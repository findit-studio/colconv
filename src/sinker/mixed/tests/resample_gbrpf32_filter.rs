//! Filter-resample coverage for the float planar-GBR family
//! ([`Gbrpf32`]) routed through the separable filter engine.
//!
//! `Gbrpf32` scatters its G/B/R planes into a source-width packed `R, G, B`
//! f32 row and bins in float on the shared `FilterStream<f32>` (the
//! `SpanKind::Filter` twin of its area `AreaStream<f32>`). The merged engine
//! filters each channel independently, so:
//!
//! 1. **Per-channel equivalence.** The binned packed row a 3-channel
//!    `Gbrpf32` filter resample produces (its `rgb_f32` output) must equal
//!    **bit-for-bit** the single-channel [`FilterStream<f32>`] resample of
//!    each source plane — the *same engine*, run per plane. The derived
//!    integer / f16 outputs must then equal a **direct** full-resolution
//!    `Gbrpf32` conversion of that binned f32 frame (the parity oracle).
//!    Covered for `Triangle` / `CatmullRom` / `Lanczos3` across a downscale
//!    (8 -> 4) and an upscale (4 -> 7), max per-channel `rgb_f32` diff 0.
//! 2. **Full-range contract.** A `CatmullRom` / `Lanczos3` edge that
//!    overshoots must push the unclamped `rgb_f32` out of `[0, 1]` (HDR > 1
//!    and ringing < 0 preserved, mirroring the area path) while the clamped
//!    integer / f16 outputs stay in range with no wrap.
//! 3. **Filter-plan-accepted regression.** A filter plan must no longer raise
//!    `UnsupportedFilter` at the `Gbrpf32` fence, and a no-output sink stays a
//!    no-op (no stream allocation).

use crate::{
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::MixedSinker,
  source::{Gbrpf32, gbrpf32_to},
};

/// LE-encode a host-native `f32` slice as the `*LE` Frame contract requires,
/// so a fixture reads back identically on LE (no-op) and BE (byte-swap) hosts.
fn as_le_f32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect()
}

/// Per-plane f32 ramps that vary per pixel and channel (so every filter
/// window sees distinct neighbours) and span HDR (> 1.0) and negative values
/// — the full-range float path carries both into `rgb_f32`. Returns
/// `(g, b, r)` host-native planes.
fn gbr_planes(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  let n = w * h;
  let mut g = std::vec![0.0f32; n];
  let mut b = std::vec![0.0f32; n];
  let mut r = std::vec![0.0f32; n];
  for i in 0..n {
    let ii = i as f32;
    r[i] = (ii * 0.013) - 0.4; // small + negative excursions
    g[i] = 1.0 + (ii * 0.05); // HDR > 1
    b[i] = ((i % 11) as f32) * 0.1 - 0.3;
  }
  (g, b, r)
}

fn frame<'a>(
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  w: usize,
  h: usize,
) -> crate::frame::Gbrpf32Frame<'a> {
  crate::frame::Gbrpf32Frame::try_new(g, b, r, w as u32, h as u32, w as u32, w as u32, w as u32)
    .unwrap()
}

/// Single-channel filter resample of one host-native f32 plane via the merged
/// engine's [`FilterStream<f32>`] (channels = 1) — the per-channel oracle. The
/// 3-channel `Gbrpf32` filter resample's binned channel must equal this
/// bit-for-bit: same engine, same coefficients, run independently per plane.
fn channel_plane_filter<K: FilterKernel>(
  kernel: K,
  plane: &[f32],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<f32> {
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<f32>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = std::vec![0.0f32; ow * oh];
  for y in 0..sh {
    stream
      .feed_row(y, &plane[y * sw..(y + 1) * sw], false, |oy, fin| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
      })
      .expect("rows in order");
  }
  out
}

/// Asserts the 3-channel `Gbrpf32` filter outputs equal the per-channel
/// single-plane [`FilterStream<f32>`] resample of the source planes, then a
/// direct full-resolution `Gbrpf32` conversion of that binned f32. Returns the
/// max per-channel `rgb_f32` diff (exactly 0 — same engine).
fn assert_filter_is_per_channel<K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> f32 {
  let (g, b, r) = gbr_planes(sw, sh);
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, sw, sh);

  let mut rgb_f32 = std::vec![0.0f32; ow * oh * 3];
  let mut rgb = std::vec![0u8; ow * oh * 3];
  let mut rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut rgba = std::vec![0u8; ow * oh * 4];
  let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
  let mut rgba_f32 = std::vec![0.0f32; ow * oh * 4];
  let mut rgb_f16 = std::vec![half::f16::ZERO; ow * oh * 3];
  let mut rgba_f16 = std::vec![half::f16::ZERO; ow * oh * 4];
  let mut luma = std::vec![0u8; ow * oh];
  let mut luma_u16 = std::vec![0u16; ow * oh];
  let mut hh = std::vec![0u8; ow * oh];
  let mut ss = std::vec![0u8; ow * oh];
  let mut vv = std::vec![0u8; ow * oh];
  {
    let mut sink = MixedSinker::<Gbrpf32, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_simd(false)
    .with_rgb_f32(&mut rgb_f32)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_rgba_f32(&mut rgba_f32)
    .unwrap()
    .with_rgb_f16(&mut rgb_f16)
    .unwrap()
    .with_rgba_f16(&mut rgba_f16)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  // 1. The `rgb_f32` output IS the binned packed `R, G, B` — each channel must
  //    equal the per-plane single-channel filter, bit-for-bit.
  let gp = channel_plane_filter(kernel, &g, sw, sh, ow, oh);
  let bp = channel_plane_filter(kernel, &b, sw, sh, ow, oh);
  let rp = channel_plane_filter(kernel, &r, sw, sh, ow, oh);
  let mut max_diff = 0.0f32;
  for px in 0..ow * oh {
    for (c, want) in [rp[px], gp[px], bp[px]].iter().enumerate() {
      let got = rgb_f32[px * 3 + c];
      let diff = (got - want).abs();
      if diff > max_diff {
        max_diff = diff;
      }
      assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{ctx} rgb_f32 px {px} c{c}: {got} vs per-plane filter {want}"
      );
    }
  }
  assert_eq!(max_diff, 0.0, "{ctx}: per-channel rgb_f32 diff must be 0");

  // 2. The derived outputs == a direct full-res `Gbrpf32` conversion of the
  //    binned f32 frame (split back into LE-wire G/B/R planes).
  let mut bg = std::vec![0.0f32; ow * oh];
  let mut bb = std::vec![0.0f32; ow * oh];
  let mut br = std::vec![0.0f32; ow * oh];
  for px in 0..ow * oh {
    br[px] = rgb_f32[px * 3];
    bg[px] = rgb_f32[px * 3 + 1];
    bb[px] = rgb_f32[px * 3 + 2];
  }
  let (bgw, bbw, brw) = (as_le_f32(&bg), as_le_f32(&bb), as_le_f32(&br));
  let mut ref_rgb = std::vec![0u8; ow * oh * 3];
  let mut ref_rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut ref_rgba = std::vec![0u8; ow * oh * 4];
  let mut ref_rgba_u16 = std::vec![0u16; ow * oh * 4];
  let mut ref_rgba_f32 = std::vec![0.0f32; ow * oh * 4];
  let mut ref_rgb_f16 = std::vec![half::f16::ZERO; ow * oh * 3];
  let mut ref_rgba_f16 = std::vec![half::f16::ZERO; ow * oh * 4];
  let mut ref_luma = std::vec![0u8; ow * oh];
  let mut ref_luma_u16 = std::vec![0u16; ow * oh];
  let mut ref_h = std::vec![0u8; ow * oh];
  let mut ref_s = std::vec![0u8; ow * oh];
  let mut ref_v = std::vec![0u8; ow * oh];
  {
    let binned = frame(&bgw, &bbw, &brw, ow, oh);
    let mut sink = MixedSinker::<Gbrpf32>::new(ow, oh)
      .with_simd(false)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_rgba_f32(&mut ref_rgba_f32)
      .unwrap()
      .with_rgb_f16(&mut ref_rgb_f16)
      .unwrap()
      .with_rgba_f16(&mut ref_rgba_f16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gbrpf32_to(&binned, &mut sink).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "{ctx} rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "{ctx} rgb_u16");
  assert_eq!(rgba, ref_rgba, "{ctx} rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "{ctx} rgba_u16");
  assert_eq!(rgba_f32, ref_rgba_f32, "{ctx} rgba_f32");
  assert_eq!(rgb_f16, ref_rgb_f16, "{ctx} rgb_f16");
  assert_eq!(rgba_f16, ref_rgba_f16, "{ctx} rgba_f16");
  assert_eq!(luma, ref_luma, "{ctx} luma");
  assert_eq!(luma_u16, ref_luma_u16, "{ctx} luma_u16");
  assert_eq!(hh, ref_h, "{ctx} hsv H");
  assert_eq!(ss, ref_s, "{ctx} hsv S");
  assert_eq!(vv, ref_v, "{ctx} hsv V");
  max_diff
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_downscale_filter_is_per_channel() {
  assert_filter_is_per_channel(Triangle, 8, 8, 4, 4, "triangle down");
  assert_filter_is_per_channel(CatmullRom, 8, 8, 4, 4, "catmullrom down");
  assert_filter_is_per_channel(Lanczos3, 8, 8, 4, 4, "lanczos3 down");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_upscale_filter_is_per_channel() {
  assert_filter_is_per_channel(Triangle, 4, 4, 7, 7, "triangle up");
  assert_filter_is_per_channel(CatmullRom, 4, 4, 7, 7, "catmullrom up");
  assert_filter_is_per_channel(Lanczos3, 4, 4, 7, 7, "lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_filter_preserves_hdr_and_negative_while_clamped_outputs_stay_in_range() {
  // A high-contrast edge of HDR green (> 1) against negative-laden black
  // drives the signed-coefficient kernels (CatmullRom / Lanczos3) to
  // overshoot. The full-range `rgb_f32` must carry the overshoot out of
  // [0, 1] in both directions; the clamped integer / f16 outputs must stay in
  // range with no wrap (each is the saturating narrow of the same rgb_f32).
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 6;
  const OH: usize = 6;

  let mut g = std::vec![0.0f32; SW * SH];
  let mut b = std::vec![0.0f32; SW * SH];
  let mut r = std::vec![0.0f32; SW * SH];
  for sy in 0..SH {
    for sx in 0..SW {
      let i = sy * SW + sx;
      if sx < SW / 2 {
        g[i] = 4.0; // HDR green
        r[i] = 1.5;
        b[i] = 1.5;
      }
      // right half stays 0 (black)
    }
  }
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, SW, SH);

  for &name in &["catmullrom", "lanczos3"] {
    let mut rgb_f32 = std::vec![0.0f32; OW * OH * 3];
    let mut rgb = std::vec![0u8; OW * OH * 3];
    let mut rgb_u16 = std::vec![0u16; OW * OH * 3];
    let mut rgb_f16 = std::vec![half::f16::ZERO; OW * OH * 3];
    macro_rules! run {
      ($k:expr) => {{
        let mut sink =
          MixedSinker::<Gbrpf32, _>::with_resampler(SW, SH, FilteredResampler::new(OW, OH, $k))
            .unwrap()
            .with_simd(false)
            .with_rgb_f32(&mut rgb_f32)
            .unwrap()
            .with_rgb(&mut rgb)
            .unwrap()
            .with_rgb_u16(&mut rgb_u16)
            .unwrap()
            .with_rgb_f16(&mut rgb_f16)
            .unwrap();
        gbrpf32_to(&src, &mut sink).unwrap();
      }};
    }
    match name {
      "catmullrom" => run!(CatmullRom),
      _ => run!(Lanczos3),
    }

    assert!(
      rgb_f32.iter().any(|&v| v > 1.0),
      "{name}: filtered rgb_f32 lost the HDR (> 1) overshoot"
    );
    assert!(
      rgb_f32.iter().any(|&v| v < 0.0),
      "{name}: filtered rgb_f32 lost the ringing undershoot (< 0)"
    );

    // The INTEGER outputs (u8 / u16) clamp `[0, 1]` and cannot wrap: each
    // equals the saturating narrow of the SAME rgb_f32 it carries (round-half-
    // up). The f16 output is the LOSSLESS narrow — by design HDR > 1 and
    // negatives are preserved (the `Rgbf32` / Xyz-`f32` full-range contract),
    // so it must equal `from_f32(rgb_f32)` UNCLAMPED, not a clamped narrow.
    for (i, &lin) in rgb_f32.iter().enumerate() {
      let clamped = lin.clamp(0.0, 1.0);
      assert_eq!(
        rgb[i],
        (clamped * 255.0 + 0.5) as u8,
        "{name}: rgb[{i}] is not the [0,1] narrow of rgb_f32 (wrap?)"
      );
      assert_eq!(
        rgb_u16[i],
        (clamped * 65535.0 + 0.5) as u16,
        "{name}: rgb_u16[{i}] is not the [0,1] narrow of rgb_f32 (wrap?)"
      );
      assert_eq!(
        rgb_f16[i].to_bits(),
        half::f16::from_f32(lin).to_bits(),
        "{name}: rgb_f16[{i}] is not the lossless (unclamped) narrow of rgb_f32"
      );
    }
    // The f16 output really did carry the HDR overshoot out of [0, 1] (it is
    // full-range, not clamped), while the integer outputs saturated.
    assert!(
      rgb_f16.iter().any(|&v| v.to_f32() > 1.0),
      "{name}: rgb_f16 lost the HDR (> 1) overshoot (clamped?)"
    );
    assert!(
      rgb.contains(&255),
      "{name}: no rgb u8 channel saturated to 255 (overshoot missing?)"
    );
    assert!(
      rgb.contains(&0),
      "{name}: no rgb u8 channel saturated to 0 (undershoot missing?)"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_filter_plan_is_accepted_and_no_output_is_a_noop() {
  // A filter plan must be accepted (no `UnsupportedFilter` at the fence) and
  // populate the output; a no-output sink stays a no-op (no stream alloc).
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (g, b, r) = gbr_planes(SW, SH);
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, SW, SH);

  let sentinel = f32::from_bits(0x7FC0_1234);
  let mut rgb_f32 = std::vec![sentinel; OW * OH * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32, FilteredResampler<Triangle>>::with_resampler(
      SW,
      SH,
      FilteredResampler::new(OW, OH, Triangle),
    )
    .unwrap()
    .with_rgb_f32(&mut rgb_f32)
    .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  assert!(
    rgb_f32.iter().all(|&v| v.to_bits() != sentinel.to_bits()),
    "filter resample must populate rgb_f32 (no UnsupportedFilter)"
  );

  let mut noop = MixedSinker::<Gbrpf32, FilteredResampler<Triangle>>::with_resampler(
    SW,
    SH,
    FilteredResampler::new(OW, OH, Triangle),
  )
  .unwrap();
  gbrpf32_to(&src, &mut noop).unwrap();
  assert!(
    !noop.rgb_filter_stream_f32_allocated(),
    "no-output filter sink allocated the f32 filter stream"
  );
}
