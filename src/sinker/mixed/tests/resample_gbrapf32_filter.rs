//! Alpha-aware filter-resample coverage for the float planar-GBR+alpha
//! family ([`Gbrapf32`]) routed through the separable filter engine.
//!
//! `Gbrapf32` scatters its G/B/R/A planes into a source-width packed
//! `R, G, B, A` f32 row and bins all four channels in float on the 4-channel
//! `FilterStream<f32>` (the `SpanKind::Filter` twin of its area
//! `AreaStream<f32>`). PIL filters R, G, B, A independently with no
//! premultiplication, so:
//!
//! 1. **Per-channel equivalence.** The binned packed row a `Gbrapf32` filter
//!    resample produces (its `rgba_f32` output) must equal **bit-for-bit** the
//!    single-channel [`FilterStream<f32>`] resample of each source plane (α
//!    included — a real filtered channel, never forced opaque). The derived
//!    outputs equal a direct full-resolution `Gbrapf32` conversion of that
//!    binned f32 (the parity oracle). Covered for `Triangle` / `CatmullRom` /
//!    `Lanczos3` across a downscale (8 -> 4) and an upscale (4 -> 7), max
//!    per-channel `rgba_f32` diff 0.
//! 2. **Full-range contract.** A signed-kernel overshoot pushes the unclamped
//!    `rgba_f32` out of `[0, 1]` while the clamped integer / f16 outputs stay
//!    in range with no wrap.
//! 3. **Premultiplied has no filter analogue.** A premultiplied-alpha filter
//!    plan surfaces the typed `UnsupportedFilter` (routed to the area tail,
//!    which un-premultiplies); straight alpha is the only filtered mode.
//! 4. **Filter-plan-accepted regression** + no-output no-op.

use crate::{
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::{AlphaMode, MixedSinker},
  source::{Gbrapf32, gbrapf32_to},
};

fn as_le_f32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect()
}

/// Per-plane f32 ramps that vary per pixel and channel (distinct filter
/// neighbours) spanning HDR (> 1.0) and negatives, plus an α plane that varies
/// (and includes 0 and HDR α). Returns `(g, b, r, a)` host-native planes.
fn gbra_planes(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
  let n = w * h;
  let mut g = std::vec![0.0f32; n];
  let mut b = std::vec![0.0f32; n];
  let mut r = std::vec![0.0f32; n];
  let mut a = std::vec![0.0f32; n];
  for i in 0..n {
    let ii = i as f32;
    r[i] = (ii * 0.013) - 0.4;
    g[i] = 1.0 + (ii * 0.05);
    b[i] = ((i % 11) as f32) * 0.1 - 0.3;
    a[i] = ((i % 5) as f32) * 0.25; // 0.0 .. 1.0 with α == 0 present
  }
  (g, b, r, a)
}

fn frame<'a>(
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  a: &'a [f32],
  w: usize,
  h: usize,
) -> crate::frame::Gbrapf32Frame<'a> {
  crate::frame::Gbrapf32Frame::try_new(
    g, b, r, a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap()
}

/// Single-channel filter resample of one host-native f32 plane via the merged
/// engine's [`FilterStream<f32>`] (channels = 1) — the per-channel oracle.
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

fn assert_filter_is_per_channel<K: FilterKernel + Copy>(
  kernel: K,
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
  ctx: &str,
) -> f32 {
  let (g, b, r, a) = gbra_planes(sw, sh);
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, sw, sh);

  let mut rgba_f32 = std::vec![0.0f32; ow * oh * 4];
  let mut rgb = std::vec![0u8; ow * oh * 3];
  let mut rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut rgb_f32 = std::vec![0.0f32; ow * oh * 3];
  let mut rgba = std::vec![0u8; ow * oh * 4];
  let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
  let mut rgb_f16 = std::vec![half::f16::ZERO; ow * oh * 3];
  let mut rgba_f16 = std::vec![half::f16::ZERO; ow * oh * 4];
  let mut luma = std::vec![0u8; ow * oh];
  let mut luma_u16 = std::vec![0u16; ow * oh];
  let mut hh = std::vec![0u8; ow * oh];
  let mut ss = std::vec![0u8; ow * oh];
  let mut vv = std::vec![0u8; ow * oh];
  {
    let mut sink = MixedSinker::<Gbrapf32, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_simd(false)
    .with_rgba_f32(&mut rgba_f32)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
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
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // 1. `rgba_f32` IS the binned packed `R, G, B, A` — each channel (α too)
  //    must equal the per-plane single-channel filter bit-for-bit.
  let gp = channel_plane_filter(kernel, &g, sw, sh, ow, oh);
  let bp = channel_plane_filter(kernel, &b, sw, sh, ow, oh);
  let rp = channel_plane_filter(kernel, &r, sw, sh, ow, oh);
  let ap = channel_plane_filter(kernel, &a, sw, sh, ow, oh);
  let mut max_diff = 0.0f32;
  for px in 0..ow * oh {
    for (c, want) in [rp[px], gp[px], bp[px], ap[px]].iter().enumerate() {
      let got = rgba_f32[px * 4 + c];
      let diff = (got - want).abs();
      if diff > max_diff {
        max_diff = diff;
      }
      assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{ctx} rgba_f32 px {px} c{c}: {got} vs per-plane filter {want}"
      );
    }
  }
  assert_eq!(max_diff, 0.0, "{ctx}: per-channel rgba_f32 diff must be 0");

  // 2. Derived outputs == a direct full-res `Gbrapf32` conversion of the
  //    binned f32 frame (straight alpha; α the real filtered channel).
  let mut bg = std::vec![0.0f32; ow * oh];
  let mut bb = std::vec![0.0f32; ow * oh];
  let mut br = std::vec![0.0f32; ow * oh];
  let mut ba = std::vec![0.0f32; ow * oh];
  for px in 0..ow * oh {
    br[px] = rgba_f32[px * 4];
    bg[px] = rgba_f32[px * 4 + 1];
    bb[px] = rgba_f32[px * 4 + 2];
    ba[px] = rgba_f32[px * 4 + 3];
  }
  let (bgw, bbw, brw, baw) = (
    as_le_f32(&bg),
    as_le_f32(&bb),
    as_le_f32(&br),
    as_le_f32(&ba),
  );
  let mut ref_rgb = std::vec![0u8; ow * oh * 3];
  let mut ref_rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut ref_rgb_f32 = std::vec![0.0f32; ow * oh * 3];
  let mut ref_rgba = std::vec![0u8; ow * oh * 4];
  let mut ref_rgba_u16 = std::vec![0u16; ow * oh * 4];
  let mut ref_rgb_f16 = std::vec![half::f16::ZERO; ow * oh * 3];
  let mut ref_rgba_f16 = std::vec![half::f16::ZERO; ow * oh * 4];
  let mut ref_luma = std::vec![0u8; ow * oh];
  let mut ref_luma_u16 = std::vec![0u16; ow * oh];
  let mut ref_h = std::vec![0u8; ow * oh];
  let mut ref_s = std::vec![0u8; ow * oh];
  let mut ref_v = std::vec![0u8; ow * oh];
  {
    let binned = frame(&bgw, &bbw, &brw, &baw, ow, oh);
    let mut sink = MixedSinker::<Gbrapf32>::new(ow, oh)
      .with_simd(false)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgb_f32(&mut ref_rgb_f32)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
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
    gbrapf32_to(&binned, &mut sink).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "{ctx} rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "{ctx} rgb_u16");
  assert_eq!(rgb_f32, ref_rgb_f32, "{ctx} rgb_f32");
  assert_eq!(rgba, ref_rgba, "{ctx} rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "{ctx} rgba_u16");
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
fn gbrapf32_downscale_filter_is_per_channel() {
  assert_filter_is_per_channel(Triangle, 8, 8, 4, 4, "triangle down");
  assert_filter_is_per_channel(CatmullRom, 8, 8, 4, 4, "catmullrom down");
  assert_filter_is_per_channel(Lanczos3, 8, 8, 4, 4, "lanczos3 down");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_upscale_filter_is_per_channel() {
  assert_filter_is_per_channel(Triangle, 4, 4, 7, 7, "triangle up");
  assert_filter_is_per_channel(CatmullRom, 4, 4, 7, 7, "catmullrom up");
  assert_filter_is_per_channel(Lanczos3, 4, 4, 7, 7, "lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_filter_preserves_hdr_and_negative_while_clamped_outputs_stay_in_range() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 6;
  const OH: usize = 6;
  let mut g = std::vec![0.0f32; SW * SH];
  let mut b = std::vec![0.0f32; SW * SH];
  let mut r = std::vec![0.0f32; SW * SH];
  let mut a = std::vec![0.0f32; SW * SH];
  for sy in 0..SH {
    for sx in 0..SW {
      let i = sy * SW + sx;
      if sx < SW / 2 {
        g[i] = 4.0;
        r[i] = 1.5;
        b[i] = 1.5;
        a[i] = 1.0;
      }
    }
  }
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SW, SH);

  for &name in &["catmullrom", "lanczos3"] {
    let mut rgba_f32 = std::vec![0.0f32; OW * OH * 4];
    let mut rgba = std::vec![0u8; OW * OH * 4];
    let mut rgba_u16 = std::vec![0u16; OW * OH * 4];
    let mut rgba_f16 = std::vec![half::f16::ZERO; OW * OH * 4];
    macro_rules! run {
      ($k:expr) => {{
        let mut sink =
          MixedSinker::<Gbrapf32, _>::with_resampler(SW, SH, FilteredResampler::new(OW, OH, $k))
            .unwrap()
            .with_simd(false)
            .with_rgba_f32(&mut rgba_f32)
            .unwrap()
            .with_rgba(&mut rgba)
            .unwrap()
            .with_rgba_u16(&mut rgba_u16)
            .unwrap()
            .with_rgba_f16(&mut rgba_f16)
            .unwrap();
        gbrapf32_to(&src, &mut sink).unwrap();
      }};
    }
    match name {
      "catmullrom" => run!(CatmullRom),
      _ => run!(Lanczos3),
    }

    assert!(
      rgba_f32.iter().any(|&v| v > 1.0),
      "{name}: filtered rgba_f32 lost the HDR (> 1) overshoot"
    );
    assert!(
      rgba_f32.iter().any(|&v| v < 0.0),
      "{name}: filtered rgba_f32 lost the ringing undershoot (< 0)"
    );
    // INTEGER outputs (u8 / u16) clamp `[0, 1]` and cannot wrap. The f16
    // output is the LOSSLESS narrow — HDR > 1 and negatives preserved by
    // design — so it equals `from_f32(rgba_f32)` UNCLAMPED.
    for (i, &lin) in rgba_f32.iter().enumerate() {
      let clamped = lin.clamp(0.0, 1.0);
      assert_eq!(
        rgba[i],
        (clamped * 255.0 + 0.5) as u8,
        "{name}: rgba[{i}] is not the [0,1] narrow of rgba_f32 (wrap?)"
      );
      assert_eq!(
        rgba_u16[i],
        (clamped * 65535.0 + 0.5) as u16,
        "{name}: rgba_u16[{i}] is not the [0,1] narrow of rgba_f32 (wrap?)"
      );
      assert_eq!(
        rgba_f16[i].to_bits(),
        half::f16::from_f32(lin).to_bits(),
        "{name}: rgba_f16[{i}] is not the lossless (unclamped) narrow of rgba_f32"
      );
    }
    assert!(
      rgba_f16.iter().any(|&v| v.to_f32() > 1.0),
      "{name}: rgba_f16 lost the HDR (> 1) overshoot (clamped?)"
    );
    assert!(
      rgba.contains(&255),
      "{name}: no rgba u8 channel saturated to 255 (overshoot missing?)"
    );
    assert!(
      rgba.contains(&0),
      "{name}: no rgba u8 channel saturated to 0 (undershoot missing?)"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_premultiplied_filter_is_rejected() {
  // Premultiplied alpha has no filter analogue; a premultiplied-alpha filter
  // plan must surface the typed `UnsupportedFilter` (routed to the area tail,
  // which un-premultiplies — the filter engine cannot).
  use crate::{resample::ResampleError, sinker::MixedSinkerError};
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (g, b, r, a) = gbra_planes(SW, SH);
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SW, SH);

  let mut rgba = std::vec![0u8; OW * OH * 4];
  let mut sink = MixedSinker::<Gbrapf32, FilteredResampler<Triangle>>::with_resampler(
    SW,
    SH,
    FilteredResampler::new(OW, OH, Triangle),
  )
  .unwrap()
  .with_alpha_mode(AlphaMode::Premultiplied)
  .with_rgba(&mut rgba)
  .unwrap();
  let err = gbrapf32_to(&src, &mut sink).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::UnsupportedFilter(_))
    ),
    "premultiplied filter plan must reject with UnsupportedFilter, got {err:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_filter_plan_is_accepted_and_no_output_is_a_noop() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (g, b, r, a) = gbra_planes(SW, SH);
  let (gw, bw, rw, aw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r), as_le_f32(&a));
  let src = frame(&gw, &bw, &rw, &aw, SW, SH);

  let sentinel = f32::from_bits(0x7FC0_1234);
  let mut rgba_f32 = std::vec![sentinel; OW * OH * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32, FilteredResampler<Triangle>>::with_resampler(
      SW,
      SH,
      FilteredResampler::new(OW, OH, Triangle),
    )
    .unwrap()
    .with_rgba_f32(&mut rgba_f32)
    .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }
  assert!(
    rgba_f32.iter().all(|&v| v.to_bits() != sentinel.to_bits()),
    "filter resample must populate rgba_f32 (no UnsupportedFilter)"
  );

  let mut noop = MixedSinker::<Gbrapf32, FilteredResampler<Triangle>>::with_resampler(
    SW,
    SH,
    FilteredResampler::new(OW, OH, Triangle),
  )
  .unwrap();
  gbrapf32_to(&src, &mut noop).unwrap();
  assert!(
    !noop.rgba_filter_stream_f32_allocated(),
    "no-output filter sink allocated the 4-channel f32 filter stream"
  );
}
