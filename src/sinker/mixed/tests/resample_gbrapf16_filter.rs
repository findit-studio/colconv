//! Alpha-aware filter-resample coverage for the half-float planar-GBR+alpha
//! family ([`Gbrapf16`]) routed through the separable filter engine.
//!
//! There is no `FilterStream<f16>`, so `Gbrapf16` widens its G/B/R/A
//! `half::f16` planes to host-native f32, scatters them into a packed
//! `R, G, B, A` f32 row, and bins all four channels in **float** on the
//! 4-channel `FilterStream<f32>`. Per finalized output row the tail resolves
//! the straight color, **rounds** the de-interleaved planes to `half::f16`,
//! and runs the exact direct `gbrapf16_*` / `gbrpf16_*` kernels. Therefore:
//!
//! 1. **Per-channel equivalence (f16-rounded).** Each channel of `rgba_f16`
//!    (α included — a real filtered channel) must equal `half::f16::from_f32`
//!    of the single-channel [`FilterStream<f32>`] resample of that source
//!    plane widened to f32.
//! 2. **Parity oracle.** Every other output is byte-identical to a **direct**
//!    full-resolution `Gbrapf16` conversion of the captured `rgba_f16`. Covered
//!    for `Triangle` / `CatmullRom` / `Lanczos3`, downscale (8 -> 4) + upscale
//!    (4 -> 7).
//! 3. **Premultiplied has no filter analogue** (typed `UnsupportedFilter`).
//! 4. **Full-range contract** + filter-plan-accepted / no-output regressions.

use crate::{
  resample::{
    CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
  },
  sinker::{AlphaMode, MixedSinker},
  source::{Gbrapf16, gbrapf16_to},
};

fn as_le_f16(host: &[half::f16]) -> Vec<half::f16> {
  host
    .iter()
    .map(|&v| half::f16::from_bits(v.to_bits().to_le()))
    .collect()
}

/// Per-plane f16 ramps that vary per pixel and channel spanning HDR (> 1.0)
/// and negatives, plus a varying α plane (α == 0 present). Returns
/// `(g, b, r, a)` host-native f16 planes.
#[allow(clippy::type_complexity)]
fn gbra_planes_f16(
  w: usize,
  h: usize,
) -> (
  Vec<half::f16>,
  Vec<half::f16>,
  Vec<half::f16>,
  Vec<half::f16>,
) {
  let n = w * h;
  let mut g = std::vec![half::f16::ZERO; n];
  let mut b = std::vec![half::f16::ZERO; n];
  let mut r = std::vec![half::f16::ZERO; n];
  let mut a = std::vec![half::f16::ZERO; n];
  for i in 0..n {
    let ii = i as f32;
    r[i] = half::f16::from_f32((ii * 0.013) - 0.4);
    g[i] = half::f16::from_f32(1.0 + (ii * 0.05));
    b[i] = half::f16::from_f32(((i % 11) as f32) * 0.1 - 0.3);
    a[i] = half::f16::from_f32(((i % 5) as f32) * 0.25);
  }
  (g, b, r, a)
}

fn frame<'a>(
  g: &'a [half::f16],
  b: &'a [half::f16],
  r: &'a [half::f16],
  a: &'a [half::f16],
  w: usize,
  h: usize,
) -> crate::frame::Gbrapf16Frame<'a> {
  crate::frame::Gbrapf16Frame::try_new(
    g, b, r, a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap()
}

fn channel_plane_filter_f32<K: FilterKernel>(
  kernel: K,
  plane: &[half::f16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<f32> {
  let wide: Vec<f32> = plane.iter().map(|&v| v.to_f32()).collect();
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
      .feed_row(y, &wide[y * sw..(y + 1) * sw], false, |oy, fin| {
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
) {
  let (g, b, r, a) = gbra_planes_f16(sw, sh);
  let (gw, bw, rw, aw) = (as_le_f16(&g), as_le_f16(&b), as_le_f16(&r), as_le_f16(&a));
  let src = frame(&gw, &bw, &rw, &aw, sw, sh);

  let mut rgba_f16 = std::vec![half::f16::ZERO; ow * oh * 4];
  let mut rgb = std::vec![0u8; ow * oh * 3];
  let mut rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut rgb_f32 = std::vec![0.0f32; ow * oh * 3];
  let mut rgb_f16 = std::vec![half::f16::ZERO; ow * oh * 3];
  let mut rgba = std::vec![0u8; ow * oh * 4];
  let mut rgba_u16 = std::vec![0u16; ow * oh * 4];
  let mut rgba_f32 = std::vec![0.0f32; ow * oh * 4];
  let mut luma = std::vec![0u8; ow * oh];
  let mut luma_u16 = std::vec![0u16; ow * oh];
  let mut hh = std::vec![0u8; ow * oh];
  let mut ss = std::vec![0u8; ow * oh];
  let mut vv = std::vec![0u8; ow * oh];
  {
    let mut sink = MixedSinker::<Gbrapf16, FilteredResampler<K>>::with_resampler(
      sw,
      sh,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_simd(false)
    .with_rgba_f16(&mut rgba_f16)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32)
    .unwrap()
    .with_rgb_f16(&mut rgb_f16)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_rgba_f32(&mut rgba_f32)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }

  // 1. `rgba_f16` channel == `from_f32(per-plane f32 filter)` (α too).
  let gp = channel_plane_filter_f32(kernel, &g, sw, sh, ow, oh);
  let bp = channel_plane_filter_f32(kernel, &b, sw, sh, ow, oh);
  let rp = channel_plane_filter_f32(kernel, &r, sw, sh, ow, oh);
  let ap = channel_plane_filter_f32(kernel, &a, sw, sh, ow, oh);
  for px in 0..ow * oh {
    for (c, want_f32) in [rp[px], gp[px], bp[px], ap[px]].iter().enumerate() {
      assert_eq!(
        rgba_f16[px * 4 + c].to_bits(),
        half::f16::from_f32(*want_f32).to_bits(),
        "{ctx} rgba_f16 px {px} c{c}: vs from_f32(per-plane filter {want_f32})"
      );
    }
  }

  // 2. Other outputs == a direct full-res `Gbrapf16` conversion of the
  //    captured rounded f16 planes (straight alpha).
  let mut bg = std::vec![half::f16::ZERO; ow * oh];
  let mut bb = std::vec![half::f16::ZERO; ow * oh];
  let mut br = std::vec![half::f16::ZERO; ow * oh];
  let mut ba = std::vec![half::f16::ZERO; ow * oh];
  for px in 0..ow * oh {
    br[px] = rgba_f16[px * 4];
    bg[px] = rgba_f16[px * 4 + 1];
    bb[px] = rgba_f16[px * 4 + 2];
    ba[px] = rgba_f16[px * 4 + 3];
  }
  let (bgw, bbw, brw, baw) = (
    as_le_f16(&bg),
    as_le_f16(&bb),
    as_le_f16(&br),
    as_le_f16(&ba),
  );
  let mut ref_rgb = std::vec![0u8; ow * oh * 3];
  let mut ref_rgb_u16 = std::vec![0u16; ow * oh * 3];
  let mut ref_rgb_f32 = std::vec![0.0f32; ow * oh * 3];
  let mut ref_rgb_f16 = std::vec![half::f16::ZERO; ow * oh * 3];
  let mut ref_rgba = std::vec![0u8; ow * oh * 4];
  let mut ref_rgba_u16 = std::vec![0u16; ow * oh * 4];
  let mut ref_rgba_f32 = std::vec![0.0f32; ow * oh * 4];
  let mut ref_luma = std::vec![0u8; ow * oh];
  let mut ref_luma_u16 = std::vec![0u16; ow * oh];
  let mut ref_h = std::vec![0u8; ow * oh];
  let mut ref_s = std::vec![0u8; ow * oh];
  let mut ref_v = std::vec![0u8; ow * oh];
  {
    let binned = frame(&bgw, &bbw, &brw, &baw, ow, oh);
    let mut sink = MixedSinker::<Gbrapf16>::new(ow, oh)
      .with_simd(false)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgb_f32(&mut ref_rgb_f32)
      .unwrap()
      .with_rgb_f16(&mut ref_rgb_f16)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_rgba_f32(&mut ref_rgba_f32)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gbrapf16_to(&binned, &mut sink).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "{ctx} rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "{ctx} rgb_u16");
  assert_eq!(rgb_f32, ref_rgb_f32, "{ctx} rgb_f32");
  assert_eq!(rgb_f16, ref_rgb_f16, "{ctx} rgb_f16");
  assert_eq!(rgba, ref_rgba, "{ctx} rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "{ctx} rgba_u16");
  assert_eq!(rgba_f32, ref_rgba_f32, "{ctx} rgba_f32");
  assert_eq!(luma, ref_luma, "{ctx} luma");
  assert_eq!(luma_u16, ref_luma_u16, "{ctx} luma_u16");
  assert_eq!(hh, ref_h, "{ctx} hsv H");
  assert_eq!(ss, ref_s, "{ctx} hsv S");
  assert_eq!(vv, ref_v, "{ctx} hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_downscale_filter_is_per_channel() {
  assert_filter_is_per_channel(Triangle, 8, 8, 4, 4, "triangle down");
  assert_filter_is_per_channel(CatmullRom, 8, 8, 4, 4, "catmullrom down");
  assert_filter_is_per_channel(Lanczos3, 8, 8, 4, 4, "lanczos3 down");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_upscale_filter_is_per_channel() {
  assert_filter_is_per_channel(Triangle, 4, 4, 7, 7, "triangle up");
  assert_filter_is_per_channel(CatmullRom, 4, 4, 7, 7, "catmullrom up");
  assert_filter_is_per_channel(Lanczos3, 4, 4, 7, 7, "lanczos3 up");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_premultiplied_filter_is_rejected() {
  use crate::{resample::ResampleError, sinker::MixedSinkerError};
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (g, b, r, a) = gbra_planes_f16(SW, SH);
  let (gw, bw, rw, aw) = (as_le_f16(&g), as_le_f16(&b), as_le_f16(&r), as_le_f16(&a));
  let src = frame(&gw, &bw, &rw, &aw, SW, SH);

  let mut rgba = std::vec![0u8; OW * OH * 4];
  let mut sink = MixedSinker::<Gbrapf16, FilteredResampler<Triangle>>::with_resampler(
    SW,
    SH,
    FilteredResampler::new(OW, OH, Triangle),
  )
  .unwrap()
  .with_alpha_mode(AlphaMode::Premultiplied)
  .with_rgba(&mut rgba)
  .unwrap();
  let err = gbrapf16_to(&src, &mut sink).unwrap_err();
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
fn gbrapf16_filter_plan_is_accepted_and_no_output_is_a_noop() {
  const SW: usize = 8;
  const SH: usize = 8;
  const OW: usize = 4;
  const OH: usize = 4;
  let (g, b, r, a) = gbra_planes_f16(SW, SH);
  let (gw, bw, rw, aw) = (as_le_f16(&g), as_le_f16(&b), as_le_f16(&r), as_le_f16(&a));
  let src = frame(&gw, &bw, &rw, &aw, SW, SH);

  let sentinel = half::f16::from_bits(0x7E01);
  let mut rgba_f16 = std::vec![sentinel; OW * OH * 4];
  {
    let mut sink = MixedSinker::<Gbrapf16, FilteredResampler<Triangle>>::with_resampler(
      SW,
      SH,
      FilteredResampler::new(OW, OH, Triangle),
    )
    .unwrap()
    .with_rgba_f16(&mut rgba_f16)
    .unwrap();
    gbrapf16_to(&src, &mut sink).unwrap();
  }
  assert!(
    rgba_f16.iter().all(|&v| v.to_bits() != sentinel.to_bits()),
    "filter resample must populate rgba_f16 (no UnsupportedFilter)"
  );

  let mut noop = MixedSinker::<Gbrapf16, FilteredResampler<Triangle>>::with_resampler(
    SW,
    SH,
    FilteredResampler::new(OW, OH, Triangle),
  )
  .unwrap();
  gbrapf16_to(&src, &mut noop).unwrap();
  assert!(
    !noop.rgba_filter_stream_f32_allocated(),
    "no-output filter sink allocated the 4-channel f32 filter stream"
  );
}
