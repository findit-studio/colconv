//! Fused-downscale coverage for `Yuv411p` — routed through the shared
//! packed-RGB resample tail. Each source row is chroma-upsampled to a
//! full-width RGB row at source width (the same `yuv_411_to_rgb_row`
//! kernel the identity path uses) in the shared scratch, then fed to
//! the area stream and the shared emit derivations.
//!
//! The differential pins **row-stage == convert-then-bin**: resampling
//! the 4:1:1 source directly must equal converting it to full-width
//! RGB first (identity sink) and then area-downscaling that RGB frame
//! through the `Rgb24` path. Both derive luma from the binned RGB with
//! the same matrix / range, so RGB and luma match byte-for-byte.

use crate::{
  ColorMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Rgb24, Yuv411p, rgb24_to, yuv411p_to},
};
use mediaframe::frame::{Rgb24Frame, Yuv411pFrame};

const SRC_W: usize = 8;
const SRC_H: usize = 8;
const OUT_W: usize = 4;
const OUT_H: usize = 4;
// FFmpeg `AV_PIX_FMT_YUV411P`: chroma row width is `width.div_ceil(4)`.
const CW: usize = SRC_W.div_ceil(4);

/// Build a `Yuv411p` frame (contiguous planes) with interior ramp
/// values across Y / U / V, so the chroma upsample and the area mean
/// both see real variation rather than a flat field.
fn ramp_yuv411p_frame() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = vec![0u8; SRC_W * SRC_H];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 32 + (i % 64) as u8; // interior, varies per pixel
  }
  let mut u = vec![0u8; CW * SRC_H];
  for (i, p) in u.iter_mut().enumerate() {
    *p = 100 + (i % 40) as u8;
  }
  let mut v = vec![0u8; CW * SRC_H];
  for (i, p) in v.iter_mut().enumerate() {
    *p = 140 - (i % 40) as u8;
  }
  (y, u, v)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_resample_matches_rgb24_of_converted_frame() {
  let (yp, up, vp) = ramp_yuv411p_frame();
  let src = Yuv411pFrame::new(
    &yp,
    &up,
    &vp,
    SRC_W as u32,
    SRC_H as u32,
    SRC_W as u32,
    CW as u32,
    CW as u32,
  );

  // Path A — resample the 4:1:1 source directly (row-stage).
  let (mut rgb_a, mut luma_a) = (vec![0u8; OUT_W * OUT_H * 3], vec![0u8; OUT_W * OUT_H]);
  {
    let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_rgb(&mut rgb_a)
    .unwrap()
    .with_luma(&mut luma_a)
    .unwrap();
    yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  // Reference, step 1 — identity-convert the 4:1:1 source to a
  // source-width RGB frame.
  let mut full_rgb = vec![0u8; SRC_W * SRC_H * 3];
  {
    let mut sink = MixedSinker::<Yuv411p>::new(SRC_W, SRC_H)
      .with_rgb(&mut full_rgb)
      .unwrap();
    yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  // Reference, step 2 — area-downscale that RGB frame via the Rgb24
  // path (convert-then-bin), same matrix / range so luma matches.
  let rgb_src = Rgb24Frame::new(&full_rgb, SRC_W as u32, SRC_H as u32, (SRC_W * 3) as u32);
  let (mut rgb_b, mut luma_b) = (vec![0u8; OUT_W * OUT_H * 3], vec![0u8; OUT_W * OUT_H]);
  {
    let mut sink = MixedSinker::<Rgb24, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_rgb(&mut rgb_b)
    .unwrap()
    .with_luma(&mut luma_b)
    .unwrap();
    rgb24_to(&rgb_src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  assert_eq!(rgb_a, rgb_b, "rgb: row-stage must equal convert-then-bin");
  assert_eq!(
    luma_a, luma_b,
    "luma: row-stage must equal convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_identity_plan_matches_new_sink() {
  // A resampler to the SAME geometry must be byte-identical to the
  // non-resampling `new()` sink for every output.
  let (yp, up, vp) = ramp_yuv411p_frame();
  let src = Yuv411pFrame::new(
    &yp,
    &up,
    &vp,
    SRC_W as u32,
    SRC_H as u32,
    SRC_W as u32,
    CW as u32,
    CW as u32,
  );

  let (mut rgb_direct, mut luma_direct) = (vec![0u8; SRC_W * SRC_H * 3], vec![0u8; SRC_W * SRC_H]);
  {
    let mut sink = MixedSinker::<Yuv411p>::new(SRC_W, SRC_H)
      .with_rgb(&mut rgb_direct)
      .unwrap()
      .with_luma(&mut luma_direct)
      .unwrap();
    yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  let (mut rgb_area, mut luma_area) = (vec![0u8; SRC_W * SRC_H * 3], vec![0u8; SRC_W * SRC_H]);
  {
    let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(SRC_W, SRC_H),
    )
    .unwrap()
    .with_rgb(&mut rgb_area)
    .unwrap()
    .with_luma(&mut luma_area)
    .unwrap();
    yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  assert_eq!(rgb_direct, rgb_area, "rgb identity-plan vs new()");
  // The identity (`new()`) path copies luma straight off the Y plane,
  // while the resample path derives it from the (1:1) binned RGB —
  // round-trip-equal here only because the geometry is unchanged.
  assert_eq!(luma_direct, luma_area, "luma identity-plan vs new()");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_resample_no_outputs_is_a_no_op() {
  // No attached outputs is legal: the resample route freezes the
  // (empty) output set and returns before allocating a stream or
  // enforcing sequencing, so even an out-of-order row is accepted.
  let (yp, up, vp) = ramp_yuv411p_frame();
  let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap();
  sink.begin_frame(SRC_W as u32, SRC_H as u32).unwrap();
  let y2 = &yp[SRC_W * 2..SRC_W * 3];
  let u2 = &up[CW * 2..CW * 3];
  let v2 = &vp[CW * 2..CW * 3];
  sink
    .process(crate::source::Yuv411pRow::new(
      y2,
      u2,
      v2,
      2,
      ColorMatrix::Bt601,
      true,
    ))
    .unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_resample_rejects_out_of_sequence_before_staging() {
  // An out-of-sequence row must be rejected before the source-width
  // scratch is grown or the chroma upsample runs — sequencing is
  // checked ahead of staging. The scratch capacity staying zero proves
  // no staging allocation happened.
  let (yp, up, vp) = ramp_yuv411p_frame();
  let mut rgb = vec![0u8; OUT_W * OUT_H * 3];
  let mut sink = MixedSinker::<Yuv411p, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap()
  .with_rgb(&mut rgb)
  .unwrap();
  sink.begin_frame(SRC_W as u32, SRC_H as u32).unwrap();
  // Skip row 0 — the stream expects strict sequencing from row 0.
  let y2 = &yp[SRC_W * 2..SRC_W * 3];
  let u2 = &up[CW * 2..CW * 3];
  let v2 = &vp[CW * 2..CW * 3];
  let err = sink
    .process(crate::source::Yuv411pRow::new(
      y2,
      u2,
      v2,
      2,
      ColorMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "got {err:?}"
  );
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "out-of-sequence row must not stage into the scratch"
  );
}
