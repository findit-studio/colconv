//! Fused-downscale coverage for the planar `Yuv422p` / `Yuv444p`
//! sources routed through the shared planar resample tail. These
//! formats are row-stage only: each source row is converted to
//! canonical RGB at source width, then fed to one 3-channel area
//! stream — exactly the `Bgr24` / `Rgb24` model. So the differential
//! tests pin **row-stage == convert-then-bin**: the routed planar
//! output must equal running the converted (identity-path) RGB frame
//! through the `Rgb24` resample path.

use crate::{
  ColorMatrix,
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{Rgb24, Yuv422p, Yuv444p, rgb24_to, yuv422p_to, yuv444p_to},
};
use mediaframe::frame::{Rgb24Frame, Yuv422pFrame, Yuv444pFrame};

const SRC: usize = 8;
const OUT: usize = 4;

/// Interior-ramp Y / U / V planes for a `SRC x SRC` 4:2:2 frame
/// (half-width, full-height chroma). Every sample is interior so the
/// YUV→RGB kernel and the area binning see real math.
fn yuv422p_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = SRC / 2;
  let mut y = vec![0u8; SRC * SRC];
  let mut u = vec![0u8; cw * SRC];
  let mut v = vec![0u8; cw * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + (i as u8) * 2;
  }
  for (i, p) in u.iter_mut().enumerate() {
    *p = 70 + ((i % cw) as u8) * 5;
  }
  for (i, p) in v.iter_mut().enumerate() {
    *p = 200 - ((i % cw) as u8) * 4;
  }
  (y, u, v)
}

/// Interior-ramp Y / U / V planes for a `SRC x SRC` 4:4:4 frame
/// (full-width, full-height chroma).
fn yuv444p_planes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let mut y = vec![0u8; SRC * SRC];
  let mut u = vec![0u8; SRC * SRC];
  let mut v = vec![0u8; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 40 + (i as u8) * 2;
  }
  for (i, p) in u.iter_mut().enumerate() {
    *p = 70 + ((i % SRC) as u8) * 5;
  }
  for (i, p) in v.iter_mut().enumerate() {
    *p = 200 - ((i % SRC) as u8) * 3;
  }
  (y, u, v)
}

/// Build the full-resolution converted RGB frame via the format's
/// **identity** sink. This is the canonical RGB the resample path bins,
/// so feeding it through the `Rgb24` resample path is the reference for
/// the routed planar output.
fn yuv422p_converted_rgb(y: &[u8], u: &[u8], v: &[u8]) -> Vec<u8> {
  let cw = SRC / 2;
  let src = Yuv422pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );
  let mut rgb = vec![0u8; SRC * SRC * 3];
  let mut sink = MixedSinker::<Yuv422p>::new(SRC, SRC)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  rgb
}

fn yuv444p_converted_rgb(y: &[u8], u: &[u8], v: &[u8]) -> Vec<u8> {
  let src = Yuv444pFrame::new(
    y, u, v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  );
  let mut rgb = vec![0u8; SRC * SRC * 3];
  let mut sink = MixedSinker::<Yuv444p>::new(SRC, SRC)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  rgb
}

/// Run the `Rgb24` resample path on the converted RGB frame — the
/// reference output set for the routed planar comparison.
fn rgb24_reference(converted: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let src = Rgb24Frame::new(converted, SRC as u32, SRC as u32, (SRC * 3) as u32);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    rgb24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  (rgb, luma)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv422p_resample_matches_rgb24_of_converted_frame() {
  let (y, u, v) = yuv422p_planes();
  let cw = SRC / 2;
  let src = Yuv422pFrame::new(
    &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );

  let mut rgb_a = vec![0u8; OUT * OUT * 3];
  let mut luma_a = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb_a)
        .unwrap()
        .with_luma(&mut luma_a)
        .unwrap();
    yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  let converted = yuv422p_converted_rgb(&y, &u, &v);
  let (rgb_b, luma_b) = rgb24_reference(&converted);

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
fn yuv444p_resample_matches_rgb24_of_converted_frame() {
  let (y, u, v) = yuv444p_planes();
  let src = Yuv444pFrame::new(
    &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  );

  let mut rgb_a = vec![0u8; OUT * OUT * 3];
  let mut luma_a = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb_a)
        .unwrap()
        .with_luma(&mut luma_a)
        .unwrap();
    yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  let converted = yuv444p_converted_rgb(&y, &u, &v);
  let (rgb_b, luma_b) = rgb24_reference(&converted);

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
fn yuv422p_identity_plan_matches_new_sink() {
  let (y, u, v) = yuv422p_planes();
  let cw = SRC / 2;
  let src = Yuv422pFrame::new(
    &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw as u32, cw as u32,
  );

  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv422p>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Yuv422p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    yuv422p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv444p_identity_plan_matches_new_sink() {
  let (y, u, v) = yuv444p_planes();
  let src = Yuv444pFrame::new(
    &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  );

  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Yuv444p>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Yuv444p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    yuv444p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}
