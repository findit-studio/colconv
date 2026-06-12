//! Geometry-split contract: output buffers validate against the
//! resampler's output geometry while `begin_frame` keeps validating the
//! walker against the source geometry.

use crate::{
  PixelSink,
  resample::{AreaResampler, NoopResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError, mixed::HsvPlane},
  source::Yuv420p,
};

const SRC: usize = 8;
const OUT: usize = 4;

fn downscaled<'a>() -> MixedSinker<'a, Yuv420p, AreaResampler> {
  MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
    .expect("8x8 -> 4x4 area plan never fails")
}

#[test]
fn new_keeps_output_geometry_equal_to_source() {
  let sink = MixedSinker::<Yuv420p>::new(SRC, SRC);
  assert_eq!((sink.width(), sink.height()), (SRC, SRC));
  assert_eq!((sink.out_width(), sink.out_height()), (SRC, SRC));
}

#[test]
fn with_resampler_noop_matches_new() {
  let sink = MixedSinker::<Yuv420p, NoopResampler>::with_resampler(SRC, SRC, NoopResampler)
    .expect("identity plan never fails");
  assert_eq!((sink.width(), sink.height()), (SRC, SRC));
  assert_eq!((sink.out_width(), sink.out_height()), (SRC, SRC));
}

#[test]
fn with_resampler_area_identity_matches_new() {
  let sink =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
      .expect("identity area plan never fails");
  assert_eq!((sink.out_width(), sink.out_height()), (SRC, SRC));
}

#[test]
fn with_resampler_downscale_shrinks_output_geometry() {
  let sink = downscaled();
  assert_eq!((sink.width(), sink.height()), (SRC, SRC));
  assert_eq!((sink.out_width(), sink.out_height()), (OUT, OUT));
}

#[test]
fn rgb_buffer_validates_against_output_geometry() {
  let mut short = vec![0u8; OUT * OUT * 3 - 1];
  let err = downscaled().with_rgb(&mut short).map(|_| ()).unwrap_err();
  match err {
    MixedSinkerError::InsufficientRgbBuffer(e) => {
      assert_eq!(e.expected(), OUT * OUT * 3);
      assert_eq!(e.actual(), OUT * OUT * 3 - 1);
    }
    other => panic!("expected InsufficientRgbBuffer, got {other:?}"),
  }

  let mut exact = vec![0u8; OUT * OUT * 3];
  assert!(downscaled().with_rgb(&mut exact).is_ok());
}

#[test]
fn luma_buffer_validates_against_output_geometry() {
  let mut short = vec![0u8; OUT * OUT - 1];
  let err = downscaled().with_luma(&mut short).map(|_| ()).unwrap_err();
  match err {
    MixedSinkerError::InsufficientLumaBuffer(e) => assert_eq!(e.expected(), OUT * OUT),
    other => panic!("expected InsufficientLumaBuffer, got {other:?}"),
  }

  let mut exact = vec![0u8; OUT * OUT];
  assert!(downscaled().with_luma(&mut exact).is_ok());
}

#[test]
fn luma_u16_buffer_validates_against_output_geometry() {
  let mut short = vec![0u16; OUT * OUT - 1];
  let err = downscaled()
    .with_luma_u16(&mut short)
    .map(|_| ())
    .unwrap_err();
  match err {
    MixedSinkerError::InsufficientLumaU16Buffer(e) => assert_eq!(e.expected(), OUT * OUT),
    other => panic!("expected InsufficientLumaU16Buffer, got {other:?}"),
  }

  let mut exact = vec![0u16; OUT * OUT];
  assert!(downscaled().with_luma_u16(&mut exact).is_ok());
}

#[test]
fn rgba_buffer_validates_against_output_geometry() {
  let mut short = vec![0u8; OUT * OUT * 4 - 1];
  let err = downscaled().with_rgba(&mut short).map(|_| ()).unwrap_err();
  match err {
    MixedSinkerError::InsufficientRgbaBuffer(e) => assert_eq!(e.expected(), OUT * OUT * 4),
    other => panic!("expected InsufficientRgbaBuffer, got {other:?}"),
  }

  let mut exact = vec![0u8; OUT * OUT * 4];
  assert!(downscaled().with_rgba(&mut exact).is_ok());
}

#[test]
fn hsv_planes_validate_against_output_geometry() {
  let mut h = vec![0u8; OUT * OUT - 1];
  let mut s = vec![0u8; OUT * OUT];
  let mut v = vec![0u8; OUT * OUT];
  let err = downscaled()
    .with_hsv(&mut h, &mut s, &mut v)
    .map(|_| ())
    .unwrap_err();
  match err {
    MixedSinkerError::InsufficientHsvPlane(e) => {
      assert_eq!(e.which(), HsvPlane::H);
      assert_eq!(e.expected(), OUT * OUT);
    }
    other => panic!("expected InsufficientHsvPlane, got {other:?}"),
  }

  let mut h = vec![0u8; OUT * OUT];
  assert!(downscaled().with_hsv(&mut h, &mut s, &mut v).is_ok());
}

#[test]
fn begin_frame_still_validates_source_geometry() {
  // Non-identity sinks deliberately have no `PixelSink` impl until the
  // streaming engine routes output geometry, so the walker contract is
  // pinned on the `with_resampler`-built identity sink.
  let mut sink = MixedSinker::<Yuv420p, NoopResampler>::with_resampler(SRC, SRC, NoopResampler)
    .expect("identity plan never fails");
  assert!(sink.begin_frame(SRC as u32, SRC as u32).is_ok());
  assert!(matches!(
    sink.begin_frame(OUT as u32, OUT as u32),
    Err(MixedSinkerError::DimensionMismatch(_))
  ));
}

#[test]
fn plan_error_surfaces_as_mixed_sinker_error() {
  let err =
    MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(0, 0))
      .map(|_| ())
      .unwrap_err();
  assert!(matches!(
    err,
    MixedSinkerError::Resample(ResampleError::ZeroOutputDimension(_))
  ));
}
