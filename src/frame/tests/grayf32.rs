use crate::frame::{Grayf32Frame, Grayf32FrameError};

#[test]
fn grayf32_frame_try_new_accepts_valid_tight() {
  // stride == width (tight); 4 px × 4 rows = 16 elements
  let buf = [0.0f32; 16];
  let f = Grayf32Frame::try_new(&buf, 4, 4, 4).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.y_stride(), 4);
  assert_eq!(f.y().len(), 16);
}

#[test]
fn grayf32_frame_try_new_accepts_padded_stride() {
  // stride = 8 (padded to 8), 4 px × 4 rows = 32 elements
  let buf = [0.0f32; 32];
  Grayf32Frame::try_new(&buf, 4, 4, 8).unwrap();
}

#[test]
fn grayf32_frame_try_new_rejects_zero_width() {
  let buf = [0.0f32; 16];
  assert!(matches!(
    Grayf32Frame::try_new(&buf, 0, 4, 4),
    Err(Grayf32FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
}

#[test]
fn grayf32_frame_try_new_rejects_zero_height() {
  let buf = [0.0f32; 16];
  assert!(matches!(
    Grayf32Frame::try_new(&buf, 4, 0, 4),
    Err(Grayf32FrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn grayf32_frame_try_new_rejects_stride_too_small() {
  let buf = [0.0f32; 16];
  assert!(matches!(
    Grayf32Frame::try_new(&buf, 4, 4, 3),
    Err(Grayf32FrameError::YStrideTooSmall {
      width: 4,
      y_stride: 3
    })
  ));
}

#[test]
fn grayf32_frame_try_new_rejects_plane_too_short() {
  // stride=4, height=4 → need 16 elements; supply only 15
  let buf = [0.0f32; 15];
  assert!(matches!(
    Grayf32Frame::try_new(&buf, 4, 4, 4),
    Err(Grayf32FrameError::YPlaneTooShort {
      expected: 16,
      actual: 15
    })
  ));
}

#[test]
#[cfg(not(target_arch = "wasm32"))] // wasm uses panic=abort; catch_unwind requires unwinding
fn grayf32_frame_new_panics_on_invalid() {
  let result = std::panic::catch_unwind(|| {
    let buf = [0.0f32; 1];
    Grayf32Frame::new(&buf, 0, 1, 1);
  });
  assert!(result.is_err(), "expected panic on zero width");
}

#[test]
fn grayf32_frame_accessors_are_correct() {
  let buf: std::vec::Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
  let f = Grayf32Frame::new(&buf, 4, 4, 4);
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.y_stride(), 4);
  // y() returns the full backing slice
  assert_eq!(f.y().len(), 16);
  assert!((f.y()[5] - 0.5f32).abs() < 1e-6);
}
