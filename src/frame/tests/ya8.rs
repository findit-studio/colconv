use crate::frame::{Ya8Frame, Ya8FrameError};

#[test]
fn ya8_frame_try_new_accepts_valid_tight() {
  // 4 px × 2 bytes/px × 4 rows = 32 bytes; stride = 8 (= width * 2)
  let buf = [0u8; 32];
  let f = Ya8Frame::try_new(&buf, 4, 4, 8).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 8);
  assert_eq!(f.packed().len(), 32);
}

#[test]
fn ya8_frame_try_new_accepts_padded_stride() {
  // stride=16 (padded), 4 px × 4 rows = 64 bytes
  let buf = [0u8; 64];
  Ya8Frame::try_new(&buf, 4, 4, 16).unwrap();
}

#[test]
fn ya8_frame_try_new_rejects_zero_width() {
  let buf = [0u8; 32];
  assert!(matches!(
    Ya8Frame::try_new(&buf, 0, 4, 8),
    Err(Ya8FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
}

#[test]
fn ya8_frame_try_new_rejects_zero_height() {
  let buf = [0u8; 32];
  assert!(matches!(
    Ya8Frame::try_new(&buf, 4, 0, 8),
    Err(Ya8FrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn ya8_frame_try_new_rejects_stride_too_small() {
  // width=4, min_stride=8; stride=7 is too small
  let buf = [0u8; 32];
  assert!(matches!(
    Ya8Frame::try_new(&buf, 4, 4, 7),
    Err(Ya8FrameError::StrideTooSmall {
      width: 4,
      stride: 7,
      min_stride: 8
    })
  ));
}

#[test]
fn ya8_frame_try_new_rejects_plane_too_short() {
  // stride=8, height=4 → need 32 bytes; supply 31
  let buf = [0u8; 31];
  assert!(matches!(
    Ya8Frame::try_new(&buf, 4, 4, 8),
    Err(Ya8FrameError::PlaneTooShort {
      expected: 32,
      actual: 31
    })
  ));
}

#[test]
fn ya8_frame_new_panics_on_invalid() {
  let result = std::panic::catch_unwind(|| {
    let buf = [0u8; 1];
    Ya8Frame::new(&buf, 4, 4, 8);
  });
  assert!(result.is_err(), "expected panic on plane too short");
}

#[test]
fn ya8_frame_accessors_are_correct() {
  // packed [Y=100, A=200, Y=50, A=150] for a 2×1 frame (stride=4)
  let buf: [u8; 4] = [100, 200, 50, 150];
  let f = Ya8Frame::new(&buf, 2, 1, 4);
  assert_eq!(f.width(), 2);
  assert_eq!(f.height(), 1);
  assert_eq!(f.stride(), 4);
  assert_eq!(f.packed(), &[100u8, 200, 50, 150]);
}
