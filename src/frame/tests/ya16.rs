use crate::frame::{Ya16Frame, Ya16FrameError};

#[test]
fn ya16_frame_try_new_accepts_valid_tight() {
  // 4 px × 2 u16/px × 4 rows = 32 u16 elements; stride = 8
  let buf = [0u16; 32];
  let f = Ya16Frame::try_new(&buf, 4, 4, 8).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 8);
  assert_eq!(f.packed().len(), 32);
}

#[test]
fn ya16_frame_try_new_accepts_padded_stride() {
  // stride=16, 4 px × 4 rows = 64 u16 elements
  let buf = [0u16; 64];
  Ya16Frame::try_new(&buf, 4, 4, 16).unwrap();
}

#[test]
fn ya16_frame_try_new_rejects_zero_width() {
  let buf = [0u16; 32];
  assert!(matches!(
    Ya16Frame::try_new(&buf, 0, 4, 8),
    Err(Ya16FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
}

#[test]
fn ya16_frame_try_new_rejects_zero_height() {
  let buf = [0u16; 32];
  assert!(matches!(
    Ya16Frame::try_new(&buf, 4, 0, 8),
    Err(Ya16FrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn ya16_frame_try_new_rejects_stride_too_small() {
  // width=4, min_stride=8; stride=7 is too small
  let buf = [0u16; 32];
  assert!(matches!(
    Ya16Frame::try_new(&buf, 4, 4, 7),
    Err(Ya16FrameError::StrideTooSmall {
      width: 4,
      stride: 7,
      min_stride: 8
    })
  ));
}

#[test]
fn ya16_frame_try_new_rejects_plane_too_short() {
  // stride=8, height=4 → need 32 u16 elements; supply 31
  let buf = [0u16; 31];
  assert!(matches!(
    Ya16Frame::try_new(&buf, 4, 4, 8),
    Err(Ya16FrameError::PlaneTooShort {
      expected: 32,
      actual: 31
    })
  ));
}

#[test]
fn ya16_frame_new_panics_on_invalid() {
  let result = std::panic::catch_unwind(|| {
    let buf = [0u16; 1];
    Ya16Frame::new(&buf, 4, 4, 8);
  });
  assert!(result.is_err(), "expected panic on plane too short");
}

#[test]
fn ya16_frame_accessors_are_correct() {
  // [Y=0x8000, A=0x4000, Y=0x1000, A=0x0800] for a 2×1 frame
  let buf: [u16; 4] = [0x8000, 0x4000, 0x1000, 0x0800];
  let f = Ya16Frame::new(&buf, 2, 1, 4);
  assert_eq!(f.width(), 2);
  assert_eq!(f.height(), 1);
  assert_eq!(f.stride(), 4);
  assert_eq!(f.packed(), &[0x8000u16, 0x4000, 0x1000, 0x0800]);
}
