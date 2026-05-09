use super::*;

// ---- Uyyvyy411Frame (Tier 5.25) ----------------------------------------
//
// Single-plane 8-bit packed YUV 4:1:1 (DV legacy). 6 bytes per
// 4-pixel block, `stride >= width * 3 / 2` (12 bpp). Width must be a
// multiple of 4.

#[test]
fn uyyvyy411_frame_try_new_accepts_valid_tight() {
  // 16 px × 4 rows tightly packed: stride = 16 * 3 / 2 = 24.
  let buf = std::vec![0u8; 24 * 4];
  Uyyvyy411Frame::try_new(&buf, 16, 4, 24).expect("valid");
}

#[test]
fn uyyvyy411_frame_try_new_accepts_oversized_stride() {
  let buf = std::vec![0u8; 48 * 4];
  Uyyvyy411Frame::try_new(&buf, 16, 4, 48).expect("padded stride is valid");
}

#[test]
fn uyyvyy411_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 24 * 4];
  assert!(matches!(
    Uyyvyy411Frame::try_new(&buf, 0, 4, 24),
    Err(Uyyvyy411FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    Uyyvyy411Frame::try_new(&buf, 16, 0, 24),
    Err(Uyyvyy411FrameError::ZeroDimension {
      width: 16,
      height: 0
    })
  ));
}

#[test]
fn uyyvyy411_frame_try_new_rejects_width_not_multiple_of_4() {
  let buf = std::vec![0u8; 100];
  // 18 = even but not divisible by 4.
  assert!(matches!(
    Uyyvyy411Frame::try_new(&buf, 18, 4, 27),
    Err(Uyyvyy411FrameError::WidthNotMultipleOf4 { width: 18 })
  ));
  // 17 = odd.
  assert!(matches!(
    Uyyvyy411Frame::try_new(&buf, 17, 4, 26),
    Err(Uyyvyy411FrameError::WidthNotMultipleOf4 { width: 17 })
  ));
}

#[test]
fn uyyvyy411_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 24 * 4];
  assert!(matches!(
    Uyyvyy411Frame::try_new(&buf, 16, 4, 23),
    Err(Uyyvyy411FrameError::StrideTooSmall {
      min_stride: 24,
      stride: 23,
    })
  ));
}

#[test]
fn uyyvyy411_frame_try_new_rejects_short_plane() {
  // Need 24 * 4 = 96 bytes; supply 32.
  let small = std::vec![0u8; 32];
  assert!(matches!(
    Uyyvyy411Frame::try_new(&small, 16, 4, 24),
    Err(Uyyvyy411FrameError::PlaneTooShort {
      expected: 96,
      actual: 32,
    })
  ));
}

#[test]
#[should_panic(expected = "invalid Uyyvyy411Frame")]
fn uyyvyy411_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = Uyyvyy411Frame::new(&buf, 16, 4, 24);
}

#[test]
fn uyyvyy411_frame_accessors_return_constructor_args() {
  let buf = std::vec![0u8; 24 * 4];
  let f = Uyyvyy411Frame::try_new(&buf, 16, 4, 24).unwrap();
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 24);
  assert_eq!(f.uyyvyy().len(), 96);
}
