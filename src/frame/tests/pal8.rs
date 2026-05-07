use super::*;

/// Shared zero-filled palette for tests that don't care about palette contents.
static PALETTE: [[u8; 4]; 256] = [[0u8; 4]; 256];

#[test]
fn pal8_try_new_happy_path() {
  // stride == width; data exactly covers the frame
  let data = std::vec![0u8; 4 * 3];
  let f = Pal8Frame::try_new(&data, &PALETTE, 4, 3, 4).expect("valid frame");
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 3);
  assert_eq!(f.stride(), 4);
  assert_eq!(f.data().len(), 12);
}

#[test]
fn pal8_try_new_zero_dimension() {
  let data = std::vec![0u8; 16];
  let e = Pal8Frame::try_new(&data, &PALETTE, 0, 4, 4).unwrap_err();
  assert!(matches!(
    e,
    Pal8FrameError::ZeroDimension {
      width: 0,
      height: 4
    }
  ));
  let e = Pal8Frame::try_new(&data, &PALETTE, 4, 0, 4).unwrap_err();
  assert!(matches!(
    e,
    Pal8FrameError::ZeroDimension {
      width: 4,
      height: 0
    }
  ));
}

#[test]
fn pal8_try_new_stride_too_small() {
  let data = std::vec![0u8; 16];
  let e = Pal8Frame::try_new(&data, &PALETTE, 8, 2, 4).unwrap_err();
  assert!(matches!(
    e,
    Pal8FrameError::StrideTooSmall {
      width: 8,
      stride: 4
    }
  ));
}

#[test]
fn pal8_try_new_plane_too_short() {
  // stride * height = 4 * 3 = 12, but data is only 11 bytes
  let data = std::vec![0u8; 11];
  let e = Pal8Frame::try_new(&data, &PALETTE, 4, 3, 4).unwrap_err();
  assert!(matches!(
    e,
    Pal8FrameError::PlaneTooShort {
      expected: 12,
      actual: 11
    }
  ));
}

#[test]
fn pal8_try_new_stride_wider_than_width() {
  // stride = 8 > width = 4; data must cover stride * height = 8 * 3 = 24
  let data = std::vec![0u8; 24];
  let f = Pal8Frame::try_new(&data, &PALETTE, 4, 3, 8).expect("padded stride valid");
  assert_eq!(f.width(), 4);
  assert_eq!(f.stride(), 8);
}

#[test]
#[should_panic(expected = "invalid Pal8Frame")]
fn pal8_new_panics_on_invalid() {
  let data = std::vec![0u8; 10];
  let _ = Pal8Frame::new(&data, &PALETTE, 16, 8, 16);
}
