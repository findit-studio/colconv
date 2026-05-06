use super::*;

// ---- Rgb48Frame tests --------------------------------------------------------

#[test]
fn rgb48_try_new_happy_path() {
  // width=2, stride=6, height=3 → plane needs 18 u16 elements
  let buf = std::vec![0u16; 18];
  let f = Rgb48Frame::try_new(&buf, 2, 3, 6).unwrap();
  assert_eq!(f.width(), 2);
  assert_eq!(f.height(), 3);
  assert_eq!(f.stride(), 6);
  assert_eq!(f.rgb48().len(), 18);
}

#[test]
fn rgb48_stride_too_small() {
  let buf = std::vec![0u16; 18];
  // stride=5 < 3*2=6
  assert!(Rgb48Frame::try_new(&buf, 2, 3, 5).is_err());
}

#[test]
fn rgb48_plane_too_short() {
  // stride=6, height=3 → need 18; supply only 17
  let buf = std::vec![0u16; 17];
  assert!(Rgb48Frame::try_new(&buf, 2, 3, 6).is_err());
}

#[test]
fn rgb48_zero_dimension() {
  let buf = std::vec![0u16; 18];
  assert!(Rgb48Frame::try_new(&buf, 0, 3, 6).is_err());
  assert!(Rgb48Frame::try_new(&buf, 2, 0, 6).is_err());
}

// ---- Bgr48Frame tests --------------------------------------------------------

#[test]
fn bgr48_try_new_happy_path() {
  let buf = std::vec![0u16; 18];
  let f = Bgr48Frame::try_new(&buf, 2, 3, 6).unwrap();
  assert_eq!(f.width(), 2);
  assert_eq!(f.height(), 3);
  assert_eq!(f.stride(), 6);
  assert_eq!(f.bgr48().len(), 18);
}

#[test]
fn bgr48_stride_too_small() {
  let buf = std::vec![0u16; 18];
  assert!(Bgr48Frame::try_new(&buf, 2, 3, 5).is_err());
}

#[test]
fn bgr48_plane_too_short() {
  let buf = std::vec![0u16; 17];
  assert!(Bgr48Frame::try_new(&buf, 2, 3, 6).is_err());
}

#[test]
fn bgr48_zero_dimension() {
  let buf = std::vec![0u16; 18];
  assert!(Bgr48Frame::try_new(&buf, 0, 3, 6).is_err());
  assert!(Bgr48Frame::try_new(&buf, 2, 0, 6).is_err());
}

// ---- Rgba64Frame tests -------------------------------------------------------

#[test]
fn rgba64_try_new_happy_path() {
  // width=2, stride=8 (4*2), height=3 → plane needs 24 u16 elements
  let buf = std::vec![0u16; 24];
  let f = Rgba64Frame::try_new(&buf, 2, 3, 8).unwrap();
  assert_eq!(f.width(), 2);
  assert_eq!(f.height(), 3);
  assert_eq!(f.stride(), 8);
  assert_eq!(f.rgba64().len(), 24);
}

#[test]
fn rgba64_stride_too_small() {
  let buf = std::vec![0u16; 24];
  // stride=7 < 4*2=8
  assert!(Rgba64Frame::try_new(&buf, 2, 3, 7).is_err());
}

#[test]
fn rgba64_plane_too_short() {
  // stride=8, height=3 → need 24; supply only 23
  let buf = std::vec![0u16; 23];
  assert!(Rgba64Frame::try_new(&buf, 2, 3, 8).is_err());
}

#[test]
fn rgba64_zero_dimension() {
  let buf = std::vec![0u16; 24];
  assert!(Rgba64Frame::try_new(&buf, 0, 3, 8).is_err());
  assert!(Rgba64Frame::try_new(&buf, 2, 0, 8).is_err());
}

// ---- Rgb48Frame overflow tests -----------------------------------------------

#[test]
fn rgb48_try_new_rejects_width_overflow() {
  let buf = std::vec![0u16; 0];
  let too_big = (u32::MAX / 3) + 1;
  assert!(matches!(
    Rgb48Frame::try_new(&buf, too_big, 1, u32::MAX),
    Err(Rgb48FrameError::WidthOverflow { width }) if width == too_big
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn rgb48_try_new_rejects_geometry_overflow() {
  // Only meaningful on 32-bit targets (wasm32, i686) where
  // `stride * height` as `usize` can overflow. Pick a width small
  // enough that `3 * width <= stride` so we pass the StrideTooSmall
  // check and reach the geometry-overflow check.
  let buf: [u16; 0] = [];
  let width: u32 = 0x5555; // 3 * width = 0xFFFF, ≤ stride
  let stride: u32 = 0x1_0000;
  let height: u32 = 0x1_0000; // stride * height = 2^32 → overflows usize on 32-bit
  let res = Rgb48Frame::try_new(&buf, width, height, stride);
  assert!(
    matches!(
      res,
      Err(Rgb48FrameError::GeometryOverflow {
        stride: 0x1_0000,
        rows: 0x1_0000,
      })
    ),
    "expected GeometryOverflow, got {:?}",
    res
  );
}

// ---- Rgba64Frame overflow tests ----------------------------------------------

#[test]
fn rgba64_try_new_rejects_width_overflow() {
  let buf = std::vec![0u16; 0];
  let too_big = (u32::MAX / 4) + 1;
  assert!(matches!(
    Rgba64Frame::try_new(&buf, too_big, 1, u32::MAX),
    Err(Rgba64FrameError::WidthOverflow { width }) if width == too_big
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn rgba64_try_new_rejects_geometry_overflow() {
  // Only meaningful on 32-bit targets (wasm32, i686) where
  // `stride * height` as `usize` can overflow. Pick a width small
  // enough that `4 * width <= stride` so we pass the StrideTooSmall
  // check and reach the geometry-overflow check.
  let buf: [u16; 0] = [];
  let width: u32 = 0x3FFF; // 4 * width = 0xFFFC, ≤ stride
  let stride: u32 = 0x1_0000;
  let height: u32 = 0x1_0000; // stride * height = 2^32 → overflows usize on 32-bit
  let res = Rgba64Frame::try_new(&buf, width, height, stride);
  assert!(
    matches!(
      res,
      Err(Rgba64FrameError::GeometryOverflow {
        stride: 0x1_0000,
        rows: 0x1_0000,
      })
    ),
    "expected GeometryOverflow, got {:?}",
    res
  );
}

// ---- Bgra64Frame tests -------------------------------------------------------

#[test]
fn bgra64_try_new_happy_path() {
  let buf = std::vec![0u16; 24];
  let f = Bgra64Frame::try_new(&buf, 2, 3, 8).unwrap();
  assert_eq!(f.width(), 2);
  assert_eq!(f.height(), 3);
  assert_eq!(f.stride(), 8);
  assert_eq!(f.bgra64().len(), 24);
}

#[test]
fn bgra64_stride_too_small() {
  let buf = std::vec![0u16; 24];
  assert!(Bgra64Frame::try_new(&buf, 2, 3, 7).is_err());
}

#[test]
fn bgra64_plane_too_short() {
  let buf = std::vec![0u16; 23];
  assert!(Bgra64Frame::try_new(&buf, 2, 3, 8).is_err());
}

#[test]
fn bgra64_zero_dimension() {
  let buf = std::vec![0u16; 24];
  assert!(Bgra64Frame::try_new(&buf, 0, 3, 8).is_err());
  assert!(Bgra64Frame::try_new(&buf, 2, 0, 8).is_err());
}
