use super::*;

// ---- Rgb565Frame -----------------------------------------------------------

#[test]
fn rgb565_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  Rgb565Frame::try_new(&buf, 16, 4, 32).expect("valid tight stride");
}

#[test]
fn rgb565_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 32];
  assert!(matches!(
    Rgb565Frame::try_new(&buf, 0, 0, 0),
    Err(LegacyRgbFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn rgb565_frame_try_new_rejects_width_overflow() {
  let buf = std::vec![0u8; 4];
  let w = u32::MAX / 2 + 1;
  assert!(matches!(
    Rgb565Frame::try_new(&buf, w, 1, w),
    Err(LegacyRgbFrameError::WidthOverflow { .. })
  ));
}

#[test]
fn rgb565_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  // min stride = 2*16 = 32; supply 31
  assert!(matches!(
    Rgb565Frame::try_new(&buf, 16, 4, 31),
    Err(LegacyRgbFrameError::StrideTooSmall { .. })
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn rgb565_frame_try_new_rejects_geometry_overflow() {
  // Only meaningful on 32-bit targets where `stride * height` as `usize` can overflow.
  // stride=0x1_0000, height=0x1_0000 → product = 2^32 → overflows 32-bit usize.
  let buf: [u8; 0] = [];
  assert!(matches!(
    Rgb565Frame::try_new(&buf, 1, 0x1_0000, 0x1_0000),
    Err(LegacyRgbFrameError::GeometryOverflow { .. })
  ));
}

#[test]
fn rgb565_frame_try_new_rejects_plane_too_short() {
  // Valid geometry: 16 px wide, 4 rows, stride=32 → need 128 bytes; give 127
  let buf = std::vec![0u8; 127];
  assert!(matches!(
    Rgb565Frame::try_new(&buf, 16, 4, 32),
    Err(LegacyRgbFrameError::PlaneTooShort { .. })
  ));
}

// ---- Bgr565Frame -----------------------------------------------------------

#[test]
fn bgr565_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  Bgr565Frame::try_new(&buf, 16, 4, 32).expect("valid tight stride");
}

#[test]
fn bgr565_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 32];
  assert!(matches!(
    Bgr565Frame::try_new(&buf, 0, 0, 0),
    Err(LegacyRgbFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn bgr565_frame_try_new_rejects_width_overflow() {
  let buf = std::vec![0u8; 4];
  let w = u32::MAX / 2 + 1;
  assert!(matches!(
    Bgr565Frame::try_new(&buf, w, 1, w),
    Err(LegacyRgbFrameError::WidthOverflow { .. })
  ));
}

#[test]
fn bgr565_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  assert!(matches!(
    Bgr565Frame::try_new(&buf, 16, 4, 31),
    Err(LegacyRgbFrameError::StrideTooSmall { .. })
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn bgr565_frame_try_new_rejects_geometry_overflow() {
  let buf: [u8; 0] = [];
  assert!(matches!(
    Bgr565Frame::try_new(&buf, 1, 0x1_0000, 0x1_0000),
    Err(LegacyRgbFrameError::GeometryOverflow { .. })
  ));
}

#[test]
fn bgr565_frame_try_new_rejects_plane_too_short() {
  let buf = std::vec![0u8; 127];
  assert!(matches!(
    Bgr565Frame::try_new(&buf, 16, 4, 32),
    Err(LegacyRgbFrameError::PlaneTooShort { .. })
  ));
}

// ---- Rgb555Frame -----------------------------------------------------------

#[test]
fn rgb555_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  Rgb555Frame::try_new(&buf, 16, 4, 32).expect("valid tight stride");
}

#[test]
fn rgb555_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 32];
  assert!(matches!(
    Rgb555Frame::try_new(&buf, 0, 0, 0),
    Err(LegacyRgbFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn rgb555_frame_try_new_rejects_width_overflow() {
  let buf = std::vec![0u8; 4];
  let w = u32::MAX / 2 + 1;
  assert!(matches!(
    Rgb555Frame::try_new(&buf, w, 1, w),
    Err(LegacyRgbFrameError::WidthOverflow { .. })
  ));
}

#[test]
fn rgb555_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  assert!(matches!(
    Rgb555Frame::try_new(&buf, 16, 4, 31),
    Err(LegacyRgbFrameError::StrideTooSmall { .. })
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn rgb555_frame_try_new_rejects_geometry_overflow() {
  let buf: [u8; 0] = [];
  assert!(matches!(
    Rgb555Frame::try_new(&buf, 1, 0x1_0000, 0x1_0000),
    Err(LegacyRgbFrameError::GeometryOverflow { .. })
  ));
}

#[test]
fn rgb555_frame_try_new_rejects_plane_too_short() {
  let buf = std::vec![0u8; 127];
  assert!(matches!(
    Rgb555Frame::try_new(&buf, 16, 4, 32),
    Err(LegacyRgbFrameError::PlaneTooShort { .. })
  ));
}

// ---- Bgr555Frame -----------------------------------------------------------

#[test]
fn bgr555_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  Bgr555Frame::try_new(&buf, 16, 4, 32).expect("valid tight stride");
}

#[test]
fn bgr555_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 32];
  assert!(matches!(
    Bgr555Frame::try_new(&buf, 0, 0, 0),
    Err(LegacyRgbFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn bgr555_frame_try_new_rejects_width_overflow() {
  let buf = std::vec![0u8; 4];
  let w = u32::MAX / 2 + 1;
  assert!(matches!(
    Bgr555Frame::try_new(&buf, w, 1, w),
    Err(LegacyRgbFrameError::WidthOverflow { .. })
  ));
}

#[test]
fn bgr555_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  assert!(matches!(
    Bgr555Frame::try_new(&buf, 16, 4, 31),
    Err(LegacyRgbFrameError::StrideTooSmall { .. })
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn bgr555_frame_try_new_rejects_geometry_overflow() {
  let buf: [u8; 0] = [];
  assert!(matches!(
    Bgr555Frame::try_new(&buf, 1, 0x1_0000, 0x1_0000),
    Err(LegacyRgbFrameError::GeometryOverflow { .. })
  ));
}

#[test]
fn bgr555_frame_try_new_rejects_plane_too_short() {
  let buf = std::vec![0u8; 127];
  assert!(matches!(
    Bgr555Frame::try_new(&buf, 16, 4, 32),
    Err(LegacyRgbFrameError::PlaneTooShort { .. })
  ));
}

// ---- Rgb444Frame -----------------------------------------------------------

#[test]
fn rgb444_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  Rgb444Frame::try_new(&buf, 16, 4, 32).expect("valid tight stride");
}

#[test]
fn rgb444_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 32];
  assert!(matches!(
    Rgb444Frame::try_new(&buf, 0, 0, 0),
    Err(LegacyRgbFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn rgb444_frame_try_new_rejects_width_overflow() {
  let buf = std::vec![0u8; 4];
  let w = u32::MAX / 2 + 1;
  assert!(matches!(
    Rgb444Frame::try_new(&buf, w, 1, w),
    Err(LegacyRgbFrameError::WidthOverflow { .. })
  ));
}

#[test]
fn rgb444_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  assert!(matches!(
    Rgb444Frame::try_new(&buf, 16, 4, 31),
    Err(LegacyRgbFrameError::StrideTooSmall { .. })
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn rgb444_frame_try_new_rejects_geometry_overflow() {
  let buf: [u8; 0] = [];
  assert!(matches!(
    Rgb444Frame::try_new(&buf, 1, 0x1_0000, 0x1_0000),
    Err(LegacyRgbFrameError::GeometryOverflow { .. })
  ));
}

#[test]
fn rgb444_frame_try_new_rejects_plane_too_short() {
  let buf = std::vec![0u8; 127];
  assert!(matches!(
    Rgb444Frame::try_new(&buf, 16, 4, 32),
    Err(LegacyRgbFrameError::PlaneTooShort { .. })
  ));
}

// ---- Bgr444Frame -----------------------------------------------------------

#[test]
fn bgr444_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  Bgr444Frame::try_new(&buf, 16, 4, 32).expect("valid tight stride");
}

#[test]
fn bgr444_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![0u8; 32];
  assert!(matches!(
    Bgr444Frame::try_new(&buf, 0, 0, 0),
    Err(LegacyRgbFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn bgr444_frame_try_new_rejects_width_overflow() {
  let buf = std::vec![0u8; 4];
  let w = u32::MAX / 2 + 1;
  assert!(matches!(
    Bgr444Frame::try_new(&buf, w, 1, w),
    Err(LegacyRgbFrameError::WidthOverflow { .. })
  ));
}

#[test]
fn bgr444_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u8; 16 * 2 * 4];
  assert!(matches!(
    Bgr444Frame::try_new(&buf, 16, 4, 31),
    Err(LegacyRgbFrameError::StrideTooSmall { .. })
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn bgr444_frame_try_new_rejects_geometry_overflow() {
  let buf: [u8; 0] = [];
  assert!(matches!(
    Bgr444Frame::try_new(&buf, 1, 0x1_0000, 0x1_0000),
    Err(LegacyRgbFrameError::GeometryOverflow { .. })
  ));
}

#[test]
fn bgr444_frame_try_new_rejects_plane_too_short() {
  let buf = std::vec![0u8; 127];
  assert!(matches!(
    Bgr444Frame::try_new(&buf, 16, 4, 32),
    Err(LegacyRgbFrameError::PlaneTooShort { .. })
  ));
}
