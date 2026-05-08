use super::*;

// ---- Xyz12Frame --------------------------------------------------------
//
// Single-plane packed-XYZ at 12 bits per sample. `stride` is in **u16
// elements** (≥ 3 * width); `plane.len() >= stride * height` u16s.
// No width parity constraint. Out-of-range samples (upper 4 bits set)
// are tolerated at construction — masking happens inside every kernel.

#[test]
fn xyz12_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0_u16; 16 * 4 * 3];
  Xyz12LeFrame::try_new(&buf, 16, 4, 48).expect("valid");
}

#[test]
fn xyz12_frame_try_new_accepts_oversized_stride() {
  let buf = std::vec![0_u16; 64 * 4];
  Xyz12LeFrame::try_new(&buf, 16, 4, 64).expect("padded stride is valid");
}

#[test]
fn xyz12_frame_try_new_accepts_be_alias() {
  let buf = std::vec![0_u16; 16 * 4 * 3];
  let f = Xyz12BeFrame::try_new(&buf, 16, 4, 48).expect("valid");
  assert!(f.big_endian());
}

#[test]
fn xyz12_frame_try_new_accepts_out_of_range_samples() {
  // Samples with upper 4 bits set are permitted at construction; the
  // masking happens inside every row kernel.
  let buf = std::vec![0xFFFF_u16; 16 * 4 * 3];
  Xyz12LeFrame::try_new(&buf, 16, 4, 48).expect("upper-4-bits dirty values allowed");
}

#[test]
fn xyz12_frame_zero_width_rejected() {
  let buf = std::vec![0_u16; 16 * 4 * 3];
  assert!(matches!(
    Xyz12LeFrame::try_new(&buf, 0, 4, 48),
    Err(Xyz12FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
}

#[test]
fn xyz12_frame_zero_height_rejected() {
  let buf = std::vec![0_u16; 16 * 4 * 3];
  assert!(matches!(
    Xyz12LeFrame::try_new(&buf, 16, 0, 48),
    Err(Xyz12FrameError::ZeroDimension {
      width: 16,
      height: 0
    })
  ));
}

#[test]
fn xyz12_frame_stride_smaller_than_3w_rejected() {
  let buf = std::vec![0_u16; 16 * 4 * 3];
  assert!(matches!(
    Xyz12LeFrame::try_new(&buf, 16, 4, 47),
    Err(Xyz12FrameError::StrideTooSmall {
      min_stride: 48,
      stride: 47,
    })
  ));
}

#[test]
fn xyz12_frame_plane_too_short_rejected() {
  let small = std::vec![0_u16; 16 * 3];
  assert!(matches!(
    Xyz12LeFrame::try_new(&small, 16, 4, 48),
    Err(Xyz12FrameError::PlaneTooShort {
      expected: 192,
      actual: 48,
    })
  ));
}

#[test]
fn xyz12_frame_width_3x_overflow_rejected() {
  // 3 * width must fit in u32: width > u32::MAX / 3 trips WidthOverflow.
  let buf: [u16; 0] = [];
  let too_big = (u32::MAX / 3) + 1;
  assert!(matches!(
    Xyz12LeFrame::try_new(&buf, too_big, 1, u32::MAX),
    Err(Xyz12FrameError::WidthOverflow { width }) if width == too_big
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn xyz12_frame_geometry_overflow_rejected() {
  // 32-bit-only: `stride * height` overflows `usize`.
  let buf: [u16; 0] = [];
  let width: u32 = 0x5555; // 3 * width = 0xFFFF, ≤ stride
  let stride: u32 = 0x1_0000;
  let height: u32 = 0x1_0000; // stride * height = 2^32 → overflows usize on 32-bit
  let res = Xyz12LeFrame::try_new(&buf, width, height, stride);
  assert!(
    matches!(
      res,
      Err(Xyz12FrameError::GeometryOverflow {
        stride: 0x1_0000,
        rows: 0x1_0000,
      })
    ),
    "expected GeometryOverflow, got {:?}",
    res
  );
}

#[test]
#[should_panic(expected = "invalid Xyz12Frame")]
fn xyz12_frame_new_panics_on_invalid() {
  let buf = std::vec![0_u16; 10];
  let _ = Xyz12LeFrame::new(&buf, 16, 4, 48);
}

#[test]
fn xyz12_frame_accessors_round_trip() {
  let buf = std::vec![0x0800_u16; 8 * 2 * 3];
  let frame = Xyz12LeFrame::try_new(&buf, 8, 2, 24).expect("valid");
  assert_eq!(frame.width(), 8);
  assert_eq!(frame.height(), 2);
  assert_eq!(frame.stride(), 24);
  assert_eq!(frame.xyz().len(), 48);
  assert_eq!(frame.xyz()[0], 0x0800);
  assert!(!frame.big_endian());
}

#[test]
fn xyz12_frame_be_flag_distinguishes_aliases() {
  let buf = std::vec![0_u16; 8 * 2 * 3];
  let le = Xyz12LeFrame::try_new(&buf, 8, 2, 24).expect("valid");
  let be = Xyz12BeFrame::try_new(&buf, 8, 2, 24).expect("valid");
  assert!(!le.big_endian());
  assert!(be.big_endian());
}
