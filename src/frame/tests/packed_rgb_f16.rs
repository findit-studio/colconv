use super::*;

// ---- Rgbf16Frame -------------------------------------------------------
//
// Single-plane packed half-precision float RGB. `stride` is in **`f16` elements**
// (≥ 3 * width); `plane.len() >= stride * height` `f16`s. No width
// parity constraint. HDR / negative values are permitted in the buffer
// — validation is purely shape-based.

#[test]
fn rgbf16_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![half::f16::ZERO; 16 * 4 * 3];
  Rgbf16LeFrame::try_new(&buf, 16, 4, 48).expect("valid");
}

#[test]
fn rgbf16_frame_try_new_accepts_oversized_stride() {
  let buf = std::vec![half::f16::ZERO; 64 * 4];
  Rgbf16LeFrame::try_new(&buf, 16, 4, 64).expect("padded stride is valid");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_frame_try_new_accepts_hdr_and_negative_values() {
  // Out-of-[0,1] f16 values are permitted; only shape is validated.
  let buf = std::vec![half::f16::from_f32(10.0); 16 * 4 * 3];
  Rgbf16LeFrame::try_new(&buf, 16, 4, 48).expect("HDR values allowed");
  let neg = std::vec![half::f16::from_f32(-1.0); 16 * 4 * 3];
  Rgbf16LeFrame::try_new(&neg, 16, 4, 48).expect("negative values allowed");
}

#[test]
fn rgbf16_frame_try_new_rejects_zero_dimension() {
  let buf = std::vec![half::f16::ZERO; 16 * 4 * 3];
  assert!(matches!(
    Rgbf16LeFrame::try_new(&buf, 0, 4, 48),
    Err(Rgbf16FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    Rgbf16LeFrame::try_new(&buf, 16, 0, 48),
    Err(Rgbf16FrameError::ZeroDimension {
      width: 16,
      height: 0
    })
  ));
}

#[test]
fn rgbf16_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![half::f16::ZERO; 16 * 4 * 3];
  assert!(matches!(
    Rgbf16LeFrame::try_new(&buf, 16, 4, 47),
    Err(Rgbf16FrameError::StrideTooSmall {
      min_stride: 48,
      stride: 47,
    })
  ));
}

#[test]
fn rgbf16_frame_try_new_rejects_short_plane() {
  let small = std::vec![half::f16::ZERO; 16 * 3];
  assert!(matches!(
    Rgbf16LeFrame::try_new(&small, 16, 4, 48),
    Err(Rgbf16FrameError::PlaneTooShort {
      expected: 192,
      actual: 48,
    })
  ));
}

#[test]
fn rgbf16_frame_try_new_rejects_width_overflow() {
  // 3 * width must fit in u32: width > u32::MAX / 3 trips WidthOverflow.
  let buf = std::vec![half::f16::ZERO; 0];
  let too_big = (u32::MAX / 3) + 1;
  assert!(matches!(
    Rgbf16LeFrame::try_new(&buf, too_big, 1, u32::MAX),
    Err(Rgbf16FrameError::WidthOverflow { width }) if width == too_big
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn rgbf16_frame_try_new_rejects_geometry_overflow() {
  // Only meaningful on 32-bit targets (wasm32, i686) where
  // `stride * height` as `usize` can overflow. Pick a width small
  // enough that `3 * width <= stride` so we pass the StrideTooSmall
  // check and reach the geometry-overflow check.
  let buf: [half::f16; 0] = [];
  let width: u32 = 0x5555; // 3 * width = 0xFFFF, ≤ stride
  let stride: u32 = 0x1_0000;
  let height: u32 = 0x1_0000; // stride * height = 2^32 → overflows usize on 32-bit
  let res = Rgbf16LeFrame::try_new(&buf, width, height, stride);
  assert!(
    matches!(
      res,
      Err(Rgbf16FrameError::GeometryOverflow {
        stride: 0x1_0000,
        rows: 0x1_0000,
      })
    ),
    "expected GeometryOverflow, got {:?}",
    res
  );
}

#[test]
#[should_panic(expected = "invalid Rgbf16Frame")]
fn rgbf16_frame_new_panics_on_invalid() {
  let buf = std::vec![half::f16::ZERO; 10];
  let _ = Rgbf16LeFrame::new(&buf, 16, 4, 48);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "half::f16::from_f32 uses inline asm (fcvt) unsupported by Miri"
)]
fn rgbf16_frame_accessors_round_trip() {
  let val = half::f16::from_f32(0.25);
  let buf = std::vec![val; 8 * 2 * 3];
  let frame = Rgbf16LeFrame::try_new(&buf, 8, 2, 24).expect("valid");
  assert_eq!(frame.width(), 8);
  assert_eq!(frame.height(), 2);
  assert_eq!(frame.stride(), 24);
  assert_eq!(frame.rgb().len(), 48);
  assert_eq!(frame.rgb()[0], val);
  // Phase 4: default `<const BE: bool = false>` exposed via `is_be()`.
  assert!(!frame.is_be());
}

#[test]
fn rgbf16_be_frame_alias_constructs() {
  // Phase 4: `Rgbf16BeFrame` alias resolves to `Rgbf16Frame<'_, true>`.
  let buf = std::vec![half::f16::ZERO; 16 * 4 * 3];
  let f = Rgbf16BeFrame::try_new(&buf, 16, 4, 48).unwrap();
  assert!(f.is_be());
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 4);
}
