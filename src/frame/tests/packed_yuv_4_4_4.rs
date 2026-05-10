use crate::frame::{
  Ayuv64BeFrame, Ayuv64FrameError, Ayuv64LeFrame, V30XFrame, V30XFrameError, V410BeFrame,
  V410FrameError, V410LeFrame, VuyaFrame, VuyaFrameError, VuyxFrame, VuyxFrameError, Xv36BeFrame,
  Xv36FrameError, Xv36LeFrame,
};

const fn zero_buf<const N: usize>() -> [u32; N] {
  [0u32; N]
}

#[test]
fn v410_frame_try_new_accepts_valid_tight() {
  // Tight stride: stride == width.
  let buf = zero_buf::<16>();
  let f = V410LeFrame::try_new(&buf, 4, 4, 4).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 4);
  assert_eq!(f.packed().len(), 16);
}

#[test]
fn v410_frame_try_new_accepts_oversized_stride() {
  let buf = zero_buf::<32>();
  V410LeFrame::try_new(&buf, 4, 4, 8).unwrap();
}

#[test]
fn v410_frame_try_new_rejects_zero_dimension() {
  let buf = zero_buf::<16>();
  assert!(matches!(
    V410LeFrame::try_new(&buf, 0, 4, 4),
    Err(V410FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    V410LeFrame::try_new(&buf, 4, 0, 4),
    Err(V410FrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn v410_frame_try_new_rejects_stride_too_small() {
  let buf = zero_buf::<16>();
  assert!(matches!(
    V410LeFrame::try_new(&buf, 4, 4, 3),
    Err(V410FrameError::StrideTooSmall {
      min_stride: 4,
      stride: 3
    })
  ));
}

#[test]
fn v410_frame_try_new_rejects_short_plane() {
  let buf = zero_buf::<8>();
  assert!(matches!(
    V410LeFrame::try_new(&buf, 4, 4, 4),
    Err(V410FrameError::PlaneTooShort {
      expected: 16,
      actual: 8
    })
  ));
}

#[test]
fn v410_frame_accessors_round_trip() {
  let buf = zero_buf::<32>();
  let f = V410LeFrame::try_new(&buf, 4, 4, 8).unwrap();
  assert_eq!(f.packed().len(), 32);
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 8);
}

#[test]
#[should_panic(expected = "invalid V410Frame:")]
fn v410_frame_new_panics_on_invalid() {
  let buf = zero_buf::<8>();
  let _ = V410LeFrame::new(&buf, 4, 4, 4); // PlaneTooShort
}

#[test]
fn v410_le_frame_default_is_le() {
  // Phase 4: default `<const BE: bool = false>` exposed via `is_be()`.
  let buf = zero_buf::<16>();
  let f = V410LeFrame::try_new(&buf, 4, 4, 4).unwrap();
  assert!(!f.is_be());
}

#[test]
fn v410_be_frame_alias_constructs() {
  // Phase 4: `V410BeFrame` alias resolves to `V410Frame<'_, true>`.
  let buf = zero_buf::<16>();
  let f = V410BeFrame::try_new(&buf, 4, 4, 4).unwrap();
  assert!(f.is_be());
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
}

#[test]
fn v30x_frame_try_new_accepts_valid_tight() {
  // Tight stride: stride == width.
  let buf = zero_buf::<16>();
  let f = V30XFrame::try_new(&buf, 4, 4, 4).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 4);
  assert_eq!(f.packed().len(), 16);
}

#[test]
fn v30x_frame_try_new_accepts_oversized_stride() {
  let buf = zero_buf::<32>();
  V30XFrame::try_new(&buf, 4, 4, 8).unwrap();
}

#[test]
fn v30x_frame_try_new_rejects_zero_dimension() {
  let buf = zero_buf::<16>();
  assert!(matches!(
    V30XFrame::try_new(&buf, 0, 4, 4),
    Err(V30XFrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    V30XFrame::try_new(&buf, 4, 0, 4),
    Err(V30XFrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn v30x_frame_try_new_rejects_stride_too_small() {
  let buf = zero_buf::<16>();
  assert!(matches!(
    V30XFrame::try_new(&buf, 4, 4, 3),
    Err(V30XFrameError::StrideTooSmall {
      min_stride: 4,
      stride: 3
    })
  ));
}

#[test]
fn v30x_frame_try_new_rejects_short_plane() {
  let buf = zero_buf::<8>();
  assert!(matches!(
    V30XFrame::try_new(&buf, 4, 4, 4),
    Err(V30XFrameError::PlaneTooShort {
      expected: 16,
      actual: 8
    })
  ));
}

#[test]
fn v30x_frame_accessors_round_trip() {
  let buf = zero_buf::<32>();
  let f = V30XFrame::try_new(&buf, 4, 4, 8).unwrap();
  assert_eq!(f.packed().len(), 32);
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 8);
}

#[test]
#[should_panic(expected = "invalid V30XFrame:")]
fn v30x_frame_new_panics_on_invalid() {
  let buf = zero_buf::<8>();
  let _ = V30XFrame::new(&buf, 4, 4, 4); // PlaneTooShort
}

#[test]
fn xv36_frame_try_new_accepts_valid_tight() {
  let buf = vec![0u16; 4 * 4 * 4]; // 4 px × 4 channels × 4 rows
  let f = Xv36LeFrame::try_new(&buf, 4, 4, 16).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 16);
  assert_eq!(f.packed().len(), 64);
}

#[test]
fn xv36_frame_try_new_accepts_oversized_stride() {
  let buf = vec![0u16; 4 * 4 * 8]; // stride=32 > width*4=16
  Xv36LeFrame::try_new(&buf, 4, 4, 32).unwrap();
}

#[test]
fn xv36_frame_try_new_rejects_zero_dimension() {
  let buf = vec![0u16; 16];
  assert!(matches!(
    Xv36LeFrame::try_new(&buf, 0, 4, 16),
    Err(Xv36FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    Xv36LeFrame::try_new(&buf, 4, 0, 16),
    Err(Xv36FrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn xv36_frame_try_new_rejects_stride_too_small() {
  let buf = vec![0u16; 64];
  // width=4, width*4=16; stride=12 < 16
  assert!(matches!(
    Xv36LeFrame::try_new(&buf, 4, 4, 12),
    Err(Xv36FrameError::StrideTooSmall {
      min_stride: 16,
      stride: 12
    })
  ));
}

#[test]
fn xv36_frame_try_new_rejects_short_plane() {
  let buf = vec![0u16; 32]; // need 16*4 = 64
  assert!(matches!(
    Xv36LeFrame::try_new(&buf, 4, 4, 16),
    Err(Xv36FrameError::PlaneTooShort {
      expected: 64,
      actual: 32
    })
  ));
}

#[test]
fn xv36_frame_try_new_checked_accepts_msb_aligned() {
  let mut buf = vec![0u16; 64];
  buf.fill(0xABC0); // low 4 bits = 0
  Xv36LeFrame::try_new_checked(&buf, 4, 4, 16).unwrap();
}

#[test]
fn xv36_frame_try_new_checked_rejects_low_bits_set() {
  let mut buf = vec![0u16; 64];
  buf[5] = 0xABCD; // low 4 bits = 0xD ≠ 0 (in active row range)
  assert!(matches!(
    Xv36LeFrame::try_new_checked(&buf, 4, 4, 16),
    Err(Xv36FrameError::SampleLowBitsSet)
  ));
}

#[test]
fn xv36_frame_accessors_round_trip() {
  let buf = vec![0u16; 64];
  let f = Xv36LeFrame::try_new(&buf, 4, 4, 16).unwrap();
  assert_eq!(f.packed().len(), 64);
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 16);
}

#[test]
fn xv36_le_frame_default_is_le() {
  // Phase 4: default `<const BE: bool = false>` exposed via `is_be()`.
  let buf = vec![0u16; 64];
  let f = Xv36LeFrame::try_new(&buf, 4, 4, 16).unwrap();
  assert!(!f.is_be());
}

#[test]
fn xv36_be_frame_alias_constructs() {
  // Phase 4: `Xv36BeFrame` alias resolves to `Xv36Frame<'_, true>`.
  let buf = vec![0u16; 64];
  let f = Xv36BeFrame::try_new(&buf, 4, 4, 16).unwrap();
  assert!(f.is_be());
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
}

#[test]
fn vuya_frame_try_new_accepts_valid_tight() {
  let buf = vec![0u8; 4 * 4 * 4]; // 4 px × 4 bytes × 4 rows
  let f = VuyaFrame::try_new(&buf, 4, 4, 16).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 16);
  assert_eq!(f.packed().len(), 64);
}

#[test]
fn vuya_frame_try_new_accepts_oversized_stride() {
  let buf = vec![0u8; 4 * 4 * 8]; // stride=32 > width*4=16
  VuyaFrame::try_new(&buf, 4, 4, 32).unwrap();
}

#[test]
fn vuya_frame_try_new_rejects_zero_dimension() {
  let buf = vec![0u8; 64];
  assert!(matches!(
    VuyaFrame::try_new(&buf, 0, 4, 16),
    Err(VuyaFrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    VuyaFrame::try_new(&buf, 4, 0, 16),
    Err(VuyaFrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn vuya_frame_try_new_rejects_stride_too_small() {
  let buf = vec![0u8; 64];
  // width=4, width*4=16 bytes; stride=12 < 16
  assert!(matches!(
    VuyaFrame::try_new(&buf, 4, 4, 12),
    Err(VuyaFrameError::StrideTooSmall {
      min_stride: 16,
      stride: 12
    })
  ));
}

#[test]
fn vuya_frame_try_new_rejects_short_plane() {
  let buf = vec![0u8; 32]; // need 16*4 = 64 bytes
  assert!(matches!(
    VuyaFrame::try_new(&buf, 4, 4, 16),
    Err(VuyaFrameError::PlaneTooShort {
      expected: 64,
      actual: 32
    })
  ));
}

#[test]
fn vuya_frame_accessors_round_trip() {
  let buf = vec![0u8; 128]; // stride=32, height=4 → 128 bytes
  let f = VuyaFrame::try_new(&buf, 4, 4, 32).unwrap();
  assert_eq!(f.packed().len(), 128);
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 32);
}

#[test]
fn vuyx_frame_try_new_accepts_valid_tight() {
  let buf = vec![0u8; 4 * 4 * 4]; // 4 px × 4 bytes × 4 rows
  let f = VuyxFrame::try_new(&buf, 4, 4, 16).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 16);
  assert_eq!(f.packed().len(), 64);
}

#[test]
fn vuyx_frame_try_new_accepts_oversized_stride() {
  let buf = vec![0u8; 4 * 4 * 8]; // stride=32 > width*4=16
  VuyxFrame::try_new(&buf, 4, 4, 32).unwrap();
}

#[test]
fn vuyx_frame_try_new_rejects_zero_dimension() {
  let buf = vec![0u8; 64];
  assert!(matches!(
    VuyxFrame::try_new(&buf, 0, 4, 16),
    Err(VuyxFrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    VuyxFrame::try_new(&buf, 4, 0, 16),
    Err(VuyxFrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn vuyx_frame_try_new_rejects_stride_too_small() {
  let buf = vec![0u8; 64];
  // width=4, width*4=16 bytes; stride=12 < 16
  assert!(matches!(
    VuyxFrame::try_new(&buf, 4, 4, 12),
    Err(VuyxFrameError::StrideTooSmall {
      min_stride: 16,
      stride: 12
    })
  ));
}

#[test]
fn vuyx_frame_try_new_rejects_short_plane() {
  let buf = vec![0u8; 32]; // need 16*4 = 64 bytes
  assert!(matches!(
    VuyxFrame::try_new(&buf, 4, 4, 16),
    Err(VuyxFrameError::PlaneTooShort {
      expected: 64,
      actual: 32
    })
  ));
}

#[test]
fn vuyx_frame_accessors_round_trip() {
  let buf = vec![0u8; 128]; // stride=32, height=4 → 128 bytes
  let f = VuyxFrame::try_new(&buf, 4, 4, 32).unwrap();
  assert_eq!(f.packed().len(), 128);
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 32);
}

#[test]
fn ayuv64_frame_try_new_accepts_valid_tight() {
  let buf = vec![0u16; 4 * 4 * 4]; // 4 px × 4 u16 channels × 4 rows
  let f = Ayuv64LeFrame::try_new(&buf, 4, 4, 16).unwrap();
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 16);
  assert_eq!(f.packed().len(), 64);
}

#[test]
fn ayuv64_frame_try_new_accepts_oversized_stride() {
  let buf = vec![0u16; 4 * 4 * 8]; // stride=32 > width*4=16
  Ayuv64LeFrame::try_new(&buf, 4, 4, 32).unwrap();
}

#[test]
fn ayuv64_frame_try_new_rejects_zero_dimension() {
  let buf = vec![0u16; 64];
  assert!(matches!(
    Ayuv64LeFrame::try_new(&buf, 0, 4, 16),
    Err(Ayuv64FrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    Ayuv64LeFrame::try_new(&buf, 4, 0, 16),
    Err(Ayuv64FrameError::ZeroDimension {
      width: 4,
      height: 0
    })
  ));
}

#[test]
fn ayuv64_frame_try_new_rejects_stride_too_small() {
  let buf = vec![0u16; 64];
  // width=4, width*4=16 u16 elements; stride=12 < 16
  assert!(matches!(
    Ayuv64LeFrame::try_new(&buf, 4, 4, 12),
    Err(Ayuv64FrameError::StrideTooSmall {
      min_stride: 16,
      stride: 12
    })
  ));
}

#[test]
fn ayuv64_frame_try_new_rejects_short_plane() {
  let buf = vec![0u16; 32]; // need 16*4 = 64 u16 elements
  assert!(matches!(
    Ayuv64LeFrame::try_new(&buf, 4, 4, 16),
    Err(Ayuv64FrameError::PlaneTooShort {
      expected: 64,
      actual: 32
    })
  ));
}

#[test]
fn ayuv64_frame_accessors_round_trip() {
  let buf = vec![0u16; 128]; // stride=32, height=4 → 128 u16 elements
  let f = Ayuv64LeFrame::try_new(&buf, 4, 4, 32).unwrap();
  assert_eq!(f.packed().len(), 128);
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
  assert_eq!(f.stride(), 32);
}

#[test]
fn ayuv64_le_frame_default_is_le() {
  // Phase 4: default `<const BE: bool = false>` exposed via `is_be()`.
  let buf = vec![0u16; 64];
  let f = Ayuv64LeFrame::try_new(&buf, 4, 4, 16).unwrap();
  assert!(!f.is_be());
}

#[test]
fn ayuv64_be_frame_alias_constructs() {
  // Phase 4: `Ayuv64BeFrame` alias resolves to `Ayuv64Frame<'_, true>`.
  let buf = vec![0u16; 64];
  let f = Ayuv64BeFrame::try_new(&buf, 4, 4, 16).unwrap();
  assert!(f.is_be());
  assert_eq!(f.width(), 4);
  assert_eq!(f.height(), 4);
}
