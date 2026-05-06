use super::*;

// ---- Gbrpf32Frame ----------------------------------------------------------
// Three f32 planes. Stride in elements. DimensionOverflow uses i32::MAX + 1.

#[test]
fn gbrpf32_frame_try_new_accepts_valid_tight() {
  // stride == width, planes exactly cover the frame.
  let g = std::vec![0.0f32; 8 * 4];
  let b = std::vec![0.0f32; 8 * 4];
  let r = std::vec![0.0f32; 8 * 4];
  let f = Gbrpf32Frame::try_new(&g, &b, &r, 8, 4, 8, 8, 8).expect("valid tight frame");
  assert_eq!(f.width(), 8);
  assert_eq!(f.height(), 4);
  assert_eq!(f.g_stride(), 8);
  assert_eq!(f.b_stride(), 8);
  assert_eq!(f.r_stride(), 8);
}

#[test]
fn gbrpf32_frame_try_new_rejects_zero_dimension() {
  let p = std::vec![0.0f32; 16];
  assert!(matches!(
    Gbrpf32Frame::try_new(&p, &p, &p, 0, 4, 8, 8, 8),
    Err(GbrFloatFrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
  assert!(matches!(
    Gbrpf32Frame::try_new(&p, &p, &p, 8, 0, 8, 8, 8),
    Err(GbrFloatFrameError::ZeroDimension {
      width: 8,
      height: 0
    })
  ));
}

#[test]
fn gbrpf32_frame_try_new_rejects_stride_below_width() {
  let p = std::vec![0.0f32; 8 * 4];
  // G stride too small
  assert!(matches!(
    Gbrpf32Frame::try_new(&p, &p, &p, 8, 4, 7, 8, 8),
    Err(GbrFloatFrameError::StrideBelowWidth {
      plane: "g",
      stride: 7,
      width: 8
    })
  ));
  // B stride too small
  assert!(matches!(
    Gbrpf32Frame::try_new(&p, &p, &p, 8, 4, 8, 7, 8),
    Err(GbrFloatFrameError::StrideBelowWidth {
      plane: "b",
      stride: 7,
      width: 8
    })
  ));
  // R stride too small
  assert!(matches!(
    Gbrpf32Frame::try_new(&p, &p, &p, 8, 4, 8, 8, 7),
    Err(GbrFloatFrameError::StrideBelowWidth {
      plane: "r",
      stride: 7,
      width: 8
    })
  ));
}

#[test]
fn gbrpf32_frame_try_new_rejects_plane_too_short() {
  // need stride*(h-1)+w = 8*3+8 = 32 elements; supply 16
  let short = std::vec![0.0f32; 16];
  let full = std::vec![0.0f32; 8 * 4];
  assert!(matches!(
    Gbrpf32Frame::try_new(&short, &full, &full, 8, 4, 8, 8, 8),
    Err(GbrFloatFrameError::PlaneTooShort {
      plane: "g",
      expected: 32,
      actual: 16
    })
  ));
  assert!(matches!(
    Gbrpf32Frame::try_new(&full, &short, &full, 8, 4, 8, 8, 8),
    Err(GbrFloatFrameError::PlaneTooShort {
      plane: "b",
      expected: 32,
      actual: 16
    })
  ));
  assert!(matches!(
    Gbrpf32Frame::try_new(&full, &full, &short, 8, 4, 8, 8, 8),
    Err(GbrFloatFrameError::PlaneTooShort {
      plane: "r",
      expected: 32,
      actual: 16
    })
  ));
}

#[test]
fn gbrpf32_frame_try_new_rejects_dimension_overflow() {
  // width * height > i32::MAX: use two values whose product is 2^31.
  let w: u32 = 1 << 16;
  let h: u32 = 1 << 15; // 2^16 * 2^15 = 2^31 > i32::MAX (= 2^31 - 1)
  let p: &[f32] = &[];
  assert!(matches!(
    Gbrpf32Frame::try_new(p, p, p, w, h, w as usize, w as usize, w as usize),
    Err(GbrFloatFrameError::DimensionOverflow { .. })
  ));
}

// ---- Gbrapf32Frame ---------------------------------------------------------
// Four f32 planes (adds alpha).

#[test]
fn gbrapf32_frame_try_new_accepts_valid_tight() {
  let p = std::vec![0.0f32; 8 * 4];
  let f = Gbrapf32Frame::try_new(&p, &p, &p, &p, 8, 4, 8, 8, 8, 8).expect("valid");
  assert_eq!(f.width(), 8);
  assert_eq!(f.height(), 4);
  assert_eq!(f.a_stride(), 8);
}

#[test]
fn gbrapf32_frame_try_new_rejects_zero_dimension() {
  let p = std::vec![0.0f32; 16];
  assert!(matches!(
    Gbrapf32Frame::try_new(&p, &p, &p, &p, 0, 4, 8, 8, 8, 8),
    Err(GbrFloatFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn gbrapf32_frame_try_new_rejects_stride_below_width() {
  let p = std::vec![0.0f32; 8 * 4];
  // A stride too small
  assert!(matches!(
    Gbrapf32Frame::try_new(&p, &p, &p, &p, 8, 4, 8, 8, 8, 7),
    Err(GbrFloatFrameError::StrideBelowWidth {
      plane: "a",
      stride: 7,
      width: 8
    })
  ));
}

#[test]
fn gbrapf32_frame_try_new_rejects_plane_too_short() {
  let short = std::vec![0.0f32; 16];
  let full = std::vec![0.0f32; 8 * 4];
  assert!(matches!(
    Gbrapf32Frame::try_new(&full, &full, &full, &short, 8, 4, 8, 8, 8, 8),
    Err(GbrFloatFrameError::PlaneTooShort {
      plane: "a",
      expected: 32,
      actual: 16
    })
  ));
}

#[test]
fn gbrapf32_frame_try_new_rejects_dimension_overflow() {
  let w: u32 = 1 << 16;
  let h: u32 = 1 << 15;
  let p: &[f32] = &[];
  assert!(matches!(
    Gbrapf32Frame::try_new(
      p, p, p, p, w, h, w as usize, w as usize, w as usize, w as usize
    ),
    Err(GbrFloatFrameError::DimensionOverflow { .. })
  ));
}

// ---- Gbrpf16Frame ----------------------------------------------------------
// Three half::f16 planes, no alpha.

fn f16_zeros(n: usize) -> std::vec::Vec<half::f16> {
  std::vec![half::f16::ZERO; n]
}

#[test]
fn gbrpf16_frame_try_new_accepts_valid_tight() {
  let p = f16_zeros(8 * 4);
  let f = Gbrpf16Frame::try_new(&p, &p, &p, 8, 4, 8, 8, 8).expect("valid");
  assert_eq!(f.width(), 8);
  assert_eq!(f.height(), 4);
  assert_eq!(f.g_stride(), 8);
}

#[test]
fn gbrpf16_frame_try_new_rejects_zero_dimension() {
  let p = f16_zeros(16);
  assert!(matches!(
    Gbrpf16Frame::try_new(&p, &p, &p, 8, 0, 8, 8, 8),
    Err(GbrFloatFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn gbrpf16_frame_try_new_rejects_stride_below_width() {
  let p = f16_zeros(8 * 4);
  assert!(matches!(
    Gbrpf16Frame::try_new(&p, &p, &p, 8, 4, 7, 8, 8),
    Err(GbrFloatFrameError::StrideBelowWidth {
      plane: "g",
      stride: 7,
      width: 8
    })
  ));
  assert!(matches!(
    Gbrpf16Frame::try_new(&p, &p, &p, 8, 4, 8, 7, 8),
    Err(GbrFloatFrameError::StrideBelowWidth {
      plane: "b",
      stride: 7,
      width: 8
    })
  ));
  assert!(matches!(
    Gbrpf16Frame::try_new(&p, &p, &p, 8, 4, 8, 8, 7),
    Err(GbrFloatFrameError::StrideBelowWidth {
      plane: "r",
      stride: 7,
      width: 8
    })
  ));
}

#[test]
fn gbrpf16_frame_try_new_rejects_plane_too_short() {
  let short = f16_zeros(16);
  let full = f16_zeros(8 * 4);
  assert!(matches!(
    Gbrpf16Frame::try_new(&short, &full, &full, 8, 4, 8, 8, 8),
    Err(GbrFloatFrameError::PlaneTooShort {
      plane: "g",
      expected: 32,
      actual: 16
    })
  ));
  assert!(matches!(
    Gbrpf16Frame::try_new(&full, &short, &full, 8, 4, 8, 8, 8),
    Err(GbrFloatFrameError::PlaneTooShort {
      plane: "b",
      expected: 32,
      actual: 16
    })
  ));
  assert!(matches!(
    Gbrpf16Frame::try_new(&full, &full, &short, 8, 4, 8, 8, 8),
    Err(GbrFloatFrameError::PlaneTooShort {
      plane: "r",
      expected: 32,
      actual: 16
    })
  ));
}

#[test]
fn gbrpf16_frame_try_new_rejects_dimension_overflow() {
  let w: u32 = 1 << 16;
  let h: u32 = 1 << 15;
  let p: &[half::f16] = &[];
  assert!(matches!(
    Gbrpf16Frame::try_new(p, p, p, w, h, w as usize, w as usize, w as usize),
    Err(GbrFloatFrameError::DimensionOverflow { .. })
  ));
}

// ---- Gbrapf16Frame ---------------------------------------------------------
// Four half::f16 planes, with alpha.

#[test]
fn gbrapf16_frame_try_new_accepts_valid_tight() {
  let p = f16_zeros(8 * 4);
  let f = Gbrapf16Frame::try_new(&p, &p, &p, &p, 8, 4, 8, 8, 8, 8).expect("valid");
  assert_eq!(f.width(), 8);
  assert_eq!(f.height(), 4);
  assert_eq!(f.a_stride(), 8);
}

#[test]
fn gbrapf16_frame_try_new_rejects_zero_dimension() {
  let p = f16_zeros(16);
  assert!(matches!(
    Gbrapf16Frame::try_new(&p, &p, &p, &p, 0, 4, 8, 8, 8, 8),
    Err(GbrFloatFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn gbrapf16_frame_try_new_rejects_stride_below_width() {
  let p = f16_zeros(8 * 4);
  assert!(matches!(
    Gbrapf16Frame::try_new(&p, &p, &p, &p, 8, 4, 8, 8, 8, 7),
    Err(GbrFloatFrameError::StrideBelowWidth {
      plane: "a",
      stride: 7,
      width: 8
    })
  ));
}

#[test]
fn gbrapf16_frame_try_new_rejects_plane_too_short() {
  let short = f16_zeros(16);
  let full = f16_zeros(8 * 4);
  assert!(matches!(
    Gbrapf16Frame::try_new(&full, &full, &full, &short, 8, 4, 8, 8, 8, 8),
    Err(GbrFloatFrameError::PlaneTooShort {
      plane: "a",
      expected: 32,
      actual: 16
    })
  ));
}

#[test]
fn gbrapf16_frame_try_new_rejects_dimension_overflow() {
  let w: u32 = 1 << 16;
  let h: u32 = 1 << 15;
  let p: &[half::f16] = &[];
  assert!(matches!(
    Gbrapf16Frame::try_new(
      p, p, p, p, w, h, w as usize, w as usize, w as usize, w as usize
    ),
    Err(GbrFloatFrameError::DimensionOverflow { .. })
  ));
}
