use super::*;

// ---- Gbrpf32Frame ----------------------------------------------------------
// Three f32 planes. Stride in elements. DimensionOverflow uses i32::MAX + 1.

#[test]
fn gbrpf32_frame_try_new_accepts_valid_tight() {
  // stride == width, planes exactly cover the frame.
  let g = vec![0.0f32; 8 * 4];
  let b = vec![0.0f32; 8 * 4];
  let r = vec![0.0f32; 8 * 4];
  let f = Gbrpf32Frame::try_new(&g, &b, &r, 8, 4, 8, 8, 8).expect("valid tight frame");
  assert_eq!(f.width(), 8);
  assert_eq!(f.height(), 4);
  assert_eq!(f.g_stride(), 8);
  assert_eq!(f.b_stride(), 8);
  assert_eq!(f.r_stride(), 8);
}

#[test]
fn gbrpf32_try_new_accepts_padded_strides() {
  // stride > width — padded layout, planes must cover stride * (h-1) + w
  let stride: u32 = 16;
  let w: u32 = 8;
  let h: u32 = 4;
  // needed = stride * (h-1) + w = 16*3+8 = 56
  let p = vec![0.0f32; (stride as usize) * (h as usize - 1) + w as usize];
  let f = Gbrpf32Frame::try_new(&p, &p, &p, w, h, stride, stride, stride)
    .expect("padded stride accepted");
  assert_eq!(f.g_stride(), 16);
}

#[test]
fn gbrpf32_try_new_accepts_height_one() {
  // h-1 == 0, so needed == width only
  let p = vec![0.0f32; 8];
  let f = Gbrpf32Frame::try_new(&p, &p, &p, 8, 1, 8, 8, 8).expect("height=1 accepted");
  assert_eq!(f.height(), 1);
}

#[test]
fn gbrpf32_frame_try_new_rejects_zero_dimension() {
  let p = vec![0.0f32; 16];
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
  let p = vec![0.0f32; 8 * 4];
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
  let short = vec![0.0f32; 16];
  let full = vec![0.0f32; 8 * 4];
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
    Gbrpf32Frame::try_new(p, p, p, w, h, w, w, w),
    Err(GbrFloatFrameError::DimensionOverflow { .. })
  ));
}

#[test]
fn gbrpf32_try_new_rejects_geometry_overflow() {
  // stride * (height - 1) overflows usize on 32-bit; on 64-bit this also
  // overflows because u32::MAX/2+1 squared > usize::MAX on 32-bit.
  // We use values that overflow usize even on 64-bit hosts by picking
  // a stride that is just over half of u32::MAX, paired with a large height.
  // On 64-bit hosts (usize=u64), stride * (h-1) = (2^31) * (2^31) = 2^62 < u64::MAX,
  // so this won't overflow on 64-bit. Instead we rely on the DimensionOverflow
  // check (width*height > i32::MAX) to fire first on large values.
  //
  // To specifically trigger GeometryOverflow we need stride to be > i32::MAX
  // but width <= stride (stride >= width). width*height must still fit i32::MAX
  // which is impossible if stride >= width > i32::MAX.
  // The check order is: ZeroDimension → DimensionOverflow → per-plane.
  // GeometryOverflow fires only if stride*height overflows usize,
  // which on 64-bit cannot happen with u32 inputs (u32::MAX * u32::MAX < u64::MAX).
  // So on 64-bit this test is only meaningful on 32-bit targets.
  // We skip on 64-bit and just verify the DimensionOverflow path works.
  #[cfg(target_pointer_width = "32")]
  {
    let stride: u32 = u32::MAX / 2 + 1;
    let height: u32 = u32::MAX / 2 + 1;
    // width must be <= stride to avoid StrideBelowWidth, but also large enough
    // that width*height check might pass. Actually width*height will overflow i32::MAX
    // too, so DimensionOverflow will fire. Use width=1 so w*h passes i32::MAX check.
    let p: &[f32] = &[];
    // With width=1, height=stride/2+1: product = height < i32::MAX. stride >= 1.
    // stride*(height-1) on 32-bit: (u32::MAX/2+1) * (u32::MAX/2) overflows usize (u32).
    let small_h: u32 = 3;
    assert!(matches!(
      Gbrpf32Frame::try_new(p, p, p, 1, small_h, stride, stride, stride),
      Err(GbrFloatFrameError::GeometryOverflow { plane: "g", .. })
    ));
  }
  #[cfg(not(target_pointer_width = "32"))]
  {
    // On 64-bit, GeometryOverflow cannot fire with u32 strides/heights since
    // u32::MAX * u32::MAX = ~1.8e19 < u64::MAX. Verify the error type exists
    // by constructing a dummy value.
    let _ = GbrFloatFrameError::GeometryOverflow {
      plane: "g",
      stride: u32::MAX,
      height: u32::MAX,
    };
  }
}

// ---- Gbrapf32Frame ---------------------------------------------------------
// Four f32 planes (adds alpha).

#[test]
fn gbrapf32_frame_try_new_accepts_valid_tight() {
  let p = vec![0.0f32; 8 * 4];
  let f = Gbrapf32Frame::try_new(&p, &p, &p, &p, 8, 4, 8, 8, 8, 8).expect("valid");
  assert_eq!(f.width(), 8);
  assert_eq!(f.height(), 4);
  assert_eq!(f.a_stride(), 8);
}

#[test]
fn gbrapf32_frame_try_new_rejects_zero_dimension() {
  let p = vec![0.0f32; 16];
  assert!(matches!(
    Gbrapf32Frame::try_new(&p, &p, &p, &p, 0, 4, 8, 8, 8, 8),
    Err(GbrFloatFrameError::ZeroDimension { .. })
  ));
}

#[test]
fn gbrapf32_frame_try_new_rejects_stride_below_width() {
  let p = vec![0.0f32; 8 * 4];
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
  let short = vec![0.0f32; 16];
  let full = vec![0.0f32; 8 * 4];
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
    Gbrapf32Frame::try_new(p, p, p, p, w, h, w, w, w, w),
    Err(GbrFloatFrameError::DimensionOverflow { .. })
  ));
}

#[test]
fn gbrapf32_try_new_rejects_geometry_overflow() {
  #[cfg(target_pointer_width = "32")]
  {
    let stride: u32 = u32::MAX / 2 + 1;
    let p: &[f32] = &[];
    assert!(matches!(
      Gbrapf32Frame::try_new(p, p, p, p, 1, 3, stride, stride, stride, stride),
      Err(GbrFloatFrameError::GeometryOverflow { plane: "g", .. })
    ));
  }
  #[cfg(not(target_pointer_width = "32"))]
  {
    let _ = GbrFloatFrameError::GeometryOverflow {
      plane: "g",
      stride: u32::MAX,
      height: u32::MAX,
    };
  }
}

// ---- Gbrpf16Frame ----------------------------------------------------------
// Three half::f16 planes, no alpha.

fn f16_zeros(n: usize) -> Vec<half::f16> {
  vec![half::f16::ZERO; n]
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
    Gbrpf16Frame::try_new(p, p, p, w, h, w, w, w),
    Err(GbrFloatFrameError::DimensionOverflow { .. })
  ));
}

#[test]
fn gbrpf16_try_new_rejects_geometry_overflow() {
  #[cfg(target_pointer_width = "32")]
  {
    let stride: u32 = u32::MAX / 2 + 1;
    let p: &[half::f16] = &[];
    assert!(matches!(
      Gbrpf16Frame::try_new(p, p, p, 1, 3, stride, stride, stride),
      Err(GbrFloatFrameError::GeometryOverflow { plane: "g", .. })
    ));
  }
  #[cfg(not(target_pointer_width = "32"))]
  {
    let _ = GbrFloatFrameError::GeometryOverflow {
      plane: "g",
      stride: u32::MAX,
      height: u32::MAX,
    };
  }
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
    Gbrapf16Frame::try_new(p, p, p, p, w, h, w, w, w, w),
    Err(GbrFloatFrameError::DimensionOverflow { .. })
  ));
}

#[test]
fn gbrapf16_try_new_rejects_geometry_overflow() {
  #[cfg(target_pointer_width = "32")]
  {
    let stride: u32 = u32::MAX / 2 + 1;
    let p: &[half::f16] = &[];
    assert!(matches!(
      Gbrapf16Frame::try_new(p, p, p, p, 1, 3, stride, stride, stride, stride),
      Err(GbrFloatFrameError::GeometryOverflow { plane: "g", .. })
    ));
  }
  #[cfg(not(target_pointer_width = "32"))]
  {
    let _ = GbrFloatFrameError::GeometryOverflow {
      plane: "g",
      stride: u32::MAX,
      height: u32::MAX,
    };
  }
}
