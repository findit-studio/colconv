use super::*;

// ---- GbrpFrame ---------------------------------------------------------
//
// Three full-width 8-bit planes in G, B, R order.  Each plane requires
// `*_stride >= width` and length `>= *_stride * height`.  No width /
// height parity constraint.

#[test]
fn gbrp_frame_try_new_accepts_valid_tight() {
  // Minimum geometry: stride == width for every plane.
  let g = std::vec![0u8; 16 * 4];
  let b = std::vec![0u8; 16 * 4];
  let r = std::vec![0u8; 16 * 4];
  GbrpFrame::try_new(&g, &b, &r, 16, 4, 16, 16, 16).expect("valid tight");
}

#[test]
fn gbrp_frame_try_new_accepts_oversized_stride() {
  // Row padding is allowed: stride > width on every plane.
  let g = std::vec![0u8; 32 * 4];
  let b = std::vec![0u8; 32 * 4];
  let r = std::vec![0u8; 32 * 4];
  GbrpFrame::try_new(&g, &b, &r, 16, 4, 32, 32, 32).expect("oversized stride is valid");
}

#[test]
fn gbrp_frame_try_new_rejects_zero_width() {
  let g = std::vec![0u8; 16];
  let b = std::vec![0u8; 16];
  let r = std::vec![0u8; 16];
  assert!(matches!(
    GbrpFrame::try_new(&g, &b, &r, 0, 4, 0, 0, 0),
    Err(GbrpFrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
}

#[test]
fn gbrp_frame_try_new_rejects_zero_height() {
  let g = std::vec![0u8; 16];
  let b = std::vec![0u8; 16];
  let r = std::vec![0u8; 16];
  assert!(matches!(
    GbrpFrame::try_new(&g, &b, &r, 16, 0, 16, 16, 16),
    Err(GbrpFrameError::ZeroDimension {
      width: 16,
      height: 0
    })
  ));
}

#[test]
fn gbrp_frame_try_new_rejects_g_stride_too_small() {
  let g = std::vec![0u8; 16 * 4];
  let b = std::vec![0u8; 16 * 4];
  let r = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrpFrame::try_new(&g, &b, &r, 16, 4, 15, 16, 16),
    Err(GbrpFrameError::GStrideTooSmall {
      width: 16,
      g_stride: 15
    })
  ));
}

#[test]
fn gbrp_frame_try_new_rejects_b_stride_too_small() {
  let g = std::vec![0u8; 16 * 4];
  let b = std::vec![0u8; 16 * 4];
  let r = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrpFrame::try_new(&g, &b, &r, 16, 4, 16, 15, 16),
    Err(GbrpFrameError::BStrideTooSmall {
      width: 16,
      b_stride: 15
    })
  ));
}

#[test]
fn gbrp_frame_try_new_rejects_r_stride_too_small() {
  let g = std::vec![0u8; 16 * 4];
  let b = std::vec![0u8; 16 * 4];
  let r = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrpFrame::try_new(&g, &b, &r, 16, 4, 16, 16, 15),
    Err(GbrpFrameError::RStrideTooSmall {
      width: 16,
      r_stride: 15
    })
  ));
}

#[test]
fn gbrp_frame_try_new_rejects_g_plane_too_short() {
  // G plane only large enough for 1 row, but height = 4.
  let g = std::vec![0u8; 16];
  let b = std::vec![0u8; 16 * 4];
  let r = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrpFrame::try_new(&g, &b, &r, 16, 4, 16, 16, 16),
    Err(GbrpFrameError::GPlaneTooShort {
      expected: 64,
      actual: 16
    })
  ));
}

#[test]
fn gbrp_frame_try_new_rejects_b_plane_too_short() {
  let g = std::vec![0u8; 16 * 4];
  let b = std::vec![0u8; 16];
  let r = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrpFrame::try_new(&g, &b, &r, 16, 4, 16, 16, 16),
    Err(GbrpFrameError::BPlaneTooShort {
      expected: 64,
      actual: 16
    })
  ));
}

#[test]
fn gbrp_frame_try_new_rejects_r_plane_too_short() {
  let g = std::vec![0u8; 16 * 4];
  let b = std::vec![0u8; 16 * 4];
  let r = std::vec![0u8; 16];
  assert!(matches!(
    GbrpFrame::try_new(&g, &b, &r, 16, 4, 16, 16, 16),
    Err(GbrpFrameError::RPlaneTooShort {
      expected: 64,
      actual: 16
    })
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn gbrp_frame_try_new_rejects_geometry_overflow() {
  let big: u32 = 0x1_0000;
  let g: [u8; 0] = [];
  let b: [u8; 0] = [];
  let r: [u8; 0] = [];
  let e = GbrpFrame::try_new(&g, &b, &r, big, big, big, big, big).unwrap_err();
  assert!(matches!(e, GbrpFrameError::GeometryOverflow { .. }));
}

#[test]
#[should_panic(expected = "invalid GbrpFrame")]
fn gbrp_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = GbrpFrame::new(&buf, &buf, &buf, 16, 4, 16, 16, 16);
}

#[test]
fn gbrp_frame_accessors_round_trip() {
  let g = std::vec![1u8; 8 * 2];
  let b = std::vec![2u8; 8 * 2];
  let r = std::vec![3u8; 8 * 2];
  let frame = GbrpFrame::try_new(&g, &b, &r, 8, 2, 8, 8, 8).unwrap();
  assert_eq!(frame.width(), 8);
  assert_eq!(frame.height(), 2);
  assert_eq!(frame.g_stride(), 8);
  assert_eq!(frame.b_stride(), 8);
  assert_eq!(frame.r_stride(), 8);
  assert_eq!(frame.g(), g.as_slice());
  assert_eq!(frame.b(), b.as_slice());
  assert_eq!(frame.r(), r.as_slice());
}

// ---- GbrapFrame --------------------------------------------------------
//
// Four full-width 8-bit planes in G, B, R, A order.  Each plane
// requires `*_stride >= width` and length `>= *_stride * height`.
// No width / height parity constraint.

#[test]
fn gbrap_frame_try_new_accepts_valid_tight() {
  let g = std::vec![0u8; 16 * 4];
  let b = std::vec![0u8; 16 * 4];
  let r = std::vec![0u8; 16 * 4];
  let a = std::vec![0u8; 16 * 4];
  GbrapFrame::try_new(&g, &b, &r, &a, 16, 4, 16, 16, 16, 16).expect("valid tight");
}

#[test]
fn gbrap_frame_try_new_accepts_oversized_stride() {
  let g = std::vec![0u8; 32 * 4];
  let b = std::vec![0u8; 32 * 4];
  let r = std::vec![0u8; 32 * 4];
  let a = std::vec![0u8; 32 * 4];
  GbrapFrame::try_new(&g, &b, &r, &a, 16, 4, 32, 32, 32, 32).expect("oversized stride is valid");
}

#[test]
fn gbrap_frame_try_new_rejects_zero_width() {
  let empty = std::vec![0u8; 4];
  assert!(matches!(
    GbrapFrame::try_new(&empty, &empty, &empty, &empty, 0, 4, 0, 0, 0, 0),
    Err(GbrapFrameError::ZeroDimension {
      width: 0,
      height: 4
    })
  ));
}

#[test]
fn gbrap_frame_try_new_rejects_zero_height() {
  let empty = std::vec![0u8; 16];
  assert!(matches!(
    GbrapFrame::try_new(&empty, &empty, &empty, &empty, 16, 0, 16, 16, 16, 16),
    Err(GbrapFrameError::ZeroDimension {
      width: 16,
      height: 0
    })
  ));
}

#[test]
fn gbrap_frame_try_new_rejects_g_stride_too_small() {
  let p = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrapFrame::try_new(&p, &p, &p, &p, 16, 4, 15, 16, 16, 16),
    Err(GbrapFrameError::GStrideTooSmall {
      width: 16,
      g_stride: 15
    })
  ));
}

#[test]
fn gbrap_frame_try_new_rejects_b_stride_too_small() {
  let p = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrapFrame::try_new(&p, &p, &p, &p, 16, 4, 16, 15, 16, 16),
    Err(GbrapFrameError::BStrideTooSmall {
      width: 16,
      b_stride: 15
    })
  ));
}

#[test]
fn gbrap_frame_try_new_rejects_r_stride_too_small() {
  let p = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrapFrame::try_new(&p, &p, &p, &p, 16, 4, 16, 16, 15, 16),
    Err(GbrapFrameError::RStrideTooSmall {
      width: 16,
      r_stride: 15
    })
  ));
}

#[test]
fn gbrap_frame_try_new_rejects_a_stride_too_small() {
  let p = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrapFrame::try_new(&p, &p, &p, &p, 16, 4, 16, 16, 16, 15),
    Err(GbrapFrameError::AStrideTooSmall {
      width: 16,
      a_stride: 15
    })
  ));
}

#[test]
fn gbrap_frame_try_new_rejects_g_plane_too_short() {
  let short = std::vec![0u8; 16];
  let full = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrapFrame::try_new(&short, &full, &full, &full, 16, 4, 16, 16, 16, 16),
    Err(GbrapFrameError::GPlaneTooShort {
      expected: 64,
      actual: 16
    })
  ));
}

#[test]
fn gbrap_frame_try_new_rejects_b_plane_too_short() {
  let short = std::vec![0u8; 16];
  let full = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrapFrame::try_new(&full, &short, &full, &full, 16, 4, 16, 16, 16, 16),
    Err(GbrapFrameError::BPlaneTooShort {
      expected: 64,
      actual: 16
    })
  ));
}

#[test]
fn gbrap_frame_try_new_rejects_r_plane_too_short() {
  let short = std::vec![0u8; 16];
  let full = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrapFrame::try_new(&full, &full, &short, &full, 16, 4, 16, 16, 16, 16),
    Err(GbrapFrameError::RPlaneTooShort {
      expected: 64,
      actual: 16
    })
  ));
}

#[test]
fn gbrap_frame_try_new_rejects_a_plane_too_short() {
  let short = std::vec![0u8; 16];
  let full = std::vec![0u8; 16 * 4];
  assert!(matches!(
    GbrapFrame::try_new(&full, &full, &full, &short, 16, 4, 16, 16, 16, 16),
    Err(GbrapFrameError::APlaneTooShort {
      expected: 64,
      actual: 16
    })
  ));
}

#[cfg(target_pointer_width = "32")]
#[test]
fn gbrap_frame_try_new_rejects_geometry_overflow() {
  let big: u32 = 0x1_0000;
  let p: [u8; 0] = [];
  let e = GbrapFrame::try_new(&p, &p, &p, &p, big, big, big, big, big, big).unwrap_err();
  assert!(matches!(e, GbrapFrameError::GeometryOverflow { .. }));
}

#[test]
#[should_panic(expected = "invalid GbrapFrame")]
fn gbrap_frame_new_panics_on_invalid() {
  let buf = std::vec![0u8; 10];
  let _ = GbrapFrame::new(&buf, &buf, &buf, &buf, 16, 4, 16, 16, 16, 16);
}

#[test]
fn gbrap_frame_accessors_round_trip() {
  let g = std::vec![1u8; 8 * 2];
  let b = std::vec![2u8; 8 * 2];
  let r = std::vec![3u8; 8 * 2];
  let a = std::vec![255u8; 8 * 2];
  let frame = GbrapFrame::try_new(&g, &b, &r, &a, 8, 2, 8, 8, 8, 8).unwrap();
  assert_eq!(frame.width(), 8);
  assert_eq!(frame.height(), 2);
  assert_eq!(frame.g_stride(), 8);
  assert_eq!(frame.b_stride(), 8);
  assert_eq!(frame.r_stride(), 8);
  assert_eq!(frame.a_stride(), 8);
  assert_eq!(frame.g(), g.as_slice());
  assert_eq!(frame.b(), b.as_slice());
  assert_eq!(frame.r(), r.as_slice());
  assert_eq!(frame.a(), a.as_slice());
}
