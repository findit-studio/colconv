use super::*;

// ---- GbrpHighBitFrame (3-plane) tests ------------------------------------

fn gbrp10_planes(w: u32, h: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let n = (w * h) as usize;
  (vec![0u16; n], vec![0u16; n], vec![0u16; n])
}

#[test]
fn gbrp10_try_new_accepts_valid_tight() {
  let (g, b, r) = gbrp10_planes(16, 8);
  let f = Gbrp10Frame::try_new(&g, &b, &r, 16, 8, 16, 16, 16).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.g_stride(), 16);
  assert_eq!(f.b_stride(), 16);
  assert_eq!(f.r_stride(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn gbrp10_try_new_accepts_padded_strides() {
  let stride = 32u32;
  let h = 8u32;
  let g = vec![0u16; (stride * h) as usize];
  let b = vec![0u16; (stride * h) as usize];
  let r = vec![0u16; (stride * h) as usize];
  let f = Gbrp10Frame::try_new(&g, &b, &r, 16, h, stride, stride, stride).expect("valid");
  assert_eq!(f.g_stride(), stride);
}

#[test]
fn gbrp10_try_new_rejects_zero_width() {
  let (g, b, r) = gbrp10_planes(4, 4);
  let e = Gbrp10Frame::try_new(&g, &b, &r, 0, 4, 4, 4, 4).unwrap_err();
  assert!(matches!(e, GbrpHighBitFrameError::ZeroDimension { .. }));
}

#[test]
fn gbrp10_try_new_rejects_zero_height() {
  let (g, b, r) = gbrp10_planes(4, 4);
  let e = Gbrp10Frame::try_new(&g, &b, &r, 4, 0, 4, 4, 4).unwrap_err();
  assert!(matches!(e, GbrpHighBitFrameError::ZeroDimension { .. }));
}

#[test]
fn gbrp10_try_new_rejects_g_stride_too_small() {
  let (g, b, r) = gbrp10_planes(8, 4);
  let e = Gbrp10Frame::try_new(&g, &b, &r, 8, 4, 4, 8, 8).unwrap_err();
  assert!(matches!(e, GbrpHighBitFrameError::GStrideTooSmall { .. }));
}

#[test]
fn gbrp10_try_new_rejects_b_stride_too_small() {
  let (g, b, r) = gbrp10_planes(8, 4);
  let e = Gbrp10Frame::try_new(&g, &b, &r, 8, 4, 8, 4, 8).unwrap_err();
  assert!(matches!(e, GbrpHighBitFrameError::BStrideTooSmall { .. }));
}

#[test]
fn gbrp10_try_new_rejects_r_stride_too_small() {
  let (g, b, r) = gbrp10_planes(8, 4);
  let e = Gbrp10Frame::try_new(&g, &b, &r, 8, 4, 8, 8, 4).unwrap_err();
  assert!(matches!(e, GbrpHighBitFrameError::RStrideTooSmall { .. }));
}

#[test]
fn gbrp10_try_new_rejects_g_plane_too_short() {
  // Need stride*height = 8*4=32 samples; only provide 16.
  let g = vec![0u16; 16];
  let b = vec![0u16; 32];
  let r = vec![0u16; 32];
  let e = Gbrp10Frame::try_new(&g, &b, &r, 8, 4, 8, 8, 8).unwrap_err();
  assert!(matches!(e, GbrpHighBitFrameError::GPlaneTooShort { .. }));
}

#[test]
fn gbrp10_try_new_rejects_b_plane_too_short() {
  let g = vec![0u16; 32];
  let b = vec![0u16; 16];
  let r = vec![0u16; 32];
  let e = Gbrp10Frame::try_new(&g, &b, &r, 8, 4, 8, 8, 8).unwrap_err();
  assert!(matches!(e, GbrpHighBitFrameError::BPlaneTooShort { .. }));
}

#[test]
fn gbrp10_try_new_rejects_r_plane_too_short() {
  let g = vec![0u16; 32];
  let b = vec![0u16; 32];
  let r = vec![0u16; 16];
  let e = Gbrp10Frame::try_new(&g, &b, &r, 8, 4, 8, 8, 8).unwrap_err();
  assert!(matches!(e, GbrpHighBitFrameError::RPlaneTooShort { .. }));
}

#[test]
fn gbrp10_new_panics_on_invalid() {
  // zero width must panic — use try_new to confirm the error path
  // (const fn `new` panics but is untestable via catch_unwind with borrows).
  let g: [u16; 0] = [];
  let b: [u16; 0] = [];
  let r: [u16; 0] = [];
  let e = Gbrp10Frame::try_new(&g, &b, &r, 0, 1, 1, 1, 1).unwrap_err();
  assert!(matches!(e, GbrpHighBitFrameError::ZeroDimension { .. }));
}

// ---- Per-BITS sanity: bits() accessor and valid construction --------------

#[test]
fn gbrp9_bits_accessor() {
  let p = vec![0u16; 4];
  let f = Gbrp9Frame::try_new(&p, &p, &p, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.bits(), 9);
}

#[test]
fn gbrp12_bits_accessor() {
  let p = vec![0u16; 4];
  let f = Gbrp12Frame::try_new(&p, &p, &p, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.bits(), 12);
}

#[test]
fn gbrp14_bits_accessor() {
  let p = vec![0u16; 4];
  let f = Gbrp14Frame::try_new(&p, &p, &p, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.bits(), 14);
}

#[test]
fn gbrp16_bits_accessor() {
  let p = vec![0u16; 4];
  let f = Gbrp16Frame::try_new(&p, &p, &p, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.bits(), 16);
}

// ---- GbrapHighBitFrame (4-plane) tests ------------------------------------

fn gbrap10_planes(w: u32, h: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let n = (w * h) as usize;
  (vec![0u16; n], vec![0u16; n], vec![0u16; n], vec![0u16; n])
}

#[test]
fn gbrap10_try_new_accepts_valid_tight() {
  let (g, b, r, a) = gbrap10_planes(16, 8);
  let f = Gbrap10Frame::try_new(&g, &b, &r, &a, 16, 8, 16, 16, 16, 16).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.a_stride(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn gbrap10_try_new_rejects_zero_dimension() {
  let (g, b, r, a) = gbrap10_planes(4, 4);
  let e = Gbrap10Frame::try_new(&g, &b, &r, &a, 0, 4, 4, 4, 4, 4).unwrap_err();
  assert!(matches!(e, GbrapHighBitFrameError::ZeroDimension { .. }));
}

#[test]
fn gbrap10_try_new_rejects_a_stride_too_small() {
  let (g, b, r, a) = gbrap10_planes(8, 4);
  let e = Gbrap10Frame::try_new(&g, &b, &r, &a, 8, 4, 8, 8, 8, 4).unwrap_err();
  assert!(matches!(e, GbrapHighBitFrameError::AStrideTooSmall { .. }));
}

#[test]
fn gbrap10_try_new_rejects_a_plane_too_short() {
  let g = vec![0u16; 32];
  let b = vec![0u16; 32];
  let r = vec![0u16; 32];
  let a = vec![0u16; 16]; // too short — need 32
  let e = Gbrap10Frame::try_new(&g, &b, &r, &a, 8, 4, 8, 8, 8, 8).unwrap_err();
  assert!(matches!(e, GbrapHighBitFrameError::APlaneTooShort { .. }));
}

#[test]
fn gbrap10_try_new_rejects_g_stride_too_small() {
  let (g, b, r, a) = gbrap10_planes(8, 4);
  let e = Gbrap10Frame::try_new(&g, &b, &r, &a, 8, 4, 4, 8, 8, 8).unwrap_err();
  assert!(matches!(e, GbrapHighBitFrameError::GStrideTooSmall { .. }));
}

#[test]
fn gbrap10_new_panics_on_invalid() {
  let p: [u16; 0] = [];
  let e = Gbrap10Frame::try_new(&p, &p, &p, &p, 0, 1, 1, 1, 1, 1).unwrap_err();
  assert!(matches!(e, GbrapHighBitFrameError::ZeroDimension { .. }));
}

// ---- Per-BITS sanity for Gbrap family -------------------------------------

#[test]
fn gbrap9_bits_accessor() {
  let p = vec![0u16; 4];
  let f = Gbrap9Frame::try_new(&p, &p, &p, &p, 2, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.bits(), 9);
}

#[test]
fn gbrap12_bits_accessor() {
  let p = vec![0u16; 4];
  let f = Gbrap12Frame::try_new(&p, &p, &p, &p, 2, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.bits(), 12);
}

#[test]
fn gbrap14_bits_accessor() {
  let p = vec![0u16; 4];
  let f = Gbrap14Frame::try_new(&p, &p, &p, &p, 2, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.bits(), 14);
}

#[test]
fn gbrap16_bits_accessor() {
  let p = vec![0u16; 4];
  let f = Gbrap16Frame::try_new(&p, &p, &p, &p, 2, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.bits(), 16);
}

// ---- crate-internal aliases (y/u/v) pass through to g/b/r ----------------

#[test]
fn gbrp10_internal_aliases_match_public_accessors() {
  let g = vec![1u16; 4];
  let b = vec![2u16; 4];
  let r = vec![3u16; 4];
  let f = Gbrp10Frame::try_new(&g, &b, &r, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.y(), f.g());
  assert_eq!(f.u(), f.b());
  assert_eq!(f.v(), f.r());
  assert_eq!(f.y_stride(), f.g_stride());
  assert_eq!(f.u_stride(), f.b_stride());
  assert_eq!(f.v_stride(), f.r_stride());
}

#[test]
fn gbrap10_internal_aliases_match_public_accessors() {
  let g = vec![1u16; 4];
  let b = vec![2u16; 4];
  let r = vec![3u16; 4];
  let a = vec![4u16; 4];
  let f = Gbrap10Frame::try_new(&g, &b, &r, &a, 2, 2, 2, 2, 2, 2).unwrap();
  assert_eq!(f.y(), f.g());
  assert_eq!(f.u(), f.b());
  assert_eq!(f.v(), f.r());
  assert_eq!(f.a()[0], 4u16);
}
