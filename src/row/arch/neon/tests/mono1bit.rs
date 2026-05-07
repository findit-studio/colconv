//! NEON mono1bit parity tests vs scalar reference.

use crate::row::{arch::neon::mono1bit as neon_mono1bit, scalar::mono1bit as scalar};

const WIDTHS: &[usize] = &[1, 7, 8, 15, 16, 17, 32, 33, 64, 65, 128, 130];

/// Generate a deterministic byte sequence (PRNG) for the given number of input bytes.
fn make_data(n_bytes: usize, seed: u32) -> std::vec::Vec<u8> {
  let mut s = seed;
  (0..n_bytes)
    .map(|_| {
      s = s.wrapping_mul(1664525).wrapping_add(1013904223);
      (s >> 16) as u8
    })
    .collect()
}

// ---- Monoblack --------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monoblack_to_rgb_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0xABCD_1234);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { neon_mono1bit::monoblack_to_rgb_row(&data, &mut simd, w) };
    scalar::monoblack_to_rgb_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monoblack→rgb width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monoblack_to_rgba_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0x5678_ABCD);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { neon_mono1bit::monoblack_to_rgba_row(&data, &mut simd, w) };
    scalar::monoblack_to_rgba_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monoblack→rgba width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monoblack_to_luma_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0xCAFE_BABE);
    let mut simd = std::vec![0u8; w];
    let mut scal = std::vec![0u8; w];
    unsafe { neon_mono1bit::monoblack_to_luma_row(&data, &mut simd, w) };
    scalar::monoblack_to_luma_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monoblack→luma width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monoblack_to_luma_u16_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0xDEAD_BEEF);
    let mut simd = std::vec![0u16; w];
    let mut scal = std::vec![0u16; w];
    unsafe { neon_mono1bit::monoblack_to_luma_u16_row(&data, &mut simd, w) };
    scalar::monoblack_to_luma_u16_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monoblack→luma_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monoblack_to_rgb_u16_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0x1234_5678);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe { neon_mono1bit::monoblack_to_rgb_u16_row(&data, &mut simd, w) };
    scalar::monoblack_to_rgb_u16_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monoblack→rgb_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monoblack_to_rgba_u16_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0xFEDC_BA98);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { neon_mono1bit::monoblack_to_rgba_u16_row(&data, &mut simd, w) };
    scalar::monoblack_to_rgba_u16_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monoblack→rgba_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monoblack_to_hsv_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0xA1B2_C3D4);
    let mut sh = std::vec![0u8; w];
    let mut ss = std::vec![0u8; w];
    let mut sv = std::vec![0u8; w];
    let mut rh = std::vec![0u8; w];
    let mut rs = std::vec![0u8; w];
    let mut rv = std::vec![0u8; w];
    unsafe { neon_mono1bit::monoblack_to_hsv_row(&data, &mut sh, &mut ss, &mut sv, w) };
    scalar::monoblack_to_hsv_row(&data, &mut rh, &mut rs, &mut rv, w);
    assert_eq!(sh, rh, "monoblack→hsv H width={w}");
    assert_eq!(ss, rs, "monoblack→hsv S width={w}");
    assert_eq!(sv, rv, "monoblack→hsv V width={w}");
  }
}

// ---- Monowhite --------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monowhite_to_rgb_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0x1111_2222);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { neon_mono1bit::monowhite_to_rgb_row(&data, &mut simd, w) };
    scalar::monowhite_to_rgb_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monowhite→rgb width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monowhite_to_rgba_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0x3333_4444);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { neon_mono1bit::monowhite_to_rgba_row(&data, &mut simd, w) };
    scalar::monowhite_to_rgba_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monowhite→rgba width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monowhite_to_luma_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0x5555_6666);
    let mut simd = std::vec![0u8; w];
    let mut scal = std::vec![0u8; w];
    unsafe { neon_mono1bit::monowhite_to_luma_row(&data, &mut simd, w) };
    scalar::monowhite_to_luma_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monowhite→luma width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monowhite_to_luma_u16_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0x7777_8888);
    let mut simd = std::vec![0u16; w];
    let mut scal = std::vec![0u16; w];
    unsafe { neon_mono1bit::monowhite_to_luma_u16_row(&data, &mut simd, w) };
    scalar::monowhite_to_luma_u16_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monowhite→luma_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monowhite_to_rgb_u16_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0x9999_AAAA);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe { neon_mono1bit::monowhite_to_rgb_u16_row(&data, &mut simd, w) };
    scalar::monowhite_to_rgb_u16_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monowhite→rgb_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monowhite_to_rgba_u16_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0xBBBB_CCCC);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { neon_mono1bit::monowhite_to_rgba_u16_row(&data, &mut simd, w) };
    scalar::monowhite_to_rgba_u16_row(&data, &mut scal, w);
    assert_eq!(simd, scal, "monowhite→rgba_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_monowhite_to_hsv_matches_scalar() {
  for &w in WIDTHS {
    let data = make_data(w.div_ceil(8), 0xDDDD_EEEE);
    let mut sh = std::vec![0u8; w];
    let mut ss = std::vec![0u8; w];
    let mut sv = std::vec![0u8; w];
    let mut rh = std::vec![0u8; w];
    let mut rs = std::vec![0u8; w];
    let mut rv = std::vec![0u8; w];
    unsafe { neon_mono1bit::monowhite_to_hsv_row(&data, &mut sh, &mut ss, &mut sv, w) };
    scalar::monowhite_to_hsv_row(&data, &mut rh, &mut rs, &mut rv, w);
    assert_eq!(sh, rh, "monowhite→hsv H width={w}");
    assert_eq!(ss, rs, "monowhite→hsv S width={w}");
    assert_eq!(sv, rv, "monowhite→hsv V width={w}");
  }
}
