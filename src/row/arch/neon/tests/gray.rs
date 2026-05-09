use super::super::*;

use crate::row::scalar::gray as scalar;

const WIDTHS: &[usize] = &[1, 7, 8, 16, 17, 32, 33, 64, 128, 130];

fn prng(out: &mut [u8], seed: u32) {
  let mut s = seed;
  for v in out.iter_mut() {
    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    *v = (s >> 16) as u8;
  }
}
fn prng16(out: &mut [u16], seed: u32) {
  let mut buf = std::vec![0u8; out.len() * 2];
  prng(&mut buf, seed);
  for (i, o) in out.iter_mut().enumerate() {
    *o = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_gray8_to_rgb_matches_scalar() {
  for &w in WIDTHS {
    let mut plane = std::vec![0u8; w];
    prng(&mut plane, 0xABCD);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gray8_to_rgb_row(&plane, &mut simd, w, true) };
    scalar::gray8_to_rgb_row(&plane, &mut scal, w, true);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_gray8_to_rgba_matches_scalar() {
  for &w in WIDTHS {
    let mut plane = std::vec![0u8; w];
    prng(&mut plane, 0x1234);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { gray8_to_rgba_row(&plane, &mut simd, w, true) };
    scalar::gray8_to_rgba_row(&plane, &mut scal, w, true);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_gray8_to_hsv_matches_scalar() {
  for &w in WIDTHS {
    let mut plane = std::vec![0u8; w];
    prng(&mut plane, 0x5678);
    let mut sh = std::vec![0u8; w];
    let mut ss = std::vec![0u8; w];
    let mut sv = std::vec![0u8; w];
    let mut rh = std::vec![0u8; w];
    let mut rs = std::vec![0u8; w];
    let mut rv = std::vec![0u8; w];
    unsafe { gray8_to_hsv_row(&plane, &mut sh, &mut ss, &mut sv, w, true) };
    scalar::gray8_to_hsv_row(&plane, &mut rh, &mut rs, &mut rv, w, true);
    assert_eq!(sh, rh, "H width={w}");
    assert_eq!(ss, rs, "S width={w}");
    assert_eq!(sv, rv, "V width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_gray_n_to_rgb_10bit_matches_scalar() {
  for &w in WIDTHS {
    let mut plane = std::vec![0u16; w];
    prng16(&mut plane, 0xABCD_1234);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gray_n_to_rgb_row::<10, false>(&plane, &mut simd, w, true) };
    scalar::gray_n_to_rgb_row::<10, false>(&plane, &mut scal, w, true);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_gray16_to_rgb_matches_scalar() {
  for &w in WIDTHS {
    let mut plane = std::vec![0u16; w];
    prng16(&mut plane, 0xDEAD_BEEF);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gray16_to_rgb_row::<false>(&plane, &mut simd, w, true) };
    scalar::gray16_to_rgb_row::<false>(&plane, &mut scal, w, true);
    assert_eq!(simd, scal, "width={w}");
  }
}

// ---- limited-range SIMD/scalar parity tests ----

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_gray8_limited_range_matches_scalar() {
  for &w in WIDTHS {
    let mut plane = std::vec![0u8; w];
    prng(&mut plane, 0xCAFE_BABEu32);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gray8_to_rgb_row(&plane, &mut simd, w, false) };
    scalar::gray8_to_rgb_row(&plane, &mut scal, w, false);
    assert_eq!(simd, scal, "width={w} limited-range");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_gray16_limited_range_matches_scalar() {
  for &w in WIDTHS {
    let mut plane = std::vec![0u16; w];
    prng16(&mut plane, 0x1234_5678);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { gray16_to_rgb_row::<false>(&plane, &mut simd, w, false) };
    scalar::gray16_to_rgb_row::<false>(&plane, &mut scal, w, false);
    assert_eq!(simd, scal, "width={w} limited-range");
  }
}

// ---- Grayf32 parity tests ---------------------------------------------------

fn prng_f32(out: &mut [f32], seed: u32) {
  let mut s = seed;
  for v in out.iter_mut() {
    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    // Values in [-0.1, 1.2] to exercise clamping.
    *v = ((s >> 8) as f32) / (u32::MAX as f32) * 1.3 - 0.1;
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_to_rgb_matches_scalar() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut plane = std::vec![0.0f32; w];
    prng_f32(&mut plane, 0xF32A_0001);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { grayf32_to_rgb_row::<false>(&plane, &mut simd, w) };
    sf::grayf32_to_rgb_row::<false>(&plane, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_to_rgba_matches_scalar() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut plane = std::vec![0.0f32; w];
    prng_f32(&mut plane, 0xF32A_0002);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { grayf32_to_rgba_row::<false>(&plane, &mut simd, w) };
    sf::grayf32_to_rgba_row::<false>(&plane, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_to_rgb_u16_matches_scalar() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut plane = std::vec![0.0f32; w];
    prng_f32(&mut plane, 0xF32A_0003);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe { grayf32_to_rgb_u16_row::<false>(&plane, &mut simd, w) };
    sf::grayf32_to_rgb_u16_row::<false>(&plane, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_to_rgba_u16_matches_scalar() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut plane = std::vec![0.0f32; w];
    prng_f32(&mut plane, 0xF32A_0004);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { grayf32_to_rgba_u16_row::<false>(&plane, &mut simd, w) };
    sf::grayf32_to_rgba_u16_row::<false>(&plane, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_to_rgb_f32_matches_scalar() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut plane = std::vec![0.0f32; w];
    prng_f32(&mut plane, 0xF32A_0005);
    let mut simd = std::vec![0.0f32; w * 3];
    let mut scal = std::vec![0.0f32; w * 3];
    unsafe { grayf32_to_rgb_f32_row::<false>(&plane, &mut simd, w) };
    sf::grayf32_to_rgb_f32_row::<false>(&plane, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_to_luma_matches_scalar() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut plane = std::vec![0.0f32; w];
    prng_f32(&mut plane, 0xF32A_0006);
    let mut simd = std::vec![0u8; w];
    let mut scal = std::vec![0u8; w];
    unsafe { grayf32_to_luma_row::<false>(&plane, &mut simd, w) };
    sf::grayf32_to_luma_row::<false>(&plane, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_to_luma_u16_matches_scalar() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut plane = std::vec![0.0f32; w];
    prng_f32(&mut plane, 0xF32A_0007);
    let mut simd = std::vec![0u16; w];
    let mut scal = std::vec![0u16; w];
    unsafe { grayf32_to_luma_u16_row::<false>(&plane, &mut simd, w) };
    sf::grayf32_to_luma_u16_row::<false>(&plane, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_to_luma_f32_matches_scalar() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut plane = std::vec![0.0f32; w];
    prng_f32(&mut plane, 0xF32A_0008);
    let mut simd = std::vec![0.0f32; w];
    let mut scal = std::vec![0.0f32; w];
    unsafe { grayf32_to_luma_f32_row::<false>(&plane, &mut simd, w) };
    sf::grayf32_to_luma_f32_row::<false>(&plane, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_to_hsv_matches_scalar() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut plane = std::vec![0.0f32; w];
    prng_f32(&mut plane, 0xF32A_0009);
    let mut sh = std::vec![0u8; w];
    let mut ss = std::vec![0u8; w];
    let mut sv = std::vec![0u8; w];
    let mut rh = std::vec![0u8; w];
    let mut rs = std::vec![0u8; w];
    let mut rv = std::vec![0u8; w];
    unsafe { grayf32_to_hsv_row::<false>(&plane, &mut sh, &mut ss, &mut sv, w) };
    sf::grayf32_to_hsv_row::<false>(&plane, &mut rh, &mut rs, &mut rv, w);
    assert_eq!(sh, rh, "H width={w}");
    assert_eq!(ss, rs, "S width={w}");
    assert_eq!(sv, rv, "V width={w}");
  }
}

// ---- Ya8 parity tests -------------------------------------------------------

fn prng_ya8(out: &mut [u8], seed: u32) {
  let mut s = seed;
  for v in out.iter_mut() {
    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    *v = (s >> 16) as u8;
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya8_to_rgb_matches_scalar() {
  use crate::row::scalar::ya8 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u8; w * 2];
    prng_ya8(&mut packed, 0xA800_0001);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { ya8_to_rgb_row(&packed, &mut simd, w) };
    sy::ya8_to_rgb_row(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya8_to_rgba_matches_scalar() {
  use crate::row::scalar::ya8 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u8; w * 2];
    prng_ya8(&mut packed, 0xA800_0002);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { ya8_to_rgba_row(&packed, &mut simd, w) };
    sy::ya8_to_rgba_row(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya8_to_rgb_u16_matches_scalar() {
  use crate::row::scalar::ya8 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u8; w * 2];
    prng_ya8(&mut packed, 0xA800_0003);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe { ya8_to_rgb_u16_row(&packed, &mut simd, w) };
    sy::ya8_to_rgb_u16_row(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya8_to_rgba_u16_matches_scalar() {
  use crate::row::scalar::ya8 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u8; w * 2];
    prng_ya8(&mut packed, 0xA800_0004);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { ya8_to_rgba_u16_row(&packed, &mut simd, w) };
    sy::ya8_to_rgba_u16_row(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya8_to_luma_matches_scalar() {
  use crate::row::scalar::ya8 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u8; w * 2];
    prng_ya8(&mut packed, 0xA800_0005);
    let mut simd = std::vec![0u8; w];
    let mut scal = std::vec![0u8; w];
    unsafe { ya8_to_luma_row(&packed, &mut simd, w) };
    sy::ya8_to_luma_row(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya8_to_luma_u16_matches_scalar() {
  use crate::row::scalar::ya8 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u8; w * 2];
    prng_ya8(&mut packed, 0xA800_0006);
    let mut simd = std::vec![0u16; w];
    let mut scal = std::vec![0u16; w];
    unsafe { ya8_to_luma_u16_row(&packed, &mut simd, w) };
    sy::ya8_to_luma_u16_row(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya8_to_hsv_matches_scalar() {
  use crate::row::scalar::ya8 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u8; w * 2];
    prng_ya8(&mut packed, 0xA800_0007);
    let mut sh = std::vec![0u8; w];
    let mut ss = std::vec![0u8; w];
    let mut sv = std::vec![0u8; w];
    let mut rh = std::vec![0u8; w];
    let mut rs = std::vec![0u8; w];
    let mut rv = std::vec![0u8; w];
    unsafe { ya8_to_hsv_row(&packed, &mut sh, &mut ss, &mut sv, w) };
    sy::ya8_to_hsv_row(&packed, &mut rh, &mut rs, &mut rv, w);
    assert_eq!(sh, rh, "H width={w}");
    assert_eq!(ss, rs, "S width={w}");
    assert_eq!(sv, rv, "V width={w}");
  }
}

// ---- Ya16 parity tests ------------------------------------------------------

fn prng_ya16(out: &mut [u16], seed: u32) {
  let mut buf = std::vec![0u8; out.len() * 2];
  prng(&mut buf, seed);
  for (i, o) in out.iter_mut().enumerate() {
    *o = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_to_rgb_matches_scalar() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 2];
    prng_ya16(&mut packed, 0xA160_0001);
    let mut simd = std::vec![0u8; w * 3];
    let mut scal = std::vec![0u8; w * 3];
    unsafe { ya16_to_rgb_row::<false>(&packed, &mut simd, w) };
    sy::ya16_to_rgb_row::<false>(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_to_rgba_matches_scalar() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 2];
    prng_ya16(&mut packed, 0xA160_0002);
    let mut simd = std::vec![0u8; w * 4];
    let mut scal = std::vec![0u8; w * 4];
    unsafe { ya16_to_rgba_row::<false>(&packed, &mut simd, w) };
    sy::ya16_to_rgba_row::<false>(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_to_rgb_u16_matches_scalar() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 2];
    prng_ya16(&mut packed, 0xA160_0003);
    let mut simd = std::vec![0u16; w * 3];
    let mut scal = std::vec![0u16; w * 3];
    unsafe { ya16_to_rgb_u16_row::<false>(&packed, &mut simd, w) };
    sy::ya16_to_rgb_u16_row::<false>(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_to_rgba_u16_matches_scalar() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 2];
    prng_ya16(&mut packed, 0xA160_0004);
    let mut simd = std::vec![0u16; w * 4];
    let mut scal = std::vec![0u16; w * 4];
    unsafe { ya16_to_rgba_u16_row::<false>(&packed, &mut simd, w) };
    sy::ya16_to_rgba_u16_row::<false>(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_to_luma_matches_scalar() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 2];
    prng_ya16(&mut packed, 0xA160_0005);
    let mut simd = std::vec![0u8; w];
    let mut scal = std::vec![0u8; w];
    unsafe { ya16_to_luma_row::<false>(&packed, &mut simd, w) };
    sy::ya16_to_luma_row::<false>(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_to_luma_u16_matches_scalar() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 2];
    prng_ya16(&mut packed, 0xA160_0006);
    let mut simd = std::vec![0u16; w];
    let mut scal = std::vec![0u16; w];
    unsafe { ya16_to_luma_u16_row::<false>(&packed, &mut simd, w) };
    sy::ya16_to_luma_u16_row::<false>(&packed, &mut scal, w);
    assert_eq!(simd, scal, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_to_hsv_matches_scalar() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 2];
    prng_ya16(&mut packed, 0xA160_0007);
    let mut sh = std::vec![0u8; w];
    let mut ss = std::vec![0u8; w];
    let mut sv = std::vec![0u8; w];
    let mut rh = std::vec![0u8; w];
    let mut rs = std::vec![0u8; w];
    let mut rv = std::vec![0u8; w];
    unsafe { ya16_to_hsv_row::<false>(&packed, &mut sh, &mut ss, &mut sv, w) };
    sy::ya16_to_hsv_row::<false>(&packed, &mut rh, &mut rs, &mut rv, w);
    assert_eq!(sh, rh, "H width={w}");
    assert_eq!(ss, rs, "S width={w}");
    assert_eq!(sv, rv, "V width={w}");
  }
}

// ---- BE parity tests: NEON BE kernel == scalar LE kernel on byte-swapped input ----

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_gray10_be_parity_rgb() {
  for &w in WIDTHS {
    let mut le = std::vec![0u16; w];
    prng16(&mut le, 0xBE10_0001);
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut simd_be = std::vec![0u8; w * 3];
    let mut scal_le = std::vec![0u8; w * 3];
    unsafe { gray_n_to_rgb_row::<10, true>(&be, &mut simd_be, w, true) };
    scalar::gray_n_to_rgb_row::<10, false>(&le, &mut scal_le, w, true);
    assert_eq!(simd_be, scal_le, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_gray16_be_parity_luma() {
  for &w in WIDTHS {
    let mut le = std::vec![0u16; w];
    prng16(&mut le, 0xBE16_0002);
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut simd_be = std::vec![0u8; w];
    let mut scal_le = std::vec![0u8; w];
    unsafe { gray16_to_luma_row::<true>(&be, &mut simd_be, w) };
    scalar::gray16_to_luma_row::<false>(&le, &mut scal_le, w);
    assert_eq!(simd_be, scal_le, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_grayf32_be_parity_luma() {
  use crate::row::scalar::grayf32 as sf;
  for &w in WIDTHS {
    let mut le = std::vec![0.0f32; w];
    prng_f32(&mut le, 0xBEF3_0003);
    let be: std::vec::Vec<f32> = le
      .iter()
      .map(|v| f32::from_bits(v.to_bits().swap_bytes()))
      .collect();
    let mut simd_be = std::vec![0u8; w];
    let mut scal_le = std::vec![0u8; w];
    unsafe { grayf32_to_luma_row::<true>(&be, &mut simd_be, w) };
    sf::grayf32_to_luma_row::<false>(&le, &mut scal_le, w);
    assert_eq!(simd_be, scal_le, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_be_parity_luma() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut le = std::vec![0u16; w * 2];
    prng_ya16(&mut le, 0xBEA1_0004);
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut simd_be = std::vec![0u8; w];
    let mut scal_le = std::vec![0u8; w];
    unsafe { ya16_to_luma_row::<true>(&be, &mut simd_be, w) };
    sy::ya16_to_luma_row::<false>(&le, &mut scal_le, w);
    assert_eq!(simd_be, scal_le, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_be_parity_rgb() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut le = std::vec![0u16; w * 2];
    prng_ya16(&mut le, 0xBEA1_0005);
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut simd_be = std::vec![0u8; w * 3];
    let mut scal_le = std::vec![0u8; w * 3];
    unsafe { ya16_to_rgb_row::<true>(&be, &mut simd_be, w) };
    sy::ya16_to_rgb_row::<false>(&le, &mut scal_le, w);
    assert_eq!(simd_be, scal_le, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_be_parity_rgba() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut le = std::vec![0u16; w * 2];
    prng_ya16(&mut le, 0xBEA1_0006);
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut simd_be = std::vec![0u8; w * 4];
    let mut scal_le = std::vec![0u8; w * 4];
    unsafe { ya16_to_rgba_row::<true>(&be, &mut simd_be, w) };
    sy::ya16_to_rgba_row::<false>(&le, &mut scal_le, w);
    assert_eq!(simd_be, scal_le, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_be_parity_rgb_u16() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut le = std::vec![0u16; w * 2];
    prng_ya16(&mut le, 0xBEA1_0007);
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut simd_be = std::vec![0u16; w * 3];
    let mut scal_le = std::vec![0u16; w * 3];
    unsafe { ya16_to_rgb_u16_row::<true>(&be, &mut simd_be, w) };
    sy::ya16_to_rgb_u16_row::<false>(&le, &mut scal_le, w);
    assert_eq!(simd_be, scal_le, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_be_parity_rgba_u16() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut le = std::vec![0u16; w * 2];
    prng_ya16(&mut le, 0xBEA1_0008);
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut simd_be = std::vec![0u16; w * 4];
    let mut scal_le = std::vec![0u16; w * 4];
    unsafe { ya16_to_rgba_u16_row::<true>(&be, &mut simd_be, w) };
    sy::ya16_to_rgba_u16_row::<false>(&le, &mut scal_le, w);
    assert_eq!(simd_be, scal_le, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_be_parity_luma_u16() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut le = std::vec![0u16; w * 2];
    prng_ya16(&mut le, 0xBEA1_0009);
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut simd_be = std::vec![0u16; w];
    let mut scal_le = std::vec![0u16; w];
    unsafe { ya16_to_luma_u16_row::<true>(&be, &mut simd_be, w) };
    sy::ya16_to_luma_u16_row::<false>(&le, &mut scal_le, w);
    assert_eq!(simd_be, scal_le, "width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
fn neon_ya16_be_parity_hsv() {
  use crate::row::scalar::ya16 as sy;
  for &w in WIDTHS {
    let mut le = std::vec![0u16; w * 2];
    prng_ya16(&mut le, 0xBEA1_000A);
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut sh_be = std::vec![0u8; w];
    let mut ss_be = std::vec![0u8; w];
    let mut sv_be = std::vec![0u8; w];
    let mut sh_le = std::vec![0u8; w];
    let mut ss_le = std::vec![0u8; w];
    let mut sv_le = std::vec![0u8; w];
    unsafe { ya16_to_hsv_row::<true>(&be, &mut sh_be, &mut ss_be, &mut sv_be, w) };
    sy::ya16_to_hsv_row::<false>(&le, &mut sh_le, &mut ss_le, &mut sv_le, w);
    assert_eq!(sh_be, sh_le, "H width={w}");
    assert_eq!(ss_be, ss_le, "S width={w}");
    assert_eq!(sv_be, sv_le, "V width={w}");
  }
}
