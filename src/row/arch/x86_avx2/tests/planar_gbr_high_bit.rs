//! AVX2 parity tests for the high-bit-depth planar-GBR kernels (Tier 10b).
//!
//! Covers BITS=10 and BITS=16 at widths [1, 7, 8, 16, 17, 32, 33, 64, 128, 130].
//! Each test early-returns when AVX2 is not available (sanitizer/Miri safety).

use super::super::*;

fn gbr_plane_u16<const BITS: u32>(width: usize, seed: u32) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  let mut state = seed;
  (0..width)
    .map(|_| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (state & mask) as u16
    })
    .collect()
}

fn gbr_plane_u16_dirty<const BITS: u32>(width: usize, dirty_upper: u16) -> std::vec::Vec<u16> {
  let clean_mask = ((1u32 << BITS) - 1) as u16;
  (0..width)
    .map(|i| (i as u16 & clean_mask) | dirty_upper)
    .collect()
}

// ---- u8 output -----------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_high_bit_matches_scalar_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgb_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_high_bit_matches_scalar_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<16, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgb_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgba_opaque_high_bit_matches_scalar_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgba_opaque_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgba_opaque_high_bit_matches_scalar_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<16, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgba_opaque_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_high_bit_matches_scalar_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbra_to_rgba_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_high_bit_matches_scalar_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbra_to_rgba_high_bit<16> diverges (width={w})"
    );
  }
}

// ---- u16 output ----------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_u16_high_bit_matches_scalar_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgb_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_u16_high_bit_matches_scalar_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgb_u16_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgba_opaque_u16_high_bit_matches_scalar_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx = std::vec![0u16; w * 4];
    scalar::gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgba_opaque_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgba_opaque_u16_high_bit_matches_scalar_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx = std::vec![0u16; w * 4];
    scalar::gbr_to_rgba_opaque_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgba_opaque_u16_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_u16_high_bit_matches_scalar_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbra_to_rgba_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_u16_high_bit_matches_scalar_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbra_to_rgba_u16_high_bit<16> diverges (width={w})"
    );
  }
}

// ---- Upper-bits masking: AVX2 must match scalar for dirty inputs ----------

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_high_bit_upper_bits_masked_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let b = gbr_plane_u16_dirty::<10>(w, 0x0800);
    let r = gbr_plane_u16_dirty::<10>(w, 0x0400);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgb_high_bit<10> dirty-input diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_high_bit_upper_bits_masked_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let b = gbr_plane_u16_dirty::<10>(w, 0x0800);
    let r = gbr_plane_u16_dirty::<10>(w, 0x0400);
    let a = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbra_to_rgba_high_bit<10> dirty-input diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_u16_high_bit_upper_bits_masked_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let b = gbr_plane_u16_dirty::<10>(w, 0x0800);
    let r = gbr_plane_u16_dirty::<10>(w, 0x0400);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbr_to_rgb_u16_high_bit<10> dirty-input diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_u16_high_bit_upper_bits_masked_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let b = gbr_plane_u16_dirty::<10>(w, 0x0800);
    let r = gbr_plane_u16_dirty::<10>(w, 0x0400);
    let a = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX2 gbra_to_rgba_u16_high_bit<10> dirty-input diverges (width={w})"
    );
  }
}

// ---- BE parity: AVX2<BITS, true> output must match AVX2<BITS, false> --------

fn byte_swap_plane(plane: &[u16]) -> std::vec::Vec<u16> {
  plane.iter().map(|v| v.swap_bytes()).collect()
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_high_bit_be_matches_le_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_le, w);
      gbr_to_rgb_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbr_to_rgb_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_high_bit_be_matches_le_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      gbr_to_rgb_high_bit_row::<16, false>(&g, &b, &r, &mut out_le, w);
      gbr_to_rgb_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbr_to_rgb_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgba_opaque_high_bit_be_matches_le_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut out_le, w);
      gbr_to_rgba_opaque_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbr_to_rgba_opaque_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgba_opaque_high_bit_be_matches_le_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<16, false>(&g, &b, &r, &mut out_le, w);
      gbr_to_rgba_opaque_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbr_to_rgba_opaque_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_high_bit_be_matches_le_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let a_be = byte_swap_plane(&a);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_le, w);
      gbra_to_rgba_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbra_to_rgba_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_high_bit_be_matches_le_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let a_be = byte_swap_plane(&a);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      gbra_to_rgba_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_le, w);
      gbra_to_rgba_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbra_to_rgba_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_u16_high_bit_be_matches_le_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let mut out_le = std::vec![0u16; w * 3];
    let mut out_be = std::vec![0u16; w * 3];
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_le, w);
      gbr_to_rgb_u16_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbr_to_rgb_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgb_u16_high_bit_be_matches_le_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let mut out_le = std::vec![0u16; w * 3];
    let mut out_be = std::vec![0u16; w * 3];
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_le, w);
      gbr_to_rgb_u16_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbr_to_rgb_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgba_opaque_u16_high_bit_be_matches_le_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_le, w);
      gbr_to_rgba_opaque_u16_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbr_to_rgba_opaque_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbr_to_rgba_opaque_u16_high_bit_be_matches_le_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_le, w);
      gbr_to_rgba_opaque_u16_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbr_to_rgba_opaque_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_u16_high_bit_be_matches_le_bits10() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let a_be = byte_swap_plane(&a);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_le, w);
      gbra_to_rgba_u16_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbra_to_rgba_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx2_gbra_to_rgba_u16_high_bit_be_matches_le_bits16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let g_be = byte_swap_plane(&g);
    let b_be = byte_swap_plane(&b);
    let r_be = byte_swap_plane(&r);
    let a_be = byte_swap_plane(&a);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_le, w);
      gbra_to_rgba_u16_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "AVX2 gbra_to_rgba_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}
