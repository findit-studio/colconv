//! NEON parity tests for the high-bit-depth planar-GBR kernels (Tier 10b).
//!
//! Covers BITS=10 and BITS=16 at widths [1, 7, 8, 16, 17, 32, 33, 64, 128, 130].
//! Each test asserts SIMD output == scalar output.

use super::*;

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

// ---- u8 output -----------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<16>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<16>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_high_bit_row::<10>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<10>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgba_opaque_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_high_bit_row::<16>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<16>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgba_opaque_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<10>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<10>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<16>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<16>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_high_bit<16> diverges (width={w})"
    );
  }
}

// ---- u16 output ----------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_u16_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<10>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<10>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_u16_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<16>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<16>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_u16_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_u16_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::gbr_to_rgba_opaque_u16_high_bit_row::<10>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<10>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgba_opaque_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_u16_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::gbr_to_rgba_opaque_u16_high_bit_row::<16>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<16>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgba_opaque_u16_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_u16_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<10>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<10>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_u16_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<16>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<16>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_u16_high_bit<16> diverges (width={w})"
    );
  }
}
