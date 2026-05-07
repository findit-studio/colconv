//! wasm-simd128 parity tests for the high-bit-depth planar-GBR kernels (Tier 10b).
//!
//! Covers BITS=10 and BITS=16 at widths [1, 7, 8, 16, 17, 32, 33, 64, 128, 130].
//!
//! This module is gated `#[cfg(target_arch = "wasm32")]` (via `row::arch::mod`)
//! and only compiles when targeting wasm32. wasm SIMD is enabled at compile
//! time via `RUSTFLAGS=-C target-feature=+simd128`; there is no runtime
//! `target_feature` detection — the kernels lower directly to wasm-simd128
//! intrinsics or fail to compile if the feature isn't enabled.

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
fn simd128_gbr_to_rgb_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbr_to_rgb_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgb_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<16, false>(&g, &b, &r, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbr_to_rgb_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgba_opaque_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbr_to_rgba_opaque_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgba_opaque_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<16, false>(&g, &b, &r, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbr_to_rgba_opaque_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbra_to_rgba_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbra_to_rgba_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbra_to_rgba_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbra_to_rgba_high_bit<16> diverges (width={w})"
    );
  }
}

// ---- u16 output ----------------------------------------------------------

#[test]
fn simd128_gbr_to_rgb_u16_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbr_to_rgb_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgb_u16_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbr_to_rgb_u16_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgba_opaque_u16_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbr_to_rgba_opaque_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgba_opaque_u16_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::gbr_to_rgba_opaque_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbr_to_rgba_opaque_u16_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbra_to_rgba_u16_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbra_to_rgba_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
fn simd128_gbra_to_rgba_u16_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_wasm, w);
    }
    assert_eq!(
      out_scalar, out_wasm,
      "simd128 gbra_to_rgba_u16_high_bit<16> diverges (width={w})"
    );
  }
}

// ---- BE parity: simd128<BITS, true> output must match simd128<BITS, false> --

fn byte_swap_plane(plane: &[u16]) -> std::vec::Vec<u16> {
  plane.iter().map(|v| v.swap_bytes()).collect()
}

#[test]
fn simd128_gbr_to_rgb_high_bit_be_matches_le_bits10() {
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
      "simd128 gbr_to_rgb_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgb_high_bit_be_matches_le_bits16() {
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
      "simd128 gbr_to_rgb_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgba_opaque_high_bit_be_matches_le_bits10() {
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
      "simd128 gbr_to_rgba_opaque_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgba_opaque_high_bit_be_matches_le_bits16() {
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
      "simd128 gbr_to_rgba_opaque_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbra_to_rgba_high_bit_be_matches_le_bits10() {
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
      "simd128 gbra_to_rgba_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbra_to_rgba_high_bit_be_matches_le_bits16() {
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
      "simd128 gbra_to_rgba_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgb_u16_high_bit_be_matches_le_bits10() {
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
      "simd128 gbr_to_rgb_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgb_u16_high_bit_be_matches_le_bits16() {
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
      "simd128 gbr_to_rgb_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgba_opaque_u16_high_bit_be_matches_le_bits10() {
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
      "simd128 gbr_to_rgba_opaque_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbr_to_rgba_opaque_u16_high_bit_be_matches_le_bits16() {
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
      "simd128 gbr_to_rgba_opaque_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbra_to_rgba_u16_high_bit_be_matches_le_bits10() {
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
      "simd128 gbra_to_rgba_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
fn simd128_gbra_to_rgba_u16_high_bit_be_matches_le_bits16() {
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
      "simd128 gbra_to_rgba_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}
