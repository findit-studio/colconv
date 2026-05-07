//! wasm-simd128 vs scalar equivalence tests for packed 16-bit RGB kernels.
//!
//! Compile-time gated via `#[cfg(target_feature = "simd128")]` — no runtime guard needed.
//! Uses Vec / assert_eq pattern per project conventions (no for-loop indexing).

use super::super::*;
use crate::row::scalar;

// ---- Helpers ---------------------------------------------------------------

fn pseudo_random_u16(n: usize, seed: u64) -> std::vec::Vec<u16> {
  // Simple LCG deterministic fill covering the full u16 range.
  let mut v = std::vec::Vec::with_capacity(n);
  let mut s = seed;
  for _ in 0..n {
    s = s
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    v.push((s >> 48) as u16);
  }
  v
}

fn widths() -> &'static [usize] {
  &[1, 7, 8, 9, 15, 16, 17, 31, 32, 33]
}

// =============================================================================
// Rgb48 kernels
// =============================================================================

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgb48_to_rgb_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 3, 0xDEAD_BEEF_1234_5678);
    let mut scalar_out = std::vec![0u8; w * 3];
    let mut simd_out = std::vec![0u8; w * 3];
    scalar::rgb48_to_rgb_row(&src, &mut scalar_out, w);
    unsafe { wasm_rgb48_to_rgb_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "rgb48→rgb diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgb48_to_rgba_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 3, 0xCAFE_BABE_DEAD_1234);
    let mut scalar_out = std::vec![0u8; w * 4];
    let mut simd_out = std::vec![0u8; w * 4];
    scalar::rgb48_to_rgba_row(&src, &mut scalar_out, w);
    unsafe { wasm_rgb48_to_rgba_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "rgb48→rgba diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgb48_to_rgb_u16_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 3, 0xFEED_FACE_ABCD_EF01);
    let mut scalar_out = std::vec![0u16; w * 3];
    let mut simd_out = std::vec![0u16; w * 3];
    scalar::rgb48_to_rgb_u16_row(&src, &mut scalar_out, w);
    unsafe { wasm_rgb48_to_rgb_u16_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "rgb48→rgb_u16 diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgb48_to_rgba_u16_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 3, 0x1234_5678_9ABC_DEF0);
    let mut scalar_out = std::vec![0u16; w * 4];
    let mut simd_out = std::vec![0u16; w * 4];
    scalar::rgb48_to_rgba_u16_row(&src, &mut scalar_out, w);
    unsafe { wasm_rgb48_to_rgba_u16_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "rgb48→rgba_u16 diverges (width={w})");
  }
}

// =============================================================================
// Bgr48 kernels
// =============================================================================

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgr48_to_rgb_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 3, 0xABCD_EF01_2345_6789);
    let mut scalar_out = std::vec![0u8; w * 3];
    let mut simd_out = std::vec![0u8; w * 3];
    scalar::bgr48_to_rgb_row(&src, &mut scalar_out, w);
    unsafe { wasm_bgr48_to_rgb_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "bgr48→rgb diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgr48_to_rgba_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 3, 0x9876_5432_10FE_DCBA);
    let mut scalar_out = std::vec![0u8; w * 4];
    let mut simd_out = std::vec![0u8; w * 4];
    scalar::bgr48_to_rgba_row(&src, &mut scalar_out, w);
    unsafe { wasm_bgr48_to_rgba_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "bgr48→rgba diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgr48_to_rgb_u16_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 3, 0x0011_2233_4455_6677);
    let mut scalar_out = std::vec![0u16; w * 3];
    let mut simd_out = std::vec![0u16; w * 3];
    scalar::bgr48_to_rgb_u16_row(&src, &mut scalar_out, w);
    unsafe { wasm_bgr48_to_rgb_u16_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "bgr48→rgb_u16 diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgr48_to_rgba_u16_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 3, 0x8899_AABB_CCDD_EEFF);
    let mut scalar_out = std::vec![0u16; w * 4];
    let mut simd_out = std::vec![0u16; w * 4];
    scalar::bgr48_to_rgba_u16_row(&src, &mut scalar_out, w);
    unsafe { wasm_bgr48_to_rgba_u16_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "bgr48→rgba_u16 diverges (width={w})");
  }
}

// =============================================================================
// Rgba64 kernels
// =============================================================================

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgba64_to_rgb_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 4, 0xF0F0_F0F0_0F0F_0F0F);
    let mut scalar_out = std::vec![0u8; w * 3];
    let mut simd_out = std::vec![0u8; w * 3];
    scalar::rgba64_to_rgb_row(&src, &mut scalar_out, w);
    unsafe { wasm_rgba64_to_rgb_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "rgba64→rgb diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgba64_to_rgba_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 4, 0x1357_9BDF_2468_ACE0);
    let mut scalar_out = std::vec![0u8; w * 4];
    let mut simd_out = std::vec![0u8; w * 4];
    scalar::rgba64_to_rgba_row(&src, &mut scalar_out, w);
    unsafe { wasm_rgba64_to_rgba_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "rgba64→rgba diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgba64_to_rgb_u16_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 4, 0x2468_ACE0_1357_9BDF);
    let mut scalar_out = std::vec![0u16; w * 3];
    let mut simd_out = std::vec![0u16; w * 3];
    scalar::rgba64_to_rgb_u16_row(&src, &mut scalar_out, w);
    unsafe { wasm_rgba64_to_rgb_u16_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "rgba64→rgb_u16 diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgba64_to_rgba_u16_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 4, 0x3C3C_C3C3_5A5A_A5A5);
    let mut scalar_out = std::vec![0u16; w * 4];
    let mut simd_out = std::vec![0u16; w * 4];
    scalar::rgba64_to_rgba_u16_row(&src, &mut scalar_out, w);
    unsafe { wasm_rgba64_to_rgba_u16_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "rgba64→rgba_u16 diverges (width={w})");
  }
}

// =============================================================================
// Bgra64 kernels
// =============================================================================

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgra64_to_rgb_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 4, 0x7654_3210_FEDC_BA98);
    let mut scalar_out = std::vec![0u8; w * 3];
    let mut simd_out = std::vec![0u8; w * 3];
    scalar::bgra64_to_rgb_row(&src, &mut scalar_out, w);
    unsafe { wasm_bgra64_to_rgb_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "bgra64→rgb diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgra64_to_rgba_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 4, 0xAABB_CCDD_EEFF_0011);
    let mut scalar_out = std::vec![0u8; w * 4];
    let mut simd_out = std::vec![0u8; w * 4];
    scalar::bgra64_to_rgba_row(&src, &mut scalar_out, w);
    unsafe { wasm_bgra64_to_rgba_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "bgra64→rgba diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgra64_to_rgb_u16_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 4, 0x5566_7788_99AA_BBCC);
    let mut scalar_out = std::vec![0u16; w * 3];
    let mut simd_out = std::vec![0u16; w * 3];
    scalar::bgra64_to_rgb_u16_row(&src, &mut scalar_out, w);
    unsafe { wasm_bgra64_to_rgb_u16_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "bgra64→rgb_u16 diverges (width={w})");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgra64_to_rgba_u16_matches_scalar() {
  for &w in widths() {
    let src = pseudo_random_u16(w * 4, 0xDDEE_FF00_1122_3344);
    let mut scalar_out = std::vec![0u16; w * 4];
    let mut simd_out = std::vec![0u16; w * 4];
    scalar::bgra64_to_rgba_u16_row(&src, &mut scalar_out, w);
    unsafe { wasm_bgra64_to_rgba_u16_row(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "bgra64→rgba_u16 diverges (width={w})");
  }
}
