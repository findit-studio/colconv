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
    scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_rgb48_to_rgb_row::<false>(&src, &mut simd_out, w) };
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
    scalar::rgb48_to_rgba_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_rgb48_to_rgba_row::<false>(&src, &mut simd_out, w) };
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
    scalar::rgb48_to_rgb_u16_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_rgb48_to_rgb_u16_row::<false>(&src, &mut simd_out, w) };
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
    scalar::rgb48_to_rgba_u16_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_rgb48_to_rgba_u16_row::<false>(&src, &mut simd_out, w) };
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
    scalar::bgr48_to_rgb_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_bgr48_to_rgb_row::<false>(&src, &mut simd_out, w) };
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
    scalar::bgr48_to_rgba_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_bgr48_to_rgba_row::<false>(&src, &mut simd_out, w) };
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
    scalar::bgr48_to_rgb_u16_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_bgr48_to_rgb_u16_row::<false>(&src, &mut simd_out, w) };
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
    scalar::bgr48_to_rgba_u16_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_bgr48_to_rgba_u16_row::<false>(&src, &mut simd_out, w) };
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
    scalar::rgba64_to_rgb_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_rgba64_to_rgb_row::<false>(&src, &mut simd_out, w) };
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
    scalar::rgba64_to_rgba_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_rgba64_to_rgba_row::<false>(&src, &mut simd_out, w) };
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
    scalar::rgba64_to_rgb_u16_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_rgba64_to_rgb_u16_row::<false>(&src, &mut simd_out, w) };
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
    scalar::rgba64_to_rgba_u16_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_rgba64_to_rgba_u16_row::<false>(&src, &mut simd_out, w) };
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
    scalar::bgra64_to_rgb_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_bgra64_to_rgb_row::<false>(&src, &mut simd_out, w) };
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
    scalar::bgra64_to_rgba_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_bgra64_to_rgba_row::<false>(&src, &mut simd_out, w) };
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
    scalar::bgra64_to_rgb_u16_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_bgra64_to_rgb_u16_row::<false>(&src, &mut simd_out, w) };
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
    scalar::bgra64_to_rgba_u16_row::<false>(&src, &mut scalar_out, w);
    unsafe { wasm_bgra64_to_rgba_u16_row::<false>(&src, &mut simd_out, w) };
    assert_eq!(scalar_out, simd_out, "bgra64→rgba_u16 diverges (width={w})");
  }
}

// =============================================================================
// Lane-order regression tests (Codex Bug 1: stride-4 deinterleave)
// =============================================================================
//
// These tests use asymmetric per-pixel channel inputs that catch
// per-pixel mixing bugs that uniform-value tests miss. The earlier
// `*_matches_scalar` tests use a pseudo-random fill that *would* catch
// the bug too, but the dedicated tests below pin down the exact
// expected per-pixel index mapping with hand-derived values, making
// regression diagnosis trivial if the deinterleave is ever broken
// again.
//
// Pattern: R[n] = n+1, G[n] = 100+n, B[n] = 200+n, A[n] = 50+n.
// Per-channel offsets (1, 100, 200, 50) are chosen so that no two
// channels alias for the first ~50 pixel indices, while still fitting
// in u16. Comparing SIMD vs scalar over the u16 outputs (which preserve
// the full 16-bit value) directly verifies that each output pixel
// pulls from the correct per-pixel channel triple/quad.

/// Build an asymmetric Rgba64 row: pixel `n` = [R=n+1, G=100+n, B=200+n, A=50+n].
fn make_rgba64_lane_order(width: usize) -> std::vec::Vec<u16> {
  let mut src = std::vec::Vec::with_capacity(width * 4);
  for n in 0..width {
    src.push((n as u16).wrapping_add(1)); // R
    src.push((n as u16).wrapping_add(100)); // G
    src.push((n as u16).wrapping_add(200)); // B
    src.push((n as u16).wrapping_add(50)); // A
  }
  src
}

/// Build an asymmetric Bgra64 row: pixel `n` memory = [B=200+n, G=100+n, R=n+1, A=50+n].
fn make_bgra64_lane_order(width: usize) -> std::vec::Vec<u16> {
  let mut src = std::vec::Vec::with_capacity(width * 4);
  for n in 0..width {
    src.push((n as u16).wrapping_add(200)); // B
    src.push((n as u16).wrapping_add(100)); // G
    src.push((n as u16).wrapping_add(1)); // R
    src.push((n as u16).wrapping_add(50)); // A
  }
  src
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgba64_to_rgba_u16_lane_order_regression() {
  // 9 pixels exercises 1 SIMD iteration (8 px) + 1-pixel scalar tail.
  let src = make_rgba64_lane_order(9);
  let mut simd_out = std::vec![0u16; 9 * 4];
  let mut scalar_out = std::vec![0u16; 9 * 4];
  unsafe { wasm_rgba64_to_rgba_u16_row(&src, &mut simd_out, 9) };
  scalar::rgba64_to_rgba_u16_row(&src, &mut scalar_out, 9);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba_u16 lane order: SIMD vs scalar mismatch (channel mixing?)"
  );
  // Also pin down the expected values explicitly: pixel n preserves
  // R=n+1, G=100+n, B=200+n, A=50+n.
  for n in 0..9 {
    assert_eq!(simd_out[n * 4], (n as u16) + 1, "R at pixel {n}");
    assert_eq!(simd_out[n * 4 + 1], (n as u16) + 100, "G at pixel {n}");
    assert_eq!(simd_out[n * 4 + 2], (n as u16) + 200, "B at pixel {n}");
    assert_eq!(simd_out[n * 4 + 3], (n as u16) + 50, "A at pixel {n}");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_rgba64_to_rgb_u16_lane_order_regression() {
  let src = make_rgba64_lane_order(9);
  let mut simd_out = std::vec![0u16; 9 * 3];
  let mut scalar_out = std::vec![0u16; 9 * 3];
  unsafe { wasm_rgba64_to_rgb_u16_row(&src, &mut simd_out, 9) };
  scalar::rgba64_to_rgb_u16_row(&src, &mut scalar_out, 9);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb_u16 lane order: SIMD vs scalar mismatch"
  );
  for n in 0..9 {
    assert_eq!(simd_out[n * 3], (n as u16) + 1, "R at pixel {n}");
    assert_eq!(simd_out[n * 3 + 1], (n as u16) + 100, "G at pixel {n}");
    assert_eq!(simd_out[n * 3 + 2], (n as u16) + 200, "B at pixel {n}");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgra64_to_rgba_u16_lane_order_regression() {
  // Bgra64 source memory: [B, G, R, A] per pixel; output is RGBA.
  let src = make_bgra64_lane_order(9);
  let mut simd_out = std::vec![0u16; 9 * 4];
  let mut scalar_out = std::vec![0u16; 9 * 4];
  unsafe { wasm_bgra64_to_rgba_u16_row(&src, &mut simd_out, 9) };
  scalar::bgra64_to_rgba_u16_row(&src, &mut scalar_out, 9);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba_u16 lane order: SIMD vs scalar mismatch (B↔R swap or alpha?)"
  );
  // Output is RGBA: R=n+1, G=100+n, B=200+n, A=50+n per pixel n.
  for n in 0..9 {
    assert_eq!(simd_out[n * 4], (n as u16) + 1, "R at pixel {n}");
    assert_eq!(simd_out[n * 4 + 1], (n as u16) + 100, "G at pixel {n}");
    assert_eq!(simd_out[n * 4 + 2], (n as u16) + 200, "B at pixel {n}");
    assert_eq!(simd_out[n * 4 + 3], (n as u16) + 50, "A at pixel {n}");
  }
}

#[cfg(target_feature = "simd128")]
#[test]
fn wasm_bgra64_to_rgb_u16_lane_order_regression() {
  let src = make_bgra64_lane_order(9);
  let mut simd_out = std::vec![0u16; 9 * 3];
  let mut scalar_out = std::vec![0u16; 9 * 3];
  unsafe { wasm_bgra64_to_rgb_u16_row(&src, &mut simd_out, 9) };
  scalar::bgra64_to_rgb_u16_row(&src, &mut scalar_out, 9);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgb_u16 lane order: SIMD vs scalar mismatch"
  );
  for n in 0..9 {
    assert_eq!(simd_out[n * 3], (n as u16) + 1, "R at pixel {n}");
    assert_eq!(simd_out[n * 3 + 1], (n as u16) + 100, "G at pixel {n}");
    assert_eq!(simd_out[n * 3 + 2], (n as u16) + 200, "B at pixel {n}");
  }
}
