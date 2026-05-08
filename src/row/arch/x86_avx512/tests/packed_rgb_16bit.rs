//! AVX-512 parity tests for the packed 16-bit RGB/BGR/RGBA/BGRA kernels
//! (Tier 8 finish).
//!
//! Each test early-returns when `avx512bw` is not detected at runtime,
//! satisfying the x86 SIMD test guard requirement so CI sanitizer / Miri
//! runs do not fail.
//!
//! Width = 33 exercises: 1 SIMD iteration (32 px) + 1-pixel scalar tail.
//! Width = 32 exercises: exact hot-loop (no tail).
//! Width = 1  exercises: tail-only (SIMD loop skipped entirely).
//!
//! Lane-order regression: asymmetric R/G/B/A inputs verify that channels
//! are not swapped by the shuffle deinterleave.

use super::super::*;

// ---- helpers ----------------------------------------------------------------

fn make_rgb48_src(width: usize, seed: u16) -> std::vec::Vec<u16> {
  (0..width * 3)
    .map(|i| (i as u16).wrapping_mul(seed).wrapping_add(0x1357))
    .collect()
}

fn make_rgba64_src(width: usize, seed: u16) -> std::vec::Vec<u16> {
  (0..width * 4)
    .map(|i| (i as u16).wrapping_mul(seed).wrapping_add(0x2468))
    .collect()
}

/// Asymmetric Rgb48 row: pixel `i` has R=0x1100+i, G=0x2200+i, B=0x3300+i.
fn make_rgb48_asymmetric(width: usize) -> std::vec::Vec<u16> {
  let mut src = std::vec::Vec::with_capacity(width * 3);
  for i in 0..width {
    src.push(0x1100u16.wrapping_add(i as u16)); // R
    src.push(0x2200u16.wrapping_add(i as u16)); // G
    src.push(0x3300u16.wrapping_add(i as u16)); // B
  }
  src
}

/// Asymmetric Rgba64 row: pixel `i` has R=0x1100+i, G=0x2200+i, B=0x3300+i,
/// A=0x4400+i.
fn make_rgba64_asymmetric(width: usize) -> std::vec::Vec<u16> {
  let mut src = std::vec::Vec::with_capacity(width * 4);
  for i in 0..width {
    src.push(0x1100u16.wrapping_add(i as u16)); // R
    src.push(0x2200u16.wrapping_add(i as u16)); // G
    src.push(0x3300u16.wrapping_add(i as u16)); // B
    src.push(0x4400u16.wrapping_add(i as u16)); // A
  }
  src
}

// =============================================================================
// Rgb48 → u8 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgb48_to_rgb_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(33, 0x0101);
  let mut simd_out = std::vec![0u8; 33 * 3];
  let mut scalar_out = std::vec![0u8; 33 * 3];
  unsafe { avx512_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb width=33: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgb48_to_rgb_exact32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(32, 0xF0F0);
  let mut simd_out = std::vec![0u8; 32 * 3];
  let mut scalar_out = std::vec![0u8; 32 * 3];
  unsafe { avx512_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 32) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 32);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb exact-32: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgb48_to_rgb_width1_tail_only() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = [0x1234u16, 0x5678, 0x9ABC];
  let mut simd_out = [0u8; 3];
  let mut scalar_out = [0u8; 3];
  unsafe { avx512_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 1) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 1);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb width=1: tail-only mismatch"
  );
}

/// Lane-order regression: R/G/B channels must not be swapped.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgb48_to_rgb_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_asymmetric(33);
  let mut simd_out = std::vec![0u8; 33 * 3];
  let mut scalar_out = std::vec![0u8; 33 * 3];
  unsafe { avx512_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb lane order: SIMD vs scalar mismatch (channel swap?)"
  );
}

// =============================================================================
// Rgb48 → u8 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgb48_to_rgba_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(33, 0x0303);
  let mut simd_out = std::vec![0u8; 33 * 4];
  let mut scalar_out = std::vec![0u8; 33 * 4];
  unsafe { avx512_rgb48_to_rgba_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgb48_to_rgba_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgba width=33: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgb48 → u16 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgb48_to_rgb_u16_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(33, 0x0505);
  let mut simd_out = std::vec![0u16; 33 * 3];
  let mut scalar_out = std::vec![0u16; 33 * 3];
  unsafe { avx512_rgb48_to_rgb_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgb48_to_rgb_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb_u16 width=33: SIMD vs scalar mismatch"
  );
}

/// Lane-order regression for u16 output.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgb48_to_rgb_u16_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_asymmetric(33);
  let mut simd_out = std::vec![0u16; 33 * 3];
  let mut scalar_out = std::vec![0u16; 33 * 3];
  unsafe { avx512_rgb48_to_rgb_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgb48_to_rgb_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb_u16 lane order: SIMD vs scalar mismatch (channel swap?)"
  );
}

// =============================================================================
// Rgb48 → u16 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgb48_to_rgba_u16_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(33, 0x0707);
  let mut simd_out = std::vec![0u16; 33 * 4];
  let mut scalar_out = std::vec![0u16; 33 * 4];
  unsafe { avx512_rgb48_to_rgba_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgb48_to_rgba_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgba_u16 width=33: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgr48 → u8 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgr48_to_rgb_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(33, 0x1111);
  let mut simd_out = std::vec![0u8; 33 * 3];
  let mut scalar_out = std::vec![0u8; 33 * 3];
  unsafe { avx512_bgr48_to_rgb_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgr48_to_rgb_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgb width=33: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgr48_to_rgb_exact32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(32, 0xA1A1);
  let mut simd_out = std::vec![0u8; 32 * 3];
  let mut scalar_out = std::vec![0u8; 32 * 3];
  unsafe { avx512_bgr48_to_rgb_row::<false>(&src, &mut simd_out, 32) };
  scalar::bgr48_to_rgb_row::<false>(&src, &mut scalar_out, 32);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgb exact-32: SIMD vs scalar mismatch"
  );
}

/// Lane-order regression: B↔R swap must produce correct channel positions.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgr48_to_rgb_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_asymmetric(33);
  let mut simd_out = std::vec![0u8; 33 * 3];
  let mut scalar_out = std::vec![0u8; 33 * 3];
  unsafe { avx512_bgr48_to_rgb_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgr48_to_rgb_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgb lane order (B↔R swap): SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgr48 → u8 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgr48_to_rgba_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(33, 0x2222);
  let mut simd_out = std::vec![0u8; 33 * 4];
  let mut scalar_out = std::vec![0u8; 33 * 4];
  unsafe { avx512_bgr48_to_rgba_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgr48_to_rgba_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgba width=33: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgr48 → u16 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgr48_to_rgb_u16_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(33, 0x3333);
  let mut simd_out = std::vec![0u16; 33 * 3];
  let mut scalar_out = std::vec![0u16; 33 * 3];
  unsafe { avx512_bgr48_to_rgb_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgr48_to_rgb_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgb_u16 width=33: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgr48 → u16 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgr48_to_rgba_u16_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgb48_src(33, 0x4444);
  let mut simd_out = std::vec![0u16; 33 * 4];
  let mut scalar_out = std::vec![0u16; 33 * 4];
  unsafe { avx512_bgr48_to_rgba_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgr48_to_rgba_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgba_u16 width=33: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgba64 → u8 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgb_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_src(33, 0xAAAA);
  let mut simd_out = std::vec![0u8; 33 * 3];
  let mut scalar_out = std::vec![0u8; 33 * 3];
  unsafe { avx512_rgba64_to_rgb_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgba64_to_rgb_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb width=33: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgb_exact32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_src(32, 0x0F0F);
  let mut simd_out = std::vec![0u8; 32 * 3];
  let mut scalar_out = std::vec![0u8; 32 * 3];
  unsafe { avx512_rgba64_to_rgb_row::<false>(&src, &mut simd_out, 32) };
  scalar::rgba64_to_rgb_row::<false>(&src, &mut scalar_out, 32);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb exact-32: SIMD vs scalar mismatch"
  );
}

/// Lane-order regression: RGBA channels and alpha-discard correct.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgb_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_asymmetric(33);
  let mut simd_out = std::vec![0u8; 33 * 3];
  let mut scalar_out = std::vec![0u8; 33 * 3];
  unsafe { avx512_rgba64_to_rgb_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgba64_to_rgb_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb lane order: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgba64 → u8 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgba_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_src(33, 0xBBBB);
  let mut simd_out = std::vec![0u8; 33 * 4];
  let mut scalar_out = std::vec![0u8; 33 * 4];
  unsafe { avx512_rgba64_to_rgba_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgba64_to_rgba_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba width=33: SIMD vs scalar mismatch"
  );
}

/// Lane-order regression: source alpha must pass through to position 3.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgba_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_asymmetric(33);
  let mut simd_out = std::vec![0u8; 33 * 4];
  let mut scalar_out = std::vec![0u8; 33 * 4];
  unsafe { avx512_rgba64_to_rgba_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgba64_to_rgba_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba lane order (alpha passthrough): SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgba64 → u16 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgb_u16_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_src(33, 0xCCCC);
  let mut simd_out = std::vec![0u16; 33 * 3];
  let mut scalar_out = std::vec![0u16; 33 * 3];
  unsafe { avx512_rgba64_to_rgb_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgba64_to_rgb_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb_u16 width=33: SIMD vs scalar mismatch"
  );
}

/// Lane-order regression for u16 output.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgb_u16_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_asymmetric(33);
  let mut simd_out = std::vec![0u16; 33 * 3];
  let mut scalar_out = std::vec![0u16; 33 * 3];
  unsafe { avx512_rgba64_to_rgb_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgba64_to_rgb_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb_u16 lane order: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgba64 → u16 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgba_u16_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_src(33, 0xDDDD);
  let mut simd_out = std::vec![0u16; 33 * 4];
  let mut scalar_out = std::vec![0u16; 33 * 4];
  unsafe { avx512_rgba64_to_rgba_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgba64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba_u16 width=33: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgba_u16_width1_tail_only() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = [0x1234u16, 0x5678, 0x9ABC, 0xDEF0]; // R, G, B, A
  let mut simd_out = [0u16; 4];
  let mut scalar_out = [0u16; 4];
  unsafe { avx512_rgba64_to_rgba_u16_row::<false>(&src, &mut simd_out, 1) };
  scalar::rgba64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 1);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba_u16 width=1: tail-only mismatch"
  );
}

/// Lane-order regression: all 4 channels preserved in correct positions.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgba_u16_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_asymmetric(33);
  let mut simd_out = std::vec![0u16; 33 * 4];
  let mut scalar_out = std::vec![0u16; 33 * 4];
  unsafe { avx512_rgba64_to_rgba_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::rgba64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba_u16 lane order (identity copy): SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgra64 → u8 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgb_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_src(33, 0x1234);
  let mut simd_out = std::vec![0u8; 33 * 3];
  let mut scalar_out = std::vec![0u8; 33 * 3];
  unsafe { avx512_bgra64_to_rgb_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgra64_to_rgb_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgb width=33: SIMD vs scalar mismatch"
  );
}

/// Lane-order regression: B↔R swap + alpha discard for BGRA source.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgb_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_asymmetric(33);
  let mut simd_out = std::vec![0u8; 33 * 3];
  let mut scalar_out = std::vec![0u8; 33 * 3];
  unsafe { avx512_bgra64_to_rgb_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgra64_to_rgb_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgb lane order (B↔R swap): SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgra64 → u8 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgba_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_src(33, 0x5678);
  let mut simd_out = std::vec![0u8; 33 * 4];
  let mut scalar_out = std::vec![0u8; 33 * 4];
  unsafe { avx512_bgra64_to_rgba_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgra64_to_rgba_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba width=33: SIMD vs scalar mismatch"
  );
}

/// Lane-order regression: B↔R swap; alpha passes through.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgba_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_asymmetric(33);
  let mut simd_out = std::vec![0u8; 33 * 4];
  let mut scalar_out = std::vec![0u8; 33 * 4];
  unsafe { avx512_bgra64_to_rgba_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgra64_to_rgba_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba lane order (B↔R swap + alpha): SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgra64 → u16 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgb_u16_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_src(33, 0x9ABC);
  let mut simd_out = std::vec![0u16; 33 * 3];
  let mut scalar_out = std::vec![0u16; 33 * 3];
  unsafe { avx512_bgra64_to_rgb_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgra64_to_rgb_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgb_u16 width=33: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgra64 → u16 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgba_u16_matches_scalar_width33() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_src(33, 0xDEF0);
  let mut simd_out = std::vec![0u16; 33 * 4];
  let mut scalar_out = std::vec![0u16; 33 * 4];
  unsafe { avx512_bgra64_to_rgba_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgra64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba_u16 width=33: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgba_u16_width1_tail_only() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = [0x1111u16, 0x2222, 0x3333, 0x4444]; // B, G, R, A
  let mut simd_out = [0u16; 4];
  let mut scalar_out = [0u16; 4];
  unsafe { avx512_bgra64_to_rgba_u16_row::<false>(&src, &mut simd_out, 1) };
  scalar::bgra64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 1);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba_u16 width=1: tail-only mismatch"
  );
}

/// Lane-order regression: B↔R swap; alpha position 3 preserved.
#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgba_u16_lane_order_regression() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_asymmetric(33);
  let mut simd_out = std::vec![0u16; 33 * 4];
  let mut scalar_out = std::vec![0u16; 33 * 4];
  unsafe { avx512_bgra64_to_rgba_u16_row::<false>(&src, &mut simd_out, 33) };
  scalar::bgra64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 33);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba_u16 lane order (B↔R swap + alpha preserve): SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Lane-order regression tests with hand-derived expected values
// (Codex Bug 2: stride-4 reshape paired the same vector twice)
// =============================================================================
//
// The earlier `*_lane_order_regression` tests use a high-byte asymmetric
// pattern (`0x1100+i`, `0x2200+i`, ...) and compare SIMD vs scalar.
// While that pattern catches per-pixel mixing in u16 outputs, the
// hand-derived per-pixel asserts below pin down the exact mapping that
// the SIMD path must produce, making regression diagnosis trivial if
// the deinterleave is ever broken again. Pattern matches the wasm
// sibling for parity:
//
//   R[n] = n+1, G[n] = 100+n, B[n] = 200+n, A[n] = 50+n
//
// Width 33 exercises 1 SIMD iteration (32 px) + 1-pixel scalar tail —
// catching reshape bugs that only manifest in the SIMD hot loop.

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

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgba_u16_lane_order_handcheck() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_lane_order(33);
  let mut simd_out = std::vec![0u16; 33 * 4];
  unsafe { avx512_rgba64_to_rgba_u16_row::<false>(&src, &mut simd_out, 33) };
  for n in 0..33 {
    assert_eq!(simd_out[n * 4], (n as u16) + 1, "R at pixel {n}");
    assert_eq!(simd_out[n * 4 + 1], (n as u16) + 100, "G at pixel {n}");
    assert_eq!(simd_out[n * 4 + 2], (n as u16) + 200, "B at pixel {n}");
    assert_eq!(simd_out[n * 4 + 3], (n as u16) + 50, "A at pixel {n}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_to_rgb_u16_lane_order_handcheck() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_rgba64_lane_order(33);
  let mut simd_out = std::vec![0u16; 33 * 3];
  unsafe { avx512_rgba64_to_rgb_u16_row::<false>(&src, &mut simd_out, 33) };
  for n in 0..33 {
    assert_eq!(simd_out[n * 3], (n as u16) + 1, "R at pixel {n}");
    assert_eq!(simd_out[n * 3 + 1], (n as u16) + 100, "G at pixel {n}");
    assert_eq!(simd_out[n * 3 + 2], (n as u16) + 200, "B at pixel {n}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgba_u16_lane_order_handcheck() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_bgra64_lane_order(33);
  let mut simd_out = std::vec![0u16; 33 * 4];
  unsafe { avx512_bgra64_to_rgba_u16_row::<false>(&src, &mut simd_out, 33) };
  // Output is RGBA: R=n+1, G=100+n, B=200+n, A=50+n per pixel n
  // (B↔R swap from source memory order).
  for n in 0..33 {
    assert_eq!(simd_out[n * 4], (n as u16) + 1, "R at pixel {n}");
    assert_eq!(simd_out[n * 4 + 1], (n as u16) + 100, "G at pixel {n}");
    assert_eq!(simd_out[n * 4 + 2], (n as u16) + 200, "B at pixel {n}");
    assert_eq!(simd_out[n * 4 + 3], (n as u16) + 50, "A at pixel {n}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_to_rgb_u16_lane_order_handcheck() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let src = make_bgra64_lane_order(33);
  let mut simd_out = std::vec![0u16; 33 * 3];
  unsafe { avx512_bgra64_to_rgb_u16_row::<false>(&src, &mut simd_out, 33) };
  for n in 0..33 {
    assert_eq!(simd_out[n * 3], (n as u16) + 1, "R at pixel {n}");
    assert_eq!(simd_out[n * 3 + 1], (n as u16) + 100, "G at pixel {n}");
    assert_eq!(simd_out[n * 3 + 2], (n as u16) + 200, "B at pixel {n}");
  }
}

// =============================================================================
// SIMD-level BE-vs-LE parity tests (probes `BE != HOST_NATIVE_BE` gate)
// =============================================================================
//
// Buffers built host-independently via `to_le_bytes` / `to_be_bytes`. Width
// 65 = 2 × 32-lane AVX-512 SIMD body + 1 scalar tail.

fn make_le_be_pair_u16(intended: &[u16]) -> (std::vec::Vec<u16>, std::vec::Vec<u16>) {
  let le_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  let be_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
  let le: std::vec::Vec<u16> = le_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  let be: std::vec::Vec<u16> = be_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  (le, be)
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgb48_be_le_simd_parity_width65() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let intended = make_rgb48_src(65, 0xACE1);
  let (le, be) = make_le_be_pair_u16(&intended);

  let mut out_le = std::vec![0u8; 65 * 3];
  let mut out_be = std::vec![0u8; 65 * 3];
  unsafe {
    avx512_rgb48_to_rgb_row::<false>(&le, &mut out_le, 65);
    avx512_rgb48_to_rgb_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "rgb48→rgb SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u8; 65 * 4];
  let mut out_be = std::vec![0u8; 65 * 4];
  unsafe {
    avx512_rgb48_to_rgba_row::<false>(&le, &mut out_le, 65);
    avx512_rgb48_to_rgba_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "rgb48→rgba SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u16; 65 * 3];
  let mut out_be = std::vec![0u16; 65 * 3];
  unsafe {
    avx512_rgb48_to_rgb_u16_row::<false>(&le, &mut out_le, 65);
    avx512_rgb48_to_rgb_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "rgb48→rgb_u16 SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 65 * 4];
  let mut out_be = std::vec![0u16; 65 * 4];
  unsafe {
    avx512_rgb48_to_rgba_u16_row::<false>(&le, &mut out_le, 65);
    avx512_rgb48_to_rgba_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "rgb48→rgba_u16 SIMD BE/LE parity (endian gate)"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgr48_be_le_simd_parity_width65() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let intended = make_rgb48_src(65, 0xBEEF);
  let (le, be) = make_le_be_pair_u16(&intended);

  let mut out_le = std::vec![0u8; 65 * 3];
  let mut out_be = std::vec![0u8; 65 * 3];
  unsafe {
    avx512_bgr48_to_rgb_row::<false>(&le, &mut out_le, 65);
    avx512_bgr48_to_rgb_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "bgr48→rgb SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u8; 65 * 4];
  let mut out_be = std::vec![0u8; 65 * 4];
  unsafe {
    avx512_bgr48_to_rgba_row::<false>(&le, &mut out_le, 65);
    avx512_bgr48_to_rgba_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "bgr48→rgba SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u16; 65 * 3];
  let mut out_be = std::vec![0u16; 65 * 3];
  unsafe {
    avx512_bgr48_to_rgb_u16_row::<false>(&le, &mut out_le, 65);
    avx512_bgr48_to_rgb_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "bgr48→rgb_u16 SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 65 * 4];
  let mut out_be = std::vec![0u16; 65 * 4];
  unsafe {
    avx512_bgr48_to_rgba_u16_row::<false>(&le, &mut out_le, 65);
    avx512_bgr48_to_rgba_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "bgr48→rgba_u16 SIMD BE/LE parity (endian gate)"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_rgba64_be_le_simd_parity_width65() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let intended = make_rgba64_src(65, 0xCAFE);
  let (le, be) = make_le_be_pair_u16(&intended);

  let mut out_le = std::vec![0u8; 65 * 3];
  let mut out_be = std::vec![0u8; 65 * 3];
  unsafe {
    avx512_rgba64_to_rgb_row::<false>(&le, &mut out_le, 65);
    avx512_rgba64_to_rgb_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "rgba64→rgb SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u8; 65 * 4];
  let mut out_be = std::vec![0u8; 65 * 4];
  unsafe {
    avx512_rgba64_to_rgba_row::<false>(&le, &mut out_le, 65);
    avx512_rgba64_to_rgba_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "rgba64→rgba SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 65 * 3];
  let mut out_be = std::vec![0u16; 65 * 3];
  unsafe {
    avx512_rgba64_to_rgb_u16_row::<false>(&le, &mut out_le, 65);
    avx512_rgba64_to_rgb_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "rgba64→rgb_u16 SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 65 * 4];
  let mut out_be = std::vec![0u16; 65 * 4];
  unsafe {
    avx512_rgba64_to_rgba_u16_row::<false>(&le, &mut out_le, 65);
    avx512_rgba64_to_rgba_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "rgba64→rgba_u16 SIMD BE/LE parity (endian gate)"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_bgra64_be_le_simd_parity_width65() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let intended = make_rgba64_src(65, 0xF00D);
  let (le, be) = make_le_be_pair_u16(&intended);

  let mut out_le = std::vec![0u8; 65 * 3];
  let mut out_be = std::vec![0u8; 65 * 3];
  unsafe {
    avx512_bgra64_to_rgb_row::<false>(&le, &mut out_le, 65);
    avx512_bgra64_to_rgb_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "bgra64→rgb SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u8; 65 * 4];
  let mut out_be = std::vec![0u8; 65 * 4];
  unsafe {
    avx512_bgra64_to_rgba_row::<false>(&le, &mut out_le, 65);
    avx512_bgra64_to_rgba_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "bgra64→rgba SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 65 * 3];
  let mut out_be = std::vec![0u16; 65 * 3];
  unsafe {
    avx512_bgra64_to_rgb_u16_row::<false>(&le, &mut out_le, 65);
    avx512_bgra64_to_rgb_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "bgra64→rgb_u16 SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 65 * 4];
  let mut out_be = std::vec![0u16; 65 * 4];
  unsafe {
    avx512_bgra64_to_rgba_u16_row::<false>(&le, &mut out_le, 65);
    avx512_bgra64_to_rgba_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(
    out_le, out_be,
    "bgra64→rgba_u16 SIMD BE/LE parity (endian gate)"
  );
}

// =============================================================================
// X2RGB10 / X2BGR10 SIMD-level BE-vs-LE parity tests
// =============================================================================
//
// Co-located here (rather than in the dead-code `tests/packed_rgb.rs` which
// is not declared in `tests/mod.rs`) so they are actually compiled and run.

fn pseudo_random_x2_intended(width: usize, seed: u32) -> std::vec::Vec<u32> {
  let mut state = seed;
  (0..width)
    .map(|_| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      state
    })
    .collect()
}

fn make_le_be_pair_x2(intended: &[u32]) -> (std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let le_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  let be_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
  (le_bytes, be_bytes)
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_x2rgb10_be_le_simd_parity_width65() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let intended = pseudo_random_x2_intended(65, 0xC0DE_BEEF);
  let (le, be) = make_le_be_pair_x2(&intended);

  let mut out_le = std::vec![0u8; 65 * 3];
  let mut out_be = std::vec![0u8; 65 * 3];
  unsafe {
    x2rgb10_to_rgb_row::<false>(&le, &mut out_le, 65);
    x2rgb10_to_rgb_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "x2rgb10→rgb SIMD BE/LE parity");

  let mut out_le = std::vec![0u8; 65 * 4];
  let mut out_be = std::vec![0u8; 65 * 4];
  unsafe {
    x2rgb10_to_rgba_row::<false>(&le, &mut out_le, 65);
    x2rgb10_to_rgba_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "x2rgb10→rgba SIMD BE/LE parity");

  let mut out_le = std::vec![0u16; 65 * 3];
  let mut out_be = std::vec![0u16; 65 * 3];
  unsafe {
    x2rgb10_to_rgb_u16_row::<false>(&le, &mut out_le, 65);
    x2rgb10_to_rgb_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "x2rgb10→rgb_u16 SIMD BE/LE parity");
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn avx512_x2bgr10_be_le_simd_parity_width65() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let intended = pseudo_random_x2_intended(65, 0xFEED_FACE);
  let (le, be) = make_le_be_pair_x2(&intended);

  let mut out_le = std::vec![0u8; 65 * 3];
  let mut out_be = std::vec![0u8; 65 * 3];
  unsafe {
    x2bgr10_to_rgb_row::<false>(&le, &mut out_le, 65);
    x2bgr10_to_rgb_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "x2bgr10→rgb SIMD BE/LE parity");

  let mut out_le = std::vec![0u8; 65 * 4];
  let mut out_be = std::vec![0u8; 65 * 4];
  unsafe {
    x2bgr10_to_rgba_row::<false>(&le, &mut out_le, 65);
    x2bgr10_to_rgba_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "x2bgr10→rgba SIMD BE/LE parity");

  let mut out_le = std::vec![0u16; 65 * 3];
  let mut out_be = std::vec![0u16; 65 * 3];
  unsafe {
    x2bgr10_to_rgb_u16_row::<false>(&le, &mut out_le, 65);
    x2bgr10_to_rgb_u16_row::<true>(&be, &mut out_be, 65);
  }
  assert_eq!(out_le, out_be, "x2bgr10→rgb_u16 SIMD BE/LE parity");
}
