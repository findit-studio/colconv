//! SSE4.1 parity tests for the packed 16-bit RGB/BGR/RGBA/BGRA kernels (Tier 8 finish).
//!
//! Each test early-returns when SSE4.1 is not detected at runtime, satisfying the
//! x86 SIMD test guard requirement so CI sanitizer / Miri runs do not fail.
//!
//! Width = 17 exercises: 2 SIMD iterations (8 px each) + 1-pixel scalar tail.
//! Width = 8 exercises: exact hot-loop, no tail.
//! Width = 1 exercises: tail-only (SIMD loop skipped entirely).

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

// =============================================================================
// Rgb48 → u8 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgb48_to_rgb_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(17, 0x0101);
  let mut simd_out = std::vec![0u8; 17 * 3];
  let mut scalar_out = std::vec![0u8; 17 * 3];
  unsafe { sse41_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb width=17: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgb48_to_rgb_exact8_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(8, 0xF0F0);
  let mut simd_out = std::vec![0u8; 8 * 3];
  let mut scalar_out = std::vec![0u8; 8 * 3];
  unsafe { sse41_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 8) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 8);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb exact-8: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgb48_to_rgb_width1_tail_only() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = [0x1234u16, 0x5678, 0x9ABC];
  let mut simd_out = [0u8; 3];
  let mut scalar_out = [0u8; 3];
  unsafe { sse41_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 1) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 1);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb width=1: tail-only mismatch"
  );
}

// =============================================================================
// Rgb48 → u8 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgb48_to_rgba_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(17, 0x0303);
  let mut simd_out = std::vec![0u8; 17 * 4];
  let mut scalar_out = std::vec![0u8; 17 * 4];
  unsafe { sse41_rgb48_to_rgba_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgb48_to_rgba_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgba width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgb48 → u16 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgb48_to_rgb_u16_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(17, 0x0505);
  let mut simd_out = std::vec![0u16; 17 * 3];
  let mut scalar_out = std::vec![0u16; 17 * 3];
  unsafe { sse41_rgb48_to_rgb_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgb48_to_rgb_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb_u16 width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgb48 → u16 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgb48_to_rgba_u16_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(17, 0x0707);
  let mut simd_out = std::vec![0u16; 17 * 4];
  let mut scalar_out = std::vec![0u16; 17 * 4];
  unsafe { sse41_rgb48_to_rgba_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgb48_to_rgba_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgba_u16 width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgr48 → u8 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgr48_to_rgb_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(17, 0x1111);
  let mut simd_out = std::vec![0u8; 17 * 3];
  let mut scalar_out = std::vec![0u8; 17 * 3];
  unsafe { sse41_bgr48_to_rgb_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgr48_to_rgb_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgb width=17: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgr48_to_rgb_exact8_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(8, 0xA1A1);
  let mut simd_out = std::vec![0u8; 8 * 3];
  let mut scalar_out = std::vec![0u8; 8 * 3];
  unsafe { sse41_bgr48_to_rgb_row::<false>(&src, &mut simd_out, 8) };
  scalar::bgr48_to_rgb_row::<false>(&src, &mut scalar_out, 8);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgb exact-8: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgr48 → u8 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgr48_to_rgba_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(17, 0x2222);
  let mut simd_out = std::vec![0u8; 17 * 4];
  let mut scalar_out = std::vec![0u8; 17 * 4];
  unsafe { sse41_bgr48_to_rgba_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgr48_to_rgba_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgba width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgr48 → u16 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgr48_to_rgb_u16_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(17, 0x3333);
  let mut simd_out = std::vec![0u16; 17 * 3];
  let mut scalar_out = std::vec![0u16; 17 * 3];
  unsafe { sse41_bgr48_to_rgb_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgr48_to_rgb_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgb_u16 width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgr48 → u16 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgr48_to_rgba_u16_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgb48_src(17, 0x4444);
  let mut simd_out = std::vec![0u16; 17 * 4];
  let mut scalar_out = std::vec![0u16; 17 * 4];
  unsafe { sse41_bgr48_to_rgba_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgr48_to_rgba_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgba_u16 width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgba64 → u8 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgba64_to_rgb_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgba64_src(17, 0xAAAA);
  let mut simd_out = std::vec![0u8; 17 * 3];
  let mut scalar_out = std::vec![0u8; 17 * 3];
  unsafe { sse41_rgba64_to_rgb_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgba64_to_rgb_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb width=17: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgba64_to_rgb_exact8_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgba64_src(8, 0x0F0F);
  let mut simd_out = std::vec![0u8; 8 * 3];
  let mut scalar_out = std::vec![0u8; 8 * 3];
  unsafe { sse41_rgba64_to_rgb_row::<false>(&src, &mut simd_out, 8) };
  scalar::rgba64_to_rgb_row::<false>(&src, &mut scalar_out, 8);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb exact-8: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgba64 → u8 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgba64_to_rgba_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgba64_src(17, 0xBBBB);
  let mut simd_out = std::vec![0u8; 17 * 4];
  let mut scalar_out = std::vec![0u8; 17 * 4];
  unsafe { sse41_rgba64_to_rgba_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgba64_to_rgba_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgba64 → u16 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgba64_to_rgb_u16_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgba64_src(17, 0xCCCC);
  let mut simd_out = std::vec![0u16; 17 * 3];
  let mut scalar_out = std::vec![0u16; 17 * 3];
  unsafe { sse41_rgba64_to_rgb_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgba64_to_rgb_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb_u16 width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Rgba64 → u16 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgba64_to_rgba_u16_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgba64_src(17, 0xDDDD);
  let mut simd_out = std::vec![0u16; 17 * 4];
  let mut scalar_out = std::vec![0u16; 17 * 4];
  unsafe { sse41_rgba64_to_rgba_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgba64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba_u16 width=17: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_rgba64_to_rgba_u16_width1_tail_only() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = [0x1234u16, 0x5678, 0x9ABC, 0xDEF0]; // R, G, B, A
  let mut simd_out = [0u16; 4];
  let mut scalar_out = [0u16; 4];
  unsafe { sse41_rgba64_to_rgba_u16_row::<false>(&src, &mut simd_out, 1) };
  scalar::rgba64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 1);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba_u16 width=1: tail-only mismatch"
  );
}

// =============================================================================
// Bgra64 → u8 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgra64_to_rgb_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgba64_src(17, 0x1234);
  let mut simd_out = std::vec![0u8; 17 * 3];
  let mut scalar_out = std::vec![0u8; 17 * 3];
  unsafe { sse41_bgra64_to_rgb_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgra64_to_rgb_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgb width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgra64 → u8 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgra64_to_rgba_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgba64_src(17, 0x5678);
  let mut simd_out = std::vec![0u8; 17 * 4];
  let mut scalar_out = std::vec![0u8; 17 * 4];
  unsafe { sse41_bgra64_to_rgba_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgra64_to_rgba_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgra64 → u16 RGB
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgra64_to_rgb_u16_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgba64_src(17, 0x9ABC);
  let mut simd_out = std::vec![0u16; 17 * 3];
  let mut scalar_out = std::vec![0u16; 17 * 3];
  unsafe { sse41_bgra64_to_rgb_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgra64_to_rgb_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgb_u16 width=17: SIMD vs scalar mismatch"
  );
}

// =============================================================================
// Bgra64 → u16 RGBA
// =============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgra64_to_rgba_u16_matches_scalar_width17() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = make_rgba64_src(17, 0xDEF0);
  let mut simd_out = std::vec![0u16; 17 * 4];
  let mut scalar_out = std::vec![0u16; 17 * 4];
  unsafe { sse41_bgra64_to_rgba_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgra64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba_u16 width=17: SIMD vs scalar mismatch"
  );
}

#[test]
#[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
fn sse41_bgra64_to_rgba_u16_width1_tail_only() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let src = [0x1111u16, 0x2222, 0x3333, 0x4444]; // B, G, R, A
  let mut simd_out = [0u16; 4];
  let mut scalar_out = [0u16; 4];
  unsafe { sse41_bgra64_to_rgba_u16_row::<false>(&src, &mut simd_out, 1) };
  scalar::bgra64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 1);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba_u16 width=1: tail-only mismatch"
  );
}
