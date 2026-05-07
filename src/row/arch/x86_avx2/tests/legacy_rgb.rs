//! AVX2 parity tests for legacy 16-bit packed-RGB kernels (Tier 7).
//!
//! Every test early-returns if AVX2 is not detected at runtime (sanitizer/Miri safety).
//! All tests carry `#[cfg_attr(miri, ignore = "...")]`.
//!
//! Widths exercised: [1, 7, 8, 15, 16, 17, 32, 33, 64, 65] — covers all boundary
//! cases around the 16-pixel loop stride and sub-stride remainders.
//!
//! Asymmetric lane-order regression tests use pixel values that set only one
//! channel at a time (R-only, G-only, B-only) to catch per-channel mask bugs
//! that symmetric all-channels patterns would miss (Tier 10b precedent).

use super::super::*;

// ---- Shared pseudo-random helper -------------------------------------------

fn legacy_rgb_plane(width: usize, seed: u32) -> std::vec::Vec<u8> {
  let mut state = seed;
  let mut out = std::vec::Vec::with_capacity(width * 2);
  for _ in 0..width {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let px = (state as u16).to_le_bytes();
    out.extend_from_slice(&px);
  }
  out
}

// ============================================================================
// RGB565 — parity tests (scalar vs AVX2)
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb565_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx2 = std::vec![0u8; w * 3];
    scalar::legacy_rgb::rgb565_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      rgb565_to_rgb_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb565_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb565_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx2 = std::vec![0u8; w * 4];
    scalar::legacy_rgb::rgb565_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      rgb565_to_rgba_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb565_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb565_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx2 = std::vec![0u16; w * 3];
    scalar::legacy_rgb::rgb565_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb565_to_rgb_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb565_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb565_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx2 = std::vec![0u16; w * 4];
    scalar::legacy_rgb::rgb565_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb565_to_rgba_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb565_to_rgba_u16 diverges (width={w})"
    );
  }
}

// ---- RGB565 lane-order regression (asymmetric R/G/B inputs) ----------------

/// Pixel 0xF800 = 0b1111_1000_0000_0000: R5=31, G6=0, B5=0 → R=255, G=0, B=0.
/// Checks that R channel mask does not bleed into G or B lanes.
#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb565_lane_order_r_only() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  // Build 17 identical pixels (crosses the 16-pixel boundary)
  let w = 17usize;
  let px: u16 = 0xF800; // R5=31, G6=0, B5=0
  let src: std::vec::Vec<u8> = std::iter::repeat(px.to_le_bytes())
    .take(w)
    .flatten()
    .collect();
  let mut out_scalar = std::vec![0u8; w * 3];
  let mut out_avx2 = std::vec![0u8; w * 3];
  scalar::legacy_rgb::rgb565_to_rgb_row(&src, &mut out_scalar, w);
  unsafe {
    rgb565_to_rgb_row(&src, &mut out_avx2, w);
  }
  assert_eq!(
    out_scalar, out_avx2,
    "AVX2 rgb565 R-only lane order mismatch"
  );
  // Verify expected values: R=255, G=0, B=0 for every pixel
  let expected: std::vec::Vec<u8> = std::iter::repeat([255u8, 0u8, 0u8])
    .take(w)
    .flatten()
    .collect();
  assert_eq!(out_avx2, expected, "AVX2 rgb565 R-only values wrong");
}

/// Pixel 0x07E0 = 0b0000_0111_1110_0000: R5=0, G6=63, B5=0 → R=0, G=255, B=0.
/// Checks that G channel mask does not bleed into R or B lanes.
#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb565_lane_order_g_only() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let w = 17usize;
  let px: u16 = 0x07E0; // R5=0, G6=63, B5=0
  let src: std::vec::Vec<u8> = std::iter::repeat(px.to_le_bytes())
    .take(w)
    .flatten()
    .collect();
  let mut out_scalar = std::vec![0u8; w * 3];
  let mut out_avx2 = std::vec![0u8; w * 3];
  scalar::legacy_rgb::rgb565_to_rgb_row(&src, &mut out_scalar, w);
  unsafe {
    rgb565_to_rgb_row(&src, &mut out_avx2, w);
  }
  assert_eq!(
    out_scalar, out_avx2,
    "AVX2 rgb565 G-only lane order mismatch"
  );
  let expected: std::vec::Vec<u8> = std::iter::repeat([0u8, 255u8, 0u8])
    .take(w)
    .flatten()
    .collect();
  assert_eq!(out_avx2, expected, "AVX2 rgb565 G-only values wrong");
}

/// Pixel 0x001F = 0b0000_0000_0001_1111: R5=0, G6=0, B5=31 → R=0, G=0, B=255.
/// Checks that B channel mask does not bleed into R or G lanes.
#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb565_lane_order_b_only() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let w = 17usize;
  let px: u16 = 0x001F; // R5=0, G6=0, B5=31
  let src: std::vec::Vec<u8> = std::iter::repeat(px.to_le_bytes())
    .take(w)
    .flatten()
    .collect();
  let mut out_scalar = std::vec![0u8; w * 3];
  let mut out_avx2 = std::vec![0u8; w * 3];
  scalar::legacy_rgb::rgb565_to_rgb_row(&src, &mut out_scalar, w);
  unsafe {
    rgb565_to_rgb_row(&src, &mut out_avx2, w);
  }
  assert_eq!(
    out_scalar, out_avx2,
    "AVX2 rgb565 B-only lane order mismatch"
  );
  let expected: std::vec::Vec<u8> = std::iter::repeat([0u8, 0u8, 255u8])
    .take(w)
    .flatten()
    .collect();
  assert_eq!(out_avx2, expected, "AVX2 rgb565 B-only values wrong");
}

// ============================================================================
// BGR565 — parity tests
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr565_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx2 = std::vec![0u8; w * 3];
    scalar::legacy_rgb::bgr565_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      bgr565_to_rgb_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr565_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr565_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx2 = std::vec![0u8; w * 4];
    scalar::legacy_rgb::bgr565_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      bgr565_to_rgba_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr565_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr565_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx2 = std::vec![0u16; w * 3];
    scalar::legacy_rgb::bgr565_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr565_to_rgb_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr565_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr565_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx2 = std::vec![0u16; w * 4];
    scalar::legacy_rgb::bgr565_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr565_to_rgba_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr565_to_rgba_u16 diverges (width={w})"
    );
  }
}

/// BGR565 lane-order: R5 is at bits [4:0]. Pixel 0x001F → output[0] must be 255 (R).
#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr565_lane_order_r_in_low_bits() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let w = 17usize;
  // BGR565: R5 at [4:0]=31, G6=0, B5=0
  let px: u16 = 0x001F;
  let src: std::vec::Vec<u8> = std::iter::repeat(px.to_le_bytes())
    .take(w)
    .flatten()
    .collect();
  let mut out_avx2 = std::vec![0u8; w * 3];
  unsafe {
    bgr565_to_rgb_row(&src, &mut out_avx2, w);
  }
  // Output is always R,G,B order: R=255, G=0, B=0
  let expected: std::vec::Vec<u8> = std::iter::repeat([255u8, 0u8, 0u8])
    .take(w)
    .flatten()
    .collect();
  assert_eq!(
    out_avx2, expected,
    "AVX2 bgr565 R-in-low-bits lane order wrong"
  );
}

// ============================================================================
// RGB555 — parity tests
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb555_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx2 = std::vec![0u8; w * 3];
    scalar::legacy_rgb::rgb555_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      rgb555_to_rgb_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb555_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb555_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx2 = std::vec![0u8; w * 4];
    scalar::legacy_rgb::rgb555_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      rgb555_to_rgba_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb555_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb555_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx2 = std::vec![0u16; w * 3];
    scalar::legacy_rgb::rgb555_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb555_to_rgb_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb555_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb555_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx2 = std::vec![0u16; w * 4];
    scalar::legacy_rgb::rgb555_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb555_to_rgba_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb555_to_rgba_u16 diverges (width={w})"
    );
  }
}

// ---- RGB555 lane-order regression ------------------------------------------

/// RGB555: R5 at bits [14:10]. Pixel 0x7C00 → R=31 → R_exp=255, G=0, B=0.
#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb555_lane_order_r_only() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let w = 17usize;
  let px: u16 = 0x7C00; // R5=31, G5=0, B5=0
  let src: std::vec::Vec<u8> = std::iter::repeat(px.to_le_bytes())
    .take(w)
    .flatten()
    .collect();
  let mut out_avx2 = std::vec![0u8; w * 3];
  unsafe {
    rgb555_to_rgb_row(&src, &mut out_avx2, w);
  }
  let expected: std::vec::Vec<u8> = std::iter::repeat([255u8, 0u8, 0u8])
    .take(w)
    .flatten()
    .collect();
  assert_eq!(out_avx2, expected, "AVX2 rgb555 R-only values wrong");
}

/// RGB555: B5 at bits [4:0]. Pixel 0x001F → R=0, G=0, B=255.
#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb555_lane_order_b_only() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let w = 17usize;
  let px: u16 = 0x001F; // R5=0, G5=0, B5=31
  let src: std::vec::Vec<u8> = std::iter::repeat(px.to_le_bytes())
    .take(w)
    .flatten()
    .collect();
  let mut out_avx2 = std::vec![0u8; w * 3];
  unsafe {
    rgb555_to_rgb_row(&src, &mut out_avx2, w);
  }
  let expected: std::vec::Vec<u8> = std::iter::repeat([0u8, 0u8, 255u8])
    .take(w)
    .flatten()
    .collect();
  assert_eq!(out_avx2, expected, "AVX2 rgb555 B-only values wrong");
}

// ============================================================================
// BGR555 — parity tests
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr555_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx2 = std::vec![0u8; w * 3];
    scalar::legacy_rgb::bgr555_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      bgr555_to_rgb_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr555_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr555_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx2 = std::vec![0u8; w * 4];
    scalar::legacy_rgb::bgr555_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      bgr555_to_rgba_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr555_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr555_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx2 = std::vec![0u16; w * 3];
    scalar::legacy_rgb::bgr555_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr555_to_rgb_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr555_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr555_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx2 = std::vec![0u16; w * 4];
    scalar::legacy_rgb::bgr555_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr555_to_rgba_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr555_to_rgba_u16 diverges (width={w})"
    );
  }
}

// ============================================================================
// RGB444 — parity tests
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb444_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx2 = std::vec![0u8; w * 3];
    scalar::legacy_rgb::rgb444_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      rgb444_to_rgb_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb444_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb444_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx2 = std::vec![0u8; w * 4];
    scalar::legacy_rgb::rgb444_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      rgb444_to_rgba_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb444_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb444_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx2 = std::vec![0u16; w * 3];
    scalar::legacy_rgb::rgb444_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb444_to_rgb_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb444_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb444_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx2 = std::vec![0u16; w * 4];
    scalar::legacy_rgb::rgb444_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb444_to_rgba_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 rgb444_to_rgba_u16 diverges (width={w})"
    );
  }
}

// ---- RGB444 lane-order regression ------------------------------------------

/// RGB444: R4 at bits [11:8]. Pixel 0x0F00 → R4=15 → R_exp=255, G=0, B=0.
#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb444_lane_order_r_only() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let w = 17usize;
  let px: u16 = 0x0F00; // R4=15, G4=0, B4=0
  let src: std::vec::Vec<u8> = std::iter::repeat(px.to_le_bytes())
    .take(w)
    .flatten()
    .collect();
  let mut out_avx2 = std::vec![0u8; w * 3];
  unsafe {
    rgb444_to_rgb_row(&src, &mut out_avx2, w);
  }
  let expected: std::vec::Vec<u8> = std::iter::repeat([255u8, 0u8, 0u8])
    .take(w)
    .flatten()
    .collect();
  assert_eq!(out_avx2, expected, "AVX2 rgb444 R-only values wrong");
}

/// RGB444: B4 at bits [3:0]. Pixel 0x000F → R=0, G=0, B=255.
#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_rgb444_lane_order_b_only() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let w = 17usize;
  let px: u16 = 0x000F; // R4=0, G4=0, B4=15
  let src: std::vec::Vec<u8> = std::iter::repeat(px.to_le_bytes())
    .take(w)
    .flatten()
    .collect();
  let mut out_avx2 = std::vec![0u8; w * 3];
  unsafe {
    rgb444_to_rgb_row(&src, &mut out_avx2, w);
  }
  let expected: std::vec::Vec<u8> = std::iter::repeat([0u8, 0u8, 255u8])
    .take(w)
    .flatten()
    .collect();
  assert_eq!(out_avx2, expected, "AVX2 rgb444 B-only values wrong");
}

// ============================================================================
// BGR444 — parity tests
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr444_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx2 = std::vec![0u8; w * 3];
    scalar::legacy_rgb::bgr444_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      bgr444_to_rgb_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr444_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr444_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx2 = std::vec![0u8; w * 4];
    scalar::legacy_rgb::bgr444_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      bgr444_to_rgba_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr444_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr444_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_avx2 = std::vec![0u16; w * 3];
    scalar::legacy_rgb::bgr444_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr444_to_rgb_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr444_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr444_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_avx2 = std::vec![0u16; w * 4];
    scalar::legacy_rgb::bgr444_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr444_to_rgba_u16_row(&src, &mut out_avx2, w);
    }
    assert_eq!(
      out_scalar, out_avx2,
      "AVX2 bgr444_to_rgba_u16 diverges (width={w})"
    );
  }
}

/// BGR444: R4 at bits [3:0]. Pixel 0x000F → output[0]=255 (R-first output).
#[test]
#[cfg_attr(miri, ignore = "x86 AVX2 SIMD intrinsics unsupported by Miri")]
fn avx2_bgr444_lane_order_r_in_low_bits() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let w = 17usize;
  // BGR444: R4=15, G4=0, B4=0
  let px: u16 = 0x000F;
  let src: std::vec::Vec<u8> = std::iter::repeat(px.to_le_bytes())
    .take(w)
    .flatten()
    .collect();
  let mut out_avx2 = std::vec![0u8; w * 3];
  unsafe {
    bgr444_to_rgb_row(&src, &mut out_avx2, w);
  }
  // Output order is always R,G,B
  let expected: std::vec::Vec<u8> = std::iter::repeat([255u8, 0u8, 0u8])
    .take(w)
    .flatten()
    .collect();
  assert_eq!(
    out_avx2, expected,
    "AVX2 bgr444 R-in-low-bits lane order wrong"
  );
}
