//! NEON kernels for legacy 16-bit packed-RGB source formats (Tier 7).
//!
//! Six source formats × 4 output variants = 24 kernels. Each format word is a
//! little-endian `u16` at 8 pixels per iteration (`vld1q_u16` = 8 × u16).
//!
//! # Bit extraction
//!
//! - **RGB565**: `vshrq_n_u16(px, 11)` + `& 0x1F` → R5;
//!   `vshrq_n_u16(px, 5)` + `& 0x3F` → G6; `px & 0x1F` → B5.
//! - **BGR565**: same shifts, but R↔B swapped in extraction (R5 at bits [4:0],
//!   B5 at bits [15:11]).
//! - **RGB555**: `vshrq_n_u16(px, 10)` + `& 0x1F` → R5; `vshrq_n_u16(px, 5)`
//!   + `& 0x1F` → G5; `px & 0x1F` → B5.
//! - **BGR555**: same as RGB555 with R↔B swapped.
//! - **RGB444**: `vshrq_n_u16(px, 8)` + `& 0x0F` → R4; `vshrq_n_u16(px, 4)`
//!   + `& 0x0F` → G4; `px & 0x0F` → B4.
//! - **BGR444**: same as RGB444 with R↔B swapped.
//!
//! # Channel expansion
//!
//! | Bits | NEON                                               |
//! |------|----------------------------------------------------|
//! | 5    | `(c << 3) \| (c >> 2)` → `vshlq_n_u16` + `vshrq_n_u16` + `vorrq_u16` |
//! | 6    | `(c << 2) \| (c >> 4)` → same                     |
//! | 4    | `(c << 4) \| c`        → same                     |
//!
//! # u8 output
//!
//! After expansion each `uint16x8_t` lane holds a value in `[0, 255]`.
//! `vqmovn_u16` narrows to `uint8x8_t`; `vst3_u8` / `vst4_u8` interleave.
//!
//! # u16 output
//!
//! Skip expansion — store the raw extracted sub-fields. `vst3q_u16` /
//! `vst4q_u16` interleave 8 pixels of 3 or 4 channels.
//!
//! # Scalar tail
//!
//! When `width % 8 ≠ 0` the remainder is handled by `scalar::legacy_rgb`.

use core::arch::aarch64::*;

use crate::row::scalar;

// ============================================================================
// Internal helpers
// ============================================================================

/// Expand a vector of 5-bit values in [0,31] to 8-bit: `(c << 3) | (c >> 2)`.
#[inline(always)]
unsafe fn expand5(c: uint16x8_t) -> uint16x8_t {
  unsafe { vorrq_u16(vshlq_n_u16::<3>(c), vshrq_n_u16::<2>(c)) }
}

/// Expand a vector of 6-bit values in [0,63] to 8-bit: `(c << 2) | (c >> 4)`.
#[inline(always)]
unsafe fn expand6(c: uint16x8_t) -> uint16x8_t {
  unsafe { vorrq_u16(vshlq_n_u16::<2>(c), vshrq_n_u16::<4>(c)) }
}

/// Expand a vector of 4-bit values in [0,15] to 8-bit: `(c << 4) | c`.
#[inline(always)]
unsafe fn expand4(c: uint16x8_t) -> uint16x8_t {
  unsafe { vorrq_u16(vshlq_n_u16::<4>(c), c) }
}

// ============================================================================
// RGB565 — R5 G6 B5, bits [15:11]=R, [10:5]=G, [4:0]=B
// ============================================================================

/// NEON RGB565 → packed `R, G, B` bytes (8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mask6 = vdupq_n_u16(0x3F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r5 = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g6 = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let b5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16(expand5(r5));
      let g_u8 = vqmovn_u16(expand6(g6));
      let b_u8 = vqmovn_u16(expand5(b5));
      vst3_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x8x3_t(r_u8, g_u8, b_u8),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON RGB565 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mask6 = vdupq_n_u16(0x3F);
    let alpha = vdup_n_u8(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r5 = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g6 = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let b5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16(expand5(r5));
      let g_u8 = vqmovn_u16(expand6(g6));
      let b_u8 = vqmovn_u16(expand5(b5));
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r_u8, g_u8, b_u8, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON RGB565 → packed `R, G, B` **u16** (native bit-width, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mask6 = vdupq_n_u16(0x3F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let b = vandq_u16(px, mask5);
      vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), uint16x8x3_t(r, g, b));
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// NEON RGB565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mask6 = vdupq_n_u16(0x3F);
    let alpha = vdupq_n_u16(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let b = vandq_u16(px, mask5);
      vst4q_u16(
        rgba_u16_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(r, g, b, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// ============================================================================
// BGR565 — B5 G6 R5, bits [15:11]=B, [10:5]=G, [4:0]=R
// ============================================================================

/// NEON BGR565 → packed `R, G, B` bytes (output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mask6 = vdupq_n_u16(0x3F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      // BGR565: B at [15:11], G at [10:5], R at [4:0]
      let b5 = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g6 = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let r5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16(expand5(r5));
      let g_u8 = vqmovn_u16(expand6(g6));
      let b_u8 = vqmovn_u16(expand5(b5));
      vst3_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x8x3_t(r_u8, g_u8, b_u8),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON BGR565 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mask6 = vdupq_n_u16(0x3F);
    let alpha = vdup_n_u8(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let b5 = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g6 = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let r5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16(expand5(r5));
      let g_u8 = vqmovn_u16(expand6(g6));
      let b_u8 = vqmovn_u16(expand5(b5));
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r_u8, g_u8, b_u8, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON BGR565 → packed `R, G, B` **u16** (native bit-width, output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mask6 = vdupq_n_u16(0x3F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let b = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let r = vandq_u16(px, mask5);
      // Output order: R, G, B
      vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), uint16x8x3_t(r, g, b));
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// NEON BGR565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mask6 = vdupq_n_u16(0x3F);
    let alpha = vdupq_n_u16(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let b = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let r = vandq_u16(px, mask5);
      vst4q_u16(
        rgba_u16_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(r, g, b, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// ============================================================================
// RGB555 — 1X R5 G5 B5, bits [14:10]=R, [9:5]=G, [4:0]=B, bit 15 ignored
// ============================================================================

/// NEON RGB555 → packed `R, G, B` bytes (8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r5 = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g5 = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let b5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16(expand5(r5));
      let g_u8 = vqmovn_u16(expand5(g5));
      let b_u8 = vqmovn_u16(expand5(b5));
      vst3_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x8x3_t(r_u8, g_u8, b_u8),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON RGB555 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let alpha = vdup_n_u8(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r5 = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g5 = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let b5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16(expand5(r5));
      let g_u8 = vqmovn_u16(expand5(g5));
      let b_u8 = vqmovn_u16(expand5(b5));
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r_u8, g_u8, b_u8, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON RGB555 → packed `R, G, B` **u16** (native bit-width, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let b = vandq_u16(px, mask5);
      vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), uint16x8x3_t(r, g, b));
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// NEON RGB555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let alpha = vdupq_n_u16(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let b = vandq_u16(px, mask5);
      vst4q_u16(
        rgba_u16_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(r, g, b, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// ============================================================================
// BGR555 — 1X B5 G5 R5, bits [14:10]=B, [9:5]=G, [4:0]=R, bit 15 ignored
// ============================================================================

/// NEON BGR555 → packed `R, G, B` bytes (output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      // BGR555: B at [14:10], G at [9:5], R at [4:0]
      let b5 = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g5 = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let r5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16(expand5(r5));
      let g_u8 = vqmovn_u16(expand5(g5));
      let b_u8 = vqmovn_u16(expand5(b5));
      vst3_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x8x3_t(r_u8, g_u8, b_u8),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON BGR555 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let alpha = vdup_n_u8(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let b5 = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g5 = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let r5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16(expand5(r5));
      let g_u8 = vqmovn_u16(expand5(g5));
      let b_u8 = vqmovn_u16(expand5(b5));
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r_u8, g_u8, b_u8, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON BGR555 → packed `R, G, B` **u16** (native bit-width, output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let b = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let r = vandq_u16(px, mask5);
      // Output order: R, G, B
      vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), uint16x8x3_t(r, g, b));
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// NEON BGR555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = vdupq_n_u16(0x1F);
    let alpha = vdupq_n_u16(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let b = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let r = vandq_u16(px, mask5);
      vst4q_u16(
        rgba_u16_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(r, g, b, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// ============================================================================
// RGB444 — 4X R4 G4 B4, bits [11:8]=R, [7:4]=G, [3:0]=B, bits [15:12] ignored
// ============================================================================

/// NEON RGB444 → packed `R, G, B` bytes (8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = vdupq_n_u16(0x0F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r4 = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g4 = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let b4 = vandq_u16(px, mask4);
      let r_u8 = vqmovn_u16(expand4(r4));
      let g_u8 = vqmovn_u16(expand4(g4));
      let b_u8 = vqmovn_u16(expand4(b4));
      vst3_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x8x3_t(r_u8, g_u8, b_u8),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON RGB444 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = vdupq_n_u16(0x0F);
    let alpha = vdup_n_u8(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r4 = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g4 = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let b4 = vandq_u16(px, mask4);
      let r_u8 = vqmovn_u16(expand4(r4));
      let g_u8 = vqmovn_u16(expand4(g4));
      let b_u8 = vqmovn_u16(expand4(b4));
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r_u8, g_u8, b_u8, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON RGB444 → packed `R, G, B` **u16** (native 4-bit width, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = vdupq_n_u16(0x0F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let b = vandq_u16(px, mask4);
      vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), uint16x8x3_t(r, g, b));
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// NEON RGB444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgb444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = vdupq_n_u16(0x0F);
    let alpha = vdupq_n_u16(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let r = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let b = vandq_u16(px, mask4);
      vst4q_u16(
        rgba_u16_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(r, g, b, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// ============================================================================
// BGR444 — 4X B4 G4 R4, bits [11:8]=B, [7:4]=G, [3:0]=R, bits [15:12] ignored
// ============================================================================

/// NEON BGR444 → packed `R, G, B` bytes (output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = vdupq_n_u16(0x0F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      // BGR444: B at [11:8], G at [7:4], R at [3:0]
      let b4 = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g4 = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let r4 = vandq_u16(px, mask4);
      let r_u8 = vqmovn_u16(expand4(r4));
      let g_u8 = vqmovn_u16(expand4(g4));
      let b_u8 = vqmovn_u16(expand4(b4));
      vst3_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x8x3_t(r_u8, g_u8, b_u8),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON BGR444 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = vdupq_n_u16(0x0F);
    let alpha = vdup_n_u8(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let b4 = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g4 = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let r4 = vandq_u16(px, mask4);
      let r_u8 = vqmovn_u16(expand4(r4));
      let g_u8 = vqmovn_u16(expand4(g4));
      let b_u8 = vqmovn_u16(expand4(b4));
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r_u8, g_u8, b_u8, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON BGR444 → packed `R, G, B` **u16** (native 4-bit width, output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = vdupq_n_u16(0x0F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let b = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let r = vandq_u16(px, mask4);
      // Output order: R, G, B
      vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), uint16x8x3_t(r, g, b));
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// NEON BGR444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bgr444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = vdupq_n_u16(0x0F);
    let alpha = vdupq_n_u16(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = vld1q_u16(src.as_ptr().add(x * 2).cast());
      let b = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let r = vandq_u16(px, mask4);
      vst4q_u16(
        rgba_u16_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(r, g, b, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}
