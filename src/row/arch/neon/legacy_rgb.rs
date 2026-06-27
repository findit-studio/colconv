//! NEON kernels for legacy 16-bit packed-RGB source formats (Tier 7).
//!
//! Six source formats x 4 output variants = 24 kernels. Each format word is a
//! little-endian `u16` at 8 pixels per iteration (`vld1q_u16` = 8 x u16).
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

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use super::miri_compat::*;
use crate::row::scalar;

// Internal helpers.
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

/// LE-explicit `u16x8` load from a packed little-endian `u16` byte stream.
///
/// `vld1q_u16` interprets the 16 source bytes as `u16` lanes in host-endian
/// order. Every `AV_PIX_FMT_*LE` source format (RGB565LE, BGR565LE, RGB555LE,
/// BGR555LE, RGB444LE, BGR444LE) stores pixels as **little-endian** `u16`
/// words, matching the scalar's `u16::from_le_bytes` contract. On big-endian
/// `aarch64_be-*` targets the host-endian load reverses the two bytes within
/// each lane, causing every subsequent shift-and-mask to operate on a swapped
/// word and produce silent color corruption while the scalar path remains
/// correct.
///
/// The `vrev16q_u8` on the big-endian branch byte-swaps each `u16` lane back
/// to LE order. On all standard (LE) `aarch64` targets the `cfg!` evaluates to
/// `false` at compile time and the load reduces to a plain `vld1q_u16`.
///
/// # Safety
///
/// `ptr` must be valid for a 16-byte aligned or unaligned `u16` read of 8
/// lanes (i.e. at least 16 bytes of readable memory starting at `ptr`).
/// Caller must ensure NEON is available.
#[inline(always)]
unsafe fn load_u16x8_le(ptr: *const u8) -> uint16x8_t {
  unsafe {
    let v = vld1q_u16(ptr.cast::<u16>());
    if cfg!(target_endian = "big") {
      vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v)))
    } else {
      v
    }
  }
}

// RGB565 — R5 G6 B5, bits [15:11]=R, [10:5]=G, [4:0]=B.
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      let r5 = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g6 = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let b5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16_compat(expand5(r5));
      let g_u8 = vqmovn_u16_compat(expand6(g6));
      let b_u8 = vqmovn_u16_compat(expand5(b5));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      let r5 = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g6 = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let b5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16_compat(expand5(r5));
      let g_u8 = vqmovn_u16_compat(expand6(g6));
      let b_u8 = vqmovn_u16_compat(expand5(b5));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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

// BGR565 — B5 G6 R5, bits [15:11]=B, [10:5]=G, [4:0]=R.
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      // BGR565: B at [15:11], G at [10:5], R at [4:0]
      let b5 = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g6 = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let r5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16_compat(expand5(r5));
      let g_u8 = vqmovn_u16_compat(expand6(g6));
      let b_u8 = vqmovn_u16_compat(expand5(b5));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      let b5 = vandq_u16(vshrq_n_u16::<11>(px), mask5);
      let g6 = vandq_u16(vshrq_n_u16::<5>(px), mask6);
      let r5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16_compat(expand5(r5));
      let g_u8 = vqmovn_u16_compat(expand6(g6));
      let b_u8 = vqmovn_u16_compat(expand5(b5));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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

// RGB555 — 1X R5 G5 B5, bits [14:10]=R, [9:5]=G, [4:0]=B, bit 15 ignored.
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      let r5 = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g5 = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let b5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16_compat(expand5(r5));
      let g_u8 = vqmovn_u16_compat(expand5(g5));
      let b_u8 = vqmovn_u16_compat(expand5(b5));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      let r5 = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g5 = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let b5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16_compat(expand5(r5));
      let g_u8 = vqmovn_u16_compat(expand5(g5));
      let b_u8 = vqmovn_u16_compat(expand5(b5));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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

// BGR555 — 1X B5 G5 R5, bits [14:10]=B, [9:5]=G, [4:0]=R, bit 15 ignored.
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      // BGR555: B at [14:10], G at [9:5], R at [4:0]
      let b5 = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g5 = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let r5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16_compat(expand5(r5));
      let g_u8 = vqmovn_u16_compat(expand5(g5));
      let b_u8 = vqmovn_u16_compat(expand5(b5));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      let b5 = vandq_u16(vshrq_n_u16::<10>(px), mask5);
      let g5 = vandq_u16(vshrq_n_u16::<5>(px), mask5);
      let r5 = vandq_u16(px, mask5);
      let r_u8 = vqmovn_u16_compat(expand5(r5));
      let g_u8 = vqmovn_u16_compat(expand5(g5));
      let b_u8 = vqmovn_u16_compat(expand5(b5));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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

// RGB444 — 4X R4 G4 B4, bits [11:8]=R, [7:4]=G, [3:0]=B, bits [15:12] ignored.
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      let r4 = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g4 = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let b4 = vandq_u16(px, mask4);
      let r_u8 = vqmovn_u16_compat(expand4(r4));
      let g_u8 = vqmovn_u16_compat(expand4(g4));
      let b_u8 = vqmovn_u16_compat(expand4(b4));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      let r4 = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g4 = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let b4 = vandq_u16(px, mask4);
      let r_u8 = vqmovn_u16_compat(expand4(r4));
      let g_u8 = vqmovn_u16_compat(expand4(g4));
      let b_u8 = vqmovn_u16_compat(expand4(b4));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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

// BGR444 — 4X B4 G4 R4, bits [11:8]=B, [7:4]=G, [3:0]=R, bits [15:12] ignored.
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      // BGR444: B at [11:8], G at [7:4], R at [3:0]
      let b4 = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g4 = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let r4 = vandq_u16(px, mask4);
      let r_u8 = vqmovn_u16_compat(expand4(r4));
      let g_u8 = vqmovn_u16_compat(expand4(g4));
      let b_u8 = vqmovn_u16_compat(expand4(b4));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
      let b4 = vandq_u16(vshrq_n_u16::<8>(px), mask4);
      let g4 = vandq_u16(vshrq_n_u16::<4>(px), mask4);
      let r4 = vandq_u16(px, mask4);
      let r_u8 = vqmovn_u16_compat(expand4(r4));
      let g_u8 = vqmovn_u16_compat(expand4(g4));
      let b_u8 = vqmovn_u16_compat(expand4(b4));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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
      let px = load_u16x8_le(src.as_ptr().add(x * 2));
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

// =========================================================================
// Legacy bit-packed RGB/BGR (8bpp 3:3:2 + 1:2:1; 4bpp 1:2:1 two-per-byte)
// (Rgb8 / Bgr8 / Rgb4Byte / Bgr4Byte — 1 byte/pixel;
//  Rgb4 / Bgr4 — 4 bits/pixel, two pixels per byte).
//
// Each iteration produces 8 pixels as a `uint16x8_t` of native source bytes
// (byte formats: widen 8 source bytes; nibble formats: de-interleave 4 source
// bytes into 8 nibble lanes), then reuses the same shift+mask extraction,
// bit-replication expansion, and `vst3`/`vst4` interleaved stores as the
// 16-bit formats above. The `width % 8` remainder defers to `scalar`.
// =========================================================================

/// Bit-replicate a vector of 1-bit values (`0`/`1`) to 8-bit: `c * 0xFF`.
#[inline(always)]
unsafe fn expand1(c: uint16x8_t) -> uint16x8_t {
  unsafe { vmulq_u16(c, vdupq_n_u16(0xFF)) }
}

/// Bit-replicate a vector of 2-bit values (`0..=3`) to 8-bit: `c * 0x55`.
#[inline(always)]
unsafe fn expand2(c: uint16x8_t) -> uint16x8_t {
  unsafe { vmulq_u16(c, vdupq_n_u16(0x55)) }
}

/// Bit-replicate a vector of 3-bit values (`0..=7`) to 8-bit:
/// `(c << 5) | (c << 2) | (c >> 1)`.
#[inline(always)]
unsafe fn expand3(c: uint16x8_t) -> uint16x8_t {
  unsafe {
    vorrq_u16(
      vorrq_u16(vshlq_n_u16::<5>(c), vshlq_n_u16::<2>(c)),
      vshrq_n_u16::<1>(c),
    )
  }
}

/// Load 8 packed 1-byte-per-pixel source bytes and widen to a `uint16x8_t`
/// of native pixel bytes.
///
/// # Safety
///
/// `ptr` must be valid for an 8-byte read; NEON must be available.
#[inline(always)]
unsafe fn load_byte_px8(ptr: *const u8) -> uint16x8_t {
  unsafe { vmovl_u8(vld1_u8(ptr)) }
}

/// Load 4 packed 2-pixel-per-byte source bytes and de-interleave the nibbles
/// into a `uint16x8_t` of 8 native pixel nibbles (`[hi0, lo0, hi1, lo1, …]` —
/// the even pixel is the high nibble `[7:4]`, the odd pixel the low nibble).
///
/// # Safety
///
/// `ptr` must be valid for a 4-byte read; NEON must be available.
#[inline(always)]
unsafe fn load_nibble_px8(ptr: *const u8) -> uint16x8_t {
  unsafe {
    // Assemble the 4 source bytes byte-order-explicitly so lane 0 is `b0` on
    // both endians (`vcreate_u8` fills lanes from the value's least-significant
    // byte up). A host-endian `read_unaligned::<u32>` would reverse the bytes
    // on big-endian AArch64, mirroring the `load_u16x8_le` normalization above.
    let bytes: [u8; 4] = core::ptr::read_unaligned(ptr.cast::<[u8; 4]>());
    let raw = u32::from_le_bytes(bytes);
    let v4 = vcreate_u8(u64::from(raw));
    // Duplicate each of the 4 low bytes: [b0, b0, b1, b1, b2, b2, b3, b3].
    let dup = vzip1_u8(v4, v4);
    let hi = vshr_n_u8::<4>(dup);
    let lo = vand_u8(dup, vdup_n_u8(0x0F));
    // Even lanes take the high nibble, odd lanes the low nibble.
    let even = vcreate_u8(0x00FF_00FF_00FF_00FF);
    vmovl_u8(vbsl_u8(even, hi, lo))
  }
}

/// Emits the four NEON output kernels (rgb / rgba / rgb_u16 / rgba_u16) for
/// one legacy bit-packed format. `$kind` is `byte` or `nibble`; each channel
/// is `(right_shift, native_mask, expand_fn)`.
macro_rules! neon_lowbit_format {
  (@rs 0, $v:expr) => { $v };
  (@rs $s:literal, $v:expr) => { vshrq_n_u16::<$s>($v) };
  (@load byte, $src:expr, $x:expr) => { load_byte_px8($src.as_ptr().add($x)) };
  (@load nibble, $src:expr, $x:expr) => { load_nibble_px8($src.as_ptr().add($x / 2)) };
  (@srcmin byte, $w:expr) => { $w };
  (@srcmin nibble, $w:expr) => { $w.div_ceil(2) };
  (@tail byte, $src:expr, $x:expr) => { &$src[$x..] };
  (@tail nibble, $src:expr, $x:expr) => { &$src[$x / 2..] };
  (
    kind: $kind:tt,
    rgb: $to_rgb:ident, rgba: $to_rgba:ident,
    rgb_u16: $to_rgb_u16:ident, rgba_u16: $to_rgba_u16:ident,
    s_rgb: $s_rgb:path, s_rgba: $s_rgba:path,
    s_rgb_u16: $s_rgb_u16:path, s_rgba_u16: $s_rgba_u16:path,
    r: ($rsh:tt, $rmask:expr, $rexp:ident),
    g: ($gsh:tt, $gmask:expr, $gexp:ident),
    b: ($bsh:tt, $bmask:expr, $bexp:ident),
  ) => {
    /// NEON: packed legacy RGB/BGR → `R, G, B` bytes (8 px/iter).
    ///
    /// # Safety
    ///
    /// NEON available; `src` and `rgb_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn $to_rgb(src: &[u8], rgb_out: &mut [u8], width: usize) {
      debug_assert!(src.len() >= neon_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
      unsafe {
        let rmask = vdupq_n_u16($rmask);
        let gmask = vdupq_n_u16($gmask);
        let bmask = vdupq_n_u16($bmask);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = neon_lowbit_format!(@load $kind, src, x);
          let r = vandq_u16(neon_lowbit_format!(@rs $rsh, px), rmask);
          let g = vandq_u16(neon_lowbit_format!(@rs $gsh, px), gmask);
          let b = vandq_u16(neon_lowbit_format!(@rs $bsh, px), bmask);
          let r_u8 = vqmovn_u16_compat($rexp(r));
          let g_u8 = vqmovn_u16_compat($gexp(g));
          let b_u8 = vqmovn_u16_compat($bexp(b));
          vst3_u8(rgb_out.as_mut_ptr().add(x * 3), uint8x8x3_t(r_u8, g_u8, b_u8));
          x += 8;
        }
        if x < width {
          $s_rgb(neon_lowbit_format!(@tail $kind, src, x), &mut rgb_out[x * 3..], width - x);
        }
      }
    }

    /// NEON: packed legacy RGB/BGR → `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
    ///
    /// # Safety
    ///
    /// NEON available; `src` and `rgba_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn $to_rgba(src: &[u8], rgba_out: &mut [u8], width: usize) {
      debug_assert!(src.len() >= neon_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
      unsafe {
        let rmask = vdupq_n_u16($rmask);
        let gmask = vdupq_n_u16($gmask);
        let bmask = vdupq_n_u16($bmask);
        let alpha = vdup_n_u8(0xFF);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = neon_lowbit_format!(@load $kind, src, x);
          let r = vandq_u16(neon_lowbit_format!(@rs $rsh, px), rmask);
          let g = vandq_u16(neon_lowbit_format!(@rs $gsh, px), gmask);
          let b = vandq_u16(neon_lowbit_format!(@rs $bsh, px), bmask);
          let r_u8 = vqmovn_u16_compat($rexp(r));
          let g_u8 = vqmovn_u16_compat($gexp(g));
          let b_u8 = vqmovn_u16_compat($bexp(b));
          vst4_u8(
            rgba_out.as_mut_ptr().add(x * 4),
            uint8x8x4_t(r_u8, g_u8, b_u8, alpha),
          );
          x += 8;
        }
        if x < width {
          $s_rgba(neon_lowbit_format!(@tail $kind, src, x), &mut rgba_out[x * 4..], width - x);
        }
      }
    }

    /// NEON: packed legacy RGB/BGR → native `R, G, B` u16 (8 px/iter).
    ///
    /// # Safety
    ///
    /// NEON available; `src` and `rgb_u16_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn $to_rgb_u16(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
      debug_assert!(src.len() >= neon_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
      unsafe {
        let rmask = vdupq_n_u16($rmask);
        let gmask = vdupq_n_u16($gmask);
        let bmask = vdupq_n_u16($bmask);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = neon_lowbit_format!(@load $kind, src, x);
          let r = vandq_u16(neon_lowbit_format!(@rs $rsh, px), rmask);
          let g = vandq_u16(neon_lowbit_format!(@rs $gsh, px), gmask);
          let b = vandq_u16(neon_lowbit_format!(@rs $bsh, px), bmask);
          vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), uint16x8x3_t(r, g, b));
          x += 8;
        }
        if x < width {
          $s_rgb_u16(
            neon_lowbit_format!(@tail $kind, src, x),
            &mut rgb_u16_out[x * 3..],
            width - x,
          );
        }
      }
    }

    /// NEON: packed legacy RGB/BGR → native `R, G, B, A` u16 (α = `0xFFFF`).
    ///
    /// # Safety
    ///
    /// NEON available; `src` and `rgba_u16_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn $to_rgba_u16(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
      debug_assert!(src.len() >= neon_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
      unsafe {
        let rmask = vdupq_n_u16($rmask);
        let gmask = vdupq_n_u16($gmask);
        let bmask = vdupq_n_u16($bmask);
        let alpha = vdupq_n_u16(0xFFFF);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = neon_lowbit_format!(@load $kind, src, x);
          let r = vandq_u16(neon_lowbit_format!(@rs $rsh, px), rmask);
          let g = vandq_u16(neon_lowbit_format!(@rs $gsh, px), gmask);
          let b = vandq_u16(neon_lowbit_format!(@rs $bsh, px), bmask);
          vst4q_u16(
            rgba_u16_out.as_mut_ptr().add(x * 4),
            uint16x8x4_t(r, g, b, alpha),
          );
          x += 8;
        }
        if x < width {
          $s_rgba_u16(
            neon_lowbit_format!(@tail $kind, src, x),
            &mut rgba_u16_out[x * 4..],
            width - x,
          );
        }
      }
    }
  };
}

neon_lowbit_format! {
  kind: byte,
  rgb: rgb8_to_rgb_row, rgba: rgb8_to_rgba_row,
  rgb_u16: rgb8_to_rgb_u16_row, rgba_u16: rgb8_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::rgb8_to_rgb_row,
  s_rgba: scalar::legacy_rgb::rgb8_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::rgb8_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::rgb8_to_rgba_u16_row,
  r: (5, 0x07, expand3),
  g: (2, 0x07, expand3),
  b: (0, 0x03, expand2),
}

neon_lowbit_format! {
  kind: byte,
  rgb: bgr8_to_rgb_row, rgba: bgr8_to_rgba_row,
  rgb_u16: bgr8_to_rgb_u16_row, rgba_u16: bgr8_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::bgr8_to_rgb_row,
  s_rgba: scalar::legacy_rgb::bgr8_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::bgr8_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::bgr8_to_rgba_u16_row,
  r: (0, 0x07, expand3),
  g: (3, 0x07, expand3),
  b: (6, 0x03, expand2),
}

neon_lowbit_format! {
  kind: byte,
  rgb: rgb4_byte_to_rgb_row, rgba: rgb4_byte_to_rgba_row,
  rgb_u16: rgb4_byte_to_rgb_u16_row, rgba_u16: rgb4_byte_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::rgb4_byte_to_rgb_row,
  s_rgba: scalar::legacy_rgb::rgb4_byte_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::rgb4_byte_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::rgb4_byte_to_rgba_u16_row,
  r: (3, 0x01, expand1),
  g: (1, 0x03, expand2),
  b: (0, 0x01, expand1),
}

neon_lowbit_format! {
  kind: byte,
  rgb: bgr4_byte_to_rgb_row, rgba: bgr4_byte_to_rgba_row,
  rgb_u16: bgr4_byte_to_rgb_u16_row, rgba_u16: bgr4_byte_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::bgr4_byte_to_rgb_row,
  s_rgba: scalar::legacy_rgb::bgr4_byte_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::bgr4_byte_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::bgr4_byte_to_rgba_u16_row,
  r: (0, 0x01, expand1),
  g: (1, 0x03, expand2),
  b: (3, 0x01, expand1),
}

neon_lowbit_format! {
  kind: nibble,
  rgb: rgb4_to_rgb_row, rgba: rgb4_to_rgba_row,
  rgb_u16: rgb4_to_rgb_u16_row, rgba_u16: rgb4_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::rgb4_to_rgb_row,
  s_rgba: scalar::legacy_rgb::rgb4_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::rgb4_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::rgb4_to_rgba_u16_row,
  r: (3, 0x01, expand1),
  g: (1, 0x03, expand2),
  b: (0, 0x01, expand1),
}

neon_lowbit_format! {
  kind: nibble,
  rgb: bgr4_to_rgb_row, rgba: bgr4_to_rgba_row,
  rgb_u16: bgr4_to_rgb_u16_row, rgba_u16: bgr4_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::bgr4_to_rgb_row,
  s_rgba: scalar::legacy_rgb::bgr4_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::bgr4_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::bgr4_to_rgba_u16_row,
  r: (0, 0x01, expand1),
  g: (1, 0x03, expand2),
  b: (3, 0x01, expand1),
}

#[cfg(all(test, feature = "std"))]
mod endian_safety_tests {
  use super::*;

  /// The 4-bpp nibble loader must de-interleave in **source byte order**
  /// `[b0, b1, b2, b3]` regardless of host endianness, yielding lanes
  /// `[hi0, lo0, hi1, lo1, …]`. A host-endian `read_unaligned::<u32>` would
  /// reverse the bytes on big-endian AArch64; the `from_le_bytes` normalization
  /// keeps the order stable. This passes on the little-endian host and would
  /// fail on a big-endian build under the pre-fix host-endian read.
  #[test]
  #[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
  fn load_nibble_px8_follows_source_byte_order() {
    let bytes = [0x12u8, 0x34, 0x56, 0x78];
    let mut lanes = [0u16; 8];
    // SAFETY: NEON is baseline on aarch64; `bytes` is 4 readable bytes.
    unsafe {
      let v = load_nibble_px8(bytes.as_ptr());
      vst1q_u16(lanes.as_mut_ptr(), v);
    }
    // hi/lo nibbles of b0=0x12, b1=0x34, b2=0x56, b3=0x78 in byte order.
    assert_eq!(lanes, [1, 2, 3, 4, 5, 6, 7, 8]);
  }
}
