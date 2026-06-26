//! NEON kernels for 32-bit packed RGB / RGBA sources (Rgb96 / Rgba128).
//!
// Kernels are wired into the dispatcher in the SIMD dispatch task; suppress
// dead_code until then.
#![allow(dead_code)]
//!
//! ## Format layouts
//!
//! | Format  | Elements per pixel | Channel order in memory |
//! |---------|--------------------|------------------------|
//! | Rgb96   | 3 u32              | R, G, B                |
//! | Rgba128 | 4 u32              | R, G, B, A             |
//!
//! ## Per-format SIMD strategy
//!
//! - **u16 output:** 4 pixels per iteration. `vld3q_u32` (Rgb96) /
//!   `vld4q_u32` (Rgba128) deinterleaves 4 pixels into per-channel
//!   `uint32x4_t` vectors; `vshrn_n_u32::<16>` narrows each to a
//!   `uint16x4_t` (D-register), stored via `vst3_u16` / `vst4_u16`.
//! - **u8 output:** 8 pixels per iteration. Two `vld3q_u32` / `vld4q_u32`
//!   loads (4 pixels each) narrow to `uint16x4_t` halves that `vcombine_u16`
//!   joins into a `uint16x8_t` per channel; a second `vshrn_n_u16::<8>`
//!   completes the `>> 24` narrow to `uint8x8_t`, stored via `vst3_u8` /
//!   `vst4_u8`.
//!
//! ## Big-endian support
//!
//! Every per-channel `uint32x4_t` produced by the deinterleaving load is
//! conditionally byte-swapped via [`super::bswap_u32x4_if_be`] before any
//! channel math. The gate is `BE != HOST_NATIVE_BE`, so the swap fires only
//! when the wire endian differs from the host's native byte order.
//!
//! ## Depth conversion
//!
//! - **u32 → u8:**  `vshrn_n_u32::<16>` then `vshrn_n_u16::<8>` — net `>> 24`,
//!   matching scalar `(v >> 24) as u8`.
//! - **u32 → u16:** `vshrn_n_u32::<16>` — `>> 16`, matching scalar
//!   `(v >> 16) as u16`.
//!
//! ## Scalar tail
//!
//! All kernels handle the `width % {4, 8}` remaining pixels via the scalar
//! reference.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use super::bswap_u32x4_if_be;
use crate::row::scalar;

// Rgb96 (R, G, B — 3 u32 elements per pixel).

/// NEON Rgb96 → packed u8 RGB. 8 pixels per SIMD iteration.
///
/// Two `vld3q_u32` loads deinterleave 8 pixels into `(R, G, B)` u32 halves;
/// each is narrowed `>> 16` to u16x4, paired via `vcombine_u16`, then
/// `vshrn_n_u16::<8>` completes the `>> 24` narrow before `vst3_u8`.
///
/// When `BE = true` each channel vector is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgb96_to_rgb_row<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let lo: uint32x4x3_t = vld3q_u32(rgb96.as_ptr().add(x * 3));
      let hi: uint32x4x3_t = vld3q_u32(rgb96.as_ptr().add((x + 4) * 3));
      let r = narrow_u32_pair_to_u8x8::<BE>(lo.0, hi.0);
      let g = narrow_u32_pair_to_u8x8::<BE>(lo.1, hi.1);
      let b = narrow_u32_pair_to_u8x8::<BE>(lo.2, hi.2);
      vst3_u8(rgb_out.as_mut_ptr().add(x * 3), uint8x8x3_t(r, g, b));
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgb_row::<BE>(&rgb96[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Rgb96 → packed u8 RGBA. 8 pixels per SIMD iteration. Alpha forced to 0xFF.
///
/// When `BE = true` each channel vector is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgb96_to_rgba_row<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdup_n_u8(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let lo: uint32x4x3_t = vld3q_u32(rgb96.as_ptr().add(x * 3));
      let hi: uint32x4x3_t = vld3q_u32(rgb96.as_ptr().add((x + 4) * 3));
      let r = narrow_u32_pair_to_u8x8::<BE>(lo.0, hi.0);
      let g = narrow_u32_pair_to_u8x8::<BE>(lo.1, hi.1);
      let b = narrow_u32_pair_to_u8x8::<BE>(lo.2, hi.2);
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r, g, b, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgba_row::<BE>(&rgb96[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON Rgb96 → native-depth u16 RGB. 4 pixels per SIMD iteration.
///
/// `vld3q_u32` deinterleaves; `vshrn_n_u32::<16>` narrows each channel `>> 16`;
/// `vst3_u16` reinterleaves.
/// When `BE = true` each channel is byte-swapped to host-native order before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgb96_to_rgb_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let px: uint32x4x3_t = vld3q_u32(rgb96.as_ptr().add(x * 3));
      vst3_u16(
        rgb_out.as_mut_ptr().add(x * 3),
        uint16x4x3_t(
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.0)),
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.1)),
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.2)),
        ),
      );
      x += 4;
    }
    if x < width {
      scalar::rgb96_to_rgb_u16_row::<BE>(&rgb96[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Rgb96 → native-depth u16 RGBA. 4 pixels per SIMD iteration. Alpha forced to 0xFFFF.
///
/// When `BE = true` each channel is byte-swapped to host-native order before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgb96_to_rgba_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdup_n_u16(0xFFFF);
    let mut x = 0usize;
    while x + 4 <= width {
      let px: uint32x4x3_t = vld3q_u32(rgb96.as_ptr().add(x * 3));
      vst4_u16(
        rgba_out.as_mut_ptr().add(x * 4),
        uint16x4x4_t(
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.0)),
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.1)),
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.2)),
          alpha,
        ),
      );
      x += 4;
    }
    if x < width {
      scalar::rgb96_to_rgba_u16_row::<BE>(&rgb96[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// Rgba128 (R, G, B, A — 4 u32 elements per pixel).

/// NEON Rgba128 → packed u8 RGB. 8 pixels per SIMD iteration. Alpha discarded.
///
/// `vld4q_u32` deinterleaves into `(R, G, B, A)`; R/G/B narrowed `>> 24`;
/// `vst3_u8` writes 3 channels.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgba128_to_rgb_row<const BE: bool>(
  rgba128: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let lo: uint32x4x4_t = vld4q_u32(rgba128.as_ptr().add(x * 4));
      let hi: uint32x4x4_t = vld4q_u32(rgba128.as_ptr().add((x + 4) * 4));
      let r = narrow_u32_pair_to_u8x8::<BE>(lo.0, hi.0);
      let g = narrow_u32_pair_to_u8x8::<BE>(lo.1, hi.1);
      let b = narrow_u32_pair_to_u8x8::<BE>(lo.2, hi.2);
      vst3_u8(rgb_out.as_mut_ptr().add(x * 3), uint8x8x3_t(r, g, b));
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgb_row::<BE>(&rgba128[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Rgba128 → packed u8 RGBA. 8 pixels per SIMD iteration. Source alpha
/// passes through (narrowed `>> 24`).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgba128_to_rgba_row<const BE: bool>(
  rgba128: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let lo: uint32x4x4_t = vld4q_u32(rgba128.as_ptr().add(x * 4));
      let hi: uint32x4x4_t = vld4q_u32(rgba128.as_ptr().add((x + 4) * 4));
      let r = narrow_u32_pair_to_u8x8::<BE>(lo.0, hi.0);
      let g = narrow_u32_pair_to_u8x8::<BE>(lo.1, hi.1);
      let b = narrow_u32_pair_to_u8x8::<BE>(lo.2, hi.2);
      let a = narrow_u32_pair_to_u8x8::<BE>(lo.3, hi.3);
      vst4_u8(rgba_out.as_mut_ptr().add(x * 4), uint8x8x4_t(r, g, b, a));
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgba_row::<BE>(&rgba128[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON Rgba128 → native-depth u16 RGB. 4 pixels per SIMD iteration. Alpha discarded.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgba128_to_rgb_u16_row<const BE: bool>(
  rgba128: &[u32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let px: uint32x4x4_t = vld4q_u32(rgba128.as_ptr().add(x * 4));
      vst3_u16(
        rgb_out.as_mut_ptr().add(x * 3),
        uint16x4x3_t(
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.0)),
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.1)),
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.2)),
        ),
      );
      x += 4;
    }
    if x < width {
      scalar::rgba128_to_rgb_u16_row::<BE>(&rgba128[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Rgba128 → native-depth u16 RGBA. 4 pixels per SIMD iteration. Source
/// alpha passes through (narrowed `>> 16`).
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgba128_to_rgba_u16_row<const BE: bool>(
  rgba128: &[u32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let px: uint32x4x4_t = vld4q_u32(rgba128.as_ptr().add(x * 4));
      vst4_u16(
        rgba_out.as_mut_ptr().add(x * 4),
        uint16x4x4_t(
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.0)),
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.1)),
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.2)),
          vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(px.3)),
        ),
      );
      x += 4;
    }
    if x < width {
      scalar::rgba128_to_rgba_u16_row::<BE>(&rgba128[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// ---- Shared narrowing helper ------------------------------------------------

/// Narrows two `uint32x4_t` channel halves (4 + 4 pixels) to a single
/// `uint8x8_t` via `>> 24`: each half is byte-swap-corrected, narrowed
/// `>> 16` to `uint16x4_t`, joined with `vcombine_u16`, then `vshrn_n_u16::<8>`
/// completes the narrow.
///
/// # Safety
///
/// NEON must be available in the caller's `target_feature` context.
#[inline(always)]
unsafe fn narrow_u32_pair_to_u8x8<const BE: bool>(lo: uint32x4_t, hi: uint32x4_t) -> uint8x8_t {
  unsafe {
    let lo16 = vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(lo));
    let hi16 = vshrn_n_u32::<16>(bswap_u32x4_if_be::<BE>(hi));
    vshrn_n_u16::<8>(vcombine_u16(lo16, hi16))
  }
}
