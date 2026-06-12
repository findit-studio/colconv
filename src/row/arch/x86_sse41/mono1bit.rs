//! SSE4.1 1-bit-per-pixel unpack kernels (Monoblack / Monowhite).
//!
//! # Bit-mask broadcast pattern (16 px / iter, 2 bytes / iter)
//!
//! Each byte covers 8 pixels (MSB first):
//! 1. `_mm_set1_epi8(byte)` — broadcast to 16 lanes.
//! 2. AND with bit-position mask `[0x80,0x40,...,0x01]` repeated twice (16 bytes).
//! 3. `_mm_cmpeq_epi8(and, zero)` → 0x00 where bit was set, 0xFF where clear.
//! 4. Negate: `_mm_xor_si128(cmp, all_ones)` gives 0xFF for set bit (Monoblack).
//! 5. For Monowhite (`INVERT=true`): skip the negate — the `cmpeq` zero result is 0xFF.
//!
//! After unpacking 16 pixels, the Y u8x16 vector is ready for output:
//! - RGB: write via `write_rgb_16` (3-channel interleave).
//! - RGBA: write via `write_rgba_16` (4-channel interleave, α=0xFF).
//! - Luma u8: store with `_mm_storeu_si128`.
//! - u16 outputs: process 8 px / iter, zero-extend the low 8 bytes to u16x8 with
//!   `_mm_unpacklo_epi8(y, zero)`, then store via `write_rgb_u16_8`
//!   / `write_rgba_u16_8` (shared SSSE3/SSE2 interleave helpers).
//!
//! Tail (remaining pixels after last full 16-px block) falls back to scalar.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::row::{
  arch::x86_common::{write_rgb_16, write_rgb_u16_8, write_rgba_16, write_rgba_u16_8},
  scalar::mono1bit as scalar,
};

/// Unpack 2 input bytes into a u8x16 luma vector (16 pixels).
/// For INVERT=false (Monoblack): bit=1 → lane=0xFF, bit=0 → lane=0x00.
/// For INVERT=true (Monowhite): bit=0 → lane=0xFF, bit=1 → lane=0x00.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn unpack_2bytes_sse41<const INVERT: bool>(b0: u8, b1: u8) -> __m128i {
  // Bit-position mask: lane i of each 8-lane half tests bit (7 - i).
  let mask = _mm_set_epi8(
    0x01u8 as i8,
    0x02u8 as i8,
    0x04u8 as i8,
    0x08u8 as i8,
    0x10u8 as i8,
    0x20u8 as i8,
    0x40u8 as i8,
    0x80u8 as i8,
    0x01u8 as i8,
    0x02u8 as i8,
    0x04u8 as i8,
    0x08u8 as i8,
    0x10u8 as i8,
    0x20u8 as i8,
    0x40u8 as i8,
    0x80u8 as i8,
  );
  // Build the 16-byte broadcast vector: low 8 lanes = b0, high 8 lanes = b1.
  let bcast = _mm_set_epi8(
    b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b0 as i8,
    b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8,
  );
  let anded = _mm_and_si128(bcast, mask);
  let zero = _mm_setzero_si128();
  // cmpeq: 0xFF where (anded == 0), i.e., where bit was NOT set.
  let cmp = _mm_cmpeq_epi8(anded, zero);
  if INVERT {
    // Monowhite: bit=0 (not set) → white (0xFF) — already what cmp gives.
    cmp
  } else {
    // Monoblack: bit=1 (set) → white (0xFF) → negate the cmpeq.
    let all_ones = _mm_set1_epi8(-1i8);
    _mm_xor_si128(cmp, all_ones)
  }
}

/// Zero-extend a u8x8 (low 8 bytes of a __m128i) to u16x8.
/// White (0xFF) maps to 0x00FF, matching Gray8's `with_luma_u16` contract.
/// Returns a full __m128i with 8 u16 values.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn expand_y_to_u16x8_sse41(y_low8: __m128i) -> __m128i {
  // _mm_unpacklo_epi8(y, zero): interleave low 8 bytes with 0x00
  // → [y0, 0, y1, 0, ..., y7, 0] as a u8x16, i.e. u16x8 of y (zero-extended).
  let zero = _mm_setzero_si128();
  _mm_unpacklo_epi8(y_low8, zero)
}

// ---- mono1bit → RGB u8 -------------------------------------------------------

/// SSE4.1 `mono1bit_to_rgb_row<INVERT>`: unpack 1-bpp → packed RGB u8.
///
/// Block size: 16 px / iter (2 input bytes). Tail: scalar.
///
/// # Safety
/// SSE4.1 must be available. `data.len() >= width.div_ceil(8)`.
/// `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn mono1bit_to_rgb_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = unpack_2bytes_sse41::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      write_rgb_16(y, y, y, out.as_mut_ptr().add(x * 3));
      x += 16;
      byte_idx += 2;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_rgb_row(&data[byte_idx..], &mut out[x * 3..width * 3], width - x);
    } else {
      scalar::monoblack_to_rgb_row(&data[byte_idx..], &mut out[x * 3..width * 3], width - x);
    }
  }
}

// ---- mono1bit → RGBA u8 -----------------------------------------------------

/// SSE4.1 `mono1bit_to_rgba_row<INVERT>`: unpack 1-bpp → packed RGBA u8, α=0xFF.
///
/// # Safety
/// SSE4.1 must be available. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn mono1bit_to_rgba_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    let alpha = _mm_set1_epi8(-1i8); // 0xFF
    while x + 16 <= width {
      let y = unpack_2bytes_sse41::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      write_rgba_16(y, y, y, alpha, out.as_mut_ptr().add(x * 4));
      x += 16;
      byte_idx += 2;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_rgba_row(&data[byte_idx..], &mut out[x * 4..width * 4], width - x);
    } else {
      scalar::monoblack_to_rgba_row(&data[byte_idx..], &mut out[x * 4..width * 4], width - x);
    }
  }
}

// ---- mono1bit → Luma u8 -----------------------------------------------------

/// SSE4.1 `mono1bit_to_luma_row<INVERT>`: unpack 1-bpp → luma u8.
///
/// # Safety
/// SSE4.1 must be available. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn mono1bit_to_luma_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = unpack_2bytes_sse41::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y);
      x += 16;
      byte_idx += 2;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_luma_row(&data[byte_idx..], &mut out[x..width], width - x);
    } else {
      scalar::monoblack_to_luma_row(&data[byte_idx..], &mut out[x..width], width - x);
    }
  }
}

// ---- mono1bit → RGB u16 -----------------------------------------------------

/// SSE4.1 `mono1bit_to_rgb_u16_row<INVERT>`: unpack 1-bpp → RGB u16.
///
/// Block size: 8 px / iter (1 input byte). Tail: scalar.
///
/// # Safety
/// SSE4.1 must be available. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn mono1bit_to_rgb_u16_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    while x + 8 <= width {
      // Process 1 byte = 8 pixels.
      let y8_128 = unpack_2bytes_sse41::<INVERT>(data[byte_idx], 0);
      // Extract low 8 bytes (our 8 pixels) and expand to u16x8.
      let y16 = expand_y_to_u16x8_sse41(y8_128);
      // Write 8 pixels × 3 channels = 24 u16 = 48 bytes via the shared
      // SSSE3 shuffle-based interleave helper (y broadcast to R, G, B).
      write_rgb_u16_8(y16, y16, y16, out.as_mut_ptr().add(x * 3));
      x += 8;
      byte_idx += 1;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_rgb_u16_row(&data[byte_idx..], &mut out[x * 3..width * 3], width - x);
    } else {
      scalar::monoblack_to_rgb_u16_row(&data[byte_idx..], &mut out[x * 3..width * 3], width - x);
    }
  }
}

// ---- mono1bit → RGBA u16 ----------------------------------------------------

/// SSE4.1 `mono1bit_to_rgba_u16_row<INVERT>`: unpack 1-bpp → RGBA u16, α=0xFFFF.
///
/// # Safety
/// SSE4.1 must be available. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn mono1bit_to_rgba_u16_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    while x + 8 <= width {
      let y8_128 = unpack_2bytes_sse41::<INVERT>(data[byte_idx], 0);
      let y16 = expand_y_to_u16x8_sse41(y8_128);
      // α=0x00FF for all 8 pixels (zero-extend of 0xFF u8). Cast to i16 since __m128i is signed.
      let alpha = _mm_set1_epi16(0x00FFu16 as i16);
      // Write 8 pixels × 4 channels = 32 u16 = 64 bytes via the shared
      // SSE2 unpack-based interleave helper (y broadcast to R, G, B).
      write_rgba_u16_8(y16, y16, y16, alpha, out.as_mut_ptr().add(x * 4));
      x += 8;
      byte_idx += 1;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_rgba_u16_row(&data[byte_idx..], &mut out[x * 4..width * 4], width - x);
    } else {
      scalar::monoblack_to_rgba_u16_row(&data[byte_idx..], &mut out[x * 4..width * 4], width - x);
    }
  }
}

// ---- mono1bit → Luma u16 ----------------------------------------------------

/// SSE4.1 `mono1bit_to_luma_u16_row<INVERT>`: unpack 1-bpp → luma u16.
///
/// # Safety
/// SSE4.1 must be available. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn mono1bit_to_luma_u16_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    while x + 8 <= width {
      let y8_128 = unpack_2bytes_sse41::<INVERT>(data[byte_idx], 0);
      let y16 = expand_y_to_u16x8_sse41(y8_128);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y16);
      x += 8;
      byte_idx += 1;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_luma_u16_row(&data[byte_idx..], &mut out[x..width], width - x);
    } else {
      scalar::monoblack_to_luma_u16_row(&data[byte_idx..], &mut out[x..width], width - x);
    }
  }
}

// ---- mono1bit → HSV ---------------------------------------------------------

/// SSE4.1 `mono1bit_to_hsv_row<INVERT>`: H=0, S=0, V=Y.
///
/// # Safety
/// SSE4.1 must be available. All output slices >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn mono1bit_to_hsv_row<const INVERT: bool>(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(h.len() >= width);
  debug_assert!(s.len() >= width);
  debug_assert!(v.len() >= width);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    let zero = _mm_setzero_si128();
    while x + 16 <= width {
      let y = unpack_2bytes_sse41::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      _mm_storeu_si128(h.as_mut_ptr().add(x).cast(), zero);
      _mm_storeu_si128(s.as_mut_ptr().add(x).cast(), zero);
      _mm_storeu_si128(v.as_mut_ptr().add(x).cast(), y);
      x += 16;
      byte_idx += 2;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_hsv_row(
        &data[byte_idx..],
        &mut h[x..width],
        &mut s[x..width],
        &mut v[x..width],
        width - x,
      );
    } else {
      scalar::monoblack_to_hsv_row(
        &data[byte_idx..],
        &mut h[x..width],
        &mut s[x..width],
        &mut v[x..width],
        width - x,
      );
    }
  }
}

// ---- Public wrappers --------------------------------------------------------

/// Monoblack → RGB u8 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monoblack_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u8 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monoblack_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<false>(data, out, width) }
}

/// Monoblack → RGB u16 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monoblack_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u16 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monoblack_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<false>(data, out, width) }
}

/// Monoblack → Luma u8 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monoblack_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<false>(data, out, width) }
}

/// Monoblack → Luma u16 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monoblack_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<false>(data, out, width) }
}

/// Monoblack → HSV (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monoblack_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<false>(data, h, s, v, width) }
}

/// Monowhite → RGB u8 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monowhite_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u8 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monowhite_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<true>(data, out, width) }
}

/// Monowhite → RGB u16 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monowhite_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u16 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monowhite_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<true>(data, out, width) }
}

/// Monowhite → Luma u8 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monowhite_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<true>(data, out, width) }
}

/// Monowhite → Luma u16 (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monowhite_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<true>(data, out, width) }
}

/// Monowhite → HSV (SSE4.1).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn monowhite_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<true>(data, h, s, v, width) }
}
