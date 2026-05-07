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
//! - u16 outputs: process 8 px / iter, unpack the low 8 bytes to u16x8 with
//!   `(y << 8) | y` via `_mm_unpacklo_epi8`.
//!
//! Tail (remaining pixels after last full 16-px block) falls back to scalar.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::x86_64::*;

use crate::row::{
  arch::x86_common::{write_rgb_16, write_rgba_16},
  scalar::mono1bit as scalar,
};

/// Unpack 2 input bytes into a u8x16 luma vector (16 pixels).
/// For INVERT=false (Monoblack): bit=1 → lane=0xFF, bit=0 → lane=0x00.
/// For INVERT=true (Monowhite): bit=0 → lane=0xFF, bit=1 → lane=0x00.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn unpack_2bytes_sse41<const INVERT: bool>(b0: u8, b1: u8) -> __m128i {
  unsafe {
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
    // Broadcast each byte into 8 lanes. b0 → low 8 lanes, b1 → high 8 lanes.
    let lo = _mm_set1_epi8(b0 as i8);
    let hi = _mm_set1_epi8(b1 as i8);
    // Combine: low half = b0 broadcast, high half = b1 broadcast.
    let both = _mm_unpacklo_epi64(
      _mm_unpacklo_epi8(lo, lo), // just need b0 in low 8 bytes
      _mm_unpacklo_epi8(hi, hi), // just need b1 in low 8 bytes
    );
    // Actually, simpler: use insert + shuffle.
    // Easiest: build the 16-byte vector directly.
    let bcast = _mm_set_epi8(
      b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b0 as i8,
      b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8,
    );
    let _ = both; // discard the earlier attempt
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
}

/// Expand a u8x8 (low 8 bytes of a __m128i) to u16x8 with `(y << 8) | y`.
/// Returns a full __m128i with 8 u16 values.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn expand_y_to_u16x8_sse41(y_low8: __m128i) -> __m128i {
  unsafe {
    // _mm_unpacklo_epi8(y, y): interleave low 8 bytes with themselves
    // → [y0, y0, y1, y1, ..., y7, y7] as a u8x16, i.e. u16x8 of (y | y << 8).
    _mm_unpacklo_epi8(y_low8, y_low8)
  }
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
      // Write 8 pixels × 3 channels = 24 u16 = 48 bytes.
      // No intrinsic interleave-3 in SSE4.1 — use scalar for the store.
      let ptr = out.as_mut_ptr().add(x * 3);
      // Extract each lane and write manually (8 pixels × 3 u16).
      // This is still faster than the full scalar bit-extraction path.
      let v = _mm_extract_epi16::<0>(y16) as u16;
      *ptr = v;
      *ptr.add(1) = v;
      *ptr.add(2) = v;
      let v = _mm_extract_epi16::<1>(y16) as u16;
      *ptr.add(3) = v;
      *ptr.add(4) = v;
      *ptr.add(5) = v;
      let v = _mm_extract_epi16::<2>(y16) as u16;
      *ptr.add(6) = v;
      *ptr.add(7) = v;
      *ptr.add(8) = v;
      let v = _mm_extract_epi16::<3>(y16) as u16;
      *ptr.add(9) = v;
      *ptr.add(10) = v;
      *ptr.add(11) = v;
      let v = _mm_extract_epi16::<4>(y16) as u16;
      *ptr.add(12) = v;
      *ptr.add(13) = v;
      *ptr.add(14) = v;
      let v = _mm_extract_epi16::<5>(y16) as u16;
      *ptr.add(15) = v;
      *ptr.add(16) = v;
      *ptr.add(17) = v;
      let v = _mm_extract_epi16::<6>(y16) as u16;
      *ptr.add(18) = v;
      *ptr.add(19) = v;
      *ptr.add(20) = v;
      let v = _mm_extract_epi16::<7>(y16) as u16;
      *ptr.add(21) = v;
      *ptr.add(22) = v;
      *ptr.add(23) = v;
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
      let ptr = out.as_mut_ptr().add(x * 4);
      let v = _mm_extract_epi16::<0>(y16) as u16;
      *ptr = v;
      *ptr.add(1) = v;
      *ptr.add(2) = v;
      *ptr.add(3) = 0xFFFF;
      let v = _mm_extract_epi16::<1>(y16) as u16;
      *ptr.add(4) = v;
      *ptr.add(5) = v;
      *ptr.add(6) = v;
      *ptr.add(7) = 0xFFFF;
      let v = _mm_extract_epi16::<2>(y16) as u16;
      *ptr.add(8) = v;
      *ptr.add(9) = v;
      *ptr.add(10) = v;
      *ptr.add(11) = 0xFFFF;
      let v = _mm_extract_epi16::<3>(y16) as u16;
      *ptr.add(12) = v;
      *ptr.add(13) = v;
      *ptr.add(14) = v;
      *ptr.add(15) = 0xFFFF;
      let v = _mm_extract_epi16::<4>(y16) as u16;
      *ptr.add(16) = v;
      *ptr.add(17) = v;
      *ptr.add(18) = v;
      *ptr.add(19) = 0xFFFF;
      let v = _mm_extract_epi16::<5>(y16) as u16;
      *ptr.add(20) = v;
      *ptr.add(21) = v;
      *ptr.add(22) = v;
      *ptr.add(23) = 0xFFFF;
      let v = _mm_extract_epi16::<6>(y16) as u16;
      *ptr.add(24) = v;
      *ptr.add(25) = v;
      *ptr.add(26) = v;
      *ptr.add(27) = 0xFFFF;
      let v = _mm_extract_epi16::<7>(y16) as u16;
      *ptr.add(28) = v;
      *ptr.add(29) = v;
      *ptr.add(30) = v;
      *ptr.add(31) = 0xFFFF;
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
