//! AVX2 1-bit-per-pixel unpack kernels (Monoblack / Monowhite).
//!
//! # Bit-mask broadcast pattern (32 px / iter, 4 bytes / iter)
//!
//! Each byte covers 8 pixels (MSB first). For 4 input bytes:
//! 1. Build a 256-bit vector with each byte broadcast to its 8 lanes.
//! 2. AND with the bit-position mask `[0x80,...,0x01]` repeated 4× (32 bytes).
//! 3. `_mm256_cmpeq_epi8(and, zero)` → 0x00 where bit was set, 0xFF where clear.
//! 4. Negate for Monoblack (`INVERT=false`).
//! 5. For u16 outputs: process 16 px / iter (2 bytes), zero-extend to u16x16.
//!
//! Tail: scalar fallback.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::row::{
  arch::x86_common::{write_rgb_16, write_rgba_16},
  scalar::mono1bit as scalar,
};

/// Unpack 4 input bytes into a u8x32 luma vector (32 pixels).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_4bytes_avx2<const INVERT: bool>(b0: u8, b1: u8, b2: u8, b3: u8) -> __m256i {
  let mask = _mm256_set_epi8(
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
  let bcast = _mm256_set_epi8(
    b3 as i8, b3 as i8, b3 as i8, b3 as i8, b3 as i8, b3 as i8, b3 as i8, b3 as i8, b2 as i8,
    b2 as i8, b2 as i8, b2 as i8, b2 as i8, b2 as i8, b2 as i8, b2 as i8, b1 as i8, b1 as i8,
    b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b0 as i8, b0 as i8, b0 as i8,
    b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8,
  );
  let anded = _mm256_and_si256(bcast, mask);
  let zero = _mm256_setzero_si256();
  let cmp = _mm256_cmpeq_epi8(anded, zero);
  if INVERT {
    cmp
  } else {
    let all_ones = _mm256_set1_epi8(-1i8);
    _mm256_xor_si256(cmp, all_ones)
  }
}

/// Unpack 2 input bytes into an SSE __m128i luma vector (16 pixels) using AVX2 context.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn unpack_2bytes_as_m128i<const INVERT: bool>(b0: u8, b1: u8) -> __m128i {
  unsafe {
    let y256 = unpack_4bytes_avx2::<INVERT>(b0, b1, 0, 0);
    _mm256_castsi256_si128(y256)
  }
}

/// Zero-extend 16 u8 pixel values (from 2 input bytes) to u16x16.
/// White (0xFF) maps to 0x00FF, matching Gray8's `with_luma_u16` contract.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn expand_2bytes_to_u16x16<const INVERT: bool>(b0: u8, b1: u8) -> __m256i {
  unsafe {
    // Unpack just 2 bytes to get 16 pixels in the low 128 bits.
    let y128 = unpack_2bytes_as_m128i::<INVERT>(b0, b1);
    // Zero-extend 16 u8 lanes to 16 u16 lanes.
    _mm256_cvtepu8_epi16(y128)
  }
}

// ---- mono1bit → RGB u8 -------------------------------------------------------

/// AVX2 `mono1bit_to_rgb_row<INVERT>`: unpack 1-bpp → packed RGB u8.
///
/// Block size: 32 px / iter (4 input bytes). Tail: scalar.
///
/// # Safety
/// AVX2 must be available. `data.len() >= width.div_ceil(8)`.
/// `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 32 <= width {
      let y256 = unpack_4bytes_avx2::<INVERT>(
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
      );
      // Write 32 pixels as two 16-pixel RGB blocks.
      let y_lo = _mm256_castsi256_si128(y256);
      let y_hi = _mm256_extracti128_si256::<1>(y256);
      write_rgb_16(y_lo, y_lo, y_lo, out.as_mut_ptr().add(x * 3));
      write_rgb_16(y_hi, y_hi, y_hi, out.as_mut_ptr().add(x * 3 + 48));
      x += 32;
      byte_idx += 4;
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

/// AVX2 `mono1bit_to_rgba_row<INVERT>`: unpack 1-bpp → packed RGBA u8, α=0xFF.
///
/// # Safety
/// AVX2 must be available. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let alpha128 = _mm_set1_epi8(-1i8);
    while x + 32 <= width {
      let y256 = unpack_4bytes_avx2::<INVERT>(
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
      );
      let y_lo = _mm256_castsi256_si128(y256);
      let y_hi = _mm256_extracti128_si256::<1>(y256);
      write_rgba_16(y_lo, y_lo, y_lo, alpha128, out.as_mut_ptr().add(x * 4));
      write_rgba_16(y_hi, y_hi, y_hi, alpha128, out.as_mut_ptr().add(x * 4 + 64));
      x += 32;
      byte_idx += 4;
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

/// AVX2 `mono1bit_to_luma_row<INVERT>`: unpack 1-bpp → luma u8.
///
/// # Safety
/// AVX2 must be available. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 32 <= width {
      let y256 = unpack_4bytes_avx2::<INVERT>(
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
      );
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), y256);
      x += 32;
      byte_idx += 4;
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

/// AVX2 `mono1bit_to_rgb_u16_row<INVERT>`: unpack 1-bpp → RGB u16.
///
/// Block size: 16 px / iter (2 input bytes). Tail: scalar.
///
/// # Safety
/// AVX2 must be available. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 16 <= width {
      // 2 bytes → 16 pixels → 16 u16 values → 48 u16 elements of RGB.
      let y256 = expand_2bytes_to_u16x16::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      let y_lo = _mm256_castsi256_si128(y256);
      let y_hi = _mm256_extracti128_si256::<1>(y256);
      // Write 8 pixels × 3 u16 = 24 u16 for each half.
      use crate::row::arch::x86_common::write_rgb_u16_8;
      write_rgb_u16_8(y_lo, y_lo, y_lo, out.as_mut_ptr().add(x * 3));
      write_rgb_u16_8(y_hi, y_hi, y_hi, out.as_mut_ptr().add(x * 3 + 24));
      x += 16;
      byte_idx += 2;
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

/// AVX2 `mono1bit_to_rgba_u16_row<INVERT>`: unpack 1-bpp → RGBA u16, α=0xFFFF.
///
/// # Safety
/// AVX2 must be available. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let alpha128 = _mm_set1_epi16(0x00FFi16); // zero-extend of 0xFF u8
    while x + 16 <= width {
      let y256 = expand_2bytes_to_u16x16::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      let y_lo = _mm256_castsi256_si128(y256);
      let y_hi = _mm256_extracti128_si256::<1>(y256);
      use crate::row::arch::x86_common::write_rgba_u16_8;
      write_rgba_u16_8(y_lo, y_lo, y_lo, alpha128, out.as_mut_ptr().add(x * 4));
      write_rgba_u16_8(y_hi, y_hi, y_hi, alpha128, out.as_mut_ptr().add(x * 4 + 32));
      x += 16;
      byte_idx += 2;
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

/// AVX2 `mono1bit_to_luma_u16_row<INVERT>`: unpack 1-bpp → luma u16.
///
/// Block size: 16 px / iter (2 input bytes). Tail: scalar.
///
/// # Safety
/// AVX2 must be available. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 16 <= width {
      let y256 = expand_2bytes_to_u16x16::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), y256);
      x += 16;
      byte_idx += 2;
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

/// AVX2 `mono1bit_to_hsv_row<INVERT>`: H=0, S=0, V=Y.
///
/// # Safety
/// AVX2 must be available. All output slices >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_si256();
    while x + 32 <= width {
      let y256 = unpack_4bytes_avx2::<INVERT>(
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
      );
      _mm256_storeu_si256(h.as_mut_ptr().add(x).cast(), zero);
      _mm256_storeu_si256(s.as_mut_ptr().add(x).cast(), zero);
      _mm256_storeu_si256(v.as_mut_ptr().add(x).cast(), y256);
      x += 32;
      byte_idx += 4;
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

/// Monoblack → RGB u8 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monoblack_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u8 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monoblack_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<false>(data, out, width) }
}

/// Monoblack → RGB u16 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monoblack_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u16 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monoblack_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<false>(data, out, width) }
}

/// Monoblack → Luma u8 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monoblack_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<false>(data, out, width) }
}

/// Monoblack → Luma u16 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monoblack_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<false>(data, out, width) }
}

/// Monoblack → HSV (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monoblack_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<false>(data, h, s, v, width) }
}

/// Monowhite → RGB u8 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monowhite_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u8 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monowhite_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<true>(data, out, width) }
}

/// Monowhite → RGB u16 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monowhite_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u16 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monowhite_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<true>(data, out, width) }
}

/// Monowhite → Luma u8 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monowhite_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<true>(data, out, width) }
}

/// Monowhite → Luma u16 (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monowhite_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<true>(data, out, width) }
}

/// Monowhite → HSV (AVX2).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn monowhite_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<true>(data, h, s, v, width) }
}
