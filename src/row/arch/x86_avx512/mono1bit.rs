//! AVX-512F+BW 1-bit-per-pixel unpack kernels (Monoblack / Monowhite).
//!
//! # Bit-mask broadcast pattern (64 px / iter, 8 bytes / iter)
//!
//! Each byte covers 8 pixels (MSB first). For 8 input bytes:
//! 1. Build a 512-bit vector with each byte broadcast to its 8 lanes.
//! 2. AND with the bit-position mask `[0x80,...,0x01]` repeated 8× (64 bytes).
//! 3. `_mm512_cmpeq_epi8_mask` + `_mm512_movm_epi8`: compare-to-mask → 0xFF per set bit.
//!    Alternative: `_mm512_cmpeq_epi8` then negate (using AVX-512BW).
//! 4. For Monowhite (`INVERT=true`): use the inverted comparison.
//! 5. For u16 outputs: process 32 px / iter (4 bytes), unpack to u16x32.
//!
//! Requires AVX-512F + AVX-512BW (no VBMI required).

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::x86_64::*;

use crate::row::{
  arch::x86_common::{write_rgb_16, write_rgb_u16_8, write_rgba_16, write_rgba_u16_8},
  scalar::mono1bit as scalar,
};

/// Unpack 8 input bytes into a u8x64 luma vector (64 pixels).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn unpack_8bytes_avx512<const INVERT: bool>(b: [u8; 8]) -> __m512i {
  let mask = _mm512_set_epi8(
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
  let bcast = _mm512_set_epi8(
    b[7] as i8, b[7] as i8, b[7] as i8, b[7] as i8, b[7] as i8, b[7] as i8, b[7] as i8,
    b[7] as i8, b[6] as i8, b[6] as i8, b[6] as i8, b[6] as i8, b[6] as i8, b[6] as i8,
    b[6] as i8, b[6] as i8, b[5] as i8, b[5] as i8, b[5] as i8, b[5] as i8, b[5] as i8,
    b[5] as i8, b[5] as i8, b[5] as i8, b[4] as i8, b[4] as i8, b[4] as i8, b[4] as i8,
    b[4] as i8, b[4] as i8, b[4] as i8, b[4] as i8, b[3] as i8, b[3] as i8, b[3] as i8,
    b[3] as i8, b[3] as i8, b[3] as i8, b[3] as i8, b[3] as i8, b[2] as i8, b[2] as i8,
    b[2] as i8, b[2] as i8, b[2] as i8, b[2] as i8, b[2] as i8, b[2] as i8, b[1] as i8,
    b[1] as i8, b[1] as i8, b[1] as i8, b[1] as i8, b[1] as i8, b[1] as i8, b[1] as i8,
    b[0] as i8, b[0] as i8, b[0] as i8, b[0] as i8, b[0] as i8, b[0] as i8, b[0] as i8,
    b[0] as i8,
  );
  let anded = _mm512_and_si512(bcast, mask);
  let zero = _mm512_setzero_si512();
  // cmpeq_epi8_mask: k=1 where anded==0 (bit not set).
  // movm_epi8: expand mask bit to 0xFF per set mask bit, 0x00 per clear.
  let eq_mask: __mmask64 = _mm512_cmpeq_epi8_mask(anded, zero);
  if INVERT {
    // Monowhite: 0xFF where bit=0 (not set) → directly use eq_mask.
    _mm512_movm_epi8(eq_mask)
  } else {
    // Monoblack: 0xFF where bit=1 (set) → use NOT eq_mask.
    _mm512_movm_epi8(!eq_mask)
  }
}

/// Unpack 4 input bytes into a u8x32 luma vector (32 pixels) in an AVX-512 context.
/// Returns a __m256i.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn unpack_4bytes_as_m256i<const INVERT: bool>(b0: u8, b1: u8, b2: u8, b3: u8) -> __m256i {
  unsafe {
    let y512 = unpack_8bytes_avx512::<INVERT>([b0, b1, b2, b3, 0, 0, 0, 0]);
    _mm512_castsi512_si256(y512)
  }
}

/// Expand 4 input bytes (32 pixels) to u16x32 with `(y << 8) | y`.
/// Returns a __m512i with 32 u16 values.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn expand_4bytes_to_u16x32<const INVERT: bool>(b0: u8, b1: u8, b2: u8, b3: u8) -> __m512i {
  unsafe {
    let y256 = unpack_4bytes_as_m256i::<INVERT>(b0, b1, b2, b3);
    let y512 = _mm512_cvtepu8_epi16(y256);
    _mm512_or_si512(_mm512_slli_epi16::<8>(y512), y512)
  }
}

// ---- mono1bit → RGB u8 -------------------------------------------------------

/// AVX-512 `mono1bit_to_rgb_row<INVERT>`: unpack 1-bpp → packed RGB u8.
///
/// Block size: 64 px / iter (8 input bytes). Tail: scalar.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `data.len() >= width.div_ceil(8)`.
/// `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 64 <= width {
      let b = [
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
        data[byte_idx + 4],
        data[byte_idx + 5],
        data[byte_idx + 6],
        data[byte_idx + 7],
      ];
      let y512 = unpack_8bytes_avx512::<INVERT>(b);
      // Write 4 × 16-pixel RGB blocks.
      let ptr = out.as_mut_ptr().add(x * 3);
      let y0: __m128i = _mm512_castsi512_si128(y512);
      let y1: __m128i = _mm512_extracti32x4_epi32::<1>(y512);
      let y2: __m128i = _mm512_extracti32x4_epi32::<2>(y512);
      let y3: __m128i = _mm512_extracti32x4_epi32::<3>(y512);
      write_rgb_16(y0, y0, y0, ptr);
      write_rgb_16(y1, y1, y1, ptr.add(48));
      write_rgb_16(y2, y2, y2, ptr.add(96));
      write_rgb_16(y3, y3, y3, ptr.add(144));
      x += 64;
      byte_idx += 8;
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

/// AVX-512 `mono1bit_to_rgba_row<INVERT>`: unpack 1-bpp → packed RGBA u8, α=0xFF.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 64 <= width {
      let b = [
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
        data[byte_idx + 4],
        data[byte_idx + 5],
        data[byte_idx + 6],
        data[byte_idx + 7],
      ];
      let y512 = unpack_8bytes_avx512::<INVERT>(b);
      let ptr = out.as_mut_ptr().add(x * 4);
      let y0: __m128i = _mm512_castsi512_si128(y512);
      let y1: __m128i = _mm512_extracti32x4_epi32::<1>(y512);
      let y2: __m128i = _mm512_extracti32x4_epi32::<2>(y512);
      let y3: __m128i = _mm512_extracti32x4_epi32::<3>(y512);
      write_rgba_16(y0, y0, y0, alpha128, ptr);
      write_rgba_16(y1, y1, y1, alpha128, ptr.add(64));
      write_rgba_16(y2, y2, y2, alpha128, ptr.add(128));
      write_rgba_16(y3, y3, y3, alpha128, ptr.add(192));
      x += 64;
      byte_idx += 8;
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

/// AVX-512 `mono1bit_to_luma_row<INVERT>`: unpack 1-bpp → luma u8.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 64 <= width {
      let b = [
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
        data[byte_idx + 4],
        data[byte_idx + 5],
        data[byte_idx + 6],
        data[byte_idx + 7],
      ];
      let y512 = unpack_8bytes_avx512::<INVERT>(b);
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), y512);
      x += 64;
      byte_idx += 8;
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

/// AVX-512 `mono1bit_to_rgb_u16_row<INVERT>`: unpack 1-bpp → RGB u16.
///
/// Block size: 32 px / iter (4 input bytes). Tail: scalar.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 32 <= width {
      // 4 bytes → 32 pixels → 32 u16 → 96 u16 elements of RGB.
      let y512 = expand_4bytes_to_u16x32::<INVERT>(
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
      );
      let ptr = out.as_mut_ptr().add(x * 3);
      // Write 4 × 8-pixel RGB chunks.
      let y0: __m128i = _mm512_castsi512_si128(y512);
      let y1: __m128i = _mm512_extracti32x4_epi32::<1>(y512);
      let y2: __m128i = _mm512_extracti32x4_epi32::<2>(y512);
      let y3: __m128i = _mm512_extracti32x4_epi32::<3>(y512);
      write_rgb_u16_8(y0, y0, y0, ptr);
      write_rgb_u16_8(y1, y1, y1, ptr.add(24));
      write_rgb_u16_8(y2, y2, y2, ptr.add(48));
      write_rgb_u16_8(y3, y3, y3, ptr.add(72));
      x += 32;
      byte_idx += 4;
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

/// AVX-512 `mono1bit_to_rgba_u16_row<INVERT>`: unpack 1-bpp → RGBA u16, α=0xFFFF.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let alpha128 = _mm_set1_epi16(-1i16); // 0xFFFF
    while x + 32 <= width {
      let y512 = expand_4bytes_to_u16x32::<INVERT>(
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
      );
      let ptr = out.as_mut_ptr().add(x * 4);
      let y0: __m128i = _mm512_castsi512_si128(y512);
      let y1: __m128i = _mm512_extracti32x4_epi32::<1>(y512);
      let y2: __m128i = _mm512_extracti32x4_epi32::<2>(y512);
      let y3: __m128i = _mm512_extracti32x4_epi32::<3>(y512);
      write_rgba_u16_8(y0, y0, y0, alpha128, ptr);
      write_rgba_u16_8(y1, y1, y1, alpha128, ptr.add(32));
      write_rgba_u16_8(y2, y2, y2, alpha128, ptr.add(64));
      write_rgba_u16_8(y3, y3, y3, alpha128, ptr.add(96));
      x += 32;
      byte_idx += 4;
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

/// AVX-512 `mono1bit_to_luma_u16_row<INVERT>`: unpack 1-bpp → luma u16.
///
/// Block size: 32 px / iter (4 input bytes). Tail: scalar.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 32 <= width {
      let y512 = expand_4bytes_to_u16x32::<INVERT>(
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
      );
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), y512);
      x += 32;
      byte_idx += 4;
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

/// AVX-512 `mono1bit_to_hsv_row<INVERT>`: H=0, S=0, V=Y.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. All output slices >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero = _mm512_setzero_si512();
    while x + 64 <= width {
      let b = [
        data[byte_idx],
        data[byte_idx + 1],
        data[byte_idx + 2],
        data[byte_idx + 3],
        data[byte_idx + 4],
        data[byte_idx + 5],
        data[byte_idx + 6],
        data[byte_idx + 7],
      ];
      let y512 = unpack_8bytes_avx512::<INVERT>(b);
      _mm512_storeu_si512(h.as_mut_ptr().add(x).cast(), zero);
      _mm512_storeu_si512(s.as_mut_ptr().add(x).cast(), zero);
      _mm512_storeu_si512(v.as_mut_ptr().add(x).cast(), y512);
      x += 64;
      byte_idx += 8;
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

/// Monoblack → RGB u8 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monoblack_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u8 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monoblack_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<false>(data, out, width) }
}

/// Monoblack → RGB u16 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monoblack_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u16 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monoblack_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<false>(data, out, width) }
}

/// Monoblack → Luma u8 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monoblack_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<false>(data, out, width) }
}

/// Monoblack → Luma u16 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monoblack_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<false>(data, out, width) }
}

/// Monoblack → HSV (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monoblack_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<false>(data, h, s, v, width) }
}

/// Monowhite → RGB u8 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monowhite_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u8 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monowhite_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<true>(data, out, width) }
}

/// Monowhite → RGB u16 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monowhite_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u16 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monowhite_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<true>(data, out, width) }
}

/// Monowhite → Luma u8 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monowhite_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<true>(data, out, width) }
}

/// Monowhite → Luma u16 (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monowhite_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<true>(data, out, width) }
}

/// Monowhite → HSV (AVX-512).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn monowhite_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<true>(data, h, s, v, width) }
}
