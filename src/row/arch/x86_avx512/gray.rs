//! AVX-512 (F + BW) gray kernel implementations.
//!
//! Gray → luma / luma_u16 / HSV paths use 32-pixel blocks (512-bit wide).
//! Packed-channel interleave paths (RGB, RGBA) delegate to scalar: the
//! 3/4-channel store pattern would need AVX-512-specific gathers/scatters
//! and the scalar implementations auto-vectorize well at -O3.
//!
//! # `full_range` parameter
//!
//! For RGB/RGBA/HSV kernels, `full_range = true` uses the existing fast AVX-512
//! path. `full_range = false` (limited-range) falls back to scalar since
//! limited-range rescaling is the less-common path and the scalar formulation
//! is simple and correct.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::row::{
  arch::x86_avx512::endian::{load_endian_u16x16, load_endian_u16x32, load_endian_u32x16},
  scalar::{bits_mask, gray as scalar},
};

// ---- Gray8 ------------------------------------------------------------------

/// AVX-512 `gray8_to_rgb_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray8_to_rgb_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray8_to_rgb_row(y_plane, out, width, full_range);
}

/// AVX-512 `gray8_to_rgba_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray8_to_rgba_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray8_to_rgba_row(y_plane, out, width, full_range);
}

/// AVX-512 `gray8_to_hsv_row`: H=0, S=0, V=Y. 64 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// AVX-512F+BW must be available. All planes have length >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray8_to_hsv_row(
  y_plane: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  if !full_range {
    return scalar::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let zero = _mm512_setzero_si512();
    while x + 64 <= width {
      let v = _mm512_loadu_si512(y_plane.as_ptr().add(x).cast());
      _mm512_storeu_si512(h_out.as_mut_ptr().add(x).cast(), zero);
      _mm512_storeu_si512(s_out.as_mut_ptr().add(x).cast(), zero);
      _mm512_storeu_si512(v_out.as_mut_ptr().add(x).cast(), v);
      x += 64;
    }
  }
  if x < width {
    scalar::gray8_to_hsv_row(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- GrayN (const BITS) -----------------------------------------------------

/// AVX-512 `gray_n_to_rgb_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// AVX-512 `gray_n_to_rgba_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// AVX-512 `gray_n_to_rgb_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// AVX-512 `gray_n_to_rgba_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// AVX-512 `gray_n_to_luma_row<BITS>`: mask + shift → u8. 32 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX-512F+BW must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray_n_to_luma_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = _mm512_set1_epi16(mask as i16);
    // Use variable-count `_mm512_srl_epi16` since `_mm512_srli_epi16::<IMM8>`
    // requires a literal const generic shift not expressible as `BITS - 8`.
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    while x + 32 <= width {
      let raw = load_endian_u16x32::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = _mm512_and_si512(raw, mask_v);
      // Shift right by (BITS - 8) to get u8-range value in u16
      let shifted = _mm512_srl_epi16(masked, shr);
      // Pack u16x32 → u8x32 via _mm512_cvtepi16_epi8 (AVX-512BW)
      let packed = _mm512_cvtepi16_epi8(shifted);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), packed);
      x += 32;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_row::<BITS, BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX-512 `gray_n_to_luma_u16_row<BITS>`: mask, store. 32 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX-512F+BW must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray_n_to_luma_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = _mm512_set1_epi16(mask as i16);
    while x + 32 <= width {
      let raw = load_endian_u16x32::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = _mm512_and_si512(raw, mask_v);
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), masked);
      x += 32;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_u16_row::<BITS, BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX-512 `gray_n_to_hsv_row<BITS>`: H=0, S=0, V = mask+shift. 32 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// AVX-512F+BW must be available. All slices have length >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray_n_to_hsv_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  if !full_range {
    return scalar::gray_n_to_hsv_row::<BITS, BE>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = _mm512_set1_epi16(mask as i16);
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero256 = _mm256_setzero_si256();
    while x + 32 <= width {
      let raw = load_endian_u16x32::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = _mm512_and_si512(raw, mask_v);
      let shifted = _mm512_srl_epi16(masked, shr);
      let packed = _mm512_cvtepi16_epi8(shifted);
      // H and S: 32 zero bytes; V: the packed bytes.
      _mm256_storeu_si256(h_out.as_mut_ptr().add(x).cast(), zero256);
      _mm256_storeu_si256(s_out.as_mut_ptr().add(x).cast(), zero256);
      _mm256_storeu_si256(v_out.as_mut_ptr().add(x).cast(), packed);
      x += 32;
    }
  }
  if x < width {
    scalar::gray_n_to_hsv_row::<BITS, BE>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- Gray16 -----------------------------------------------------------------

/// AVX-512 `gray16_to_rgb_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray16_to_rgb_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_row::<BE>(y_plane, out, width, full_range);
}

/// AVX-512 `gray16_to_rgba_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray16_to_rgba_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_row::<BE>(y_plane, out, width, full_range);
}

/// AVX-512 `gray16_to_rgb_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray16_to_rgb_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
}

/// AVX-512 `gray16_to_rgba_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray16_to_rgba_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
}

/// AVX-512 `gray16_to_luma_row`: `>> 8`, pack to u8. 32 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX-512F+BW must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray16_to_luma_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 32 <= width {
      let raw = load_endian_u16x32::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let shifted = _mm512_srli_epi16(raw, 8);
      let packed = _mm512_cvtepi16_epi8(shifted);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), packed);
      x += 32;
    }
  }
  if x < width {
    scalar::gray16_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX-512 `gray16_to_luma_u16_row`: identity copy. 32 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX-512F+BW must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray16_to_luma_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 32 <= width {
      let y = load_endian_u16x32::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), y);
      x += 32;
    }
  }
  if x < width {
    scalar::gray16_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX-512 `gray16_to_hsv_row`: `>> 8`, H=0, S=0, V=Y8. 32 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// AVX-512F+BW must be available. All slices have length >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray16_to_hsv_row<const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  if !full_range {
    return scalar::gray16_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let zero256 = _mm256_setzero_si256();
    while x + 32 <= width {
      let raw = load_endian_u16x32::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let shifted = _mm512_srli_epi16(raw, 8);
      let packed = _mm512_cvtepi16_epi8(shifted);
      _mm256_storeu_si256(h_out.as_mut_ptr().add(x).cast(), zero256);
      _mm256_storeu_si256(s_out.as_mut_ptr().add(x).cast(), zero256);
      _mm256_storeu_si256(v_out.as_mut_ptr().add(x).cast(), packed);
      x += 32;
    }
  }
  if x < width {
    scalar::gray16_to_hsv_row::<BE>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- Grayf32 ----------------------------------------------------------------

/// AVX-512 `grayf32_to_rgb_row`: clamp [0,1] x 255 → u8, broadcast Y → R=G=B.
///
/// Uses MXCSR-independent round-half-up: `+ 0.5` then `_mm512_cvttps_epi32`
/// (matches the scalar `(y * scale + 0.5) as T` contract). Block: 16 px.
///
/// # Safety
/// AVX-512F must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn grayf32_to_rgb_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = _mm512_set1_ps(255.0);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = _mm512_castsi512_ps(load_endian_u32x16::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm512_min_ps(_mm512_max_ps(y, _mm512_setzero_ps()), _mm512_set1_ps(1.0));
      // Round-half-up: + 0.5 then truncate (matches scalar).
      let int32 = _mm512_cvttps_epi32(_mm512_add_ps(
        _mm512_mul_ps(clamped, scale),
        _mm512_set1_ps(0.5),
      ));
      // 16xi32 → 16xu8 via saturating narrow.
      let pack8: __m128i = _mm512_cvtusepi32_epi8(int32);
      // Store 16 bytes then scatter to RGB triples.
      let mut ybuf = [0u8; 16];
      _mm_storeu_si128(ybuf.as_mut_ptr().cast(), pack8);
      for (i, &v) in ybuf.iter().enumerate() {
        let base = (x + i) * 3;
        out[base] = v;
        out[base + 1] = v;
        out[base + 2] = v;
      }
      x += 16;
    }
  }
  if x < width {
    scalar::grayf32_to_rgb_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// AVX-512 `grayf32_to_rgba_row`: clamp [0,1] x 255 → u8, broadcast + α=0xFF.
///
/// # Safety
/// AVX-512F must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn grayf32_to_rgba_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = _mm512_set1_ps(255.0);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = _mm512_castsi512_ps(load_endian_u32x16::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm512_min_ps(_mm512_max_ps(y, _mm512_setzero_ps()), _mm512_set1_ps(1.0));
      let int32 = _mm512_cvttps_epi32(_mm512_add_ps(
        _mm512_mul_ps(clamped, scale),
        _mm512_set1_ps(0.5),
      ));
      let pack8: __m128i = _mm512_cvtusepi32_epi8(int32);
      let mut ybuf = [0u8; 16];
      _mm_storeu_si128(ybuf.as_mut_ptr().cast(), pack8);
      for (i, &v) in ybuf.iter().enumerate() {
        let base = (x + i) * 4;
        out[base] = v;
        out[base + 1] = v;
        out[base + 2] = v;
        out[base + 3] = 0xFF;
      }
      x += 16;
    }
  }
  if x < width {
    scalar::grayf32_to_rgba_row::<BE>(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// AVX-512 `grayf32_to_rgb_u16_row`: clamp [0,1] x 65535 → u16, broadcast.
///
/// # Safety
/// AVX-512F must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn grayf32_to_rgb_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = _mm512_set1_ps(65535.0);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = _mm512_castsi512_ps(load_endian_u32x16::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm512_min_ps(_mm512_max_ps(y, _mm512_setzero_ps()), _mm512_set1_ps(1.0));
      // Round-to-nearest with embedded rounding.
      let int32 = _mm512_cvttps_epi32(_mm512_add_ps(
        _mm512_mul_ps(clamped, scale),
        _mm512_set1_ps(0.5),
      ));
      // 16xi32 → 16xu16 via _mm512_cvtusepi32_epi16 (saturating, but values in [0,65535]).
      let pack16: __m256i = _mm512_cvtusepi32_epi16(int32);
      // Store 16 u16 values then scatter to 3-channel output.
      let mut vbuf = [0u16; 16];
      _mm256_storeu_si256(vbuf.as_mut_ptr().cast(), pack16);
      for (i, &v) in vbuf.iter().enumerate() {
        let base = (x + i) * 3;
        out[base] = v;
        out[base + 1] = v;
        out[base + 2] = v;
      }
      x += 16;
    }
  }
  if x < width {
    scalar::grayf32_to_rgb_u16_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// AVX-512 `grayf32_to_rgba_u16_row`: clamp [0,1] x 65535 → u16, broadcast + α=0xFFFF.
///
/// # Safety
/// AVX-512F must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn grayf32_to_rgba_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = _mm512_set1_ps(65535.0);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = _mm512_castsi512_ps(load_endian_u32x16::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm512_min_ps(_mm512_max_ps(y, _mm512_setzero_ps()), _mm512_set1_ps(1.0));
      let int32 = _mm512_cvttps_epi32(_mm512_add_ps(
        _mm512_mul_ps(clamped, scale),
        _mm512_set1_ps(0.5),
      ));
      let pack16: __m256i = _mm512_cvtusepi32_epi16(int32);
      let mut vbuf = [0u16; 16];
      _mm256_storeu_si256(vbuf.as_mut_ptr().cast(), pack16);
      for (i, &v) in vbuf.iter().enumerate() {
        let base = (x + i) * 4;
        out[base] = v;
        out[base + 1] = v;
        out[base + 2] = v;
        out[base + 3] = 0xFFFF;
      }
      x += 16;
    }
  }
  if x < width {
    scalar::grayf32_to_rgba_u16_row::<BE>(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AVX-512 `grayf32_to_rgb_f32_row`: lossless replicate Y → R=G=B.
///
/// # Safety
/// AVX-512F must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn grayf32_to_rgb_f32_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [f32],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::grayf32_to_rgb_f32_row::<BE>(y_plane, out, width);
}

/// AVX-512 `grayf32_to_luma_row`: clamp [0,1] x 255 → u8. 16 px/iter.
///
/// # Safety
/// AVX-512F must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn grayf32_to_luma_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = _mm512_set1_ps(255.0);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = _mm512_castsi512_ps(load_endian_u32x16::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm512_min_ps(_mm512_max_ps(y, _mm512_setzero_ps()), _mm512_set1_ps(1.0));
      let int32 = _mm512_cvttps_epi32(_mm512_add_ps(
        _mm512_mul_ps(clamped, scale),
        _mm512_set1_ps(0.5),
      ));
      let pack8: __m128i = _mm512_cvtusepi32_epi8(int32);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), pack8);
      x += 16;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX-512 `grayf32_to_luma_u16_row`: clamp [0,1] x 65535 → u16. 16 px/iter.
///
/// # Safety
/// AVX-512F must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn grayf32_to_luma_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = _mm512_set1_ps(65535.0);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = _mm512_castsi512_ps(load_endian_u32x16::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm512_min_ps(_mm512_max_ps(y, _mm512_setzero_ps()), _mm512_set1_ps(1.0));
      let int32 = _mm512_cvttps_epi32(_mm512_add_ps(
        _mm512_mul_ps(clamped, scale),
        _mm512_set1_ps(0.5),
      ));
      let pack16: __m256i = _mm512_cvtusepi32_epi16(int32);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), pack16);
      x += 16;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX-512 `grayf32_to_luma_f32_row`: memcpy pass-through.
///
/// # Safety
/// AVX-512F must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn grayf32_to_luma_f32_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [f32],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  scalar::grayf32_to_luma_f32_row::<BE>(y_plane, out, width);
}

/// AVX-512 `grayf32_to_hsv_row`: H=0, S=0, V = clamp(Y,0,1)x255. 16 px/iter.
///
/// # Safety
/// AVX-512F must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn grayf32_to_hsv_row<const BE: bool>(
  y_plane: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  let scale = _mm512_set1_ps(255.0);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = _mm512_castsi512_ps(load_endian_u32x16::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm512_min_ps(_mm512_max_ps(y, _mm512_setzero_ps()), _mm512_set1_ps(1.0));
      let int32 = _mm512_cvttps_epi32(_mm512_add_ps(
        _mm512_mul_ps(clamped, scale),
        _mm512_set1_ps(0.5),
      ));
      let pack8: __m128i = _mm512_cvtusepi32_epi8(int32);
      let zero128 = _mm_setzero_si128();
      _mm_storeu_si128(h_out.as_mut_ptr().add(x).cast(), zero128);
      _mm_storeu_si128(s_out.as_mut_ptr().add(x).cast(), zero128);
      _mm_storeu_si128(v_out.as_mut_ptr().add(x).cast(), pack8);
      x += 16;
    }
  }
  if x < width {
    scalar::grayf32_to_hsv_row::<BE>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Ya8 -------------------------------------------------------------------

/// AVX-512 `ya8_to_rgb_row`: deinterleave [Y,A] packed u8, broadcast Y → R=G=B.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya8_to_rgb_row(packed: &[u8], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  unsafe {
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 14, 12, 10, 8, 6, 4, 2, 0,
    );
    while x + 8 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let y_bytes = _mm_shuffle_epi8(chunk, y_mask);
      let val = _mm_cvtsi128_si64(y_bytes) as u64;
      let ybuf = val.to_le_bytes();
      let base = x * 3;
      for i in 0..8usize {
        out[base + i * 3] = ybuf[i];
        out[base + i * 3 + 1] = ybuf[i];
        out[base + i * 3 + 2] = ybuf[i];
      }
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_rgb_row(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// AVX-512 `ya8_to_rgba_row`: deinterleave [Y,A], broadcast Y → R=G=B, pass α.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya8_to_rgba_row(packed: &[u8], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  unsafe {
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 14, 12, 10, 8, 6, 4, 2, 0,
    );
    let a_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 15, 13, 11, 9, 7, 5, 3, 1,
    );
    while x + 8 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let y_bytes = _mm_shuffle_epi8(chunk, y_mask);
      let a_bytes = _mm_shuffle_epi8(chunk, a_mask);
      let y_lo = _mm_cvtsi128_si64(y_bytes) as u64;
      let a_lo = _mm_cvtsi128_si64(a_bytes) as u64;
      let ybuf = y_lo.to_le_bytes();
      let abuf = a_lo.to_le_bytes();
      let base = x * 4;
      for i in 0..8usize {
        out[base + i * 4] = ybuf[i];
        out[base + i * 4 + 1] = ybuf[i];
        out[base + i * 4 + 2] = ybuf[i];
        out[base + i * 4 + 3] = abuf[i];
      }
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_rgba_row(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AVX-512 `ya8_to_rgb_u16_row`: zero-extend Y → u16, broadcast R=G=B.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya8_to_rgb_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  scalar::ya8_to_rgb_u16_row(packed, out, width);
}

/// AVX-512 `ya8_to_rgba_u16_row`: zero-extend Y and A → u16.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya8_to_rgba_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  scalar::ya8_to_rgba_u16_row(packed, out, width);
}

/// AVX-512 `ya8_to_luma_row`: extract Y bytes. 8 px/iter.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya8_to_luma_row(packed: &[u8], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 14, 12, 10, 8, 6, 4, 2, 0,
    );
    while x + 8 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let y_bytes = _mm_shuffle_epi8(chunk, y_mask);
      let val = _mm_cvtsi128_si64(y_bytes) as u64;
      out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_luma_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// AVX-512 `ya8_to_luma_u16_row`: zero-extend Y → u16.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya8_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  scalar::ya8_to_luma_u16_row(packed, out, width);
}

/// AVX-512 `ya8_to_hsv_row`: H=0, S=0, V=Y. α dropped.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya8_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  let mut x = 0usize;
  unsafe {
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 14, 12, 10, 8, 6, 4, 2, 0,
    );
    while x + 8 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let y_bytes = _mm_shuffle_epi8(chunk, y_mask);
      let val = _mm_cvtsi128_si64(y_bytes) as u64;
      let vbytes = val.to_le_bytes();
      h_out[x..x + 8].fill(0);
      s_out[x..x + 8].fill(0);
      v_out[x..x + 8].copy_from_slice(&vbytes);
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_hsv_row(
      &packed[x * 2..width * 2],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Ya16 ------------------------------------------------------------------

/// Host-endian gate for Ya16 SIMD bodies.
///
/// The AVX-512 Ya16 SIMD bodies use `_mm512_loadu_si512` + fixed shuffle masks
/// that gather the **host-native** high byte of each Ya16 word. They are only
/// correct when the encoded byte order matches the host. Truth table:
///
/// | data BE | host BE | `BE != HOST_NATIVE_BE` | path   | correct via       |
/// |---------|---------|------------------------|--------|-------------------|
/// | false   | false   | false                  | SIMD   | host-native LE    |
/// | false   | true    | true                   | scalar | `from_le`         |
/// | true    | false   | true                   | scalar | `from_be`         |
/// | true    | true    | false                  | SIMD   | host-native BE    |
///
/// The narrower `if BE { scalar }` gate from `7cb64c6` only covered rows 1+3
/// (LE host); the LE-source-on-BE-host quadrant would still run the SIMD body
/// and corrupt Y/A. This constant + comparison covers all 4 quadrants.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// AVX-512 `ya16_to_rgb_row`: deinterleave [Y,A] u16, Y `>> 8` → u8, broadcast.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya16_to_rgb_row<const BE: bool>(packed: &[u16], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  if BE != HOST_NATIVE_BE {
    // Source byte order differs from host → SIMD host-native shuffle wrong; fall through.
    return scalar::ya16_to_rgb_row::<BE>(packed, out, width);
  }
  let mut x = 0usize;
  unsafe {
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 13, 12, 9, 8, 5, 4, 1, 0,
    );
    while x + 4 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast::<__m128i>());
      let y_words = _mm_shuffle_epi8(chunk, y_mask);
      let y_shifted = _mm_srli_epi16(y_words, 8);
      let pack8 = _mm_packus_epi16(y_shifted, _mm_setzero_si128());
      let val = _mm_cvtsi128_si32(pack8) as u32;
      let ybuf = val.to_le_bytes();
      let base = x * 3;
      for i in 0..4usize {
        out[base + i * 3] = ybuf[i];
        out[base + i * 3 + 1] = ybuf[i];
        out[base + i * 3 + 2] = ybuf[i];
      }
      x += 4;
    }
  }
  if x < width {
    scalar::ya16_to_rgb_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// AVX-512 `ya16_to_rgba_row`: Y `>> 8`, A `>> 8`, broadcast Y to R=G=B.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya16_to_rgba_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  if BE != HOST_NATIVE_BE {
    // Source byte order differs from host → SIMD host-native shuffle wrong; fall through.
    return scalar::ya16_to_rgba_row::<BE>(packed, out, width);
  }
  let mut x = 0usize;
  unsafe {
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 13, 12, 9, 8, 5, 4, 1, 0,
    );
    let a_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 15, 14, 11, 10, 7, 6, 3, 2,
    );
    while x + 4 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast::<__m128i>());
      let y_words = _mm_shuffle_epi8(chunk, y_mask);
      let a_words = _mm_shuffle_epi8(chunk, a_mask);
      let y_shifted = _mm_srli_epi16(y_words, 8);
      let a_shifted = _mm_srli_epi16(a_words, 8);
      let zero = _mm_setzero_si128();
      let y8 = _mm_packus_epi16(y_shifted, zero);
      let a8 = _mm_packus_epi16(a_shifted, zero);
      let yval = _mm_cvtsi128_si32(y8) as u32;
      let aval = _mm_cvtsi128_si32(a8) as u32;
      let ybuf = yval.to_le_bytes();
      let abuf = aval.to_le_bytes();
      let base = x * 4;
      for i in 0..4usize {
        out[base + i * 4] = ybuf[i];
        out[base + i * 4 + 1] = ybuf[i];
        out[base + i * 4 + 2] = ybuf[i];
        out[base + i * 4 + 3] = abuf[i];
      }
      x += 4;
    }
  }
  if x < width {
    scalar::ya16_to_rgba_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AVX-512 `ya16_to_rgb_u16_row`: native Y u16, broadcast R=G=B.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya16_to_rgb_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  scalar::ya16_to_rgb_u16_row::<BE>(packed, out, width);
}

/// AVX-512 `ya16_to_rgba_u16_row`: native Y and A u16.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya16_to_rgba_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  scalar::ya16_to_rgba_u16_row::<BE>(packed, out, width);
}

/// AVX-512 `ya16_to_luma_row`: Y `>> 8` → u8. 4 px/iter.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya16_to_luma_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  if BE != HOST_NATIVE_BE {
    // Source byte order differs from host → SIMD host-native shuffle wrong; fall through.
    return scalar::ya16_to_luma_row::<BE>(packed, out, width);
  }
  let mut x = 0usize;
  unsafe {
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 13, 12, 9, 8, 5, 4, 1, 0,
    );
    while x + 4 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast::<__m128i>());
      let y_words = _mm_shuffle_epi8(chunk, y_mask);
      let y_shifted = _mm_srli_epi16(y_words, 8);
      let pack8 = _mm_packus_epi16(y_shifted, _mm_setzero_si128());
      let val = _mm_cvtsi128_si32(pack8) as u32;
      out[x..x + 4].copy_from_slice(&val.to_le_bytes());
      x += 4;
    }
  }
  if x < width {
    scalar::ya16_to_luma_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// AVX-512 `ya16_to_luma_u16_row`: native Y pass-through.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya16_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  scalar::ya16_to_luma_u16_row::<BE>(packed, out, width);
}

/// AVX-512 `ya16_to_hsv_row`: H=0, S=0, V = Y `>> 8`. α dropped.
///
/// # Safety
/// AVX-512F+BW must be available.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn ya16_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  if BE != HOST_NATIVE_BE {
    // Source byte order differs from host → SIMD host-native shuffle wrong; fall through.
    return scalar::ya16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
  }
  let mut x = 0usize;
  unsafe {
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 13, 12, 9, 8, 5, 4, 1, 0,
    );
    while x + 4 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast::<__m128i>());
      let y_words = _mm_shuffle_epi8(chunk, y_mask);
      let y_shifted = _mm_srli_epi16(y_words, 8);
      let pack8 = _mm_packus_epi16(y_shifted, _mm_setzero_si128());
      let val = _mm_cvtsi128_si32(pack8) as u32;
      let vbytes = val.to_le_bytes();
      h_out[x..x + 4].fill(0);
      s_out[x..x + 4].fill(0);
      v_out[x..x + 4].copy_from_slice(&vbytes);
      x += 4;
    }
  }
  if x < width {
    scalar::ya16_to_hsv_row::<BE>(
      &packed[x * 2..width * 2],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Grayf16 ----------------------------------------------------------------
//
// Strategy: widen a chunk of `half::f16` luma to `f32` into a stack buffer with
// F16C + AVX-512F (`_mm512_cvtph_ps`), then delegate to the existing AVX-512
// `grayf32` downstream kernels with `HOST_NATIVE_BE` (the widened buffer is
// host-native). The half-float twin of the `grayf32` AVX-512 kernels; the f16
// reading mirrors the Rgbf16 AVX-512 path (`load_endian_u16x16` + `_mm512_cvtph_ps`).

/// `BE` value that makes the `grayf32` row kernels treat their input as
/// host-native (no-op swap) after the F16C widen produced host-native f32.
const GRAYF16_HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Widen 16 x f16 (32 bytes at `ptr`) to `out[0..16]` (host-native f32).
/// For `BE = true` the f16 bits are byte-swapped before widening.
///
/// # Safety
/// AVX-512F + F16C must be available. `ptr` valid for 32 bytes; `out` for 16 f32.
#[inline]
#[target_feature(enable = "avx512f,f16c")]
unsafe fn widen_f16x16_avx512_buf<const BE: bool>(ptr: *const half::f16, out: *mut f32) {
  unsafe {
    let m = _mm512_cvtph_ps(load_endian_u16x16::<BE>(ptr.cast::<u8>()));
    _mm512_storeu_ps(out, m);
  }
}

/// AVX-512 `grayf16_to_rgb_row`: widen f16 → f32, clamp [0,1] x 255 → u8, broadcast.
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn grayf16_to_rgb_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 16];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_rgb_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x * 3..(x + 16) * 3], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::grayf16_to_rgb_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// AVX-512 `grayf16_to_rgba_row`: widen f16 → f32, clamp [0,1] x 255, broadcast, α=0xFF.
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn grayf16_to_rgba_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 16];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_rgba_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x * 4..(x + 16) * 4], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::grayf16_to_rgba_row::<BE>(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// AVX-512 `grayf16_to_rgb_u16_row`: widen f16 → f32, clamp [0,1] x 65535 → u16, broadcast.
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn grayf16_to_rgb_u16_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 16];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_rgb_u16_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x * 3..(x + 16) * 3], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::grayf16_to_rgb_u16_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// AVX-512 `grayf16_to_rgba_u16_row`: widen f16 → f32, clamp [0,1] x 65535, broadcast, α=0xFFFF.
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn grayf16_to_rgba_u16_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 16];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_rgba_u16_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x * 4..(x + 16) * 4], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::grayf16_to_rgba_u16_row::<BE>(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AVX-512 `grayf16_to_luma_row`: widen f16 → f32, clamp [0,1] x 255 → u8 luma.
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn grayf16_to_luma_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 16];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_luma_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x..x + 16], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::grayf16_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX-512 `grayf16_to_luma_u16_row`: widen f16 → f32, clamp [0,1] x 65535 → u16 luma.
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn grayf16_to_luma_u16_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 16];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_luma_u16_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x..x + 16], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::grayf16_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX-512 `grayf16_to_hsv_row`: H=0, S=0, V = clamp(widen(Y),0,1) x 255.
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `y_plane.len() >= width`; H/S/V out `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn grayf16_to_hsv_row<const BE: bool>(
  y_plane: &[half::f16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 16];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_hsv_row::<GRAYF16_HOST_NATIVE_BE>(
        &buf,
        &mut h_out[x..x + 16],
        &mut s_out[x..x + 16],
        &mut v_out[x..x + 16],
        16,
      );
    }
    x += 16;
  }
  if x < width {
    scalar::grayf16_to_hsv_row::<BE>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Yaf32 ------------------------------------------------------------------
//
// Packed `[Y, A]` f32 source. Each 16-pixel chunk deinterleaves Y (and A for
// RGBA outputs) with 128-bit `_mm_shuffle_ps` (lane-crossing-free) into a host
// -native f32 stack buffer, then delegates to the proven `grayf32` AVX-512
// kernels for the clamp / scale / round math (Y broadcast R=G=B; A patched into
// the RGBA α channel via `grayf32_to_luma*`). Like the `ya16` path, the
// host-native deinterleave is only correct when the source byte order matches
// the host, so `BE != HOST_NATIVE_BE` falls through to scalar.

/// Deinterleave `n` (multiple of 4) Y elements from packed `[Y, A]` f32 into
/// `ybuf` via 128-bit `_mm_shuffle_ps`. Host-native (caller guards byte order).
///
/// # Safety
/// AVX-512F must be available. `packed` valid for `n * 2` f32; `ybuf` for `n` f32.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn yaf32_deint_y_avx512(packed: *const f32, ybuf: *mut f32, n: usize) {
  unsafe {
    let mut i = 0usize;
    while i + 4 <= n {
      let ya0 = _mm_loadu_ps(packed.add(i * 2));
      let ya1 = _mm_loadu_ps(packed.add(i * 2 + 4));
      _mm_storeu_ps(ybuf.add(i), _mm_shuffle_ps::<0x88>(ya0, ya1));
      i += 4;
    }
  }
}

/// Deinterleave `n` (multiple of 4) A elements from packed `[Y, A]` f32 into
/// `abuf` via 128-bit `_mm_shuffle_ps`. Host-native (caller guards byte order).
///
/// # Safety
/// AVX-512F must be available. `packed` valid for `n * 2` f32; `abuf` for `n` f32.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn yaf32_deint_a_avx512(packed: *const f32, abuf: *mut f32, n: usize) {
  unsafe {
    let mut i = 0usize;
    while i + 4 <= n {
      let ya0 = _mm_loadu_ps(packed.add(i * 2));
      let ya1 = _mm_loadu_ps(packed.add(i * 2 + 4));
      _mm_storeu_ps(abuf.add(i), _mm_shuffle_ps::<0xDD>(ya0, ya1));
      i += 4;
    }
  }
}

/// AVX-512 `yaf32_to_rgb_row`: deinterleave `[Y,A]` f32, clamp Y [0,1] x 255 → u8, broadcast.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yaf32_to_rgb_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf32 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  if BE != HOST_NATIVE_BE {
    return scalar::yaf32_to_rgb_row::<BE>(packed, out, width);
  }
  let mut x = 0usize;
  while x + 16 <= width {
    let mut ybuf = [0.0f32; 16];
    unsafe {
      yaf32_deint_y_avx512(packed.as_ptr().add(x * 2), ybuf.as_mut_ptr(), 16);
      grayf32_to_rgb_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x * 3..(x + 16) * 3], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf32_to_rgb_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// AVX-512 `yaf32_to_rgba_row`: clamp Y x 255 broadcast, α = clamp(A) x 255.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `packed.len() >= width * 2`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yaf32_to_rgba_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf32 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  if BE != HOST_NATIVE_BE {
    return scalar::yaf32_to_rgba_row::<BE>(packed, out, width);
  }
  let mut x = 0usize;
  while x + 16 <= width {
    let mut ybuf = [0.0f32; 16];
    let mut abuf = [0.0f32; 16];
    let mut a8 = [0u8; 16];
    unsafe {
      yaf32_deint_y_avx512(packed.as_ptr().add(x * 2), ybuf.as_mut_ptr(), 16);
      yaf32_deint_a_avx512(packed.as_ptr().add(x * 2), abuf.as_mut_ptr(), 16);
      grayf32_to_rgba_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x * 4..(x + 16) * 4], 16);
      grayf32_to_luma_row::<HOST_NATIVE_BE>(&abuf, &mut a8, 16);
    }
    for i in 0..16 {
      out[(x + i) * 4 + 3] = a8[i];
    }
    x += 16;
  }
  if x < width {
    scalar::yaf32_to_rgba_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AVX-512 `yaf32_to_rgb_u16_row`: clamp Y [0,1] x 65535 → u16, broadcast.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yaf32_to_rgb_u16_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::yaf32 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  if BE != HOST_NATIVE_BE {
    return scalar::yaf32_to_rgb_u16_row::<BE>(packed, out, width);
  }
  let mut x = 0usize;
  while x + 16 <= width {
    let mut ybuf = [0.0f32; 16];
    unsafe {
      yaf32_deint_y_avx512(packed.as_ptr().add(x * 2), ybuf.as_mut_ptr(), 16);
      grayf32_to_rgb_u16_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x * 3..(x + 16) * 3], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf32_to_rgb_u16_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// AVX-512 `yaf32_to_rgba_u16_row`: clamp Y x 65535 broadcast, α = clamp(A) x 65535.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `packed.len() >= width * 2`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yaf32_to_rgba_u16_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::yaf32 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  if BE != HOST_NATIVE_BE {
    return scalar::yaf32_to_rgba_u16_row::<BE>(packed, out, width);
  }
  let mut x = 0usize;
  while x + 16 <= width {
    let mut ybuf = [0.0f32; 16];
    let mut abuf = [0.0f32; 16];
    let mut a16 = [0u16; 16];
    unsafe {
      yaf32_deint_y_avx512(packed.as_ptr().add(x * 2), ybuf.as_mut_ptr(), 16);
      yaf32_deint_a_avx512(packed.as_ptr().add(x * 2), abuf.as_mut_ptr(), 16);
      grayf32_to_rgba_u16_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x * 4..(x + 16) * 4], 16);
      grayf32_to_luma_u16_row::<HOST_NATIVE_BE>(&abuf, &mut a16, 16);
    }
    for i in 0..16 {
      out[(x + i) * 4 + 3] = a16[i];
    }
    x += 16;
  }
  if x < width {
    scalar::yaf32_to_rgba_u16_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AVX-512 `yaf32_to_luma_row`: clamp Y [0,1] x 255 → u8 luma.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yaf32_to_luma_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf32 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  if BE != HOST_NATIVE_BE {
    return scalar::yaf32_to_luma_row::<BE>(packed, out, width);
  }
  let mut x = 0usize;
  while x + 16 <= width {
    let mut ybuf = [0.0f32; 16];
    unsafe {
      yaf32_deint_y_avx512(packed.as_ptr().add(x * 2), ybuf.as_mut_ptr(), 16);
      grayf32_to_luma_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x..x + 16], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf32_to_luma_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// AVX-512 `yaf32_to_luma_u16_row`: clamp Y [0,1] x 65535 → u16 luma.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yaf32_to_luma_u16_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::yaf32 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  if BE != HOST_NATIVE_BE {
    return scalar::yaf32_to_luma_u16_row::<BE>(packed, out, width);
  }
  let mut x = 0usize;
  while x + 16 <= width {
    let mut ybuf = [0.0f32; 16];
    unsafe {
      yaf32_deint_y_avx512(packed.as_ptr().add(x * 2), ybuf.as_mut_ptr(), 16);
      grayf32_to_luma_u16_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x..x + 16], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf32_to_luma_u16_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// AVX-512 `yaf32_to_hsv_row`: H=0, S=0, V = clamp(Y,0,1) x 255. α dropped.
///
/// # Safety
/// AVX-512F + AVX-512BW must be available. `packed.len() >= width * 2`; H/S/V out `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn yaf32_to_hsv_row<const BE: bool>(
  packed: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf32 as scalar;
  debug_assert!(packed.len() >= width * 2);
  if BE != HOST_NATIVE_BE {
    return scalar::yaf32_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
  }
  let mut x = 0usize;
  while x + 16 <= width {
    let mut ybuf = [0.0f32; 16];
    unsafe {
      yaf32_deint_y_avx512(packed.as_ptr().add(x * 2), ybuf.as_mut_ptr(), 16);
      grayf32_to_hsv_row::<HOST_NATIVE_BE>(
        &ybuf,
        &mut h_out[x..x + 16],
        &mut s_out[x..x + 16],
        &mut v_out[x..x + 16],
        16,
      );
    }
    x += 16;
  }
  if x < width {
    scalar::yaf32_to_hsv_row::<BE>(
      &packed[x * 2..width * 2],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Yaf16 ------------------------------------------------------------------
//
// Widen each 16-pixel chunk of packed `[Y, A]` f16 (32 f16) to a host-native
// f32 stack buffer with the F16C `_mm512_cvtph_ps` (`widen_f16x16_avx512_buf`),
// then delegate to the `yaf32` AVX-512 kernels with `HOST_NATIVE_BE`. The
// half-float twin of the `yaf32` AVX-512 path.

/// AVX-512 `yaf16_to_rgb_row`: widen `[Y,A]` f16 → f32, clamp Y x 255 → u8, broadcast.
///
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn yaf16_to_rgb_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 32];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2 + 16), buf.as_mut_ptr().add(16));
      yaf32_to_rgb_row::<HOST_NATIVE_BE>(&buf, &mut out[x * 3..(x + 16) * 3], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf16_to_rgb_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// AVX-512 `yaf16_to_rgba_row`: widen `[Y,A]` f16 → f32, clamp Y x 255 broadcast, α = clamp(A) x 255.
///
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn yaf16_to_rgba_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 32];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2 + 16), buf.as_mut_ptr().add(16));
      yaf32_to_rgba_row::<HOST_NATIVE_BE>(&buf, &mut out[x * 4..(x + 16) * 4], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf16_to_rgba_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AVX-512 `yaf16_to_rgb_u16_row`: widen `[Y,A]` f16 → f32, clamp Y x 65535 → u16, broadcast.
///
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn yaf16_to_rgb_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 32];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2 + 16), buf.as_mut_ptr().add(16));
      yaf32_to_rgb_u16_row::<HOST_NATIVE_BE>(&buf, &mut out[x * 3..(x + 16) * 3], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf16_to_rgb_u16_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// AVX-512 `yaf16_to_rgba_u16_row`: widen `[Y,A]` f16 → f32, clamp Y x 65535 broadcast, α = clamp(A) x 65535.
///
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn yaf16_to_rgba_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 32];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2 + 16), buf.as_mut_ptr().add(16));
      yaf32_to_rgba_u16_row::<HOST_NATIVE_BE>(&buf, &mut out[x * 4..(x + 16) * 4], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf16_to_rgba_u16_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AVX-512 `yaf16_to_luma_row`: widen `[Y,A]` f16 → f32, clamp Y x 255 → u8 luma.
///
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn yaf16_to_luma_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 32];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2 + 16), buf.as_mut_ptr().add(16));
      yaf32_to_luma_row::<HOST_NATIVE_BE>(&buf, &mut out[x..x + 16], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf16_to_luma_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// AVX-512 `yaf16_to_luma_u16_row`: widen `[Y,A]` f16 → f32, clamp Y x 65535 → u16 luma.
///
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn yaf16_to_luma_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 32];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2 + 16), buf.as_mut_ptr().add(16));
      yaf32_to_luma_u16_row::<HOST_NATIVE_BE>(&buf, &mut out[x..x + 16], 16);
    }
    x += 16;
  }
  if x < width {
    scalar::yaf16_to_luma_u16_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// AVX-512 `yaf16_to_hsv_row`: widen `[Y,A]` f16 → f32, H=0, S=0, V = clamp(Y,0,1) x 255. α dropped.
///
/// # Safety
/// AVX-512F + AVX-512BW + F16C must be available. `packed.len() >= width * 2`; H/S/V out `>= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn yaf16_to_hsv_row<const BE: bool>(
  packed: &[half::f16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  let mut x = 0usize;
  while x + 16 <= width {
    let mut buf = [0.0f32; 32];
    unsafe {
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x16_avx512_buf::<BE>(packed.as_ptr().add(x * 2 + 16), buf.as_mut_ptr().add(16));
      yaf32_to_hsv_row::<HOST_NATIVE_BE>(
        &buf,
        &mut h_out[x..x + 16],
        &mut s_out[x..x + 16],
        &mut v_out[x..x + 16],
        16,
      );
    }
    x += 16;
  }
  if x < width {
    scalar::yaf16_to_hsv_row::<BE>(
      &packed[x * 2..width * 2],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}
