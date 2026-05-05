//! SSE4.1 gray kernel implementations.
//!
//! Gray output kernels (broadcast, HSV H=S=0 V=Y, depth-shift) don't
//! have complex Q15 chroma math — the bottleneck is memory bandwidth.
//! The scalar kernels already auto-vectorize well with -O3; here we
//! provide explicit SSE4.1 versions that use `_mm_loadu_si128` + store
//! patterns and delegate to scalar for tail handling.
//!
//! # `full_range` parameter
//!
//! For RGB/RGBA/HSV kernels, `full_range = true` uses the existing fast SSE4.1
//! path. `full_range = false` (limited-range) falls back to scalar since
//! limited-range rescaling is the less-common path and the scalar formulation
//! is simple and correct.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::x86_64::*;

use crate::row::scalar::{bits_mask, gray as scalar};

// ---- Gray8 ------------------------------------------------------------------

/// SSE4.1 `gray8_to_rgb_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray8_to_rgb_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  // SSE4.1 doesn't have a 3-channel interleave store like NEON's vst3q_u8.
  // Use scalar (which auto-vectorizes) for the whole row here, or implement
  // manually with repeated shuffle. We delegate to scalar to stay correct and
  // simple; the dispatch will auto-promote to AVX2 when available.
  scalar::gray8_to_rgb_row(y_plane, out, width, full_range);
}

/// SSE4.1 `gray8_to_rgba_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray8_to_rgba_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  // SSE4.1 4-channel interleave without SSSE3 shuffle tables is verbose;
  // delegate to scalar (which auto-vectorizes well at -O3).
  scalar::gray8_to_rgba_row(y_plane, out, width, full_range);
}

/// SSE4.1 `gray8_to_hsv_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
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
  // H and S planes: memset 0. V plane: memcpy Y.
  // SSE4.1 can do 16-byte stores efficiently.
  let mut x = 0usize;
  unsafe {
    let zero = _mm_setzero_si128();
    while x + 16 <= width {
      let v = _mm_loadu_si128(y_plane.as_ptr().add(x).cast());
      _mm_storeu_si128(h_out.as_mut_ptr().add(x).cast(), zero);
      _mm_storeu_si128(s_out.as_mut_ptr().add(x).cast(), zero);
      _mm_storeu_si128(v_out.as_mut_ptr().add(x).cast(), v);
      x += 16;
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

// ---- GrayN (const BITS) ------------------------------------------------

/// SSE4.1 `gray_n_to_rgb_row<BITS>`: mask, shift to u8, scalar store.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray_n_to_rgb_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_row::<BITS>(y_plane, out, width, full_range);
}

/// SSE4.1 `gray_n_to_rgba_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray_n_to_rgba_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  // SSE4.1 4-channel interleave without SSSE3 shuffle tables is complex;
  // delegate to scalar (which auto-vectorizes well at -O3).
  scalar::gray_n_to_rgba_row::<BITS>(y_plane, out, width, full_range);
}

/// SSE4.1 `gray_n_to_rgb_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray_n_to_rgb_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_u16_row::<BITS>(y_plane, out, width, full_range);
}

/// SSE4.1 `gray_n_to_rgba_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray_n_to_rgba_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray_n_to_rgba_u16_row::<BITS>(y_plane, out, width, full_range);
}

/// SSE4.1 `gray_n_to_luma_row<BITS>`: mask, shift, pack, store.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray_n_to_luma_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = _mm_set1_epi16(mask as i16);
    // `_mm_srli_epi16::<IMM8>` requires a literal const generic shift, which
    // is not available for `BITS - 8` on stable Rust.  Use the variable-count
    // variant `_mm_srl_epi16` with a count vector built from the shift amount.
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    while x + 8 <= width {
      let raw = _mm_loadu_si128(y_plane.as_ptr().add(x).cast());
      let masked = _mm_and_si128(raw, mask_v);
      let shifted = _mm_srl_epi16(masked, shr);
      let zero = _mm_setzero_si128();
      let packed = _mm_packus_epi16(shifted, zero); // u8x16 low 8 valid
      // Store 8 bytes
      let val = _mm_cvtsi128_si64(packed) as u64;
      let bytes = val.to_le_bytes();
      out[x..x + 8].copy_from_slice(&bytes);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_row::<BITS>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `gray_n_to_luma_u16_row<BITS>`: mask, store.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray_n_to_luma_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = _mm_set1_epi16(mask as i16);
    while x + 8 <= width {
      let raw = _mm_loadu_si128(y_plane.as_ptr().add(x).cast());
      let masked = _mm_and_si128(raw, mask_v);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), masked);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_u16_row::<BITS>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `gray_n_to_hsv_row<BITS>`: H=0, S=0, V = mask+shift.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray_n_to_hsv_row<const BITS: u32>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  if !full_range {
    return scalar::gray_n_to_hsv_row::<BITS>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = _mm_set1_epi16(mask as i16);
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero = _mm_setzero_si128();
    while x + 8 <= width {
      let raw = _mm_loadu_si128(y_plane.as_ptr().add(x).cast());
      let masked = _mm_and_si128(raw, mask_v);
      let shifted = _mm_srl_epi16(masked, shr);
      let packed = _mm_packus_epi16(shifted, zero);
      let val = _mm_cvtsi128_si64(packed) as u64;
      let bytes = val.to_le_bytes();
      // H and S: 8 zero bytes; V: the packed bytes.
      h_out[x..x + 8].fill(0);
      s_out[x..x + 8].fill(0);
      v_out[x..x + 8].copy_from_slice(&bytes);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_hsv_row::<BITS>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- Gray16 ----------------------------------------------------------

/// SSE4.1 `gray16_to_rgb_row`: `>> 8` → pack → scatter (scalar fallback).
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_rgb_row(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_row(y_plane, out, width, full_range);
}

/// SSE4.1 `gray16_to_rgba_row`: `>> 8` → RGBA u8.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_rgba_row(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_row(y_plane, out, width, full_range);
}

/// SSE4.1 `gray16_to_rgb_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_rgb_u16_row(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_u16_row(y_plane, out, width, full_range);
}

/// SSE4.1 `gray16_to_rgba_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_rgba_u16_row(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_u16_row(y_plane, out, width, full_range);
}

/// SSE4.1 `gray16_to_luma_row`: `>> 8`, pack, store.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_luma_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    let zero = _mm_setzero_si128();
    while x + 8 <= width {
      let raw = _mm_loadu_si128(y_plane.as_ptr().add(x).cast());
      let shifted = _mm_srli_epi16(raw, 8);
      let packed = _mm_packus_epi16(shifted, zero);
      let val = _mm_cvtsi128_si64(packed) as u64;
      out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_luma_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `gray16_to_luma_u16_row`: identity copy via SSE4.1 stores.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_luma_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = _mm_loadu_si128(y_plane.as_ptr().add(x).cast());
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_luma_u16_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `gray16_to_hsv_row`: `>> 8`, H=0, S=0, V=Y8.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_hsv_row(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  if !full_range {
    return scalar::gray16_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let zero16 = _mm_setzero_si128();
    while x + 8 <= width {
      let raw = _mm_loadu_si128(y_plane.as_ptr().add(x).cast());
      let shifted = _mm_srli_epi16(raw, 8);
      let packed = _mm_packus_epi16(shifted, zero16);
      let val = _mm_cvtsi128_si64(packed) as u64;
      h_out[x..x + 8].fill(0);
      s_out[x..x + 8].fill(0);
      v_out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_hsv_row(
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

/// SSE4.1 `grayf32_to_rgb_row`: clamp [0,1] × 255 → u8, broadcast Y → R=G=B.
///
/// Uses MXCSR-independent rounding: `_mm_round_ps` + `_mm_cvttps_epi32`.
/// Block size: 4 px / iter.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgb_row(y_plane: &[f32], out: &mut [u8], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = _mm_set1_ps(255.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_loadu_ps(y_plane.as_ptr().add(x));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
      let int32 = _mm_cvttps_epi32(rounded);
      let pack16 = _mm_packs_epi32(int32, int32);
      let pack8 = _mm_packus_epi16(pack16, pack16);
      // Extract 4 bytes and scatter to RGB triples.
      let v0 = _mm_extract_epi8(pack8, 0) as u8;
      let v1 = _mm_extract_epi8(pack8, 1) as u8;
      let v2 = _mm_extract_epi8(pack8, 2) as u8;
      let v3 = _mm_extract_epi8(pack8, 3) as u8;
      let base = x * 3;
      out[base] = v0;
      out[base + 1] = v0;
      out[base + 2] = v0;
      out[base + 3] = v1;
      out[base + 4] = v1;
      out[base + 5] = v1;
      out[base + 6] = v2;
      out[base + 7] = v2;
      out[base + 8] = v2;
      out[base + 9] = v3;
      out[base + 10] = v3;
      out[base + 11] = v3;
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_rgb_row(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// SSE4.1 `grayf32_to_rgba_row`: clamp [0,1] × 255 → u8, broadcast + α=0xFF.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgba_row(y_plane: &[f32], out: &mut [u8], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = _mm_set1_ps(255.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_loadu_ps(y_plane.as_ptr().add(x));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
      let int32 = _mm_cvttps_epi32(rounded);
      let pack16 = _mm_packs_epi32(int32, int32);
      let pack8 = _mm_packus_epi16(pack16, pack16);
      let v0 = _mm_extract_epi8(pack8, 0) as u8;
      let v1 = _mm_extract_epi8(pack8, 1) as u8;
      let v2 = _mm_extract_epi8(pack8, 2) as u8;
      let v3 = _mm_extract_epi8(pack8, 3) as u8;
      let base = x * 4;
      out[base] = v0;
      out[base + 1] = v0;
      out[base + 2] = v0;
      out[base + 3] = 0xFF;
      out[base + 4] = v1;
      out[base + 5] = v1;
      out[base + 6] = v1;
      out[base + 7] = 0xFF;
      out[base + 8] = v2;
      out[base + 9] = v2;
      out[base + 10] = v2;
      out[base + 11] = 0xFF;
      out[base + 12] = v3;
      out[base + 13] = v3;
      out[base + 14] = v3;
      out[base + 15] = 0xFF;
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_rgba_row(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// SSE4.1 `grayf32_to_rgb_u16_row`: clamp [0,1] × 65535 → u16, broadcast.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgb_u16_row(y_plane: &[f32], out: &mut [u16], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = _mm_set1_ps(65535.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_loadu_ps(y_plane.as_ptr().add(x));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
      let int32 = _mm_cvttps_epi32(rounded);
      // Extract 4 u16 values (unrolled — _mm_extract_epi32 needs const lane).
      let base = x * 3;
      let v0 = _mm_extract_epi32::<0>(int32) as u16;
      let v1 = _mm_extract_epi32::<1>(int32) as u16;
      let v2 = _mm_extract_epi32::<2>(int32) as u16;
      let v3 = _mm_extract_epi32::<3>(int32) as u16;
      out[base] = v0;
      out[base + 1] = v0;
      out[base + 2] = v0;
      out[base + 3] = v1;
      out[base + 4] = v1;
      out[base + 5] = v1;
      out[base + 6] = v2;
      out[base + 7] = v2;
      out[base + 8] = v2;
      out[base + 9] = v3;
      out[base + 10] = v3;
      out[base + 11] = v3;
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_rgb_u16_row(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// SSE4.1 `grayf32_to_rgba_u16_row`: clamp [0,1] × 65535 → u16, broadcast + α=0xFFFF.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgba_u16_row(y_plane: &[f32], out: &mut [u16], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = _mm_set1_ps(65535.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_loadu_ps(y_plane.as_ptr().add(x));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
      let int32 = _mm_cvttps_epi32(rounded);
      let base = x * 4;
      let v0 = _mm_extract_epi32::<0>(int32) as u16;
      let v1 = _mm_extract_epi32::<1>(int32) as u16;
      let v2 = _mm_extract_epi32::<2>(int32) as u16;
      let v3 = _mm_extract_epi32::<3>(int32) as u16;
      out[base] = v0;
      out[base + 1] = v0;
      out[base + 2] = v0;
      out[base + 3] = 0xFFFF;
      out[base + 4] = v1;
      out[base + 5] = v1;
      out[base + 6] = v1;
      out[base + 7] = 0xFFFF;
      out[base + 8] = v2;
      out[base + 9] = v2;
      out[base + 10] = v2;
      out[base + 11] = 0xFFFF;
      out[base + 12] = v3;
      out[base + 13] = v3;
      out[base + 14] = v3;
      out[base + 15] = 0xFFFF;
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_rgba_u16_row(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// SSE4.1 `grayf32_to_rgb_f32_row`: lossless replicate Y → R=G=B.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgb_f32_row(y_plane: &[f32], out: &mut [f32], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  // f32 triplet broadcast: scalar is already optimal here.
  scalar::grayf32_to_rgb_f32_row(y_plane, out, width);
}

/// SSE4.1 `grayf32_to_luma_row`: clamp [0,1] × 255 → u8.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_luma_row(y_plane: &[f32], out: &mut [u8], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = _mm_set1_ps(255.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_loadu_ps(y_plane.as_ptr().add(x));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
      let int32 = _mm_cvttps_epi32(rounded);
      let pack16 = _mm_packs_epi32(int32, int32);
      let pack8 = _mm_packus_epi16(pack16, pack16);
      // Store 4 bytes: low 4 of pack8.
      let val = _mm_cvtsi128_si32(pack8) as u32;
      out[x..x + 4].copy_from_slice(&val.to_le_bytes());
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `grayf32_to_luma_u16_row`: clamp [0,1] × 65535 → u16.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_luma_u16_row(y_plane: &[f32], out: &mut [u16], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = _mm_set1_ps(65535.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_loadu_ps(y_plane.as_ptr().add(x));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
      let int32 = _mm_cvttps_epi32(rounded);
      // Narrow i32x4 → u16x4 via packus (values in [0, 65535] so no saturation clipping).
      let pack16 = _mm_packus_epi32(int32, int32);
      // Store 8 bytes (4 u16 values) via unaligned store to the u16 output.
      // _mm_storel_epi64 writes 8 bytes (low 64-bit lane) unaligned.
      _mm_storel_epi64(out.as_mut_ptr().add(x).cast::<__m128i>(), pack16);
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_u16_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `grayf32_to_luma_f32_row`: memcpy pass-through.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_luma_f32_row(y_plane: &[f32], out: &mut [f32], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  scalar::grayf32_to_luma_f32_row(y_plane, out, width);
}

/// SSE4.1 `grayf32_to_hsv_row`: H=0, S=0, V = clamp(Y,0,1)×255.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_hsv_row(
  y_plane: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  let scale = _mm_set1_ps(255.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let zero128 = _mm_setzero_si128();
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_loadu_ps(y_plane.as_ptr().add(x));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled);
      let int32 = _mm_cvttps_epi32(rounded);
      let pack16 = _mm_packs_epi32(int32, int32);
      let pack8 = _mm_packus_epi16(pack16, pack16);
      let val = _mm_cvtsi128_si32(pack8) as u32;
      let vbytes = val.to_le_bytes();
      // H and S: 4 zero bytes; V: the packed bytes.
      h_out[x..x + 4].copy_from_slice(&[0u8; 4]);
      s_out[x..x + 4].copy_from_slice(&[0u8; 4]);
      v_out[x..x + 4].copy_from_slice(&vbytes);
      let _ = zero128;
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_hsv_row(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Ya8 -------------------------------------------------------------------

/// SSE4.1 `ya8_to_rgb_row`: deinterleave [Y,A] packed u8, broadcast Y → R=G=B.
///
/// Block size: 8 px / iter (16 bytes = 8 Ya8 pixels via `_mm_loadu_si128`).
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya8_to_rgb_row(packed: &[u8], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  // SSE4.1 lacks 3-channel interleave; use shuffle to extract Y then scatter.
  let mut x = 0usize;
  unsafe {
    // Mask to extract Y bytes (even bytes) from 16-byte Ya8 block (8 pixels).
    // Positions 0,2,4,6,8,10,12,14 are Y; set upper 8 bytes to 0x80 (=zero pad).
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 14, 12, 10, 8, 6, 4, 2, 0,
    );
    while x + 8 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast());
      let y_bytes = _mm_shuffle_epi8(chunk, y_mask); // 8 Y values in low 8 bytes
      let y_lo = _mm_cvtsi128_si64(y_bytes) as u64;
      let ybuf = y_lo.to_le_bytes();
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

/// SSE4.1 `ya8_to_rgba_row`: deinterleave [Y,A], broadcast Y → R=G=B, pass α.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `ya8_to_rgb_u16_row`: zero-extend Y → u16, broadcast R=G=B.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya8_to_rgb_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  scalar::ya8_to_rgb_u16_row(packed, out, width);
}

/// SSE4.1 `ya8_to_rgba_u16_row`: zero-extend Y and A → u16.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya8_to_rgba_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  scalar::ya8_to_rgba_u16_row(packed, out, width);
}

/// SSE4.1 `ya8_to_luma_row`: extract Y bytes.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `ya8_to_luma_u16_row`: zero-extend Y → u16.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya8_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  scalar::ya8_to_luma_u16_row(packed, out, width);
}

/// SSE4.1 `ya8_to_hsv_row`: H=0, S=0, V=Y. α dropped.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `ya16_to_rgb_row`: deinterleave [Y,A] u16, Y `>> 8` → u8, broadcast.
///
/// Block size: 4 px / iter (16 bytes = 4 Ya16 pixels).
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya16_to_rgb_row(packed: &[u16], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  unsafe {
    // Extract Y words (positions 0,2,4,6 in u16 terms = bytes 0,4,8,12).
    // We load 16 bytes (4 Ya16 pixels), shuffle Y to low 4 words, shift >> 8.
    let y_mask = _mm_set_epi8(
      -128, -128, -128, -128, -128, -128, -128, -128, 13, 12, 9, 8, 5, 4, 1, 0,
    );
    while x + 4 <= width {
      let chunk = _mm_loadu_si128(packed.as_ptr().add(x * 2).cast::<__m128i>());
      let y_words = _mm_shuffle_epi8(chunk, y_mask); // 4 Y u16 in low 8 bytes
      let y_shifted = _mm_srli_epi16(y_words, 8); // >> 8 → u8 in low byte of each u16
      let pack8 = _mm_packus_epi16(y_shifted, _mm_setzero_si128()); // 4 Y bytes in low 4
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
    scalar::ya16_to_rgb_row(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// SSE4.1 `ya16_to_rgba_row`: Y `>> 8`, A `>> 8`, broadcast Y to R=G=B.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya16_to_rgba_row(packed: &[u16], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
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
    scalar::ya16_to_rgba_row(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// SSE4.1 `ya16_to_rgb_u16_row`: native Y u16, broadcast R=G=B.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya16_to_rgb_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  scalar::ya16_to_rgb_u16_row(packed, out, width);
}

/// SSE4.1 `ya16_to_rgba_u16_row`: native Y and A u16.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya16_to_rgba_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  scalar::ya16_to_rgba_u16_row(packed, out, width);
}

/// SSE4.1 `ya16_to_luma_row`: Y `>> 8` → u8.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya16_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
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
    scalar::ya16_to_luma_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `ya16_to_luma_u16_row`: native Y pass-through.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya16_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  scalar::ya16_to_luma_u16_row(packed, out, width);
}

/// SSE4.1 `ya16_to_hsv_row`: H=0, S=0, V = Y `>> 8`. α dropped.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn ya16_to_hsv_row(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
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
    scalar::ya16_to_hsv_row(
      &packed[x * 2..width * 2],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use crate::row::scalar::gray as scalar;

  const WIDTHS: &[usize] = &[1, 7, 8, 16, 17, 32, 33, 64, 128, 130];

  fn prng(out: &mut [u8], seed: u32) {
    let mut s = seed;
    for v in out.iter_mut() {
      s = s.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (s >> 16) as u8;
    }
  }
  fn prng16(out: &mut [u16], seed: u32) {
    let mut buf = std::vec![0u8; out.len() * 2];
    prng(&mut buf, seed);
    for (i, o) in out.iter_mut().enumerate() {
      *o = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
    }
  }

  #[test]
  fn sse41_gray8_to_rgb_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      prng(&mut plane, 0xABCD);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::gray8_to_rgb_row(&plane, &mut simd, w, true) };
      scalar::gray8_to_rgb_row(&plane, &mut scal, w, true);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  fn sse41_gray8_to_hsv_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      prng(&mut plane, 0x5678);
      let mut sh = std::vec![0u8; w];
      let mut ss = std::vec![0u8; w];
      let mut sv = std::vec![0u8; w];
      let mut rh = std::vec![0u8; w];
      let mut rs = std::vec![0u8; w];
      let mut rv = std::vec![0u8; w];
      unsafe { super::gray8_to_hsv_row(&plane, &mut sh, &mut ss, &mut sv, w, true) };
      scalar::gray8_to_hsv_row(&plane, &mut rh, &mut rs, &mut rv, w, true);
      assert_eq!(sh, rh, "H width={w}");
      assert_eq!(ss, rs, "S width={w}");
      assert_eq!(sv, rv, "V width={w}");
    }
  }

  #[test]
  fn sse41_gray_n_to_luma_u16_10bit_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0xCAFE_BABE);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::gray_n_to_luma_u16_row::<10>(&plane, &mut simd, w) };
      scalar::gray_n_to_luma_u16_row::<10>(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  fn sse41_gray16_to_luma_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0xDEAD_BEEF);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::gray16_to_luma_u16_row(&plane, &mut simd, w) };
      scalar::gray16_to_luma_u16_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  fn sse41_gray16_to_luma_row_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0x1234_5678);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::gray16_to_luma_row(&plane, &mut simd, w) };
      scalar::gray16_to_luma_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  fn sse41_gray8_limited_range_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      prng(&mut plane, 0xCAFE_BABEu32);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::gray8_to_rgb_row(&plane, &mut simd, w, false) };
      scalar::gray8_to_rgb_row(&plane, &mut scal, w, false);
      assert_eq!(simd, scal, "width={w} limited-range");
    }
  }

  #[test]
  fn sse41_gray16_limited_range_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0x1234_5678);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::gray16_to_rgb_row(&plane, &mut simd, w, false) };
      scalar::gray16_to_rgb_row(&plane, &mut scal, w, false);
      assert_eq!(simd, scal, "width={w} limited-range");
    }
  }

  // ---- Grayf32 parity tests ---------------------------------------------------

  fn prng_f32(out: &mut [f32], seed: u32) {
    let mut s = seed;
    for v in out.iter_mut() {
      s = s.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = ((s >> 8) as f32) / (u32::MAX as f32) * 1.3 - 0.1;
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_grayf32_to_rgb_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF320_0001);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::grayf32_to_rgb_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgb_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_grayf32_to_rgba_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF320_0002);
      let mut simd = std::vec![0u8; w * 4];
      let mut scal = std::vec![0u8; w * 4];
      unsafe { super::grayf32_to_rgba_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgba_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_grayf32_to_rgb_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF320_0003);
      let mut simd = std::vec![0u16; w * 3];
      let mut scal = std::vec![0u16; w * 3];
      unsafe { super::grayf32_to_rgb_u16_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgb_u16_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_grayf32_to_rgba_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF320_0004);
      let mut simd = std::vec![0u16; w * 4];
      let mut scal = std::vec![0u16; w * 4];
      unsafe { super::grayf32_to_rgba_u16_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgba_u16_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_grayf32_to_rgb_f32_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF320_0005);
      let mut simd = std::vec![0.0f32; w * 3];
      let mut scal = std::vec![0.0f32; w * 3];
      unsafe { super::grayf32_to_rgb_f32_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgb_f32_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_grayf32_to_luma_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF320_0006);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::grayf32_to_luma_row(&plane, &mut simd, w) };
      sf::grayf32_to_luma_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_grayf32_to_luma_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF320_0007);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::grayf32_to_luma_u16_row(&plane, &mut simd, w) };
      sf::grayf32_to_luma_u16_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_grayf32_to_luma_f32_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF320_0008);
      let mut simd = std::vec![0.0f32; w];
      let mut scal = std::vec![0.0f32; w];
      unsafe { super::grayf32_to_luma_f32_row(&plane, &mut simd, w) };
      sf::grayf32_to_luma_f32_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_grayf32_to_hsv_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF320_0009);
      let mut sh = std::vec![0u8; w];
      let mut ss = std::vec![0u8; w];
      let mut sv = std::vec![0u8; w];
      let mut rh = std::vec![0u8; w];
      let mut rs = std::vec![0u8; w];
      let mut rv = std::vec![0u8; w];
      unsafe { super::grayf32_to_hsv_row(&plane, &mut sh, &mut ss, &mut sv, w) };
      sf::grayf32_to_hsv_row(&plane, &mut rh, &mut rs, &mut rv, w);
      assert_eq!(sh, rh, "H width={w}");
      assert_eq!(ss, rs, "S width={w}");
      assert_eq!(sv, rv, "V width={w}");
    }
  }

  // ---- Ya8 parity tests -------------------------------------------------------

  fn prng_ya8(out: &mut [u8], seed: u32) {
    let mut s = seed;
    for v in out.iter_mut() {
      s = s.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (s >> 16) as u8;
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya8_to_rgb_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA480_0001);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::ya8_to_rgb_row(&packed, &mut simd, w) };
      sy::ya8_to_rgb_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya8_to_rgba_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA480_0002);
      let mut simd = std::vec![0u8; w * 4];
      let mut scal = std::vec![0u8; w * 4];
      unsafe { super::ya8_to_rgba_row(&packed, &mut simd, w) };
      sy::ya8_to_rgba_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya8_to_rgb_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA480_0003);
      let mut simd = std::vec![0u16; w * 3];
      let mut scal = std::vec![0u16; w * 3];
      unsafe { super::ya8_to_rgb_u16_row(&packed, &mut simd, w) };
      sy::ya8_to_rgb_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya8_to_rgba_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA480_0004);
      let mut simd = std::vec![0u16; w * 4];
      let mut scal = std::vec![0u16; w * 4];
      unsafe { super::ya8_to_rgba_u16_row(&packed, &mut simd, w) };
      sy::ya8_to_rgba_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya8_to_luma_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA480_0005);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::ya8_to_luma_row(&packed, &mut simd, w) };
      sy::ya8_to_luma_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya8_to_luma_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA480_0006);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::ya8_to_luma_u16_row(&packed, &mut simd, w) };
      sy::ya8_to_luma_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya8_to_hsv_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA480_0007);
      let mut sh = std::vec![0u8; w];
      let mut ss = std::vec![0u8; w];
      let mut sv = std::vec![0u8; w];
      let mut rh = std::vec![0u8; w];
      let mut rs = std::vec![0u8; w];
      let mut rv = std::vec![0u8; w];
      unsafe { super::ya8_to_hsv_row(&packed, &mut sh, &mut ss, &mut sv, w) };
      sy::ya8_to_hsv_row(&packed, &mut rh, &mut rs, &mut rv, w);
      assert_eq!(sh, rh, "H width={w}");
      assert_eq!(ss, rs, "S width={w}");
      assert_eq!(sv, rv, "V width={w}");
    }
  }

  // ---- Ya16 parity tests ------------------------------------------------------

  fn prng_ya16(out: &mut [u16], seed: u32) {
    let mut buf = std::vec![0u8; out.len() * 2];
    prng(&mut buf, seed);
    for (i, o) in out.iter_mut().enumerate() {
      *o = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya16_to_rgb_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0001);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::ya16_to_rgb_row(&packed, &mut simd, w) };
      sy::ya16_to_rgb_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya16_to_rgba_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0002);
      let mut simd = std::vec![0u8; w * 4];
      let mut scal = std::vec![0u8; w * 4];
      unsafe { super::ya16_to_rgba_row(&packed, &mut simd, w) };
      sy::ya16_to_rgba_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya16_to_rgb_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0003);
      let mut simd = std::vec![0u16; w * 3];
      let mut scal = std::vec![0u16; w * 3];
      unsafe { super::ya16_to_rgb_u16_row(&packed, &mut simd, w) };
      sy::ya16_to_rgb_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya16_to_rgba_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0004);
      let mut simd = std::vec![0u16; w * 4];
      let mut scal = std::vec![0u16; w * 4];
      unsafe { super::ya16_to_rgba_u16_row(&packed, &mut simd, w) };
      sy::ya16_to_rgba_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya16_to_luma_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0005);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::ya16_to_luma_row(&packed, &mut simd, w) };
      sy::ya16_to_luma_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya16_to_luma_u16_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0006);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::ya16_to_luma_u16_row(&packed, &mut simd, w) };
      sy::ya16_to_luma_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn sse41_ya16_to_hsv_matches_scalar() {
    if !is_x86_feature_detected!("sse4.1") {
      return;
    }
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0007);
      let mut sh = std::vec![0u8; w];
      let mut ss = std::vec![0u8; w];
      let mut sv = std::vec![0u8; w];
      let mut rh = std::vec![0u8; w];
      let mut rs = std::vec![0u8; w];
      let mut rv = std::vec![0u8; w];
      unsafe { super::ya16_to_hsv_row(&packed, &mut sh, &mut ss, &mut sv, w) };
      sy::ya16_to_hsv_row(&packed, &mut rh, &mut rs, &mut rv, w);
      assert_eq!(sh, rh, "H width={w}");
      assert_eq!(ss, rs, "S width={w}");
      assert_eq!(sv, rv, "V width={w}");
    }
  }
}
