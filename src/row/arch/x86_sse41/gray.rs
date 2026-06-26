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

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::row::{
  arch::x86_sse41::endian::{load_endian_u16x4, load_endian_u16x8, load_endian_u32x4},
  scalar::{bits_mask, gray as scalar},
};

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

/// SSE4.1 `gray_n_to_rgba_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  // SSE4.1 4-channel interleave without SSSE3 shuffle tables is complex;
  // delegate to scalar (which auto-vectorizes well at -O3).
  scalar::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
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

/// SSE4.1 `gray_n_to_rgba_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `gray_n_to_luma_row<BITS>`: mask, shift, pack, store.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
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
    let mask_v = _mm_set1_epi16(mask as i16);
    // `_mm_srli_epi16::<IMM8>` requires a literal const generic shift, which
    // is not available for `BITS - 8` on stable Rust.  Use the variable-count
    // variant `_mm_srl_epi16` with a count vector built from the shift amount.
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
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
    scalar::gray_n_to_luma_row::<BITS, BE>(&y_plane[x..width], &mut out[x..width], width - x);
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
    let mask_v = _mm_set1_epi16(mask as i16);
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = _mm_and_si128(raw, mask_v);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), masked);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_u16_row::<BITS, BE>(&y_plane[x..width], &mut out[x..width], width - x);
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
    let mask_v = _mm_set1_epi16(mask as i16);
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero = _mm_setzero_si128();
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
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

/// SSE4.1 `gray16_to_rgba_row`: `>> 8` → RGBA u8.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `gray16_to_rgb_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `gray16_to_rgba_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `gray16_to_luma_row`: `>> 8`, pack, store.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_luma_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    let zero = _mm_setzero_si128();
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let shifted = _mm_srli_epi16(raw, 8);
      let packed = _mm_packus_epi16(shifted, zero);
      let val = _mm_cvtsi128_si64(packed) as u64;
      out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
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
pub(crate) unsafe fn gray16_to_luma_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), y);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
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
    let zero16 = _mm_setzero_si128();
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
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

/// SSE4.1 `grayf32_to_rgb_row`: clamp [0,1] × 255 → u8, broadcast Y → R=G=B.
///
/// Uses MXCSR-independent round-half-up: `+ 0.5` then `_mm_cvttps_epi32`
/// (matches the scalar `(y * scale + 0.5) as T` contract).
/// Block size: 4 px / iter.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgb_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = _mm_set1_ps(255.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_castsi128_ps(load_endian_u32x4::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let int32 = _mm_cvttps_epi32(_mm_add_ps(scaled, _mm_set1_ps(0.5)));
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
    scalar::grayf32_to_rgb_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// SSE4.1 `grayf32_to_rgba_row`: clamp [0,1] × 255 → u8, broadcast + α=0xFF.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgba_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = _mm_set1_ps(255.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_castsi128_ps(load_endian_u32x4::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let int32 = _mm_cvttps_epi32(_mm_add_ps(scaled, _mm_set1_ps(0.5)));
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
    scalar::grayf32_to_rgba_row::<BE>(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// SSE4.1 `grayf32_to_rgb_u16_row`: clamp [0,1] × 65535 → u16, broadcast.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgb_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = _mm_set1_ps(65535.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_castsi128_ps(load_endian_u32x4::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let int32 = _mm_cvttps_epi32(_mm_add_ps(scaled, _mm_set1_ps(0.5)));
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
    scalar::grayf32_to_rgb_u16_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// SSE4.1 `grayf32_to_rgba_u16_row`: clamp [0,1] × 65535 → u16, broadcast + α=0xFFFF.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgba_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = _mm_set1_ps(65535.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_castsi128_ps(load_endian_u32x4::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let int32 = _mm_cvttps_epi32(_mm_add_ps(scaled, _mm_set1_ps(0.5)));
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
    scalar::grayf32_to_rgba_u16_row::<BE>(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// SSE4.1 `grayf32_to_rgb_f32_row`: lossless replicate Y → R=G=B.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_rgb_f32_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [f32],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  // f32 triplet broadcast: scalar is already optimal here.
  scalar::grayf32_to_rgb_f32_row::<BE>(y_plane, out, width);
}

/// SSE4.1 `grayf32_to_luma_row`: clamp [0,1] × 255 → u8.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_luma_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = _mm_set1_ps(255.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_castsi128_ps(load_endian_u32x4::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let int32 = _mm_cvttps_epi32(_mm_add_ps(scaled, _mm_set1_ps(0.5)));
      let pack16 = _mm_packs_epi32(int32, int32);
      let pack8 = _mm_packus_epi16(pack16, pack16);
      // Store 4 bytes: low 4 of pack8.
      let val = _mm_cvtsi128_si32(pack8) as u32;
      out[x..x + 4].copy_from_slice(&val.to_le_bytes());
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `grayf32_to_luma_u16_row`: clamp [0,1] × 65535 → u16.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_luma_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = _mm_set1_ps(65535.0);
  let zero = _mm_setzero_ps();
  let one = _mm_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = _mm_castsi128_ps(load_endian_u32x4::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let int32 = _mm_cvttps_epi32(_mm_add_ps(scaled, _mm_set1_ps(0.5)));
      // Narrow i32x4 → u16x4 via packus (values in [0, 65535] so no saturation clipping).
      let pack16 = _mm_packus_epi32(int32, int32);
      // Store 8 bytes (4 u16 values) via unaligned store to the u16 output.
      // _mm_storel_epi64 writes 8 bytes (low 64-bit lane) unaligned.
      _mm_storel_epi64(out.as_mut_ptr().add(x).cast::<__m128i>(), pack16);
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `grayf32_to_luma_f32_row`: memcpy pass-through.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `grayf32_to_hsv_row`: H=0, S=0, V = clamp(Y,0,1)×255.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn grayf32_to_hsv_row<const BE: bool>(
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
      let y = _mm_castsi128_ps(load_endian_u32x4::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm_min_ps(_mm_max_ps(y, zero), one);
      let scaled = _mm_mul_ps(clamped, scale);
      let int32 = _mm_cvttps_epi32(_mm_add_ps(scaled, _mm_set1_ps(0.5)));
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

/// Host-endian gate for Ya16 SIMD bodies.
///
/// The SSE4.1 Ya16 SIMD bodies use `_mm_loadu_si128` + fixed `_mm_shuffle_epi8`
/// masks (`13,12,9,8,5,4,1,0` etc.) that gather the **host-native** high byte
/// of each Ya16 word. They are only correct when the encoded byte order matches
/// the host. Truth table:
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

/// SSE4.1 `ya16_to_rgb_row`: deinterleave [Y,A] u16, Y `>> 8` → u8, broadcast.
///
/// Block size: 4 px / iter (16 bytes = 4 Ya16 pixels).
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    scalar::ya16_to_rgb_row::<BE>(
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

/// SSE4.1 `ya16_to_rgb_u16_row`: native Y u16, broadcast R=G=B.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `ya16_to_rgba_u16_row`: native Y and A u16.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `ya16_to_luma_row`: Y `>> 8` → u8.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `ya16_to_luma_u16_row`: native Y pass-through.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
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

/// SSE4.1 `ya16_to_hsv_row`: H=0, S=0, V = Y `>> 8`. α dropped.
///
/// # Safety
/// SSE4.1 must be available.
#[inline]
#[target_feature(enable = "sse4.1")]
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
// F16C (`_mm_cvtph_ps`), then delegate to the existing SSE4.1 `grayf32`
// downstream kernels with `HOST_NATIVE_BE` (the widened buffer is host-native).
// The half-float twin of the `grayf32` SSE4.1 kernels; the f16 reading mirrors
// the Rgbf16 SSE4.1 path (`load_endian_u16x4` + `_mm_cvtph_ps`).

/// `BE` value that makes the `grayf32` row kernels treat their input as
/// host-native (no-op swap) after the F16C widen produced host-native f32.
const GRAYF16_HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Widen 4 x f16 (8 bytes at `ptr`) to `out[0..4]` (host-native f32).
/// For `BE = true` the f16 bits are byte-swapped before widening.
///
/// # Safety
/// SSE4.1 + F16C must be available. `ptr` valid for 8 bytes; `out` for 4 f32.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
unsafe fn widen_f16x4_sse_buf<const BE: bool>(ptr: *const half::f16, out: *mut f32) {
  unsafe {
    let m = _mm_cvtph_ps(load_endian_u16x4::<BE>(ptr.cast::<u8>()));
    _mm_storeu_ps(out, m);
  }
}

/// SSE4.1 `grayf16_to_rgb_row`: widen f16 → f32, clamp [0,1] x 255 → u8, broadcast.
/// # Safety
/// SSE4.1 + F16C must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn grayf16_to_rgb_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 4];
    unsafe {
      widen_f16x4_sse_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_rgb_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x * 3..(x + 4) * 3], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::grayf16_to_rgb_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// SSE4.1 `grayf16_to_rgba_row`: widen f16 → f32, clamp [0,1] x 255, broadcast, α=0xFF.
/// # Safety
/// SSE4.1 + F16C must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn grayf16_to_rgba_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 4];
    unsafe {
      widen_f16x4_sse_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_rgba_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x * 4..(x + 4) * 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::grayf16_to_rgba_row::<BE>(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// SSE4.1 `grayf16_to_rgb_u16_row`: widen f16 → f32, clamp [0,1] x 65535 → u16, broadcast.
/// # Safety
/// SSE4.1 + F16C must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn grayf16_to_rgb_u16_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 4];
    unsafe {
      widen_f16x4_sse_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_rgb_u16_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x * 3..(x + 4) * 3], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::grayf16_to_rgb_u16_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// SSE4.1 `grayf16_to_rgba_u16_row`: widen f16 → f32, clamp [0,1] x 65535, broadcast, α=0xFFFF.
/// # Safety
/// SSE4.1 + F16C must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn grayf16_to_rgba_u16_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 4];
    unsafe {
      widen_f16x4_sse_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_rgba_u16_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x * 4..(x + 4) * 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::grayf16_to_rgba_u16_row::<BE>(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// SSE4.1 `grayf16_to_luma_row`: widen f16 → f32, clamp [0,1] x 255 → u8 luma.
/// # Safety
/// SSE4.1 + F16C must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn grayf16_to_luma_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 4];
    unsafe {
      widen_f16x4_sse_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_luma_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x..x + 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::grayf16_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `grayf16_to_luma_u16_row`: widen f16 → f32, clamp [0,1] x 65535 → u16 luma.
/// # Safety
/// SSE4.1 + F16C must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn grayf16_to_luma_u16_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf16 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 4];
    unsafe {
      widen_f16x4_sse_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_luma_u16_row::<GRAYF16_HOST_NATIVE_BE>(&buf, &mut out[x..x + 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::grayf16_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `grayf16_to_hsv_row`: H=0, S=0, V = clamp(widen(Y),0,1) x 255.
/// # Safety
/// SSE4.1 + F16C must be available. `y_plane.len() >= width`; H/S/V out `>= width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
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
  while x + 4 <= width {
    let mut buf = [0.0f32; 4];
    unsafe {
      widen_f16x4_sse_buf::<BE>(y_plane.as_ptr().add(x), buf.as_mut_ptr());
      grayf32_to_hsv_row::<GRAYF16_HOST_NATIVE_BE>(
        &buf,
        &mut h_out[x..x + 4],
        &mut s_out[x..x + 4],
        &mut v_out[x..x + 4],
        4,
      );
    }
    x += 4;
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
// Packed `[Y, A]` f32 source. Each 4-pixel chunk deinterleaves Y (and A for
// RGBA outputs) with `_mm_shuffle_ps` (128-bit, lane-crossing-free) into a host
// -native f32 stack buffer, then delegates to the proven `grayf32` SSE4.1
// kernels for the clamp / scale / round math (Y broadcast R=G=B; A patched into
// the RGBA α channel via `grayf32_to_luma*`). Like the `ya16` SSE path, the
// host-native deinterleave is only correct when the source byte order matches
// the host, so `BE != HOST_NATIVE_BE` falls through to scalar.

/// SSE4.1 `yaf32_to_rgb_row`: deinterleave `[Y,A]` f32, clamp Y [0,1] x 255 → u8, broadcast.
///
/// # Safety
/// SSE4.1 must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
  while x + 4 <= width {
    let mut ybuf = [0.0f32; 4];
    unsafe {
      let ya0 = _mm_loadu_ps(packed.as_ptr().add(x * 2));
      let ya1 = _mm_loadu_ps(packed.as_ptr().add(x * 2 + 4));
      _mm_storeu_ps(ybuf.as_mut_ptr(), _mm_shuffle_ps::<0x88>(ya0, ya1));
      grayf32_to_rgb_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x * 3..(x + 4) * 3], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf32_to_rgb_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// SSE4.1 `yaf32_to_rgba_row`: clamp Y x 255 broadcast, α = clamp(A) x 255.
///
/// # Safety
/// SSE4.1 must be available. `packed.len() >= width * 2`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
  while x + 4 <= width {
    let mut ybuf = [0.0f32; 4];
    let mut abuf = [0.0f32; 4];
    let mut a8 = [0u8; 4];
    unsafe {
      let ya0 = _mm_loadu_ps(packed.as_ptr().add(x * 2));
      let ya1 = _mm_loadu_ps(packed.as_ptr().add(x * 2 + 4));
      _mm_storeu_ps(ybuf.as_mut_ptr(), _mm_shuffle_ps::<0x88>(ya0, ya1));
      _mm_storeu_ps(abuf.as_mut_ptr(), _mm_shuffle_ps::<0xDD>(ya0, ya1));
      grayf32_to_rgba_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x * 4..(x + 4) * 4], 4);
      grayf32_to_luma_row::<HOST_NATIVE_BE>(&abuf, &mut a8, 4);
    }
    for i in 0..4 {
      out[(x + i) * 4 + 3] = a8[i];
    }
    x += 4;
  }
  if x < width {
    scalar::yaf32_to_rgba_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// SSE4.1 `yaf32_to_rgb_u16_row`: clamp Y [0,1] x 65535 → u16, broadcast.
///
/// # Safety
/// SSE4.1 must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
  while x + 4 <= width {
    let mut ybuf = [0.0f32; 4];
    unsafe {
      let ya0 = _mm_loadu_ps(packed.as_ptr().add(x * 2));
      let ya1 = _mm_loadu_ps(packed.as_ptr().add(x * 2 + 4));
      _mm_storeu_ps(ybuf.as_mut_ptr(), _mm_shuffle_ps::<0x88>(ya0, ya1));
      grayf32_to_rgb_u16_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x * 3..(x + 4) * 3], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf32_to_rgb_u16_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// SSE4.1 `yaf32_to_rgba_u16_row`: clamp Y x 65535 broadcast, α = clamp(A) x 65535.
///
/// # Safety
/// SSE4.1 must be available. `packed.len() >= width * 2`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
  while x + 4 <= width {
    let mut ybuf = [0.0f32; 4];
    let mut abuf = [0.0f32; 4];
    let mut a16 = [0u16; 4];
    unsafe {
      let ya0 = _mm_loadu_ps(packed.as_ptr().add(x * 2));
      let ya1 = _mm_loadu_ps(packed.as_ptr().add(x * 2 + 4));
      _mm_storeu_ps(ybuf.as_mut_ptr(), _mm_shuffle_ps::<0x88>(ya0, ya1));
      _mm_storeu_ps(abuf.as_mut_ptr(), _mm_shuffle_ps::<0xDD>(ya0, ya1));
      grayf32_to_rgba_u16_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x * 4..(x + 4) * 4], 4);
      grayf32_to_luma_u16_row::<HOST_NATIVE_BE>(&abuf, &mut a16, 4);
    }
    for i in 0..4 {
      out[(x + i) * 4 + 3] = a16[i];
    }
    x += 4;
  }
  if x < width {
    scalar::yaf32_to_rgba_u16_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// SSE4.1 `yaf32_to_luma_row`: clamp Y [0,1] x 255 → u8 luma.
///
/// # Safety
/// SSE4.1 must be available. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
  while x + 4 <= width {
    let mut ybuf = [0.0f32; 4];
    unsafe {
      let ya0 = _mm_loadu_ps(packed.as_ptr().add(x * 2));
      let ya1 = _mm_loadu_ps(packed.as_ptr().add(x * 2 + 4));
      _mm_storeu_ps(ybuf.as_mut_ptr(), _mm_shuffle_ps::<0x88>(ya0, ya1));
      grayf32_to_luma_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x..x + 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf32_to_luma_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `yaf32_to_luma_u16_row`: clamp Y [0,1] x 65535 → u16 luma.
///
/// # Safety
/// SSE4.1 must be available. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
  while x + 4 <= width {
    let mut ybuf = [0.0f32; 4];
    unsafe {
      let ya0 = _mm_loadu_ps(packed.as_ptr().add(x * 2));
      let ya1 = _mm_loadu_ps(packed.as_ptr().add(x * 2 + 4));
      _mm_storeu_ps(ybuf.as_mut_ptr(), _mm_shuffle_ps::<0x88>(ya0, ya1));
      grayf32_to_luma_u16_row::<HOST_NATIVE_BE>(&ybuf, &mut out[x..x + 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf32_to_luma_u16_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `yaf32_to_hsv_row`: H=0, S=0, V = clamp(Y,0,1) x 255. α dropped.
///
/// # Safety
/// SSE4.1 must be available. `packed.len() >= width * 2`; H/S/V out `>= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
  while x + 4 <= width {
    let mut ybuf = [0.0f32; 4];
    unsafe {
      let ya0 = _mm_loadu_ps(packed.as_ptr().add(x * 2));
      let ya1 = _mm_loadu_ps(packed.as_ptr().add(x * 2 + 4));
      _mm_storeu_ps(ybuf.as_mut_ptr(), _mm_shuffle_ps::<0x88>(ya0, ya1));
      grayf32_to_hsv_row::<HOST_NATIVE_BE>(
        &ybuf,
        &mut h_out[x..x + 4],
        &mut s_out[x..x + 4],
        &mut v_out[x..x + 4],
        4,
      );
    }
    x += 4;
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
// Widen each 4-pixel chunk of packed `[Y, A]` f16 (8 f16) to a host-native f32
// stack buffer with the F16C `_mm_cvtph_ps` (`widen_f16x4_sse_buf`), then
// delegate to the `yaf32` SSE4.1 kernels with `HOST_NATIVE_BE`. The half-float
// twin of the `yaf32` SSE path.

/// SSE4.1 `yaf16_to_rgb_row`: widen `[Y,A]` f16 → f32, clamp Y x 255 → u8, broadcast.
///
/// # Safety
/// SSE4.1 + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn yaf16_to_rgb_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 8];
    unsafe {
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2 + 4), buf.as_mut_ptr().add(4));
      yaf32_to_rgb_row::<HOST_NATIVE_BE>(&buf, &mut out[x * 3..(x + 4) * 3], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf16_to_rgb_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// SSE4.1 `yaf16_to_rgba_row`: widen `[Y,A]` f16 → f32, clamp Y x 255 broadcast, α = clamp(A) x 255.
///
/// # Safety
/// SSE4.1 + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn yaf16_to_rgba_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 8];
    unsafe {
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2 + 4), buf.as_mut_ptr().add(4));
      yaf32_to_rgba_row::<HOST_NATIVE_BE>(&buf, &mut out[x * 4..(x + 4) * 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf16_to_rgba_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// SSE4.1 `yaf16_to_rgb_u16_row`: widen `[Y,A]` f16 → f32, clamp Y x 65535 → u16, broadcast.
///
/// # Safety
/// SSE4.1 + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn yaf16_to_rgb_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 8];
    unsafe {
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2 + 4), buf.as_mut_ptr().add(4));
      yaf32_to_rgb_u16_row::<HOST_NATIVE_BE>(&buf, &mut out[x * 3..(x + 4) * 3], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf16_to_rgb_u16_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// SSE4.1 `yaf16_to_rgba_u16_row`: widen `[Y,A]` f16 → f32, clamp Y x 65535 broadcast, α = clamp(A) x 65535.
///
/// # Safety
/// SSE4.1 + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn yaf16_to_rgba_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 8];
    unsafe {
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2 + 4), buf.as_mut_ptr().add(4));
      yaf32_to_rgba_u16_row::<HOST_NATIVE_BE>(&buf, &mut out[x * 4..(x + 4) * 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf16_to_rgba_u16_row::<BE>(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// SSE4.1 `yaf16_to_luma_row`: widen `[Y,A]` f16 → f32, clamp Y x 255 → u8 luma.
///
/// # Safety
/// SSE4.1 + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn yaf16_to_luma_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 8];
    unsafe {
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2 + 4), buf.as_mut_ptr().add(4));
      yaf32_to_luma_row::<HOST_NATIVE_BE>(&buf, &mut out[x..x + 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf16_to_luma_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `yaf16_to_luma_u16_row`: widen `[Y,A]` f16 → f32, clamp Y x 65535 → u16 luma.
///
/// # Safety
/// SSE4.1 + F16C must be available. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn yaf16_to_luma_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::yaf16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  while x + 4 <= width {
    let mut buf = [0.0f32; 8];
    unsafe {
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2 + 4), buf.as_mut_ptr().add(4));
      yaf32_to_luma_u16_row::<HOST_NATIVE_BE>(&buf, &mut out[x..x + 4], 4);
    }
    x += 4;
  }
  if x < width {
    scalar::yaf16_to_luma_u16_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// SSE4.1 `yaf16_to_hsv_row`: widen `[Y,A]` f16 → f32, H=0, S=0, V = clamp(Y,0,1) x 255. α dropped.
///
/// # Safety
/// SSE4.1 + F16C must be available. `packed.len() >= width * 2`; H/S/V out `>= width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
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
  while x + 4 <= width {
    let mut buf = [0.0f32; 8];
    unsafe {
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2), buf.as_mut_ptr());
      widen_f16x4_sse_buf::<BE>(packed.as_ptr().add(x * 2 + 4), buf.as_mut_ptr().add(4));
      yaf32_to_hsv_row::<HOST_NATIVE_BE>(
        &buf,
        &mut h_out[x..x + 4],
        &mut s_out[x..x + 4],
        &mut v_out[x..x + 4],
        4,
      );
    }
    x += 4;
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
