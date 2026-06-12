//! AVX2 gray kernel implementations.
//!
//! Gray → luma / luma_u16 / HSV paths get full AVX2 (32-px / 16-px blocks).
//! Packed-channel interleave paths (RGB, RGBA) delegate to scalar: the
//! 3/4-channel store pattern is verbose without SSSE3 shuffle tables and
//! the scalar implementations auto-vectorize well at -O3.
//!
//! # `full_range` parameter
//!
//! For RGB/RGBA/HSV kernels, `full_range = true` uses the existing fast AVX2
//! path. `full_range = false` (limited-range) falls back to scalar since
//! limited-range rescaling is the less-common path and the scalar formulation
//! is simple and correct.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::row::{
  arch::x86_avx2::endian::{load_endian_u16x16, load_endian_u32x8},
  scalar::{bits_mask, gray as scalar},
};

// ---- Gray8 ------------------------------------------------------------------

/// AVX2 `gray8_to_rgb_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray8_to_rgb_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  // 3-channel interleave without SSSE3 cross-lane shuffle is verbose under
  // AVX2. Scalar (which auto-vectorizes) handles this path.
  scalar::gray8_to_rgb_row(y_plane, out, width, full_range);
}

/// AVX2 `gray8_to_rgba_row`: broadcast Y into RGBA u8, 32 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray8_to_rgba_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  // AVX2: 32 u8 → 128 bytes RGBA (4-ch interleave). Requires manual
  // unpacklo/hi chains. Delegate to scalar for simplicity and correctness.
  scalar::gray8_to_rgba_row(y_plane, out, width, full_range);
}

/// AVX2 `gray8_to_hsv_row`: H=0, S=0, V=Y. 32 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// AVX2 must be available. All planes have length >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_si256();
    while x + 32 <= width {
      let v = _mm256_loadu_si256(y_plane.as_ptr().add(x).cast());
      _mm256_storeu_si256(h_out.as_mut_ptr().add(x).cast(), zero);
      _mm256_storeu_si256(s_out.as_mut_ptr().add(x).cast(), zero);
      _mm256_storeu_si256(v_out.as_mut_ptr().add(x).cast(), v);
      x += 32;
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

/// AVX2 `gray_n_to_rgb_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `gray_n_to_rgba_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `gray_n_to_rgb_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `gray_n_to_rgba_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `gray_n_to_luma_row<BITS>`: mask + shift to u8. 16 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let mask_v = _mm256_set1_epi16(mask as i16);
    // Use variable-count `_mm256_srl_epi16` since `_mm256_srli_epi16::<IMM8>`
    // requires a literal const generic shift not expressible as `BITS - 8`.
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    while x + 16 <= width {
      let raw = load_endian_u16x16::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = _mm256_and_si256(raw, mask_v);
      let shifted = _mm256_srl_epi16(masked, shr);
      // Pack u16x16 → u8x16 (with lane-cross fixup via permute4x64)
      let zero = _mm256_setzero_si256();
      // _mm256_packus_epi16 produces [lo_lo8, hi_lo8, lo_hi8, hi_hi8] order
      // (per 128-bit lane). permute4x64 with 0xD8 = [0,2,1,3] restores
      // natural order: [lo8, hi8] in a single 128-bit lane.
      let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(shifted, zero));
      // Extract low 128 bits = 16 valid u8 pixels.
      let lo = _mm256_castsi256_si128(packed);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), lo);
      x += 16;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_row::<BITS, BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `gray_n_to_luma_u16_row<BITS>`: mask, store. 16 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let mask_v = _mm256_set1_epi16(mask as i16);
    while x + 16 <= width {
      let raw = load_endian_u16x16::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = _mm256_and_si256(raw, mask_v);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), masked);
      x += 16;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_u16_row::<BITS, BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `gray_n_to_hsv_row<BITS>`: H=0, S=0, V = mask+shift. 16 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// AVX2 must be available. All slices have length >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let mask_v = _mm256_set1_epi16(mask as i16);
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero256 = _mm256_setzero_si256();
    while x + 16 <= width {
      let raw = load_endian_u16x16::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = _mm256_and_si256(raw, mask_v);
      let shifted = _mm256_srl_epi16(masked, shr);
      let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(shifted, zero256));
      let lo = _mm256_castsi256_si128(packed);
      // H and S: 16 zero bytes; V: the packed bytes.
      _mm_storeu_si128(
        h_out.as_mut_ptr().add(x).cast(),
        _mm256_castsi256_si128(zero256),
      );
      _mm_storeu_si128(
        s_out.as_mut_ptr().add(x).cast(),
        _mm256_castsi256_si128(zero256),
      );
      _mm_storeu_si128(v_out.as_mut_ptr().add(x).cast(), lo);
      x += 16;
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

/// AVX2 `gray16_to_rgb_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `gray16_to_rgba_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `gray16_to_rgb_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `gray16_to_rgba_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `gray16_to_luma_row`: `>> 8`, pack, store. 16 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray16_to_luma_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    let zero = _mm256_setzero_si256();
    while x + 16 <= width {
      let raw = load_endian_u16x16::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let shifted = _mm256_srli_epi16(raw, 8);
      // Pack u16x16 → u8x16 with lane-cross fixup.
      let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(shifted, zero));
      let lo = _mm256_castsi256_si128(packed);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), lo);
      x += 16;
    }
  }
  if x < width {
    scalar::gray16_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `gray16_to_luma_u16_row`: identity copy. 16 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray16_to_luma_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = load_endian_u16x16::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), y);
      x += 16;
    }
  }
  if x < width {
    scalar::gray16_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `gray16_to_hsv_row`: `>> 8`, H=0, S=0, V=Y8. 16 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// AVX2 must be available. All slices have length >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_si256();
    while x + 16 <= width {
      let raw = load_endian_u16x16::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let shifted = _mm256_srli_epi16(raw, 8);
      let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(shifted, zero));
      let lo = _mm256_castsi256_si128(packed);
      _mm_storeu_si128(
        h_out.as_mut_ptr().add(x).cast(),
        _mm256_castsi256_si128(zero),
      );
      _mm_storeu_si128(
        s_out.as_mut_ptr().add(x).cast(),
        _mm256_castsi256_si128(zero),
      );
      _mm_storeu_si128(v_out.as_mut_ptr().add(x).cast(), lo);
      x += 16;
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

/// AVX2 `grayf32_to_rgb_row`: clamp [0,1] x 255 → u8, broadcast Y → R=G=B.
///
/// Uses MXCSR-independent round-half-up: `+ 0.5` then `_mm256_cvttps_epi32`
/// (matches the scalar `(y * scale + 0.5) as T` contract).
/// Block size: 8 px / iter.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn grayf32_to_rgb_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = _mm256_set1_ps(255.0);
  let zero = _mm256_setzero_ps();
  let one = _mm256_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = _mm256_castsi256_ps(load_endian_u32x8::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm256_min_ps(_mm256_max_ps(y, zero), one);
      let scaled = _mm256_mul_ps(clamped, scale);
      let int32 = _mm256_cvttps_epi32(_mm256_add_ps(scaled, _mm256_set1_ps(0.5)));
      // Narrow 8xi32 → 8xu8 via two pack steps.
      let lo = _mm256_castsi256_si128(int32);
      let hi = _mm256_extracti128_si256::<1>(int32);
      let pack16 = _mm_packs_epi32(lo, hi); // 8xi16
      let pack8 = _mm_packus_epi16(pack16, pack16); // 8xu8 in low 8 bytes
      // Extract 8 bytes and scatter to RGB triples.
      let val = _mm_cvtsi128_si64(pack8) as u64;
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
    scalar::grayf32_to_rgb_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// AVX2 `grayf32_to_rgba_row`: clamp [0,1] x 255 → u8, broadcast + α=0xFF.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn grayf32_to_rgba_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = _mm256_set1_ps(255.0);
  let zero = _mm256_setzero_ps();
  let one = _mm256_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = _mm256_castsi256_ps(load_endian_u32x8::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm256_min_ps(_mm256_max_ps(y, zero), one);
      let scaled = _mm256_mul_ps(clamped, scale);
      let int32 = _mm256_cvttps_epi32(_mm256_add_ps(scaled, _mm256_set1_ps(0.5)));
      let lo = _mm256_castsi256_si128(int32);
      let hi = _mm256_extracti128_si256::<1>(int32);
      let pack16 = _mm_packs_epi32(lo, hi);
      let pack8 = _mm_packus_epi16(pack16, pack16);
      let val = _mm_cvtsi128_si64(pack8) as u64;
      let ybuf = val.to_le_bytes();
      let base = x * 4;
      for i in 0..8usize {
        out[base + i * 4] = ybuf[i];
        out[base + i * 4 + 1] = ybuf[i];
        out[base + i * 4 + 2] = ybuf[i];
        out[base + i * 4 + 3] = 0xFF;
      }
      x += 8;
    }
  }
  if x < width {
    scalar::grayf32_to_rgba_row::<BE>(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// AVX2 `grayf32_to_rgb_u16_row`: clamp [0,1] x 65535 → u16, broadcast.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn grayf32_to_rgb_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = _mm256_set1_ps(65535.0);
  let zero = _mm256_setzero_ps();
  let one = _mm256_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = _mm256_castsi256_ps(load_endian_u32x8::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm256_min_ps(_mm256_max_ps(y, zero), one);
      let scaled = _mm256_mul_ps(clamped, scale);
      let int32 = _mm256_cvttps_epi32(_mm256_add_ps(scaled, _mm256_set1_ps(0.5)));
      // Narrow 8xi32 → 8xu16 using AVX2 packus path.
      let lo = _mm256_castsi256_si128(int32);
      let hi = _mm256_extracti128_si256::<1>(int32);
      let pack16 = _mm_packus_epi32(lo, hi); // 8xu16 (values in [0,65535], no saturation)
      // Extract u16 values unrolled (const lane requirement).
      let base = x * 3;
      let v0 = _mm_extract_epi16::<0>(pack16) as u16;
      let v1 = _mm_extract_epi16::<1>(pack16) as u16;
      let v2 = _mm_extract_epi16::<2>(pack16) as u16;
      let v3 = _mm_extract_epi16::<3>(pack16) as u16;
      let v4 = _mm_extract_epi16::<4>(pack16) as u16;
      let v5 = _mm_extract_epi16::<5>(pack16) as u16;
      let v6 = _mm_extract_epi16::<6>(pack16) as u16;
      let v7 = _mm_extract_epi16::<7>(pack16) as u16;
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
      out[base + 12] = v4;
      out[base + 13] = v4;
      out[base + 14] = v4;
      out[base + 15] = v5;
      out[base + 16] = v5;
      out[base + 17] = v5;
      out[base + 18] = v6;
      out[base + 19] = v6;
      out[base + 20] = v6;
      out[base + 21] = v7;
      out[base + 22] = v7;
      out[base + 23] = v7;
      x += 8;
    }
  }
  if x < width {
    scalar::grayf32_to_rgb_u16_row::<BE>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// AVX2 `grayf32_to_rgba_u16_row`: clamp [0,1] x 65535 → u16, broadcast + α=0xFFFF.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn grayf32_to_rgba_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = _mm256_set1_ps(65535.0);
  let zero = _mm256_setzero_ps();
  let one = _mm256_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = _mm256_castsi256_ps(load_endian_u32x8::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm256_min_ps(_mm256_max_ps(y, zero), one);
      let scaled = _mm256_mul_ps(clamped, scale);
      let int32 = _mm256_cvttps_epi32(_mm256_add_ps(scaled, _mm256_set1_ps(0.5)));
      let lo = _mm256_castsi256_si128(int32);
      let hi = _mm256_extracti128_si256::<1>(int32);
      let pack16 = _mm_packus_epi32(lo, hi);
      let base = x * 4;
      let v0 = _mm_extract_epi16::<0>(pack16) as u16;
      let v1 = _mm_extract_epi16::<1>(pack16) as u16;
      let v2 = _mm_extract_epi16::<2>(pack16) as u16;
      let v3 = _mm_extract_epi16::<3>(pack16) as u16;
      let v4 = _mm_extract_epi16::<4>(pack16) as u16;
      let v5 = _mm_extract_epi16::<5>(pack16) as u16;
      let v6 = _mm_extract_epi16::<6>(pack16) as u16;
      let v7 = _mm_extract_epi16::<7>(pack16) as u16;
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
      out[base + 16] = v4;
      out[base + 17] = v4;
      out[base + 18] = v4;
      out[base + 19] = 0xFFFF;
      out[base + 20] = v5;
      out[base + 21] = v5;
      out[base + 22] = v5;
      out[base + 23] = 0xFFFF;
      out[base + 24] = v6;
      out[base + 25] = v6;
      out[base + 26] = v6;
      out[base + 27] = 0xFFFF;
      out[base + 28] = v7;
      out[base + 29] = v7;
      out[base + 30] = v7;
      out[base + 31] = 0xFFFF;
      x += 8;
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

/// AVX2 `grayf32_to_rgb_f32_row`: lossless replicate Y → R=G=B.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `grayf32_to_luma_row`: clamp [0,1] x 255 → u8. 8 px/iter.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn grayf32_to_luma_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = _mm256_set1_ps(255.0);
  let zero = _mm256_setzero_ps();
  let one = _mm256_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = _mm256_castsi256_ps(load_endian_u32x8::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm256_min_ps(_mm256_max_ps(y, zero), one);
      let scaled = _mm256_mul_ps(clamped, scale);
      let int32 = _mm256_cvttps_epi32(_mm256_add_ps(scaled, _mm256_set1_ps(0.5)));
      let lo = _mm256_castsi256_si128(int32);
      let hi = _mm256_extracti128_si256::<1>(int32);
      let pack16 = _mm_packs_epi32(lo, hi);
      let pack8 = _mm_packus_epi16(pack16, pack16);
      let val = _mm_cvtsi128_si64(pack8) as u64;
      out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `grayf32_to_luma_u16_row`: clamp [0,1] x 65535 → u16. 8 px/iter.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn grayf32_to_luma_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = _mm256_set1_ps(65535.0);
  let zero = _mm256_setzero_ps();
  let one = _mm256_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = _mm256_castsi256_ps(load_endian_u32x8::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm256_min_ps(_mm256_max_ps(y, zero), one);
      let scaled = _mm256_mul_ps(clamped, scale);
      let int32 = _mm256_cvttps_epi32(_mm256_add_ps(scaled, _mm256_set1_ps(0.5)));
      let lo = _mm256_castsi256_si128(int32);
      let hi = _mm256_extracti128_si256::<1>(int32);
      let pack16 = _mm_packus_epi32(lo, hi); // 8xu16
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast::<__m128i>(), pack16);
      x += 8;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `grayf32_to_luma_f32_row`: memcpy pass-through.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `grayf32_to_hsv_row`: H=0, S=0, V = clamp(Y,0,1)x255. 8 px/iter.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn grayf32_to_hsv_row<const BE: bool>(
  y_plane: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  let scale = _mm256_set1_ps(255.0);
  let zero = _mm256_setzero_ps();
  let one = _mm256_set1_ps(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = _mm256_castsi256_ps(load_endian_u32x8::<BE>(
        y_plane.as_ptr().cast::<u8>().add(x * 4),
      ));
      let clamped = _mm256_min_ps(_mm256_max_ps(y, zero), one);
      let scaled = _mm256_mul_ps(clamped, scale);
      let int32 = _mm256_cvttps_epi32(_mm256_add_ps(scaled, _mm256_set1_ps(0.5)));
      let lo = _mm256_castsi256_si128(int32);
      let hi = _mm256_extracti128_si256::<1>(int32);
      let pack16 = _mm_packs_epi32(lo, hi);
      let pack8 = _mm_packus_epi16(pack16, pack16);
      let val = _mm_cvtsi128_si64(pack8) as u64;
      let vbytes = val.to_le_bytes();
      h_out[x..x + 8].fill(0);
      s_out[x..x + 8].fill(0);
      v_out[x..x + 8].copy_from_slice(&vbytes);
      x += 8;
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

/// AVX2 `ya8_to_rgb_row`: deinterleave [Y,A] packed u8, broadcast Y → R=G=B.
///
/// Block size: 8 px / iter (16 bytes via SSE, use AVX2 register path falls through
/// to SSE or scalar for the 3-channel scatter).
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `ya8_to_rgba_row`: deinterleave [Y,A], broadcast Y → R=G=B, pass α.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `ya8_to_rgb_u16_row`: zero-extend Y → u16, broadcast R=G=B.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ya8_to_rgb_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  scalar::ya8_to_rgb_u16_row(packed, out, width);
}

/// AVX2 `ya8_to_rgba_u16_row`: zero-extend Y and A → u16.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ya8_to_rgba_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  scalar::ya8_to_rgba_u16_row(packed, out, width);
}

/// AVX2 `ya8_to_luma_row`: extract Y bytes. 8 px/iter.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `ya8_to_luma_u16_row`: zero-extend Y → u16.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn ya8_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  scalar::ya8_to_luma_u16_row(packed, out, width);
}

/// AVX2 `ya8_to_hsv_row`: H=0, S=0, V=Y. α dropped.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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
/// The AVX2 Ya16 SIMD bodies use `_mm256_loadu_si256` + fixed shuffle masks
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

/// AVX2 `ya16_to_rgb_row`: deinterleave [Y,A] u16, Y `>> 8` → u8, broadcast.
///
/// Block size: 4 px / iter (16 bytes = 4 Ya16 pixels via SSE path in AVX2 fn).
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `ya16_to_rgba_row`: Y `>> 8`, A `>> 8`, broadcast Y to R=G=B.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `ya16_to_rgb_u16_row`: native Y u16, broadcast R=G=B.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `ya16_to_rgba_u16_row`: native Y and A u16.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `ya16_to_luma_row`: Y `>> 8` → u8. 4 px/iter.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `ya16_to_luma_u16_row`: native Y pass-through.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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

/// AVX2 `ya16_to_hsv_row`: H=0, S=0, V = Y `>> 8`. α dropped.
///
/// # Safety
/// AVX2 must be available.
#[inline]
#[target_feature(enable = "avx2")]
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
