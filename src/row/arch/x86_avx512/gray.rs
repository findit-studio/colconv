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

use core::arch::x86_64::*;

use crate::row::scalar::{bits_mask, gray as scalar};

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

/// AVX-512 `gray_n_to_rgba_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray_n_to_rgba_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray_n_to_rgba_row::<BITS>(y_plane, out, width, full_range);
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

/// AVX-512 `gray_n_to_rgba_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 `gray_n_to_luma_row<BITS>`: mask + shift → u8. 32 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX-512F+BW must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let mask_v = _mm512_set1_epi16(mask as i16);
    // Use variable-count `_mm512_srl_epi16` since `_mm512_srli_epi16::<IMM8>`
    // requires a literal const generic shift not expressible as `BITS - 8`.
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    while x + 32 <= width {
      let raw = _mm512_loadu_si512(y_plane.as_ptr().add(x).cast());
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
    scalar::gray_n_to_luma_row::<BITS>(&y_plane[x..width], &mut out[x..width], width - x);
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
    let mask_v = _mm512_set1_epi16(mask as i16);
    while x + 32 <= width {
      let raw = _mm512_loadu_si512(y_plane.as_ptr().add(x).cast());
      let masked = _mm512_and_si512(raw, mask_v);
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), masked);
      x += 32;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_u16_row::<BITS>(&y_plane[x..width], &mut out[x..width], width - x);
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
    let mask_v = _mm512_set1_epi16(mask as i16);
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero512 = _mm512_setzero_si512();
    let zero256 = _mm256_setzero_si256();
    while x + 32 <= width {
      let raw = _mm512_loadu_si512(y_plane.as_ptr().add(x).cast());
      let masked = _mm512_and_si512(raw, mask_v);
      let shifted = _mm512_srl_epi16(masked, shr);
      let packed = _mm512_cvtepi16_epi8(shifted);
      // H and S: 32 zero bytes; V: the packed bytes.
      _mm256_storeu_si256(h_out.as_mut_ptr().add(x).cast(), zero256);
      _mm256_storeu_si256(s_out.as_mut_ptr().add(x).cast(), zero256);
      _mm256_storeu_si256(v_out.as_mut_ptr().add(x).cast(), packed);
      let _ = zero512;
      x += 32;
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

/// AVX-512 `gray16_to_rgba_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 `gray16_to_rgb_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 `gray16_to_rgba_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// AVX-512F+BW must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

/// AVX-512 `gray16_to_luma_row`: `>> 8`, pack to u8. 32 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// AVX-512F+BW must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gray16_to_luma_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 32 <= width {
      let raw = _mm512_loadu_si512(y_plane.as_ptr().add(x).cast());
      let shifted = _mm512_srli_epi16(raw, 8);
      let packed = _mm512_cvtepi16_epi8(shifted);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), packed);
      x += 32;
    }
  }
  if x < width {
    scalar::gray16_to_luma_row(&y_plane[x..width], &mut out[x..width], width - x);
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
pub(crate) unsafe fn gray16_to_luma_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 32 <= width {
      let y = _mm512_loadu_si512(y_plane.as_ptr().add(x).cast());
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), y);
      x += 32;
    }
  }
  if x < width {
    scalar::gray16_to_luma_u16_row(&y_plane[x..width], &mut out[x..width], width - x);
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
    let zero256 = _mm256_setzero_si256();
    while x + 32 <= width {
      let raw = _mm512_loadu_si512(y_plane.as_ptr().add(x).cast());
      let shifted = _mm512_srli_epi16(raw, 8);
      let packed = _mm512_cvtepi16_epi8(shifted);
      _mm256_storeu_si256(h_out.as_mut_ptr().add(x).cast(), zero256);
      _mm256_storeu_si256(s_out.as_mut_ptr().add(x).cast(), zero256);
      _mm256_storeu_si256(v_out.as_mut_ptr().add(x).cast(), packed);
      x += 32;
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

#[cfg(all(test, feature = "std"))]
mod tests {
  use crate::row::scalar::gray as scalar;

  const WIDTHS: &[usize] = &[1, 7, 8, 15, 16, 31, 32, 33, 64, 65];

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
  fn avx512_gray8_to_hsv_matches_scalar() {
    if !is_x86_feature_detected!("avx512bw") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      prng(&mut plane, 0xFEED_BEEF);
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
  fn avx512_gray_n_to_luma_u16_row_10bit_matches_scalar() {
    if !is_x86_feature_detected!("avx512bw") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0x1234_ABCD);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::gray_n_to_luma_u16_row::<10>(&plane, &mut simd, w) };
      scalar::gray_n_to_luma_u16_row::<10>(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  fn avx512_gray16_to_luma_row_matches_scalar() {
    if !is_x86_feature_detected!("avx512bw") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0xCAFE_BABE);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::gray16_to_luma_row(&plane, &mut simd, w) };
      scalar::gray16_to_luma_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  fn avx512_gray16_to_luma_u16_row_matches_scalar() {
    if !is_x86_feature_detected!("avx512bw") {
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
  fn avx512_gray8_limited_range_matches_scalar() {
    if !is_x86_feature_detected!("avx512bw") {
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
  fn avx512_gray16_limited_range_matches_scalar() {
    if !is_x86_feature_detected!("avx512bw") {
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
}
