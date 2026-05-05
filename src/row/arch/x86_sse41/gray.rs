//! SSE4.1 gray kernel implementations.
//!
//! Gray output kernels (broadcast, HSV H=S=0 V=Y, depth-shift) don't
//! have complex Q15 chroma math — the bottleneck is memory bandwidth.
//! The scalar kernels already auto-vectorize well with -O3; here we
//! provide explicit SSE4.1 versions that use `_mm_loadu_si128` + store
//! patterns and delegate to scalar for tail handling.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::x86_64::*;

use crate::row::scalar::{bits_mask, gray as scalar};

// ---- Gray8 ------------------------------------------------------------------

/// SSE4.1 `gray8_to_rgb_row`.
///
/// # Safety
/// SSE4.1 must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray8_to_rgb_row(y_plane: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  // SSE4.1 doesn't have a 3-channel interleave store like NEON's vst3q_u8.
  // Use scalar (which auto-vectorizes) for the whole row here, or implement
  // manually with repeated shuffle. We delegate to scalar to stay correct and
  // simple; the dispatch will auto-promote to AVX2 when available.
  scalar::gray8_to_rgb_row(y_plane, out, width);
}

/// SSE4.1 `gray8_to_rgba_row`.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray8_to_rgba_row(y_plane: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  // SSE4.1 4-channel interleave without SSSE3 shuffle tables is verbose;
  // delegate to scalar (which auto-vectorizes well at -O3).
  scalar::gray8_to_rgba_row(y_plane, out, width);
}

/// SSE4.1 `gray8_to_hsv_row`.
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
) {
  debug_assert!(y_plane.len() >= width);
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
    );
  }
}

// ---- GrayN (const BITS) ------------------------------------------------

/// SSE4.1 `gray_n_to_rgb_row<BITS>`: mask, shift to u8, scalar store.
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
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_row::<BITS>(y_plane, out, width);
}

/// SSE4.1 `gray_n_to_rgba_row<BITS>`.
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
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  // SSE4.1 4-channel interleave without SSSE3 shuffle tables is complex;
  // delegate to scalar (which auto-vectorizes well at -O3).
  scalar::gray_n_to_rgba_row::<BITS>(y_plane, out, width);
}

/// SSE4.1 `gray_n_to_rgb_u16_row<BITS>`.
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
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_u16_row::<BITS>(y_plane, out, width);
}

/// SSE4.1 `gray_n_to_rgba_u16_row<BITS>`.
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
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray_n_to_rgba_u16_row::<BITS>(y_plane, out, width);
}

/// SSE4.1 `gray_n_to_luma_row<BITS>`: mask, shift, pack, store.
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
) {
  debug_assert!(y_plane.len() >= width);
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
    );
  }
}

// ---- Gray16 ----------------------------------------------------------

/// SSE4.1 `gray16_to_rgb_row`: `>> 8` → pack → scatter (scalar fallback).
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_rgb_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_row(y_plane, out, width);
}

/// SSE4.1 `gray16_to_rgba_row`: `>> 8` → RGBA u8.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_rgba_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_row(y_plane, out, width);
}

/// SSE4.1 `gray16_to_rgb_u16_row`.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_rgb_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_u16_row(y_plane, out, width);
}

/// SSE4.1 `gray16_to_rgba_u16_row`.
///
/// # Safety
/// SSE4.1 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gray16_to_rgba_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_u16_row(y_plane, out, width);
}

/// SSE4.1 `gray16_to_luma_row`: `>> 8`, pack, store.
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
) {
  debug_assert!(y_plane.len() >= width);
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
    );
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use crate::row::scalar::gray as scalar;

  const WIDTHS: &[usize] = &[1, 7, 8, 15, 16, 17, 32, 33, 64];

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
      unsafe { super::gray8_to_rgb_row(&plane, &mut simd, w) };
      scalar::gray8_to_rgb_row(&plane, &mut scal, w);
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
      unsafe { super::gray8_to_hsv_row(&plane, &mut sh, &mut ss, &mut sv, w) };
      scalar::gray8_to_hsv_row(&plane, &mut rh, &mut rs, &mut rv, w);
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
}
