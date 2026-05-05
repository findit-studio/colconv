//! AVX2 gray kernel implementations.
//!
//! Gray → luma / luma_u16 / HSV paths get full AVX2 (32-px / 16-px blocks).
//! Packed-channel interleave paths (RGB, RGBA) delegate to scalar: the
//! 3/4-channel store pattern is verbose without SSSE3 shuffle tables and
//! the scalar implementations auto-vectorize well at -O3.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::x86_64::*;

use crate::row::scalar::{bits_mask, gray as scalar};

// ---- Gray8 ------------------------------------------------------------------

/// AVX2 `gray8_to_rgb_row`.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray8_to_rgb_row(y_plane: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  // 3-channel interleave without SSSE3 cross-lane shuffle is verbose under
  // AVX2. Scalar (which auto-vectorizes) handles this path.
  scalar::gray8_to_rgb_row(y_plane, out, width);
}

/// AVX2 `gray8_to_rgba_row`: broadcast Y into RGBA u8, 32 pixels/iter.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray8_to_rgba_row(y_plane: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  // AVX2: 32 u8 → 128 bytes RGBA (4-ch interleave). Requires manual
  // unpacklo/hi chains. Delegate to scalar for simplicity and correctness.
  scalar::gray8_to_rgba_row(y_plane, out, width);
}

/// AVX2 `gray8_to_hsv_row`: H=0, S=0, V=Y. 32 pixels/iter.
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
) {
  debug_assert!(y_plane.len() >= width);
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
    );
  }
}

// ---- GrayN (const BITS) -----------------------------------------------------

/// AVX2 `gray_n_to_rgb_row<BITS>`.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray_n_to_rgb_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_row::<BITS>(y_plane, out, width);
}

/// AVX2 `gray_n_to_rgba_row<BITS>`.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray_n_to_rgba_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray_n_to_rgba_row::<BITS>(y_plane, out, width);
}

/// AVX2 `gray_n_to_rgb_u16_row<BITS>`.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray_n_to_rgb_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_u16_row::<BITS>(y_plane, out, width);
}

/// AVX2 `gray_n_to_rgba_u16_row<BITS>`.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray_n_to_rgba_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray_n_to_rgba_u16_row::<BITS>(y_plane, out, width);
}

/// AVX2 `gray_n_to_luma_row<BITS>`: mask + shift to u8. 16 pixels/iter.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let mask_v = _mm256_set1_epi16(mask as i16);
    // Use variable-count `_mm256_srl_epi16` since `_mm256_srli_epi16::<IMM8>`
    // requires a literal const generic shift not expressible as `BITS - 8`.
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    while x + 16 <= width {
      let raw = _mm256_loadu_si256(y_plane.as_ptr().add(x).cast());
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
    scalar::gray_n_to_luma_row::<BITS>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `gray_n_to_luma_u16_row<BITS>`: mask, store. 16 pixels/iter.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let mask_v = _mm256_set1_epi16(mask as i16);
    while x + 16 <= width {
      let raw = _mm256_loadu_si256(y_plane.as_ptr().add(x).cast());
      let masked = _mm256_and_si256(raw, mask_v);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), masked);
      x += 16;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_u16_row::<BITS>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `gray_n_to_hsv_row<BITS>`: H=0, S=0, V = mask+shift. 16 pixels/iter.
///
/// # Safety
/// AVX2 must be available. All slices have length >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let mask_v = _mm256_set1_epi16(mask as i16);
    let shr = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero256 = _mm256_setzero_si256();
    while x + 16 <= width {
      let raw = _mm256_loadu_si256(y_plane.as_ptr().add(x).cast());
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
    scalar::gray_n_to_hsv_row::<BITS>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Gray16 -----------------------------------------------------------------

/// AVX2 `gray16_to_rgb_row`.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray16_to_rgb_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_row(y_plane, out, width);
}

/// AVX2 `gray16_to_rgba_row`.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray16_to_rgba_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_row(y_plane, out, width);
}

/// AVX2 `gray16_to_rgb_u16_row`.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray16_to_rgb_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_u16_row(y_plane, out, width);
}

/// AVX2 `gray16_to_rgba_u16_row`.
///
/// # Safety
/// AVX2 must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray16_to_rgba_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_u16_row(y_plane, out, width);
}

/// AVX2 `gray16_to_luma_row`: `>> 8`, pack, store. 16 pixels/iter.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray16_to_luma_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    let zero = _mm256_setzero_si256();
    while x + 16 <= width {
      let raw = _mm256_loadu_si256(y_plane.as_ptr().add(x).cast());
      let shifted = _mm256_srli_epi16(raw, 8);
      // Pack u16x16 → u8x16 with lane-cross fixup.
      let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(shifted, zero));
      let lo = _mm256_castsi256_si128(packed);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), lo);
      x += 16;
    }
  }
  if x < width {
    scalar::gray16_to_luma_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `gray16_to_luma_u16_row`: identity copy. 16 pixels/iter.
///
/// # Safety
/// AVX2 must be available. `y_plane.len() >= width`. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gray16_to_luma_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = _mm256_loadu_si256(y_plane.as_ptr().add(x).cast());
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), y);
      x += 16;
    }
  }
  if x < width {
    scalar::gray16_to_luma_u16_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// AVX2 `gray16_to_hsv_row`: `>> 8`, H=0, S=0, V=Y8. 16 pixels/iter.
///
/// # Safety
/// AVX2 must be available. All slices have length >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx2")]
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
    let zero = _mm256_setzero_si256();
    while x + 16 <= width {
      let raw = _mm256_loadu_si256(y_plane.as_ptr().add(x).cast());
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
  fn avx2_gray8_to_hsv_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      prng(&mut plane, 0x1234);
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
  fn avx2_gray_n_to_luma_row_10bit_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0xABCD_1234);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::gray_n_to_luma_row::<10>(&plane, &mut simd, w) };
      scalar::gray_n_to_luma_row::<10>(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  fn avx2_gray_n_to_luma_u16_row_12bit_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0xDEAD_CAFE);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::gray_n_to_luma_u16_row::<12>(&plane, &mut simd, w) };
      scalar::gray_n_to_luma_u16_row::<12>(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  fn avx2_gray16_to_luma_row_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0xBEEF_CAFE);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::gray16_to_luma_row(&plane, &mut simd, w) };
      scalar::gray16_to_luma_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  fn avx2_gray16_to_luma_u16_row_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0x1234_5678);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::gray16_to_luma_u16_row(&plane, &mut simd, w) };
      scalar::gray16_to_luma_u16_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }
}
