//! AVX2 `y_plane_to_luma_u16_row` — zero-extends a u8 Y plane to u16.
//!
//! Processes 32 pixels per iteration using two `_mm_loadu_si128` loads
//! (16 u8 each) followed by `_mm256_cvtepu8_epi16` to produce two
//! `__m256i` vectors of 16 u16 each, stored via `_mm256_storeu_si256`.
//! Scalar tail delegates to the reference implementation.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::row::scalar::y_plane_to_luma_u16 as scalar;

/// AVX2 zero-extension: `out[x] = plane[x] as u16` for `x in 0..width`.
///
/// Block size: 32 px / iter (two 16-px `_mm256_cvtepu8_epi16` calls).
///
/// # Safety
///
/// AVX2 must be available. `plane.len() >= width`; `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn y_plane_to_luma_u16_row(plane: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(out.len() >= width, "out too short");

  let mut x = 0usize;
  // SAFETY: loop guard `x + 32 <= width` plus debug_asserts guarantee both
  // the two 16-byte loads and the two 32-byte stores stay in bounds.
  unsafe {
    while x + 32 <= width {
      // Load two 16-byte chunks; _mm256_cvtepu8_epi16 takes a __m128i.
      let lo_src = _mm_loadu_si128(plane.as_ptr().add(x).cast());
      let hi_src = _mm_loadu_si128(plane.as_ptr().add(x + 16).cast());
      let lo = _mm256_cvtepu8_epi16(lo_src);
      let hi = _mm256_cvtepu8_epi16(hi_src);
      _mm256_storeu_si256(out.as_mut_ptr().add(x).cast(), lo);
      _mm256_storeu_si256(out.as_mut_ptr().add(x + 16).cast(), hi);
      x += 32;
    }
  }

  if x < width {
    scalar::y_plane_to_luma_u16_row(&plane[x..width], &mut out[x..width], width - x);
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use crate::row::scalar::y_plane_to_luma_u16 as scalar;

  fn pseudo_random_u8(out: &mut [u8], seed: u32) {
    let mut state = seed;
    for v in out.iter_mut() {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (state >> 16) as u8;
    }
  }

  const WIDTHS: &[usize] = &[1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 130];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx2_y_plane_to_luma_u16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      pseudo_random_u8(&mut plane, 0xC0FFEE);
      let mut out_simd = std::vec![0u16; w];
      let mut out_scalar = std::vec![0u16; w];
      unsafe { super::y_plane_to_luma_u16_row(&plane, &mut out_simd, w) };
      scalar::y_plane_to_luma_u16_row(&plane, &mut out_scalar, w);
      assert_eq!(out_simd, out_scalar, "width={w}");
    }
  }
}
