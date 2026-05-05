//! AVX-512 (F + BW) `y_plane_to_luma_u16_row` — zero-extends a u8 Y
//! plane to u16.
//!
//! Processes 32 pixels per iteration: loads 32 u8 via `_mm256_loadu_si256`
//! then zero-extends to 32 u16 in one `_mm512_cvtepu8_epi16` instruction,
//! stored via `_mm512_storeu_si512`. Scalar tail delegates to the
//! reference implementation.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::row::scalar::y_plane_to_luma_u16 as scalar;

/// AVX-512 zero-extension: `out[x] = plane[x] as u16` for `x in 0..width`.
///
/// Block size: 32 px / iter (`_mm512_cvtepu8_epi16` from a `__m256i`).
///
/// # Safety
///
/// AVX-512F and AVX-512BW must be available. `plane.len() >= width`;
/// `out.len() >= width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn y_plane_to_luma_u16_row(plane: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(out.len() >= width, "out too short");

  let mut x = 0usize;
  // SAFETY: loop guard `x + 32 <= width` plus debug_asserts guarantee the
  // 32-byte load and 64-byte store stay in bounds.
  unsafe {
    while x + 32 <= width {
      let src = _mm256_loadu_si256(plane.as_ptr().add(x).cast());
      let wide = _mm512_cvtepu8_epi16(src);
      _mm512_storeu_si512(out.as_mut_ptr().add(x).cast(), wide);
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
  fn avx512_y_plane_to_luma_u16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512f") {
      return;
    }
    if !std::arch::is_x86_feature_detected!("avx512bw") {
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
