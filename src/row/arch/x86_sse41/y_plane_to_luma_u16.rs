//! SSE4.1 `y_plane_to_luma_u16_row` — zero-extends a u8 Y plane to u16.
//!
//! Processes 16 pixels per iteration: load 16 u8 via `_mm_loadu_si128`,
//! zero-extend low 8 via `_mm_cvtepu8_epi16`, extract high 8 with
//! `_mm_srli_si128::<8>` + `_mm_cvtepu8_epi16`, then store both halves
//! via two `_mm_storeu_si128` calls. Scalar tail delegates to the
//! reference implementation.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(target_arch = "x86_64")]
#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use crate::row::scalar::y_plane_to_luma_u16 as scalar;

/// SSE4.1 zero-extension: `out[x] = plane[x] as u16` for `x in 0..width`.
///
/// Block size: 16 px / iter.
///
/// # Safety
///
/// SSE4.1 must be available. `plane.len() >= width`; `out.len() >= width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn y_plane_to_luma_u16_row(plane: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(out.len() >= width, "out too short");

  let mut x = 0usize;
  // SAFETY: loop guard `x + 16 <= width` plus debug_asserts guarantee both
  // the 16-byte load and the two 16-byte stores stay in bounds.
  unsafe {
    while x + 16 <= width {
      let v = _mm_loadu_si128(plane.as_ptr().add(x).cast());
      // Zero-extend low 8 u8 lanes → 8 u16 lanes.
      let low = _mm_cvtepu8_epi16(v);
      // Shift right by 8 bytes to expose the high 8 u8 lanes in the low half.
      let v_hi = _mm_srli_si128::<8>(v);
      let high = _mm_cvtepu8_epi16(v_hi);
      _mm_storeu_si128(out.as_mut_ptr().add(x).cast(), low);
      _mm_storeu_si128(out.as_mut_ptr().add(x + 8).cast(), high);
      x += 16;
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
  fn sse41_y_plane_to_luma_u16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
