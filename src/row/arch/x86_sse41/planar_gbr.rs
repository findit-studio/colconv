//! SSE4.1 kernels for planar GBR sources (Tier 10).
//!
//! Reuses the shared [`super::write_rgb_16`] / [`super::write_rgba_16`]
//! helpers from `x86_common` — the planar→packed interleave is exactly
//! the inverse-of-three (or four) shuffle pattern those helpers
//! implement. Each kernel processes 16 pixels per iteration with a
//! scalar tail.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use super::*;

/// SSE4.1 G/B/R planar → packed `R, G, B`.
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbr_to_rgb_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  // SAFETY: SSE4.1 (incl. SSSE3 for `_mm_shuffle_epi8` in
  // `write_rgb_16`) is available per caller obligation. All pointer
  // adds are bounded by the `while x + 16 <= width` condition.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
      let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
      write_rgb_16(r_v, g_v, b_v, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::gbr_to_rgb_row(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// SSE4.1 G/B/R/A planar → packed `R, G, B, A`. Alpha is sourced from
/// the `a` plane (real per-pixel α).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbra_to_rgba_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  a: &[u8],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  // SAFETY: see `gbr_to_rgb_row`.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
      let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
      let a_v = _mm_loadu_si128(a.as_ptr().add(x).cast());
      write_rgba_16(r_v, g_v, b_v, a_v, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::gbra_to_rgba_row(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &a[x..width],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// SSE4.1 G/B/R planar → packed `R, G, B, A` with constant `α = 0xFF`.
/// Used by `Gbrp` (no alpha plane) for the standalone `with_rgba`
/// path.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbr_to_rgba_opaque_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  // SAFETY: see `gbr_to_rgb_row`.
  unsafe {
    let opaque = _mm_set1_epi8(-1); // 0xFF as i8 = -1
    let mut x = 0usize;
    while x + 16 <= width {
      let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
      let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
      write_rgba_16(r_v, g_v, b_v, opaque, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::gbr_to_rgba_opaque_row(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}
