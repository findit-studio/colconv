//! AVX2 kernels for planar GBR sources (Tier 10).
//!
//! AVX2's planar→packed-RGB interleave has no widely-used native
//! 256-bit primitive (`vpshufb` works per-lane, but the 3-channel
//! pattern crosses 128-bit lanes); the cleanest approach is to reuse
//! the shared 128-bit [`super::write_rgb_16`] / [`super::write_rgba_16`]
//! helpers twice per iteration and process 32 pixels per outer-loop
//! step. Same scalar tail as the SSE4.1 kernel.

use core::arch::x86_64::*;

use super::*;

/// AVX2 G/B/R planar → packed `R, G, B`. Processes 32 pixels per
/// iteration via two calls to the shared 128-bit
/// [`super::write_rgb_16`] helper.
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation). AVX2 implies SSE4.1
///    (and SSSE3), so the underlying `_mm_shuffle_epi8` in
///    `write_rgb_16` is legal.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 (incl. SSSE3 superset) per caller obligation.
  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let g_lo = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_lo = _mm_loadu_si128(b.as_ptr().add(x).cast());
      let r_lo = _mm_loadu_si128(r.as_ptr().add(x).cast());
      write_rgb_16(r_lo, g_lo, b_lo, rgb_out.as_mut_ptr().add(x * 3));

      let g_hi = _mm_loadu_si128(g.as_ptr().add(x + 16).cast());
      let b_hi = _mm_loadu_si128(b.as_ptr().add(x + 16).cast());
      let r_hi = _mm_loadu_si128(r.as_ptr().add(x + 16).cast());
      write_rgb_16(r_hi, g_hi, b_hi, rgb_out.as_mut_ptr().add((x + 16) * 3));

      x += 32;
    }
    // Process any remaining 16-pixel block before the scalar tail.
    if x + 16 <= width {
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

/// AVX2 G/B/R/A planar → packed `R, G, B, A` (real alpha plane).
///
/// # Safety
///
/// Same as [`gbr_to_rgb_row`] plus `a.len()` ≥ `width`,
/// `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 32 <= width {
      let g_lo = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_lo = _mm_loadu_si128(b.as_ptr().add(x).cast());
      let r_lo = _mm_loadu_si128(r.as_ptr().add(x).cast());
      let a_lo = _mm_loadu_si128(a.as_ptr().add(x).cast());
      write_rgba_16(r_lo, g_lo, b_lo, a_lo, rgba_out.as_mut_ptr().add(x * 4));

      let g_hi = _mm_loadu_si128(g.as_ptr().add(x + 16).cast());
      let b_hi = _mm_loadu_si128(b.as_ptr().add(x + 16).cast());
      let r_hi = _mm_loadu_si128(r.as_ptr().add(x + 16).cast());
      let a_hi = _mm_loadu_si128(a.as_ptr().add(x + 16).cast());
      write_rgba_16(
        r_hi,
        g_hi,
        b_hi,
        a_hi,
        rgba_out.as_mut_ptr().add((x + 16) * 4),
      );

      x += 32;
    }
    if x + 16 <= width {
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

/// AVX2 G/B/R planar → packed `R, G, B, A` with constant `α = 0xFF`.
///
/// # Safety
///
/// Same as [`gbr_to_rgb_row`] plus `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let opaque = _mm_set1_epi8(-1);
    let mut x = 0usize;
    while x + 32 <= width {
      let g_lo = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_lo = _mm_loadu_si128(b.as_ptr().add(x).cast());
      let r_lo = _mm_loadu_si128(r.as_ptr().add(x).cast());
      write_rgba_16(r_lo, g_lo, b_lo, opaque, rgba_out.as_mut_ptr().add(x * 4));

      let g_hi = _mm_loadu_si128(g.as_ptr().add(x + 16).cast());
      let b_hi = _mm_loadu_si128(b.as_ptr().add(x + 16).cast());
      let r_hi = _mm_loadu_si128(r.as_ptr().add(x + 16).cast());
      write_rgba_16(
        r_hi,
        g_hi,
        b_hi,
        opaque,
        rgba_out.as_mut_ptr().add((x + 16) * 4),
      );

      x += 32;
    }
    if x + 16 <= width {
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
