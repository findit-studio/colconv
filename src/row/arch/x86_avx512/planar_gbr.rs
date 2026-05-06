//! AVX-512 (F + BW) kernels for planar GBR sources (Tier 10).
//!
//! AVX-512 has no clean cross-lane planar→packed-RGB primitive
//! either; reuse the shared 128-bit
//! [`super::write_rgb_16`] / [`super::write_rgba_16`] helpers four
//! times per 64-pixel iteration. Same scalar tail as SSE4.1 / AVX2.

use core::arch::x86_64::*;

use super::*;

/// AVX-512 G/B/R planar → packed `R, G, B`.
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: AVX-512BW (incl. SSSE3 superset) per caller obligation.
  unsafe {
    let mut x = 0usize;
    while x + 64 <= width {
      // Process the four 16-pixel chunks in a 64-pixel block.
      for off in [0, 16, 32, 48] {
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + off).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + off).cast());
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + off).cast());
        write_rgb_16(r_v, g_v, b_v, rgb_out.as_mut_ptr().add((x + off) * 3));
      }
      x += 64;
    }
    // Drain remaining 16-pixel blocks before the scalar tail.
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

/// AVX-512 G/B/R/A planar → packed `R, G, B, A` (real alpha plane).
///
/// # Safety
///
/// Same as [`gbr_to_rgb_row`] plus `a.len()` ≥ `width`,
/// `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 64 <= width {
      for off in [0, 16, 32, 48] {
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + off).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + off).cast());
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + off).cast());
        let a_v = _mm_loadu_si128(a.as_ptr().add(x + off).cast());
        write_rgba_16(r_v, g_v, b_v, a_v, rgba_out.as_mut_ptr().add((x + off) * 4));
      }
      x += 64;
    }
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

/// AVX-512 G/B/R planar → packed `R, G, B, A` with constant `α = 0xFF`.
///
/// # Safety
///
/// Same as [`gbr_to_rgb_row`] plus `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 64 <= width {
      for off in [0, 16, 32, 48] {
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + off).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + off).cast());
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + off).cast());
        write_rgba_16(
          r_v,
          g_v,
          b_v,
          opaque,
          rgba_out.as_mut_ptr().add((x + off) * 4),
        );
      }
      x += 64;
    }
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
