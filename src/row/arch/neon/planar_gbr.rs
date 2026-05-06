//! NEON kernels for planar GBR sources (Tier 10).
//!
//! NEON's `vst3q_u8` / `vst4q_u8` make the planar→packed interleave
//! nearly free — load three (or four) channel vectors and store them
//! interleaved in one shot. Both kernels process 16 pixels per
//! iteration with a scalar tail.

use core::arch::aarch64::*;

use crate::row::scalar;

/// Interleaves three planar G/B/R rows into packed `R, G, B`. Loads
/// each plane as a `uint8x16_t` (16 bytes), assembles a
/// `uint8x16x3_t` in **R, G, B** order, and stores via `vst3q_u8` —
/// which interleaves the three vectors into one packed RGB stream
/// without per-pixel byte arithmetic.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
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

  // SAFETY: NEON is available per caller obligation. All pointer adds
  // are bounded by the `while x + 16 <= width` condition.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let g_v = vld1q_u8(g.as_ptr().add(x));
      let b_v = vld1q_u8(b.as_ptr().add(x));
      let r_v = vld1q_u8(r.as_ptr().add(x));
      let triple = uint8x16x3_t(r_v, g_v, b_v);
      vst3q_u8(rgb_out.as_mut_ptr().add(x * 3), triple);
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

/// Interleaves four planar G/B/R/A rows into packed `R, G, B, A`.
/// Same shape as [`gbr_to_rgb_row`] with `vst4q_u8` and an extra
/// alpha lane.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
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
      let g_v = vld1q_u8(g.as_ptr().add(x));
      let b_v = vld1q_u8(b.as_ptr().add(x));
      let r_v = vld1q_u8(r.as_ptr().add(x));
      let a_v = vld1q_u8(a.as_ptr().add(x));
      let quad = uint8x16x4_t(r_v, g_v, b_v, a_v);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), quad);
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

/// Interleaves three planar G/B/R rows into packed `R, G, B, A` with
/// constant `α = 0xFF`. Used by `Gbrp` (no alpha plane) for the
/// `with_rgba` standalone path.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
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
    let opaque = vdupq_n_u8(0xFF);
    let mut x = 0usize;
    while x + 16 <= width {
      let g_v = vld1q_u8(g.as_ptr().add(x));
      let b_v = vld1q_u8(b.as_ptr().add(x));
      let r_v = vld1q_u8(r.as_ptr().add(x));
      let quad = uint8x16x4_t(r_v, g_v, b_v, opaque);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), quad);
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
