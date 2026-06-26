//! NEON kernels for MSB-aligned high-bit planar GBR sources
//! (`AV_PIX_FMT_GBRP10MSB{LE,BE}` / `AV_PIX_FMT_GBRP12MSB{LE,BE}`).
//!
//! The MSB-aligned twins of [`planar_gbr_high_bit`](super::planar_gbr_high_bit).
//! The active sample is in the high `BITS` bits of each `u16`, so recovery is
//! a logical right-shift by `16 - BITS` (`vshlq_u16` with a negative count)
//! rather than the low-bit family's `vandq_u16` mask. Once recovered the
//! sample is in `[0, (1 << BITS) - 1]` and every downstream step is identical.
//! These formats have no alpha plane, so only the 3-plane kernels exist.
//!
//! Lane width: 8 pixels per iteration (`vld1q_u16` = 8 × u16). `vst3q_u16`
//! / `vst4q_u16` do the interleave; scalar tails handle the remainder.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use crate::row::scalar;

use super::{endian::load_endian_u16x8, miri_compat::*};

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// NEON MSB-aligned G/B/R planar → packed `R, G, B` **bytes**.
/// Recovers each sample (`>> (16 - BITS)`), downshifts by `BITS - 8`, narrows
/// to u8.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr_to_rgb_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let align = vdupq_n_s16(-((16 - BITS) as i16));
    let shr = vdupq_n_s16(-((BITS - 8) as i16));

    let mut x = 0usize;
    while x + 8 <= width {
      let g_v = vshlq_u16(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), align);
      let b_v = vshlq_u16(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), align);
      let r_v = vshlq_u16(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), align);

      let r_sh = vqmovn_u16_compat(vshlq_u16(r_v, shr));
      let g_sh = vqmovn_u16_compat(vshlq_u16(g_v, shr));
      let b_sh = vqmovn_u16_compat(vshlq_u16(b_v, shr));

      vst3_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x8x3_t(r_sh, g_sh, b_sh),
      );

      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgb_msb_row::<BITS, BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

// ---- u8 output, 4-channel RGBA with constant opaque alpha ----------------

/// NEON MSB-aligned G/B/R planar → packed `R, G, B, A` **bytes** with constant
/// opaque alpha (`0xFF`).
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr_to_rgba_opaque_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let align = vdupq_n_s16(-((16 - BITS) as i16));
    let shr = vdupq_n_s16(-((BITS - 8) as i16));
    let opaque = vdup_n_u8(0xFF);

    let mut x = 0usize;
    while x + 8 <= width {
      let g_v = vshlq_u16(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), align);
      let b_v = vshlq_u16(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), align);
      let r_v = vshlq_u16(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), align);

      let r_sh = vqmovn_u16_compat(vshlq_u16(r_v, shr));
      let g_sh = vqmovn_u16_compat(vshlq_u16(g_v, shr));
      let b_sh = vqmovn_u16_compat(vshlq_u16(b_v, shr));

      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r_sh, g_sh, b_sh, opaque),
      );

      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgba_opaque_msb_row::<BITS, BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ---- u16 output, 3-channel (RGB) ----------------------------------------

/// NEON MSB-aligned G/B/R planar → packed `R, G, B` **u16** samples at native
/// depth. Recovers each sample (`>> (16 - BITS)`).
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_u16_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr_to_rgb_u16_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let align = vdupq_n_s16(-((16 - BITS) as i16));
    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = vshlq_u16(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), align);
      let g_v = vshlq_u16(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), align);
      let b_v = vshlq_u16(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), align);
      let triple = uint16x8x3_t(r_v, g_v, b_v);
      vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), triple);
      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgb_u16_msb_row::<BITS, BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgb_u16_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

// ---- u16 output, 4-channel RGBA with constant opaque alpha ---------------

/// NEON MSB-aligned G/B/R planar → packed `R, G, B, A` **u16** samples with
/// constant opaque alpha `(1 << BITS) - 1`.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr_to_rgba_opaque_u16_msb_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );

  unsafe {
    let align = vdupq_n_s16(-((16 - BITS) as i16));
    let opaque = vdupq_n_u16(((1u32 << BITS) - 1) as u16);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = vshlq_u16(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), align);
      let g_v = vshlq_u16(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), align);
      let b_v = vshlq_u16(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), align);
      let quad = uint16x8x4_t(r_v, g_v, b_v, opaque);
      vst4q_u16(rgba_u16_out.as_mut_ptr().add(x * 4), quad);
      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgba_opaque_u16_msb_row::<BITS, BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgba_u16_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}
