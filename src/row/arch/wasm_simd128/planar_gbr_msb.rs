//! wasm-simd128 kernels for MSB-aligned high-bit planar GBR sources
//! (`AV_PIX_FMT_GBRP10MSB{LE,BE}` / `AV_PIX_FMT_GBRP12MSB{LE,BE}`).
//!
//! The MSB-aligned twins of [`planar_gbr_high_bit`](super::planar_gbr_high_bit).
//! The active sample is in the high `BITS` bits of each `u16`, so recovery is
//! `u16x8_shr(v, 16 - BITS)` rather than the low-bit family's `v128_and` mask.
//! These formats have no alpha plane, so only the 3-plane kernels exist.
//!
//! Lane width: 8 pixels per iteration (8 × u16 per `v128`). Scalar tail
//! handles the remainder.

use core::arch::wasm32::*;

use crate::row::scalar;

use super::{endian::load_endian_u16x8, *};

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// wasm-simd128 MSB-aligned G/B/R planar → packed `R, G, B` **bytes**.
/// Recovers each sample (`>> (16 - BITS)`), downshifts by `BITS - 8`, narrows
/// to u8.
/// When `BE = true`, input u16 lanes are byte-swapped before processing.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    let align = (16 - BITS) as u32;
    let shift = (BITS - 8) as u32;
    let zero = i16x8_splat(0);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = u16x8_shr(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), align);
      let g_v = u16x8_shr(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), align);
      let b_v = u16x8_shr(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), align);

      let r_sh = u16x8_shr(r_v, shift);
      let g_sh = u16x8_shr(g_v, shift);
      let b_sh = u16x8_shr(b_v, shift);

      let r_u8 = u8x16_narrow_i16x8(r_sh, zero);
      let g_u8 = u8x16_narrow_i16x8(g_sh, zero);
      let b_u8 = u8x16_narrow_i16x8(b_sh, zero);

      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);

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

/// wasm-simd128 MSB-aligned G/B/R planar → packed `R, G, B, A` **bytes** with
/// constant opaque alpha (`0xFF`).
/// When `BE = true`, input u16 lanes are byte-swapped before processing.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    let align = (16 - BITS) as u32;
    let shift = (BITS - 8) as u32;
    let zero = i16x8_splat(0);
    let opaque_u8 = u8x16_splat(0xFF);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = u16x8_shr(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), align);
      let g_v = u16x8_shr(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), align);
      let b_v = u16x8_shr(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), align);

      let r_sh = u16x8_shr(r_v, shift);
      let g_sh = u16x8_shr(g_v, shift);
      let b_sh = u16x8_shr(b_v, shift);

      let r_u8 = u8x16_narrow_i16x8(r_sh, zero);
      let g_u8 = u8x16_narrow_i16x8(g_sh, zero);
      let b_u8 = u8x16_narrow_i16x8(b_sh, zero);

      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);

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

/// wasm-simd128 MSB-aligned G/B/R planar → packed `R, G, B` **u16** samples.
/// Recovers each sample (`>> (16 - BITS)`), reorders G/B/R → R/G/B.
/// When `BE = true`, input u16 lanes are byte-swapped before processing.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_u16_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    let align = (16 - BITS) as u32;
    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = u16x8_shr(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), align);
      let g_v = u16x8_shr(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), align);
      let b_v = u16x8_shr(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), align);
      write_rgb_u16_8(r_v, g_v, b_v, rgb_u16_out.as_mut_ptr().add(x * 3));
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

/// wasm-simd128 MSB-aligned G/B/R planar → packed `R, G, B, A` **u16** samples
/// with constant opaque alpha `(1 << BITS) - 1`.
/// When `BE = true`, input u16 lanes are byte-swapped before processing.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    let align = (16 - BITS) as u32;
    let opaque = u16x8_splat(((1u32 << BITS) - 1) as u16);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = u16x8_shr(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), align);
      let g_v = u16x8_shr(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), align);
      let b_v = u16x8_shr(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), align);
      write_rgba_u16_8(r_v, g_v, b_v, opaque, rgba_u16_out.as_mut_ptr().add(x * 4));
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
