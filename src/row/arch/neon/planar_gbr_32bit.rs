//! NEON kernels for 32-bit planar GBR + alpha sources
//! (`AV_PIX_FMT_GBRAP32{LE,BE}`).
//!
//! Each `u32` plane is loaded via [`load_endian_u32x4`] (byte-swapped per `BE`
//! at monomorphisation), narrowed `>> 16` (native u16) or `>> 24` (u8), and the
//! per-channel lane vectors are interleaved straight to packed `R, G, B[, A]`
//! via `vst3`/`vst4`. This combines Gray32's u32 narrow with the planar-GBR
//! `vst3`/`vst4` interleave.
//!
//! Lane width: 4 pixels per iteration for u16 outputs (one `u32x4` per plane),
//! 8 pixels for u8 outputs (two `u32x4` per plane combined to a `u16x8` then
//! shifted-narrowed to `u8x8`). Scalar tails handle the remainder.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use crate::row::{arch::neon::endian::load_endian_u32x4, scalar::planar_gbr_32bit as scalar};

/// Narrow one `u32x4` plane chunk `>> 16` into the low 4 lanes of a `u16x4`.
///
/// # Safety
/// NEON available; `ptr` points to ≥ 4 readable `u32`.
#[inline(always)]
unsafe fn narrow_u16x4<const BE: bool>(ptr: *const u32) -> uint16x4_t {
  unsafe { vmovn_u32(vshrq_n_u32::<16>(load_endian_u32x4::<BE>(ptr.cast()))) }
}

/// Narrow 8 pixels of one `u32` plane to `u8x8`: two `u32x4` loads → `>> 16`
/// each → combine to `u16x8` → `>> 8` shift-narrow to `u8x8` (net `>> 24`).
///
/// # Safety
/// NEON available; `ptr` points to ≥ 8 readable `u32`.
#[inline(always)]
unsafe fn narrow_u8x8<const BE: bool>(ptr: *const u32) -> uint8x8_t {
  unsafe {
    let lo = narrow_u16x4::<BE>(ptr);
    let hi = narrow_u16x4::<BE>(ptr.add(4));
    vshrn_n_u16::<8>(vcombine_u16(lo, hi))
  }
}

/// NEON `gbr32_to_rgb_row`: drop α, `>> 24` → packed `R, G, B` u8.
///
/// # Safety
/// 1. NEON available. 2. `g`/`b`/`r` ≥ `width`. 3. `rgb_out` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr32_to_rgb_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let r8 = narrow_u8x8::<BE>(r.as_ptr().add(x));
      let g8 = narrow_u8x8::<BE>(g.as_ptr().add(x));
      let b8 = narrow_u8x8::<BE>(b.as_ptr().add(x));
      vst3_u8(rgb_out.as_mut_ptr().add(x * 3), uint8x8x3_t(r8, g8, b8));
      x += 8;
    }
    if x < width {
      scalar::gbr32_to_rgb_row::<BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// NEON `gbr32_to_rgb_u16_row`: drop α, `>> 16` → packed `R, G, B` u16.
///
/// # Safety
/// 1. NEON available. 2. `g`/`b`/`r` ≥ `width`. 3. `rgb_u16_out` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr32_to_rgb_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let r16 = narrow_u16x4::<BE>(r.as_ptr().add(x));
      let g16 = narrow_u16x4::<BE>(g.as_ptr().add(x));
      let b16 = narrow_u16x4::<BE>(b.as_ptr().add(x));
      vst3_u16(
        rgb_u16_out.as_mut_ptr().add(x * 3),
        uint16x4x3_t(r16, g16, b16),
      );
      x += 4;
    }
    if x < width {
      scalar::gbr32_to_rgb_u16_row::<BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgb_u16_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// NEON `gbra32_to_rgba_row`: `>> 24` all 4 channels → packed `R, G, B, A` u8.
///
/// # Safety
/// 1. NEON available. 2. `g`/`b`/`r`/`a` ≥ `width`. 3. `rgba_out` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbra32_to_rgba_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let r8 = narrow_u8x8::<BE>(r.as_ptr().add(x));
      let g8 = narrow_u8x8::<BE>(g.as_ptr().add(x));
      let b8 = narrow_u8x8::<BE>(b.as_ptr().add(x));
      let a8 = narrow_u8x8::<BE>(a.as_ptr().add(x));
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r8, g8, b8, a8),
      );
      x += 8;
    }
    if x < width {
      scalar::gbra32_to_rgba_row::<BE>(
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

/// NEON `gbra32_to_rgba_u16_row`: `>> 16` all 4 channels → packed
/// `R, G, B, A` u16.
///
/// # Safety
/// 1. NEON available. 2. `g`/`b`/`r`/`a` ≥ `width`. 3. `rgba_u16_out` ≥
///    `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbra32_to_rgba_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let r16 = narrow_u16x4::<BE>(r.as_ptr().add(x));
      let g16 = narrow_u16x4::<BE>(g.as_ptr().add(x));
      let b16 = narrow_u16x4::<BE>(b.as_ptr().add(x));
      let a16 = narrow_u16x4::<BE>(a.as_ptr().add(x));
      vst4_u16(
        rgba_u16_out.as_mut_ptr().add(x * 4),
        uint16x4x4_t(r16, g16, b16, a16),
      );
      x += 4;
    }
    if x < width {
      scalar::gbra32_to_rgba_u16_row::<BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &a[x..width],
        &mut rgba_u16_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}
