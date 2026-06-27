//! wasm `simd128` kernels for 32-bit planar GBR + alpha sources
//! (`AV_PIX_FMT_GBRAP32{LE,BE}`).
//!
//! Lane width: 8 pixels per iteration. Each plane is loaded as two `u32x4`
//! (`load_endian_u32x4`, byte-swapped per `BE`), narrowed `>> 16` /`>> 24`, and
//! packed to a single `u16x8` / `u8x16`(-low-8) channel vector via
//! `u16x8_narrow_i32x4` / `u8x16_narrow_i16x8` (inputs fit their target width
//! after the shift, so the saturating narrows are exact). The narrowed channel
//! vectors feed the shared `write_rgb_u16_8` / `write_rgba_u16_8` /
//! `write_rgb_16` / `write_rgba_16` interleave helpers — the same tail the
//! `Gbrap16` u16 kernels use. Scalar tails handle the remainder.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use super::{endian::load_endian_u32x4, *};
use crate::row::scalar::planar_gbr_32bit as scalar;

/// Load 8 pixels of one `u32` plane, narrow `>> 16`, pack to a `u16x8`.
///
/// # Safety
/// `simd128` available; `ptr` points to ≥ 8 readable `u32`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn narrow_u16x8<const BE: bool>(ptr: *const u32) -> v128 {
  unsafe {
    let lo = load_endian_u32x4::<BE>(ptr.cast());
    let hi = load_endian_u32x4::<BE>(ptr.add(4).cast());
    u16x8_narrow_i32x4(u32x4_shr(lo, 16), u32x4_shr(hi, 16))
  }
}

/// Load 8 pixels of one `u32` plane, narrow `>> 24`, pack to a `u8x16` whose
/// low 8 bytes carry the 8 valid samples.
///
/// # Safety
/// `simd128` available; `ptr` points to ≥ 8 readable `u32`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn narrow_u8x8<const BE: bool>(ptr: *const u32) -> v128 {
  unsafe {
    let lo = load_endian_u32x4::<BE>(ptr.cast());
    let hi = load_endian_u32x4::<BE>(ptr.add(4).cast());
    let u16 = u16x8_narrow_i32x4(u32x4_shr(lo, 24), u32x4_shr(hi, 24));
    u8x16_narrow_i16x8(u16, u16x8_splat(0))
  }
}

/// wasm `gbr32_to_rgb_row`: drop α, `>> 24` → packed `R, G, B` u8.
///
/// # Safety
/// 1. `simd128` available. 2. `g`/`b`/`r` ≥ `width`. 3. `rgb_out` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
      let r = narrow_u8x8::<BE>(r.as_ptr().add(x));
      let g = narrow_u8x8::<BE>(g.as_ptr().add(x));
      let b = narrow_u8x8::<BE>(b.as_ptr().add(x));
      let mut tmp = [0u8; 48];
      write_rgb_16(r, g, b, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
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

/// wasm `gbr32_to_rgb_u16_row`: drop α, `>> 16` → packed `R, G, B` u16.
///
/// # Safety
/// 1. `simd128` available. 2. `g`/`b`/`r` ≥ `width`. 3. `rgb_u16_out` ≥
///    `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    while x + 8 <= width {
      let r = narrow_u16x8::<BE>(r.as_ptr().add(x));
      let g = narrow_u16x8::<BE>(g.as_ptr().add(x));
      let b = narrow_u16x8::<BE>(b.as_ptr().add(x));
      write_rgb_u16_8(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 8;
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

/// wasm `gbra32_to_rgba_row`: `>> 24` all 4 channels → packed `R, G, B, A` u8.
///
/// # Safety
/// 1. `simd128` available. 2. `g`/`b`/`r`/`a` ≥ `width`. 3. `rgba_out` ≥
///    `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
      let r = narrow_u8x8::<BE>(r.as_ptr().add(x));
      let g = narrow_u8x8::<BE>(g.as_ptr().add(x));
      let b = narrow_u8x8::<BE>(b.as_ptr().add(x));
      let a = narrow_u8x8::<BE>(a.as_ptr().add(x));
      let mut tmp = [0u8; 64];
      write_rgba_16(r, g, b, a, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
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

/// wasm `gbra32_to_rgba_u16_row`: `>> 16` all 4 channels → packed
/// `R, G, B, A` u16.
///
/// # Safety
/// 1. `simd128` available. 2. `g`/`b`/`r`/`a` ≥ `width`. 3. `rgba_u16_out` ≥
///    `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    while x + 8 <= width {
      let r = narrow_u16x8::<BE>(r.as_ptr().add(x));
      let g = narrow_u16x8::<BE>(g.as_ptr().add(x));
      let b = narrow_u16x8::<BE>(b.as_ptr().add(x));
      let a = narrow_u16x8::<BE>(a.as_ptr().add(x));
      write_rgba_u16_8(r, g, b, a, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 8;
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
