//! wasm-simd128 kernels for high-bit-depth planar GBR sources (Tier 10b).
//!
//! All functions are const-generic over `BITS ∈ {9, 10, 12, 14, 16}`.
//! Lane width: 8 pixels per iteration (8 × u16 per `v128`).
//! Scalar tail handles the remainder.
//!
//! # u8 output
//!
//! `u16x8_shr(v, BITS - 8)` shifts all 8 u16 lanes right, then
//! `u8x16_narrow_i16x8(shifted, zero)` narrows to 8 u8 bytes (in the
//! low half of the output v128). The write helpers operate on the
//! narrowed u8 vectors — same swizzle pattern as the 8-bit planar_gbr
//! helpers in this backend.
//!
//! # u16 output
//!
//! Use the existing `write_rgb_u16_8` / `write_rgba_u16_8` helpers from
//! this backend's mod.rs which interleave 8 u16 lanes per channel.

#[cfg(target_feature = "simd128")]
use core::arch::wasm32::*;

#[cfg(target_feature = "simd128")]
use crate::row::scalar;

#[cfg(target_feature = "simd128")]
use super::*;

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// wasm-simd128 high-bit-depth G/B/R planar → packed `R, G, B` **bytes**.
/// Downshifts each sample by `BITS - 8` and narrows to u8.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[cfg(target_feature = "simd128")]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbr_to_rgb_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  // SAFETY: simd128 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let shift = (BITS - 8) as u32;
    let zero = i16x8_splat(0);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = v128_load(r.as_ptr().add(x).cast());
      let g_v = v128_load(g.as_ptr().add(x).cast());
      let b_v = v128_load(b.as_ptr().add(x).cast());

      // Shift right by BITS-8, then narrow u16x8 → u8x8 (in low half).
      let r_sh = u16x8_shr(r_v, shift);
      let g_sh = u16x8_shr(g_v, shift);
      let b_sh = u16x8_shr(b_v, shift);

      // u8x16_narrow_i16x8(lo, hi): lo → low 8 bytes, hi → high 8 bytes.
      // Pass zero for hi to get 8 valid u8 pixels in the low half.
      let r_u8 = u8x16_narrow_i16x8(r_sh, zero);
      let g_u8 = u8x16_narrow_i16x8(g_sh, zero);
      let b_u8 = u8x16_narrow_i16x8(b_sh, zero);

      // write_rgb_16 (from planar_gbr.rs) writes 16-pixel RGB (48 bytes);
      // only the first 8 pixels (24 bytes) are valid since our u8 vectors
      // have data only in the low 8 bytes. Write to a temp buffer and copy.
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);

      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgb_high_bit_row::<BITS>(
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

/// wasm-simd128 high-bit-depth G/B/R planar → packed `R, G, B, A` **bytes**
/// with constant opaque alpha (`0xFF`).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[cfg(target_feature = "simd128")]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbr_to_rgba_opaque_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: simd128 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shift = (BITS - 8) as u32;
    let zero = i16x8_splat(0);
    let opaque_u8 = u8x16_splat(0xFF);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = v128_load(r.as_ptr().add(x).cast());
      let g_v = v128_load(g.as_ptr().add(x).cast());
      let b_v = v128_load(b.as_ptr().add(x).cast());

      let r_sh = u16x8_shr(r_v, shift);
      let g_sh = u16x8_shr(g_v, shift);
      let b_sh = u16x8_shr(b_v, shift);

      let r_u8 = u8x16_narrow_i16x8(r_sh, zero);
      let g_u8 = u8x16_narrow_i16x8(g_sh, zero);
      let b_u8 = u8x16_narrow_i16x8(b_sh, zero);

      // write_rgba_16 writes 16-pixel RGBA (64 bytes); only 32 bytes valid.
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);

      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgba_opaque_high_bit_row::<BITS>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ---- u8 output, 4-channel RGBA with source alpha -------------------------

/// wasm-simd128 high-bit-depth G/B/R/A planar → packed `R, G, B, A` **bytes**.
/// Alpha sourced from the `a` plane, downshifted by `BITS - 8`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[cfg(target_feature = "simd128")]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbra_to_rgba_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: simd128 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shift = (BITS - 8) as u32;
    let zero = i16x8_splat(0);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = v128_load(r.as_ptr().add(x).cast());
      let g_v = v128_load(g.as_ptr().add(x).cast());
      let b_v = v128_load(b.as_ptr().add(x).cast());
      let a_v = v128_load(a.as_ptr().add(x).cast());

      let r_sh = u16x8_shr(r_v, shift);
      let g_sh = u16x8_shr(g_v, shift);
      let b_sh = u16x8_shr(b_v, shift);
      let a_sh = u16x8_shr(a_v, shift);

      let r_u8 = u8x16_narrow_i16x8(r_sh, zero);
      let g_u8 = u8x16_narrow_i16x8(g_sh, zero);
      let b_u8 = u8x16_narrow_i16x8(b_sh, zero);
      let a_u8 = u8x16_narrow_i16x8(a_sh, zero);

      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, a_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);

      x += 8;
    }
    if x < width {
      scalar::gbra_to_rgba_high_bit_row::<BITS>(
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

// ---- u16 output, 3-channel (RGB) ----------------------------------------

/// wasm-simd128 high-bit-depth G/B/R planar → packed `R, G, B` **u16** samples.
/// No shift — values copied directly, reordered G/B/R → R/G/B.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_u16_out.len()` ≥ `3 * width`.
#[cfg(target_feature = "simd128")]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbr_to_rgb_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: simd128 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = v128_load(r.as_ptr().add(x).cast());
      let g_v = v128_load(g.as_ptr().add(x).cast());
      let b_v = v128_load(b.as_ptr().add(x).cast());
      write_rgb_u16_8(r_v, g_v, b_v, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgb_u16_high_bit_row::<BITS>(
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

/// wasm-simd128 high-bit-depth G/B/R planar → packed `R, G, B, A` **u16** samples
/// with constant opaque alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[cfg(target_feature = "simd128")]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbr_to_rgba_opaque_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: simd128 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );

  unsafe {
    let opaque = u16x8_splat(((1u32 << BITS) - 1) as u16);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = v128_load(r.as_ptr().add(x).cast());
      let g_v = v128_load(g.as_ptr().add(x).cast());
      let b_v = v128_load(b.as_ptr().add(x).cast());
      write_rgba_u16_8(r_v, g_v, b_v, opaque, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgba_opaque_u16_high_bit_row::<BITS>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgba_u16_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ---- u16 output, 4-channel RGBA with source alpha ------------------------

/// wasm-simd128 high-bit-depth G/B/R/A planar → packed `R, G, B, A` **u16** samples.
/// Alpha sourced from the `a` plane at native depth (no shift).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[cfg(target_feature = "simd128")]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbra_to_rgba_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: simd128 verified available by caller.
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
      let r_v = v128_load(r.as_ptr().add(x).cast());
      let g_v = v128_load(g.as_ptr().add(x).cast());
      let b_v = v128_load(b.as_ptr().add(x).cast());
      let a_v = v128_load(a.as_ptr().add(x).cast());
      write_rgba_u16_8(r_v, g_v, b_v, a_v, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::gbra_to_rgba_u16_high_bit_row::<BITS>(
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
