//! NEON kernels for high-bit-depth planar GBR sources (Tier 10b).
//!
//! All functions are const-generic over `BITS ∈ {9, 10, 12, 14, 16}` and
//! `BE: bool` (endianness of the source u16 planes).
//! Lane width: 8 pixels per iteration (`vld1q_u16` = 8 x u16).
//! `vst3q_u16` / `vst4q_u16` do the 3-way / 4-way u16 interleave in a
//! single hardware instruction. Scalar tails handle the remainder.
//!
//! # u8 downshift
//!
//! For u8-output kernels, each u16 sample is right-shifted by `BITS - 8`
//! using a negative-count vector shift (`vshlq_u16` with a negative
//! shift), then narrowed with `vqmovn_u16` to u8x8. Two such halves are
//! recombined with `vcombine_u8` before `vst3q_u8` / `vst4q_u8`.
//!
//! # Big-endian (`BE = true`) mode
//!
//! When `BE = true` each 8-pixel NEON load goes through
//! `load_endian_u16x8::<BE>` (defined in `endian.rs`) which applies a
//! per-lane byte-swap via `vrev16q_u8`. The branch is resolved at
//! monomorphisation — `BE = false` compiles to a plain `vld1q_u16`.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use crate::row::scalar;

use super::endian::load_endian_u16x8;

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// NEON high-bit-depth G/B/R planar → packed `R, G, B` **bytes**.
/// Downshifts each sample by `BITS - 8` and narrows to u8.
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
pub(crate) unsafe fn gbr_to_rgb_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  // SAFETY: NEON verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    // Shift amount as a negative (right) vector shift for vshlq_u16.
    let shr = vdupq_n_s16(-((BITS - 8) as i16));
    let mask_v = vdupq_n_u16(((1u32 << BITS) - 1) as u16);

    let mut x = 0usize;
    while x + 8 <= width {
      let g_raw = load_endian_u16x8::<BE>(g.as_ptr().add(x).cast());
      let b_raw = load_endian_u16x8::<BE>(b.as_ptr().add(x).cast());
      let r_raw = load_endian_u16x8::<BE>(r.as_ptr().add(x).cast());

      let g_v = vandq_u16(g_raw, mask_v);
      let b_v = vandq_u16(b_raw, mask_v);
      let r_v = vandq_u16(r_raw, mask_v);

      // Right-shift each 8-pixel vector by BITS-8, then narrow to u8x8.
      let r_sh = vqmovn_u16(vshlq_u16(r_v, shr));
      let g_sh = vqmovn_u16(vshlq_u16(g_v, shr));
      let b_sh = vqmovn_u16(vshlq_u16(b_v, shr));

      // Direct 8-pixel interleaved store via vst3_u8: 24 bytes written
      // straight to the output. This replaces the previous
      // vcombine_u8 → vst3q_u8 → 48-byte stack temp → 24-byte memcpy
      // dance, which was a workaround for the 16-pixel-wide vst3q_u8
      // when only 8 pixels were available.
      vst3_u8(
        rgb_out.as_mut_ptr().add(x * 3),
        uint8x8x3_t(r_sh, g_sh, b_sh),
      );

      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgb_high_bit_row::<BITS, BE>(
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

/// NEON high-bit-depth G/B/R planar → packed `R, G, B, A` **bytes**
/// with constant opaque alpha (`0xFF`). Used by `Gbrp*` (no alpha plane).
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
pub(crate) unsafe fn gbr_to_rgba_opaque_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: NEON verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shr = vdupq_n_s16(-((BITS - 8) as i16));
    let mask_v = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    let opaque = vdup_n_u8(0xFF);

    let mut x = 0usize;
    while x + 8 <= width {
      let g_raw = load_endian_u16x8::<BE>(g.as_ptr().add(x).cast());
      let b_raw = load_endian_u16x8::<BE>(b.as_ptr().add(x).cast());
      let r_raw = load_endian_u16x8::<BE>(r.as_ptr().add(x).cast());

      let g_v = vandq_u16(g_raw, mask_v);
      let b_v = vandq_u16(b_raw, mask_v);
      let r_v = vandq_u16(r_raw, mask_v);

      let r_sh = vqmovn_u16(vshlq_u16(r_v, shr));
      let g_sh = vqmovn_u16(vshlq_u16(g_v, shr));
      let b_sh = vqmovn_u16(vshlq_u16(b_v, shr));

      // Direct 8-pixel interleaved store via vst4_u8: 32 bytes written
      // straight to the output (replaces the prior vst4q_u8 + 64-byte
      // temp + 32-byte memcpy workaround).
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r_sh, g_sh, b_sh, opaque),
      );

      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgba_opaque_high_bit_row::<BITS, BE>(
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

/// NEON high-bit-depth G/B/R/A planar → packed `R, G, B, A` **bytes**.
/// Alpha sourced from the `a` plane, downshifted by `BITS - 8`.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbra_to_rgba_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: NEON verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shr = vdupq_n_s16(-((BITS - 8) as i16));
    let mask_v = vdupq_n_u16(((1u32 << BITS) - 1) as u16);

    let mut x = 0usize;
    while x + 8 <= width {
      let g_raw = load_endian_u16x8::<BE>(g.as_ptr().add(x).cast());
      let b_raw = load_endian_u16x8::<BE>(b.as_ptr().add(x).cast());
      let r_raw = load_endian_u16x8::<BE>(r.as_ptr().add(x).cast());
      let a_raw = load_endian_u16x8::<BE>(a.as_ptr().add(x).cast());

      let g_v = vandq_u16(g_raw, mask_v);
      let b_v = vandq_u16(b_raw, mask_v);
      let r_v = vandq_u16(r_raw, mask_v);
      let a_v = vandq_u16(a_raw, mask_v);

      let r_sh = vqmovn_u16(vshlq_u16(r_v, shr));
      let g_sh = vqmovn_u16(vshlq_u16(g_v, shr));
      let b_sh = vqmovn_u16(vshlq_u16(b_v, shr));
      let a_sh = vqmovn_u16(vshlq_u16(a_v, shr));

      // Direct 8-pixel interleaved store via vst4_u8: 32 bytes written
      // straight to the output (replaces the prior vst4q_u8 + 64-byte
      // temp + 32-byte memcpy workaround).
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r_sh, g_sh, b_sh, a_sh),
      );

      x += 8;
    }
    if x < width {
      scalar::gbra_to_rgba_high_bit_row::<BITS, BE>(
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

/// NEON high-bit-depth G/B/R planar → packed `R, G, B` **u16** samples.
/// Copies samples without shifting — output values in `[0, (1<<BITS)-1]`.
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
pub(crate) unsafe fn gbr_to_rgb_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: NEON verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let mask_v = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = vandq_u16(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask_v);
      let g_v = vandq_u16(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask_v);
      let b_v = vandq_u16(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask_v);
      // vst3q_u16 stores 8x3 = 24 u16 interleaved as R,G,B per pixel.
      let triple = uint16x8x3_t(r_v, g_v, b_v);
      vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), triple);
      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgb_u16_high_bit_row::<BITS, BE>(
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

/// NEON high-bit-depth G/B/R planar → packed `R, G, B, A` **u16** samples
/// with constant opaque alpha `(1 << BITS) - 1`.
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
pub(crate) unsafe fn gbr_to_rgba_opaque_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: NEON verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );

  unsafe {
    let mask_v = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    let opaque = mask_v;

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = vandq_u16(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask_v);
      let g_v = vandq_u16(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask_v);
      let b_v = vandq_u16(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask_v);
      let quad = uint16x8x4_t(r_v, g_v, b_v, opaque);
      vst4q_u16(rgba_u16_out.as_mut_ptr().add(x * 4), quad);
      x += 8;
    }
    if x < width {
      scalar::gbr_to_rgba_opaque_u16_high_bit_row::<BITS, BE>(
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

/// NEON high-bit-depth G/B/R/A planar → packed `R, G, B, A` **u16** samples.
/// Alpha sourced from the `a` plane at native depth (no shift).
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbra_to_rgba_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: NEON verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );

  unsafe {
    let mask_v = vdupq_n_u16(((1u32 << BITS) - 1) as u16);
    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = vandq_u16(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask_v);
      let g_v = vandq_u16(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask_v);
      let b_v = vandq_u16(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask_v);
      let a_v = vandq_u16(load_endian_u16x8::<BE>(a.as_ptr().add(x).cast()), mask_v);
      let quad = uint16x8x4_t(r_v, g_v, b_v, a_v);
      vst4q_u16(rgba_u16_out.as_mut_ptr().add(x * 4), quad);
      x += 8;
    }
    if x < width {
      scalar::gbra_to_rgba_u16_high_bit_row::<BITS, BE>(
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
