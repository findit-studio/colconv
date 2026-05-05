//! NEON kernels for high-bit-depth planar GBR sources (Tier 10b).
//!
//! All functions are const-generic over `BITS ∈ {9, 10, 12, 14, 16}`.
//! Lane width: 8 pixels per iteration (`vld1q_u16` = 8 × u16).
//! `vst3q_u16` / `vst4q_u16` do the 3-way / 4-way u16 interleave in a
//! single hardware instruction. Scalar tails handle the remainder.
//!
//! # u8 downshift
//!
//! For u8-output kernels, each u16 sample is right-shifted by `BITS - 8`
//! using a negative-count vector shift (`vshlq_u16` with a negative
//! shift), then narrowed with `vqmovn_u16` to u8x8. Two such halves are
//! recombined with `vcombine_u8` before `vst3q_u8` / `vst4q_u8`.

use core::arch::aarch64::*;

use crate::row::scalar;

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// NEON high-bit-depth G/B/R planar → packed `R, G, B` **bytes**.
/// Downshifts each sample by `BITS - 8` and narrows to u8.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr_to_rgb_high_bit_row<const BITS: u32>(
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

    let mut x = 0usize;
    while x + 8 <= width {
      let g_v = vld1q_u16(g.as_ptr().add(x));
      let b_v = vld1q_u16(b.as_ptr().add(x));
      let r_v = vld1q_u16(r.as_ptr().add(x));

      // Right-shift each 8-pixel vector by BITS-8, then narrow to u8x8.
      let r_sh = vqmovn_u16(vshlq_u16(r_v, shr));
      let g_sh = vqmovn_u16(vshlq_u16(g_v, shr));
      let b_sh = vqmovn_u16(vshlq_u16(b_v, shr));

      // Combine two u8x8 halves into u8x16 for vst3q_u8 (needs 16-wide).
      // We only have 8 pixels, so pair with zeros and store as 8 pixels.
      // Use a direct store of the triple via vst3q_u8 with combined.
      // Since vst3q_u8 writes 16 pixels, we combine with zeros and store
      // only 8×3=24 bytes.
      let r_16 = vcombine_u8(r_sh, vdup_n_u8(0));
      let g_16 = vcombine_u8(g_sh, vdup_n_u8(0));
      let b_16 = vcombine_u8(b_sh, vdup_n_u8(0));
      // vst3q_u8 stores 3*16=48 bytes; we'll write exactly 24 of them.
      // To avoid writing past the buffer, check if we have 16 pixels of
      // space. If rgb_out is large enough, do it; otherwise fall back.
      // Instead, use a temp buffer approach for safety.
      let triple = uint8x16x3_t(r_16, g_16, b_16);
      let mut tmp = [0u8; 48];
      vst3q_u8(tmp.as_mut_ptr(), triple);
      // Copy only the first 24 bytes (8 pixels * 3 channels).
      let dst = rgb_out.as_mut_ptr().add(x * 3);
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, 24);

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

/// NEON high-bit-depth G/B/R planar → packed `R, G, B, A` **bytes**
/// with constant opaque alpha (`0xFF`). Used by `Gbrp*` (no alpha plane).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr_to_rgba_opaque_high_bit_row<const BITS: u32>(
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
    let opaque = vdup_n_u8(0xFF);

    let mut x = 0usize;
    while x + 8 <= width {
      let g_v = vld1q_u16(g.as_ptr().add(x));
      let b_v = vld1q_u16(b.as_ptr().add(x));
      let r_v = vld1q_u16(r.as_ptr().add(x));

      let r_sh = vqmovn_u16(vshlq_u16(r_v, shr));
      let g_sh = vqmovn_u16(vshlq_u16(g_v, shr));
      let b_sh = vqmovn_u16(vshlq_u16(b_v, shr));

      let r_16 = vcombine_u8(r_sh, vdup_n_u8(0));
      let g_16 = vcombine_u8(g_sh, vdup_n_u8(0));
      let b_16 = vcombine_u8(b_sh, vdup_n_u8(0));
      let a_16 = vcombine_u8(opaque, vdup_n_u8(0));

      let quad = uint8x16x4_t(r_16, g_16, b_16, a_16);
      let mut tmp = [0u8; 64];
      vst4q_u8(tmp.as_mut_ptr(), quad);
      let dst = rgba_out.as_mut_ptr().add(x * 4);
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, 32);

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

/// NEON high-bit-depth G/B/R/A planar → packed `R, G, B, A` **bytes**.
/// Alpha sourced from the `a` plane, downshifted by `BITS - 8`.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbra_to_rgba_high_bit_row<const BITS: u32>(
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

    let mut x = 0usize;
    while x + 8 <= width {
      let g_v = vld1q_u16(g.as_ptr().add(x));
      let b_v = vld1q_u16(b.as_ptr().add(x));
      let r_v = vld1q_u16(r.as_ptr().add(x));
      let a_v = vld1q_u16(a.as_ptr().add(x));

      let r_sh = vqmovn_u16(vshlq_u16(r_v, shr));
      let g_sh = vqmovn_u16(vshlq_u16(g_v, shr));
      let b_sh = vqmovn_u16(vshlq_u16(b_v, shr));
      let a_sh = vqmovn_u16(vshlq_u16(a_v, shr));

      let r_16 = vcombine_u8(r_sh, vdup_n_u8(0));
      let g_16 = vcombine_u8(g_sh, vdup_n_u8(0));
      let b_16 = vcombine_u8(b_sh, vdup_n_u8(0));
      let a_16 = vcombine_u8(a_sh, vdup_n_u8(0));

      let quad = uint8x16x4_t(r_16, g_16, b_16, a_16);
      let mut tmp = [0u8; 64];
      vst4q_u8(tmp.as_mut_ptr(), quad);
      let dst = rgba_out.as_mut_ptr().add(x * 4);
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, 32);

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

/// NEON high-bit-depth G/B/R planar → packed `R, G, B` **u16** samples.
/// Copies samples without shifting — output values in `[0, (1<<BITS)-1]`.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_u16_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr_to_rgb_u16_high_bit_row<const BITS: u32>(
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
    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = vld1q_u16(r.as_ptr().add(x));
      let g_v = vld1q_u16(g.as_ptr().add(x));
      let b_v = vld1q_u16(b.as_ptr().add(x));
      // vst3q_u16 stores 8×3 = 24 u16 interleaved as R,G,B per pixel.
      let triple = uint16x8x3_t(r_v, g_v, b_v);
      vst3q_u16(rgb_u16_out.as_mut_ptr().add(x * 3), triple);
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

/// NEON high-bit-depth G/B/R planar → packed `R, G, B, A` **u16** samples
/// with constant opaque alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbr_to_rgba_opaque_u16_high_bit_row<const BITS: u32>(
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
    let opaque = vdupq_n_u16(((1u32 << BITS) - 1) as u16);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = vld1q_u16(r.as_ptr().add(x));
      let g_v = vld1q_u16(g.as_ptr().add(x));
      let b_v = vld1q_u16(b.as_ptr().add(x));
      let quad = uint16x8x4_t(r_v, g_v, b_v, opaque);
      vst4q_u16(rgba_u16_out.as_mut_ptr().add(x * 4), quad);
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

/// NEON high-bit-depth G/B/R/A planar → packed `R, G, B, A` **u16** samples.
/// Alpha sourced from the `a` plane at native depth (no shift).
///
/// # Safety
///
/// 1. NEON must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gbra_to_rgba_u16_high_bit_row<const BITS: u32>(
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
    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = vld1q_u16(r.as_ptr().add(x));
      let g_v = vld1q_u16(g.as_ptr().add(x));
      let b_v = vld1q_u16(b.as_ptr().add(x));
      let a_v = vld1q_u16(a.as_ptr().add(x));
      let quad = uint16x8x4_t(r_v, g_v, b_v, a_v);
      vst4q_u16(rgba_u16_out.as_mut_ptr().add(x * 4), quad);
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
