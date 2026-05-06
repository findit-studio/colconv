//! SSE4.1 kernels for high-bit-depth planar GBR sources (Tier 10b).
//!
//! All functions are const-generic over `BITS ∈ {9, 10, 12, 14, 16}`.
//! Lane width: 8 pixels per iteration (8 × u16 per `__m128i`).
//! Scalar tail handles the remainder.
//!
//! # u8 output
//!
//! `_mm_srl_epi16(v, count_vec)` (variable-count shift) right-shifts each
//! u16 lane by `BITS - 8`. `_mm_srli_epi16::<IMM8>` requires a literal
//! const generic that is not yet stable for `BITS - 8`, so we use
//! `_mm_srl_epi16` with a count built via `_mm_cvtsi32_si128(BITS - 8)`.
//! Then `_mm_packus_epi16(shifted, zero)` packs 8 u16 → 8 u8 (low half).
//!
//! # u16 output
//!
//! Use the existing `write_rgb_u16_8` / `write_rgba_u16_8` helpers from
//! `x86_common` which interleave 8 u16 lanes per channel into packed
//! RGB / RGBA u16 output.

use core::arch::x86_64::*;

use super::*;

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// SSE4.1 high-bit-depth G/B/R planar → packed `R, G, B` **bytes**.
/// Downshifts each sample by `BITS - 8` and packs to u8.
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbr_to_rgb_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  // SAFETY: SSE4.1 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    // Build a count vector for _mm_srl_epi16 (variable-count shift).
    // `_mm_srli_epi16::<IMM8>` requires a literal const — not stable for
    // `BITS - 8` as a const-generic expression. See SSE4.1 yuv high-bit
    // kernel for the same pattern.
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero = _mm_setzero_si128();
    let mask_v = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = _mm_and_si128(_mm_loadu_si128(r.as_ptr().add(x).cast()), mask_v);
      let g_v = _mm_and_si128(_mm_loadu_si128(g.as_ptr().add(x).cast()), mask_v);
      let b_v = _mm_and_si128(_mm_loadu_si128(b.as_ptr().add(x).cast()), mask_v);

      // Variable-count logical right-shift by BITS-8 per u16 lane.
      let r_sh = _mm_srl_epi16(r_v, shr_count);
      let g_sh = _mm_srl_epi16(g_v, shr_count);
      let b_sh = _mm_srl_epi16(b_v, shr_count);

      // Pack u16x8 + zero → u8x16 (8 valid bytes in the low half).
      let r_u8 = _mm_packus_epi16(r_sh, zero);
      let g_u8 = _mm_packus_epi16(g_sh, zero);
      let b_u8 = _mm_packus_epi16(b_sh, zero);

      // write_rgb_16 writes 16 pixels (48 bytes); only the first 8 pixels
      // (24 bytes) are valid. Write to a temp buffer and copy 24 bytes.
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

/// SSE4.1 high-bit-depth G/B/R planar → packed `R, G, B, A` **bytes**
/// with constant opaque alpha (`0xFF`).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbr_to_rgba_opaque_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: SSE4.1 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero = _mm_setzero_si128();
    let mask_v = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    // Opaque u8 value as u16, packed into u8 with zero upper half.
    let opaque_u16 = _mm_set1_epi16(0x00FF_u16 as i16);
    let opaque_u8 = _mm_packus_epi16(opaque_u16, zero);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = _mm_and_si128(_mm_loadu_si128(r.as_ptr().add(x).cast()), mask_v);
      let g_v = _mm_and_si128(_mm_loadu_si128(g.as_ptr().add(x).cast()), mask_v);
      let b_v = _mm_and_si128(_mm_loadu_si128(b.as_ptr().add(x).cast()), mask_v);

      let r_sh = _mm_srl_epi16(r_v, shr_count);
      let g_sh = _mm_srl_epi16(g_v, shr_count);
      let b_sh = _mm_srl_epi16(b_v, shr_count);

      let r_u8 = _mm_packus_epi16(r_sh, zero);
      let g_u8 = _mm_packus_epi16(g_sh, zero);
      let b_u8 = _mm_packus_epi16(b_sh, zero);

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

/// SSE4.1 high-bit-depth G/B/R/A planar → packed `R, G, B, A` **bytes**.
/// Alpha sourced from the `a` plane, downshifted by `BITS - 8`.
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbra_to_rgba_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: SSE4.1 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero = _mm_setzero_si128();
    let mask_v = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = _mm_and_si128(_mm_loadu_si128(r.as_ptr().add(x).cast()), mask_v);
      let g_v = _mm_and_si128(_mm_loadu_si128(g.as_ptr().add(x).cast()), mask_v);
      let b_v = _mm_and_si128(_mm_loadu_si128(b.as_ptr().add(x).cast()), mask_v);
      let a_v = _mm_and_si128(_mm_loadu_si128(a.as_ptr().add(x).cast()), mask_v);

      let r_sh = _mm_srl_epi16(r_v, shr_count);
      let g_sh = _mm_srl_epi16(g_v, shr_count);
      let b_sh = _mm_srl_epi16(b_v, shr_count);
      let a_sh = _mm_srl_epi16(a_v, shr_count);

      let r_u8 = _mm_packus_epi16(r_sh, zero);
      let g_u8 = _mm_packus_epi16(g_sh, zero);
      let b_u8 = _mm_packus_epi16(b_sh, zero);
      let a_u8 = _mm_packus_epi16(a_sh, zero);

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

/// SSE4.1 high-bit-depth G/B/R planar → packed `R, G, B` **u16** samples.
/// No shift — values copied directly, reordered G/B/R → R/G/B.
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_u16_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbr_to_rgb_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: SSE4.1 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let mask_v = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = _mm_and_si128(_mm_loadu_si128(r.as_ptr().add(x).cast()), mask_v);
      let g_v = _mm_and_si128(_mm_loadu_si128(g.as_ptr().add(x).cast()), mask_v);
      let b_v = _mm_and_si128(_mm_loadu_si128(b.as_ptr().add(x).cast()), mask_v);
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

/// SSE4.1 high-bit-depth G/B/R planar → packed `R, G, B, A` **u16** samples
/// with constant opaque alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbr_to_rgba_opaque_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: SSE4.1 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );

  unsafe {
    // `(1 << BITS) - 1` as u16, reinterpreted as i16 for _mm_set1_epi16.
    let mask_v = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    let opaque = mask_v;

    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = _mm_and_si128(_mm_loadu_si128(r.as_ptr().add(x).cast()), mask_v);
      let g_v = _mm_and_si128(_mm_loadu_si128(g.as_ptr().add(x).cast()), mask_v);
      let b_v = _mm_and_si128(_mm_loadu_si128(b.as_ptr().add(x).cast()), mask_v);
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

/// SSE4.1 high-bit-depth G/B/R/A planar → packed `R, G, B, A` **u16** samples.
/// Alpha sourced from the `a` plane at native depth (no shift).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn gbra_to_rgba_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: SSE4.1 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );

  unsafe {
    let mask_v = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let r_v = _mm_and_si128(_mm_loadu_si128(r.as_ptr().add(x).cast()), mask_v);
      let g_v = _mm_and_si128(_mm_loadu_si128(g.as_ptr().add(x).cast()), mask_v);
      let b_v = _mm_and_si128(_mm_loadu_si128(b.as_ptr().add(x).cast()), mask_v);
      let a_v = _mm_and_si128(_mm_loadu_si128(a.as_ptr().add(x).cast()), mask_v);
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
