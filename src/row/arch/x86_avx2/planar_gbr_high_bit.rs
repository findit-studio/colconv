//! AVX2 kernels for high-bit-depth planar GBR sources (Tier 10b).
//!
//! All functions are const-generic over `BITS ∈ {9, 10, 12, 14, 16}` and
//! `BE: bool` (endianness of the source u16 planes).
//! Lane width: 16 pixels per iteration (16 × u16 per `__m256i`).
//! Scalar tail handles the remainder.
//!
//! # u8 output strategy
//!
//! `_mm256_srl_epi16(v, count_vec)` (variable-count shift — same pattern as
//! SSE4.1 high-bit kernels, since `_mm256_srli_epi16::<IMM8>` requires a
//! literal const not yet stable for `BITS - 8`). After shifting, the 16
//! u16 lanes are packed to 16 u8 bytes via `_mm256_packus_epi16(lo, zero256)`
//! and then `_mm256_permute4x64_epi64::<0xD8>(packed)` to fix the AVX2
//! cross-lane ordering. The resulting 128-bit low half holds 16 valid u8
//! pixels and is fed to `write_rgb_16` / `write_rgba_16` (16-pixel helpers).
//!
//! # u16 output strategy
//!
//! Process 16 u16 pixels per outer iteration via two calls to the 128-bit
//! `write_rgb_u16_8` / `write_rgba_u16_8` helpers (8 pixels each).
//!
//! # Big-endian (`BE = true`) mode
//!
//! Wide (16-pixel) iterations use `load_endian_u16x16::<BE>` from this
//! backend's own `endian.rs` (256-bit shuffle). 8-pixel tail iterations use
//! `load_endian_u16x8::<BE>` from the SSE4.1 `endian.rs` (128-bit shuffle).
//! Both branches are resolved at monomorphisation time.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use super::{endian::load_endian_u16x16, *};
use crate::row::arch::x86_sse41::endian::load_endian_u16x8;

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// AVX2 high-bit-depth G/B/R planar → packed `R, G, B` **bytes**.
/// Downshifts each sample by `BITS - 8` and packs to u8.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gbr_to_rgb_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  // SAFETY: AVX2 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    // Variable-count shift count as a 128-bit vector (used by both
    // _mm256_srl_epi16 and _mm_srl_epi16 — both accept a 128-bit count).
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero256 = _mm256_setzero_si256();
    let mask256 = _mm256_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    let mask128 = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);

    let mut x = 0usize;
    while x + 16 <= width {
      let r_v = _mm256_and_si256(load_endian_u16x16::<BE>(r.as_ptr().add(x).cast()), mask256);
      let g_v = _mm256_and_si256(load_endian_u16x16::<BE>(g.as_ptr().add(x).cast()), mask256);
      let b_v = _mm256_and_si256(load_endian_u16x16::<BE>(b.as_ptr().add(x).cast()), mask256);

      // Variable-count logical right-shift for all 16 u16 lanes.
      let r_sh = _mm256_srl_epi16(r_v, shr_count);
      let g_sh = _mm256_srl_epi16(g_v, shr_count);
      let b_sh = _mm256_srl_epi16(b_v, shr_count);

      // Pack 16 u16 → 16 u8. AVX2 packus_epi16 interleaves per 128-bit
      // lane; permute to fix ordering → low 128 bits = 16 valid u8 pixels.
      let r_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(r_sh, zero256));
      let g_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(g_sh, zero256));
      let b_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(b_sh, zero256));

      let r_u8 = _mm256_castsi256_si128(r_packed);
      let g_u8 = _mm256_castsi256_si128(g_packed);
      let b_u8 = _mm256_castsi256_si128(b_packed);

      // write_rgb_16 writes exactly 16 pixels (48 bytes).
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }
    // Drain remaining 8-pixel blocks with the SSE-width path.
    if x + 8 <= width {
      let zero = _mm_setzero_si128();
      let r_v = _mm_and_si128(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask128);
      let g_v = _mm_and_si128(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask128);
      let b_v = _mm_and_si128(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask128);
      let r_sh = _mm_srl_epi16(r_v, shr_count);
      let g_sh = _mm_srl_epi16(g_v, shr_count);
      let b_sh = _mm_srl_epi16(b_v, shr_count);
      let r_u8 = _mm_packus_epi16(r_sh, zero);
      let g_u8 = _mm_packus_epi16(g_sh, zero);
      let b_u8 = _mm_packus_epi16(b_sh, zero);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
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

/// AVX2 high-bit-depth G/B/R planar → packed `R, G, B, A` **bytes**
/// with constant opaque alpha (`0xFF`).
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gbr_to_rgba_opaque_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: AVX2 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero256 = _mm256_setzero_si256();
    let zero = _mm_setzero_si128();
    let mask256 = _mm256_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    let mask128 = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    // All 16 alpha bytes must be 0xFF; _mm_set1_epi8(-1) fills all lanes.
    let opaque_u8 = _mm_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 16 <= width {
      let r_v = _mm256_and_si256(load_endian_u16x16::<BE>(r.as_ptr().add(x).cast()), mask256);
      let g_v = _mm256_and_si256(load_endian_u16x16::<BE>(g.as_ptr().add(x).cast()), mask256);
      let b_v = _mm256_and_si256(load_endian_u16x16::<BE>(b.as_ptr().add(x).cast()), mask256);

      let r_sh = _mm256_srl_epi16(r_v, shr_count);
      let g_sh = _mm256_srl_epi16(g_v, shr_count);
      let b_sh = _mm256_srl_epi16(b_v, shr_count);

      let r_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(r_sh, zero256));
      let g_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(g_sh, zero256));
      let b_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(b_sh, zero256));

      let r_u8 = _mm256_castsi256_si128(r_packed);
      let g_u8 = _mm256_castsi256_si128(g_packed);
      let b_u8 = _mm256_castsi256_si128(b_packed);

      write_rgba_16(
        r_u8,
        g_u8,
        b_u8,
        opaque_u8,
        rgba_out.as_mut_ptr().add(x * 4),
      );

      x += 16;
    }
    if x + 8 <= width {
      let r_v = _mm_and_si128(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask128);
      let g_v = _mm_and_si128(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask128);
      let b_v = _mm_and_si128(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask128);
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

/// AVX2 high-bit-depth G/B/R/A planar → packed `R, G, B, A` **bytes**.
/// Alpha sourced from the `a` plane, downshifted by `BITS - 8`.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gbra_to_rgba_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: AVX2 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero256 = _mm256_setzero_si256();
    let zero = _mm_setzero_si128();
    let mask256 = _mm256_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    let mask128 = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);

    let mut x = 0usize;
    while x + 16 <= width {
      let r_v = _mm256_and_si256(load_endian_u16x16::<BE>(r.as_ptr().add(x).cast()), mask256);
      let g_v = _mm256_and_si256(load_endian_u16x16::<BE>(g.as_ptr().add(x).cast()), mask256);
      let b_v = _mm256_and_si256(load_endian_u16x16::<BE>(b.as_ptr().add(x).cast()), mask256);
      let a_v = _mm256_and_si256(load_endian_u16x16::<BE>(a.as_ptr().add(x).cast()), mask256);

      let r_sh = _mm256_srl_epi16(r_v, shr_count);
      let g_sh = _mm256_srl_epi16(g_v, shr_count);
      let b_sh = _mm256_srl_epi16(b_v, shr_count);
      let a_sh = _mm256_srl_epi16(a_v, shr_count);

      let r_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(r_sh, zero256));
      let g_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(g_sh, zero256));
      let b_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(b_sh, zero256));
      let a_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(a_sh, zero256));

      let r_u8 = _mm256_castsi256_si128(r_packed);
      let g_u8 = _mm256_castsi256_si128(g_packed);
      let b_u8 = _mm256_castsi256_si128(b_packed);
      let a_u8 = _mm256_castsi256_si128(a_packed);

      write_rgba_16(r_u8, g_u8, b_u8, a_u8, rgba_out.as_mut_ptr().add(x * 4));

      x += 16;
    }
    if x + 8 <= width {
      let r_v = _mm_and_si128(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask128);
      let g_v = _mm_and_si128(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask128);
      let b_v = _mm_and_si128(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask128);
      let a_v = _mm_and_si128(load_endian_u16x8::<BE>(a.as_ptr().add(x).cast()), mask128);
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

/// AVX2 high-bit-depth G/B/R planar → packed `R, G, B` **u16** samples.
/// No shift — values copied directly, reordered G/B/R → R/G/B.
/// Processes 16 pixels per outer loop via two 8-pixel `write_rgb_u16_8` calls.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_u16_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gbr_to_rgb_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: AVX2 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let mask128 = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      // Two 8-pixel halves using the SSE helper.
      let r_lo = _mm_and_si128(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask128);
      let g_lo = _mm_and_si128(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask128);
      let b_lo = _mm_and_si128(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask128);
      write_rgb_u16_8(r_lo, g_lo, b_lo, rgb_u16_out.as_mut_ptr().add(x * 3));

      let r_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(r.as_ptr().add(x + 8).cast()),
        mask128,
      );
      let g_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(g.as_ptr().add(x + 8).cast()),
        mask128,
      );
      let b_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(b.as_ptr().add(x + 8).cast()),
        mask128,
      );
      write_rgb_u16_8(r_hi, g_hi, b_hi, rgb_u16_out.as_mut_ptr().add((x + 8) * 3));

      x += 16;
    }
    if x + 8 <= width {
      let r_v = _mm_and_si128(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask128);
      let g_v = _mm_and_si128(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask128);
      let b_v = _mm_and_si128(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask128);
      write_rgb_u16_8(r_v, g_v, b_v, rgb_u16_out.as_mut_ptr().add(x * 3));
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

/// AVX2 high-bit-depth G/B/R planar → packed `R, G, B, A` **u16** samples
/// with constant opaque alpha `(1 << BITS) - 1`.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gbr_to_rgba_opaque_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: AVX2 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );

  unsafe {
    let mask128 = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    let opaque = mask128;

    let mut x = 0usize;
    while x + 16 <= width {
      let r_lo = _mm_and_si128(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask128);
      let g_lo = _mm_and_si128(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask128);
      let b_lo = _mm_and_si128(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask128);
      write_rgba_u16_8(
        r_lo,
        g_lo,
        b_lo,
        opaque,
        rgba_u16_out.as_mut_ptr().add(x * 4),
      );

      let r_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(r.as_ptr().add(x + 8).cast()),
        mask128,
      );
      let g_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(g.as_ptr().add(x + 8).cast()),
        mask128,
      );
      let b_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(b.as_ptr().add(x + 8).cast()),
        mask128,
      );
      write_rgba_u16_8(
        r_hi,
        g_hi,
        b_hi,
        opaque,
        rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
      );

      x += 16;
    }
    if x + 8 <= width {
      let r_v = _mm_and_si128(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask128);
      let g_v = _mm_and_si128(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask128);
      let b_v = _mm_and_si128(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask128);
      write_rgba_u16_8(r_v, g_v, b_v, opaque, rgba_u16_out.as_mut_ptr().add(x * 4));
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

/// AVX2 high-bit-depth G/B/R/A planar → packed `R, G, B, A` **u16** samples.
/// Alpha sourced from the `a` plane at native depth (no shift).
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gbra_to_rgba_u16_high_bit_row<const BITS: u32, const BE: bool>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: AVX2 verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );

  unsafe {
    let mask128 = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let r_lo = _mm_and_si128(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask128);
      let g_lo = _mm_and_si128(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask128);
      let b_lo = _mm_and_si128(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask128);
      let a_lo = _mm_and_si128(load_endian_u16x8::<BE>(a.as_ptr().add(x).cast()), mask128);
      write_rgba_u16_8(r_lo, g_lo, b_lo, a_lo, rgba_u16_out.as_mut_ptr().add(x * 4));

      let r_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(r.as_ptr().add(x + 8).cast()),
        mask128,
      );
      let g_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(g.as_ptr().add(x + 8).cast()),
        mask128,
      );
      let b_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(b.as_ptr().add(x + 8).cast()),
        mask128,
      );
      let a_hi = _mm_and_si128(
        load_endian_u16x8::<BE>(a.as_ptr().add(x + 8).cast()),
        mask128,
      );
      write_rgba_u16_8(
        r_hi,
        g_hi,
        b_hi,
        a_hi,
        rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
      );

      x += 16;
    }
    if x + 8 <= width {
      let r_v = _mm_and_si128(load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()), mask128);
      let g_v = _mm_and_si128(load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()), mask128);
      let b_v = _mm_and_si128(load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()), mask128);
      let a_v = _mm_and_si128(load_endian_u16x8::<BE>(a.as_ptr().add(x).cast()), mask128);
      write_rgba_u16_8(r_v, g_v, b_v, a_v, rgba_u16_out.as_mut_ptr().add(x * 4));
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
