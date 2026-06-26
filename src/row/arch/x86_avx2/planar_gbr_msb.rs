//! AVX2 kernels for MSB-aligned high-bit planar GBR sources
//! (`AV_PIX_FMT_GBRP10MSB{LE,BE}` / `AV_PIX_FMT_GBRP12MSB{LE,BE}`).
//!
//! The MSB-aligned twins of [`planar_gbr_high_bit`](super::planar_gbr_high_bit).
//! The active sample is in the high `BITS` bits of each `u16`, so recovery is
//! `_mm256_srl_epi16(v, align)` (variable right-shift by `16 - BITS`, a 128-bit
//! count vector shared by the 256- and 128-bit shift forms) rather than the
//! low-bit family's `_mm256_and_si256` mask. These formats have no alpha
//! plane, so only the 3-plane kernels exist.
//!
//! Lane width: 16 pixels per iteration (16 × u16 per `__m256i`), draining an
//! 8-pixel block then a scalar tail.

use super::{endian::load_endian_u16x16, *};
use crate::row::arch::x86_sse41::endian::load_endian_u16x8;

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// AVX2 MSB-aligned G/B/R planar → packed `R, G, B` **bytes**.
/// Recovers each sample (`>> (16 - BITS)`), downshifts by `BITS - 8`, packs
/// to u8.
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
    let align_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero256 = _mm256_setzero_si256();

    let mut x = 0usize;
    while x + 16 <= width {
      let r_v = _mm256_srl_epi16(
        load_endian_u16x16::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_v = _mm256_srl_epi16(
        load_endian_u16x16::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_v = _mm256_srl_epi16(
        load_endian_u16x16::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );

      let r_sh = _mm256_srl_epi16(r_v, shr_count);
      let g_sh = _mm256_srl_epi16(g_v, shr_count);
      let b_sh = _mm256_srl_epi16(b_v, shr_count);

      let r_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(r_sh, zero256));
      let g_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(g_sh, zero256));
      let b_packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(b_sh, zero256));

      let r_u8 = _mm256_castsi256_si128(r_packed);
      let g_u8 = _mm256_castsi256_si128(g_packed);
      let b_u8 = _mm256_castsi256_si128(b_packed);

      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));

      x += 16;
    }
    if x + 8 <= width {
      let zero = _mm_setzero_si128();
      let r_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );
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

/// AVX2 MSB-aligned G/B/R planar → packed `R, G, B, A` **bytes** with constant
/// opaque alpha (`0xFF`).
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
    let align_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero256 = _mm256_setzero_si256();
    let zero = _mm_setzero_si128();
    let opaque_u8 = _mm_set1_epi8(-1i8);

    let mut x = 0usize;
    while x + 16 <= width {
      let r_v = _mm256_srl_epi16(
        load_endian_u16x16::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_v = _mm256_srl_epi16(
        load_endian_u16x16::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_v = _mm256_srl_epi16(
        load_endian_u16x16::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );

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
      let r_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );
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

/// AVX2 MSB-aligned G/B/R planar → packed `R, G, B` **u16** samples.
/// Recovers each sample (`>> (16 - BITS)`), reorders G/B/R → R/G/B. Processes
/// 16 pixels per outer loop via two 8-pixel `write_rgb_u16_8` calls.
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
    let align_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let mut x = 0usize;
    while x + 16 <= width {
      let r_lo = _mm_srl_epi16(
        load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_lo = _mm_srl_epi16(
        load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_lo = _mm_srl_epi16(
        load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );
      write_rgb_u16_8(r_lo, g_lo, b_lo, rgb_u16_out.as_mut_ptr().add(x * 3));

      let r_hi = _mm_srl_epi16(
        load_endian_u16x8::<BE>(r.as_ptr().add(x + 8).cast()),
        align_count,
      );
      let g_hi = _mm_srl_epi16(
        load_endian_u16x8::<BE>(g.as_ptr().add(x + 8).cast()),
        align_count,
      );
      let b_hi = _mm_srl_epi16(
        load_endian_u16x8::<BE>(b.as_ptr().add(x + 8).cast()),
        align_count,
      );
      write_rgb_u16_8(r_hi, g_hi, b_hi, rgb_u16_out.as_mut_ptr().add((x + 8) * 3));

      x += 16;
    }
    if x + 8 <= width {
      let r_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );
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

/// AVX2 MSB-aligned G/B/R planar → packed `R, G, B, A` **u16** samples with
/// constant opaque alpha `(1 << BITS) - 1`.
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
    let align_count = _mm_cvtsi32_si128((16 - BITS) as i32);
    let opaque = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);

    let mut x = 0usize;
    while x + 16 <= width {
      let r_lo = _mm_srl_epi16(
        load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_lo = _mm_srl_epi16(
        load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_lo = _mm_srl_epi16(
        load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );
      write_rgba_u16_8(
        r_lo,
        g_lo,
        b_lo,
        opaque,
        rgba_u16_out.as_mut_ptr().add(x * 4),
      );

      let r_hi = _mm_srl_epi16(
        load_endian_u16x8::<BE>(r.as_ptr().add(x + 8).cast()),
        align_count,
      );
      let g_hi = _mm_srl_epi16(
        load_endian_u16x8::<BE>(g.as_ptr().add(x + 8).cast()),
        align_count,
      );
      let b_hi = _mm_srl_epi16(
        load_endian_u16x8::<BE>(b.as_ptr().add(x + 8).cast()),
        align_count,
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
      let r_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_v = _mm_srl_epi16(
        load_endian_u16x8::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );
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
