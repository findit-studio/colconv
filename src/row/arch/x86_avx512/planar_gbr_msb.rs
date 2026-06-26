//! AVX-512 (F + BW) kernels for MSB-aligned high-bit planar GBR sources
//! (`AV_PIX_FMT_GBRP10MSB{LE,BE}` / `AV_PIX_FMT_GBRP12MSB{LE,BE}`).
//!
//! The MSB-aligned twins of [`planar_gbr_high_bit`](super::planar_gbr_high_bit).
//! The active sample is in the high `BITS` bits of each `u16`, so recovery is
//! `_mm512_srl_epi16(v, align)` (variable right-shift by `16 - BITS`, a 128-bit
//! count vector) rather than the low-bit family's `_mm512_and_si512` mask.
//! These formats have no alpha plane, so only the 3-plane kernels exist.
//!
//! Lane width: 32 pixels per iteration (32 × u16 per `__m512i`), draining
//! 8-pixel blocks then a scalar tail. NO VBMI — only F+BW tier intrinsics.

use super::{endian::load_endian_u16x32, *};
use crate::row::arch::x86_sse41::endian::load_endian_u16x8;

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// AVX-512 (F+BW) MSB-aligned G/B/R planar → packed `R, G, B` **bytes**.
/// Recovers each sample (`>> (16 - BITS)`), downshifts by `BITS - 8`, packs
/// to u8.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero128 = _mm_setzero_si128();

    let mut x = 0usize;
    while x + 32 <= width {
      let r_v = _mm512_srl_epi16(
        load_endian_u16x32::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_v = _mm512_srl_epi16(
        load_endian_u16x32::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_v = _mm512_srl_epi16(
        load_endian_u16x32::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );

      let r_sh = _mm512_srl_epi16(r_v, shr_count);
      let g_sh = _mm512_srl_epi16(g_v, shr_count);
      let b_sh = _mm512_srl_epi16(b_v, shr_count);

      // Quarter 0
      {
        let r_q = _mm512_extracti32x4_epi32::<0>(r_sh);
        let g_q = _mm512_extracti32x4_epi32::<0>(g_sh);
        let b_q = _mm512_extracti32x4_epi32::<0>(b_sh);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 48];
        write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      }
      // Quarter 1
      {
        let r_q = _mm512_extracti32x4_epi32::<1>(r_sh);
        let g_q = _mm512_extracti32x4_epi32::<1>(g_sh);
        let b_q = _mm512_extracti32x4_epi32::<1>(b_sh);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 48];
        write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add((x + 8) * 3), 24);
      }
      // Quarter 2
      {
        let r_q = _mm512_extracti32x4_epi32::<2>(r_sh);
        let g_q = _mm512_extracti32x4_epi32::<2>(g_sh);
        let b_q = _mm512_extracti32x4_epi32::<2>(b_sh);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 48];
        write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add((x + 16) * 3), 24);
      }
      // Quarter 3
      {
        let r_q = _mm512_extracti32x4_epi32::<3>(r_sh);
        let g_q = _mm512_extracti32x4_epi32::<3>(g_sh);
        let b_q = _mm512_extracti32x4_epi32::<3>(b_sh);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 48];
        write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add((x + 24) * 3), 24);
      }

      x += 32;
    }
    while x + 8 <= width {
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
      let r_u8 = _mm_packus_epi16(r_sh, zero128);
      let g_u8 = _mm_packus_epi16(g_sh, zero128);
      let b_u8 = _mm_packus_epi16(b_sh, zero128);
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

/// AVX-512 (F+BW) MSB-aligned G/B/R planar → packed `R, G, B, A` **bytes**
/// with constant opaque alpha (`0xFF`).
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    let zero128 = _mm_setzero_si128();
    let opaque_u16 = _mm_set1_epi16(0x00FF_u16 as i16);
    let opaque_u8 = _mm_packus_epi16(opaque_u16, zero128);

    let mut x = 0usize;
    while x + 32 <= width {
      let r_v = _mm512_srl_epi16(
        load_endian_u16x32::<BE>(r.as_ptr().add(x).cast()),
        align_count,
      );
      let g_v = _mm512_srl_epi16(
        load_endian_u16x32::<BE>(g.as_ptr().add(x).cast()),
        align_count,
      );
      let b_v = _mm512_srl_epi16(
        load_endian_u16x32::<BE>(b.as_ptr().add(x).cast()),
        align_count,
      );

      let r_sh = _mm512_srl_epi16(r_v, shr_count);
      let g_sh = _mm512_srl_epi16(g_v, shr_count);
      let b_sh = _mm512_srl_epi16(b_v, shr_count);

      // Quarter 0
      {
        let r_q = _mm512_extracti32x4_epi32::<0>(r_sh);
        let g_q = _mm512_extracti32x4_epi32::<0>(g_sh);
        let b_q = _mm512_extracti32x4_epi32::<0>(b_sh);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      }
      // Quarter 1
      {
        let r_q = _mm512_extracti32x4_epi32::<1>(r_sh);
        let g_q = _mm512_extracti32x4_epi32::<1>(g_sh);
        let b_q = _mm512_extracti32x4_epi32::<1>(b_sh);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add((x + 8) * 4), 32);
      }
      // Quarter 2
      {
        let r_q = _mm512_extracti32x4_epi32::<2>(r_sh);
        let g_q = _mm512_extracti32x4_epi32::<2>(g_sh);
        let b_q = _mm512_extracti32x4_epi32::<2>(b_sh);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add((x + 16) * 4), 32);
      }
      // Quarter 3
      {
        let r_q = _mm512_extracti32x4_epi32::<3>(r_sh);
        let g_q = _mm512_extracti32x4_epi32::<3>(g_sh);
        let b_q = _mm512_extracti32x4_epi32::<3>(b_sh);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add((x + 24) * 4), 32);
      }

      x += 32;
    }
    while x + 8 <= width {
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
      let r_u8 = _mm_packus_epi16(r_sh, zero128);
      let g_u8 = _mm_packus_epi16(g_sh, zero128);
      let b_u8 = _mm_packus_epi16(b_sh, zero128);
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

/// AVX-512 (F+BW) MSB-aligned G/B/R planar → packed `R, G, B` **u16** samples.
/// Recovers each sample (`>> (16 - BITS)`), reorders G/B/R → R/G/B. Processes
/// 32 pixels per outer loop via four 8-pixel `write_rgb_u16_8` calls.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_u16_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    // The u16 path uses only the 128-bit SSE helper (8 px/block); the AVX-512
    // tier just unrolls four blocks per 32-pixel stride, matching the sibling.
    let mut x = 0usize;
    while x + 8 <= width {
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

/// AVX-512 (F+BW) MSB-aligned G/B/R planar → packed `R, G, B, A` **u16**
/// samples with constant opaque alpha `(1 << BITS) - 1`.
///
/// When `BE = true` each source u16 element is byte-swapped on load.
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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
    while x + 8 <= width {
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
