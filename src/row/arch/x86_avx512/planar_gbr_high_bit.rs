//! AVX-512 (F + BW) kernels for high-bit-depth planar GBR sources (Tier 10b).
//!
//! All functions are const-generic over `BITS ∈ {9, 10, 12, 14, 16}`.
//! Lane width: 32 pixels per iteration (32 × u16 per `__m512i`).
//! Scalar tail handles the remainder.
//!
//! NO `_mm512_permutex2var_epi8` (VBMI) — only F+BW tier intrinsics.
//!
//! # u8 output strategy
//!
//! `_mm512_srl_epi16(v, count_vec)` (variable-count, F+BW tier) shifts all
//! 32 u16 lanes right by `BITS - 8`. The count vector is a `__m128i`
//! built via `_mm_cvtsi32_si128(BITS - 8)` — the same pattern used by the
//! SSE4.1 and AVX2 high-bit kernels. After shifting, each 128-bit quarter
//! is extracted with a **literal** constant index (0..=3) as required by
//! `_mm512_extracti32x4_epi32`, packed to u8, and fed to `write_rgb_16` /
//! `write_rgba_16` (8 pixels each → 24 / 32 bytes, temp-buffered).
//!
//! # u16 output strategy
//!
//! Process 32 pixels via four calls to `write_rgb_u16_8` /
//! `write_rgba_u16_8` (8 pixels each, SSE4.1 128-bit helpers).

use core::arch::x86_64::*;

use super::*;

// ---- u8 output, 3-channel (RGB) -----------------------------------------

/// AVX-512 (F+BW) high-bit-depth G/B/R planar → packed `R, G, B` **bytes**.
/// Downshifts each sample by `BITS - 8` and packs to u8.
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbr_to_rgb_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  // SAFETY: AVX-512BW verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    // Variable-count shift using a 128-bit count vector (F+BW pattern).
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero128 = _mm_setzero_si128();

    let mut x = 0usize;
    while x + 32 <= width {
      // Load 32 u16 pixels per plane via 512-bit loads.
      let r_v = _mm512_loadu_si512(r.as_ptr().add(x).cast());
      let g_v = _mm512_loadu_si512(g.as_ptr().add(x).cast());
      let b_v = _mm512_loadu_si512(b.as_ptr().add(x).cast());

      // Shift all 32 u16 lanes right by BITS-8.
      let r_sh = _mm512_srl_epi16(r_v, shr_count);
      let g_sh = _mm512_srl_epi16(g_v, shr_count);
      let b_sh = _mm512_srl_epi16(b_v, shr_count);

      // Extract each 128-bit quarter with a literal IMM2 (required by the intrinsic),
      // pack to u8, and write 8 pixels (24 bytes) per quarter.
      // Quarter 0
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 0);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 0);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 0);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 48];
        write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      }
      // Quarter 1
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 1);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 1);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 1);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 48];
        write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add((x + 8) * 3), 24);
      }
      // Quarter 2
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 2);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 2);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 2);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 48];
        write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add((x + 16) * 3), 24);
      }
      // Quarter 3
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 3);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 3);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 3);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 48];
        write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add((x + 24) * 3), 24);
      }

      x += 32;
    }
    // Drain remaining 8-pixel blocks before scalar tail.
    while x + 8 <= width {
      let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
      let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
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

/// AVX-512 (F+BW) high-bit-depth G/B/R planar → packed `R, G, B, A` **bytes**
/// with constant opaque alpha (`0xFF`).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbr_to_rgba_opaque_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: AVX-512BW verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero128 = _mm_setzero_si128();
    let opaque_u16 = _mm_set1_epi16(0x00FF_u16 as i16);
    let opaque_u8 = _mm_packus_epi16(opaque_u16, zero128);

    let mut x = 0usize;
    while x + 32 <= width {
      let r_v = _mm512_loadu_si512(r.as_ptr().add(x).cast());
      let g_v = _mm512_loadu_si512(g.as_ptr().add(x).cast());
      let b_v = _mm512_loadu_si512(b.as_ptr().add(x).cast());

      let r_sh = _mm512_srl_epi16(r_v, shr_count);
      let g_sh = _mm512_srl_epi16(g_v, shr_count);
      let b_sh = _mm512_srl_epi16(b_v, shr_count);

      // Quarter 0
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 0);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 0);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 0);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      }
      // Quarter 1
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 1);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 1);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 1);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add((x + 8) * 4), 32);
      }
      // Quarter 2
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 2);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 2);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 2);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add((x + 16) * 4), 32);
      }
      // Quarter 3
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 3);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 3);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 3);
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
      let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
      let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
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

/// AVX-512 (F+BW) high-bit-depth G/B/R/A planar → packed `R, G, B, A` **bytes**.
/// Alpha sourced from the `a` plane, downshifted by `BITS - 8`.
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbra_to_rgba_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  // SAFETY: AVX-512BW verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let shr_count = _mm_cvtsi32_si128((BITS - 8) as i32);
    let zero128 = _mm_setzero_si128();

    let mut x = 0usize;
    while x + 32 <= width {
      let r_v = _mm512_loadu_si512(r.as_ptr().add(x).cast());
      let g_v = _mm512_loadu_si512(g.as_ptr().add(x).cast());
      let b_v = _mm512_loadu_si512(b.as_ptr().add(x).cast());
      let a_v = _mm512_loadu_si512(a.as_ptr().add(x).cast());

      let r_sh = _mm512_srl_epi16(r_v, shr_count);
      let g_sh = _mm512_srl_epi16(g_v, shr_count);
      let b_sh = _mm512_srl_epi16(b_v, shr_count);
      let a_sh = _mm512_srl_epi16(a_v, shr_count);

      // Quarter 0
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 0);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 0);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 0);
        let a_q = _mm512_extracti32x4_epi32(a_sh, 0);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let a_u8 = _mm_packus_epi16(a_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, a_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      }
      // Quarter 1
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 1);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 1);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 1);
        let a_q = _mm512_extracti32x4_epi32(a_sh, 1);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let a_u8 = _mm_packus_epi16(a_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, a_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add((x + 8) * 4), 32);
      }
      // Quarter 2
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 2);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 2);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 2);
        let a_q = _mm512_extracti32x4_epi32(a_sh, 2);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let a_u8 = _mm_packus_epi16(a_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, a_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add((x + 16) * 4), 32);
      }
      // Quarter 3
      {
        let r_q = _mm512_extracti32x4_epi32(r_sh, 3);
        let g_q = _mm512_extracti32x4_epi32(g_sh, 3);
        let b_q = _mm512_extracti32x4_epi32(b_sh, 3);
        let a_q = _mm512_extracti32x4_epi32(a_sh, 3);
        let r_u8 = _mm_packus_epi16(r_q, zero128);
        let g_u8 = _mm_packus_epi16(g_q, zero128);
        let b_u8 = _mm_packus_epi16(b_q, zero128);
        let a_u8 = _mm_packus_epi16(a_q, zero128);
        let mut tmp = [0u8; 64];
        write_rgba_16(r_u8, g_u8, b_u8, a_u8, tmp.as_mut_ptr());
        core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add((x + 24) * 4), 32);
      }

      x += 32;
    }
    while x + 8 <= width {
      let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
      let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
      let a_v = _mm_loadu_si128(a.as_ptr().add(x).cast());
      let r_sh = _mm_srl_epi16(r_v, shr_count);
      let g_sh = _mm_srl_epi16(g_v, shr_count);
      let b_sh = _mm_srl_epi16(b_v, shr_count);
      let a_sh = _mm_srl_epi16(a_v, shr_count);
      let r_u8 = _mm_packus_epi16(r_sh, zero128);
      let g_u8 = _mm_packus_epi16(g_sh, zero128);
      let b_u8 = _mm_packus_epi16(b_sh, zero128);
      let a_u8 = _mm_packus_epi16(a_sh, zero128);
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

/// AVX-512 (F+BW) high-bit-depth G/B/R planar → packed `R, G, B` **u16** samples.
/// No shift — values copied directly, reordered G/B/R → R/G/B.
/// Processes 32 pixels per outer loop via four 8-pixel `write_rgb_u16_8` calls.
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_u16_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbr_to_rgb_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: AVX-512BW verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      // Four 8-pixel blocks (offsets 0, 8, 16, 24).
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
        write_rgb_u16_8(r_v, g_v, b_v, rgb_u16_out.as_mut_ptr().add(x * 3));
      }
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + 8).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + 8).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + 8).cast());
        write_rgb_u16_8(r_v, g_v, b_v, rgb_u16_out.as_mut_ptr().add((x + 8) * 3));
      }
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + 16).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + 16).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + 16).cast());
        write_rgb_u16_8(r_v, g_v, b_v, rgb_u16_out.as_mut_ptr().add((x + 16) * 3));
      }
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + 24).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + 24).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + 24).cast());
        write_rgb_u16_8(r_v, g_v, b_v, rgb_u16_out.as_mut_ptr().add((x + 24) * 3));
      }
      x += 32;
    }
    while x + 8 <= width {
      let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
      let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
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

/// AVX-512 (F+BW) high-bit-depth G/B/R planar → packed `R, G, B, A` **u16** samples
/// with constant opaque alpha `(1 << BITS) - 1`.
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbr_to_rgba_opaque_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: AVX-512BW verified available by caller.
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );

  unsafe {
    let opaque = _mm_set1_epi16(((1u32 << BITS) - 1) as u16 as i16);

    let mut x = 0usize;
    while x + 32 <= width {
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
        write_rgba_u16_8(r_v, g_v, b_v, opaque, rgba_u16_out.as_mut_ptr().add(x * 4));
      }
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + 8).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + 8).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + 8).cast());
        write_rgba_u16_8(
          r_v,
          g_v,
          b_v,
          opaque,
          rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
        );
      }
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + 16).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + 16).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + 16).cast());
        write_rgba_u16_8(
          r_v,
          g_v,
          b_v,
          opaque,
          rgba_u16_out.as_mut_ptr().add((x + 16) * 4),
        );
      }
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + 24).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + 24).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + 24).cast());
        write_rgba_u16_8(
          r_v,
          g_v,
          b_v,
          opaque,
          rgba_u16_out.as_mut_ptr().add((x + 24) * 4),
        );
      }
      x += 32;
    }
    while x + 8 <= width {
      let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
      let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
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

/// AVX-512 (F+BW) high-bit-depth G/B/R/A planar → packed `R, G, B, A` **u16** samples.
/// Alpha sourced from the `a` plane at native depth (no shift).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `g.len()`, `b.len()`, `r.len()`, `a.len()` ≥ `width`.
/// 3. `rgba_u16_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbra_to_rgba_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  // SAFETY: AVX-512BW verified available by caller.
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
    while x + 32 <= width {
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
        let a_v = _mm_loadu_si128(a.as_ptr().add(x).cast());
        write_rgba_u16_8(r_v, g_v, b_v, a_v, rgba_u16_out.as_mut_ptr().add(x * 4));
      }
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + 8).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + 8).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + 8).cast());
        let a_v = _mm_loadu_si128(a.as_ptr().add(x + 8).cast());
        write_rgba_u16_8(
          r_v,
          g_v,
          b_v,
          a_v,
          rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
        );
      }
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + 16).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + 16).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + 16).cast());
        let a_v = _mm_loadu_si128(a.as_ptr().add(x + 16).cast());
        write_rgba_u16_8(
          r_v,
          g_v,
          b_v,
          a_v,
          rgba_u16_out.as_mut_ptr().add((x + 16) * 4),
        );
      }
      {
        let r_v = _mm_loadu_si128(r.as_ptr().add(x + 24).cast());
        let g_v = _mm_loadu_si128(g.as_ptr().add(x + 24).cast());
        let b_v = _mm_loadu_si128(b.as_ptr().add(x + 24).cast());
        let a_v = _mm_loadu_si128(a.as_ptr().add(x + 24).cast());
        write_rgba_u16_8(
          r_v,
          g_v,
          b_v,
          a_v,
          rgba_u16_out.as_mut_ptr().add((x + 24) * 4),
        );
      }
      x += 32;
    }
    while x + 8 <= width {
      let r_v = _mm_loadu_si128(r.as_ptr().add(x).cast());
      let g_v = _mm_loadu_si128(g.as_ptr().add(x).cast());
      let b_v = _mm_loadu_si128(b.as_ptr().add(x).cast());
      let a_v = _mm_loadu_si128(a.as_ptr().add(x).cast());
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
