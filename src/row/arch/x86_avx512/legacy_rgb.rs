//! AVX-512 (F + BW) kernels for legacy 16-bit packed-RGB source formats (Tier 7).
//!
//! Six source formats x 4 output variants = 24 kernels. Each format word is a
//! little-endian `u16` at 32 pixels per iteration (`_mm512_loadu_si512` = 32 x u16).
//!
//! # Bit extraction (AVX-512)
//!
//! - **RGB565**: `_mm512_srli_epi16(px, 11)` + `& 0x1F` → R5;
//!   `_mm512_srli_epi16(px, 5)` + `& 0x3F` → G6; `px & 0x1F` → B5.
//! - **BGR565**: same shifts, R↔B swapped (R5 at bits [4:0], B5 at bits [15:11]).
//! - **RGB555**: `_mm512_srli_epi16(px, 10)` + `& 0x1F` → R5;
//!   `_mm512_srli_epi16(px, 5)` + `& 0x1F` → G5; `px & 0x1F` → B5.
//! - **BGR555**: same as RGB555 with R↔B swapped.
//! - **RGB444**: `_mm512_srli_epi16(px, 8)` + `& 0x0F` → R4;
//!   `_mm512_srli_epi16(px, 4)` + `& 0x0F` → G4; `px & 0x0F` → B4.
//! - **BGR444**: same as RGB444 with R↔B swapped.
//!
//! # Channel expansion
//!
//! | Bits | AVX-512 (shift + OR)                                                                 |
//! |------|--------------------------------------------------------------------------------------|
//! | 5    | `_mm512_or_si512(_mm512_slli_epi16(c, 3), _mm512_srli_epi16(c, 2))` → [0, 255]    |
//! | 6    | `_mm512_or_si512(_mm512_slli_epi16(c, 2), _mm512_srli_epi16(c, 4))` → [0, 255]    |
//! | 4    | `_mm512_or_si512(_mm512_slli_epi16(c, 4), c)`                          → [0, 255] |
//!
//! # u8 output strategy
//!
//! After expansion each i16 lane holds a value in `[0, 255]`. Extract four 128-bit
//! quarters via literal-index `_mm512_extracti32x4_epi32::<{0,1,2,3}>`. Pack each
//! quarter with `_mm_packus_epi16(q, zero)` → 8 valid u8 pixels in the low 8 bytes.
//! Write with `write_rgb_16` / `write_rgba_16` (8 pixels = 24 / 32 bytes), using a
//! tmp buffer and `copy_nonoverlapping` to avoid pointer-aliasing issues.
//!
//! # u16 output strategy
//!
//! Skip `_mm_packus_epi16`. Extract four 128-bit quarters of each channel vector
//! and call `write_rgb_u16_8` / `write_rgba_u16_8` (8 pixels each).
//!
//! # Scalar tail
//!
//! When `width % 32 ≠ 0` the remainder is handled by `scalar::legacy_rgb`.
//!
//! NO `_mm512_permutex2var_epi8` (VBMI) — only F+BW tier intrinsics.

use super::*;

// Internal helpers.
/// Expand a vector of 5-bit values in [0, 31] to 8-bit: `(c << 3) | (c >> 2)`.
#[inline(always)]
unsafe fn expand5(c: __m512i) -> __m512i {
  unsafe { _mm512_or_si512(_mm512_slli_epi16(c, 3), _mm512_srli_epi16(c, 2)) }
}

/// Expand a vector of 6-bit values in [0, 63] to 8-bit: `(c << 2) | (c >> 4)`.
#[inline(always)]
unsafe fn expand6(c: __m512i) -> __m512i {
  unsafe { _mm512_or_si512(_mm512_slli_epi16(c, 2), _mm512_srli_epi16(c, 4)) }
}

/// Expand a vector of 4-bit values in [0, 15] to 8-bit: `(c << 4) | c`.
#[inline(always)]
unsafe fn expand4(c: __m512i) -> __m512i {
  unsafe { _mm512_or_si512(_mm512_slli_epi16(c, 4), c) }
}

/// Write 32 expanded u8-range pixels (in 4 quarters of a `__m512i`) as packed RGB bytes.
///
/// Each quarter holds 8 u16 lanes in [0, 255]. Pack each to u8, then write via
/// `write_rgb_16` through a 48-byte tmp buffer.
///
/// # Safety
///
/// `ptr` must point to at least 96 writable bytes (32 pixels x 3 bytes).
/// Caller must be in an `avx512f,avx512bw` target-feature context.
#[inline(always)]
unsafe fn write_rgb_32_from_u16lanes(
  r: __m512i,
  g: __m512i,
  b: __m512i,
  zero128: __m128i,
  ptr: *mut u8,
) {
  unsafe {
    // Quarter 0 — pixels [0..8)
    {
      let rq = _mm512_extracti32x4_epi32::<0>(r);
      let gq = _mm512_extracti32x4_epi32::<0>(g);
      let bq = _mm512_extracti32x4_epi32::<0>(b);
      let r_u8 = _mm_packus_epi16(rq, zero128);
      let g_u8 = _mm_packus_epi16(gq, zero128);
      let b_u8 = _mm_packus_epi16(bq, zero128);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), ptr, 24);
    }
    // Quarter 1 — pixels [8..16)
    {
      let rq = _mm512_extracti32x4_epi32::<1>(r);
      let gq = _mm512_extracti32x4_epi32::<1>(g);
      let bq = _mm512_extracti32x4_epi32::<1>(b);
      let r_u8 = _mm_packus_epi16(rq, zero128);
      let g_u8 = _mm_packus_epi16(gq, zero128);
      let b_u8 = _mm_packus_epi16(bq, zero128);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), ptr.add(24), 24);
    }
    // Quarter 2 — pixels [16..24)
    {
      let rq = _mm512_extracti32x4_epi32::<2>(r);
      let gq = _mm512_extracti32x4_epi32::<2>(g);
      let bq = _mm512_extracti32x4_epi32::<2>(b);
      let r_u8 = _mm_packus_epi16(rq, zero128);
      let g_u8 = _mm_packus_epi16(gq, zero128);
      let b_u8 = _mm_packus_epi16(bq, zero128);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), ptr.add(48), 24);
    }
    // Quarter 3 — pixels [24..32)
    {
      let rq = _mm512_extracti32x4_epi32::<3>(r);
      let gq = _mm512_extracti32x4_epi32::<3>(g);
      let bq = _mm512_extracti32x4_epi32::<3>(b);
      let r_u8 = _mm_packus_epi16(rq, zero128);
      let g_u8 = _mm_packus_epi16(gq, zero128);
      let b_u8 = _mm_packus_epi16(bq, zero128);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), ptr.add(72), 24);
    }
  }
}

/// Write 32 expanded u8-range pixels as packed RGBA bytes (constant opaque alpha).
///
/// # Safety
///
/// `ptr` must point to at least 128 writable bytes (32 pixels x 4 bytes).
/// `alpha_u8` must be a valid `__m128i` of u8 lanes all set to `0xFF`.
/// Caller must be in an `avx512f,avx512bw` target-feature context.
#[inline(always)]
unsafe fn write_rgba_32_from_u16lanes(
  r: __m512i,
  g: __m512i,
  b: __m512i,
  alpha_u8: __m128i,
  zero128: __m128i,
  ptr: *mut u8,
) {
  unsafe {
    // Quarter 0
    {
      let rq = _mm512_extracti32x4_epi32::<0>(r);
      let gq = _mm512_extracti32x4_epi32::<0>(g);
      let bq = _mm512_extracti32x4_epi32::<0>(b);
      let r_u8 = _mm_packus_epi16(rq, zero128);
      let g_u8 = _mm_packus_epi16(gq, zero128);
      let b_u8 = _mm_packus_epi16(bq, zero128);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), ptr, 32);
    }
    // Quarter 1
    {
      let rq = _mm512_extracti32x4_epi32::<1>(r);
      let gq = _mm512_extracti32x4_epi32::<1>(g);
      let bq = _mm512_extracti32x4_epi32::<1>(b);
      let r_u8 = _mm_packus_epi16(rq, zero128);
      let g_u8 = _mm_packus_epi16(gq, zero128);
      let b_u8 = _mm_packus_epi16(bq, zero128);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), ptr.add(32), 32);
    }
    // Quarter 2
    {
      let rq = _mm512_extracti32x4_epi32::<2>(r);
      let gq = _mm512_extracti32x4_epi32::<2>(g);
      let bq = _mm512_extracti32x4_epi32::<2>(b);
      let r_u8 = _mm_packus_epi16(rq, zero128);
      let g_u8 = _mm_packus_epi16(gq, zero128);
      let b_u8 = _mm_packus_epi16(bq, zero128);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), ptr.add(64), 32);
    }
    // Quarter 3
    {
      let rq = _mm512_extracti32x4_epi32::<3>(r);
      let gq = _mm512_extracti32x4_epi32::<3>(g);
      let bq = _mm512_extracti32x4_epi32::<3>(b);
      let r_u8 = _mm_packus_epi16(rq, zero128);
      let g_u8 = _mm_packus_epi16(gq, zero128);
      let b_u8 = _mm_packus_epi16(bq, zero128);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), ptr.add(96), 32);
    }
  }
}

/// Write 32 u16 pixels as packed RGB u16 output via four `write_rgb_u16_8` calls.
///
/// # Safety
///
/// `ptr` must point to at least 96 writable `u16` elements (32 pixels x 3).
/// Caller must be in an `avx512f,avx512bw` target-feature context.
#[inline(always)]
unsafe fn write_rgb_u16_32_quarters(r: __m512i, g: __m512i, b: __m512i, ptr: *mut u16) {
  unsafe {
    write_rgb_u16_8(
      _mm512_castsi512_si128(r),
      _mm512_castsi512_si128(g),
      _mm512_castsi512_si128(b),
      ptr,
    );
    write_rgb_u16_8(
      _mm512_extracti32x4_epi32::<1>(r),
      _mm512_extracti32x4_epi32::<1>(g),
      _mm512_extracti32x4_epi32::<1>(b),
      ptr.add(24),
    );
    write_rgb_u16_8(
      _mm512_extracti32x4_epi32::<2>(r),
      _mm512_extracti32x4_epi32::<2>(g),
      _mm512_extracti32x4_epi32::<2>(b),
      ptr.add(48),
    );
    write_rgb_u16_8(
      _mm512_extracti32x4_epi32::<3>(r),
      _mm512_extracti32x4_epi32::<3>(g),
      _mm512_extracti32x4_epi32::<3>(b),
      ptr.add(72),
    );
  }
}

/// Write 32 u16 pixels as packed RGBA u16 output via four `write_rgba_u16_8` calls.
///
/// # Safety
///
/// `ptr` must point to at least 128 writable `u16` elements (32 pixels x 4).
/// `alpha` is a splatted `__m128i` with all 8 u16 lanes = `0xFFFF`.
/// Caller must be in an `avx512f,avx512bw` target-feature context.
#[inline(always)]
unsafe fn write_rgba_u16_32_quarters(
  r: __m512i,
  g: __m512i,
  b: __m512i,
  alpha: __m128i,
  ptr: *mut u16,
) {
  unsafe {
    write_rgba_u16_8(
      _mm512_castsi512_si128(r),
      _mm512_castsi512_si128(g),
      _mm512_castsi512_si128(b),
      alpha,
      ptr,
    );
    write_rgba_u16_8(
      _mm512_extracti32x4_epi32::<1>(r),
      _mm512_extracti32x4_epi32::<1>(g),
      _mm512_extracti32x4_epi32::<1>(b),
      alpha,
      ptr.add(32),
    );
    write_rgba_u16_8(
      _mm512_extracti32x4_epi32::<2>(r),
      _mm512_extracti32x4_epi32::<2>(g),
      _mm512_extracti32x4_epi32::<2>(b),
      alpha,
      ptr.add(64),
    );
    write_rgba_u16_8(
      _mm512_extracti32x4_epi32::<3>(r),
      _mm512_extracti32x4_epi32::<3>(g),
      _mm512_extracti32x4_epi32::<3>(b),
      alpha,
      ptr.add(96),
    );
  }
}

// RGB565 — R5 G6 B5, bits [15:11]=R, [10:5]=G, [4:0]=B.
/// AVX-512 (F+BW) RGB565 → packed `R, G, B` bytes (32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask_r5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm512_set1_epi16(0x3F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r5 = _mm512_and_si512(_mm512_srli_epi16(px, 11), mask_r5);
      let g6 = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask_g6);
      let b5 = _mm512_and_si512(px, mask_r5);
      write_rgb_32_from_u16lanes(
        expand5(r5),
        expand6(g6),
        expand5(b5),
        zero128,
        rgb_out.as_mut_ptr().add(x * 3),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 (F+BW) RGB565 → packed `R, G, B, A` bytes (α = `0xFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask_r5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm512_set1_epi16(0x3F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r5 = _mm512_and_si512(_mm512_srli_epi16(px, 11), mask_r5);
      let g6 = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask_g6);
      let b5 = _mm512_and_si512(px, mask_r5);
      write_rgba_32_from_u16lanes(
        expand5(r5),
        expand6(g6),
        expand5(b5),
        alpha_u8,
        zero128,
        rgba_out.as_mut_ptr().add(x * 4),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 (F+BW) RGB565 → packed `R, G, B` **u16** (native bit-width, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask_r5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm512_set1_epi16(0x3F_u16 as i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r = _mm512_and_si512(_mm512_srli_epi16(px, 11), mask_r5);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask_g6);
      let b = _mm512_and_si512(px, mask_r5);
      write_rgb_u16_32_quarters(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// AVX-512 (F+BW) RGB565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask_r5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm512_set1_epi16(0x3F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r = _mm512_and_si512(_mm512_srli_epi16(px, 11), mask_r5);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask_g6);
      let b = _mm512_and_si512(px, mask_r5);
      write_rgba_u16_32_quarters(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// BGR565 — B5 G6 R5, bits [15:11]=B, [10:5]=G, [4:0]=R.
/// AVX-512 (F+BW) BGR565 → packed `R, G, B` bytes (output R-first, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask_r5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm512_set1_epi16(0x3F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      // BGR565: B at [15:11], G at [10:5], R at [4:0]
      let b5 = _mm512_and_si512(_mm512_srli_epi16(px, 11), mask_r5);
      let g6 = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask_g6);
      let r5 = _mm512_and_si512(px, mask_r5);
      write_rgb_32_from_u16lanes(
        expand5(r5),
        expand6(g6),
        expand5(b5),
        zero128,
        rgb_out.as_mut_ptr().add(x * 3),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 (F+BW) BGR565 → packed `R, G, B, A` bytes (α = `0xFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask_r5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm512_set1_epi16(0x3F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let b5 = _mm512_and_si512(_mm512_srli_epi16(px, 11), mask_r5);
      let g6 = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask_g6);
      let r5 = _mm512_and_si512(px, mask_r5);
      write_rgba_32_from_u16lanes(
        expand5(r5),
        expand6(g6),
        expand5(b5),
        alpha_u8,
        zero128,
        rgba_out.as_mut_ptr().add(x * 4),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 (F+BW) BGR565 → packed `R, G, B` **u16** (native bit-width, output R-first,
/// 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask_r5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm512_set1_epi16(0x3F_u16 as i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      // BGR565: B at [15:11], G at [10:5], R at [4:0]. Output order: R, G, B.
      let b = _mm512_and_si512(_mm512_srli_epi16(px, 11), mask_r5);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask_g6);
      let r = _mm512_and_si512(px, mask_r5);
      write_rgb_u16_32_quarters(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// AVX-512 (F+BW) BGR565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask_r5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm512_set1_epi16(0x3F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let b = _mm512_and_si512(_mm512_srli_epi16(px, 11), mask_r5);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask_g6);
      let r = _mm512_and_si512(px, mask_r5);
      write_rgba_u16_32_quarters(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// RGB555 — 1X R5 G5 B5, bits [14:10]=R, [9:5]=G, [4:0]=B, bit 15 ignored.
/// AVX-512 (F+BW) RGB555 → packed `R, G, B` bytes (32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r5 = _mm512_and_si512(_mm512_srli_epi16(px, 10), mask5);
      let g5 = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask5);
      let b5 = _mm512_and_si512(px, mask5);
      write_rgb_32_from_u16lanes(
        expand5(r5),
        expand5(g5),
        expand5(b5),
        zero128,
        rgb_out.as_mut_ptr().add(x * 3),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 (F+BW) RGB555 → packed `R, G, B, A` bytes (α = `0xFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r5 = _mm512_and_si512(_mm512_srli_epi16(px, 10), mask5);
      let g5 = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask5);
      let b5 = _mm512_and_si512(px, mask5);
      write_rgba_32_from_u16lanes(
        expand5(r5),
        expand5(g5),
        expand5(b5),
        alpha_u8,
        zero128,
        rgba_out.as_mut_ptr().add(x * 4),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 (F+BW) RGB555 → packed `R, G, B` **u16** (native bit-width, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r = _mm512_and_si512(_mm512_srli_epi16(px, 10), mask5);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask5);
      let b = _mm512_and_si512(px, mask5);
      write_rgb_u16_32_quarters(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// AVX-512 (F+BW) RGB555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r = _mm512_and_si512(_mm512_srli_epi16(px, 10), mask5);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask5);
      let b = _mm512_and_si512(px, mask5);
      write_rgba_u16_32_quarters(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// BGR555 — 1X B5 G5 R5, bits [14:10]=B, [9:5]=G, [4:0]=R, bit 15 ignored.
/// AVX-512 (F+BW) BGR555 → packed `R, G, B` bytes (output R-first, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      // BGR555: B at [14:10], G at [9:5], R at [4:0]
      let b5 = _mm512_and_si512(_mm512_srli_epi16(px, 10), mask5);
      let g5 = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask5);
      let r5 = _mm512_and_si512(px, mask5);
      write_rgb_32_from_u16lanes(
        expand5(r5),
        expand5(g5),
        expand5(b5),
        zero128,
        rgb_out.as_mut_ptr().add(x * 3),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 (F+BW) BGR555 → packed `R, G, B, A` bytes (α = `0xFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let b5 = _mm512_and_si512(_mm512_srli_epi16(px, 10), mask5);
      let g5 = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask5);
      let r5 = _mm512_and_si512(px, mask5);
      write_rgba_32_from_u16lanes(
        expand5(r5),
        expand5(g5),
        expand5(b5),
        alpha_u8,
        zero128,
        rgba_out.as_mut_ptr().add(x * 4),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 (F+BW) BGR555 → packed `R, G, B` **u16** (native bit-width, output R-first,
/// 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      // BGR555: B at [14:10], G at [9:5], R at [4:0]. Output order: R, G, B.
      let b = _mm512_and_si512(_mm512_srli_epi16(px, 10), mask5);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask5);
      let r = _mm512_and_si512(px, mask5);
      write_rgb_u16_32_quarters(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// AVX-512 (F+BW) BGR555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = _mm512_set1_epi16(0x1F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let b = _mm512_and_si512(_mm512_srli_epi16(px, 10), mask5);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 5), mask5);
      let r = _mm512_and_si512(px, mask5);
      write_rgba_u16_32_quarters(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// RGB444 — 4X R4 G4 B4, bits [11:8]=R, [7:4]=G, [3:0]=B, bits [15:12] ignored.
/// AVX-512 (F+BW) RGB444 → packed `R, G, B` bytes (32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = _mm512_set1_epi16(0x0F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r4 = _mm512_and_si512(_mm512_srli_epi16(px, 8), mask4);
      let g4 = _mm512_and_si512(_mm512_srli_epi16(px, 4), mask4);
      let b4 = _mm512_and_si512(px, mask4);
      write_rgb_32_from_u16lanes(
        expand4(r4),
        expand4(g4),
        expand4(b4),
        zero128,
        rgb_out.as_mut_ptr().add(x * 3),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 (F+BW) RGB444 → packed `R, G, B, A` bytes (α = `0xFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = _mm512_set1_epi16(0x0F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r4 = _mm512_and_si512(_mm512_srli_epi16(px, 8), mask4);
      let g4 = _mm512_and_si512(_mm512_srli_epi16(px, 4), mask4);
      let b4 = _mm512_and_si512(px, mask4);
      write_rgba_32_from_u16lanes(
        expand4(r4),
        expand4(g4),
        expand4(b4),
        alpha_u8,
        zero128,
        rgba_out.as_mut_ptr().add(x * 4),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 (F+BW) RGB444 → packed `R, G, B` **u16** (native 4-bit width, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = _mm512_set1_epi16(0x0F_u16 as i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r = _mm512_and_si512(_mm512_srli_epi16(px, 8), mask4);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 4), mask4);
      let b = _mm512_and_si512(px, mask4);
      write_rgb_u16_32_quarters(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// AVX-512 (F+BW) RGB444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgb444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = _mm512_set1_epi16(0x0F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let r = _mm512_and_si512(_mm512_srli_epi16(px, 8), mask4);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 4), mask4);
      let b = _mm512_and_si512(px, mask4);
      write_rgba_u16_32_quarters(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// BGR444 — 4X B4 G4 R4, bits [11:8]=B, [7:4]=G, [3:0]=R, bits [15:12] ignored.
/// AVX-512 (F+BW) BGR444 → packed `R, G, B` bytes (output R-first, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = _mm512_set1_epi16(0x0F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      // BGR444: B at [11:8], G at [7:4], R at [3:0]
      let b4 = _mm512_and_si512(_mm512_srli_epi16(px, 8), mask4);
      let g4 = _mm512_and_si512(_mm512_srli_epi16(px, 4), mask4);
      let r4 = _mm512_and_si512(px, mask4);
      write_rgb_32_from_u16lanes(
        expand4(r4),
        expand4(g4),
        expand4(b4),
        zero128,
        rgb_out.as_mut_ptr().add(x * 3),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 (F+BW) BGR444 → packed `R, G, B, A` bytes (α = `0xFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = _mm512_set1_epi16(0x0F_u16 as i16);
    let zero128 = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let b4 = _mm512_and_si512(_mm512_srli_epi16(px, 8), mask4);
      let g4 = _mm512_and_si512(_mm512_srli_epi16(px, 4), mask4);
      let r4 = _mm512_and_si512(px, mask4);
      write_rgba_32_from_u16lanes(
        expand4(r4),
        expand4(g4),
        expand4(b4),
        alpha_u8,
        zero128,
        rgba_out.as_mut_ptr().add(x * 4),
      );
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 (F+BW) BGR444 → packed `R, G, B` **u16** (native 4-bit width, output R-first,
/// 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = _mm512_set1_epi16(0x0F_u16 as i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      // BGR444: B at [11:8], G at [7:4], R at [3:0]. Output order: R, G, B.
      let b = _mm512_and_si512(_mm512_srli_epi16(px, 8), mask4);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 4), mask4);
      let r = _mm512_and_si512(px, mask4);
      write_rgb_u16_32_quarters(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgb_u16_row(
        &src[x * 2..],
        &mut rgb_u16_out[x * 3..],
        width - x,
      );
    }
  }
}

/// AVX-512 (F+BW) BGR444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 32 px/iter).
///
/// # Safety
///
/// 1. AVX-512BW must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bgr444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = _mm512_set1_epi16(0x0F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let px = _mm512_loadu_si512(src.as_ptr().add(x * 2).cast());
      let b = _mm512_and_si512(_mm512_srli_epi16(px, 8), mask4);
      let g = _mm512_and_si512(_mm512_srli_epi16(px, 4), mask4);
      let r = _mm512_and_si512(px, mask4);
      write_rgba_u16_32_quarters(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 32;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgba_u16_row(
        &src[x * 2..],
        &mut rgba_u16_out[x * 4..],
        width - x,
      );
    }
  }
}

// =========================================================================
// Legacy bit-packed RGB/BGR (8bpp 3:3:2 + 1:2:1; 4bpp 1:2:1 two-per-byte)
// (Rgb8 / Bgr8 / Rgb4Byte / Bgr4Byte — 1 byte/pixel;
//  Rgb4 / Bgr4 — 4 bits/pixel, two pixels per byte).
//
// Each iteration produces 32 pixels as 32 u16 lanes of native source bytes
// (byte formats: widen 32 source bytes via `_mm512_cvtepu8_epi16`; nibble
// formats: de-interleave 16 source bytes into 32 nibble lanes), then reuses
// the same shift+mask extraction, bit-replication expansion, and quarter-wise
// `write_*_32` stores as the 16-bit formats above. The `width % 32` remainder
// defers to `scalar`. The nibble de-interleave uses 256-bit AVX2
// broadcast+shuffle intrinsics, which `avx512f` implies (so the kernels stay
// `avx512f,avx512bw`-declared and gate on `avx512_available()`, matching every
// other AVX-512 backend in this crate — e.g. `yuv_planar_8bit`).
// =========================================================================

/// Bit-replicate u16 lanes of 1-bit values (`0`/`1`) to 8-bit: `c * 0xFF`.
#[inline(always)]
unsafe fn expand1(c: __m512i) -> __m512i {
  unsafe { _mm512_mullo_epi16(c, _mm512_set1_epi16(0xFF)) }
}

/// Bit-replicate u16 lanes of 2-bit values (`0..=3`) to 8-bit: `c * 0x55`.
#[inline(always)]
unsafe fn expand2(c: __m512i) -> __m512i {
  unsafe { _mm512_mullo_epi16(c, _mm512_set1_epi16(0x55)) }
}

/// Bit-replicate u16 lanes of 3-bit values (`0..=7`) to 8-bit:
/// `(c << 5) | (c << 2) | (c >> 1)`.
#[inline(always)]
unsafe fn expand3(c: __m512i) -> __m512i {
  unsafe {
    _mm512_or_si512(
      _mm512_or_si512(_mm512_slli_epi16(c, 5), _mm512_slli_epi16(c, 2)),
      _mm512_srli_epi16(c, 1),
    )
  }
}

/// Load 32 packed 1-byte-per-pixel source bytes and widen to 32 u16 lanes.
///
/// # Safety
///
/// `ptr` valid for a 32-byte read; AVX-512BW available.
#[inline(always)]
unsafe fn load_byte_px32(ptr: *const u8) -> __m512i {
  unsafe { _mm512_cvtepu8_epi16(_mm256_loadu_si256(ptr.cast())) }
}

/// Load 16 packed 2-pixel-per-byte source bytes and de-interleave the nibbles
/// into 32 u16 lanes (even pixel = high nibble `[7:4]`, odd = low nibble).
///
/// Uses 256-bit AVX2 broadcast+shuffle intrinsics, which `avx512f` implies; the
/// caller is `avx512f,avx512bw`-gated like every other AVX-512 kernel.
///
/// # Safety
///
/// `ptr` valid for a 16-byte read; AVX-512BW available.
#[inline(always)]
unsafe fn load_nibble_px32(ptr: *const u8) -> __m512i {
  unsafe {
    let v = _mm_loadu_si128(ptr.cast());
    // Duplicate each of the 16 bytes → [b0, b0, b1, b1, …, b15, b15] (32 bytes).
    let bcast = _mm256_broadcastsi128_si256(v);
    let dupmask = _mm256_setr_epi8(
      0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
      14, 14, 15, 15,
    );
    let dup = _mm256_shuffle_epi8(bcast, dupmask);
    let w = _mm512_cvtepu8_epi16(dup);
    let hi = _mm512_srli_epi16(w, 4);
    let lo = _mm512_and_si512(w, _mm512_set1_epi16(0x0F));
    // Even lanes take the high nibble (mask 0x5555_5555).
    _mm512_mask_blend_epi16(0x5555_5555, lo, hi)
  }
}

/// Emits the four AVX-512 output kernels (rgb / rgba / rgb_u16 / rgba_u16) for
/// one legacy bit-packed format. `$kind` is `byte` or `nibble`; each channel
/// is `(right_shift, native_mask, expand_fn)`.
macro_rules! avx512_lowbit_format {
  (@load byte, $src:expr, $x:expr) => { load_byte_px32($src.as_ptr().add($x)) };
  (@load nibble, $src:expr, $x:expr) => { load_nibble_px32($src.as_ptr().add($x / 2)) };
  (@srcmin byte, $w:expr) => { $w };
  (@srcmin nibble, $w:expr) => { $w.div_ceil(2) };
  (@tail byte, $src:expr, $x:expr) => { &$src[$x..] };
  (@tail nibble, $src:expr, $x:expr) => { &$src[$x / 2..] };
  (
    kind: $kind:tt,
    rgb: $to_rgb:ident, rgba: $to_rgba:ident,
    rgb_u16: $to_rgb_u16:ident, rgba_u16: $to_rgba_u16:ident,
    s_rgb: $s_rgb:path, s_rgba: $s_rgba:path,
    s_rgb_u16: $s_rgb_u16:path, s_rgba_u16: $s_rgba_u16:path,
    r: ($rsh:literal, $rmask:expr, $rexp:ident),
    g: ($gsh:literal, $gmask:expr, $gexp:ident),
    b: ($bsh:literal, $bmask:expr, $bexp:ident),
  ) => {
    /// AVX-512 (F+BW): packed legacy RGB/BGR → `R, G, B` bytes (32 px/iter).
    ///
    /// # Safety
    ///
    /// AVX-512BW available; `src` and `rgb_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn $to_rgb(src: &[u8], rgb_out: &mut [u8], width: usize) {
      debug_assert!(src.len() >= avx512_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
      unsafe {
        let rmask = _mm512_set1_epi16($rmask);
        let gmask = _mm512_set1_epi16($gmask);
        let bmask = _mm512_set1_epi16($bmask);
        let zero128 = _mm_setzero_si128();
        let mut x = 0usize;
        while x + 32 <= width {
          let px = avx512_lowbit_format!(@load $kind, src, x);
          let r = _mm512_and_si512(_mm512_srli_epi16(px, $rsh), rmask);
          let g = _mm512_and_si512(_mm512_srli_epi16(px, $gsh), gmask);
          let b = _mm512_and_si512(_mm512_srli_epi16(px, $bsh), bmask);
          write_rgb_32_from_u16lanes(
            $rexp(r),
            $gexp(g),
            $bexp(b),
            zero128,
            rgb_out.as_mut_ptr().add(x * 3),
          );
          x += 32;
        }
        if x < width {
          $s_rgb(avx512_lowbit_format!(@tail $kind, src, x), &mut rgb_out[x * 3..], width - x);
        }
      }
    }

    /// AVX-512 (F+BW): packed legacy RGB/BGR → `R, G, B, A` bytes (α = `0xFF`).
    ///
    /// # Safety
    ///
    /// AVX-512BW available; `src` and `rgba_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn $to_rgba(src: &[u8], rgba_out: &mut [u8], width: usize) {
      debug_assert!(src.len() >= avx512_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
      unsafe {
        let rmask = _mm512_set1_epi16($rmask);
        let gmask = _mm512_set1_epi16($gmask);
        let bmask = _mm512_set1_epi16($bmask);
        let zero128 = _mm_setzero_si128();
        let alpha_u8 = _mm_set1_epi8(-1i8);
        let mut x = 0usize;
        while x + 32 <= width {
          let px = avx512_lowbit_format!(@load $kind, src, x);
          let r = _mm512_and_si512(_mm512_srli_epi16(px, $rsh), rmask);
          let g = _mm512_and_si512(_mm512_srli_epi16(px, $gsh), gmask);
          let b = _mm512_and_si512(_mm512_srli_epi16(px, $bsh), bmask);
          write_rgba_32_from_u16lanes(
            $rexp(r),
            $gexp(g),
            $bexp(b),
            alpha_u8,
            zero128,
            rgba_out.as_mut_ptr().add(x * 4),
          );
          x += 32;
        }
        if x < width {
          $s_rgba(avx512_lowbit_format!(@tail $kind, src, x), &mut rgba_out[x * 4..], width - x);
        }
      }
    }

    /// AVX-512 (F+BW): packed legacy RGB/BGR → native `R, G, B` u16 (32 px/iter).
    ///
    /// # Safety
    ///
    /// AVX-512BW available; `src` and `rgb_u16_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn $to_rgb_u16(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
      debug_assert!(src.len() >= avx512_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
      unsafe {
        let rmask = _mm512_set1_epi16($rmask);
        let gmask = _mm512_set1_epi16($gmask);
        let bmask = _mm512_set1_epi16($bmask);
        let mut x = 0usize;
        while x + 32 <= width {
          let px = avx512_lowbit_format!(@load $kind, src, x);
          let r = _mm512_and_si512(_mm512_srli_epi16(px, $rsh), rmask);
          let g = _mm512_and_si512(_mm512_srli_epi16(px, $gsh), gmask);
          let b = _mm512_and_si512(_mm512_srli_epi16(px, $bsh), bmask);
          write_rgb_u16_32_quarters(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
          x += 32;
        }
        if x < width {
          $s_rgb_u16(
            avx512_lowbit_format!(@tail $kind, src, x),
            &mut rgb_u16_out[x * 3..],
            width - x,
          );
        }
      }
    }

    /// AVX-512 (F+BW): packed legacy RGB/BGR → native `R, G, B, A` u16
    /// (α = `0xFFFF`, 32 px/iter).
    ///
    /// # Safety
    ///
    /// AVX-512BW available; `src` and `rgba_u16_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(crate) unsafe fn $to_rgba_u16(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
      debug_assert!(src.len() >= avx512_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
      unsafe {
        let rmask = _mm512_set1_epi16($rmask);
        let gmask = _mm512_set1_epi16($gmask);
        let bmask = _mm512_set1_epi16($bmask);
        let alpha = _mm_set1_epi16(-1i16);
        let mut x = 0usize;
        while x + 32 <= width {
          let px = avx512_lowbit_format!(@load $kind, src, x);
          let r = _mm512_and_si512(_mm512_srli_epi16(px, $rsh), rmask);
          let g = _mm512_and_si512(_mm512_srli_epi16(px, $gsh), gmask);
          let b = _mm512_and_si512(_mm512_srli_epi16(px, $bsh), bmask);
          write_rgba_u16_32_quarters(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
          x += 32;
        }
        if x < width {
          $s_rgba_u16(
            avx512_lowbit_format!(@tail $kind, src, x),
            &mut rgba_u16_out[x * 4..],
            width - x,
          );
        }
      }
    }
  };
}

avx512_lowbit_format! {
  kind: byte,
  rgb: rgb8_to_rgb_row, rgba: rgb8_to_rgba_row,
  rgb_u16: rgb8_to_rgb_u16_row, rgba_u16: rgb8_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::rgb8_to_rgb_row,
  s_rgba: scalar::legacy_rgb::rgb8_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::rgb8_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::rgb8_to_rgba_u16_row,
  r: (5, 0x07, expand3),
  g: (2, 0x07, expand3),
  b: (0, 0x03, expand2),
}

avx512_lowbit_format! {
  kind: byte,
  rgb: bgr8_to_rgb_row, rgba: bgr8_to_rgba_row,
  rgb_u16: bgr8_to_rgb_u16_row, rgba_u16: bgr8_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::bgr8_to_rgb_row,
  s_rgba: scalar::legacy_rgb::bgr8_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::bgr8_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::bgr8_to_rgba_u16_row,
  r: (0, 0x07, expand3),
  g: (3, 0x07, expand3),
  b: (6, 0x03, expand2),
}

avx512_lowbit_format! {
  kind: byte,
  rgb: rgb4_byte_to_rgb_row, rgba: rgb4_byte_to_rgba_row,
  rgb_u16: rgb4_byte_to_rgb_u16_row, rgba_u16: rgb4_byte_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::rgb4_byte_to_rgb_row,
  s_rgba: scalar::legacy_rgb::rgb4_byte_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::rgb4_byte_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::rgb4_byte_to_rgba_u16_row,
  r: (3, 0x01, expand1),
  g: (1, 0x03, expand2),
  b: (0, 0x01, expand1),
}

avx512_lowbit_format! {
  kind: byte,
  rgb: bgr4_byte_to_rgb_row, rgba: bgr4_byte_to_rgba_row,
  rgb_u16: bgr4_byte_to_rgb_u16_row, rgba_u16: bgr4_byte_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::bgr4_byte_to_rgb_row,
  s_rgba: scalar::legacy_rgb::bgr4_byte_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::bgr4_byte_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::bgr4_byte_to_rgba_u16_row,
  r: (0, 0x01, expand1),
  g: (1, 0x03, expand2),
  b: (3, 0x01, expand1),
}

avx512_lowbit_format! {
  kind: nibble,
  rgb: rgb4_to_rgb_row, rgba: rgb4_to_rgba_row,
  rgb_u16: rgb4_to_rgb_u16_row, rgba_u16: rgb4_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::rgb4_to_rgb_row,
  s_rgba: scalar::legacy_rgb::rgb4_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::rgb4_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::rgb4_to_rgba_u16_row,
  r: (3, 0x01, expand1),
  g: (1, 0x03, expand2),
  b: (0, 0x01, expand1),
}

avx512_lowbit_format! {
  kind: nibble,
  rgb: bgr4_to_rgb_row, rgba: bgr4_to_rgba_row,
  rgb_u16: bgr4_to_rgb_u16_row, rgba_u16: bgr4_to_rgba_u16_row,
  s_rgb: scalar::legacy_rgb::bgr4_to_rgb_row,
  s_rgba: scalar::legacy_rgb::bgr4_to_rgba_row,
  s_rgb_u16: scalar::legacy_rgb::bgr4_to_rgb_u16_row,
  s_rgba_u16: scalar::legacy_rgb::bgr4_to_rgba_u16_row,
  r: (0, 0x01, expand1),
  g: (1, 0x03, expand2),
  b: (3, 0x01, expand1),
}
