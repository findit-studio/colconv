//! AVX2 kernels for legacy 16-bit packed-RGB source formats (Tier 7).
//!
//! Six source formats × 4 output variants = 24 kernels. Each format word is a
//! little-endian `u16` at 16 pixels per iteration (`_mm256_loadu_si256` = 16 × u16).
//!
//! # Bit extraction
//!
//! - **RGB565**: `_mm256_srli_epi16(px, 11)` + `& 0x1F` → R5;
//!   `_mm256_srli_epi16(px, 5)` + `& 0x3F` → G6; `px & 0x1F` → B5.
//! - **BGR565**: same shifts, but R↔B swapped in extraction (R5 at bits [4:0],
//!   B5 at bits [15:11]).
//! - **RGB555**: `_mm256_srli_epi16(px, 10)` + `& 0x1F` → R5;
//!   `_mm256_srli_epi16(px, 5)` + `& 0x1F` → G5; `px & 0x1F` → B5.
//! - **BGR555**: same as RGB555 with R↔B swapped.
//! - **RGB444**: `_mm256_srli_epi16(px, 8)` + `& 0x0F` → R4;
//!   `_mm256_srli_epi16(px, 4)` + `& 0x0F` → G4; `px & 0x0F` → B4.
//! - **BGR444**: same as RGB444 with R↔B swapped.
//!
//! # Channel expansion
//!
//! | Bits | AVX2 (shift + OR)                                                             |
//! |------|-------------------------------------------------------------------------------|
//! | 5    | `_mm256_or_si256(_mm256_slli_epi16(c, 3), _mm256_srli_epi16(c, 2))` → [0,255] |
//! | 6    | `_mm256_or_si256(_mm256_slli_epi16(c, 2), _mm256_srli_epi16(c, 4))` → [0,255] |
//! | 4    | `_mm256_or_si256(_mm256_slli_epi16(c, 4), c)`                    → [0,255] |
//!
//! # u8 output
//!
//! After expansion each i16 lane holds a value in `[0, 255]`.
//! `_mm256_packus_epi16(expanded, zero256)` packs 16 u16 → 16 u8 but in AVX2
//! cross-lane order; `_mm256_permute4x64_epi64::<0xD8>` fixes lane order. The
//! resulting low 128 bits hold 16 valid u8 pixels and are fed to `write_rgb_16` /
//! `write_rgba_16` (16-pixel helpers writing exactly 48 / 64 bytes).
//!
//! # u16 output
//!
//! Skip `_mm256_packus_epi16`; process as two 8-pixel 128-bit halves:
//! low via `_mm256_castsi256_si128`, high via `_mm256_extracti128_si256::<1>`.
//! Feed each half to `write_rgb_u16_8` / `write_rgba_u16_8` (8-pixel helpers).
//!
//! # Scalar tail
//!
//! When `width % 16 ≠ 0` the remainder is handled by `scalar::legacy_rgb`.

use super::*;

// Internal helpers.
/// Expand a vector of 5-bit values in [0, 31] to 8-bit: `(c << 3) | (c >> 2)`.
#[inline(always)]
unsafe fn expand5(c: __m256i) -> __m256i {
  unsafe { _mm256_or_si256(_mm256_slli_epi16(c, 3), _mm256_srli_epi16(c, 2)) }
}

/// Expand a vector of 6-bit values in [0, 63] to 8-bit: `(c << 2) | (c >> 4)`.
#[inline(always)]
unsafe fn expand6(c: __m256i) -> __m256i {
  unsafe { _mm256_or_si256(_mm256_slli_epi16(c, 2), _mm256_srli_epi16(c, 4)) }
}

/// Expand a vector of 4-bit values in [0, 15] to 8-bit: `(c << 4) | c`.
#[inline(always)]
unsafe fn expand4(c: __m256i) -> __m256i {
  unsafe { _mm256_or_si256(_mm256_slli_epi16(c, 4), c) }
}

/// Pack 16 u16 lanes to 16 u8 with correct lane order.
/// `_mm256_packus_epi16(v, zero)` produces cross-lane interleaved output;
/// `_mm256_permute4x64_epi64::<0xD8>` restores natural order so the low
/// 128-bit half contains the 16 valid u8 pixels.
#[inline(always)]
unsafe fn pack_u8(v: __m256i, zero256: __m256i) -> __m256i {
  unsafe { _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(v, zero256)) }
}

// RGB565 — R5 G6 B5, bits [15:11]=R, [10:5]=G, [4:0]=B.
/// AVX2 RGB565 → packed `R, G, B` bytes (16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask_r5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm256_set1_epi16(0x3F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r5 = _mm256_and_si256(_mm256_srli_epi16(px, 11), mask_r5);
      let g6 = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask_g6);
      let b5 = _mm256_and_si256(px, mask_r5);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand5(r5), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand6(g6), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand5(b5), zero256));
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 RGB565 → packed `R, G, B, A` bytes (α = `0xFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask_r5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm256_set1_epi16(0x3F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r5 = _mm256_and_si256(_mm256_srli_epi16(px, 11), mask_r5);
      let g6 = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask_g6);
      let b5 = _mm256_and_si256(px, mask_r5);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand5(r5), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand6(g6), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand5(b5), zero256));
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 RGB565 → packed `R, G, B` **u16** (native bit-width, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask_r5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm256_set1_epi16(0x3F_u16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r = _mm256_and_si256(_mm256_srli_epi16(px, 11), mask_r5);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask_g6);
      let b = _mm256_and_si256(px, mask_r5);
      // Two 8-pixel halves via 128-bit SSE helpers.
      let r_lo = _mm256_castsi256_si128(r);
      let g_lo = _mm256_castsi256_si128(g);
      let b_lo = _mm256_castsi256_si128(b);
      write_rgb_u16_8(r_lo, g_lo, b_lo, rgb_u16_out.as_mut_ptr().add(x * 3));
      let r_hi = _mm256_extracti128_si256::<1>(r);
      let g_hi = _mm256_extracti128_si256::<1>(g);
      let b_hi = _mm256_extracti128_si256::<1>(b);
      write_rgb_u16_8(r_hi, g_hi, b_hi, rgb_u16_out.as_mut_ptr().add((x + 8) * 3));
      x += 16;
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

/// AVX2 RGB565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask_r5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm256_set1_epi16(0x3F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r = _mm256_and_si256(_mm256_srli_epi16(px, 11), mask_r5);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask_g6);
      let b = _mm256_and_si256(px, mask_r5);
      let r_lo = _mm256_castsi256_si128(r);
      let g_lo = _mm256_castsi256_si128(g);
      let b_lo = _mm256_castsi256_si128(b);
      write_rgba_u16_8(
        r_lo,
        g_lo,
        b_lo,
        alpha,
        rgba_u16_out.as_mut_ptr().add(x * 4),
      );
      let r_hi = _mm256_extracti128_si256::<1>(r);
      let g_hi = _mm256_extracti128_si256::<1>(g);
      let b_hi = _mm256_extracti128_si256::<1>(b);
      write_rgba_u16_8(
        r_hi,
        g_hi,
        b_hi,
        alpha,
        rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
      );
      x += 16;
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
/// AVX2 BGR565 → packed `R, G, B` bytes (output R-first, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask_r5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm256_set1_epi16(0x3F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      // BGR565: B at [15:11], G at [10:5], R at [4:0]
      let b5 = _mm256_and_si256(_mm256_srli_epi16(px, 11), mask_r5);
      let g6 = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask_g6);
      let r5 = _mm256_and_si256(px, mask_r5);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand5(r5), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand6(g6), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand5(b5), zero256));
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 BGR565 → packed `R, G, B, A` bytes (α = `0xFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask_r5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm256_set1_epi16(0x3F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let b5 = _mm256_and_si256(_mm256_srli_epi16(px, 11), mask_r5);
      let g6 = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask_g6);
      let r5 = _mm256_and_si256(px, mask_r5);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand5(r5), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand6(g6), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand5(b5), zero256));
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 BGR565 → packed `R, G, B` **u16** (native bit-width, output R-first, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask_r5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm256_set1_epi16(0x3F_u16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      // BGR565: B at [15:11], G at [10:5], R at [4:0]. Output order: R, G, B.
      let b = _mm256_and_si256(_mm256_srli_epi16(px, 11), mask_r5);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask_g6);
      let r = _mm256_and_si256(px, mask_r5);
      write_rgb_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        rgb_u16_out.as_mut_ptr().add(x * 3),
      );
      write_rgb_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        rgb_u16_out.as_mut_ptr().add((x + 8) * 3),
      );
      x += 16;
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

/// AVX2 BGR565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask_r5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm256_set1_epi16(0x3F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let b = _mm256_and_si256(_mm256_srli_epi16(px, 11), mask_r5);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask_g6);
      let r = _mm256_and_si256(px, mask_r5);
      write_rgba_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add(x * 4),
      );
      write_rgba_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
      );
      x += 16;
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
/// AVX2 RGB555 → packed `R, G, B` bytes (16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r5 = _mm256_and_si256(_mm256_srli_epi16(px, 10), mask5);
      let g5 = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask5);
      let b5 = _mm256_and_si256(px, mask5);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand5(r5), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand5(g5), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand5(b5), zero256));
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 RGB555 → packed `R, G, B, A` bytes (α = `0xFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r5 = _mm256_and_si256(_mm256_srli_epi16(px, 10), mask5);
      let g5 = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask5);
      let b5 = _mm256_and_si256(px, mask5);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand5(r5), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand5(g5), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand5(b5), zero256));
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 RGB555 → packed `R, G, B` **u16** (native bit-width, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r = _mm256_and_si256(_mm256_srli_epi16(px, 10), mask5);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask5);
      let b = _mm256_and_si256(px, mask5);
      write_rgb_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        rgb_u16_out.as_mut_ptr().add(x * 3),
      );
      write_rgb_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        rgb_u16_out.as_mut_ptr().add((x + 8) * 3),
      );
      x += 16;
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

/// AVX2 RGB555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r = _mm256_and_si256(_mm256_srli_epi16(px, 10), mask5);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask5);
      let b = _mm256_and_si256(px, mask5);
      write_rgba_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add(x * 4),
      );
      write_rgba_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
      );
      x += 16;
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
/// AVX2 BGR555 → packed `R, G, B` bytes (output R-first, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      // BGR555: B at [14:10], G at [9:5], R at [4:0]
      let b5 = _mm256_and_si256(_mm256_srli_epi16(px, 10), mask5);
      let g5 = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask5);
      let r5 = _mm256_and_si256(px, mask5);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand5(r5), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand5(g5), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand5(b5), zero256));
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 BGR555 → packed `R, G, B, A` bytes (α = `0xFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let b5 = _mm256_and_si256(_mm256_srli_epi16(px, 10), mask5);
      let g5 = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask5);
      let r5 = _mm256_and_si256(px, mask5);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand5(r5), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand5(g5), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand5(b5), zero256));
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 BGR555 → packed `R, G, B` **u16** (native bit-width, output R-first, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      // BGR555: B at [14:10], G at [9:5], R at [4:0]. Output order: R, G, B.
      let b = _mm256_and_si256(_mm256_srli_epi16(px, 10), mask5);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask5);
      let r = _mm256_and_si256(px, mask5);
      write_rgb_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        rgb_u16_out.as_mut_ptr().add(x * 3),
      );
      write_rgb_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        rgb_u16_out.as_mut_ptr().add((x + 8) * 3),
      );
      x += 16;
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

/// AVX2 BGR555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = _mm256_set1_epi16(0x1F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let b = _mm256_and_si256(_mm256_srli_epi16(px, 10), mask5);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 5), mask5);
      let r = _mm256_and_si256(px, mask5);
      write_rgba_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add(x * 4),
      );
      write_rgba_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
      );
      x += 16;
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
/// AVX2 RGB444 → packed `R, G, B` bytes (16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = _mm256_set1_epi16(0x0F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r4 = _mm256_and_si256(_mm256_srli_epi16(px, 8), mask4);
      let g4 = _mm256_and_si256(_mm256_srli_epi16(px, 4), mask4);
      let b4 = _mm256_and_si256(px, mask4);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand4(r4), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand4(g4), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand4(b4), zero256));
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 RGB444 → packed `R, G, B, A` bytes (α = `0xFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = _mm256_set1_epi16(0x0F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r4 = _mm256_and_si256(_mm256_srli_epi16(px, 8), mask4);
      let g4 = _mm256_and_si256(_mm256_srli_epi16(px, 4), mask4);
      let b4 = _mm256_and_si256(px, mask4);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand4(r4), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand4(g4), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand4(b4), zero256));
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 RGB444 → packed `R, G, B` **u16** (native 4-bit width, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = _mm256_set1_epi16(0x0F_u16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r = _mm256_and_si256(_mm256_srli_epi16(px, 8), mask4);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 4), mask4);
      let b = _mm256_and_si256(px, mask4);
      write_rgb_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        rgb_u16_out.as_mut_ptr().add(x * 3),
      );
      write_rgb_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        rgb_u16_out.as_mut_ptr().add((x + 8) * 3),
      );
      x += 16;
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

/// AVX2 RGB444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgb444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = _mm256_set1_epi16(0x0F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let r = _mm256_and_si256(_mm256_srli_epi16(px, 8), mask4);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 4), mask4);
      let b = _mm256_and_si256(px, mask4);
      write_rgba_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add(x * 4),
      );
      write_rgba_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
      );
      x += 16;
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
/// AVX2 BGR444 → packed `R, G, B` bytes (output R-first, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = _mm256_set1_epi16(0x0F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      // BGR444: B at [11:8], G at [7:4], R at [3:0]
      let b4 = _mm256_and_si256(_mm256_srli_epi16(px, 8), mask4);
      let g4 = _mm256_and_si256(_mm256_srli_epi16(px, 4), mask4);
      let r4 = _mm256_and_si256(px, mask4);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand4(r4), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand4(g4), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand4(b4), zero256));
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 BGR444 → packed `R, G, B, A` bytes (α = `0xFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = _mm256_set1_epi16(0x0F_u16 as i16);
    let zero256 = _mm256_setzero_si256();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let b4 = _mm256_and_si256(_mm256_srli_epi16(px, 8), mask4);
      let g4 = _mm256_and_si256(_mm256_srli_epi16(px, 4), mask4);
      let r4 = _mm256_and_si256(px, mask4);
      let r_u8 = _mm256_castsi256_si128(pack_u8(expand4(r4), zero256));
      let g_u8 = _mm256_castsi256_si128(pack_u8(expand4(g4), zero256));
      let b_u8 = _mm256_castsi256_si128(pack_u8(expand4(b4), zero256));
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 BGR444 → packed `R, G, B` **u16** (native 4-bit width, output R-first, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = _mm256_set1_epi16(0x0F_u16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      // BGR444: B at [11:8], G at [7:4], R at [3:0]. Output order: R, G, B.
      let b = _mm256_and_si256(_mm256_srli_epi16(px, 8), mask4);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 4), mask4);
      let r = _mm256_and_si256(px, mask4);
      write_rgb_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        rgb_u16_out.as_mut_ptr().add(x * 3),
      );
      write_rgb_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        rgb_u16_out.as_mut_ptr().add((x + 8) * 3),
      );
      x += 16;
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

/// AVX2 BGR444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 16 px/iter).
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bgr444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = _mm256_set1_epi16(0x0F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let px = _mm256_loadu_si256(src.as_ptr().add(x * 2).cast());
      let b = _mm256_and_si256(_mm256_srli_epi16(px, 8), mask4);
      let g = _mm256_and_si256(_mm256_srli_epi16(px, 4), mask4);
      let r = _mm256_and_si256(px, mask4);
      write_rgba_u16_8(
        _mm256_castsi256_si128(r),
        _mm256_castsi256_si128(g),
        _mm256_castsi256_si128(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add(x * 4),
      );
      write_rgba_u16_8(
        _mm256_extracti128_si256::<1>(r),
        _mm256_extracti128_si256::<1>(g),
        _mm256_extracti128_si256::<1>(b),
        alpha,
        rgba_u16_out.as_mut_ptr().add((x + 8) * 4),
      );
      x += 16;
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
