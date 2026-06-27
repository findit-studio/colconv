//! SSE4.1 kernels for legacy 16-bit packed-RGB source formats (Tier 7).
//!
//! Six source formats × 4 output variants = 24 kernels. Each format word is a
//! little-endian `u16` at 8 pixels per iteration (`_mm_loadu_si128` = 8 × u16).
//!
//! # Bit extraction
//!
//! - **RGB565**: `_mm_srli_epi16(px, 11)` + `& 0x1F` → R5;
//!   `_mm_srli_epi16(px, 5)` + `& 0x3F` → G6; `px & 0x1F` → B5.
//! - **BGR565**: same shifts, but R↔B swapped in extraction (R5 at bits [4:0],
//!   B5 at bits [15:11]).
//! - **RGB555**: `_mm_srli_epi16(px, 10)` + `& 0x1F` → R5;
//!   `_mm_srli_epi16(px, 5)` + `& 0x1F` → G5; `px & 0x1F` → B5.
//! - **BGR555**: same as RGB555 with R↔B swapped.
//! - **RGB444**: `_mm_srli_epi16(px, 8)` + `& 0x0F` → R4;
//!   `_mm_srli_epi16(px, 4)` + `& 0x0F` → G4; `px & 0x0F` → B4.
//! - **BGR444**: same as RGB444 with R↔B swapped.
//!
//! # Channel expansion
//!
//! | Bits | SSE4.1 (shift + OR)                                                   |
//! |------|-----------------------------------------------------------------------|
//! | 5    | `_mm_or_si128(_mm_slli_epi16(c, 3), _mm_srli_epi16(c, 2))` → [0,255] |
//! | 6    | `_mm_or_si128(_mm_slli_epi16(c, 2), _mm_srli_epi16(c, 4))` → [0,255] |
//! | 4    | `_mm_or_si128(_mm_slli_epi16(c, 4), c)`                    → [0,255] |
//!
//! # u8 output
//!
//! After expansion each i16 lane holds a value in `[0, 255]`. `_mm_packus_epi16`
//! narrows to u8 (8 valid bytes in the low half). The 48-byte `write_rgb_16` /
//! 64-byte `write_rgba_16` helpers write 16 pixels; we use a local temp buffer
//! and `core::ptr::copy_nonoverlapping` to emit only 24 / 32 bytes for 8 pixels.
//!
//! # u16 output
//!
//! Skip `_mm_packus_epi16`; feed the raw extracted (or expanded) u16 lanes
//! directly into `write_rgb_u16_8` / `write_rgba_u16_8` which write exactly
//! 8 pixels (24 / 32 u16 elements).
//!
//! # Scalar tail
//!
//! When `width % 8 ≠ 0` the remainder is handled by `scalar::legacy_rgb`.

use super::*;

// Internal helpers.
/// Expand a vector of 5-bit values in [0, 31] to 8-bit: `(c << 3) | (c >> 2)`.
#[inline(always)]
unsafe fn expand5(c: __m128i) -> __m128i {
  unsafe { _mm_or_si128(_mm_slli_epi16(c, 3), _mm_srli_epi16(c, 2)) }
}

/// Expand a vector of 6-bit values in [0, 63] to 8-bit: `(c << 2) | (c >> 4)`.
#[inline(always)]
unsafe fn expand6(c: __m128i) -> __m128i {
  unsafe { _mm_or_si128(_mm_slli_epi16(c, 2), _mm_srli_epi16(c, 4)) }
}

/// Expand a vector of 4-bit values in [0, 15] to 8-bit: `(c << 4) | c`.
#[inline(always)]
unsafe fn expand4(c: __m128i) -> __m128i {
  unsafe { _mm_or_si128(_mm_slli_epi16(c, 4), c) }
}

// RGB565 — R5 G6 B5, bits [15:11]=R, [10:5]=G, [4:0]=B.
/// SSE4.1 RGB565 → packed `R, G, B` bytes (8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask_r5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm_set1_epi16(0x3F_u16 as i16);
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r5 = _mm_and_si128(_mm_srli_epi16(px, 11), mask_r5);
      let g6 = _mm_and_si128(_mm_srli_epi16(px, 5), mask_g6);
      let b5 = _mm_and_si128(px, mask_r5);
      let r_exp = expand5(r5);
      let g_exp = expand6(g6);
      let b_exp = expand5(b5);
      let r_u8 = _mm_packus_epi16(r_exp, zero);
      let g_u8 = _mm_packus_epi16(g_exp, zero);
      let b_u8 = _mm_packus_epi16(b_exp, zero);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 RGB565 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask_r5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm_set1_epi16(0x3F_u16 as i16);
    let zero = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r5 = _mm_and_si128(_mm_srli_epi16(px, 11), mask_r5);
      let g6 = _mm_and_si128(_mm_srli_epi16(px, 5), mask_g6);
      let b5 = _mm_and_si128(px, mask_r5);
      let r_u8 = _mm_packus_epi16(expand5(r5), zero);
      let g_u8 = _mm_packus_epi16(expand6(g6), zero);
      let b_u8 = _mm_packus_epi16(expand5(b5), zero);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb565_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// SSE4.1 RGB565 → packed `R, G, B` **u16** (native bit-width, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask_r5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm_set1_epi16(0x3F_u16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r = _mm_and_si128(_mm_srli_epi16(px, 11), mask_r5);
      let g = _mm_and_si128(_mm_srli_epi16(px, 5), mask_g6);
      let b = _mm_and_si128(px, mask_r5);
      write_rgb_u16_8(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 8;
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

/// SSE4.1 RGB565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask_r5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm_set1_epi16(0x3F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r = _mm_and_si128(_mm_srli_epi16(px, 11), mask_r5);
      let g = _mm_and_si128(_mm_srli_epi16(px, 5), mask_g6);
      let b = _mm_and_si128(px, mask_r5);
      write_rgba_u16_8(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 8;
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
/// SSE4.1 BGR565 → packed `R, G, B` bytes (output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask_r5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm_set1_epi16(0x3F_u16 as i16);
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      // BGR565: B at [15:11], G at [10:5], R at [4:0]
      let b5 = _mm_and_si128(_mm_srli_epi16(px, 11), mask_r5);
      let g6 = _mm_and_si128(_mm_srli_epi16(px, 5), mask_g6);
      let r5 = _mm_and_si128(px, mask_r5);
      let r_u8 = _mm_packus_epi16(expand5(r5), zero);
      let g_u8 = _mm_packus_epi16(expand6(g6), zero);
      let b_u8 = _mm_packus_epi16(expand5(b5), zero);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 BGR565 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask_r5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm_set1_epi16(0x3F_u16 as i16);
    let zero = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let b5 = _mm_and_si128(_mm_srli_epi16(px, 11), mask_r5);
      let g6 = _mm_and_si128(_mm_srli_epi16(px, 5), mask_g6);
      let r5 = _mm_and_si128(px, mask_r5);
      let r_u8 = _mm_packus_epi16(expand5(r5), zero);
      let g_u8 = _mm_packus_epi16(expand6(g6), zero);
      let b_u8 = _mm_packus_epi16(expand5(b5), zero);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr565_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// SSE4.1 BGR565 → packed `R, G, B` **u16** (native bit-width, output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask_r5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm_set1_epi16(0x3F_u16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      // BGR565: B at [15:11], G at [10:5], R at [4:0]. Output order: R, G, B.
      let b = _mm_and_si128(_mm_srli_epi16(px, 11), mask_r5);
      let g = _mm_and_si128(_mm_srli_epi16(px, 5), mask_g6);
      let r = _mm_and_si128(px, mask_r5);
      write_rgb_u16_8(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 8;
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

/// SSE4.1 BGR565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask_r5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mask_g6 = _mm_set1_epi16(0x3F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let b = _mm_and_si128(_mm_srli_epi16(px, 11), mask_r5);
      let g = _mm_and_si128(_mm_srli_epi16(px, 5), mask_g6);
      let r = _mm_and_si128(px, mask_r5);
      write_rgba_u16_8(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 8;
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
/// SSE4.1 RGB555 → packed `R, G, B` bytes (8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = _mm_set1_epi16(0x1F_u16 as i16);
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r5 = _mm_and_si128(_mm_srli_epi16(px, 10), mask5);
      let g5 = _mm_and_si128(_mm_srli_epi16(px, 5), mask5);
      let b5 = _mm_and_si128(px, mask5);
      let r_u8 = _mm_packus_epi16(expand5(r5), zero);
      let g_u8 = _mm_packus_epi16(expand5(g5), zero);
      let b_u8 = _mm_packus_epi16(expand5(b5), zero);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 RGB555 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = _mm_set1_epi16(0x1F_u16 as i16);
    let zero = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r5 = _mm_and_si128(_mm_srli_epi16(px, 10), mask5);
      let g5 = _mm_and_si128(_mm_srli_epi16(px, 5), mask5);
      let b5 = _mm_and_si128(px, mask5);
      let r_u8 = _mm_packus_epi16(expand5(r5), zero);
      let g_u8 = _mm_packus_epi16(expand5(g5), zero);
      let b_u8 = _mm_packus_epi16(expand5(b5), zero);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb555_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// SSE4.1 RGB555 → packed `R, G, B` **u16** (native bit-width, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r = _mm_and_si128(_mm_srli_epi16(px, 10), mask5);
      let g = _mm_and_si128(_mm_srli_epi16(px, 5), mask5);
      let b = _mm_and_si128(px, mask5);
      write_rgb_u16_8(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 8;
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

/// SSE4.1 RGB555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = _mm_set1_epi16(0x1F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r = _mm_and_si128(_mm_srli_epi16(px, 10), mask5);
      let g = _mm_and_si128(_mm_srli_epi16(px, 5), mask5);
      let b = _mm_and_si128(px, mask5);
      write_rgba_u16_8(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 8;
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
/// SSE4.1 BGR555 → packed `R, G, B` bytes (output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = _mm_set1_epi16(0x1F_u16 as i16);
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      // BGR555: B at [14:10], G at [9:5], R at [4:0]
      let b5 = _mm_and_si128(_mm_srli_epi16(px, 10), mask5);
      let g5 = _mm_and_si128(_mm_srli_epi16(px, 5), mask5);
      let r5 = _mm_and_si128(px, mask5);
      let r_u8 = _mm_packus_epi16(expand5(r5), zero);
      let g_u8 = _mm_packus_epi16(expand5(g5), zero);
      let b_u8 = _mm_packus_epi16(expand5(b5), zero);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 BGR555 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = _mm_set1_epi16(0x1F_u16 as i16);
    let zero = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let b5 = _mm_and_si128(_mm_srli_epi16(px, 10), mask5);
      let g5 = _mm_and_si128(_mm_srli_epi16(px, 5), mask5);
      let r5 = _mm_and_si128(px, mask5);
      let r_u8 = _mm_packus_epi16(expand5(r5), zero);
      let g_u8 = _mm_packus_epi16(expand5(g5), zero);
      let b_u8 = _mm_packus_epi16(expand5(b5), zero);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr555_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// SSE4.1 BGR555 → packed `R, G, B` **u16** (native bit-width, output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = _mm_set1_epi16(0x1F_u16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      // BGR555: B at [14:10], G at [9:5], R at [4:0]. Output order: R, G, B.
      let b = _mm_and_si128(_mm_srli_epi16(px, 10), mask5);
      let g = _mm_and_si128(_mm_srli_epi16(px, 5), mask5);
      let r = _mm_and_si128(px, mask5);
      write_rgb_u16_8(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 8;
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

/// SSE4.1 BGR555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = _mm_set1_epi16(0x1F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let b = _mm_and_si128(_mm_srli_epi16(px, 10), mask5);
      let g = _mm_and_si128(_mm_srli_epi16(px, 5), mask5);
      let r = _mm_and_si128(px, mask5);
      write_rgba_u16_8(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 8;
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
/// SSE4.1 RGB444 → packed `R, G, B` bytes (8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = _mm_set1_epi16(0x0F_u16 as i16);
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r4 = _mm_and_si128(_mm_srli_epi16(px, 8), mask4);
      let g4 = _mm_and_si128(_mm_srli_epi16(px, 4), mask4);
      let b4 = _mm_and_si128(px, mask4);
      let r_u8 = _mm_packus_epi16(expand4(r4), zero);
      let g_u8 = _mm_packus_epi16(expand4(g4), zero);
      let b_u8 = _mm_packus_epi16(expand4(b4), zero);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 RGB444 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = _mm_set1_epi16(0x0F_u16 as i16);
    let zero = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r4 = _mm_and_si128(_mm_srli_epi16(px, 8), mask4);
      let g4 = _mm_and_si128(_mm_srli_epi16(px, 4), mask4);
      let b4 = _mm_and_si128(px, mask4);
      let r_u8 = _mm_packus_epi16(expand4(r4), zero);
      let g_u8 = _mm_packus_epi16(expand4(g4), zero);
      let b_u8 = _mm_packus_epi16(expand4(b4), zero);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::rgb444_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// SSE4.1 RGB444 → packed `R, G, B` **u16** (native 4-bit width, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = _mm_set1_epi16(0x0F_u16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r = _mm_and_si128(_mm_srli_epi16(px, 8), mask4);
      let g = _mm_and_si128(_mm_srli_epi16(px, 4), mask4);
      let b = _mm_and_si128(px, mask4);
      write_rgb_u16_8(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 8;
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

/// SSE4.1 RGB444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgb444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = _mm_set1_epi16(0x0F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let r = _mm_and_si128(_mm_srli_epi16(px, 8), mask4);
      let g = _mm_and_si128(_mm_srli_epi16(px, 4), mask4);
      let b = _mm_and_si128(px, mask4);
      write_rgba_u16_8(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 8;
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
/// SSE4.1 BGR444 → packed `R, G, B` bytes (output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = _mm_set1_epi16(0x0F_u16 as i16);
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      // BGR444: B at [11:8], G at [7:4], R at [3:0]
      let b4 = _mm_and_si128(_mm_srli_epi16(px, 8), mask4);
      let g4 = _mm_and_si128(_mm_srli_epi16(px, 4), mask4);
      let r4 = _mm_and_si128(px, mask4);
      let r_u8 = _mm_packus_epi16(expand4(r4), zero);
      let g_u8 = _mm_packus_epi16(expand4(g4), zero);
      let b_u8 = _mm_packus_epi16(expand4(b4), zero);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgb_row(&src[x * 2..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 BGR444 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = _mm_set1_epi16(0x0F_u16 as i16);
    let zero = _mm_setzero_si128();
    let alpha_u8 = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let b4 = _mm_and_si128(_mm_srli_epi16(px, 8), mask4);
      let g4 = _mm_and_si128(_mm_srli_epi16(px, 4), mask4);
      let r4 = _mm_and_si128(px, mask4);
      let r_u8 = _mm_packus_epi16(expand4(r4), zero);
      let g_u8 = _mm_packus_epi16(expand4(g4), zero);
      let b_u8 = _mm_packus_epi16(expand4(b4), zero);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::legacy_rgb::bgr444_to_rgba_row(&src[x * 2..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// SSE4.1 BGR444 → packed `R, G, B` **u16** (native 4-bit width, output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = _mm_set1_epi16(0x0F_u16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      // BGR444: B at [11:8], G at [7:4], R at [3:0]. Output order: R, G, B.
      let b = _mm_and_si128(_mm_srli_epi16(px, 8), mask4);
      let g = _mm_and_si128(_mm_srli_epi16(px, 4), mask4);
      let r = _mm_and_si128(px, mask4);
      write_rgb_u16_8(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
      x += 8;
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

/// SSE4.1 BGR444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn bgr444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = _mm_set1_epi16(0x0F_u16 as i16);
    let alpha = _mm_set1_epi16(-1i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = _mm_loadu_si128(src.as_ptr().add(x * 2).cast());
      let b = _mm_and_si128(_mm_srli_epi16(px, 8), mask4);
      let g = _mm_and_si128(_mm_srli_epi16(px, 4), mask4);
      let r = _mm_and_si128(px, mask4);
      write_rgba_u16_8(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
      x += 8;
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
// Each iteration produces 8 pixels as 8 u16 lanes of native source bytes
// (byte formats: widen 8 source bytes via `_mm_cvtepu8_epi16`; nibble
// formats: de-interleave 4 source bytes into 8 nibble lanes), then reuses the
// same shift+mask extraction, bit-replication expansion, and
// `write_rgb_*` / `write_rgba_*` interleaved stores as the 16-bit formats
// above. The `width % 8` remainder defers to `scalar`.
// =========================================================================

/// Bit-replicate u16 lanes of 1-bit values (`0`/`1`) to 8-bit: `c * 0xFF`.
#[inline(always)]
unsafe fn expand1(c: __m128i) -> __m128i {
  unsafe { _mm_mullo_epi16(c, _mm_set1_epi16(0xFF)) }
}

/// Bit-replicate u16 lanes of 2-bit values (`0..=3`) to 8-bit: `c * 0x55`.
#[inline(always)]
unsafe fn expand2(c: __m128i) -> __m128i {
  unsafe { _mm_mullo_epi16(c, _mm_set1_epi16(0x55)) }
}

/// Bit-replicate u16 lanes of 3-bit values (`0..=7`) to 8-bit:
/// `(c << 5) | (c << 2) | (c >> 1)`.
#[inline(always)]
unsafe fn expand3(c: __m128i) -> __m128i {
  unsafe {
    _mm_or_si128(
      _mm_or_si128(_mm_slli_epi16(c, 5), _mm_slli_epi16(c, 2)),
      _mm_srli_epi16(c, 1),
    )
  }
}

/// Load 8 packed 1-byte-per-pixel source bytes and widen to 8 u16 lanes.
///
/// # Safety
///
/// `ptr` valid for an 8-byte read; SSE4.1 available.
#[inline(always)]
unsafe fn load_byte_px8(ptr: *const u8) -> __m128i {
  unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(ptr.cast())) }
}

/// Load 4 packed 2-pixel-per-byte source bytes and de-interleave the nibbles
/// into 8 u16 lanes (even pixel = high nibble `[7:4]`, odd = low nibble).
///
/// # Safety
///
/// `ptr` valid for a 4-byte read; SSE4.1 available.
#[inline(always)]
unsafe fn load_nibble_px8(ptr: *const u8) -> __m128i {
  unsafe {
    let raw = _mm_cvtsi32_si128(core::ptr::read_unaligned(ptr.cast::<u32>()) as i32);
    // Duplicate each of the 4 low bytes: [b0, b0, b1, b1, b2, b2, b3, b3, …].
    let dup = _mm_unpacklo_epi8(raw, raw);
    let w = _mm_cvtepu8_epi16(dup);
    let hi = _mm_srli_epi16(w, 4);
    let lo = _mm_and_si128(w, _mm_set1_epi16(0x0F));
    // Even lanes take the high nibble (imm 0x55 selects lanes 0,2,4,6 from `hi`).
    _mm_blend_epi16(lo, hi, 0x55)
  }
}

/// Emits the four SSE4.1 output kernels (rgb / rgba / rgb_u16 / rgba_u16) for
/// one legacy bit-packed format. `$kind` is `byte` or `nibble`; each channel
/// is `(right_shift, native_mask, expand_fn)`.
macro_rules! sse_lowbit_format {
  (@load byte, $src:expr, $x:expr) => { load_byte_px8($src.as_ptr().add($x)) };
  (@load nibble, $src:expr, $x:expr) => { load_nibble_px8($src.as_ptr().add($x / 2)) };
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
    /// SSE4.1: packed legacy RGB/BGR → `R, G, B` bytes (8 px/iter).
    ///
    /// # Safety
    ///
    /// SSE4.1 available; `src` and `rgb_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn $to_rgb(src: &[u8], rgb_out: &mut [u8], width: usize) {
      debug_assert!(src.len() >= sse_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
      unsafe {
        let rmask = _mm_set1_epi16($rmask);
        let gmask = _mm_set1_epi16($gmask);
        let bmask = _mm_set1_epi16($bmask);
        let zero = _mm_setzero_si128();
        let mut x = 0usize;
        while x + 8 <= width {
          let px = sse_lowbit_format!(@load $kind, src, x);
          let r = _mm_and_si128(_mm_srli_epi16(px, $rsh), rmask);
          let g = _mm_and_si128(_mm_srli_epi16(px, $gsh), gmask);
          let b = _mm_and_si128(_mm_srli_epi16(px, $bsh), bmask);
          let r_u8 = _mm_packus_epi16($rexp(r), zero);
          let g_u8 = _mm_packus_epi16($gexp(g), zero);
          let b_u8 = _mm_packus_epi16($bexp(b), zero);
          let mut tmp = [0u8; 48];
          write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
          core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
          x += 8;
        }
        if x < width {
          $s_rgb(sse_lowbit_format!(@tail $kind, src, x), &mut rgb_out[x * 3..], width - x);
        }
      }
    }

    /// SSE4.1: packed legacy RGB/BGR → `R, G, B, A` bytes (α = `0xFF`).
    ///
    /// # Safety
    ///
    /// SSE4.1 available; `src` and `rgba_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn $to_rgba(src: &[u8], rgba_out: &mut [u8], width: usize) {
      debug_assert!(src.len() >= sse_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
      unsafe {
        let rmask = _mm_set1_epi16($rmask);
        let gmask = _mm_set1_epi16($gmask);
        let bmask = _mm_set1_epi16($bmask);
        let zero = _mm_setzero_si128();
        let alpha = _mm_set1_epi8(-1i8);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = sse_lowbit_format!(@load $kind, src, x);
          let r = _mm_and_si128(_mm_srli_epi16(px, $rsh), rmask);
          let g = _mm_and_si128(_mm_srli_epi16(px, $gsh), gmask);
          let b = _mm_and_si128(_mm_srli_epi16(px, $bsh), bmask);
          let r_u8 = _mm_packus_epi16($rexp(r), zero);
          let g_u8 = _mm_packus_epi16($gexp(g), zero);
          let b_u8 = _mm_packus_epi16($bexp(b), zero);
          let mut tmp = [0u8; 64];
          write_rgba_16(r_u8, g_u8, b_u8, alpha, tmp.as_mut_ptr());
          core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
          x += 8;
        }
        if x < width {
          $s_rgba(sse_lowbit_format!(@tail $kind, src, x), &mut rgba_out[x * 4..], width - x);
        }
      }
    }

    /// SSE4.1: packed legacy RGB/BGR → native `R, G, B` u16 (8 px/iter).
    ///
    /// # Safety
    ///
    /// SSE4.1 available; `src` and `rgb_u16_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn $to_rgb_u16(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
      debug_assert!(src.len() >= sse_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
      unsafe {
        let rmask = _mm_set1_epi16($rmask);
        let gmask = _mm_set1_epi16($gmask);
        let bmask = _mm_set1_epi16($bmask);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = sse_lowbit_format!(@load $kind, src, x);
          let r = _mm_and_si128(_mm_srli_epi16(px, $rsh), rmask);
          let g = _mm_and_si128(_mm_srli_epi16(px, $gsh), gmask);
          let b = _mm_and_si128(_mm_srli_epi16(px, $bsh), bmask);
          write_rgb_u16_8(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
          x += 8;
        }
        if x < width {
          $s_rgb_u16(
            sse_lowbit_format!(@tail $kind, src, x),
            &mut rgb_u16_out[x * 3..],
            width - x,
          );
        }
      }
    }

    /// SSE4.1: packed legacy RGB/BGR → native `R, G, B, A` u16 (α = `0xFFFF`).
    ///
    /// # Safety
    ///
    /// SSE4.1 available; `src` and `rgba_u16_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) unsafe fn $to_rgba_u16(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
      debug_assert!(src.len() >= sse_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
      unsafe {
        let rmask = _mm_set1_epi16($rmask);
        let gmask = _mm_set1_epi16($gmask);
        let bmask = _mm_set1_epi16($bmask);
        let alpha = _mm_set1_epi16(-1i16);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = sse_lowbit_format!(@load $kind, src, x);
          let r = _mm_and_si128(_mm_srli_epi16(px, $rsh), rmask);
          let g = _mm_and_si128(_mm_srli_epi16(px, $gsh), gmask);
          let b = _mm_and_si128(_mm_srli_epi16(px, $bsh), bmask);
          write_rgba_u16_8(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
          x += 8;
        }
        if x < width {
          $s_rgba_u16(
            sse_lowbit_format!(@tail $kind, src, x),
            &mut rgba_u16_out[x * 4..],
            width - x,
          );
        }
      }
    }
  };
}

sse_lowbit_format! {
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

sse_lowbit_format! {
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

sse_lowbit_format! {
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

sse_lowbit_format! {
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

sse_lowbit_format! {
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

sse_lowbit_format! {
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
