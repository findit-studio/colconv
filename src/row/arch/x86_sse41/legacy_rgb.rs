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
