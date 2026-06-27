//! wasm-simd128 kernels for legacy 16-bit packed-RGB source formats (Tier 7).
//!
//! Six source formats × 4 output variants = 24 kernels. Each format word is a
//! little-endian `u16` at 8 pixels per iteration (`v128_load` = 8 × u16).
//!
//! # Bit extraction
//!
//! - **RGB565**: `u16x8_shr(px, 11)` + `& 0x1F` → R5;
//!   `u16x8_shr(px, 5)` + `& 0x3F` → G6; `px & 0x1F` → B5.
//! - **BGR565**: same shifts, but R↔B swapped (R5 at bits [4:0],
//!   B5 at bits [15:11]).
//! - **RGB555**: `u16x8_shr(px, 10)` + `& 0x1F` → R5;
//!   `u16x8_shr(px, 5)` + `& 0x1F` → G5; `px & 0x1F` → B5.
//! - **BGR555**: same as RGB555 with R↔B swapped.
//! - **RGB444**: `u16x8_shr(px, 8)` + `& 0x0F` → R4;
//!   `u16x8_shr(px, 4)` + `& 0x0F` → G4; `px & 0x0F` → B4.
//! - **BGR444**: same as RGB444 with R↔B swapped.
//!
//! # Channel expansion
//!
//! | Bits | wasm (shift + OR)                                          |
//! |------|-----------------------------------------------------------|
//! | 5    | `v128_or(u16x8_shl(c, 3), u16x8_shr(c, 2))` → [0, 255]  |
//! | 6    | `v128_or(u16x8_shl(c, 2), u16x8_shr(c, 4))` → [0, 255]  |
//! | 4    | `v128_or(u16x8_shl(c, 4), c)`               → [0, 255]  |
//!
//! # u8 output
//!
//! After expansion each u16 lane holds a value in `[0, 255]`.
//! `u8x16_narrow_i16x8(expanded, zero)` narrows to 8 u8 bytes (in the
//! low half of the output v128). The 48-byte `write_rgb_16` / 64-byte
//! `write_rgba_16` helpers write 16 pixels; we use a local temp buffer
//! and `core::ptr::copy_nonoverlapping` to emit only 24 / 32 bytes for 8 pixels.
//!
//! # u16 output
//!
//! Skip `u8x16_narrow_i16x8`; feed the raw extracted (or expanded) u16
//! lanes directly into `write_rgb_u16_8` / `write_rgba_u16_8` which write
//! exactly 8 pixels (24 / 32 u16 elements).
//!
//! # Scalar tail
//!
//! When `width % 8 ≠ 0` the remainder is handled by `scalar::legacy_rgb`.

use core::arch::wasm32::*;

use super::*;

// Internal helpers.
/// Expand a v128 of 5-bit values in [0, 31] to 8-bit: `(c << 3) | (c >> 2)`.
#[inline(always)]
unsafe fn expand5(c: v128) -> v128 {
  v128_or(u16x8_shl(c, 3), u16x8_shr(c, 2))
}

/// Expand a v128 of 6-bit values in [0, 63] to 8-bit: `(c << 2) | (c >> 4)`.
#[inline(always)]
unsafe fn expand6(c: v128) -> v128 {
  v128_or(u16x8_shl(c, 2), u16x8_shr(c, 4))
}

/// Expand a v128 of 4-bit values in [0, 15] to 8-bit: `(c << 4) | c`.
#[inline(always)]
unsafe fn expand4(c: v128) -> v128 {
  v128_or(u16x8_shl(c, 4), c)
}

// RGB565 — R5 G6 B5, bits [15:11]=R, [10:5]=G, [4:0]=B.
/// wasm-simd128 RGB565 → packed `R, G, B` bytes (8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask_r5 = u16x8_splat(0x1F);
    let mask_g6 = u16x8_splat(0x3F);
    let zero = i16x8_splat(0);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r5 = v128_and(u16x8_shr(px, 11), mask_r5);
      let g6 = v128_and(u16x8_shr(px, 5), mask_g6);
      let b5 = v128_and(px, mask_r5);
      let r_u8 = u8x16_narrow_i16x8(expand5(r5), zero);
      let g_u8 = u8x16_narrow_i16x8(expand6(g6), zero);
      let b_u8 = u8x16_narrow_i16x8(expand5(b5), zero);
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

/// wasm-simd128 RGB565 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask_r5 = u16x8_splat(0x1F);
    let mask_g6 = u16x8_splat(0x3F);
    let zero = i16x8_splat(0);
    let alpha_u8 = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r5 = v128_and(u16x8_shr(px, 11), mask_r5);
      let g6 = v128_and(u16x8_shr(px, 5), mask_g6);
      let b5 = v128_and(px, mask_r5);
      let r_u8 = u8x16_narrow_i16x8(expand5(r5), zero);
      let g_u8 = u8x16_narrow_i16x8(expand6(g6), zero);
      let b_u8 = u8x16_narrow_i16x8(expand5(b5), zero);
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

/// wasm-simd128 RGB565 → packed `R, G, B` **u16** (native bit-width, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask_r5 = u16x8_splat(0x1F);
    let mask_g6 = u16x8_splat(0x3F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r = v128_and(u16x8_shr(px, 11), mask_r5);
      let g = v128_and(u16x8_shr(px, 5), mask_g6);
      let b = v128_and(px, mask_r5);
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

/// wasm-simd128 RGB565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask_r5 = u16x8_splat(0x1F);
    let mask_g6 = u16x8_splat(0x3F);
    let alpha = u16x8_splat(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r = v128_and(u16x8_shr(px, 11), mask_r5);
      let g = v128_and(u16x8_shr(px, 5), mask_g6);
      let b = v128_and(px, mask_r5);
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
/// wasm-simd128 BGR565 → packed `R, G, B` bytes (output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask_r5 = u16x8_splat(0x1F);
    let mask_g6 = u16x8_splat(0x3F);
    let zero = i16x8_splat(0);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      // BGR565: B at [15:11], G at [10:5], R at [4:0]
      let b5 = v128_and(u16x8_shr(px, 11), mask_r5);
      let g6 = v128_and(u16x8_shr(px, 5), mask_g6);
      let r5 = v128_and(px, mask_r5);
      let r_u8 = u8x16_narrow_i16x8(expand5(r5), zero);
      let g_u8 = u8x16_narrow_i16x8(expand6(g6), zero);
      let b_u8 = u8x16_narrow_i16x8(expand5(b5), zero);
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

/// wasm-simd128 BGR565 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask_r5 = u16x8_splat(0x1F);
    let mask_g6 = u16x8_splat(0x3F);
    let zero = i16x8_splat(0);
    let alpha_u8 = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let b5 = v128_and(u16x8_shr(px, 11), mask_r5);
      let g6 = v128_and(u16x8_shr(px, 5), mask_g6);
      let r5 = v128_and(px, mask_r5);
      let r_u8 = u8x16_narrow_i16x8(expand5(r5), zero);
      let g_u8 = u8x16_narrow_i16x8(expand6(g6), zero);
      let b_u8 = u8x16_narrow_i16x8(expand5(b5), zero);
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

/// wasm-simd128 BGR565 → packed `R, G, B` **u16** (native bit-width, output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask_r5 = u16x8_splat(0x1F);
    let mask_g6 = u16x8_splat(0x3F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      // BGR565: B at [15:11], G at [10:5], R at [4:0]. Output order: R, G, B.
      let b = v128_and(u16x8_shr(px, 11), mask_r5);
      let g = v128_and(u16x8_shr(px, 5), mask_g6);
      let r = v128_and(px, mask_r5);
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

/// wasm-simd128 BGR565 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask_r5 = u16x8_splat(0x1F);
    let mask_g6 = u16x8_splat(0x3F);
    let alpha = u16x8_splat(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let b = v128_and(u16x8_shr(px, 11), mask_r5);
      let g = v128_and(u16x8_shr(px, 5), mask_g6);
      let r = v128_and(px, mask_r5);
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
/// wasm-simd128 RGB555 → packed `R, G, B` bytes (8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = u16x8_splat(0x1F);
    let zero = i16x8_splat(0);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r5 = v128_and(u16x8_shr(px, 10), mask5);
      let g5 = v128_and(u16x8_shr(px, 5), mask5);
      let b5 = v128_and(px, mask5);
      let r_u8 = u8x16_narrow_i16x8(expand5(r5), zero);
      let g_u8 = u8x16_narrow_i16x8(expand5(g5), zero);
      let b_u8 = u8x16_narrow_i16x8(expand5(b5), zero);
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

/// wasm-simd128 RGB555 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = u16x8_splat(0x1F);
    let zero = i16x8_splat(0);
    let alpha_u8 = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r5 = v128_and(u16x8_shr(px, 10), mask5);
      let g5 = v128_and(u16x8_shr(px, 5), mask5);
      let b5 = v128_and(px, mask5);
      let r_u8 = u8x16_narrow_i16x8(expand5(r5), zero);
      let g_u8 = u8x16_narrow_i16x8(expand5(g5), zero);
      let b_u8 = u8x16_narrow_i16x8(expand5(b5), zero);
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

/// wasm-simd128 RGB555 → packed `R, G, B` **u16** (native bit-width, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = u16x8_splat(0x1F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r = v128_and(u16x8_shr(px, 10), mask5);
      let g = v128_and(u16x8_shr(px, 5), mask5);
      let b = v128_and(px, mask5);
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

/// wasm-simd128 RGB555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = u16x8_splat(0x1F);
    let alpha = u16x8_splat(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r = v128_and(u16x8_shr(px, 10), mask5);
      let g = v128_and(u16x8_shr(px, 5), mask5);
      let b = v128_and(px, mask5);
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
/// wasm-simd128 BGR555 → packed `R, G, B` bytes (output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask5 = u16x8_splat(0x1F);
    let zero = i16x8_splat(0);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      // BGR555: B at [14:10], G at [9:5], R at [4:0]
      let b5 = v128_and(u16x8_shr(px, 10), mask5);
      let g5 = v128_and(u16x8_shr(px, 5), mask5);
      let r5 = v128_and(px, mask5);
      let r_u8 = u8x16_narrow_i16x8(expand5(r5), zero);
      let g_u8 = u8x16_narrow_i16x8(expand5(g5), zero);
      let b_u8 = u8x16_narrow_i16x8(expand5(b5), zero);
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

/// wasm-simd128 BGR555 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask5 = u16x8_splat(0x1F);
    let zero = i16x8_splat(0);
    let alpha_u8 = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let b5 = v128_and(u16x8_shr(px, 10), mask5);
      let g5 = v128_and(u16x8_shr(px, 5), mask5);
      let r5 = v128_and(px, mask5);
      let r_u8 = u8x16_narrow_i16x8(expand5(r5), zero);
      let g_u8 = u8x16_narrow_i16x8(expand5(g5), zero);
      let b_u8 = u8x16_narrow_i16x8(expand5(b5), zero);
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

/// wasm-simd128 BGR555 → packed `R, G, B` **u16** (native bit-width, output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask5 = u16x8_splat(0x1F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      // BGR555: B at [14:10], G at [9:5], R at [4:0]. Output order: R, G, B.
      let b = v128_and(u16x8_shr(px, 10), mask5);
      let g = v128_and(u16x8_shr(px, 5), mask5);
      let r = v128_and(px, mask5);
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

/// wasm-simd128 BGR555 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask5 = u16x8_splat(0x1F);
    let alpha = u16x8_splat(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let b = v128_and(u16x8_shr(px, 10), mask5);
      let g = v128_and(u16x8_shr(px, 5), mask5);
      let r = v128_and(px, mask5);
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
/// wasm-simd128 RGB444 → packed `R, G, B` bytes (8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = u16x8_splat(0x0F);
    let zero = i16x8_splat(0);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r4 = v128_and(u16x8_shr(px, 8), mask4);
      let g4 = v128_and(u16x8_shr(px, 4), mask4);
      let b4 = v128_and(px, mask4);
      let r_u8 = u8x16_narrow_i16x8(expand4(r4), zero);
      let g_u8 = u8x16_narrow_i16x8(expand4(g4), zero);
      let b_u8 = u8x16_narrow_i16x8(expand4(b4), zero);
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

/// wasm-simd128 RGB444 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = u16x8_splat(0x0F);
    let zero = i16x8_splat(0);
    let alpha_u8 = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r4 = v128_and(u16x8_shr(px, 8), mask4);
      let g4 = v128_and(u16x8_shr(px, 4), mask4);
      let b4 = v128_and(px, mask4);
      let r_u8 = u8x16_narrow_i16x8(expand4(r4), zero);
      let g_u8 = u8x16_narrow_i16x8(expand4(g4), zero);
      let b_u8 = u8x16_narrow_i16x8(expand4(b4), zero);
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

/// wasm-simd128 RGB444 → packed `R, G, B` **u16** (native 4-bit width, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = u16x8_splat(0x0F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r = v128_and(u16x8_shr(px, 8), mask4);
      let g = v128_and(u16x8_shr(px, 4), mask4);
      let b = v128_and(px, mask4);
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

/// wasm-simd128 RGB444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgb444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = u16x8_splat(0x0F);
    let alpha = u16x8_splat(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let r = v128_and(u16x8_shr(px, 8), mask4);
      let g = v128_and(u16x8_shr(px, 4), mask4);
      let b = v128_and(px, mask4);
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
/// wasm-simd128 BGR444 → packed `R, G, B` bytes (output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  unsafe {
    let mask4 = u16x8_splat(0x0F);
    let zero = i16x8_splat(0);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      // BGR444: B at [11:8], G at [7:4], R at [3:0]
      let b4 = v128_and(u16x8_shr(px, 8), mask4);
      let g4 = v128_and(u16x8_shr(px, 4), mask4);
      let r4 = v128_and(px, mask4);
      let r_u8 = u8x16_narrow_i16x8(expand4(r4), zero);
      let g_u8 = u8x16_narrow_i16x8(expand4(g4), zero);
      let b_u8 = u8x16_narrow_i16x8(expand4(b4), zero);
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

/// wasm-simd128 BGR444 → packed `R, G, B, A` bytes (α = `0xFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  unsafe {
    let mask4 = u16x8_splat(0x0F);
    let zero = i16x8_splat(0);
    let alpha_u8 = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let b4 = v128_and(u16x8_shr(px, 8), mask4);
      let g4 = v128_and(u16x8_shr(px, 4), mask4);
      let r4 = v128_and(px, mask4);
      let r_u8 = u8x16_narrow_i16x8(expand4(r4), zero);
      let g_u8 = u8x16_narrow_i16x8(expand4(g4), zero);
      let b_u8 = u8x16_narrow_i16x8(expand4(b4), zero);
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

/// wasm-simd128 BGR444 → packed `R, G, B` **u16** (native 4-bit width, output R-first, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgb_u16_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  unsafe {
    let mask4 = u16x8_splat(0x0F);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      // BGR444: B at [11:8], G at [7:4], R at [3:0]. Output order: R, G, B.
      let b = v128_and(u16x8_shr(px, 8), mask4);
      let g = v128_and(u16x8_shr(px, 4), mask4);
      let r = v128_and(px, mask4);
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

/// wasm-simd128 BGR444 → packed `R, G, B, A` **u16** (α = `0xFFFF`, 8 px/iter).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time (caller obligation).
/// 2. `src.len() >= width * 2`.
/// 3. `rgba_u16_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn bgr444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  unsafe {
    let mask4 = u16x8_splat(0x0F);
    let alpha = u16x8_splat(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px = v128_load(src.as_ptr().add(x * 2).cast::<v128>());
      let b = v128_and(u16x8_shr(px, 8), mask4);
      let g = v128_and(u16x8_shr(px, 4), mask4);
      let r = v128_and(px, mask4);
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
// Each iteration produces 8 pixels as a `v128` of native source bytes (byte
// formats: widen 8 source bytes; nibble formats: de-interleave 4 source bytes
// into 8 nibble lanes), then reuses the same shift+mask extraction,
// bit-replication expansion, and `write_rgb_*` / `write_rgba_*` interleaved
// stores as the 16-bit formats above. The `width % 8` remainder defers to
// `scalar`.
// =========================================================================

/// Bit-replicate a v128 of 1-bit values (`0`/`1`) to 8-bit: `c * 0xFF`.
#[inline(always)]
unsafe fn expand1(c: v128) -> v128 {
  u16x8_mul(c, u16x8_splat(0xFF))
}

/// Bit-replicate a v128 of 2-bit values (`0..=3`) to 8-bit: `c * 0x55`.
#[inline(always)]
unsafe fn expand2(c: v128) -> v128 {
  u16x8_mul(c, u16x8_splat(0x55))
}

/// Bit-replicate a v128 of 3-bit values (`0..=7`) to 8-bit:
/// `(c << 5) | (c << 2) | (c >> 1)`.
#[inline(always)]
unsafe fn expand3(c: v128) -> v128 {
  v128_or(v128_or(u16x8_shl(c, 5), u16x8_shl(c, 2)), u16x8_shr(c, 1))
}

/// Load 8 packed 1-byte-per-pixel source bytes and widen to a `v128` of
/// native pixel bytes (8 u16 lanes).
///
/// # Safety
///
/// `ptr` must be valid for an 8-byte read; simd128 enabled.
#[inline(always)]
unsafe fn load_byte_px8(ptr: *const u8) -> v128 {
  unsafe { u16x8_extend_low_u8x16(v128_load64_zero(ptr.cast())) }
}

/// Load 4 packed 2-pixel-per-byte source bytes and de-interleave the nibbles
/// into a `v128` of 8 native pixel nibbles (even pixel = high nibble `[7:4]`,
/// odd pixel = low nibble `[3:0]`).
///
/// # Safety
///
/// `ptr` must be valid for a 4-byte read; simd128 enabled.
#[inline(always)]
unsafe fn load_nibble_px8(ptr: *const u8) -> v128 {
  unsafe {
    let raw = v128_load32_zero(ptr.cast());
    // Duplicate each of the 4 low bytes: [b0, b0, b1, b1, b2, b2, b3, b3, …].
    let dup = u8x16_shuffle::<0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0>(raw, raw);
    let w = u16x8_extend_low_u8x16(dup);
    let hi = u16x8_shr(w, 4);
    let lo = v128_and(w, u16x8_splat(0x0F));
    // u16 lanes [0xFFFF, 0, 0xFFFF, 0, …]: even lanes select the high nibble.
    let even = u32x4_splat(0x0000_FFFF);
    v128_bitselect(hi, lo, even)
  }
}

/// Emits the four wasm-simd128 output kernels (rgb / rgba / rgb_u16 /
/// rgba_u16) for one legacy bit-packed format. `$kind` is `byte` or `nibble`;
/// each channel is `(right_shift, native_mask, expand_fn)`.
macro_rules! wasm_lowbit_format {
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
    /// wasm-simd128: packed legacy RGB/BGR → `R, G, B` bytes (8 px/iter).
    ///
    /// # Safety
    ///
    /// simd128 enabled; `src` and `rgb_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn $to_rgb(src: &[u8], rgb_out: &mut [u8], width: usize) {
      debug_assert!(src.len() >= wasm_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
      unsafe {
        let rmask = u16x8_splat($rmask);
        let gmask = u16x8_splat($gmask);
        let bmask = u16x8_splat($bmask);
        let zero = i16x8_splat(0);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = wasm_lowbit_format!(@load $kind, src, x);
          let r = v128_and(u16x8_shr(px, $rsh), rmask);
          let g = v128_and(u16x8_shr(px, $gsh), gmask);
          let b = v128_and(u16x8_shr(px, $bsh), bmask);
          let r_u8 = u8x16_narrow_i16x8($rexp(r), zero);
          let g_u8 = u8x16_narrow_i16x8($gexp(g), zero);
          let b_u8 = u8x16_narrow_i16x8($bexp(b), zero);
          let mut tmp = [0u8; 48];
          write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
          core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
          x += 8;
        }
        if x < width {
          $s_rgb(wasm_lowbit_format!(@tail $kind, src, x), &mut rgb_out[x * 3..], width - x);
        }
      }
    }

    /// wasm-simd128: packed legacy RGB/BGR → `R, G, B, A` bytes (α = `0xFF`).
    ///
    /// # Safety
    ///
    /// simd128 enabled; `src` and `rgba_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn $to_rgba(src: &[u8], rgba_out: &mut [u8], width: usize) {
      debug_assert!(src.len() >= wasm_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
      unsafe {
        let rmask = u16x8_splat($rmask);
        let gmask = u16x8_splat($gmask);
        let bmask = u16x8_splat($bmask);
        let zero = i16x8_splat(0);
        let alpha = u8x16_splat(0xFF);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = wasm_lowbit_format!(@load $kind, src, x);
          let r = v128_and(u16x8_shr(px, $rsh), rmask);
          let g = v128_and(u16x8_shr(px, $gsh), gmask);
          let b = v128_and(u16x8_shr(px, $bsh), bmask);
          let r_u8 = u8x16_narrow_i16x8($rexp(r), zero);
          let g_u8 = u8x16_narrow_i16x8($gexp(g), zero);
          let b_u8 = u8x16_narrow_i16x8($bexp(b), zero);
          let mut tmp = [0u8; 64];
          write_rgba_16(r_u8, g_u8, b_u8, alpha, tmp.as_mut_ptr());
          core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
          x += 8;
        }
        if x < width {
          $s_rgba(wasm_lowbit_format!(@tail $kind, src, x), &mut rgba_out[x * 4..], width - x);
        }
      }
    }

    /// wasm-simd128: packed legacy RGB/BGR → native `R, G, B` u16 (8 px/iter).
    ///
    /// # Safety
    ///
    /// simd128 enabled; `src` and `rgb_u16_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn $to_rgb_u16(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
      debug_assert!(src.len() >= wasm_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
      unsafe {
        let rmask = u16x8_splat($rmask);
        let gmask = u16x8_splat($gmask);
        let bmask = u16x8_splat($bmask);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = wasm_lowbit_format!(@load $kind, src, x);
          let r = v128_and(u16x8_shr(px, $rsh), rmask);
          let g = v128_and(u16x8_shr(px, $gsh), gmask);
          let b = v128_and(u16x8_shr(px, $bsh), bmask);
          write_rgb_u16_8(r, g, b, rgb_u16_out.as_mut_ptr().add(x * 3));
          x += 8;
        }
        if x < width {
          $s_rgb_u16(
            wasm_lowbit_format!(@tail $kind, src, x),
            &mut rgb_u16_out[x * 3..],
            width - x,
          );
        }
      }
    }

    /// wasm-simd128: packed legacy RGB/BGR → native `R, G, B, A` u16
    /// (α = `0xFFFF`, 8 px/iter).
    ///
    /// # Safety
    ///
    /// simd128 enabled; `src` and `rgba_u16_out` long enough for `width`.
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn $to_rgba_u16(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
      debug_assert!(src.len() >= wasm_lowbit_format!(@srcmin $kind, width), "src too short");
      debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
      unsafe {
        let rmask = u16x8_splat($rmask);
        let gmask = u16x8_splat($gmask);
        let bmask = u16x8_splat($bmask);
        let alpha = u16x8_splat(0xFFFF);
        let mut x = 0usize;
        while x + 8 <= width {
          let px = wasm_lowbit_format!(@load $kind, src, x);
          let r = v128_and(u16x8_shr(px, $rsh), rmask);
          let g = v128_and(u16x8_shr(px, $gsh), gmask);
          let b = v128_and(u16x8_shr(px, $bsh), bmask);
          write_rgba_u16_8(r, g, b, alpha, rgba_u16_out.as_mut_ptr().add(x * 4));
          x += 8;
        }
        if x < width {
          $s_rgba_u16(
            wasm_lowbit_format!(@tail $kind, src, x),
            &mut rgba_u16_out[x * 4..],
            width - x,
          );
        }
      }
    }
  };
}

wasm_lowbit_format! {
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

wasm_lowbit_format! {
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

wasm_lowbit_format! {
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

wasm_lowbit_format! {
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

wasm_lowbit_format! {
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

wasm_lowbit_format! {
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
