//! NEON 1-bit-per-pixel unpack kernels (Monoblack / Monowhite).
//!
//! # Bit-mask broadcast pattern (16 px / iter, 2 bytes / iter)
//!
//! Each byte covers 8 pixels (MSB first). For a pair of input bytes:
//! 1. Broadcast each byte to 8 u8 NEON lanes with `vdup_n_u8`.
//! 2. AND with the bit-position mask `[0x80,0x40,0x20,0x10,0x08,0x04,0x02,0x01]`.
//! 3. `vtstq_u8(v, mask)` tests "is bit non-zero?" → 0xFF per set bit, 0x00 per clear bit.
//! 4. `INVERT=true` (Monowhite): `vmvnq_u8` to flip all bits.
//! 5. For RGB/RGBA: `vst3q_u8` / `vst4q_u8` broadcast the Y vector across channels.
//! 6. For u16 outputs: Y = y as u16 (zero-extend via `vmovl_u8`); white maps to 0x00FF.
//!
//! Tail (< 16 pixels) falls back to scalar.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use crate::row::scalar::mono1bit as scalar;

/// Bit-position mask: `[0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01]`.
/// Lane i tests bit (7 - i) of the input byte.
const BIT_MASK: [u8; 8] = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];

/// Unpack 1 input byte into a u8x8 luma vector (8 pixels).
/// Each lane is 0xFF (pixel=1) or 0x00 (pixel=0) for INVERT=false (Monoblack).
/// For INVERT=true (Monowhite), the result is complemented.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_byte<const INVERT: bool>(b: u8) -> uint8x8_t {
  unsafe {
    let mask = vld1_u8(BIT_MASK.as_ptr());
    let v = vdup_n_u8(b);
    // vtst_u8: lane-wise (v & mask) != 0 → 0xFF / 0x00
    let result = vtst_u8(v, mask);
    if INVERT { vmvn_u8(result) } else { result }
  }
}

/// Unpack 2 consecutive input bytes into a u8x16 luma vector (16 pixels).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn unpack_2bytes<const INVERT: bool>(b0: u8, b1: u8) -> uint8x16_t {
  let lo = unsafe { unpack_byte::<INVERT>(b0) };
  let hi = unsafe { unpack_byte::<INVERT>(b1) };
  vcombine_u8(lo, hi)
}

/// Expand a u8x8 luma vector to u16x8 by zero-extending each u8 to u16.
/// White (0xFF) maps to 0x00FF, matching Gray8's `with_luma_u16` contract.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn expand_y_to_u16x8(y8: uint8x8_t) -> uint16x8_t {
  vmovl_u8(y8)
}

// ---- Monoblack / Monowhite → RGB u8 -----------------------------------------

/// NEON `mono1bit_to_rgb_row<INVERT>`: unpack 1-bpp → packed RGB u8.
///
/// Block size: 16 px / iter (2 input bytes per block). Tail: scalar.
///
/// # Safety
/// NEON must be available. `data.len() >= width.div_ceil(8)`.
/// `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mono1bit_to_rgb_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = unpack_2bytes::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      let rgb = uint8x16x3_t(y, y, y);
      vst3q_u8(out.as_mut_ptr().add(x * 3), rgb);
      x += 16;
      byte_idx += 2;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_rgb_row(&data[byte_idx..], &mut out[x * 3..width * 3], width - x);
    } else {
      scalar::monoblack_to_rgb_row(&data[byte_idx..], &mut out[x * 3..width * 3], width - x);
    }
  }
}

// ---- Monoblack / Monowhite → RGBA u8 ----------------------------------------

/// NEON `mono1bit_to_rgba_row<INVERT>`: unpack 1-bpp → packed RGBA u8, α=0xFF.
///
/// # Safety
/// NEON must be available. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mono1bit_to_rgba_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    let alpha = vdupq_n_u8(0xFF);
    while x + 16 <= width {
      let y = unpack_2bytes::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      let rgba = uint8x16x4_t(y, y, y, alpha);
      vst4q_u8(out.as_mut_ptr().add(x * 4), rgba);
      x += 16;
      byte_idx += 2;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_rgba_row(&data[byte_idx..], &mut out[x * 4..width * 4], width - x);
    } else {
      scalar::monoblack_to_rgba_row(&data[byte_idx..], &mut out[x * 4..width * 4], width - x);
    }
  }
}

// ---- Monoblack / Monowhite → Luma u8 ----------------------------------------

/// NEON `mono1bit_to_luma_row<INVERT>`: unpack 1-bpp → luma u8.
///
/// # Safety
/// NEON must be available. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mono1bit_to_luma_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    while x + 16 <= width {
      let y = unpack_2bytes::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      vst1q_u8(out.as_mut_ptr().add(x), y);
      x += 16;
      byte_idx += 2;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_luma_row(&data[byte_idx..], &mut out[x..width], width - x);
    } else {
      scalar::monoblack_to_luma_row(&data[byte_idx..], &mut out[x..width], width - x);
    }
  }
}

// ---- Monoblack / Monowhite → RGB u16 ----------------------------------------

/// NEON `mono1bit_to_rgb_u16_row<INVERT>`: unpack 1-bpp → RGB u16.
///
/// Block size: 8 px / iter (1 input byte per block). Tail: scalar.
///
/// # Safety
/// NEON must be available. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mono1bit_to_rgb_u16_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    while x + 8 <= width {
      let y8 = unpack_byte::<INVERT>(data[byte_idx]);
      let y16 = expand_y_to_u16x8(y8);
      let rgb = uint16x8x3_t(y16, y16, y16);
      vst3q_u16(out.as_mut_ptr().add(x * 3), rgb);
      x += 8;
      byte_idx += 1;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_rgb_u16_row(&data[byte_idx..], &mut out[x * 3..width * 3], width - x);
    } else {
      scalar::monoblack_to_rgb_u16_row(&data[byte_idx..], &mut out[x * 3..width * 3], width - x);
    }
  }
}

// ---- Monoblack / Monowhite → RGBA u16 ---------------------------------------

/// NEON `mono1bit_to_rgba_u16_row<INVERT>`: unpack 1-bpp → RGBA u16, α=0xFFFF.
///
/// # Safety
/// NEON must be available. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mono1bit_to_rgba_u16_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    let alpha = vdupq_n_u16(0x00FF);
    while x + 8 <= width {
      let y8 = unpack_byte::<INVERT>(data[byte_idx]);
      let y16 = expand_y_to_u16x8(y8);
      let rgba = uint16x8x4_t(y16, y16, y16, alpha);
      vst4q_u16(out.as_mut_ptr().add(x * 4), rgba);
      x += 8;
      byte_idx += 1;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_rgba_u16_row(&data[byte_idx..], &mut out[x * 4..width * 4], width - x);
    } else {
      scalar::monoblack_to_rgba_u16_row(&data[byte_idx..], &mut out[x * 4..width * 4], width - x);
    }
  }
}

// ---- Monoblack / Monowhite → Luma u16 ----------------------------------------

/// NEON `mono1bit_to_luma_u16_row<INVERT>`: unpack 1-bpp → luma u16.
///
/// # Safety
/// NEON must be available. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mono1bit_to_luma_u16_row<const INVERT: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    while x + 8 <= width {
      let y8 = unpack_byte::<INVERT>(data[byte_idx]);
      let y16 = expand_y_to_u16x8(y8);
      vst1q_u16(out.as_mut_ptr().add(x), y16);
      x += 8;
      byte_idx += 1;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_luma_u16_row(&data[byte_idx..], &mut out[x..width], width - x);
    } else {
      scalar::monoblack_to_luma_u16_row(&data[byte_idx..], &mut out[x..width], width - x);
    }
  }
}

// ---- Monoblack / Monowhite → HSV --------------------------------------------

/// NEON `mono1bit_to_hsv_row<INVERT>`: H=0, S=0, V=Y.
///
/// # Safety
/// NEON must be available. All output slices >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mono1bit_to_hsv_row<const INVERT: bool>(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(h.len() >= width);
  debug_assert!(s.len() >= width);
  debug_assert!(v.len() >= width);
  let mut x = 0usize;
  let mut byte_idx = 0usize;
  unsafe {
    let zero = vdupq_n_u8(0);
    while x + 16 <= width {
      let y = unpack_2bytes::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      vst1q_u8(h.as_mut_ptr().add(x), zero);
      vst1q_u8(s.as_mut_ptr().add(x), zero);
      vst1q_u8(v.as_mut_ptr().add(x), y);
      x += 16;
      byte_idx += 2;
    }
  }
  if x < width {
    if INVERT {
      scalar::monowhite_to_hsv_row(
        &data[byte_idx..],
        &mut h[x..width],
        &mut s[x..width],
        &mut v[x..width],
        width - x,
      );
    } else {
      scalar::monoblack_to_hsv_row(
        &data[byte_idx..],
        &mut h[x..width],
        &mut s[x..width],
        &mut v[x..width],
        width - x,
      );
    }
  }
}

// ---- Public wrappers for the dispatch layer ---------------------------------

/// Monoblack → RGB u8 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monoblack_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u8 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monoblack_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<false>(data, out, width) }
}

/// Monoblack → RGB u16 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monoblack_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u16 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monoblack_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<false>(data, out, width) }
}

/// Monoblack → Luma u8 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monoblack_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<false>(data, out, width) }
}

/// Monoblack → Luma u16 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monoblack_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<false>(data, out, width) }
}

/// Monoblack → HSV (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monoblack_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<false>(data, h, s, v, width) }
}

/// Monowhite → RGB u8 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monowhite_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u8 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monowhite_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<true>(data, out, width) }
}

/// Monowhite → RGB u16 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monowhite_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u16 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monowhite_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<true>(data, out, width) }
}

/// Monowhite → Luma u8 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monowhite_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<true>(data, out, width) }
}

/// Monowhite → Luma u16 (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monowhite_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<true>(data, out, width) }
}

/// Monowhite → HSV (NEON).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn monowhite_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<true>(data, h, s, v, width) }
}
