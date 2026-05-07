//! wasm-simd128 1-bit-per-pixel unpack kernels (Monoblack / Monowhite).
//!
//! # Bit-mask broadcast pattern (16 px / iter, 2 bytes / iter)
//!
//! Each byte covers 8 pixels (MSB first). For a pair of input bytes:
//! 1. `u8x16_splat(byte)` — broadcast to 16 lanes.
//! 2. `v128_and` with bit-position mask `[0x80,...,0x01]` repeated twice (16 bytes).
//! 3. `i8x16_eq(and, zero)` → 0x00 where bit was set, 0xFF where clear.
//! 4. For Monoblack (`INVERT=false`): negate with `v128_not`.
//! 5. For u16 outputs: process 8 px / iter (1 byte), zero-extend via `u16x8_extend_low_u8x16`.
//!
//! Tail: scalar fallback.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::wasm32::*;

use crate::row::{
  arch::wasm_simd128::{write_rgb_16, write_rgb_u16_8, write_rgba_16, write_rgba_u16_8},
  scalar::mono1bit as scalar,
};

/// Bit-position mask for 8 pixels (1 byte): [0x80, 0x40, ..., 0x01].
#[inline]
unsafe fn bit_mask_8() -> v128 {
  i8x16(
    0x80u8 as i8,
    0x40u8 as i8,
    0x20u8 as i8,
    0x10u8 as i8,
    0x08u8 as i8,
    0x04u8 as i8,
    0x02u8 as i8,
    0x01u8 as i8,
    0x80u8 as i8,
    0x40u8 as i8,
    0x20u8 as i8,
    0x10u8 as i8,
    0x08u8 as i8,
    0x04u8 as i8,
    0x02u8 as i8,
    0x01u8 as i8,
  )
}

/// Unpack 2 input bytes into a u8x16 luma vector (16 pixels).
/// INVERT=false (Monoblack): bit=1 → 0xFF; INVERT=true (Monowhite): bit=0 → 0xFF.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn unpack_2bytes_wasm<const INVERT: bool>(b0: u8, b1: u8) -> v128 {
  let mask = unsafe { bit_mask_8() };
  // Build the broadcast: b0 in low 8 lanes, b1 in high 8 lanes.
  let bcast = i8x16(
    b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b0 as i8, b1 as i8,
    b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8, b1 as i8,
  );
  let anded = v128_and(bcast, mask);
  let zero = i8x16_splat(0);
  // i8x16_eq: 0xFF where anded == 0 (bit NOT set), 0x00 where set.
  let cmp = i8x16_eq(anded, zero);
  if INVERT {
    cmp // Monowhite: 0xFF for bit=0
  } else {
    v128_not(cmp) // Monoblack: 0xFF for bit=1
  }
}

/// Zero-extend 8 u8 pixel values (low 8 lanes of a v128) to u16x8.
/// White (0xFF) maps to 0x00FF, matching Gray8's `with_luma_u16` contract.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn expand_y_to_u16x8_wasm(y_low8: v128) -> v128 {
  // u16x8_extend_low_u8x16 zero-extends the low 8 bytes to u16x8.
  u16x8_extend_low_u8x16(y_low8)
}

// ---- mono1bit → RGB u8 -------------------------------------------------------

/// wasm-simd128 `mono1bit_to_rgb_row<INVERT>`: unpack 1-bpp → packed RGB u8.
///
/// Block size: 16 px / iter (2 input bytes). Tail: scalar.
///
/// # Safety
/// simd128 must be enabled at compile time. `data.len() >= width.div_ceil(8)`.
/// `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
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
      let y = unpack_2bytes_wasm::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      write_rgb_16(y, y, y, out.as_mut_ptr().add(x * 3));
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

// ---- mono1bit → RGBA u8 -----------------------------------------------------

/// wasm-simd128 `mono1bit_to_rgba_row<INVERT>`: unpack 1-bpp → packed RGBA u8, α=0xFF.
///
/// # Safety
/// simd128 must be enabled. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
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
    let alpha = u8x16_splat(0xFF);
    while x + 16 <= width {
      let y = unpack_2bytes_wasm::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      write_rgba_16(y, y, y, alpha, out.as_mut_ptr().add(x * 4));
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

// ---- mono1bit → Luma u8 -----------------------------------------------------

/// wasm-simd128 `mono1bit_to_luma_row<INVERT>`: unpack 1-bpp → luma u8.
///
/// # Safety
/// simd128 must be enabled. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
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
      let y = unpack_2bytes_wasm::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      v128_store(out.as_mut_ptr().add(x).cast(), y);
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

// ---- mono1bit → RGB u16 -----------------------------------------------------

/// wasm-simd128 `mono1bit_to_rgb_u16_row<INVERT>`: unpack 1-bpp → RGB u16.
///
/// Block size: 8 px / iter (1 input byte). Tail: scalar.
///
/// # Safety
/// simd128 must be enabled. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
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
      let y_raw = unpack_2bytes_wasm::<INVERT>(data[byte_idx], 0);
      let y16 = expand_y_to_u16x8_wasm(y_raw);
      write_rgb_u16_8(y16, y16, y16, out.as_mut_ptr().add(x * 3));
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

// ---- mono1bit → RGBA u16 ----------------------------------------------------

/// wasm-simd128 `mono1bit_to_rgba_u16_row<INVERT>`: unpack 1-bpp → RGBA u16, α=0xFFFF.
///
/// # Safety
/// simd128 must be enabled. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
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
    let alpha = u16x8_splat(0x00FF);
    while x + 8 <= width {
      let y_raw = unpack_2bytes_wasm::<INVERT>(data[byte_idx], 0);
      let y16 = expand_y_to_u16x8_wasm(y_raw);
      write_rgba_u16_8(y16, y16, y16, alpha, out.as_mut_ptr().add(x * 4));
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

// ---- mono1bit → Luma u16 ----------------------------------------------------

/// wasm-simd128 `mono1bit_to_luma_u16_row<INVERT>`: unpack 1-bpp → luma u16.
///
/// Block size: 8 px / iter (1 input byte). Tail: scalar.
///
/// # Safety
/// simd128 must be enabled. `out.len() >= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
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
      let y_raw = unpack_2bytes_wasm::<INVERT>(data[byte_idx], 0);
      let y16 = expand_y_to_u16x8_wasm(y_raw);
      v128_store(out.as_mut_ptr().add(x).cast(), y16);
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

// ---- mono1bit → HSV ---------------------------------------------------------

/// wasm-simd128 `mono1bit_to_hsv_row<INVERT>`: H=0, S=0, V=Y.
///
/// # Safety
/// simd128 must be enabled. All output slices >= width.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
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
    let zero = i8x16_splat(0);
    while x + 16 <= width {
      let y = unpack_2bytes_wasm::<INVERT>(data[byte_idx], data[byte_idx + 1]);
      v128_store(h.as_mut_ptr().add(x).cast(), zero);
      v128_store(s.as_mut_ptr().add(x).cast(), zero);
      v128_store(v.as_mut_ptr().add(x).cast(), y);
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

// ---- Public wrappers --------------------------------------------------------

/// Monoblack → RGB u8 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monoblack_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u8 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monoblack_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<false>(data, out, width) }
}

/// Monoblack → RGB u16 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monoblack_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<false>(data, out, width) }
}

/// Monoblack → RGBA u16 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monoblack_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<false>(data, out, width) }
}

/// Monoblack → Luma u8 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monoblack_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<false>(data, out, width) }
}

/// Monoblack → Luma u16 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monoblack_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<false>(data, out, width) }
}

/// Monoblack → HSV (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monoblack_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<false>(data, h, s, v, width) }
}

/// Monowhite → RGB u8 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monowhite_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgb_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u8 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monowhite_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_rgba_row::<true>(data, out, width) }
}

/// Monowhite → RGB u16 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monowhite_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgb_u16_row::<true>(data, out, width) }
}

/// Monowhite → RGBA u16 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monowhite_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_rgba_u16_row::<true>(data, out, width) }
}

/// Monowhite → Luma u8 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monowhite_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  unsafe { mono1bit_to_luma_row::<true>(data, out, width) }
}

/// Monowhite → Luma u16 (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monowhite_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  unsafe { mono1bit_to_luma_u16_row::<true>(data, out, width) }
}

/// Monowhite → HSV (wasm-simd128).
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn monowhite_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  unsafe { mono1bit_to_hsv_row::<true>(data, h, s, v, width) }
}
