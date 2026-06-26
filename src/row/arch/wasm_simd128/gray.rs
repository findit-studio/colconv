//! WebAssembly simd128 gray kernel implementations.
//!
//! Gray → luma / luma_u16 / HSV paths get simd128 (16-px / 8-px blocks).
//! Packed-channel interleave paths (RGB, RGBA) delegate to scalar since
//! the 3/4-channel store pattern is verbose and scalar auto-vectorizes well.
//!
//! # `full_range` parameter
//!
//! For RGB/RGBA/HSV kernels, `full_range = true` uses the existing fast
//! simd128 path. `full_range = false` (limited-range) falls back to scalar
//! since limited-range rescaling is the less-common path and the scalar
//! formulation is simple and correct.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::wasm32::*;

use crate::row::{
  arch::wasm_simd128::endian::{load_endian_u16x8, load_endian_u32x4},
  scalar::{bits_mask, gray as scalar, grayf16, grayf32, ya8, ya16, yaf16, yaf32},
};

// ---- Gray8 ------------------------------------------------------------------

/// wasm-simd128 `gray8_to_rgb_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray8_to_rgb_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray8_to_rgb_row(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray8_to_rgba_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray8_to_rgba_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray8_to_rgba_row(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray8_to_hsv_row`: H=0, S=0, V=Y. 16 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// simd128 must be enabled. All planes have length >= width.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray8_to_hsv_row(
  y_plane: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  if !full_range {
    return scalar::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let zero = i64x2(0, 0);
    while x + 16 <= width {
      let v = v128_load(y_plane.as_ptr().add(x).cast());
      v128_store(h_out.as_mut_ptr().add(x).cast(), zero);
      v128_store(s_out.as_mut_ptr().add(x).cast(), zero);
      v128_store(v_out.as_mut_ptr().add(x).cast(), v);
      x += 16;
    }
  }
  if x < width {
    scalar::gray8_to_hsv_row(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- GrayN (const BITS) -----------------------------------------------------

/// wasm-simd128 `gray_n_to_rgb_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray_n_to_rgba_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray_n_to_rgba_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray_n_to_rgb_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray_n_to_rgba_u16_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray_n_to_luma_row<BITS>`: mask + shift → u8. 8 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// simd128 must be enabled. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray_n_to_luma_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  let mut x = 0usize;
  unsafe {
    let mask_v = u16x8_splat(mask);
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = v128_and(raw, mask_v);
      let shifted = u16x8_shr(masked, shift);
      // Narrow u16x8 → u8x8 via u8x16_narrow_i16x8 (saturation, but values
      // are already in [0, 255] after the shift so no saturation occurs).
      let zero = i64x2(0, 0);
      let narrowed = u8x16_narrow_i16x8(shifted, zero);
      // Store low 8 bytes (8 pixels).
      let val = i64x2_extract_lane::<0>(narrowed) as u64;
      out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_row::<BITS, BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `gray_n_to_luma_u16_row<BITS>`: mask, store. 8 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// simd128 must be enabled. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray_n_to_luma_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = u16x8_splat(mask);
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = v128_and(raw, mask_v);
      v128_store(out.as_mut_ptr().add(x).cast(), masked);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_u16_row::<BITS, BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `gray_n_to_hsv_row<BITS>`: H=0, S=0, V = mask+shift.
/// 8 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// simd128 must be enabled. All slices have length >= width.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray_n_to_hsv_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  if !full_range {
    return scalar::gray_n_to_hsv_row::<BITS, BE>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  let mut x = 0usize;
  unsafe {
    let mask_v = u16x8_splat(mask);
    let zero = i64x2(0, 0);
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let masked = v128_and(raw, mask_v);
      let shifted = u16x8_shr(masked, shift);
      let narrowed = u8x16_narrow_i16x8(shifted, zero);
      let val = i64x2_extract_lane::<0>(narrowed) as u64;
      let bytes = val.to_le_bytes();
      h_out[x..x + 8].fill(0);
      s_out[x..x + 8].fill(0);
      v_out[x..x + 8].copy_from_slice(&bytes);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_hsv_row::<BITS, BE>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- Gray16 -----------------------------------------------------------------

/// wasm-simd128 `gray16_to_rgb_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray16_to_rgb_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_row::<BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray16_to_rgba_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray16_to_rgba_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_row::<BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray16_to_rgb_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray16_to_rgb_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray16_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray16_to_rgba_u16_row`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// simd128 must be enabled.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray16_to_rgba_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray16_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray16_to_luma_row`: `>> 8` → u8. 8 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// simd128 must be enabled. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray16_to_luma_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    let zero = i64x2(0, 0);
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let shifted = u16x8_shr(raw, 8);
      let narrowed = u8x16_narrow_i16x8(shifted, zero);
      let val = i64x2_extract_lane::<0>(narrowed) as u64;
      out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `gray16_to_luma_u16_row`: identity copy. 8 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// simd128 must be enabled. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray16_to_luma_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      v128_store(out.as_mut_ptr().add(x).cast(), y);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `gray16_to_hsv_row`: `>> 8`, H=0, S=0, V=Y8. 8 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// simd128 must be enabled. All slices have length >= width.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray16_to_hsv_row<const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  if !full_range {
    return scalar::gray16_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let zero = i64x2(0, 0);
    while x + 8 <= width {
      let raw = load_endian_u16x8::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 2));
      let shifted = u16x8_shr(raw, 8);
      let narrowed = u8x16_narrow_i16x8(shifted, zero);
      let val = i64x2_extract_lane::<0>(narrowed) as u64;
      let bytes = val.to_le_bytes();
      h_out[x..x + 8].fill(0);
      s_out[x..x + 8].fill(0);
      v_out[x..x + 8].copy_from_slice(&bytes);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_hsv_row::<BE>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- Gray32 -----------------------------------------------------------------
//
// Full-bit integer twin of Gray16, widened u16 → u32. As with Gray16, the
// packed-RGB(A) paths fall back to scalar; luma / luma_u16 / hsv get simd128
// bodies processing 8 px per iter (two u32x4 loads). The u32 narrows reuse the
// `load_endian_u32x4` loader the grayf32 path established: `u32x4_shr` then
// `u16x8_narrow_i32x4` lands the native u16 sample (`>> 16`), and a further
// `u8x16_narrow_i16x8` lands the u8 sample (`>> 24`).

/// wasm-simd128 `gray32_to_rgb_row`: `>> 24` → broadcast (scalar fallback).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray32_to_rgb_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray32_to_rgb_row::<BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray32_to_rgba_row`: `>> 24` → RGBA u8 (scalar fallback).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray32_to_rgba_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray32_to_rgba_row::<BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray32_to_rgb_u16_row`: `>> 16` → RGB u16 (scalar fallback).
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray32_to_rgb_u16_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  scalar::gray32_to_rgb_u16_row::<BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray32_to_rgba_u16_row`: `>> 16` → RGBA u16 (scalar fallback).
///
/// # Safety
/// simd128 must be enabled.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray32_to_rgba_u16_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  scalar::gray32_to_rgba_u16_row::<BE>(y_plane, out, width, full_range);
}

/// wasm-simd128 `gray32_to_luma_row`: `>> 24`, narrow u32 → u8. 8 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// simd128 must be enabled. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray32_to_luma_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    let zero = i64x2(0, 0);
    while x + 8 <= width {
      let lo = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 4));
      let hi = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add((x + 4) * 4));
      let u16v = u16x8_narrow_i32x4(u32x4_shr(lo, 24), u32x4_shr(hi, 24));
      let narrowed = u8x16_narrow_i16x8(u16v, zero);
      let val = i64x2_extract_lane::<0>(narrowed) as u64;
      out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    scalar::gray32_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `gray32_to_luma_u16_row`: `>> 16`, narrow u32 → u16. 8 pixels/iter.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// simd128 must be enabled. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray32_to_luma_u16_row<const BE: bool>(
  y_plane: &[u32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let lo = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 4));
      let hi = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add((x + 4) * 4));
      let u16v = u16x8_narrow_i32x4(u32x4_shr(lo, 16), u32x4_shr(hi, 16));
      v128_store(out.as_mut_ptr().add(x).cast(), u16v);
      x += 8;
    }
  }
  if x < width {
    scalar::gray32_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `gray32_to_hsv_row`: `>> 24`, H=0, S=0, V=Y8. 8 pixels/iter.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// simd128 must be enabled. All slices have length >= width.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gray32_to_hsv_row<const BE: bool>(
  y_plane: &[u32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  if !full_range {
    return scalar::gray32_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let zero = i64x2(0, 0);
    while x + 8 <= width {
      let lo = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 4));
      let hi = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add((x + 4) * 4));
      let u16v = u16x8_narrow_i32x4(u32x4_shr(lo, 24), u32x4_shr(hi, 24));
      let narrowed = u8x16_narrow_i16x8(u16v, zero);
      let val = i64x2_extract_lane::<0>(narrowed) as u64;
      let bytes = val.to_le_bytes();
      h_out[x..x + 8].fill(0);
      s_out[x..x + 8].fill(0);
      v_out[x..x + 8].copy_from_slice(&bytes);
      x += 8;
    }
  }
  if x < width {
    scalar::gray32_to_hsv_row::<BE>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- Grayf32 ----------------------------------------------------------------

/// wasm-simd128 `grayf32_to_rgb_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf32_to_rgb_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  grayf32::grayf32_to_rgb_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf32_to_rgba_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf32_to_rgba_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  grayf32::grayf32_to_rgba_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf32_to_rgb_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf32_to_rgb_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  grayf32::grayf32_to_rgb_u16_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf32_to_rgba_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf32_to_rgba_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  grayf32::grayf32_to_rgba_u16_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf32_to_rgb_f32_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)] // dispatcher always uses scalar; function is exercised by tests only
pub(crate) unsafe fn grayf32_to_rgb_f32_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  grayf32::grayf32_to_rgb_f32_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf32_to_luma_row`: clamp→scale→round→u8. 4 pixels/iter.
///
/// Uses `f32x4_add(0.5)` + `i32x4_trunc_sat_f32x4` for MXCSR-independent
/// round-to-nearest (ties round up, matching scalar `+0.5 as u8`).
///
/// # Safety
/// simd128 must be enabled. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf32_to_luma_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = f32x4_splat(255.0);
  let zero4 = f32x4_splat(0.0);
  let one4 = f32x4_splat(1.0);
  let half = f32x4_splat(0.5);
  let zero16 = i64x2(0, 0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 4));
      let clamped = f32x4_min(f32x4_max(y, zero4), one4);
      let scaled = f32x4_mul(clamped, scale);
      let rounded = i32x4_trunc_sat_f32x4(f32x4_add(scaled, half));
      // Narrow i32x4 → i16x8 → u8x16, then extract low 4 bytes.
      let narrow16 = i16x8_narrow_i32x4(rounded, zero16);
      let narrow8 = u8x16_narrow_i16x8(narrow16, zero16);
      let val = i32x4_extract_lane::<0>(narrow8) as u32;
      let bytes = val.to_le_bytes();
      out[x..x + 4].copy_from_slice(&bytes);
      x += 4;
    }
  }
  if x < width {
    grayf32::grayf32_to_luma_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `grayf32_to_luma_u16_row`: clamp→scale→round→u16. 4 pixels/iter.
///
/// # Safety
/// simd128 must be enabled. `y_plane.len() >= width`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf32_to_luma_u16_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = f32x4_splat(65535.0);
  let zero4 = f32x4_splat(0.0);
  let one4 = f32x4_splat(1.0);
  let half = f32x4_splat(0.5);
  let zero16 = i64x2(0, 0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 4));
      let clamped = f32x4_min(f32x4_max(y, zero4), one4);
      let scaled = f32x4_mul(clamped, scale);
      let rounded = i32x4_trunc_sat_f32x4(f32x4_add(scaled, half));
      // Narrow i32x4 → u16x8 via unsigned saturation, then extract lanes.
      let narrow16 = u16x8_narrow_i32x4(rounded, zero16);
      out[x] = u16x8_extract_lane::<0>(narrow16);
      out[x + 1] = u16x8_extract_lane::<1>(narrow16);
      out[x + 2] = u16x8_extract_lane::<2>(narrow16);
      out[x + 3] = u16x8_extract_lane::<3>(narrow16);
      x += 4;
    }
  }
  if x < width {
    grayf32::grayf32_to_luma_u16_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `grayf32_to_luma_f32_row`: identity copy. 4 pixels/iter.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)] // dispatcher always uses scalar; function is exercised by tests only
pub(crate) unsafe fn grayf32_to_luma_f32_row<const BE: bool>(
  y_plane: &[f32],
  out: &mut [f32],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 4));
      v128_store(out.as_mut_ptr().add(x).cast(), y);
      x += 4;
    }
  }
  if x < width {
    grayf32::grayf32_to_luma_f32_row::<BE>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `grayf32_to_hsv_row`: H=0, S=0, V=luma(Y). 4 pixels/iter.
///
/// # Safety
/// simd128 must be enabled. All slices have length >= width.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf32_to_hsv_row<const BE: bool>(
  y_plane: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  let scale = f32x4_splat(255.0);
  let zero4 = f32x4_splat(0.0);
  let one4 = f32x4_splat(1.0);
  let half = f32x4_splat(0.5);
  let zero16 = i64x2(0, 0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = load_endian_u32x4::<BE>(y_plane.as_ptr().cast::<u8>().add(x * 4));
      let clamped = f32x4_min(f32x4_max(y, zero4), one4);
      let scaled = f32x4_mul(clamped, scale);
      let rounded = i32x4_trunc_sat_f32x4(f32x4_add(scaled, half));
      let narrow16 = i16x8_narrow_i32x4(rounded, zero16);
      let narrow8 = u8x16_narrow_i16x8(narrow16, zero16);
      let val = i32x4_extract_lane::<0>(narrow8) as u32;
      let bytes = val.to_le_bytes();
      h_out[x..x + 4].fill(0);
      s_out[x..x + 4].fill(0);
      v_out[x..x + 4].copy_from_slice(&bytes);
      x += 4;
    }
  }
  if x < width {
    grayf32::grayf32_to_hsv_row::<BE>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Ya8 --------------------------------------------------------------------

/// wasm-simd128 `ya8_to_rgb_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya8_to_rgb_row(packed: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  ya8::ya8_to_rgb_row(packed, out, width);
}

/// wasm-simd128 `ya8_to_rgba_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya8_to_rgba_row(packed: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  ya8::ya8_to_rgba_row(packed, out, width);
}

/// wasm-simd128 `ya8_to_rgb_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya8_to_rgb_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  ya8::ya8_to_rgb_u16_row(packed, out, width);
}

/// wasm-simd128 `ya8_to_rgba_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya8_to_rgba_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  ya8::ya8_to_rgba_u16_row(packed, out, width);
}

/// wasm-simd128 `ya8_to_luma_row`: deinterleave Y from [Y,A,...], 8 px/iter.
///
/// Loads 16 packed bytes (8 Ya8 pairs), shuffles even bytes = Y values.
///
/// # Safety
/// simd128 must be enabled. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya8_to_luma_row(packed: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  // Shuffle mask: extract even bytes (Y bytes at indices 0,2,4,6,8,10,12,14).
  let shuf = i8x16(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let src = v128_load(packed.as_ptr().add(x * 2).cast());
      let y8 = i8x16_swizzle(src, shuf);
      let val = i64x2_extract_lane::<0>(y8) as u64;
      out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    ya8::ya8_to_luma_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `ya8_to_luma_u16_row`: deinterleave Y → zero-extend to u16.
/// 8 pixels/iter.
///
/// # Safety
/// simd128 must be enabled. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya8_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let shuf = i8x16(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let src = v128_load(packed.as_ptr().add(x * 2).cast());
      let y8 = i8x16_swizzle(src, shuf);
      // Zero-extend: interleave 8 Y bytes with 8 zero bytes → 8 u16.
      let y16 = u16x8_extend_low_u8x16(y8);
      v128_store(out.as_mut_ptr().add(x).cast(), y16);
      x += 8;
    }
  }
  if x < width {
    ya8::ya8_to_luma_u16_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `ya8_to_hsv_row`: H=0, S=0, V=Y. 8 pixels/iter.
///
/// # Safety
/// simd128 must be enabled. All slices have length >= width.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya8_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  let shuf = i8x16(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let src = v128_load(packed.as_ptr().add(x * 2).cast());
      let y8 = i8x16_swizzle(src, shuf);
      let val = i64x2_extract_lane::<0>(y8) as u64;
      let bytes = val.to_le_bytes();
      h_out[x..x + 8].fill(0);
      s_out[x..x + 8].fill(0);
      v_out[x..x + 8].copy_from_slice(&bytes);
      x += 8;
    }
  }
  if x < width {
    ya8::ya8_to_hsv_row(
      &packed[x * 2..width * 2],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Ya16 -------------------------------------------------------------------

/// Host-endian gate for Ya16 SIMD bodies (`luma_row`, `luma_u16_row`, `hsv_row`).
///
/// The wasm-simd128 Ya16 SIMD bodies use `v128_load` + fixed swizzle masks
/// (`0,1,4,5,8,9,12,13,...`) plus `u16x8_shr` that gather the **host-native**
/// high byte of each Ya16 word. They are only correct when the encoded byte
/// order matches the host. Truth table:
///
/// | data BE | host BE | `BE != HOST_NATIVE_BE` | path   | correct via       |
/// |---------|---------|------------------------|--------|-------------------|
/// | false   | false   | false                  | SIMD   | host-native LE    |
/// | false   | true    | true                   | scalar | `from_le`         |
/// | true    | false   | true                   | scalar | `from_be`         |
/// | true    | true    | false                  | SIMD   | host-native BE    |
///
/// The narrower `if BE { scalar }` gate from `7cb64c6` only covered rows 1+3
/// (LE host); the LE-source-on-BE-host quadrant would still run the SIMD body
/// and corrupt Y/V. The pure-delegate entries (`rgb_row`, `rgba_row`,
/// `rgb_u16_row`, `rgba_u16_row`) don't need the gate because they call into
/// scalar directly.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// wasm-simd128 `ya16_to_rgb_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya16_to_rgb_row<const BE: bool>(packed: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  ya16::ya16_to_rgb_row::<BE>(packed, out, width);
}

/// wasm-simd128 `ya16_to_rgba_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya16_to_rgba_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  ya16::ya16_to_rgba_row::<BE>(packed, out, width);
}

/// wasm-simd128 `ya16_to_rgb_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya16_to_rgb_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  ya16::ya16_to_rgb_u16_row::<BE>(packed, out, width);
}

/// wasm-simd128 `ya16_to_rgba_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya16_to_rgba_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  ya16::ya16_to_rgba_u16_row::<BE>(packed, out, width);
}

/// wasm-simd128 `ya16_to_luma_row`: deinterleave Y u16 → `>> 8` → u8.
/// 8 pixels/iter.
///
/// Loads 8 packed u16 pairs (16 u16 values), extracts even words = Y,
/// then shifts right 8 to get u8.
///
/// # Safety
/// simd128 must be enabled. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya16_to_luma_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  if BE != HOST_NATIVE_BE {
    // Source byte order differs from host → SIMD host-native swizzle wrong; fall through.
    return ya16::ya16_to_luma_row::<BE>(packed, out, width);
  }
  // Shuffle mask: gather words at indices 0,2,4,6 (byte offsets 0-1,4-5,8-9,12-13)
  // into the low 8 bytes. Each word is little-endian so bytes are [lo,hi,...].
  // We want word[0]=bytes[0,1], word[2]=bytes[4,5], word[4]=bytes[8,9], word[6]=bytes[12,13].
  let shuf_lo = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
  let zero16 = i64x2(0, 0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      // Load 8 Ya16 pairs = 16 u16 = two 128-bit vectors.
      let src0 = v128_load(packed.as_ptr().add(x * 2).cast::<v128>());
      let src1 = v128_load(packed.as_ptr().add(x * 2 + 8).cast::<v128>());
      // Extract Y words from each half (every other word starting at index 0).
      let y0 = i8x16_swizzle(src0, shuf_lo); // 4 Y words in low 8 bytes
      let y1 = i8x16_swizzle(src1, shuf_lo); // 4 Y words in low 8 bytes
      // Combine into 8 u16 in one vector.
      let y_words = v128_or(
        y0,
        i8x16_swizzle(
          y1,
          i8x16(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7),
        ),
      );
      // Shift right 8 to get u8 values, narrow to u8.
      let shifted = u16x8_shr(y_words, 8);
      let narrowed = u8x16_narrow_i16x8(shifted, zero16);
      let val = i64x2_extract_lane::<0>(narrowed) as u64;
      out[x..x + 8].copy_from_slice(&val.to_le_bytes());
      x += 8;
    }
  }
  if x < width {
    ya16::ya16_to_luma_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `ya16_to_luma_u16_row`: deinterleave Y u16 passthrough.
/// 8 pixels/iter.
///
/// # Safety
/// simd128 must be enabled. `packed.len() >= width * 2`. `out.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya16_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  if BE != HOST_NATIVE_BE {
    // Source byte order differs from host → SIMD host-native swizzle wrong; fall through.
    return ya16::ya16_to_luma_u16_row::<BE>(packed, out, width);
  }
  let shuf_lo = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let src0 = v128_load(packed.as_ptr().add(x * 2).cast::<v128>());
      let src1 = v128_load(packed.as_ptr().add(x * 2 + 8).cast::<v128>());
      let y0 = i8x16_swizzle(src0, shuf_lo);
      let y1 = i8x16_swizzle(src1, shuf_lo);
      let y_words = v128_or(
        y0,
        i8x16_swizzle(
          y1,
          i8x16(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7),
        ),
      );
      v128_store(out.as_mut_ptr().add(x).cast(), y_words);
      x += 8;
    }
  }
  if x < width {
    ya16::ya16_to_luma_u16_row::<BE>(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// wasm-simd128 `ya16_to_hsv_row`: H=0, S=0, V=`Y>>8`. 8 pixels/iter.
///
/// # Safety
/// simd128 must be enabled. All slices have length >= width.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn ya16_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  if BE != HOST_NATIVE_BE {
    // Source byte order differs from host → SIMD host-native swizzle wrong; fall through.
    return ya16::ya16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
  }
  let shuf_lo = i8x16(0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
  let zero16 = i64x2(0, 0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let src0 = v128_load(packed.as_ptr().add(x * 2).cast::<v128>());
      let src1 = v128_load(packed.as_ptr().add(x * 2 + 8).cast::<v128>());
      let y0 = i8x16_swizzle(src0, shuf_lo);
      let y1 = i8x16_swizzle(src1, shuf_lo);
      let y_words = v128_or(
        y0,
        i8x16_swizzle(
          y1,
          i8x16(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7),
        ),
      );
      let shifted = u16x8_shr(y_words, 8);
      let narrowed = u8x16_narrow_i16x8(shifted, zero16);
      let val = i64x2_extract_lane::<0>(narrowed) as u64;
      let bytes = val.to_le_bytes();
      h_out[x..x + 8].fill(0);
      s_out[x..x + 8].fill(0);
      v_out[x..x + 8].copy_from_slice(&bytes);
      x += 8;
    }
  }
  if x < width {
    ya16::ya16_to_hsv_row::<BE>(
      &packed[x * 2..width * 2],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Grayf16 ----------------------------------------------------------------
//
// wasm-simd128 has no half-precision widening intrinsic, so every Grayf16 row
// delegates to the scalar `grayf16` kernels (which widen each f16 to f32 via
// `half::f16::to_f32`), mirroring the `grayf32` wasm delegation.

/// wasm-simd128 `grayf16_to_rgb_row`: delegates to scalar.
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf16_to_rgb_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  grayf16::grayf16_to_rgb_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf16_to_rgba_row`: delegates to scalar.
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf16_to_rgba_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  grayf16::grayf16_to_rgba_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf16_to_rgb_u16_row`: delegates to scalar.
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf16_to_rgb_u16_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  grayf16::grayf16_to_rgb_u16_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf16_to_rgba_u16_row`: delegates to scalar.
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf16_to_rgba_u16_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  grayf16::grayf16_to_rgba_u16_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf16_to_luma_row`: delegates to scalar.
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf16_to_luma_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  grayf16::grayf16_to_luma_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf16_to_luma_u16_row`: delegates to scalar.
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf16_to_luma_u16_row<const BE: bool>(
  y_plane: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  grayf16::grayf16_to_luma_u16_row::<BE>(y_plane, out, width);
}

/// wasm-simd128 `grayf16_to_hsv_row`: delegates to scalar.
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn grayf16_to_hsv_row<const BE: bool>(
  y_plane: &[half::f16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  grayf16::grayf16_to_hsv_row::<BE>(y_plane, h_out, s_out, v_out, width);
}

// ---- Yaf32 ------------------------------------------------------------------
//
// Packed `[Y, A]` f32 source. The wasm-simd128 backend delegates every Yaf32
// kernel to scalar: the deinterleave-then-clamp/scale offers little SIMD upside
// on simd128, and the sibling `ya16` rgb / rgba paths likewise delegate. Routed
// through these wrappers so the dispatcher keeps a uniform per-backend shape.

/// wasm-simd128 `yaf32_to_rgb_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf32_to_rgb_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  yaf32::yaf32_to_rgb_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf32_to_rgba_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf32_to_rgba_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  yaf32::yaf32_to_rgba_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf32_to_rgb_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf32_to_rgb_u16_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  yaf32::yaf32_to_rgb_u16_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf32_to_rgba_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf32_to_rgba_u16_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  yaf32::yaf32_to_rgba_u16_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf32_to_luma_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf32_to_luma_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  yaf32::yaf32_to_luma_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf32_to_luma_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf32_to_luma_u16_row<const BE: bool>(
  packed: &[f32],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  yaf32::yaf32_to_luma_u16_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf32_to_hsv_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf32_to_hsv_row<const BE: bool>(
  packed: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  yaf32::yaf32_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
}

// ---- Yaf16 ------------------------------------------------------------------
//
// Packed `[Y, A]` f16 source. As with the `grayf16` wasm path (and `yaf32`
// above), every kernel delegates to scalar — the f16 widen has no simd128
// hardware accel here.

/// wasm-simd128 `yaf16_to_rgb_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf16_to_rgb_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  yaf16::yaf16_to_rgb_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf16_to_rgba_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf16_to_rgba_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  yaf16::yaf16_to_rgba_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf16_to_rgb_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf16_to_rgb_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  yaf16::yaf16_to_rgb_u16_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf16_to_rgba_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf16_to_rgba_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  yaf16::yaf16_to_rgba_u16_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf16_to_luma_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf16_to_luma_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  yaf16::yaf16_to_luma_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf16_to_luma_u16_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf16_to_luma_u16_row<const BE: bool>(
  packed: &[half::f16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  yaf16::yaf16_to_luma_u16_row::<BE>(packed, out, width);
}

/// wasm-simd128 `yaf16_to_hsv_row`: delegates to scalar.
///
/// # Safety
/// simd128 must be enabled.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yaf16_to_hsv_row<const BE: bool>(
  packed: &[half::f16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2);
  yaf16::yaf16_to_hsv_row::<BE>(packed, h_out, s_out, v_out, width);
}
