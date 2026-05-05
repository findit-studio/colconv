//! NEON gray → {RGB, RGBA, HSV, luma, luma_u16} kernels.
//!
//! Gray sources are achromatic: every output just broadcasts Y (or shifts
//! it). NEON provides vectorised interleave stores (`vst3q_u8`, `vst4q_u8`)
//! and vectorised shift-and-narrow for the depth-conversion paths.
//!
//! # `full_range` parameter
//!
//! For RGB/RGBA/HSV kernels, `full_range = true` uses the existing fast NEON
//! path. `full_range = false` (limited-range) falls back to scalar since
//! limited-range rescaling is the less-common path and the scalar formulation
//! is simple and correct.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::aarch64::*;

use crate::row::scalar::{bits_mask, gray as scalar};

// ---- helpers -----------------------------------------------------------------

/// Broadcast a u8x16 vector to three-plane interleaved RGB (48 bytes).
///
/// # Safety
/// NEON must be available. `out.len() >= x * 3 + 48` at call site.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_rgb_16x(v: uint8x16_t, out: &mut [u8], x: usize) {
  unsafe {
    let rgb = uint8x16x3_t(v, v, v);
    vst3q_u8(out.as_mut_ptr().add(x * 3), rgb);
  }
}

/// Broadcast a u8x16 vector to four-plane interleaved RGBA (64 bytes), α=0xFF.
///
/// # Safety
/// NEON must be available. `out.len() >= x * 4 + 64` at call site.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_rgba_16x(v: uint8x16_t, out: &mut [u8], x: usize) {
  unsafe {
    let rgba = uint8x16x4_t(v, v, v, vdupq_n_u8(0xFF));
    vst4q_u8(out.as_mut_ptr().add(x * 4), rgba);
  }
}

// ---- Gray8 -------------------------------------------------------------------

/// NEON `gray8_to_rgb_row`: broadcast Y → packed RGB.
///
/// Block size: 16 px / iter (48 bytes written per block via `vst3q_u8`).
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray8_to_rgb_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  if !full_range {
    return scalar::gray8_to_rgb_row(y_plane, out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let v = vld1q_u8(y_plane.as_ptr().add(x));
      store_rgb_16x(v, out, x);
      x += 16;
    }
  }
  if x < width {
    scalar::gray8_to_rgb_row(
      &y_plane[x..width],
      &mut out[x * 3..width * 3],
      width - x,
      true,
    );
  }
}

/// NEON `gray8_to_rgba_row`: broadcast Y → packed RGBA, α=0xFF.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray8_to_rgba_row(
  y_plane: &[u8],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  if !full_range {
    return scalar::gray8_to_rgba_row(y_plane, out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let v = vld1q_u8(y_plane.as_ptr().add(x));
      store_rgba_16x(v, out, x);
      x += 16;
    }
  }
  if x < width {
    scalar::gray8_to_rgba_row(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
      true,
    );
  }
}

/// NEON `gray8_to_hsv_row`: H=0, S=0, V=Y — stores three memset-like planes.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// NEON must be available. All slices `>= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray8_to_hsv_row(
  y_plane: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);
  if !full_range {
    return scalar::gray8_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let zero = vdupq_n_u8(0);
    while x + 16 <= width {
      let v = vld1q_u8(y_plane.as_ptr().add(x));
      vst1q_u8(h_out.as_mut_ptr().add(x), zero);
      vst1q_u8(s_out.as_mut_ptr().add(x), zero);
      vst1q_u8(v_out.as_mut_ptr().add(x), v);
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

// ---- GrayN (const BITS) ------------------------------------------------------

/// NEON `gray_n_to_rgb_row<BITS>`: mask → shift → broadcast → packed RGB u8.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available. Slices sized correctly for `width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_rgb_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  if !full_range {
    return scalar::gray_n_to_rgb_row::<BITS>(y_plane, out, width, full_range);
  }
  let shift = (BITS - 8) as i32;
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = vdupq_n_u16(mask);
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let masked = vandq_u16(raw, mask_v);
      let shifted = vshlq_u16(masked, vdupq_n_s16(-(shift as i16)));
      // narrow u16x8 → u8x8
      let narrow = vmovn_u16(shifted);
      // need 8 pixels → 24 bytes via vst3 on 8-lane
      let rgb8 = uint8x8x3_t(narrow, narrow, narrow);
      vst3_u8(out.as_mut_ptr().add(x * 3), rgb8);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_rgb_row::<BITS>(
      &y_plane[x..width],
      &mut out[x * 3..width * 3],
      width - x,
      true,
    );
  }
}

/// NEON `gray_n_to_rgba_row<BITS>`.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available. Slices sized correctly for `width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_rgba_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  if !full_range {
    return scalar::gray_n_to_rgba_row::<BITS>(y_plane, out, width, full_range);
  }
  let shift = (BITS - 8) as i32;
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = vdupq_n_u16(mask);
    let alpha = vdup_n_u8(0xFF);
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let masked = vandq_u16(raw, mask_v);
      let shifted = vshlq_u16(masked, vdupq_n_s16(-(shift as i16)));
      let narrow = vmovn_u16(shifted);
      let rgba8 = uint8x8x4_t(narrow, narrow, narrow, alpha);
      vst4_u8(out.as_mut_ptr().add(x * 4), rgba8);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_rgba_row::<BITS>(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
      true,
    );
  }
}

/// NEON `gray_n_to_rgb_u16_row<BITS>`: mask → broadcast 3x.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_rgb_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  if !full_range {
    return scalar::gray_n_to_rgb_u16_row::<BITS>(y_plane, out, width, full_range);
  }
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = vdupq_n_u16(mask);
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let y = vandq_u16(raw, mask_v);
      let rgb = uint16x8x3_t(y, y, y);
      vst3q_u16(out.as_mut_ptr().add(x * 3), rgb);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_rgb_u16_row::<BITS>(
      &y_plane[x..width],
      &mut out[x * 3..width * 3],
      width - x,
      true,
    );
  }
}

/// NEON `gray_n_to_rgba_u16_row<BITS>`: mask → broadcast 3x + α = bits_mask.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_rgba_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  if !full_range {
    return scalar::gray_n_to_rgba_u16_row::<BITS>(y_plane, out, width, full_range);
  }
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = vdupq_n_u16(mask);
    let alpha_v = vdupq_n_u16(mask); // full-range max for BITS
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let y = vandq_u16(raw, mask_v);
      let rgba = uint16x8x4_t(y, y, y, alpha_v);
      vst4q_u16(out.as_mut_ptr().add(x * 4), rgba);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_rgba_u16_row::<BITS>(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
      true,
    );
  }
}

/// NEON `gray_n_to_luma_row<BITS>`: mask → shift → u8.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_luma_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let shift = (BITS - 8) as i32;
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = vdupq_n_u16(mask);
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let masked = vandq_u16(raw, mask_v);
      let shifted = vshlq_u16(masked, vdupq_n_s16(-(shift as i16)));
      let narrow = vmovn_u16(shifted);
      vst1_u8(out.as_mut_ptr().add(x), narrow);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_row::<BITS>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// NEON `gray_n_to_luma_u16_row<BITS>`: mask only.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_luma_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = vdupq_n_u16(mask);
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let masked = vandq_u16(raw, mask_v);
      vst1q_u16(out.as_mut_ptr().add(x), masked);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_luma_u16_row::<BITS>(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// NEON `gray_n_to_hsv_row<BITS>`: H=0, S=0, V=Y8.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_hsv_row<const BITS: u32>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);
  if !full_range {
    return scalar::gray_n_to_hsv_row::<BITS>(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let shift = (BITS - 8) as i32;
  let mask = bits_mask::<BITS>();
  let mut x = 0usize;
  unsafe {
    let mask_v = vdupq_n_u16(mask);
    let zero = vdup_n_u8(0);
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let masked = vandq_u16(raw, mask_v);
      let shifted = vshlq_u16(masked, vdupq_n_s16(-(shift as i16)));
      let narrow = vmovn_u16(shifted);
      vst1_u8(h_out.as_mut_ptr().add(x), zero);
      vst1_u8(s_out.as_mut_ptr().add(x), zero);
      vst1_u8(v_out.as_mut_ptr().add(x), narrow);
      x += 8;
    }
  }
  if x < width {
    scalar::gray_n_to_hsv_row::<BITS>(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- Gray16 ------------------------------------------------------------------

/// NEON `gray16_to_rgb_row`: `>> 8` → broadcast → packed RGB u8.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_rgb_row(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  if !full_range {
    return scalar::gray16_to_rgb_row(y_plane, out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let y8 = vshrn_n_u16::<8>(raw);
      let rgb = uint8x8x3_t(y8, y8, y8);
      vst3_u8(out.as_mut_ptr().add(x * 3), rgb);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_rgb_row(
      &y_plane[x..width],
      &mut out[x * 3..width * 3],
      width - x,
      true,
    );
  }
}

/// NEON `gray16_to_rgba_row`: `>> 8` → broadcast → packed RGBA u8, α=0xFF.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_rgba_row(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  if !full_range {
    return scalar::gray16_to_rgba_row(y_plane, out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let alpha = vdup_n_u8(0xFF);
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let y8 = vshrn_n_u16::<8>(raw);
      let rgba = uint8x8x4_t(y8, y8, y8, alpha);
      vst4_u8(out.as_mut_ptr().add(x * 4), rgba);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_rgba_row(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
      true,
    );
  }
}

/// NEON `gray16_to_rgb_u16_row`: identity broadcast × 3.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_rgb_u16_row(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  if !full_range {
    return scalar::gray16_to_rgb_u16_row(y_plane, out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = vld1q_u16(y_plane.as_ptr().add(x));
      let rgb = uint16x8x3_t(y, y, y);
      vst3q_u16(out.as_mut_ptr().add(x * 3), rgb);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_rgb_u16_row(
      &y_plane[x..width],
      &mut out[x * 3..width * 3],
      width - x,
      true,
    );
  }
}

/// NEON `gray16_to_rgba_u16_row`: identity broadcast × 3 + α=0xFFFF.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling).
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_rgba_u16_row(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  if !full_range {
    return scalar::gray16_to_rgba_u16_row(y_plane, out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let alpha = vdupq_n_u16(0xFFFF);
    while x + 8 <= width {
      let y = vld1q_u16(y_plane.as_ptr().add(x));
      let rgba = uint16x8x4_t(y, y, y, alpha);
      vst4q_u16(out.as_mut_ptr().add(x * 4), rgba);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_rgba_u16_row(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
      true,
    );
  }
}

/// NEON `gray16_to_luma_row`: `>> 8` → u8.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_luma_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let y8 = vshrn_n_u16::<8>(raw);
      vst1_u8(out.as_mut_ptr().add(x), y8);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_luma_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// NEON `gray16_to_luma_u16_row`: identity copy via NEON loads/stores.
///
/// Luma outputs always pass Y through without `full_range` rescaling.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_luma_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y = vld1q_u16(y_plane.as_ptr().add(x));
      vst1q_u16(out.as_mut_ptr().add(x), y);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_luma_u16_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// NEON `gray16_to_hsv_row`: `>> 8` → H=0, S=0, V=Y8.
///
/// For `full_range = false`, falls back to scalar (limited-range rescaling
/// applied to V channel).
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_hsv_row(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);
  if !full_range {
    return scalar::gray16_to_hsv_row(y_plane, h_out, s_out, v_out, width, full_range);
  }
  let mut x = 0usize;
  unsafe {
    let zero = vdup_n_u8(0);
    while x + 8 <= width {
      let raw = vld1q_u16(y_plane.as_ptr().add(x));
      let y8 = vshrn_n_u16::<8>(raw);
      vst1_u8(h_out.as_mut_ptr().add(x), zero);
      vst1_u8(s_out.as_mut_ptr().add(x), zero);
      vst1_u8(v_out.as_mut_ptr().add(x), y8);
      x += 8;
    }
  }
  if x < width {
    scalar::gray16_to_hsv_row(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
      true,
    );
  }
}

// ---- Grayf32 -----------------------------------------------------------------

/// NEON `grayf32_to_rgb_row`: clamp [0,1] × 255 → u8, broadcast Y → R=G=B.
///
/// Block size 8 px (two float32x4_t loads, vst3_u8 interleave store).
///
/// # Safety
/// NEON must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn grayf32_to_rgb_row(y_plane: &[f32], out: &mut [u8], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = vdupq_n_f32(255.0);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y0 = vld1q_f32(y_plane.as_ptr().add(x));
      let y1 = vld1q_f32(y_plane.as_ptr().add(x + 4));
      let c0 = vmulq_f32(vmaxq_f32(vminq_f32(y0, one), zero), scale);
      let c1 = vmulq_f32(vmaxq_f32(vminq_f32(y1, one), zero), scale);
      // vcvtaq_u32_f32: round-to-nearest-even, no FPCR manipulation needed.
      let u0 = vcvtaq_u32_f32(c0);
      let u1 = vcvtaq_u32_f32(c1);
      let n0 = vmovn_u32(u0); // u32x4 → u16x4
      let n1 = vmovn_u32(u1);
      let n8 = vmovn_u16(vcombine_u16(n0, n1)); // u16x8 → u8x8
      let rgb = uint8x8x3_t(n8, n8, n8);
      vst3_u8(out.as_mut_ptr().add(x * 3), rgb);
      x += 8;
    }
  }
  if x < width {
    scalar::grayf32_to_rgb_row(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// NEON `grayf32_to_rgba_row`: clamp [0,1] × 255 → u8, broadcast Y → R=G=B, α=0xFF.
///
/// # Safety
/// NEON must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn grayf32_to_rgba_row(y_plane: &[f32], out: &mut [u8], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = vdupq_n_f32(255.0);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let alpha = vdup_n_u8(0xFF);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y0 = vld1q_f32(y_plane.as_ptr().add(x));
      let y1 = vld1q_f32(y_plane.as_ptr().add(x + 4));
      let c0 = vmulq_f32(vmaxq_f32(vminq_f32(y0, one), zero), scale);
      let c1 = vmulq_f32(vmaxq_f32(vminq_f32(y1, one), zero), scale);
      let u0 = vcvtaq_u32_f32(c0);
      let u1 = vcvtaq_u32_f32(c1);
      let n8 = vmovn_u16(vcombine_u16(vmovn_u32(u0), vmovn_u32(u1)));
      let rgba = uint8x8x4_t(n8, n8, n8, alpha);
      vst4_u8(out.as_mut_ptr().add(x * 4), rgba);
      x += 8;
    }
  }
  if x < width {
    scalar::grayf32_to_rgba_row(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// NEON `grayf32_to_rgb_u16_row`: clamp [0,1] × 65535 → u16, broadcast Y → R=G=B.
///
/// # Safety
/// NEON must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn grayf32_to_rgb_u16_row(y_plane: &[f32], out: &mut [u16], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let scale = vdupq_n_f32(65535.0);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = vld1q_f32(y_plane.as_ptr().add(x));
      let c = vmulq_f32(vmaxq_f32(vminq_f32(y, one), zero), scale);
      let u32v = vcvtaq_u32_f32(c);
      let u16v = vqmovn_u32(u32v); // saturating narrow to u16
      let rgb = uint16x4x3_t(u16v, u16v, u16v);
      vst3_u16(out.as_mut_ptr().add(x * 3), rgb);
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_rgb_u16_row(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// NEON `grayf32_to_rgba_u16_row`: clamp [0,1] × 65535 → u16, broadcast + α=0xFFFF.
///
/// # Safety
/// NEON must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn grayf32_to_rgba_u16_row(y_plane: &[f32], out: &mut [u16], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let scale = vdupq_n_f32(65535.0);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let alpha = vdup_n_u16(0xFFFF);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = vld1q_f32(y_plane.as_ptr().add(x));
      let c = vmulq_f32(vmaxq_f32(vminq_f32(y, one), zero), scale);
      let u16v = vqmovn_u32(vcvtaq_u32_f32(c));
      let rgba = uint16x4x4_t(u16v, u16v, u16v, alpha);
      vst4_u16(out.as_mut_ptr().add(x * 4), rgba);
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_rgba_u16_row(
      &y_plane[x..width],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// NEON `grayf32_to_rgb_f32_row`: lossless replicate Y → R=G=B (no clamp).
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn grayf32_to_rgb_f32_row(y_plane: &[f32], out: &mut [f32], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = vld1q_f32(y_plane.as_ptr().add(x));
      let rgb = float32x4x3_t(y, y, y);
      vst3q_f32(out.as_mut_ptr().add(x * 3), rgb);
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_rgb_f32_row(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// NEON `grayf32_to_luma_row`: clamp [0,1] × 255 → u8 luma.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn grayf32_to_luma_row(y_plane: &[f32], out: &mut [u8], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = vdupq_n_f32(255.0);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y0 = vld1q_f32(y_plane.as_ptr().add(x));
      let y1 = vld1q_f32(y_plane.as_ptr().add(x + 4));
      let c0 = vmulq_f32(vmaxq_f32(vminq_f32(y0, one), zero), scale);
      let c1 = vmulq_f32(vmaxq_f32(vminq_f32(y1, one), zero), scale);
      let n8 = vmovn_u16(vcombine_u16(vmovn_u32(vcvtaq_u32_f32(c0)), vmovn_u32(vcvtaq_u32_f32(c1))));
      vst1_u8(out.as_mut_ptr().add(x), n8);
      x += 8;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// NEON `grayf32_to_luma_u16_row`: clamp [0,1] × 65535 → u16 luma.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn grayf32_to_luma_u16_row(y_plane: &[f32], out: &mut [u16], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let scale = vdupq_n_f32(65535.0);
  let zero = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = vld1q_f32(y_plane.as_ptr().add(x));
      let c = vmulq_f32(vmaxq_f32(vminq_f32(y, one), zero), scale);
      let u16v = vqmovn_u32(vcvtaq_u32_f32(c));
      vst1_u16(out.as_mut_ptr().add(x), u16v);
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_u16_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// NEON `grayf32_to_luma_f32_row`: memcpy pass-through.
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)] // dispatcher uses scalar directly for lossless f32 paths
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn grayf32_to_luma_f32_row(y_plane: &[f32], out: &mut [f32], width: usize) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 4 <= width {
      let y = vld1q_f32(y_plane.as_ptr().add(x));
      vst1q_f32(out.as_mut_ptr().add(x), y);
      x += 4;
    }
  }
  if x < width {
    scalar::grayf32_to_luma_f32_row(&y_plane[x..width], &mut out[x..width], width - x);
  }
}

/// NEON `grayf32_to_hsv_row`: H=0, S=0, V = clamp(Y,0,1)×255. Gray fast-path.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn grayf32_to_hsv_row(
  y_plane: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::grayf32 as scalar;
  debug_assert!(y_plane.len() >= width);
  let scale = vdupq_n_f32(255.0);
  let zero_f = vdupq_n_f32(0.0);
  let one = vdupq_n_f32(1.0);
  let zero_u8 = vdup_n_u8(0);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let y0 = vld1q_f32(y_plane.as_ptr().add(x));
      let y1 = vld1q_f32(y_plane.as_ptr().add(x + 4));
      let c0 = vmulq_f32(vmaxq_f32(vminq_f32(y0, one), zero_f), scale);
      let c1 = vmulq_f32(vmaxq_f32(vminq_f32(y1, one), zero_f), scale);
      let v8 = vmovn_u16(vcombine_u16(vmovn_u32(vcvtaq_u32_f32(c0)), vmovn_u32(vcvtaq_u32_f32(c1))));
      vst1_u8(h_out.as_mut_ptr().add(x), zero_u8);
      vst1_u8(s_out.as_mut_ptr().add(x), zero_u8);
      vst1_u8(v_out.as_mut_ptr().add(x), v8);
      x += 8;
    }
  }
  if x < width {
    scalar::grayf32_to_hsv_row(
      &y_plane[x..width],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Ya8 ---------------------------------------------------------------------

/// NEON `ya8_to_rgb_row`: deinterleave `[Y,A]` pairs, broadcast Y → R=G=B.
///
/// Block size: 8 px / iter (vld2_u8 deinterleaves 16 bytes = 8 Ya8 pixels).
///
/// # Safety
/// NEON must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya8_to_rgb_row(packed: &[u8], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2_u8(packed.as_ptr().add(x * 2));
      let y = ya.0; // uint8x8_t Y values
      let rgb = uint8x8x3_t(y, y, y);
      vst3_u8(out.as_mut_ptr().add(x * 3), rgb);
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_rgb_row(&packed[x * 2..width * 2], &mut out[x * 3..width * 3], width - x);
  }
}

/// NEON `ya8_to_rgba_row`: deinterleave `[Y,A]`, broadcast Y → R=G=B, pass α.
///
/// # Safety
/// NEON must be available. `packed.len() >= width * 2`. `out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya8_to_rgba_row(packed: &[u8], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2_u8(packed.as_ptr().add(x * 2));
      let y = ya.0;
      let a = ya.1;
      let rgba = uint8x8x4_t(y, y, y, a);
      vst4_u8(out.as_mut_ptr().add(x * 4), rgba);
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_rgba_row(&packed[x * 2..width * 2], &mut out[x * 4..width * 4], width - x);
  }
}

/// NEON `ya8_to_rgb_u16_row`: zero-extend Y → u16, broadcast R=G=B.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya8_to_rgb_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2_u8(packed.as_ptr().add(x * 2));
      let y8 = ya.0;
      // zero-extend u8x8 → u16x8
      let y16 = vmovl_u8(y8);
      // broadcast to 3 channels in two 4-px halves
      let ylo = vget_low_u16(y16);
      let yhi = vget_high_u16(y16);
      let rgb_lo = uint16x4x3_t(ylo, ylo, ylo);
      let rgb_hi = uint16x4x3_t(yhi, yhi, yhi);
      vst3_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
      vst3_u16(out.as_mut_ptr().add((x + 4) * 3), rgb_hi);
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_rgb_u16_row(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// NEON `ya8_to_rgba_u16_row`: zero-extend Y and A → u16, broadcast Y to R=G=B.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya8_to_rgba_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2_u8(packed.as_ptr().add(x * 2));
      let y16 = vmovl_u8(ya.0);
      let a16 = vmovl_u8(ya.1);
      let ylo = vget_low_u16(y16);
      let yhi = vget_high_u16(y16);
      let alo = vget_low_u16(a16);
      let ahi = vget_high_u16(a16);
      let rgba_lo = uint16x4x4_t(ylo, ylo, ylo, alo);
      let rgba_hi = uint16x4x4_t(yhi, yhi, yhi, ahi);
      vst4_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
      vst4_u16(out.as_mut_ptr().add((x + 4) * 4), rgba_hi);
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_rgba_u16_row(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// NEON `ya8_to_luma_row`: extract Y bytes (`out[x] = packed[2*x]`).
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya8_to_luma_row(packed: &[u8], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2_u8(packed.as_ptr().add(x * 2));
      vst1_u8(out.as_mut_ptr().add(x), ya.0);
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_luma_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// NEON `ya8_to_luma_u16_row`: zero-extend Y → u16.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya8_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2_u8(packed.as_ptr().add(x * 2));
      let y16 = vmovl_u8(ya.0);
      vst1q_u16(out.as_mut_ptr().add(x), y16);
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_luma_u16_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// NEON `ya8_to_hsv_row`: H=0, S=0, V=Y. α dropped.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya8_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::ya8 as scalar;
  debug_assert!(packed.len() >= width * 2);
  let mut x = 0usize;
  unsafe {
    let zero = vdup_n_u8(0);
    while x + 8 <= width {
      let ya = vld2_u8(packed.as_ptr().add(x * 2));
      vst1_u8(h_out.as_mut_ptr().add(x), zero);
      vst1_u8(s_out.as_mut_ptr().add(x), zero);
      vst1_u8(v_out.as_mut_ptr().add(x), ya.0);
      x += 8;
    }
  }
  if x < width {
    scalar::ya8_to_hsv_row(
      &packed[x * 2..width * 2],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

// ---- Ya16 --------------------------------------------------------------------

/// NEON `ya16_to_rgb_row`: deinterleave `[Y,A]` u16 pairs, Y `>> 8` → u8, broadcast.
///
/// Block size: 8 px / iter (vld2q_u16).
///
/// # Safety
/// NEON must be available. `packed.len() >= width * 2`. `out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya16_to_rgb_row(packed: &[u16], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2q_u16(packed.as_ptr().add(x * 2));
      let y8 = vshrn_n_u16::<8>(ya.0); // u16x8 → u8x8 via high byte
      let rgb = uint8x8x3_t(y8, y8, y8);
      vst3_u8(out.as_mut_ptr().add(x * 3), rgb);
      x += 8;
    }
  }
  if x < width {
    scalar::ya16_to_rgb_row(&packed[x * 2..width * 2], &mut out[x * 3..width * 3], width - x);
  }
}

/// NEON `ya16_to_rgba_row`: Y `>> 8`, A `>> 8`, broadcast Y → R=G=B.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya16_to_rgba_row(packed: &[u16], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2q_u16(packed.as_ptr().add(x * 2));
      let y8 = vshrn_n_u16::<8>(ya.0);
      let a8 = vshrn_n_u16::<8>(ya.1);
      let rgba = uint8x8x4_t(y8, y8, y8, a8);
      vst4_u8(out.as_mut_ptr().add(x * 4), rgba);
      x += 8;
    }
  }
  if x < width {
    scalar::ya16_to_rgba_row(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// NEON `ya16_to_rgb_u16_row`: native Y u16, broadcast R=G=B.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya16_to_rgb_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2q_u16(packed.as_ptr().add(x * 2));
      let ylo = vget_low_u16(ya.0);
      let yhi = vget_high_u16(ya.0);
      let rgb_lo = uint16x4x3_t(ylo, ylo, ylo);
      let rgb_hi = uint16x4x3_t(yhi, yhi, yhi);
      vst3_u16(out.as_mut_ptr().add(x * 3), rgb_lo);
      vst3_u16(out.as_mut_ptr().add((x + 4) * 3), rgb_hi);
      x += 8;
    }
  }
  if x < width {
    scalar::ya16_to_rgb_u16_row(
      &packed[x * 2..width * 2],
      &mut out[x * 3..width * 3],
      width - x,
    );
  }
}

/// NEON `ya16_to_rgba_u16_row`: native Y and A u16, broadcast Y to R=G=B.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya16_to_rgba_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2q_u16(packed.as_ptr().add(x * 2));
      let ylo = vget_low_u16(ya.0);
      let yhi = vget_high_u16(ya.0);
      let alo = vget_low_u16(ya.1);
      let ahi = vget_high_u16(ya.1);
      let rgba_lo = uint16x4x4_t(ylo, ylo, ylo, alo);
      let rgba_hi = uint16x4x4_t(yhi, yhi, yhi, ahi);
      vst4_u16(out.as_mut_ptr().add(x * 4), rgba_lo);
      vst4_u16(out.as_mut_ptr().add((x + 4) * 4), rgba_hi);
      x += 8;
    }
  }
  if x < width {
    scalar::ya16_to_rgba_u16_row(
      &packed[x * 2..width * 2],
      &mut out[x * 4..width * 4],
      width - x,
    );
  }
}

/// NEON `ya16_to_luma_row`: Y `>> 8` → u8.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya16_to_luma_row(packed: &[u16], out: &mut [u8], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2q_u16(packed.as_ptr().add(x * 2));
      let y8 = vshrn_n_u16::<8>(ya.0);
      vst1_u8(out.as_mut_ptr().add(x), y8);
      x += 8;
    }
  }
  if x < width {
    scalar::ya16_to_luma_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// NEON `ya16_to_luma_u16_row`: native Y pass-through.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya16_to_luma_u16_row(packed: &[u16], out: &mut [u16], width: usize) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  debug_assert!(out.len() >= width);
  let mut x = 0usize;
  unsafe {
    while x + 8 <= width {
      let ya = vld2q_u16(packed.as_ptr().add(x * 2));
      vst1q_u16(out.as_mut_ptr().add(x), ya.0);
      x += 8;
    }
  }
  if x < width {
    scalar::ya16_to_luma_u16_row(&packed[x * 2..width * 2], &mut out[x..width], width - x);
  }
}

/// NEON `ya16_to_hsv_row`: H=0, S=0, V = Y `>> 8`. α dropped.
///
/// # Safety
/// NEON must be available.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn ya16_to_hsv_row(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  use crate::row::scalar::ya16 as scalar;
  debug_assert!(packed.len() >= width * 2);
  let mut x = 0usize;
  unsafe {
    let zero = vdup_n_u8(0);
    while x + 8 <= width {
      let ya = vld2q_u16(packed.as_ptr().add(x * 2));
      let y8 = vshrn_n_u16::<8>(ya.0);
      vst1_u8(h_out.as_mut_ptr().add(x), zero);
      vst1_u8(s_out.as_mut_ptr().add(x), zero);
      vst1_u8(v_out.as_mut_ptr().add(x), y8);
      x += 8;
    }
  }
  if x < width {
    scalar::ya16_to_hsv_row(
      &packed[x * 2..width * 2],
      &mut h_out[x..width],
      &mut s_out[x..width],
      &mut v_out[x..width],
      width - x,
    );
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use crate::row::scalar::gray as scalar;

  const WIDTHS: &[usize] = &[1, 7, 8, 16, 17, 32, 33, 64, 128, 130];

  fn prng(out: &mut [u8], seed: u32) {
    let mut s = seed;
    for v in out.iter_mut() {
      s = s.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (s >> 16) as u8;
    }
  }
  fn prng16(out: &mut [u16], seed: u32) {
    let mut buf = std::vec![0u8; out.len() * 2];
    prng(&mut buf, seed);
    for (i, o) in out.iter_mut().enumerate() {
      *o = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_gray8_to_rgb_matches_scalar() {
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      prng(&mut plane, 0xABCD);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::gray8_to_rgb_row(&plane, &mut simd, w, true) };
      scalar::gray8_to_rgb_row(&plane, &mut scal, w, true);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_gray8_to_rgba_matches_scalar() {
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      prng(&mut plane, 0x1234);
      let mut simd = std::vec![0u8; w * 4];
      let mut scal = std::vec![0u8; w * 4];
      unsafe { super::gray8_to_rgba_row(&plane, &mut simd, w, true) };
      scalar::gray8_to_rgba_row(&plane, &mut scal, w, true);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_gray8_to_hsv_matches_scalar() {
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      prng(&mut plane, 0x5678);
      let mut sh = std::vec![0u8; w];
      let mut ss = std::vec![0u8; w];
      let mut sv = std::vec![0u8; w];
      let mut rh = std::vec![0u8; w];
      let mut rs = std::vec![0u8; w];
      let mut rv = std::vec![0u8; w];
      unsafe { super::gray8_to_hsv_row(&plane, &mut sh, &mut ss, &mut sv, w, true) };
      scalar::gray8_to_hsv_row(&plane, &mut rh, &mut rs, &mut rv, w, true);
      assert_eq!(sh, rh, "H width={w}");
      assert_eq!(ss, rs, "S width={w}");
      assert_eq!(sv, rv, "V width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_gray_n_to_rgb_10bit_matches_scalar() {
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0xABCD_1234);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::gray_n_to_rgb_row::<10>(&plane, &mut simd, w, true) };
      scalar::gray_n_to_rgb_row::<10>(&plane, &mut scal, w, true);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_gray16_to_rgb_matches_scalar() {
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0xDEAD_BEEF);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::gray16_to_rgb_row(&plane, &mut simd, w, true) };
      scalar::gray16_to_rgb_row(&plane, &mut scal, w, true);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  // ---- limited-range SIMD/scalar parity tests ----

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_gray8_limited_range_matches_scalar() {
    for &w in WIDTHS {
      let mut plane = std::vec![0u8; w];
      prng(&mut plane, 0xCAFE_BABEu32);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::gray8_to_rgb_row(&plane, &mut simd, w, false) };
      scalar::gray8_to_rgb_row(&plane, &mut scal, w, false);
      assert_eq!(simd, scal, "width={w} limited-range");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_gray16_limited_range_matches_scalar() {
    for &w in WIDTHS {
      let mut plane = std::vec![0u16; w];
      prng16(&mut plane, 0x1234_5678);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::gray16_to_rgb_row(&plane, &mut simd, w, false) };
      scalar::gray16_to_rgb_row(&plane, &mut scal, w, false);
      assert_eq!(simd, scal, "width={w} limited-range");
    }
  }

  // ---- Grayf32 parity tests ---------------------------------------------------

  fn prng_f32(out: &mut [f32], seed: u32) {
    let mut s = seed;
    for v in out.iter_mut() {
      s = s.wrapping_mul(1664525).wrapping_add(1013904223);
      // Values in [-0.1, 1.2] to exercise clamping.
      *v = ((s >> 8) as f32) / (u32::MAX as f32) * 1.3 - 0.1;
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_grayf32_to_rgb_matches_scalar() {
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF32A_0001);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::grayf32_to_rgb_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgb_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_grayf32_to_rgba_matches_scalar() {
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF32A_0002);
      let mut simd = std::vec![0u8; w * 4];
      let mut scal = std::vec![0u8; w * 4];
      unsafe { super::grayf32_to_rgba_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgba_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_grayf32_to_rgb_u16_matches_scalar() {
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF32A_0003);
      let mut simd = std::vec![0u16; w * 3];
      let mut scal = std::vec![0u16; w * 3];
      unsafe { super::grayf32_to_rgb_u16_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgb_u16_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_grayf32_to_rgba_u16_matches_scalar() {
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF32A_0004);
      let mut simd = std::vec![0u16; w * 4];
      let mut scal = std::vec![0u16; w * 4];
      unsafe { super::grayf32_to_rgba_u16_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgba_u16_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_grayf32_to_rgb_f32_matches_scalar() {
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF32A_0005);
      let mut simd = std::vec![0.0f32; w * 3];
      let mut scal = std::vec![0.0f32; w * 3];
      unsafe { super::grayf32_to_rgb_f32_row(&plane, &mut simd, w) };
      sf::grayf32_to_rgb_f32_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_grayf32_to_luma_matches_scalar() {
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF32A_0006);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::grayf32_to_luma_row(&plane, &mut simd, w) };
      sf::grayf32_to_luma_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_grayf32_to_luma_u16_matches_scalar() {
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF32A_0007);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::grayf32_to_luma_u16_row(&plane, &mut simd, w) };
      sf::grayf32_to_luma_u16_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_grayf32_to_luma_f32_matches_scalar() {
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF32A_0008);
      let mut simd = std::vec![0.0f32; w];
      let mut scal = std::vec![0.0f32; w];
      unsafe { super::grayf32_to_luma_f32_row(&plane, &mut simd, w) };
      sf::grayf32_to_luma_f32_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_grayf32_to_hsv_matches_scalar() {
    use crate::row::scalar::grayf32 as sf;
    for &w in WIDTHS {
      let mut plane = std::vec![0.0f32; w];
      prng_f32(&mut plane, 0xF32A_0009);
      let mut sh = std::vec![0u8; w];
      let mut ss = std::vec![0u8; w];
      let mut sv = std::vec![0u8; w];
      let mut rh = std::vec![0u8; w];
      let mut rs = std::vec![0u8; w];
      let mut rv = std::vec![0u8; w];
      unsafe { super::grayf32_to_hsv_row(&plane, &mut sh, &mut ss, &mut sv, w) };
      sf::grayf32_to_hsv_row(&plane, &mut rh, &mut rs, &mut rv, w);
      assert_eq!(sh, rh, "H width={w}");
      assert_eq!(ss, rs, "S width={w}");
      assert_eq!(sv, rv, "V width={w}");
    }
  }

  // ---- Ya8 parity tests -------------------------------------------------------

  fn prng_ya8(out: &mut [u8], seed: u32) {
    let mut s = seed;
    for v in out.iter_mut() {
      s = s.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (s >> 16) as u8;
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya8_to_rgb_matches_scalar() {
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA800_0001);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::ya8_to_rgb_row(&packed, &mut simd, w) };
      sy::ya8_to_rgb_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya8_to_rgba_matches_scalar() {
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA800_0002);
      let mut simd = std::vec![0u8; w * 4];
      let mut scal = std::vec![0u8; w * 4];
      unsafe { super::ya8_to_rgba_row(&packed, &mut simd, w) };
      sy::ya8_to_rgba_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya8_to_rgb_u16_matches_scalar() {
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA800_0003);
      let mut simd = std::vec![0u16; w * 3];
      let mut scal = std::vec![0u16; w * 3];
      unsafe { super::ya8_to_rgb_u16_row(&packed, &mut simd, w) };
      sy::ya8_to_rgb_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya8_to_rgba_u16_matches_scalar() {
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA800_0004);
      let mut simd = std::vec![0u16; w * 4];
      let mut scal = std::vec![0u16; w * 4];
      unsafe { super::ya8_to_rgba_u16_row(&packed, &mut simd, w) };
      sy::ya8_to_rgba_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya8_to_luma_matches_scalar() {
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA800_0005);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::ya8_to_luma_row(&packed, &mut simd, w) };
      sy::ya8_to_luma_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya8_to_luma_u16_matches_scalar() {
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA800_0006);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::ya8_to_luma_u16_row(&packed, &mut simd, w) };
      sy::ya8_to_luma_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya8_to_hsv_matches_scalar() {
    use crate::row::scalar::ya8 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 2];
      prng_ya8(&mut packed, 0xA800_0007);
      let mut sh = std::vec![0u8; w];
      let mut ss = std::vec![0u8; w];
      let mut sv = std::vec![0u8; w];
      let mut rh = std::vec![0u8; w];
      let mut rs = std::vec![0u8; w];
      let mut rv = std::vec![0u8; w];
      unsafe { super::ya8_to_hsv_row(&packed, &mut sh, &mut ss, &mut sv, w) };
      sy::ya8_to_hsv_row(&packed, &mut rh, &mut rs, &mut rv, w);
      assert_eq!(sh, rh, "H width={w}");
      assert_eq!(ss, rs, "S width={w}");
      assert_eq!(sv, rv, "V width={w}");
    }
  }

  // ---- Ya16 parity tests ------------------------------------------------------

  fn prng_ya16(out: &mut [u16], seed: u32) {
    let mut buf = std::vec![0u8; out.len() * 2];
    prng(&mut buf, seed);
    for (i, o) in out.iter_mut().enumerate() {
      *o = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya16_to_rgb_matches_scalar() {
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0001);
      let mut simd = std::vec![0u8; w * 3];
      let mut scal = std::vec![0u8; w * 3];
      unsafe { super::ya16_to_rgb_row(&packed, &mut simd, w) };
      sy::ya16_to_rgb_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya16_to_rgba_matches_scalar() {
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0002);
      let mut simd = std::vec![0u8; w * 4];
      let mut scal = std::vec![0u8; w * 4];
      unsafe { super::ya16_to_rgba_row(&packed, &mut simd, w) };
      sy::ya16_to_rgba_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya16_to_rgb_u16_matches_scalar() {
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0003);
      let mut simd = std::vec![0u16; w * 3];
      let mut scal = std::vec![0u16; w * 3];
      unsafe { super::ya16_to_rgb_u16_row(&packed, &mut simd, w) };
      sy::ya16_to_rgb_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya16_to_rgba_u16_matches_scalar() {
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0004);
      let mut simd = std::vec![0u16; w * 4];
      let mut scal = std::vec![0u16; w * 4];
      unsafe { super::ya16_to_rgba_u16_row(&packed, &mut simd, w) };
      sy::ya16_to_rgba_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya16_to_luma_matches_scalar() {
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0005);
      let mut simd = std::vec![0u8; w];
      let mut scal = std::vec![0u8; w];
      unsafe { super::ya16_to_luma_row(&packed, &mut simd, w) };
      sy::ya16_to_luma_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya16_to_luma_u16_matches_scalar() {
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0006);
      let mut simd = std::vec![0u16; w];
      let mut scal = std::vec![0u16; w];
      unsafe { super::ya16_to_luma_u16_row(&packed, &mut simd, w) };
      sy::ya16_to_luma_u16_row(&packed, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn neon_ya16_to_hsv_matches_scalar() {
    use crate::row::scalar::ya16 as sy;
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 2];
      prng_ya16(&mut packed, 0xA160_0007);
      let mut sh = std::vec![0u8; w];
      let mut ss = std::vec![0u8; w];
      let mut sv = std::vec![0u8; w];
      let mut rh = std::vec![0u8; w];
      let mut rs = std::vec![0u8; w];
      let mut rv = std::vec![0u8; w];
      unsafe { super::ya16_to_hsv_row(&packed, &mut sh, &mut ss, &mut sv, w) };
      sy::ya16_to_hsv_row(&packed, &mut rh, &mut rs, &mut rv, w);
      assert_eq!(sh, rh, "H width={w}");
      assert_eq!(ss, rs, "S width={w}");
      assert_eq!(sv, rv, "V width={w}");
    }
  }
}
