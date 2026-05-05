//! NEON gray → {RGB, RGBA, HSV, luma, luma_u16} kernels.
//!
//! Gray sources are achromatic: every output just broadcasts Y (or shifts
//! it). NEON provides vectorised interleave stores (`vst3q_u8`, `vst4q_u8`)
//! and vectorised shift-and-narrow for the depth-conversion paths.

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
///
/// # Safety
/// NEON must be available. `y_plane.len() >= width`. `out.len() >= width * 3`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray8_to_rgb_row(y_plane: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let v = vld1q_u8(y_plane.as_ptr().add(x));
      store_rgb_16x(v, out, x);
      x += 16;
    }
  }
  if x < width {
    scalar::gray8_to_rgb_row(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// NEON `gray8_to_rgba_row`: broadcast Y → packed RGBA, α=0xFF.
///
/// # Safety
/// NEON must be available. `y_plane.len() >= width`. `out.len() >= width * 4`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray8_to_rgba_row(y_plane: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
  let mut x = 0usize;
  unsafe {
    while x + 16 <= width {
      let v = vld1q_u8(y_plane.as_ptr().add(x));
      store_rgba_16x(v, out, x);
      x += 16;
    }
  }
  if x < width {
    scalar::gray8_to_rgba_row(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// NEON `gray8_to_hsv_row`: H=0, S=0, V=Y — stores three memset-like planes.
///
/// # Safety
/// NEON must be available. All slices `>= width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray8_to_hsv_row(
  y_plane: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);
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
    );
  }
}

// ---- GrayN (const BITS) ------------------------------------------------------

/// NEON `gray_n_to_rgb_row<BITS>`: mask → shift → broadcast → packed RGB u8.
///
/// # Safety
/// NEON must be available. Slices sized correctly for `width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_rgb_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
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
    scalar::gray_n_to_rgb_row::<BITS>(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// NEON `gray_n_to_rgba_row<BITS>`.
///
/// # Safety
/// NEON must be available. Slices sized correctly for `width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_rgba_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
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
    scalar::gray_n_to_rgba_row::<BITS>(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// NEON `gray_n_to_rgb_u16_row<BITS>`: mask → broadcast 3x.
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_rgb_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
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
    );
  }
}

/// NEON `gray_n_to_rgba_u16_row<BITS>`: mask → broadcast 3x + α = bits_mask.
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_rgba_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
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
    );
  }
}

/// NEON `gray_n_to_luma_row<BITS>`: mask → shift → u8.
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
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
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
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
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray_n_to_hsv_row<const BITS: u32>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);
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
    );
  }
}

// ---- Gray16 ------------------------------------------------------------------

/// NEON `gray16_to_rgb_row`: `>> 8` → broadcast → packed RGB u8.
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_rgb_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
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
    scalar::gray16_to_rgb_row(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// NEON `gray16_to_rgba_row`: `>> 8` → broadcast → packed RGBA u8, α=0xFF.
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_rgba_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
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
    scalar::gray16_to_rgba_row(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// NEON `gray16_to_rgb_u16_row`: identity broadcast × 3.
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_rgb_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 3);
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
    scalar::gray16_to_rgb_u16_row(&y_plane[x..width], &mut out[x * 3..width * 3], width - x);
  }
}

/// NEON `gray16_to_rgba_u16_row`: identity broadcast × 3 + α=0xFFFF.
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_rgba_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(out.len() >= width * 4);
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
    scalar::gray16_to_rgba_u16_row(&y_plane[x..width], &mut out[x * 4..width * 4], width - x);
  }
}

/// NEON `gray16_to_luma_row`: `>> 8` → u8.
///
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
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
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
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
/// # Safety
/// NEON must be available.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn gray16_to_hsv_row(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width);
  debug_assert!(h_out.len() >= width);
  debug_assert!(s_out.len() >= width);
  debug_assert!(v_out.len() >= width);
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
    );
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use crate::row::scalar::gray as scalar;

  const WIDTHS: &[usize] = &[1, 7, 8, 15, 16, 17, 32, 33, 64, 65, 128, 130];

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
      unsafe { super::gray8_to_rgb_row(&plane, &mut simd, w) };
      scalar::gray8_to_rgb_row(&plane, &mut scal, w);
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
      unsafe { super::gray8_to_rgba_row(&plane, &mut simd, w) };
      scalar::gray8_to_rgba_row(&plane, &mut scal, w);
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
      unsafe { super::gray8_to_hsv_row(&plane, &mut sh, &mut ss, &mut sv, w) };
      scalar::gray8_to_hsv_row(&plane, &mut rh, &mut rs, &mut rv, w);
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
      unsafe { super::gray_n_to_rgb_row::<10>(&plane, &mut simd, w) };
      scalar::gray_n_to_rgb_row::<10>(&plane, &mut scal, w);
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
      unsafe { super::gray16_to_rgb_row(&plane, &mut simd, w) };
      scalar::gray16_to_rgb_row(&plane, &mut scal, w);
      assert_eq!(simd, scal, "width={w}");
    }
  }
}
