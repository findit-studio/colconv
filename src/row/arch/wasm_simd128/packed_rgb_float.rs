//! WASM simd128 kernels for the Tier 9 packed-float-RGB (`Rgbf32`)
//! source. simd128 is a 128-bit ISA so each register holds 4 `f32`
//! lanes — same shape as the SSE4.1 backend.
//!
//! `i32x4_trunc_sat_f32x4` performs a saturating truncate-toward-zero
//! cast; round-to-nearest-even isn't a primitive on simd128 so we
//! preface the cast with `f32x4_nearest` (round to nearest even, the
//! IEEE 754 default) — matching the scalar path's
//! `round_ties_even_nonneg` helper.

use core::arch::wasm32::*;

use super::scalar;

#[inline(always)]
fn clamp_scale_to_i32(v: v128, zero: v128, one: v128, scale: v128) -> v128 {
  let clamped = f32x4_min(f32x4_max(v, zero), one);
  let scaled = f32x4_mul(clamped, scale);
  // Round to nearest even (IEEE default), then saturating-truncate to i32.
  let rounded = f32x4_nearest(scaled);
  i32x4_trunc_sat_f32x4(rounded)
}

/// f32 RGB → u8 RGB.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgb_row(rgb_in: &[f32], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(255.0);

  let total_lanes = width * 3;
  let mut lane = 0usize;
  // 4 pixels = 12 lanes per iter.
  while lane + 12 <= total_lanes {
    unsafe {
      let v0 = v128_load(rgb_in.as_ptr().add(lane) as *const v128);
      let v1 = v128_load(rgb_in.as_ptr().add(lane + 4) as *const v128);
      let v2 = v128_load(rgb_in.as_ptr().add(lane + 8) as *const v128);

      let i0 = clamp_scale_to_i32(v0, zero, one, scale);
      let i1 = clamp_scale_to_i32(v1, zero, one, scale);
      let i2 = clamp_scale_to_i32(v2, zero, one, scale);

      // i32x4 → i16x8 (saturating signed narrow); each yields 4 i16 lanes.
      let h01 = i16x8_narrow_i32x4(i0, i1);
      let h22 = i16x8_narrow_i32x4(i2, i2);

      // i16x8 → u8x16 (saturating unsigned narrow). 12 valid bytes per iter.
      let bytes = u8x16_narrow_i16x8(h01, h22);

      let mut tmp = [0u8; 16];
      v128_store(tmp.as_mut_ptr() as *mut v128, bytes);
      rgb_out
        .get_unchecked_mut(lane..lane + 12)
        .copy_from_slice(&tmp[..12]);
    }
    lane += 12;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf32_to_rgb_row(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f32 RGB → u8 RGBA (alpha forced to `0xFF`).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgba_row(rgb_in: &[f32], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(255.0);

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 12 <= total_lanes {
    unsafe {
      let v0 = v128_load(rgb_in.as_ptr().add(lane) as *const v128);
      let v1 = v128_load(rgb_in.as_ptr().add(lane + 4) as *const v128);
      let v2 = v128_load(rgb_in.as_ptr().add(lane + 8) as *const v128);

      let i0 = clamp_scale_to_i32(v0, zero, one, scale);
      let i1 = clamp_scale_to_i32(v1, zero, one, scale);
      let i2 = clamp_scale_to_i32(v2, zero, one, scale);

      let h01 = i16x8_narrow_i32x4(i0, i1);
      let h22 = i16x8_narrow_i32x4(i2, i2);
      let bytes = u8x16_narrow_i16x8(h01, h22);

      let mut tmp = [0u8; 16];
      v128_store(tmp.as_mut_ptr() as *mut v128, bytes);
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16);
      for p in 0..4 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFF;
      }
    }
    lane += 12;
    pix += 4;
  }
  if pix < width {
    scalar::rgbf32_to_rgba_row(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f32 RGB → u16 RGB.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgb_u16_row(rgb_in: &[f32], rgb_out: &mut [u16], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(65535.0);

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 12 <= total_lanes {
    unsafe {
      let v0 = v128_load(rgb_in.as_ptr().add(lane) as *const v128);
      let v1 = v128_load(rgb_in.as_ptr().add(lane + 4) as *const v128);
      let v2 = v128_load(rgb_in.as_ptr().add(lane + 8) as *const v128);

      let i0 = clamp_scale_to_i32(v0, zero, one, scale);
      let i1 = clamp_scale_to_i32(v1, zero, one, scale);
      let i2 = clamp_scale_to_i32(v2, zero, one, scale);

      // i32x4 → u16x8 saturating narrow (`u16x8_narrow_i32x4` saturates
      // negatives to 0 and clamps at 65535). After clamp+scale our
      // values are already in [0, 65535] so saturation is a no-op
      // semantically but still required by the type system.
      let u01 = u16x8_narrow_i32x4(i0, i1);
      let u22 = u16x8_narrow_i32x4(i2, i2);

      let mut tmp = [0u16; 16];
      v128_store(tmp.as_mut_ptr() as *mut v128, u01);
      v128_store(tmp.as_mut_ptr().add(8) as *mut v128, u22);
      rgb_out
        .get_unchecked_mut(lane..lane + 12)
        .copy_from_slice(&tmp[..12]);
    }
    lane += 12;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf32_to_rgb_u16_row(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f32 RGB → u16 RGBA (alpha forced to `0xFFFF`).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgba_u16_row(rgb_in: &[f32], rgba_out: &mut [u16], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_u16_out row too short");

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(65535.0);

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 12 <= total_lanes {
    unsafe {
      let v0 = v128_load(rgb_in.as_ptr().add(lane) as *const v128);
      let v1 = v128_load(rgb_in.as_ptr().add(lane + 4) as *const v128);
      let v2 = v128_load(rgb_in.as_ptr().add(lane + 8) as *const v128);

      let i0 = clamp_scale_to_i32(v0, zero, one, scale);
      let i1 = clamp_scale_to_i32(v1, zero, one, scale);
      let i2 = clamp_scale_to_i32(v2, zero, one, scale);

      let u01 = u16x8_narrow_i32x4(i0, i1);
      let u22 = u16x8_narrow_i32x4(i2, i2);

      let mut tmp = [0u16; 16];
      v128_store(tmp.as_mut_ptr() as *mut v128, u01);
      v128_store(tmp.as_mut_ptr().add(8) as *mut v128, u22);
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16);
      for p in 0..4 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFFFF;
      }
    }
    lane += 12;
    pix += 4;
  }
  if pix < width {
    scalar::rgbf32_to_rgba_u16_row(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f32 RGB → f32 RGB lossless pass-through.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgb_f32_row(rgb_in: &[f32], rgb_out: &mut [f32], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  let total = width * 3;
  let mut i = 0usize;
  while i + 4 <= total {
    unsafe {
      let v = v128_load(rgb_in.as_ptr().add(i) as *const v128);
      v128_store(rgb_out.as_mut_ptr().add(i) as *mut v128, v);
    }
    i += 4;
  }
  while i < total {
    unsafe {
      *rgb_out.get_unchecked_mut(i) = *rgb_in.get_unchecked(i);
    }
    i += 1;
  }
}
