//! SSE4.1 kernels for the Tier 9 packed-float-RGB (`Rgbf32`) source.
//!
//! Each kernel processes 4 `f32` lanes per `__m128` register; the
//! integer-output kernels use `_mm_min_ps` / `_mm_max_ps` for the
//! `[0, 1]` clamp, `_mm_mul_ps` for the scale, `_mm_cvtps_epi32` for
//! the round-to-nearest-even cast (uses the current MXCSR rounding
//! mode — round-to-nearest-even is the default), and `_mm_packus_*`
//! for the saturating narrow.
//!
//! Pixel-aligned chunks (4 pixels = 12 lanes per iter for the u8/u16
//! integer-output paths) keep the loop boundary on a pixel boundary
//! so the scalar tail handles only the final 0–3 pixels.

use core::arch::x86_64::*;

use super::scalar;

#[inline(always)]
unsafe fn clamp_scale_to_u32(v: __m128, zero: __m128, one: __m128, scale: __m128) -> __m128i {
  unsafe {
    let clamped = _mm_min_ps(_mm_max_ps(v, zero), one);
    let scaled = _mm_mul_ps(clamped, scale);
    // `_mm_cvtps_epi32` uses MXCSR rounding (round-to-nearest-even by
    // default). After clamping to `[0, scale]` the result fits in i32
    // safely (0..=scale, scale ≤ 65535).
    _mm_cvtps_epi32(scaled)
  }
}

/// f32 RGB → u8 RGB. Clamp `[0, 1]` × 255, saturating cast.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgb_row(rgb_in: &[f32], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    while lane + 12 <= total_lanes {
      let v0 = _mm_loadu_ps(rgb_in.as_ptr().add(lane));
      let v1 = _mm_loadu_ps(rgb_in.as_ptr().add(lane + 4));
      let v2 = _mm_loadu_ps(rgb_in.as_ptr().add(lane + 8));

      let i0 = clamp_scale_to_u32(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32(v2, zero, one, scale);

      // Saturating narrow i32x4 → i16x8 (pack pairs of vectors).
      let i01 = _mm_packs_epi32(i0, i1);
      let i22 = _mm_packs_epi32(i2, i2);
      // Saturating narrow i16x16 → u8x16. We need 12 of those bytes.
      let bytes = _mm_packus_epi16(i01, i22);

      // Store 12 bytes from the low half of the u8x16 vector.
      let mut tmp = [0u8; 16];
      _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, bytes);
      rgb_out
        .get_unchecked_mut(lane..lane + 12)
        .copy_from_slice(&tmp[..12]);

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
}

/// f32 RGB → u8 RGBA (alpha forced to `0xFF`).
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgba_row(rgb_in: &[f32], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    let mut pix = 0usize;
    // Process 4 pixels (12 lanes) per iteration; emit 16 RGBA bytes
    // via per-pixel scalar interleave (the input is already in
    // R, G, B, R, G, B, … layout, so we widen the 12 bytes to 16 by
    // inserting alpha at the trailing position of each 4-byte group).
    while lane + 12 <= total_lanes {
      let v0 = _mm_loadu_ps(rgb_in.as_ptr().add(lane));
      let v1 = _mm_loadu_ps(rgb_in.as_ptr().add(lane + 4));
      let v2 = _mm_loadu_ps(rgb_in.as_ptr().add(lane + 8));

      let i0 = clamp_scale_to_u32(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32(v2, zero, one, scale);

      let i01 = _mm_packs_epi32(i0, i1);
      let i22 = _mm_packs_epi32(i2, i2);
      let bytes = _mm_packus_epi16(i01, i22);

      let mut tmp = [0u8; 16];
      _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, bytes);
      // Now tmp[0..12] = R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3.
      // Interleave alpha at positions 3, 7, 11, 15.
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16);
      for p in 0..4 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFF;
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
}

/// f32 RGB → u16 RGB. Clamp `[0, 1]` × 65535, saturating cast.
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_row`] but `rgb_out` is `&mut [u16]` with
/// `len() >= 3 * width` u16 elements.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgb_u16_row(rgb_in: &[f32], rgb_out: &mut [u16], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    while lane + 12 <= total_lanes {
      let v0 = _mm_loadu_ps(rgb_in.as_ptr().add(lane));
      let v1 = _mm_loadu_ps(rgb_in.as_ptr().add(lane + 4));
      let v2 = _mm_loadu_ps(rgb_in.as_ptr().add(lane + 8));

      let i0 = clamp_scale_to_u32(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32(v2, zero, one, scale);

      // Saturating-narrow i32 → u16 via `_mm_packus_epi32` (SSE4.1).
      let u01 = _mm_packus_epi32(i0, i1);
      let u22 = _mm_packus_epi32(i2, i2);

      // Store 8 + 4 u16 elements.
      _mm_storeu_si128(rgb_out.as_mut_ptr().add(lane) as *mut __m128i, u01);
      // The low half of u22 is 4 u16 elements (8 bytes).
      _mm_storel_epi64(rgb_out.as_mut_ptr().add(lane + 8) as *mut __m128i, u22);

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
}

/// f32 RGB → u16 RGBA (alpha forced to `0xFFFF`).
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_u16_row`] but the output is `&mut [u16]`
/// with `len() >= 4 * width` u16 elements.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgba_u16_row(rgb_in: &[f32], rgba_out: &mut [u16], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_u16_out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    let mut pix = 0usize;
    while lane + 12 <= total_lanes {
      let v0 = _mm_loadu_ps(rgb_in.as_ptr().add(lane));
      let v1 = _mm_loadu_ps(rgb_in.as_ptr().add(lane + 4));
      let v2 = _mm_loadu_ps(rgb_in.as_ptr().add(lane + 8));

      let i0 = clamp_scale_to_u32(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32(v2, zero, one, scale);

      let u01 = _mm_packus_epi32(i0, i1);
      let u22 = _mm_packus_epi32(i2, i2);

      let mut tmp = [0u16; 16];
      _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, u01);
      _mm_storel_epi64(tmp.as_mut_ptr().add(8) as *mut __m128i, u22);
      // Now tmp[0..12] holds 4 pixels of R, G, B u16 elements.
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16);
      for p in 0..4 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFFFF;
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
}

/// f32 RGB → f32 RGB lossless pass-through.
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_row`] but `rgb_out` is `&mut [f32]` with
/// `len() >= 3 * width` f32 elements.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgb_f32_row(rgb_in: &[f32], rgb_out: &mut [f32], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  unsafe {
    let total = width * 3;
    let mut i = 0usize;
    while i + 4 <= total {
      let v = _mm_loadu_ps(rgb_in.as_ptr().add(i));
      _mm_storeu_ps(rgb_out.as_mut_ptr().add(i), v);
      i += 4;
    }
    while i < total {
      *rgb_out.get_unchecked_mut(i) = *rgb_in.get_unchecked(i);
      i += 1;
    }
  }
}
