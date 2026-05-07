//! SSE4.1 kernels for the Tier 9 packed-float-RGB (`Rgbf32`) source.
//!
//! Each kernel processes 4 `f32` lanes per `__m128` register; the
//! integer-output kernels use `_mm_min_ps` / `_mm_max_ps` for the
//! `[0, 1]` clamp, `_mm_mul_ps` for the scale,
//! `_mm_round_ps::<TO_NEAREST_INT | NO_EXC>` followed by
//! `_mm_cvttps_epi32` (truncate — MXCSR-independent) for the
//! round-to-nearest-even cast, and `_mm_packus_*` for the saturating
//! narrow.
//!
//! For `<const BE: bool>` kernels, each 4-lane f32 load is replaced by
//! `load_endian_u32x4::<BE>` (a `__m128i` with byte-swapped u32 lanes
//! for BE inputs) followed by `_mm_castsi128_ps` to reinterpret as f32.
//!
//! Pixel-aligned chunks (4 pixels = 12 lanes per iter for the u8/u16
//! integer-output paths) keep the loop boundary on a pixel boundary
//! so the scalar tail handles only the final 0–3 pixels.

use core::arch::x86_64::*;

use super::{endian::load_endian_u32x4, scalar};

/// Load 4 f32 lanes from `ptr` in endian-aware fashion.
///
/// # Safety
///
/// SSE4.1 + SSSE3 must be available; `ptr` must be valid for 16 bytes.
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn load_f32x4<const BE: bool>(ptr: *const f32) -> __m128 {
  unsafe {
    let u = load_endian_u32x4::<BE>(ptr as *const u8);
    _mm_castsi128_ps(u)
  }
}

#[inline(always)]
unsafe fn clamp_scale_to_u32(v: __m128, zero: __m128, one: __m128, scale: __m128) -> __m128i {
  unsafe {
    let clamped = _mm_min_ps(_mm_max_ps(v, zero), one);
    let scaled = _mm_mul_ps(clamped, scale);
    // Round nearest-even independent of the ambient MXCSR rounding mode:
    // `_mm_round_ps` with `TO_NEAREST_INT | NO_EXC` forces banker's
    // rounding and suppresses inexact exceptions; `_mm_cvttps_epi32`
    // (truncate) then converts the already-rounded value to i32 without
    // re-reading MXCSR.
    _mm_cvttps_epi32(_mm_round_ps::<
      { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
    >(scaled))
  }
}

/// f32 RGB → u8 RGB. Clamp `[0, 1]` × 255, saturating cast.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgb_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(255.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    while lane + 12 <= total_lanes {
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

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
      scalar::rgbf32_to_rgb_row::<BE>(
        &rgb_in[pix_done * 3..width * 3],
        &mut rgb_out[pix_done * 3..width * 3],
        width - pix_done,
      );
    }
  }
}

/// f32 RGB → u8 RGBA (alpha forced to `0xFF`).
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgba_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
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
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

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
      scalar::rgbf32_to_rgba_row::<BE>(
        &rgb_in[pix * 3..width * 3],
        &mut rgba_out[pix * 4..width * 4],
        width - pix,
      );
    }
  }
}

/// f32 RGB → u16 RGB. Clamp `[0, 1]` × 65535, saturating cast.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_row`] but `rgb_out` is `&mut [u16]` with
/// `len() >= 3 * width` u16 elements.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let scale = _mm_set1_ps(65535.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    while lane + 12 <= total_lanes {
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

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
      scalar::rgbf32_to_rgb_u16_row::<BE>(
        &rgb_in[pix_done * 3..width * 3],
        &mut rgb_out[pix_done * 3..width * 3],
        width - pix_done,
      );
    }
  }
}

/// f32 RGB → u16 RGBA (alpha forced to `0xFFFF`).
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_u16_row`] but the output is `&mut [u16]`
/// with `len() >= 4 * width` u16 elements.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
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
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

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
      scalar::rgbf32_to_rgba_u16_row::<BE>(
        &rgb_in[pix * 3..width * 3],
        &mut rgba_out[pix * 4..width * 4],
        width - pix,
      );
    }
  }
}

/// f32 RGB → f32 RGB lossless pass-through.
///
/// When `BE = true` the input values are byte-swapped to host-native
/// before being written.
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_row`] but `rgb_out` is `&mut [f32]` with
/// `len() >= 3 * width` f32 elements.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn rgbf32_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  unsafe {
    let total = width * 3;
    let mut i = 0usize;
    if BE {
      while i + 4 <= total {
        let v = load_f32x4::<BE>(rgb_in.as_ptr().add(i));
        _mm_storeu_ps(rgb_out.as_mut_ptr().add(i), v);
        i += 4;
      }
      while i < total {
        let bits = (*rgb_in.get_unchecked(i)).to_bits().swap_bytes();
        *rgb_out.get_unchecked_mut(i) = f32::from_bits(bits);
        i += 1;
      }
    } else {
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
}

// ---- Tier 9 — Rgbf16 SSE4.1 + F16C entry points ---------------------------
//
// `_mm_cvtph_ps` (F16C) widens 4 × f16 (stored as 4 × i16 in the low 64 bits
// of a __m128i) to 4 × f32 in a __m128.  We load 8 bytes (4 f16 values) via
// `_mm_loadl_epi64` (64-bit load into the low half of __m128i).
//
// For BE: load 8 bytes via `load_endian_u16x8::<BE>` which byte-swaps each
// u16 for big-endian inputs, then call `_mm_cvtph_ps` on the result.
//
// `#[target_feature(enable = "sse4.1,f16c")]` ensures both features are active
// in the body even though F16C is an independent feature bit.

use super::endian::load_endian_u16x4;

/// Widen 4 × f16 (at `ptr`, 8 bytes) to 4 × f32 (returned as `__m128`).
///
/// For `BE = true` the f16 values are stored big-endian; bytes are swapped
/// before the F16C widening conversion. The loader reads exactly 8 bytes
/// regardless of `BE` so the caller's `ptr` only needs 8 readable bytes
/// (a 16-byte load via `load_endian_u16x8` would tail-overread the 4 × f16
/// region the kernel actually owns).
///
/// # Safety
///
/// * SSE4.1 + F16C must be available.
/// * `ptr` must be valid for 8 bytes (4 × u16 / f16).
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
unsafe fn widen_f16x4_sse<const BE: bool>(ptr: *const half::f16) -> __m128 {
  unsafe {
    // 8-byte load (low 64 bits of __m128i, upper half zero). For `BE = true`
    // the loader byte-swaps each u16 in place; for `BE = false` it's a plain
    // load. `_mm_cvtph_ps` reads only the low 4 × f16 (low 64 bits), so the
    // upper half being zero is harmless.
    let raw = load_endian_u16x4::<BE>(ptr as *const u8);
    _mm_cvtph_ps(raw)
  }
}

/// f16 RGB → u8 RGB (SSE4.1 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// 1. SSE4.1 and F16C must be available.
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  // Process 4 pixels (12 f16 lanes) per iteration.
  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 12 <= total_lanes {
    let mut buf = [0.0f32; 12];
    unsafe {
      let f0 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane + 4));
      let f2 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane + 8));
      _mm_storeu_ps(buf.as_mut_ptr(), f0);
      _mm_storeu_ps(buf.as_mut_ptr().add(4), f1);
      _mm_storeu_ps(buf.as_mut_ptr().add(8), f2);
      rgbf32_to_rgb_row::<false>(&buf, rgb_out.get_unchecked_mut(lane..lane + 12), 4);
    }
    lane += 12;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf16_to_rgb_row::<BE>(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f16 RGB → u8 RGBA (alpha `0xFF`) (SSE4.1 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn rgbf16_to_rgba_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 12 <= total_lanes {
    let mut buf = [0.0f32; 12];
    unsafe {
      let f0 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane + 4));
      let f2 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane + 8));
      _mm_storeu_ps(buf.as_mut_ptr(), f0);
      _mm_storeu_ps(buf.as_mut_ptr().add(4), f1);
      _mm_storeu_ps(buf.as_mut_ptr().add(8), f2);
      rgbf32_to_rgba_row::<false>(&buf, rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16), 4);
    }
    lane += 12;
    pix += 4;
  }
  if pix < width {
    scalar::rgbf16_to_rgba_row::<BE>(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f16 RGB → u16 RGB (SSE4.1 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [u16]` with
/// `len() >= 3 * width` u16 elements.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 12 <= total_lanes {
    let mut buf = [0.0f32; 12];
    unsafe {
      let f0 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane + 4));
      let f2 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane + 8));
      _mm_storeu_ps(buf.as_mut_ptr(), f0);
      _mm_storeu_ps(buf.as_mut_ptr().add(4), f1);
      _mm_storeu_ps(buf.as_mut_ptr().add(8), f2);
      rgbf32_to_rgb_u16_row::<false>(&buf, rgb_out.get_unchecked_mut(lane..lane + 12), 4);
    }
    lane += 12;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf16_to_rgb_u16_row::<BE>(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f16 RGB → u16 RGBA (alpha `0xFFFF`) (SSE4.1 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_u16_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn rgbf16_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_u16_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 12 <= total_lanes {
    let mut buf = [0.0f32; 12];
    unsafe {
      let f0 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane + 4));
      let f2 = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane + 8));
      _mm_storeu_ps(buf.as_mut_ptr(), f0);
      _mm_storeu_ps(buf.as_mut_ptr().add(4), f1);
      _mm_storeu_ps(buf.as_mut_ptr().add(8), f2);
      rgbf32_to_rgba_u16_row::<false>(&buf, rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16), 4);
    }
    lane += 12;
    pix += 4;
  }
  if pix < width {
    scalar::rgbf16_to_rgba_u16_row::<BE>(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f16 RGB → f32 RGB (lossless widen) (SSE4.1 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [f32]` with
/// `len() >= 3 * width` f32 elements.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 4 <= total_lanes {
    unsafe {
      let f = widen_f16x4_sse::<BE>(rgb_in.as_ptr().add(lane));
      _mm_storeu_ps(rgb_out.as_mut_ptr().add(lane), f);
    }
    lane += 4;
  }
  // Scalar tail for the last 0-3 lanes.
  for i in lane..total_lanes {
    unsafe {
      let v = load_f16_scalar::<BE>(rgb_in, i);
      *rgb_out.get_unchecked_mut(i) = v.to_f32();
    }
  }
}

/// f16 RGB → f16 RGB lossless pass-through (SSE4.1 + F16C).
///
/// When `BE = true` the input values are byte-swapped to host-native order.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [half::f16]` with
/// `len() >= 3 * width` f16 elements.
#[inline]
#[target_feature(enable = "sse4.1,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_f16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f16_out row too short");
  scalar::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
}

/// Scalar f16 load helper for tail loops (SSE4.1 module).
#[inline(always)]
fn load_f16_scalar<const BE: bool>(rgb_in: &[half::f16], i: usize) -> half::f16 {
  let bits = rgb_in[i].to_bits();
  half::f16::from_bits(if BE { bits.swap_bytes() } else { bits })
}
