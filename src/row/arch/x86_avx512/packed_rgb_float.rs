//! AVX-512 (F + BW) kernels for the Tier 9 packed-float-RGB
//! (`Rgbf32`) source. 16-lane `__m512` registers; same lane-aligned
//! pixel chunking as the AVX2 backend at twice the throughput.
//!
//! Process 16 pixels = 48 lanes per iteration so the loop boundary
//! lands on a pixel boundary; the scalar tail handles the leftover
//! 0–15 pixels.

use core::arch::x86_64::*;

use super::scalar;

#[inline(always)]
unsafe fn clamp_scale_to_u32_512(v: __m512, zero: __m512, one: __m512, scale: __m512) -> __m512i {
  unsafe {
    let clamped = _mm512_min_ps(_mm512_max_ps(v, zero), one);
    let scaled = _mm512_mul_ps(clamped, scale);
    // AVX-512 embedded rounding: `_mm512_cvt_roundps_epi32` with
    // `TO_NEAREST_INT | NO_EXC` forces banker's rounding in a single
    // instruction, independent of the ambient MXCSR rounding mode.
    _mm512_cvt_roundps_epi32::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(scaled)
  }
}

/// f32 RGB → u8 RGB.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgbf32_to_rgb_row(rgb_in: &[f32], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    // 16 pixels = 48 lanes per iter (3 × 16-lane f32 loads).
    while lane + 48 <= total_lanes {
      let v0 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane));
      let v1 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane + 16));
      let v2 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane + 32));

      let i0 = clamp_scale_to_u32_512(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32_512(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32_512(v2, zero, one, scale);

      // i32x16 → u8x16 saturating narrow via `_mm512_cvtusepi32_epi8`
      // (AVX-512F). Each result is a 128-bit vector of 16 bytes — write
      // 16 bytes for each of i0/i1/i2 sequentially → 48 bytes total.
      let b0 = _mm512_cvtusepi32_epi8(i0);
      let b1 = _mm512_cvtusepi32_epi8(i1);
      let b2 = _mm512_cvtusepi32_epi8(i2);

      _mm_storeu_si128(rgb_out.as_mut_ptr().add(lane) as *mut __m128i, b0);
      _mm_storeu_si128(rgb_out.as_mut_ptr().add(lane + 16) as *mut __m128i, b1);
      _mm_storeu_si128(rgb_out.as_mut_ptr().add(lane + 32) as *mut __m128i, b2);

      lane += 48;
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
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgbf32_to_rgba_row(rgb_in: &[f32], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(255.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    let mut pix = 0usize;
    while lane + 48 <= total_lanes {
      let v0 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane));
      let v1 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane + 16));
      let v2 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane + 32));

      let i0 = clamp_scale_to_u32_512(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32_512(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32_512(v2, zero, one, scale);

      let b0 = _mm512_cvtusepi32_epi8(i0);
      let b1 = _mm512_cvtusepi32_epi8(i1);
      let b2 = _mm512_cvtusepi32_epi8(i2);

      let mut tmp = [0u8; 48];
      _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, b0);
      _mm_storeu_si128(tmp.as_mut_ptr().add(16) as *mut __m128i, b1);
      _mm_storeu_si128(tmp.as_mut_ptr().add(32) as *mut __m128i, b2);
      // tmp[0..48] = 16 RGB pixels. Interleave alpha at trailing slot
      // of each 4-byte group → 64 RGBA bytes.
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 64);
      for p in 0..16 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFF;
      }

      lane += 48;
      pix += 16;
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

/// f32 RGB → u16 RGB.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgbf32_to_rgb_u16_row(rgb_in: &[f32], rgb_out: &mut [u16], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    while lane + 48 <= total_lanes {
      let v0 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane));
      let v1 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane + 16));
      let v2 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane + 32));

      let i0 = clamp_scale_to_u32_512(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32_512(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32_512(v2, zero, one, scale);

      // i32x16 → u16x16 saturating narrow via `_mm512_cvtusepi32_epi16`
      // (AVX-512F). Output is a 256-bit vector of 16 u16 elements.
      let h0 = _mm512_cvtusepi32_epi16(i0);
      let h1 = _mm512_cvtusepi32_epi16(i1);
      let h2 = _mm512_cvtusepi32_epi16(i2);

      _mm256_storeu_si256(rgb_out.as_mut_ptr().add(lane) as *mut __m256i, h0);
      _mm256_storeu_si256(rgb_out.as_mut_ptr().add(lane + 16) as *mut __m256i, h1);
      _mm256_storeu_si256(rgb_out.as_mut_ptr().add(lane + 32) as *mut __m256i, h2);

      lane += 48;
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
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgbf32_to_rgba_u16_row(rgb_in: &[f32], rgba_out: &mut [u16], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_u16_out row too short");

  unsafe {
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let scale = _mm512_set1_ps(65535.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    let mut pix = 0usize;
    while lane + 48 <= total_lanes {
      let v0 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane));
      let v1 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane + 16));
      let v2 = _mm512_loadu_ps(rgb_in.as_ptr().add(lane + 32));

      let i0 = clamp_scale_to_u32_512(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32_512(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32_512(v2, zero, one, scale);

      let h0 = _mm512_cvtusepi32_epi16(i0);
      let h1 = _mm512_cvtusepi32_epi16(i1);
      let h2 = _mm512_cvtusepi32_epi16(i2);

      let mut tmp = [0u16; 48];
      _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, h0);
      _mm256_storeu_si256(tmp.as_mut_ptr().add(16) as *mut __m256i, h1);
      _mm256_storeu_si256(tmp.as_mut_ptr().add(32) as *mut __m256i, h2);
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 64);
      for p in 0..16 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFFFF;
      }

      lane += 48;
      pix += 16;
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
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn rgbf32_to_rgb_f32_row(rgb_in: &[f32], rgb_out: &mut [f32], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  unsafe {
    let total = width * 3;
    let mut i = 0usize;
    while i + 16 <= total {
      let v = _mm512_loadu_ps(rgb_in.as_ptr().add(i));
      _mm512_storeu_ps(rgb_out.as_mut_ptr().add(i), v);
      i += 16;
    }
    while i < total {
      *rgb_out.get_unchecked_mut(i) = *rgb_in.get_unchecked(i);
      i += 1;
    }
  }
}

// ---- Tier 9 — Rgbf16 AVX-512 + F16C entry points ---------------------------
//
// `_mm512_cvtph_ps` (F16C + AVX-512F) widens 16 × f16 (stored as 16 × i16 in
// a __m256i) to 16 × f32 in a __m512.  We load 32 bytes (16 f16 values) via
// `_mm256_loadu_si256`.
//
// Downstream: after widening a 48-lane chunk (= 16 pixels) to f32, we call the
// existing AVX-512 Rgbf32 kernels.  The scalar tail uses
// `crate::row::scalar::rgbf16_to_*_row`.
//
// `#[target_feature(enable = "avx512f,f16c")]` — avx512bw is implicitly
// available whenever avx512f is, and f16c is the narrowing/widening extension.

/// Widen 16 × f16 (at `ptr`, 32 bytes) to 16 × f32 (returned as `__m512`).
///
/// # Safety
///
/// * AVX-512F + F16C must be available.
/// * `ptr` must be valid for 32 bytes (16 × u16 / f16).
#[inline]
#[target_feature(enable = "avx512f,f16c")]
unsafe fn widen_f16x16_avx512(ptr: *const half::f16) -> __m512 {
  unsafe {
    let raw = _mm256_loadu_si256(ptr as *const __m256i);
    _mm512_cvtph_ps(raw)
  }
}

/// f16 RGB → u8 RGB (AVX-512F + F16C).
///
/// # Safety
///
/// 1. AVX-512F, AVX-512BW, and F16C must be available.
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_row(rgb_in: &[half::f16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  // Process 16 pixels (48 f16 lanes) per iteration.
  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 48 <= total_lanes {
    let mut buf = [0.0f32; 48];
    unsafe {
      let f0 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane + 16));
      let f2 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane + 32));
      _mm512_storeu_ps(buf.as_mut_ptr(), f0);
      _mm512_storeu_ps(buf.as_mut_ptr().add(16), f1);
      _mm512_storeu_ps(buf.as_mut_ptr().add(32), f2);
      rgbf32_to_rgb_row(&buf, rgb_out.get_unchecked_mut(lane..lane + 48), 16);
    }
    lane += 48;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf16_to_rgb_row(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f16 RGB → u8 RGBA (alpha `0xFF`) (AVX-512F + F16C).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn rgbf16_to_rgba_row(rgb_in: &[half::f16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 48 <= total_lanes {
    let mut buf = [0.0f32; 48];
    unsafe {
      let f0 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane + 16));
      let f2 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane + 32));
      _mm512_storeu_ps(buf.as_mut_ptr(), f0);
      _mm512_storeu_ps(buf.as_mut_ptr().add(16), f1);
      _mm512_storeu_ps(buf.as_mut_ptr().add(32), f2);
      rgbf32_to_rgba_row(&buf, rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 64), 16);
    }
    lane += 48;
    pix += 16;
  }
  if pix < width {
    scalar::rgbf16_to_rgba_row(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f16 RGB → u16 RGB (AVX-512F + F16C).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [u16]` with
/// `len() >= 3 * width` u16 elements.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_u16_row(
  rgb_in: &[half::f16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 48 <= total_lanes {
    let mut buf = [0.0f32; 48];
    unsafe {
      let f0 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane + 16));
      let f2 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane + 32));
      _mm512_storeu_ps(buf.as_mut_ptr(), f0);
      _mm512_storeu_ps(buf.as_mut_ptr().add(16), f1);
      _mm512_storeu_ps(buf.as_mut_ptr().add(32), f2);
      rgbf32_to_rgb_u16_row(&buf, rgb_out.get_unchecked_mut(lane..lane + 48), 16);
    }
    lane += 48;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf16_to_rgb_u16_row(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f16 RGB → u16 RGBA (alpha `0xFFFF`) (AVX-512F + F16C).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_u16_row`] but `rgba_out.len() >= 4 * width`.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn rgbf16_to_rgba_u16_row(
  rgb_in: &[half::f16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_u16_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 48 <= total_lanes {
    let mut buf = [0.0f32; 48];
    unsafe {
      let f0 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane + 16));
      let f2 = widen_f16x16_avx512(rgb_in.as_ptr().add(lane + 32));
      _mm512_storeu_ps(buf.as_mut_ptr(), f0);
      _mm512_storeu_ps(buf.as_mut_ptr().add(16), f1);
      _mm512_storeu_ps(buf.as_mut_ptr().add(32), f2);
      rgbf32_to_rgba_u16_row(&buf, rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 64), 16);
    }
    lane += 48;
    pix += 16;
  }
  if pix < width {
    scalar::rgbf16_to_rgba_u16_row(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f16 RGB → f32 RGB (lossless widen) (AVX-512F + F16C).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [f32]` with
/// `len() >= 3 * width` f32 elements.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_f32_row(
  rgb_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 16 <= total_lanes {
    unsafe {
      let f = widen_f16x16_avx512(rgb_in.as_ptr().add(lane));
      _mm512_storeu_ps(rgb_out.as_mut_ptr().add(lane), f);
    }
    lane += 16;
  }
  // Scalar tail for the last 0-15 lanes.
  for i in lane..total_lanes {
    unsafe {
      *rgb_out.get_unchecked_mut(i) = rgb_in.get_unchecked(i).to_f32();
    }
  }
}

/// f16 RGB → f16 RGB lossless pass-through (AVX-512F + F16C).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [half::f16]` with
/// `len() >= 3 * width` f16 elements.
#[allow(dead_code)]
#[inline]
#[target_feature(enable = "avx512f,avx512bw,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_f16_row(
  rgb_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f16_out row too short");
  scalar::rgbf16_to_rgb_f16_row(rgb_in, rgb_out, width);
}
