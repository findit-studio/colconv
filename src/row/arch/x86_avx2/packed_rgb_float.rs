//! AVX2 kernels for the Tier 9 packed-float-RGB (`Rgbf32`) source.
//!
//! 8-lane `__m256` registers; same pipeline as SSE4.1 but doubled
//! throughput. AVX2 supplies `_mm256_packus_epi32` (i32 → u16 saturating
//! narrow) and the standard `_mm_packus_epi16` (i16 → u8 saturating
//! narrow); cross-lane unpacks need `_mm256_permute4x64_epi64` to fix
//! the 128-bit lane interleave that AVX2 packs leave behind.
//!
//! For `<const BE: bool>` kernels, each 8-lane f32 load is replaced by
//! `load_endian_u32x8::<BE>` (a `__m256i` with byte-swapped u32 lanes
//! for BE inputs) followed by `_mm256_castsi256_ps` to reinterpret as f32.
//!
//! Pixel-aligned chunks of 8 pixels = 24 lanes per iteration so the
//! tail handles 0–7 leftover pixels.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

use super::endian::load_endian_u32x8;
// For f16 widen we need 128-bit u16 load (8 x u16).
use super::scalar;
use crate::row::arch::x86_sse41::endian::load_endian_u16x8;

/// `BE` value that makes the f32 row loaders treat their input as host-native
/// (a no-op byte-swap). Used by f16→f32 widen-then-convert paths whose stack
/// buffer is already host-native after `_mm256_cvtph_ps`. On a LE target,
/// host-native == LE so `BE = false`; on a BE target, host-native == BE so
/// `BE = true`. Without this routing the downstream `rgbf32_to_*::<false>`
/// would byte-swap an already-decoded host-native f32 buffer on BE hosts.
///
/// Also used by the `rgbf32_to_rgb_f32_row` pass-through fast path: the raw
/// `_mm256_loadu_ps`/`_mm256_storeu_ps` copy is byte-correct only when the
/// source encoding (`BE`) matches the host's native endian, so the kernel
/// falls through to the endian-aware `load_f32x8::<BE>` slow path otherwise.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Load 8 f32 lanes from `ptr` in endian-aware fashion.
///
/// # Safety
///
/// AVX2 must be available; `ptr` must be valid for 32 bytes.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_f32x8<const BE: bool>(ptr: *const f32) -> __m256 {
  unsafe {
    let u = load_endian_u32x8::<BE>(ptr as *const u8);
    _mm256_castsi256_ps(u)
  }
}

#[inline(always)]
unsafe fn clamp_scale_to_u32_256(v: __m256, zero: __m256, one: __m256, scale: __m256) -> __m256i {
  unsafe {
    let clamped = _mm256_min_ps(_mm256_max_ps(v, zero), one);
    let scaled = _mm256_mul_ps(clamped, scale);
    // Round nearest-even independent of MXCSR: `_mm256_round_ps` with
    // `TO_NEAREST_INT | NO_EXC` forces banker's rounding; `_mm256_cvttps_epi32`
    // (truncate) converts the already-rounded value without re-reading MXCSR.
    _mm256_cvttps_epi32(_mm256_round_ps::<
      { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
    >(scaled))
  }
}

/// f32 RGB → u8 RGB.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbf32_to_rgb_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    // 8 pixels = 24 lanes per iter. Three 256-bit f32 loads → 24 lanes.
    while lane + 24 <= total_lanes {
      let v0 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane + 8));
      let v2 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane + 16));

      let i0 = clamp_scale_to_u32_256(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32_256(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32_256(v2, zero, one, scale);

      // Two-step narrow: i32x8x3 → u8x24. We saturate-pack i32→i16 then
      // i16→u8.
      let i01 = _mm256_packs_epi32(i0, i1);
      // After `_mm256_packs_epi32` the 128-bit lanes are interleaved
      // (a0..a3 b0..b3 a4..a7 b4..b7 — each lane a/b coming from the
      // matching 128-bit half of i0 / i1). Permute the 64-bit chunks
      // to get sequential `[a0..a7, b0..b7]` order.
      let i01 = _mm256_permute4x64_epi64::<0b11_01_10_00>(i01);
      let i22 = _mm256_packs_epi32(i2, i2);
      let i22 = _mm256_permute4x64_epi64::<0b11_01_10_00>(i22);

      let bytes_lo = _mm256_packus_epi16(i01, i22);
      let bytes_lo = _mm256_permute4x64_epi64::<0b11_01_10_00>(bytes_lo);

      let mut tmp = [0u8; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, bytes_lo);
      // After packing: tmp[0..16] = bytes from i01 (16 i16 → 16 u8),
      // tmp[16..24] = bytes from i22 (low 8 of 16 — only the first
      // 8 are valid since we duplicated i2 with itself). Combine
      // sequentially into 24 output bytes.
      rgb_out
        .get_unchecked_mut(lane..lane + 16)
        .copy_from_slice(&tmp[..16]);
      rgb_out
        .get_unchecked_mut(lane + 16..lane + 24)
        .copy_from_slice(&tmp[16..24]);

      lane += 24;
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
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbf32_to_rgba_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(255.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    let mut pix = 0usize;
    while lane + 24 <= total_lanes {
      let v0 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane + 8));
      let v2 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane + 16));

      let i0 = clamp_scale_to_u32_256(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32_256(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32_256(v2, zero, one, scale);

      let i01 = _mm256_packs_epi32(i0, i1);
      let i01 = _mm256_permute4x64_epi64::<0b11_01_10_00>(i01);
      let i22 = _mm256_packs_epi32(i2, i2);
      let i22 = _mm256_permute4x64_epi64::<0b11_01_10_00>(i22);
      let bytes = _mm256_packus_epi16(i01, i22);
      let bytes = _mm256_permute4x64_epi64::<0b11_01_10_00>(bytes);

      let mut tmp = [0u8; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, bytes);
      // tmp[0..24] = R0..B7 (8 RGB pixels, 24 bytes). Interleave alpha
      // at trailing position of each 4-byte group.
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 32);
      for p in 0..8 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFF;
      }

      lane += 24;
      pix += 8;
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

/// f32 RGB → u16 RGB.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbf32_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    while lane + 24 <= total_lanes {
      let v0 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane + 8));
      let v2 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane + 16));

      let i0 = clamp_scale_to_u32_256(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32_256(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32_256(v2, zero, one, scale);

      // i32x8 → u16x16 saturating narrow + lane fixup.
      let u01 = _mm256_packus_epi32(i0, i1);
      let u01 = _mm256_permute4x64_epi64::<0b11_01_10_00>(u01);
      let u22 = _mm256_packus_epi32(i2, i2);
      let u22 = _mm256_permute4x64_epi64::<0b11_01_10_00>(u22);

      let mut tmp = [0u16; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, u01);
      _mm256_storeu_si256(tmp.as_mut_ptr().add(16) as *mut __m256i, u22);
      // tmp[0..16] = first 16 u16 elements; tmp[16..24] = next 8.
      rgb_out
        .get_unchecked_mut(lane..lane + 16)
        .copy_from_slice(&tmp[..16]);
      rgb_out
        .get_unchecked_mut(lane + 16..lane + 24)
        .copy_from_slice(&tmp[16..24]);

      lane += 24;
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
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbf32_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_u16_out row too short");

  unsafe {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let scale = _mm256_set1_ps(65535.0);

    let total_lanes = width * 3;
    let mut lane = 0usize;
    let mut pix = 0usize;
    while lane + 24 <= total_lanes {
      let v0 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane + 8));
      let v2 = load_f32x8::<BE>(rgb_in.as_ptr().add(lane + 16));

      let i0 = clamp_scale_to_u32_256(v0, zero, one, scale);
      let i1 = clamp_scale_to_u32_256(v1, zero, one, scale);
      let i2 = clamp_scale_to_u32_256(v2, zero, one, scale);

      let u01 = _mm256_packus_epi32(i0, i1);
      let u01 = _mm256_permute4x64_epi64::<0b11_01_10_00>(u01);
      let u22 = _mm256_packus_epi32(i2, i2);
      let u22 = _mm256_permute4x64_epi64::<0b11_01_10_00>(u22);

      let mut tmp = [0u16; 32];
      _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, u01);
      _mm256_storeu_si256(tmp.as_mut_ptr().add(16) as *mut __m256i, u22);
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 32);
      for p in 0..8 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFFFF;
      }

      lane += 24;
      pix += 8;
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
/// When `BE = true` the input values are byte-swapped to host-native before
/// being written.
#[inline]
#[target_feature(enable = "avx2")]
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
    // Fast path: when the requested encoding (BE) matches the host's native
    // endian, the bytes can be copied verbatim — `_mm256_loadu_ps` reads host-
    // native bytes which is exactly what we need to emit. Otherwise we must
    // decode through `load_f32x8::<BE>` (which byte-swaps when BE differs from
    // host-native) so the stored host-native f32 round-trips to the same value.
    if BE == HOST_NATIVE_BE {
      while i + 8 <= total {
        let v = _mm256_loadu_ps(rgb_in.as_ptr().add(i));
        _mm256_storeu_ps(rgb_out.as_mut_ptr().add(i), v);
        i += 8;
      }
      while i < total {
        *rgb_out.get_unchecked_mut(i) = *rgb_in.get_unchecked(i);
        i += 1;
      }
    } else {
      while i + 8 <= total {
        let v = load_f32x8::<BE>(rgb_in.as_ptr().add(i));
        _mm256_storeu_ps(rgb_out.as_mut_ptr().add(i), v);
        i += 8;
      }
      while i < total {
        let bits = (*rgb_in.get_unchecked(i)).to_bits();
        let host_bits = if BE {
          u32::from_be(bits)
        } else {
          u32::from_le(bits)
        };
        *rgb_out.get_unchecked_mut(i) = f32::from_bits(host_bits);
        i += 1;
      }
    }
  }
}

// ---- Tier 9 — Rgbf16 AVX2 + F16C entry points ------------------------------
//
// `_mm256_cvtph_ps` (F16C) widens 8 x f16 (stored as 8 x i16 in a __m128i)
// to 8 x f32 in a __m256.  We load 16 bytes (8 f16 values) via
// `load_endian_u16x8::<BE>` which routes to a host-native pass-through when
// the on-disk encoding matches host-native and to a byte-swap shuffle
// otherwise — correct on both LE and BE hosts.
//
// `#[target_feature(enable = "avx2,f16c")]` ensures both features are active.

/// Widen 8 x f16 (at `ptr`, 16 bytes) to 8 x f32 (returned as `__m256`).
///
/// For `BE = true` the f16 values are stored big-endian; bytes are swapped
/// before the F16C widening conversion. The historical `BE = false` branch
/// used a raw `_mm_loadu_si128` which assumed LE-encoded input on a LE host;
/// `load_endian_u16x8::<BE>` is correct on both LE and BE hosts because it
/// monomorphizes to a no-op load when on-disk encoding matches host-native
/// and to a byte-swap shuffle otherwise.
///
/// # Safety
///
/// * AVX2 + F16C must be available.
/// * `ptr` must be valid for 16 bytes (8 x u16 / f16).
#[inline]
#[target_feature(enable = "avx2,f16c")]
unsafe fn widen_f16x8_avx<const BE: bool>(ptr: *const half::f16) -> __m256 {
  unsafe {
    let raw = load_endian_u16x8::<BE>(ptr as *const u8);
    _mm256_cvtph_ps(raw)
  }
}

/// f16 RGB → u8 RGB (AVX2 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// 1. AVX2 and F16C must be available.
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  // Process 8 pixels (24 f16 lanes) per iteration.
  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 24 <= total_lanes {
    let mut buf = [0.0f32; 24];
    unsafe {
      let f0 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane + 8));
      let f2 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane + 16));
      _mm256_storeu_ps(buf.as_mut_ptr(), f0);
      _mm256_storeu_ps(buf.as_mut_ptr().add(8), f1);
      _mm256_storeu_ps(buf.as_mut_ptr().add(16), f2);
      // Buffer is host-native f32 after _mm256_cvtph_ps; route via
      // HOST_NATIVE_BE so the f32 loaders perform a no-op byte-swap on
      // both LE and BE hosts.
      rgbf32_to_rgb_row::<HOST_NATIVE_BE>(&buf, rgb_out.get_unchecked_mut(lane..lane + 24), 8);
    }
    lane += 24;
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

/// f16 RGB → u8 RGBA (alpha `0xFF`) (AVX2 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
  while lane + 24 <= total_lanes {
    let mut buf = [0.0f32; 24];
    unsafe {
      let f0 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane + 8));
      let f2 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane + 16));
      _mm256_storeu_ps(buf.as_mut_ptr(), f0);
      _mm256_storeu_ps(buf.as_mut_ptr().add(8), f1);
      _mm256_storeu_ps(buf.as_mut_ptr().add(16), f2);
      // Buffer is host-native f32; route via HOST_NATIVE_BE.
      rgbf32_to_rgba_row::<HOST_NATIVE_BE>(
        &buf,
        rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 32),
        8,
      );
    }
    lane += 24;
    pix += 8;
  }
  if pix < width {
    scalar::rgbf16_to_rgba_row::<BE>(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f16 RGB → u16 RGB (AVX2 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [u16]` with
/// `len() >= 3 * width` u16 elements.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 24 <= total_lanes {
    let mut buf = [0.0f32; 24];
    unsafe {
      let f0 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane + 8));
      let f2 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane + 16));
      _mm256_storeu_ps(buf.as_mut_ptr(), f0);
      _mm256_storeu_ps(buf.as_mut_ptr().add(8), f1);
      _mm256_storeu_ps(buf.as_mut_ptr().add(16), f2);
      // Buffer is host-native f32; route via HOST_NATIVE_BE.
      rgbf32_to_rgb_u16_row::<HOST_NATIVE_BE>(&buf, rgb_out.get_unchecked_mut(lane..lane + 24), 8);
    }
    lane += 24;
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

/// f16 RGB → u16 RGBA (alpha `0xFFFF`) (AVX2 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_u16_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
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
  while lane + 24 <= total_lanes {
    let mut buf = [0.0f32; 24];
    unsafe {
      let f0 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane));
      let f1 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane + 8));
      let f2 = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane + 16));
      _mm256_storeu_ps(buf.as_mut_ptr(), f0);
      _mm256_storeu_ps(buf.as_mut_ptr().add(8), f1);
      _mm256_storeu_ps(buf.as_mut_ptr().add(16), f2);
      // Buffer is host-native f32; route via HOST_NATIVE_BE.
      rgbf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
        &buf,
        rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 32),
        8,
      );
    }
    lane += 24;
    pix += 8;
  }
  if pix < width {
    scalar::rgbf16_to_rgba_u16_row::<BE>(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f16 RGB → f32 RGB (lossless widen) (AVX2 + F16C).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [f32]` with
/// `len() >= 3 * width` f32 elements.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 8 <= total_lanes {
    unsafe {
      let f = widen_f16x8_avx::<BE>(rgb_in.as_ptr().add(lane));
      _mm256_storeu_ps(rgb_out.as_mut_ptr().add(lane), f);
    }
    lane += 8;
  }
  // Scalar tail for the last 0-7 lanes.
  #[allow(clippy::needless_range_loop)]
  for i in lane..total_lanes {
    let bits = rgb_in[i].to_bits();
    let h = half::f16::from_bits(if BE {
      u16::from_be(bits)
    } else {
      u16::from_le(bits)
    });
    unsafe {
      *rgb_out.get_unchecked_mut(i) = h.to_f32();
    }
  }
}

/// f16 RGB → f16 RGB lossless pass-through (AVX2 + F16C).
///
/// When `BE = true` the input values are byte-swapped to host-native order.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [half::f16]` with
/// `len() >= 3 * width` f16 elements.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbf16_to_rgb_f16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f16_out row too short");
  scalar::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
}

// ---- Tier 9 — Rgbaf32 AVX2 entry points (4-channel, real alpha) -----------
//
// Thin delegates over the `rgbf32_*` kernels above. The conversions are
// per-element, so:
//   - **real-alpha** kernels feed the flat `R, G, B, A` array through the
//     elementwise RGB sibling in 12-lane chunks (12 = LCM(3,4), so each
//     chunk is a whole number of both RGB-lane and RGBA-pixel groups) and
//     finish the 0–2 pixel tail with the scalar reference.
//   - **drop-alpha** kernels gather the `R, G, B` lanes (skipping α) into a
//     small stack buffer of raw (wire-encoded) `f32`, then run the RGB
//     sibling (which BE-decodes), scalar tail for the remainder.

/// Packed RGBA f32 → packed RGB u8 (drop α).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgb_out.len() >= 3*width`,
/// no aliasing.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbaf32_to_rgb_row<const BE: bool>(
  rgba_in: &[f32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  const CHUNK: usize = 4;
  let mut p = 0usize;
  let mut buf = [0.0f32; CHUNK * 3];
  while p + CHUNK <= width {
    for k in 0..CHUNK {
      let s = (p + k) * 4;
      buf[k * 3] = rgba_in[s];
      buf[k * 3 + 1] = rgba_in[s + 1];
      buf[k * 3 + 2] = rgba_in[s + 2];
    }
    unsafe {
      rgbf32_to_rgb_row::<BE>(
        &buf,
        rgb_out.get_unchecked_mut(p * 3..(p + CHUNK) * 3),
        CHUNK,
      );
    }
    p += CHUNK;
  }
  if p < width {
    scalar::rgbaf32_to_rgb_row::<BE>(
      &rgba_in[p * 4..width * 4],
      &mut rgb_out[p * 3..width * 3],
      width - p,
    );
  }
}

/// Packed RGBA f32 → packed RGBA u8 (real α).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgba_out.len() >= 4*width`,
/// no aliasing.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbaf32_to_rgba_row<const BE: bool>(
  rgba_in: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let total = width * 4;
  let mut lane = 0usize;
  while lane + 12 <= total {
    unsafe {
      rgbf32_to_rgb_row::<BE>(
        rgba_in.get_unchecked(lane..lane + 12),
        rgba_out.get_unchecked_mut(lane..lane + 12),
        4,
      );
    }
    lane += 12;
  }
  let pix_done = lane / 4;
  if pix_done < width {
    scalar::rgbaf32_to_rgba_row::<BE>(
      &rgba_in[pix_done * 4..total],
      &mut rgba_out[pix_done * 4..total],
      width - pix_done,
    );
  }
}

/// Packed RGBA f32 → packed RGB u16 (drop α).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgb_out.len() >= 3*width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbaf32_to_rgb_u16_row<const BE: bool>(
  rgba_in: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  const CHUNK: usize = 4;
  let mut p = 0usize;
  let mut buf = [0.0f32; CHUNK * 3];
  while p + CHUNK <= width {
    for k in 0..CHUNK {
      let s = (p + k) * 4;
      buf[k * 3] = rgba_in[s];
      buf[k * 3 + 1] = rgba_in[s + 1];
      buf[k * 3 + 2] = rgba_in[s + 2];
    }
    unsafe {
      rgbf32_to_rgb_u16_row::<BE>(
        &buf,
        rgb_out.get_unchecked_mut(p * 3..(p + CHUNK) * 3),
        CHUNK,
      );
    }
    p += CHUNK;
  }
  if p < width {
    scalar::rgbaf32_to_rgb_u16_row::<BE>(
      &rgba_in[p * 4..width * 4],
      &mut rgb_out[p * 3..width * 3],
      width - p,
    );
  }
}

/// Packed RGBA f32 → packed RGBA u16 (real α).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgba_out.len() >= 4*width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbaf32_to_rgba_u16_row<const BE: bool>(
  rgba_in: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let total = width * 4;
  let mut lane = 0usize;
  while lane + 12 <= total {
    unsafe {
      rgbf32_to_rgb_u16_row::<BE>(
        rgba_in.get_unchecked(lane..lane + 12),
        rgba_out.get_unchecked_mut(lane..lane + 12),
        4,
      );
    }
    lane += 12;
  }
  let pix_done = lane / 4;
  if pix_done < width {
    scalar::rgbaf32_to_rgba_u16_row::<BE>(
      &rgba_in[pix_done * 4..total],
      &mut rgba_out[pix_done * 4..total],
      width - pix_done,
    );
  }
}

/// Packed RGBA f32 → packed RGB f32 (drop α, lossless).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgb_out.len() >= 3*width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbaf32_to_rgb_f32_row<const BE: bool>(
  rgba_in: &[f32],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");
  const CHUNK: usize = 4;
  let mut p = 0usize;
  let mut buf = [0.0f32; CHUNK * 3];
  while p + CHUNK <= width {
    for k in 0..CHUNK {
      let s = (p + k) * 4;
      buf[k * 3] = rgba_in[s];
      buf[k * 3 + 1] = rgba_in[s + 1];
      buf[k * 3 + 2] = rgba_in[s + 2];
    }
    unsafe {
      rgbf32_to_rgb_f32_row::<BE>(
        &buf,
        rgb_out.get_unchecked_mut(p * 3..(p + CHUNK) * 3),
        CHUNK,
      );
    }
    p += CHUNK;
  }
  if p < width {
    scalar::rgbaf32_to_rgb_f32_row::<BE>(
      &rgba_in[p * 4..width * 4],
      &mut rgb_out[p * 3..width * 3],
      width - p,
    );
  }
}

/// Packed RGBA f32 → packed RGBA f32 (lossless 4-channel pass-through).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgba_out.len() >= 4*width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbaf32_to_rgba_f32_row<const BE: bool>(
  rgba_in: &[f32],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_f32_out row too short");
  // Pure elementwise byte-order normalize — feed the flat 4*width array
  // through the f32 pass-through sibling in 12-lane chunks, scalar tail.
  let total = width * 4;
  let mut lane = 0usize;
  while lane + 12 <= total {
    unsafe {
      rgbf32_to_rgb_f32_row::<BE>(
        rgba_in.get_unchecked(lane..lane + 12),
        rgba_out.get_unchecked_mut(lane..lane + 12),
        4,
      );
    }
    lane += 12;
  }
  let pix_done = lane / 4;
  if pix_done < width {
    scalar::rgbaf32_to_rgba_f32_row::<BE>(
      &rgba_in[pix_done * 4..total],
      &mut rgba_out[pix_done * 4..total],
      width - pix_done,
    );
  }
}

// ---- Tier 9 — Rgbaf16 AVX2 entry points (4-channel, real alpha) -----------
//
// Same delegate strategy as Rgbaf32 but over the `rgbf16_*` widen-then-
// convert siblings (so each carries the `fp16` target feature the sibling
// needs). The lossless `_f16` pass-through delegates straight to scalar
// (no f16 hardware needed for a bit copy), matching the `rgbf16_to_rgb_f16_row`
// sibling.

/// Packed RGBA f16 → packed RGB u8 (drop α).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgb_out.len() >= 3*width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbaf16_to_rgb_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  const CHUNK: usize = 4;
  let mut p = 0usize;
  let mut buf = [half::f16::ZERO; CHUNK * 3];
  while p + CHUNK <= width {
    for k in 0..CHUNK {
      let s = (p + k) * 4;
      buf[k * 3] = rgba_in[s];
      buf[k * 3 + 1] = rgba_in[s + 1];
      buf[k * 3 + 2] = rgba_in[s + 2];
    }
    unsafe {
      rgbf16_to_rgb_row::<BE>(
        &buf,
        rgb_out.get_unchecked_mut(p * 3..(p + CHUNK) * 3),
        CHUNK,
      );
    }
    p += CHUNK;
  }
  if p < width {
    scalar::rgbaf16_to_rgb_row::<BE>(
      &rgba_in[p * 4..width * 4],
      &mut rgb_out[p * 3..width * 3],
      width - p,
    );
  }
}

/// Packed RGBA f16 → packed RGBA u8 (real α).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgba_out.len() >= 4*width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbaf16_to_rgba_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let total = width * 4;
  let mut lane = 0usize;
  while lane + 12 <= total {
    unsafe {
      rgbf16_to_rgb_row::<BE>(
        rgba_in.get_unchecked(lane..lane + 12),
        rgba_out.get_unchecked_mut(lane..lane + 12),
        4,
      );
    }
    lane += 12;
  }
  let pix_done = lane / 4;
  if pix_done < width {
    scalar::rgbaf16_to_rgba_row::<BE>(
      &rgba_in[pix_done * 4..total],
      &mut rgba_out[pix_done * 4..total],
      width - pix_done,
    );
  }
}

/// Packed RGBA f16 → packed RGB u16 (drop α).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgb_out.len() >= 3*width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbaf16_to_rgb_u16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  const CHUNK: usize = 4;
  let mut p = 0usize;
  let mut buf = [half::f16::ZERO; CHUNK * 3];
  while p + CHUNK <= width {
    for k in 0..CHUNK {
      let s = (p + k) * 4;
      buf[k * 3] = rgba_in[s];
      buf[k * 3 + 1] = rgba_in[s + 1];
      buf[k * 3 + 2] = rgba_in[s + 2];
    }
    unsafe {
      rgbf16_to_rgb_u16_row::<BE>(
        &buf,
        rgb_out.get_unchecked_mut(p * 3..(p + CHUNK) * 3),
        CHUNK,
      );
    }
    p += CHUNK;
  }
  if p < width {
    scalar::rgbaf16_to_rgb_u16_row::<BE>(
      &rgba_in[p * 4..width * 4],
      &mut rgb_out[p * 3..width * 3],
      width - p,
    );
  }
}

/// Packed RGBA f16 → packed RGBA u16 (real α).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgba_out.len() >= 4*width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbaf16_to_rgba_u16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let total = width * 4;
  let mut lane = 0usize;
  while lane + 12 <= total {
    unsafe {
      rgbf16_to_rgb_u16_row::<BE>(
        rgba_in.get_unchecked(lane..lane + 12),
        rgba_out.get_unchecked_mut(lane..lane + 12),
        4,
      );
    }
    lane += 12;
  }
  let pix_done = lane / 4;
  if pix_done < width {
    scalar::rgbaf16_to_rgba_u16_row::<BE>(
      &rgba_in[pix_done * 4..total],
      &mut rgba_out[pix_done * 4..total],
      width - pix_done,
    );
  }
}

/// Packed RGBA f16 → packed RGB f32 (drop α, widen).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgb_out.len() >= 3*width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbaf16_to_rgb_f32_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");
  const CHUNK: usize = 4;
  let mut p = 0usize;
  let mut buf = [half::f16::ZERO; CHUNK * 3];
  while p + CHUNK <= width {
    for k in 0..CHUNK {
      let s = (p + k) * 4;
      buf[k * 3] = rgba_in[s];
      buf[k * 3 + 1] = rgba_in[s + 1];
      buf[k * 3 + 2] = rgba_in[s + 2];
    }
    unsafe {
      rgbf16_to_rgb_f32_row::<BE>(
        &buf,
        rgb_out.get_unchecked_mut(p * 3..(p + CHUNK) * 3),
        CHUNK,
      );
    }
    p += CHUNK;
  }
  if p < width {
    scalar::rgbaf16_to_rgb_f32_row::<BE>(
      &rgba_in[p * 4..width * 4],
      &mut rgb_out[p * 3..width * 3],
      width - p,
    );
  }
}

/// Packed RGBA f16 → packed RGBA f32 (widen, 4-channel).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgba_out.len() >= 4*width`.
#[inline]
#[target_feature(enable = "avx2,f16c")]
pub(crate) unsafe fn rgbaf16_to_rgba_f32_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgba_in.len() >= width * 4, "rgbaf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_f32_out row too short");
  // Elementwise widen — feed the flat 4*width array through the f16→f32
  // widen sibling in 12-lane chunks, scalar tail.
  let total = width * 4;
  let mut lane = 0usize;
  while lane + 12 <= total {
    unsafe {
      rgbf16_to_rgb_f32_row::<BE>(
        rgba_in.get_unchecked(lane..lane + 12),
        rgba_out.get_unchecked_mut(lane..lane + 12),
        4,
      );
    }
    lane += 12;
  }
  let pix_done = lane / 4;
  if pix_done < width {
    scalar::rgbaf16_to_rgba_f32_row::<BE>(
      &rgba_in[pix_done * 4..total],
      &mut rgba_out[pix_done * 4..total],
      width - pix_done,
    );
  }
}

/// Packed RGBA f16 → packed RGB f16 (drop α, lossless). Delegates to scalar
/// (a bit copy / byte-swap, no f16 hardware needed — matches the
/// `rgbf16_to_rgb_f16_row` sibling).
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgb_out.len() >= 3*width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbaf16_to_rgb_f16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  scalar::rgbaf16_to_rgb_f16_row::<BE>(rgba_in, rgb_out, width);
}

/// Packed RGBA f16 → packed RGBA f16 (lossless 4-channel). Delegates to
/// scalar.
///
/// # Safety
/// AVX2 available; `rgba_in.len() >= 4*width`, `rgba_out.len() >= 4*width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rgbaf16_to_rgba_f16_row<const BE: bool>(
  rgba_in: &[half::f16],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  scalar::rgbaf16_to_rgba_f16_row::<BE>(rgba_in, rgba_out, width);
}
