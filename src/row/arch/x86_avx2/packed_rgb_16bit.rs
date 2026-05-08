//! AVX2 kernels for 16-bit packed RGB/BGR/RGBA/BGRA sources (Tier 8 finish).
//!
//! ## Format layouts
//!
//! | Format | Elements per pixel | Channel order in memory |
//! |--------|--------------------|------------------------|
//! | Rgb48  | 3 u16              | R, G, B                |
//! | Bgr48  | 3 u16              | B, G, R                |
//! | Rgba64 | 4 u16              | R, G, B, A             |
//! | Bgra64 | 4 u16              | B, G, R, A             |
//!
//! ## Per-format SIMD strategy (16 pixels per outer iteration)
//!
//! ### Rgb48 / Bgr48 (stride-3)
//!
//! 16 pixels = 48 u16 = 96 bytes.  Processed as **two** 8-pixel SSE4.1-style
//! half-iterations (each 24 u16, 3 × 128-bit loads) under the AVX2
//! `target_feature` context.  The deinterleave helper is the same
//! `_mm_shuffle_epi8` + OR approach as the SSE4.1 sibling; both SSE4.1 and
//! SSSE3 are subsets of AVX2 so `_mm_*` intrinsics are freely available.
//!
//! This avoids the much more complex 3 × 256-bit cross-lane shuffle that a
//! single 16-pixel deinterleave would require (the stride-3 pattern does not
//! tile cleanly across two 128-bit lanes of one `__m256i`).  The outer loop
//! still advances 16 pixels per iteration, matching the plan's stated
//! throughput.
//!
//! ### Rgba64 / Bgra64 (stride-4)
//!
//! 16 pixels = 64 u16 = 128 bytes = 4 × `_mm256_loadu_si256`.
//!
//! The deinterleave uses the `_mm256_permute2x128_si256` reshape +
//! 3-level `_mm256_unpacklo/hi_epi16` + `_mm256_unpackhi/lo_epi64` cascade
//! already proven in `xv36.rs` (4-channel variant). The reshape pre-strides
//! the four 256-bit loads so that the per-128-bit-lane unpack cascade
//! lands in natural pixel order — no `_mm256_permute4x64_epi64` lane-fixup
//! permute is needed (an earlier version applied one and produced
//! `[evens; odds]` order). Produces four `__m256i` channel vectors each
//! holding 16 u16 samples in natural pixel order.
//!
//! ## Depth conversion
//!
//! - **u16 → u8:** `_mm256_srli_epi16::<8>` + `_mm256_packus_epi16(v, zero)` +
//!   `_mm256_permute4x64_epi64::<0xD8>` lane fix → 16 u8 in the low 128-bit
//!   half of a 256-bit register.  The low half is then passed to
//!   `write_rgb_16` / `write_rgba_16`.
//! - **u16 → u16:** write 8-pixel halves via the `write_rgb_u16_8` /
//!   `write_rgba_u16_8` helpers (for 3-ch stride-3 path) or extract 128-bit
//!   halves of the 256-bit channel vectors and call the same helpers (for
//!   stride-4 path).
//!
//! ## Scalar tail
//!
//! All kernels handle `width % 16` remaining pixels via the scalar reference.
// Kernels are wired into the dispatcher in the SIMD dispatch task; suppress
// dead_code until then.
#![allow(dead_code)]

use core::arch::x86_64::*;

use super::*;

// =============================================================================
// Rgb48 / Bgr48 helpers — stride-3, 8-pixel deinterleave (SSE4.1 width under
// AVX2 target_feature)
// =============================================================================
//
// Re-use the SSE4.1 byte-shuffle deinterleave pattern.  Each call handles
// 8 pixels (24 u16 = 3 × 128-bit loads), and the outer loop calls it twice
// to process 16 pixels per iteration.
//
// The byte layout for 8 pixels of stride-3 u16:
//
//   v0 = [R0,G0,B0,R1,G1,B1,R2,G2] (u16 positions 0–7)
//   v1 = [B2,R3,G3,B3,R4,G4,B4,R5] (u16 positions 0–7)
//   v2 = [G5,B5,R6,G6,B6,R7,G7,B7] (u16 positions 0–7)
//
// See the SSE4.1 sibling for detailed channel mapping.

/// Deinterleave 8 pixels of stride-3 u16 into (ch0, ch1, ch2) channel vectors.
///
/// For Rgb48: `ch0=R`, `ch1=G`, `ch2=B`.
/// For Bgr48: `ch0=B`, `ch1=G`, `ch2=R`; swap on output.
///
/// # Safety
///
/// Caller must have verified AVX2 availability (SSSE3 / SSE4.1 are subsets).
#[inline(always)]
unsafe fn deinterleave_rgb48_8px(
  v0: __m128i,
  v1: __m128i,
  v2: __m128i,
) -> (__m128i, __m128i, __m128i) {
  unsafe {
    // ch0 (first channel)
    let ch0_v0 = _mm_setr_epi8(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let ch0_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1);
    let ch0_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11);
    let ch0 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, ch0_v0), _mm_shuffle_epi8(v1, ch0_v1)),
      _mm_shuffle_epi8(v2, ch0_v2),
    );

    // ch1 (middle channel — G for both Rgb48 and Bgr48)
    let ch1_v0 = _mm_setr_epi8(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let ch1_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1);
    let ch1_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13);
    let ch1 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, ch1_v0), _mm_shuffle_epi8(v1, ch1_v1)),
      _mm_shuffle_epi8(v2, ch1_v2),
    );

    // ch2 (third channel)
    let ch2_v0 = _mm_setr_epi8(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let ch2_v1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1);
    let ch2_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15);
    let ch2 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, ch2_v0), _mm_shuffle_epi8(v1, ch2_v1)),
      _mm_shuffle_epi8(v2, ch2_v2),
    );

    (ch0, ch1, ch2)
  }
}

// =============================================================================
// Rgba64 / Bgra64 helpers — stride-4, 16-pixel deinterleave (`__m256i`)
// =============================================================================
//
// 16 pixels × 4 u16 channels = 64 u16 = 128 bytes.
//
// Layout after 4 contiguous `_mm256_loadu_si256`:
//
//   raw0  = [C0_0..C3_0, C0_1..C3_1,  C0_2..C3_2,  C0_3..C3_3]   (pixels 0-3)
//   raw1  = [C0_4..C3_4, C0_5..C3_5,  C0_6..C3_6,  C0_7..C3_7]   (pixels 4-7)
//   raw2  = [C0_8..C3_8, C0_9..C3_9,  C0_10..C3_10,C0_11..C3_11] (pixels 8-11)
//   raw3  = [C0_12..C3_12,...,         C0_15..C3_15]               (pixels 12-15)
//
// Goal: separate ch0=[C0_0..C0_15], ch1=[C1_0..C1_15], ... (each in natural order).
//
// Strategy (mirroring xv36.rs):
// 1. Reshape 4 contiguous loads into the strided layout the unpack cascade
//    expects: each register should hold lo=P_n..P_{n+1}, hi=P_{n+8}..P_{n+9}.
//    This is done by cross-lane permute2x128 on pairs of registers.
// 2. Apply 3-level unpack cascade to separate channels.
// 3. Apply 0xD8 permute4x64 on each result to fix lane-split ordering.

/// Reshape 4 contiguous stride-4 loads into the strided layout expected by
/// the 3-level `_mm256_unpack*_epi16` deinterleave cascade.
///
/// Input: raw0=pixels 0-3, raw1=pixels 4-7, raw2=pixels 8-11, raw3=pixels 12-15.
/// Output: r0=pixels (0-1, 8-9), r1=pixels (2-3, 10-11), r2=pixels (4-5, 12-13),
///         r3=pixels (6-7, 14-15).
///
/// `_mm256_permute2x128_si256::<0x20>` selects lo128 of first + lo128 of second.
/// `_mm256_permute2x128_si256::<0x31>` selects hi128 of first + hi128 of second.
///
/// # Safety
///
/// Caller must have verified AVX2 availability.
#[inline(always)]
unsafe fn reshape_rgba64_for_cascade(
  raw0: __m256i,
  raw1: __m256i,
  raw2: __m256i,
  raw3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
  unsafe {
    // r0: lo128 = pixels 0-1 (from raw0 lo), hi128 = pixels 8-9 (from raw2 lo)
    let r0 = _mm256_permute2x128_si256::<0x20>(raw0, raw2);
    // r1: lo128 = pixels 2-3 (from raw0 hi), hi128 = pixels 10-11 (from raw2 hi)
    let r1 = _mm256_permute2x128_si256::<0x31>(raw0, raw2);
    // r2: lo128 = pixels 4-5 (from raw1 lo), hi128 = pixels 12-13 (from raw3 lo)
    let r2 = _mm256_permute2x128_si256::<0x20>(raw1, raw3);
    // r3: lo128 = pixels 6-7 (from raw1 hi), hi128 = pixels 14-15 (from raw3 hi)
    let r3 = _mm256_permute2x128_si256::<0x31>(raw1, raw3);
    (r0, r1, r2, r3)
  }
}

/// 3-level `_mm256_unpacklo/hi_epi16` + `_mm256_unpackhi/lo_epi64` cascade
/// that separates 4 interleaved channels from the reshaped input registers
/// into 4 channel vectors **in natural pixel order** (no fixup permute needed).
///
/// After the cross-lane reshape upstream, the per-lane layout of `r0..r3` is:
///   r0: lo=P0,P1   hi=P8,P9
///   r1: lo=P2,P3   hi=P10,P11
///   r2: lo=P4,P5   hi=P12,P13
///   r3: lo=P6,P7   hi=P14,P15
/// where each P_n is a 4-channel pixel: `[C0_n, C1_n, C2_n, C3_n]`.
///
/// Cascade (mirrors `xv36.rs` AVX2 deinterleave precedent):
///
/// Level 1 — `unpacklo/hi_epi16` of `(r0,r1)` and `(r2,r3)` separately
///   per 128-bit lane. Within each lane:
///   `s1_lo = [C0_a,C0_{a+1}, C1_a,C1_{a+1}, C2_a,C2_{a+1}, C3_a,C3_{a+1}]`
///   etc. for `(a,a+1) = (0,2)/(8,10)` (s1 pair from r0/r1) and
///   `(4,6)/(12,14)` (s2 pair from r2/r3); `s1_hi` / `s2_hi` carry the
///   adjacent odd pixels (1/3, 5/7, 9/11, 13/15).
///
/// Level 2 — `unpacklo/hi_epi16` of `(s1_lo, s1_hi)` and `(s2_lo, s2_hi)`
///   per lane interleaves the even/odd halves so each 128-bit lane holds
///   four consecutive pixels of two channels:
///   `s3_lo lo lane = [C0_0..C0_3, C1_0..C1_3]`,
///   `s3_lo hi lane = [C0_8..C0_11, C1_8..C1_11]`,
///   `s4_lo lo lane = [C0_4..C0_7, C1_4..C1_7]`,
///   `s4_lo hi lane = [C0_12..C0_15, C1_12..C1_15]`,
///   and similarly `s3_hi`/`s4_hi` carry channels 2/3.
///
/// Level 3 — `unpacklo/hi_epi64` of `(s3_lo, s4_lo)` and `(s3_hi, s4_hi)`
///   per lane concatenates the two 64-bit halves so each output channel
///   vector holds 8 consecutive pixels per 128-bit lane in natural order:
///   `ch0 lo = [C0_0..C0_7]`, `ch0 hi = [C0_8..C0_15]`, etc.
///
/// Because the upstream reshape pre-strided the inputs (lo=P_n,P_{n+1};
/// hi=P_{n+8},P_{n+9}), no `_mm256_permute4x64_epi64::<0xD8>` lane fixup is
/// needed — the cascade output already lands in natural order. (An earlier
/// version of this kernel applied that permute and produced
/// `[evens; odds]` order, which the current `*_matches_scalar_width17`
/// regression tests catch.)
///
/// # Safety
///
/// Caller must have verified AVX2 availability.
#[inline(always)]
unsafe fn deinterleave_rgba64_cascade(
  r0: __m256i,
  r1: __m256i,
  r2: __m256i,
  r3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
  unsafe {
    // Level 1: pair r0/r1 (yielding the (0,2,8,10) / (1,3,9,11) pixel
    // halves in lo/hi lanes) and r2/r3 (yielding (4,6,12,14) / (5,7,13,15)).
    let s1_lo = _mm256_unpacklo_epi16(r0, r1);
    let s1_hi = _mm256_unpackhi_epi16(r0, r1);
    let s2_lo = _mm256_unpacklo_epi16(r2, r3);
    let s2_hi = _mm256_unpackhi_epi16(r2, r3);

    // Level 2: interleave the even/odd halves within each pair to assemble
    // four consecutive pixels per 128-bit lane (channels 0/1 in s3_lo/s4_lo,
    // channels 2/3 in s3_hi/s4_hi).
    let s3_lo = _mm256_unpacklo_epi16(s1_lo, s1_hi);
    let s3_hi = _mm256_unpackhi_epi16(s1_lo, s1_hi);
    let s4_lo = _mm256_unpacklo_epi16(s2_lo, s2_hi);
    let s4_hi = _mm256_unpackhi_epi16(s2_lo, s2_hi);

    // Level 3: concatenate the 64-bit halves so each lane holds 8
    // consecutive pixels of one channel in natural order.
    let ch0 = _mm256_unpacklo_epi64(s3_lo, s4_lo);
    let ch1 = _mm256_unpackhi_epi64(s3_lo, s4_lo);
    let ch2 = _mm256_unpacklo_epi64(s3_hi, s4_hi);
    let ch3 = _mm256_unpackhi_epi64(s3_hi, s4_hi);

    (ch0, ch1, ch2, ch3)
  }
}

/// Deinterleave 16 pixels of stride-4 u16 (Rgba64 or Bgra64) from four
/// `__m256i` loads into four separate u16×16 channel vectors in natural order.
///
/// Returns `(ch0, ch1, ch2, ch3)` in memory order.
/// For Rgba64: `(R, G, B, A)`. For Bgra64: `(B, G, R, A)`.
///
/// # Safety
///
/// Caller must have verified AVX2 availability.
#[inline(always)]
unsafe fn deinterleave_rgba64_16px(
  raw0: __m256i,
  raw1: __m256i,
  raw2: __m256i,
  raw3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
  unsafe {
    let (r0, r1, r2, r3) = reshape_rgba64_for_cascade(raw0, raw1, raw2, raw3);
    deinterleave_rgba64_cascade(r0, r1, r2, r3)
  }
}

// =============================================================================
// u16 → u8 narrowing for __m256i: `>> 8` + `packus_epi16` + lane fix
// =============================================================================

/// Narrow a u16×16 vector to u8×16 (in the low 128-bit half) via logical
/// right-shift by 8, then `packus_epi16`, then `permute4x64_epi64::<0xD8>`.
///
/// Equivalent to scalar `(v >> 8) as u8`.
#[inline(always)]
unsafe fn narrow_u16x16_to_u8x16(v: __m256i, zero: __m256i) -> __m128i {
  unsafe {
    let shifted = _mm256_srli_epi16::<8>(v);
    let packed = _mm256_packus_epi16(shifted, zero);
    // Fix AVX2 lane-split: low lane bytes in [0,7] and [8,15], hi lane in [16,23] and [24,31]
    // → after 0xD8: natural u8 order in low 128 bits.
    _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0xD8>(packed))
  }
}

// ---- endian byte-swap helpers -----------------------------------------------

/// Compile-time host endianness. `true` on BE targets, `false` on LE.
///
/// Used by the byte-swap helpers below to gate the swap on
/// `BE != HOST_NATIVE_BE`, covering all four `wire × host` quadrants. Mirrors
/// the gate established in `gray.rs` and the canonical NEON
/// `bswap_u16x8_if_be` helper.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Conditionally byte-swap every u16 lane in a `__m128i` so the returned
/// value is in **host-native** byte order regardless of the host endianness.
///
/// The gate is `BE != HOST_NATIVE_BE` — see [`byteswap256_if_be`] for the
/// full truth table. Uses `_mm_shuffle_epi8` (SSSE3 subset of AVX2).
#[inline(always)]
unsafe fn byteswap128_if_be<const BE: bool>(v: __m128i) -> __m128i {
  if BE != HOST_NATIVE_BE {
    const MASK: __m128i =
      unsafe { core::mem::transmute([1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]) };
    unsafe { _mm_shuffle_epi8(v, MASK) }
  } else {
    v
  }
}

/// Conditionally byte-swap every u16 lane in a `__m256i` so the returned
/// value is in **host-native** byte order regardless of the host endianness.
///
/// The gate is `BE != HOST_NATIVE_BE`:
///
/// | wire `BE` | host | gate    | action            |
/// |-----------|------|---------|-------------------|
/// | `false`   | LE   | `false` | no swap (LE→LE)   |
/// | `false`   | BE   | `true`  | swap (LE→BE)      |
/// | `true`    | LE   | `true`  | swap (BE→LE)      |
/// | `true`    | BE   | `false` | no swap (BE→BE)   |
///
/// Uses `_mm256_shuffle_epi8` (AVX2). The unused branch folds at compile
/// time since both `BE` and `HOST_NATIVE_BE` are constants.
#[inline(always)]
unsafe fn byteswap256_if_be<const BE: bool>(v: __m256i) -> __m256i {
  if BE != HOST_NATIVE_BE {
    // Same u16-lane byte-swap mask, broadcast to both 128-bit lanes.
    const MASK: __m256i = unsafe {
      core::mem::transmute([
        1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11,
        10, 13, 12, 15, 14,
      ])
    };
    unsafe { _mm256_shuffle_epi8(v, MASK) }
  } else {
    v
  }
}

// =============================================================================
// Rgb48 (R, G, B — 3 u16 elements per pixel)
// =============================================================================

/// AVX2 Rgb48 → packed u8 RGB. 16 pixels per outer iteration.
///
/// Processes two 8-pixel halves (3 × 128-bit loads each) under the AVX2
/// target_feature, exploiting that SSE4.1/SSSE3 are AVX2 subsets. Each half
/// deinterleaves with shuffle masks, narrows via `>> 8`, writes 8 pixels
/// (24 bytes). 16 pixels are produced per outer loop iteration.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available (caller obligation).
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_rgb48_to_rgb_row<const BE: bool>(
  rgb48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    // Process 16 pixels per outer iteration (2 × 8-pixel halves).
    while x + 16 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);

      // First half: pixels x..x+7
      let v0 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (r0, g0, b0) = deinterleave_rgb48_8px(v0, v1, v2);
      let r0u8 = narrow_u16x8_to_u8x8(r0, zero);
      let g0u8 = narrow_u16x8_to_u8x8(g0, zero);
      let b0u8 = narrow_u16x8_to_u8x8(b0, zero);
      let mut tmp0 = [0u8; 48];
      write_rgb_16(r0u8, g0u8, b0u8, tmp0.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp0.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);

      // Second half: pixels x+8..x+15
      let ptr8 = ptr.add(24); // 24 u16 ahead = 8 pixels × 3 channels
      let v3 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.cast()));
      let v4 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(8).cast()));
      let v5 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(16).cast()));
      let (r1, g1, b1) = deinterleave_rgb48_8px(v3, v4, v5);
      let r1u8 = narrow_u16x8_to_u8x8(r1, zero);
      let g1u8 = narrow_u16x8_to_u8x8(g1, zero);
      let b1u8 = narrow_u16x8_to_u8x8(b1, zero);
      let mut tmp1 = [0u8; 48];
      write_rgb_16(r1u8, g1u8, b1u8, tmp1.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp1.as_ptr(), rgb_out.as_mut_ptr().add((x + 8) * 3), 24);

      x += 16;
    }
    // Handle remaining pixels (< 16) via scalar fallback.
    if x < width {
      scalar::rgb48_to_rgb_row::<BE>(&rgb48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 Rgb48 → packed u8 RGBA. 16 pixels per outer iteration. Alpha forced to 0xFF.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_rgb48_to_rgba_row<const BE: bool>(
  rgb48: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let opaque_u16 = _mm_set1_epi16(0x00FFu16 as i16);
    let opaque_u8 = _mm_packus_epi16(opaque_u16, zero);
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);

      let v0 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (r0, g0, b0) = deinterleave_rgb48_8px(v0, v1, v2);
      let r0u8 = narrow_u16x8_to_u8x8(r0, zero);
      let g0u8 = narrow_u16x8_to_u8x8(g0, zero);
      let b0u8 = narrow_u16x8_to_u8x8(b0, zero);
      let mut tmp0 = [0u8; 64];
      write_rgba_16(r0u8, g0u8, b0u8, opaque_u8, tmp0.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp0.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);

      let ptr8 = ptr.add(24);
      let v3 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.cast()));
      let v4 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(8).cast()));
      let v5 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(16).cast()));
      let (r1, g1, b1) = deinterleave_rgb48_8px(v3, v4, v5);
      let r1u8 = narrow_u16x8_to_u8x8(r1, zero);
      let g1u8 = narrow_u16x8_to_u8x8(g1, zero);
      let b1u8 = narrow_u16x8_to_u8x8(b1, zero);
      let mut tmp1 = [0u8; 64];
      write_rgba_16(r1u8, g1u8, b1u8, opaque_u8, tmp1.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp1.as_ptr(), rgba_out.as_mut_ptr().add((x + 8) * 4), 32);

      x += 16;
    }
    if x < width {
      scalar::rgb48_to_rgba_row::<BE>(&rgb48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 Rgb48 → native-depth u16 RGB (identity repack). 16 pixels per iteration.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_rgb48_to_rgb_u16_row<const BE: bool>(
  rgb48: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);

      let v0 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (r0, g0, b0) = deinterleave_rgb48_8px(v0, v1, v2);
      write_rgb_u16_8(r0, g0, b0, rgb_out.as_mut_ptr().add(x * 3));

      let ptr8 = ptr.add(24);
      let v3 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.cast()));
      let v4 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(8).cast()));
      let v5 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(16).cast()));
      let (r1, g1, b1) = deinterleave_rgb48_8px(v3, v4, v5);
      write_rgb_u16_8(r1, g1, b1, rgb_out.as_mut_ptr().add((x + 8) * 3));

      x += 16;
    }
    if x < width {
      scalar::rgb48_to_rgb_u16_row::<BE>(&rgb48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 Rgb48 → native-depth u16 RGBA. 16 pixels per iteration. Alpha forced to 0xFFFF.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_rgb48_to_rgba_u16_row<const BE: bool>(
  rgb48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = _mm_set1_epi16(0xFFFFu16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);

      let v0 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (r0, g0, b0) = deinterleave_rgb48_8px(v0, v1, v2);
      write_rgba_u16_8(r0, g0, b0, opaque, rgba_out.as_mut_ptr().add(x * 4));

      let ptr8 = ptr.add(24);
      let v3 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.cast()));
      let v4 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(8).cast()));
      let v5 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(16).cast()));
      let (r1, g1, b1) = deinterleave_rgb48_8px(v3, v4, v5);
      write_rgba_u16_8(r1, g1, b1, opaque, rgba_out.as_mut_ptr().add((x + 8) * 4));

      x += 16;
    }
    if x < width {
      scalar::rgb48_to_rgba_u16_row::<BE>(&rgb48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Bgr48 (B, G, R — 3 u16 elements per pixel)
// =============================================================================

/// AVX2 Bgr48 → packed u8 RGB. 16 pixels per outer iteration.
///
/// `deinterleave_rgb48_8px` yields `(B, G, R)` in source memory order;
/// the B↔R swap is applied by passing them as `(R=ch2, G=ch1, B=ch0)`.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_bgr48_to_rgb_row<const BE: bool>(
  bgr48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);

      let v0 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (b0, g0, r0) = deinterleave_rgb48_8px(v0, v1, v2);
      let r0u8 = narrow_u16x8_to_u8x8(r0, zero);
      let g0u8 = narrow_u16x8_to_u8x8(g0, zero);
      let b0u8 = narrow_u16x8_to_u8x8(b0, zero);
      let mut tmp0 = [0u8; 48];
      write_rgb_16(r0u8, g0u8, b0u8, tmp0.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp0.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);

      let ptr8 = ptr.add(24);
      let v3 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.cast()));
      let v4 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(8).cast()));
      let v5 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(16).cast()));
      let (b1, g1, r1) = deinterleave_rgb48_8px(v3, v4, v5);
      let r1u8 = narrow_u16x8_to_u8x8(r1, zero);
      let g1u8 = narrow_u16x8_to_u8x8(g1, zero);
      let b1u8 = narrow_u16x8_to_u8x8(b1, zero);
      let mut tmp1 = [0u8; 48];
      write_rgb_16(r1u8, g1u8, b1u8, tmp1.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp1.as_ptr(), rgb_out.as_mut_ptr().add((x + 8) * 3), 24);

      x += 16;
    }
    if x < width {
      scalar::bgr48_to_rgb_row::<BE>(&bgr48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 Bgr48 → packed u8 RGBA. 16 pixels per outer iteration.
/// B↔R swap; alpha forced to 0xFF.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_bgr48_to_rgba_row<const BE: bool>(
  bgr48: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let opaque_u16 = _mm_set1_epi16(0x00FFu16 as i16);
    let opaque_u8 = _mm_packus_epi16(opaque_u16, zero);
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);

      let v0 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (b0, g0, r0) = deinterleave_rgb48_8px(v0, v1, v2);
      let r0u8 = narrow_u16x8_to_u8x8(r0, zero);
      let g0u8 = narrow_u16x8_to_u8x8(g0, zero);
      let b0u8 = narrow_u16x8_to_u8x8(b0, zero);
      let mut tmp0 = [0u8; 64];
      write_rgba_16(r0u8, g0u8, b0u8, opaque_u8, tmp0.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp0.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);

      let ptr8 = ptr.add(24);
      let v3 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.cast()));
      let v4 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(8).cast()));
      let v5 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(16).cast()));
      let (b1, g1, r1) = deinterleave_rgb48_8px(v3, v4, v5);
      let r1u8 = narrow_u16x8_to_u8x8(r1, zero);
      let g1u8 = narrow_u16x8_to_u8x8(g1, zero);
      let b1u8 = narrow_u16x8_to_u8x8(b1, zero);
      let mut tmp1 = [0u8; 64];
      write_rgba_16(r1u8, g1u8, b1u8, opaque_u8, tmp1.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp1.as_ptr(), rgba_out.as_mut_ptr().add((x + 8) * 4), 32);

      x += 16;
    }
    if x < width {
      scalar::bgr48_to_rgba_row::<BE>(&bgr48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 Bgr48 → native-depth u16 RGB. 16 pixels per outer iteration.
/// B↔R swap; values unchanged.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_bgr48_to_rgb_u16_row<const BE: bool>(
  bgr48: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);

      let v0 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (b0, g0, r0) = deinterleave_rgb48_8px(v0, v1, v2);
      write_rgb_u16_8(r0, g0, b0, rgb_out.as_mut_ptr().add(x * 3));

      let ptr8 = ptr.add(24);
      let v3 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.cast()));
      let v4 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(8).cast()));
      let v5 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(16).cast()));
      let (b1, g1, r1) = deinterleave_rgb48_8px(v3, v4, v5);
      write_rgb_u16_8(r1, g1, b1, rgb_out.as_mut_ptr().add((x + 8) * 3));

      x += 16;
    }
    if x < width {
      scalar::bgr48_to_rgb_u16_row::<BE>(&bgr48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 Bgr48 → native-depth u16 RGBA. 16 pixels per outer iteration.
/// B↔R swap; alpha forced to 0xFFFF.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_bgr48_to_rgba_u16_row<const BE: bool>(
  bgr48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = _mm_set1_epi16(0xFFFFu16 as i16);
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);

      let v0 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (b0, g0, r0) = deinterleave_rgb48_8px(v0, v1, v2);
      write_rgba_u16_8(r0, g0, b0, opaque, rgba_out.as_mut_ptr().add(x * 4));

      let ptr8 = ptr.add(24);
      let v3 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.cast()));
      let v4 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(8).cast()));
      let v5 = byteswap128_if_be::<BE>(_mm_loadu_si128(ptr8.add(16).cast()));
      let (b1, g1, r1) = deinterleave_rgb48_8px(v3, v4, v5);
      write_rgba_u16_8(r1, g1, b1, opaque, rgba_out.as_mut_ptr().add((x + 8) * 4));

      x += 16;
    }
    if x < width {
      scalar::bgr48_to_rgba_u16_row::<BE>(&bgr48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Rgba64 (R, G, B, A — 4 u16 elements per pixel)
// =============================================================================

/// AVX2 Rgba64 → packed u8 RGB. 16 pixels per SIMD iteration. Alpha discarded.
///
/// Loads 4 × `__m256i` (64 u16 = 16 pixels), deinterleaves via the
/// cascade helper, narrows via `>> 8` + `packus_epi16` + lane fix, writes
/// 16 pixels (48 bytes) via `write_rgb_16` on the low 128 bits.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_rgba64_to_rgb_row<const BE: bool>(
  rgba64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.cast()));
      let raw1 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(16).cast()));
      let raw2 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(32).cast()));
      let raw3 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(48).cast()));
      let (r_u16, g_u16, b_u16, _a) = deinterleave_rgba64_16px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x16_to_u8x16(r_u16, zero256);
      let g_u8 = narrow_u16x16_to_u8x16(g_u16, zero256);
      let b_u8 = narrow_u16x16_to_u8x16(b_u16, zero256);
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::rgba64_to_rgb_row::<BE>(&rgba64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 Rgba64 → packed u8 RGBA. 16 pixels per SIMD iteration.
/// Source alpha passes through (narrowed via `>> 8`).
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_rgba64_to_rgba_row<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.cast()));
      let raw1 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(16).cast()));
      let raw2 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(32).cast()));
      let raw3 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(48).cast()));
      let (r_u16, g_u16, b_u16, a_u16) = deinterleave_rgba64_16px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x16_to_u8x16(r_u16, zero256);
      let g_u8 = narrow_u16x16_to_u8x16(g_u16, zero256);
      let b_u8 = narrow_u16x16_to_u8x16(b_u16, zero256);
      let a_u8 = narrow_u16x16_to_u8x16(a_u16, zero256);
      write_rgba_16(r_u8, g_u8, b_u8, a_u8, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::rgba64_to_rgba_row::<BE>(&rgba64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 Rgba64 → native-depth u16 RGB. 16 pixels per SIMD iteration. Alpha discarded.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_rgba64_to_rgb_u16_row<const BE: bool>(
  rgba64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.cast()));
      let raw1 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(16).cast()));
      let raw2 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(32).cast()));
      let raw3 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(48).cast()));
      let (r_u16, g_u16, b_u16, _a) = deinterleave_rgba64_16px(raw0, raw1, raw2, raw3);
      // Write in two 8-pixel halves using the existing 128-bit helper.
      write_rgb_u16_8(
        _mm256_castsi256_si128(r_u16),
        _mm256_castsi256_si128(g_u16),
        _mm256_castsi256_si128(b_u16),
        rgb_out.as_mut_ptr().add(x * 3),
      );
      write_rgb_u16_8(
        _mm256_extracti128_si256::<1>(r_u16),
        _mm256_extracti128_si256::<1>(g_u16),
        _mm256_extracti128_si256::<1>(b_u16),
        rgb_out.as_mut_ptr().add(x * 3 + 24),
      );
      x += 16;
    }
    if x < width {
      scalar::rgba64_to_rgb_u16_row::<BE>(&rgba64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 Rgba64 → native-depth u16 RGBA (identity copy). 16 pixels per iteration.
/// Source alpha preserved.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_rgba64_to_rgba_u16_row<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.cast()));
      let raw1 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(16).cast()));
      let raw2 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(32).cast()));
      let raw3 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(48).cast()));
      let (r_u16, g_u16, b_u16, a_u16) = deinterleave_rgba64_16px(raw0, raw1, raw2, raw3);
      write_rgba_u16_8(
        _mm256_castsi256_si128(r_u16),
        _mm256_castsi256_si128(g_u16),
        _mm256_castsi256_si128(b_u16),
        _mm256_castsi256_si128(a_u16),
        rgba_out.as_mut_ptr().add(x * 4),
      );
      write_rgba_u16_8(
        _mm256_extracti128_si256::<1>(r_u16),
        _mm256_extracti128_si256::<1>(g_u16),
        _mm256_extracti128_si256::<1>(b_u16),
        _mm256_extracti128_si256::<1>(a_u16),
        rgba_out.as_mut_ptr().add(x * 4 + 32),
      );
      x += 16;
    }
    if x < width {
      scalar::rgba64_to_rgba_u16_row::<BE>(&rgba64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Bgra64 (B, G, R, A — 4 u16 elements per pixel)
// =============================================================================

/// AVX2 Bgra64 → packed u8 RGB. 16 pixels per SIMD iteration.
/// B↔R swap; alpha discarded.
///
/// `deinterleave_rgba64_16px` yields `(B, G, R, A)` in source memory order.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_bgra64_to_rgb_row<const BE: bool>(
  bgra64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.cast()));
      let raw1 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(16).cast()));
      let raw2 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(32).cast()));
      let raw3 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(48).cast()));
      // ch0=B, ch1=G, ch2=R, ch3=A (source BGRA order)
      let (b_u16, g_u16, r_u16, _a) = deinterleave_rgba64_16px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x16_to_u8x16(r_u16, zero256);
      let g_u8 = narrow_u16x16_to_u8x16(g_u16, zero256);
      let b_u8 = narrow_u16x16_to_u8x16(b_u16, zero256);
      write_rgb_16(r_u8, g_u8, b_u8, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::bgra64_to_rgb_row::<BE>(&bgra64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 Bgra64 → packed u8 RGBA. 16 pixels per SIMD iteration.
/// B↔R swap; source alpha passes through (narrowed via `>> 8`).
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_bgra64_to_rgba_row<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero256 = _mm256_setzero_si256();
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.cast()));
      let raw1 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(16).cast()));
      let raw2 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(32).cast()));
      let raw3 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(48).cast()));
      let (b_u16, g_u16, r_u16, a_u16) = deinterleave_rgba64_16px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x16_to_u8x16(r_u16, zero256);
      let g_u8 = narrow_u16x16_to_u8x16(g_u16, zero256);
      let b_u8 = narrow_u16x16_to_u8x16(b_u16, zero256);
      let a_u8 = narrow_u16x16_to_u8x16(a_u16, zero256);
      write_rgba_16(r_u8, g_u8, b_u8, a_u8, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::bgra64_to_rgba_row::<BE>(&bgra64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX2 Bgra64 → native-depth u16 RGB. 16 pixels per SIMD iteration.
/// B↔R swap; alpha discarded.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_bgra64_to_rgb_u16_row<const BE: bool>(
  bgra64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.cast()));
      let raw1 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(16).cast()));
      let raw2 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(32).cast()));
      let raw3 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(48).cast()));
      let (b_u16, g_u16, r_u16, _a) = deinterleave_rgba64_16px(raw0, raw1, raw2, raw3);
      // Swap B↔R: store (R, G, B)
      write_rgb_u16_8(
        _mm256_castsi256_si128(r_u16),
        _mm256_castsi256_si128(g_u16),
        _mm256_castsi256_si128(b_u16),
        rgb_out.as_mut_ptr().add(x * 3),
      );
      write_rgb_u16_8(
        _mm256_extracti128_si256::<1>(r_u16),
        _mm256_extracti128_si256::<1>(g_u16),
        _mm256_extracti128_si256::<1>(b_u16),
        rgb_out.as_mut_ptr().add(x * 3 + 24),
      );
      x += 16;
    }
    if x < width {
      scalar::bgra64_to_rgb_u16_row::<BE>(&bgra64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX2 Bgra64 → native-depth u16 RGBA. 16 pixels per SIMD iteration.
/// B↔R swap; source alpha preserved at position 3.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. AVX2 must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_bgra64_to_rgba_u16_row<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.cast()));
      let raw1 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(16).cast()));
      let raw2 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(32).cast()));
      let raw3 = byteswap256_if_be::<BE>(_mm256_loadu_si256(ptr.add(48).cast()));
      // Swap B↔R: (R=ch2, G=ch1, B=ch0, A=ch3)
      let (b_u16, g_u16, r_u16, a_u16) = deinterleave_rgba64_16px(raw0, raw1, raw2, raw3);
      write_rgba_u16_8(
        _mm256_castsi256_si128(r_u16),
        _mm256_castsi256_si128(g_u16),
        _mm256_castsi256_si128(b_u16),
        _mm256_castsi256_si128(a_u16),
        rgba_out.as_mut_ptr().add(x * 4),
      );
      write_rgba_u16_8(
        _mm256_extracti128_si256::<1>(r_u16),
        _mm256_extracti128_si256::<1>(g_u16),
        _mm256_extracti128_si256::<1>(b_u16),
        _mm256_extracti128_si256::<1>(a_u16),
        rgba_out.as_mut_ptr().add(x * 4 + 32),
      );
      x += 16;
    }
    if x < width {
      scalar::bgra64_to_rgba_u16_row::<BE>(&bgra64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Helper: narrow u16×8 (128-bit) to u8×8 (used by stride-3 paths)
// =============================================================================

/// Narrow a u16×8 vector to u8×8 (in the low half) via logical right-shift by 8.
///
/// Equivalent to scalar `(v >> 8) as u8`. Zero-packs the high half.
#[inline(always)]
unsafe fn narrow_u16x8_to_u8x8(v: __m128i, zero: __m128i) -> __m128i {
  unsafe { _mm_packus_epi16(_mm_srli_epi16::<8>(v), zero) }
}
