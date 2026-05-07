//! AVX-512 (F + BW) kernels for 16-bit packed RGB/BGR/RGBA/BGRA sources
//! (Tier 8 finish).
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
//! ## Per-format SIMD strategy (32 pixels per outer iteration)
//!
//! ### Rgb48 / Bgr48 (stride-3)
//!
//! 32 pixels = 96 u16 = 192 bytes. Processed as **four** 8-pixel SSE4.1-style
//! half-iterations (each 24 u16, 3 × 128-bit loads) under the AVX-512
//! `target_feature` context. SSE4.1 and SSSE3 are subsets of AVX-512 so
//! `_mm_*` intrinsics are freely available. This avoids complex stride-3
//! cross-lane permutes in 512-bit registers that do not tile cleanly.
//!
//! ### Rgba64 / Bgra64 (stride-4)
//!
//! 32 pixels = 128 u16 = 256 bytes = 4 × `_mm512_loadu_si512`.
//!
//! The deinterleave uses a 3-level `_mm512_unpacklo/hi_epi16` cascade mirroring
//! the AVX2 sibling (xv36.rs pattern), followed by `_mm512_permutexvar_epi64`
//! lane-cross fixup. Produces four `__m512i` channel vectors each holding 32
//! u16 samples in natural order.
//!
//! ## Depth conversion
//!
//! - **u16 → u8:** `_mm512_srli_epi16::<8>` + `_mm512_cvtusepi16_epi8`
//!   (saturating narrow to u8x32). The 256-bit result is stored via a 256-bit
//!   unaligned store.
//! - **u16 → u16:** write 8-pixel chunks via `write_rgb_u16_8` / `write_rgba_u16_8`
//!   (for stride-3 path) or via the `write_rgb_u16_32` / `write_rgba_u16_32`
//!   helpers from the AVX-512 mod (for stride-4 path).
//!
//! ## Scalar tail
//!
//! All kernels handle `width % 32` remaining pixels via the scalar reference.
// Kernels are wired into the dispatcher in the dispatch-wiring step; suppress
// dead_code until then.
#![allow(dead_code)]

use core::arch::x86_64::*;

use super::*;

// =============================================================================
// Shared deinterleave helper (stride-3, 8-pixel, SSE4.1-level — same masks as
// the AVX2 and SSE4.1 siblings)
// =============================================================================

/// Deinterleave 8 pixels of stride-3 u16 from three `__m128i` loads into
/// `(ch0, ch1, ch2)` channel vectors, each holding 8 u16 values.
///
/// For Rgb48: `ch0=R`, `ch1=G`, `ch2=B`.
/// For Bgr48: `ch0=B`, `ch1=G`, `ch2=R`; caller swaps on output.
///
/// # Safety
///
/// Caller must hold AVX-512F + AVX-512BW `target_feature` (SSSE3/SSE4.1 are
/// subsets).
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
// Rgba64 / Bgra64 helpers — stride-4, 32-pixel deinterleave (__m512i)
// =============================================================================
//
// 32 pixels × 4 u16 channels = 128 u16 = 256 bytes.
//
// Layout after 4 contiguous `_mm512_loadu_si512` (each 32 u16 = 8 pixels):
//
//   raw0 = [C0_0..C3_0, C0_1..C3_1, ..., C0_7..C3_7]    (pixels 0-7)
//   raw1 = [C0_8..C3_8, ...,             C0_15..C3_15]   (pixels 8-15)
//   raw2 = [C0_16..C3_16, ...,           C0_23..C3_23]   (pixels 16-23)
//   raw3 = [C0_24..C3_24, ...,           C0_31..C3_31]   (pixels 24-31)
//
// Goal: ch0=[C0_0..C0_31], ch1=[C1_0..C1_31], ... in natural pixel order.
//
// Strategy (mirrors the AVX2 / xv36.rs cascade, scaled to 512-bit):
// 1. Reshape 4 contiguous loads into the strided layout via
//    `_mm512_permutex2var_epi64` cross-lane permutes.
// 2. Apply 3-level unpack cascade to separate channels.
// 3. Apply 0xD8-analog permute via `_mm512_permutexvar_epi64` with the
//    pack_fixup index to restore natural element order.

/// Reshape 4 contiguous stride-4 512-bit loads into the strided layout
/// expected by the 3-level `_mm512_unpack*_epi16` cascade.
///
/// Input:  raw0=pixels 0-7, raw1=pixels 8-15, raw2=pixels 16-23, raw3=pixels 24-31.
/// Output: r0=pixels (0-1, 16-17), r1=pixels (2-3, 18-19), r2=pixels (4-5, 20-21),
///         r3=pixels (6-7, 22-23), r4=pixels (8-9, 24-25), r5=pixels (10-11, 26-27),
///         r6=pixels (12-13, 28-29), r7=pixels (14-15, 30-31).
///
/// Uses `_mm512_permutex2var_epi64` to cross-mix 64-bit chunks between pairs.
///
/// # Safety
///
/// Caller must hold AVX-512F + AVX-512BW `target_feature`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn reshape_rgba64_32px_for_cascade(
  raw0: __m512i,
  raw1: __m512i,
  raw2: __m512i,
  raw3: __m512i,
) -> (
  __m512i,
  __m512i,
  __m512i,
  __m512i,
  __m512i,
  __m512i,
  __m512i,
  __m512i,
) {
  unsafe {
    // Each __m512i contains 8 pixels × 4 u16 = 32 u16 = 8 × 64-bit chunks.
    // We want to interleave pairs from raw0 and raw2 (distance-16 partners):
    //   r0: chunks [0,1] from raw0, chunks [0,1] from raw2  → pixels (0-1, 16-17)
    //   r1: chunks [2,3] from raw0, chunks [2,3] from raw2  → pixels (2-3, 18-19)
    //   r2: chunks [4,5] from raw0, chunks [4,5] from raw2  → pixels (4-5, 20-21)
    //   r3: chunks [6,7] from raw0, chunks [6,7] from raw2  → pixels (6-7, 22-23)
    // Similarly for raw1 and raw3:
    //   r4..r7: pixels (8-9,24-25), (10-11,26-27), (12-13,28-29), (14-15,30-31)

    // permutex2var indices for "lo 256 from a, lo 256 from b":
    let idx_lo = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    // "hi 256 from a, hi 256 from b":
    let idx_hi = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);

    let r01 = _mm512_permutex2var_epi64(raw0, idx_lo, raw2);
    let r23 = _mm512_permutex2var_epi64(raw0, idx_hi, raw2);
    let r45 = _mm512_permutex2var_epi64(raw1, idx_lo, raw3);
    let r67 = _mm512_permutex2var_epi64(raw1, idx_hi, raw3);

    // Now split each 512-bit register into two 256-bit-equivalent 512-bit halves
    // by extracting / combining 64-bit chunks:
    //   r0: lo256 of r01 → pixels (0-1, 16-17) in one 512-bit reg (chunks 0,1,4,5 → repack)
    //   r1: hi256 of r01 → pixels (2-3, 18-19)
    //   etc.
    // Simpler: use permutex2var again to split lo/hi 256 of each r-pair into separate regs.
    let idx_even = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    let idx_odd = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

    let r0 = _mm512_permutex2var_epi64(r01, idx_even, r01);
    let r1 = _mm512_permutex2var_epi64(r01, idx_odd, r01);
    let r2 = _mm512_permutex2var_epi64(r23, idx_even, r23);
    let r3 = _mm512_permutex2var_epi64(r23, idx_odd, r23);
    let r4 = _mm512_permutex2var_epi64(r45, idx_even, r45);
    let r5 = _mm512_permutex2var_epi64(r45, idx_odd, r45);
    let r6 = _mm512_permutex2var_epi64(r67, idx_even, r67);
    let r7 = _mm512_permutex2var_epi64(r67, idx_odd, r67);

    (r0, r1, r2, r3, r4, r5, r6, r7)
  }
}

/// 3-level `_mm512_unpacklo/hi_epi16` cascade to deinterleave 4 channels
/// from 2 reshaped `__m512i` inputs into 4 channel vectors (each holding
/// 16 u16 values in natural order, occupying the low 256 bits).
///
/// This is the AVX-512 analog of the AVX2 `deinterleave_rgba64_cascade`.
/// Used twice per 32-pixel block (once for pixels 0-15, once for 16-31).
///
/// # Safety
///
/// Caller must hold AVX-512F + AVX-512BW `target_feature`.
#[inline(always)]
unsafe fn cascade_4ch_epi16(
  r0: __m512i,
  r1: __m512i,
  r2: __m512i,
  r3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
  unsafe {
    // Level 1
    let lo_01 = _mm512_unpacklo_epi16(r0, r1);
    let hi_01 = _mm512_unpackhi_epi16(r0, r1);
    let lo_23 = _mm512_unpacklo_epi16(r2, r3);
    let hi_23 = _mm512_unpackhi_epi16(r2, r3);

    // Level 2
    let lo_lo = _mm512_unpacklo_epi32(lo_01, lo_23);
    let lo_hi = _mm512_unpackhi_epi32(lo_01, lo_23);
    let hi_lo = _mm512_unpacklo_epi32(hi_01, hi_23);
    let hi_hi = _mm512_unpackhi_epi32(hi_01, hi_23);

    // Level 3
    let ch0_raw = _mm512_unpacklo_epi64(lo_lo, hi_lo);
    let ch1_raw = _mm512_unpackhi_epi64(lo_lo, hi_lo);
    let ch2_raw = _mm512_unpacklo_epi64(lo_hi, hi_hi);
    let ch3_raw = _mm512_unpackhi_epi64(lo_hi, hi_hi);

    // Lane-cross fixup: AVX-512 unpack operates per 128-bit lane.
    // After 3 levels, 64-bit chunks are in the lane-split order
    // [lo0..lo3 | hi0..hi3] per 256-bit half → need to reassemble into
    // natural order. The pack_fixup [0,2,4,6,1,3,5,7] (64-bit chunks)
    // restores [lo0,lo1,lo2,lo3, hi0,hi1,hi2,hi3] → flat u16x32 in order.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let ch0 = _mm512_permutexvar_epi64(pack_fixup, ch0_raw);
    let ch1 = _mm512_permutexvar_epi64(pack_fixup, ch1_raw);
    let ch2 = _mm512_permutexvar_epi64(pack_fixup, ch2_raw);
    let ch3 = _mm512_permutexvar_epi64(pack_fixup, ch3_raw);

    (ch0, ch1, ch2, ch3)
  }
}

/// Deinterleave 32 pixels of stride-4 u16 (Rgba64 or Bgra64) from four
/// `__m512i` loads into four separate u16×32 channel vectors in natural order.
///
/// Returns `(ch0, ch1, ch2, ch3)` in memory order.
/// For Rgba64: `(R, G, B, A)`. For Bgra64: `(B, G, R, A)`.
///
/// # Safety
///
/// Caller must hold AVX-512F + AVX-512BW `target_feature`.
#[inline(always)]
unsafe fn deinterleave_rgba64_32px(
  raw0: __m512i,
  raw1: __m512i,
  raw2: __m512i,
  raw3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
  unsafe {
    let (r0, r1, r2, r3, r4, r5, r6, r7) = reshape_rgba64_32px_for_cascade(raw0, raw1, raw2, raw3);
    // Process two groups of 4 registers (each group = 16 pixels).
    // Group A (pixels 0-15): r0, r1, r2, r3
    let (ch0a, ch1a, ch2a, ch3a) = cascade_4ch_epi16(r0, r1, r2, r3);
    // Group B (pixels 16-31): r4, r5, r6, r7
    let (ch0b, ch1b, ch2b, ch3b) = cascade_4ch_epi16(r4, r5, r6, r7);

    // Combine the two halves: each ch_a holds pixels 0-15 in low 256b,
    // ch_b holds pixels 16-31 in low 256b.
    // Merge into a single __m512i via permutex2var.
    let idx_merge = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    let ch0 = _mm512_permutex2var_epi64(ch0a, idx_merge, ch0b);
    let ch1 = _mm512_permutex2var_epi64(ch1a, idx_merge, ch1b);
    let ch2 = _mm512_permutex2var_epi64(ch2a, idx_merge, ch2b);
    let ch3 = _mm512_permutex2var_epi64(ch3a, idx_merge, ch3b);
    (ch0, ch1, ch2, ch3)
  }
}

// =============================================================================
// u16 → u8 narrowing via srli::<8> + cvtusepi16_epi8
// =============================================================================

/// Narrow a u16×32 vector to u8×32 (256-bit result) via logical right-shift
/// by 8, then saturating unsigned narrow with `_mm512_cvtusepi16_epi8`.
///
/// Equivalent to scalar `(v >> 8) as u8`.
#[inline(always)]
unsafe fn narrow_u16x32_to_u8x32(v: __m512i) -> __m256i {
  unsafe { _mm512_cvtusepi16_epi8(_mm512_srli_epi16::<8>(v)) }
}

// =============================================================================
// Rgb48 (R, G, B — 3 u16 elements per pixel)
// =============================================================================

/// AVX-512 Rgb48 → packed u8 RGB. 32 pixels per outer iteration.
///
/// Processes four 8-pixel halves (3 × 128-bit loads each) under the
/// AVX-512 target_feature context (SSE4.1/SSSE3 are subsets). Narrows
/// each channel via `>> 8` and writes 8 pixels (24 bytes) per half.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available (caller obligation).
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgb48_to_rgb_row(rgb48: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    // Process 32 pixels per outer iteration (4 × 8-pixel halves).
    while x + 32 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);
      // Half 0: pixels x..x+7
      let v0 = _mm_loadu_si128(ptr.cast());
      let v1 = _mm_loadu_si128(ptr.add(8).cast());
      let v2 = _mm_loadu_si128(ptr.add(16).cast());
      let (r0, g0, b0) = deinterleave_rgb48_8px(v0, v1, v2);
      let r0u8 = narrow_u16x8_to_u8x8(r0, zero);
      let g0u8 = narrow_u16x8_to_u8x8(g0, zero);
      let b0u8 = narrow_u16x8_to_u8x8(b0, zero);
      let mut tmp0 = [0u8; 48];
      write_rgb_16(r0u8, g0u8, b0u8, tmp0.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp0.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);

      // Half 1: pixels x+8..x+15
      let ptr8 = ptr.add(24);
      let v3 = _mm_loadu_si128(ptr8.cast());
      let v4 = _mm_loadu_si128(ptr8.add(8).cast());
      let v5 = _mm_loadu_si128(ptr8.add(16).cast());
      let (r1, g1, b1) = deinterleave_rgb48_8px(v3, v4, v5);
      let r1u8 = narrow_u16x8_to_u8x8(r1, zero);
      let g1u8 = narrow_u16x8_to_u8x8(g1, zero);
      let b1u8 = narrow_u16x8_to_u8x8(b1, zero);
      let mut tmp1 = [0u8; 48];
      write_rgb_16(r1u8, g1u8, b1u8, tmp1.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp1.as_ptr(), rgb_out.as_mut_ptr().add((x + 8) * 3), 24);

      // Half 2: pixels x+16..x+23
      let ptr16 = ptr.add(48);
      let v6 = _mm_loadu_si128(ptr16.cast());
      let v7 = _mm_loadu_si128(ptr16.add(8).cast());
      let v8 = _mm_loadu_si128(ptr16.add(16).cast());
      let (r2, g2, b2) = deinterleave_rgb48_8px(v6, v7, v8);
      let r2u8 = narrow_u16x8_to_u8x8(r2, zero);
      let g2u8 = narrow_u16x8_to_u8x8(g2, zero);
      let b2u8 = narrow_u16x8_to_u8x8(b2, zero);
      let mut tmp2 = [0u8; 48];
      write_rgb_16(r2u8, g2u8, b2u8, tmp2.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp2.as_ptr(), rgb_out.as_mut_ptr().add((x + 16) * 3), 24);

      // Half 3: pixels x+24..x+31
      let ptr24 = ptr.add(72);
      let v9 = _mm_loadu_si128(ptr24.cast());
      let v10 = _mm_loadu_si128(ptr24.add(8).cast());
      let v11 = _mm_loadu_si128(ptr24.add(16).cast());
      let (r3, g3, b3) = deinterleave_rgb48_8px(v9, v10, v11);
      let r3u8 = narrow_u16x8_to_u8x8(r3, zero);
      let g3u8 = narrow_u16x8_to_u8x8(g3, zero);
      let b3u8 = narrow_u16x8_to_u8x8(b3, zero);
      let mut tmp3 = [0u8; 48];
      write_rgb_16(r3u8, g3u8, b3u8, tmp3.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp3.as_ptr(), rgb_out.as_mut_ptr().add((x + 24) * 3), 24);

      x += 32;
    }
    // Scalar tail: remaining < 32 pixels.
    if x < width {
      scalar::rgb48_to_rgb_row(&rgb48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 Rgb48 → packed u8 RGBA. 32 pixels per outer iteration. Alpha
/// forced to 0xFF.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgb48_to_rgba_row(rgb48: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let opaque_u16 = _mm_set1_epi16(0x00FFu16 as i16);
    let opaque_u8 = _mm_packus_epi16(opaque_u16, zero);
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);

      macro_rules! process_half {
        ($ptr:expr, $out_off:expr) => {{
          let v0 = _mm_loadu_si128($ptr.cast());
          let v1 = _mm_loadu_si128($ptr.add(8).cast());
          let v2 = _mm_loadu_si128($ptr.add(16).cast());
          let (r, g, b) = deinterleave_rgb48_8px(v0, v1, v2);
          let ru8 = narrow_u16x8_to_u8x8(r, zero);
          let gu8 = narrow_u16x8_to_u8x8(g, zero);
          let bu8 = narrow_u16x8_to_u8x8(b, zero);
          let mut tmp = [0u8; 64];
          write_rgba_16(ru8, gu8, bu8, opaque_u8, tmp.as_mut_ptr());
          core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add($out_off), 32);
        }};
      }

      process_half!(ptr, x * 4);
      process_half!(ptr.add(24), (x + 8) * 4);
      process_half!(ptr.add(48), (x + 16) * 4);
      process_half!(ptr.add(72), (x + 24) * 4);

      x += 32;
    }
    if x < width {
      scalar::rgb48_to_rgba_row(&rgb48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 Rgb48 → native-depth u16 RGB (identity repack). 32 pixels per iter.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgb48_to_rgb_u16_row(rgb48: &[u16], rgb_out: &mut [u16], width: usize) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);

      macro_rules! process_half_u16 {
        ($ptr:expr, $out_off:expr) => {{
          let v0 = _mm_loadu_si128($ptr.cast());
          let v1 = _mm_loadu_si128($ptr.add(8).cast());
          let v2 = _mm_loadu_si128($ptr.add(16).cast());
          let (r, g, b) = deinterleave_rgb48_8px(v0, v1, v2);
          write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add($out_off));
        }};
      }

      process_half_u16!(ptr, x * 3);
      process_half_u16!(ptr.add(24), (x + 8) * 3);
      process_half_u16!(ptr.add(48), (x + 16) * 3);
      process_half_u16!(ptr.add(72), (x + 24) * 3);

      x += 32;
    }
    if x < width {
      scalar::rgb48_to_rgb_u16_row(&rgb48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 Rgb48 → native-depth u16 RGBA. 32 pixels per iter. Alpha forced to
/// 0xFFFF.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgb48_to_rgba_u16_row(
  rgb48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = _mm_set1_epi16(0xFFFFu16 as i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);

      macro_rules! process_half_rgba_u16 {
        ($ptr:expr, $out_off:expr) => {{
          let v0 = _mm_loadu_si128($ptr.cast());
          let v1 = _mm_loadu_si128($ptr.add(8).cast());
          let v2 = _mm_loadu_si128($ptr.add(16).cast());
          let (r, g, b) = deinterleave_rgb48_8px(v0, v1, v2);
          write_rgba_u16_8(r, g, b, opaque, rgba_out.as_mut_ptr().add($out_off));
        }};
      }

      process_half_rgba_u16!(ptr, x * 4);
      process_half_rgba_u16!(ptr.add(24), (x + 8) * 4);
      process_half_rgba_u16!(ptr.add(48), (x + 16) * 4);
      process_half_rgba_u16!(ptr.add(72), (x + 24) * 4);

      x += 32;
    }
    if x < width {
      scalar::rgb48_to_rgba_u16_row(&rgb48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Bgr48 (B, G, R — 3 u16 elements per pixel)
// =============================================================================

/// AVX-512 Bgr48 → packed u8 RGB. 32 pixels per outer iteration.
/// B↔R swap via passing `(ch2, ch1, ch0)` to write helpers.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_bgr48_to_rgb_row(bgr48: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);

      macro_rules! process_half_bgr {
        ($ptr:expr, $out_off:expr) => {{
          let v0 = _mm_loadu_si128($ptr.cast());
          let v1 = _mm_loadu_si128($ptr.add(8).cast());
          let v2 = _mm_loadu_si128($ptr.add(16).cast());
          let (b, g, r) = deinterleave_rgb48_8px(v0, v1, v2);
          let ru8 = narrow_u16x8_to_u8x8(r, zero);
          let gu8 = narrow_u16x8_to_u8x8(g, zero);
          let bu8 = narrow_u16x8_to_u8x8(b, zero);
          let mut tmp = [0u8; 48];
          write_rgb_16(ru8, gu8, bu8, tmp.as_mut_ptr());
          core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add($out_off), 24);
        }};
      }

      process_half_bgr!(ptr, x * 3);
      process_half_bgr!(ptr.add(24), (x + 8) * 3);
      process_half_bgr!(ptr.add(48), (x + 16) * 3);
      process_half_bgr!(ptr.add(72), (x + 24) * 3);

      x += 32;
    }
    if x < width {
      scalar::bgr48_to_rgb_row(&bgr48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 Bgr48 → packed u8 RGBA. 32 pixels per iter.
/// B↔R swap; alpha forced to 0xFF.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_bgr48_to_rgba_row(bgr48: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let opaque_u16 = _mm_set1_epi16(0x00FFu16 as i16);
    let opaque_u8 = _mm_packus_epi16(opaque_u16, zero);
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);

      macro_rules! process_half_bgr_rgba {
        ($ptr:expr, $out_off:expr) => {{
          let v0 = _mm_loadu_si128($ptr.cast());
          let v1 = _mm_loadu_si128($ptr.add(8).cast());
          let v2 = _mm_loadu_si128($ptr.add(16).cast());
          let (b, g, r) = deinterleave_rgb48_8px(v0, v1, v2);
          let ru8 = narrow_u16x8_to_u8x8(r, zero);
          let gu8 = narrow_u16x8_to_u8x8(g, zero);
          let bu8 = narrow_u16x8_to_u8x8(b, zero);
          let mut tmp = [0u8; 64];
          write_rgba_16(ru8, gu8, bu8, opaque_u8, tmp.as_mut_ptr());
          core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add($out_off), 32);
        }};
      }

      process_half_bgr_rgba!(ptr, x * 4);
      process_half_bgr_rgba!(ptr.add(24), (x + 8) * 4);
      process_half_bgr_rgba!(ptr.add(48), (x + 16) * 4);
      process_half_bgr_rgba!(ptr.add(72), (x + 24) * 4);

      x += 32;
    }
    if x < width {
      scalar::bgr48_to_rgba_row(&bgr48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 Bgr48 → native-depth u16 RGB. 32 pixels per iter.
/// B↔R swap; values unchanged.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_bgr48_to_rgb_u16_row(bgr48: &[u16], rgb_out: &mut [u16], width: usize) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);

      macro_rules! process_half_bgr_u16 {
        ($ptr:expr, $out_off:expr) => {{
          let v0 = _mm_loadu_si128($ptr.cast());
          let v1 = _mm_loadu_si128($ptr.add(8).cast());
          let v2 = _mm_loadu_si128($ptr.add(16).cast());
          let (b, g, r) = deinterleave_rgb48_8px(v0, v1, v2);
          write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add($out_off));
        }};
      }

      process_half_bgr_u16!(ptr, x * 3);
      process_half_bgr_u16!(ptr.add(24), (x + 8) * 3);
      process_half_bgr_u16!(ptr.add(48), (x + 16) * 3);
      process_half_bgr_u16!(ptr.add(72), (x + 24) * 3);

      x += 32;
    }
    if x < width {
      scalar::bgr48_to_rgb_u16_row(&bgr48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 Bgr48 → native-depth u16 RGBA. 32 pixels per iter.
/// B↔R swap; alpha forced to 0xFFFF.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_bgr48_to_rgba_u16_row(
  bgr48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = _mm_set1_epi16(0xFFFFu16 as i16);
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);

      macro_rules! process_half_bgr_rgba_u16 {
        ($ptr:expr, $out_off:expr) => {{
          let v0 = _mm_loadu_si128($ptr.cast());
          let v1 = _mm_loadu_si128($ptr.add(8).cast());
          let v2 = _mm_loadu_si128($ptr.add(16).cast());
          let (b, g, r) = deinterleave_rgb48_8px(v0, v1, v2);
          write_rgba_u16_8(r, g, b, opaque, rgba_out.as_mut_ptr().add($out_off));
        }};
      }

      process_half_bgr_rgba_u16!(ptr, x * 4);
      process_half_bgr_rgba_u16!(ptr.add(24), (x + 8) * 4);
      process_half_bgr_rgba_u16!(ptr.add(48), (x + 16) * 4);
      process_half_bgr_rgba_u16!(ptr.add(72), (x + 24) * 4);

      x += 32;
    }
    if x < width {
      scalar::bgr48_to_rgba_u16_row(&bgr48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Rgba64 (R, G, B, A — 4 u16 elements per pixel)
// =============================================================================

/// AVX-512 Rgba64 → packed u8 RGB. 32 pixels per SIMD iteration.
/// Loads 4 × `__m512i` (128 u16 = 32 pixels), deinterleaves via the
/// AVX-512 cascade helper, narrows via `>> 8` + `cvtusepi16_epi8`, writes
/// 32 pixels (96 bytes) via `write_rgb_16` on 128-bit quarters.
///
/// Alpha discarded.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgba64_to_rgb_row(rgba64: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = _mm512_loadu_si512(ptr.cast());
      let raw1 = _mm512_loadu_si512(ptr.add(32).cast());
      let raw2 = _mm512_loadu_si512(ptr.add(64).cast());
      let raw3 = _mm512_loadu_si512(ptr.add(96).cast());
      let (r_u16, g_u16, b_u16, _a) = deinterleave_rgba64_32px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x32_to_u8x32(r_u16);
      let g_u8 = narrow_u16x32_to_u8x32(g_u16);
      let b_u8 = narrow_u16x32_to_u8x32(b_u16);
      // write_rgb_16 takes __m128i; split the 256-bit result into two 128-bit halves.
      let out_ptr = rgb_out.as_mut_ptr().add(x * 3);
      write_rgb_16(
        _mm256_castsi256_si128(r_u8),
        _mm256_castsi256_si128(g_u8),
        _mm256_castsi256_si128(b_u8),
        out_ptr,
      );
      write_rgb_16(
        _mm256_extracti128_si256::<1>(r_u8),
        _mm256_extracti128_si256::<1>(g_u8),
        _mm256_extracti128_si256::<1>(b_u8),
        out_ptr.add(48),
      );
      x += 32;
    }
    if x < width {
      scalar::rgba64_to_rgb_row(&rgba64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 Rgba64 → packed u8 RGBA. 32 pixels per SIMD iteration.
/// Source alpha passes through (narrowed via `>> 8`).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgba64_to_rgba_row(rgba64: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = _mm512_loadu_si512(ptr.cast());
      let raw1 = _mm512_loadu_si512(ptr.add(32).cast());
      let raw2 = _mm512_loadu_si512(ptr.add(64).cast());
      let raw3 = _mm512_loadu_si512(ptr.add(96).cast());
      let (r_u16, g_u16, b_u16, a_u16) = deinterleave_rgba64_32px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x32_to_u8x32(r_u16);
      let g_u8 = narrow_u16x32_to_u8x32(g_u16);
      let b_u8 = narrow_u16x32_to_u8x32(b_u16);
      let a_u8 = narrow_u16x32_to_u8x32(a_u16);
      let out_ptr = rgba_out.as_mut_ptr().add(x * 4);
      write_rgba_16(
        _mm256_castsi256_si128(r_u8),
        _mm256_castsi256_si128(g_u8),
        _mm256_castsi256_si128(b_u8),
        _mm256_castsi256_si128(a_u8),
        out_ptr,
      );
      write_rgba_16(
        _mm256_extracti128_si256::<1>(r_u8),
        _mm256_extracti128_si256::<1>(g_u8),
        _mm256_extracti128_si256::<1>(b_u8),
        _mm256_extracti128_si256::<1>(a_u8),
        out_ptr.add(64),
      );
      x += 32;
    }
    if x < width {
      scalar::rgba64_to_rgba_row(&rgba64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 Rgba64 → native-depth u16 RGB. 32 pixels per SIMD iteration.
/// Alpha discarded.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgba64_to_rgb_u16_row(
  rgba64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = _mm512_loadu_si512(ptr.cast());
      let raw1 = _mm512_loadu_si512(ptr.add(32).cast());
      let raw2 = _mm512_loadu_si512(ptr.add(64).cast());
      let raw3 = _mm512_loadu_si512(ptr.add(96).cast());
      let (r_u16, g_u16, b_u16, _a) = deinterleave_rgba64_32px(raw0, raw1, raw2, raw3);
      // Use the shared write_rgb_u16_32 helper (writes 32 px = 4 × 8-px chunks).
      write_rgb_u16_32(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 32;
    }
    if x < width {
      scalar::rgba64_to_rgb_u16_row(&rgba64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 Rgba64 → native-depth u16 RGBA (identity copy). 32 pixels per iter.
/// Source alpha preserved.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgba64_to_rgba_u16_row(
  rgba64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = _mm512_loadu_si512(ptr.cast());
      let raw1 = _mm512_loadu_si512(ptr.add(32).cast());
      let raw2 = _mm512_loadu_si512(ptr.add(64).cast());
      let raw3 = _mm512_loadu_si512(ptr.add(96).cast());
      let (r_u16, g_u16, b_u16, a_u16) = deinterleave_rgba64_32px(raw0, raw1, raw2, raw3);
      let opaque = _mm_set1_epi16(-1i16); // 0xFFFF placeholder — not used; a_u16 has real alpha
      let out_ptr = rgba_out.as_mut_ptr().add(x * 4);
      // write_rgba_u16_32 passes a constant alpha __m128i, but we have a per-pixel alpha.
      // Instead split into four 8-pixel chunks using write_rgba_u16_8.
      write_rgba_u16_8(
        _mm512_extracti32x4_epi32::<0>(r_u16),
        _mm512_extracti32x4_epi32::<0>(g_u16),
        _mm512_extracti32x4_epi32::<0>(b_u16),
        _mm512_extracti32x4_epi32::<0>(a_u16),
        out_ptr,
      );
      write_rgba_u16_8(
        _mm512_extracti32x4_epi32::<1>(r_u16),
        _mm512_extracti32x4_epi32::<1>(g_u16),
        _mm512_extracti32x4_epi32::<1>(b_u16),
        _mm512_extracti32x4_epi32::<1>(a_u16),
        out_ptr.add(32),
      );
      write_rgba_u16_8(
        _mm512_extracti32x4_epi32::<2>(r_u16),
        _mm512_extracti32x4_epi32::<2>(g_u16),
        _mm512_extracti32x4_epi32::<2>(b_u16),
        _mm512_extracti32x4_epi32::<2>(a_u16),
        out_ptr.add(64),
      );
      write_rgba_u16_8(
        _mm512_extracti32x4_epi32::<3>(r_u16),
        _mm512_extracti32x4_epi32::<3>(g_u16),
        _mm512_extracti32x4_epi32::<3>(b_u16),
        _mm512_extracti32x4_epi32::<3>(a_u16),
        out_ptr.add(96),
      );
      let _ = opaque; // suppress unused-variable warning
      x += 32;
    }
    if x < width {
      scalar::rgba64_to_rgba_u16_row(&rgba64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Bgra64 (B, G, R, A — 4 u16 elements per pixel)
// =============================================================================

/// AVX-512 Bgra64 → packed u8 RGB. 32 pixels per SIMD iteration.
/// B↔R swap; alpha discarded.
///
/// `deinterleave_rgba64_32px` yields `(B, G, R, A)` in source memory order.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_bgra64_to_rgb_row(bgra64: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = _mm512_loadu_si512(ptr.cast());
      let raw1 = _mm512_loadu_si512(ptr.add(32).cast());
      let raw2 = _mm512_loadu_si512(ptr.add(64).cast());
      let raw3 = _mm512_loadu_si512(ptr.add(96).cast());
      // ch0=B, ch1=G, ch2=R, ch3=A (source BGRA order)
      let (b_u16, g_u16, r_u16, _a) = deinterleave_rgba64_32px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x32_to_u8x32(r_u16);
      let g_u8 = narrow_u16x32_to_u8x32(g_u16);
      let b_u8 = narrow_u16x32_to_u8x32(b_u16);
      let out_ptr = rgb_out.as_mut_ptr().add(x * 3);
      write_rgb_16(
        _mm256_castsi256_si128(r_u8),
        _mm256_castsi256_si128(g_u8),
        _mm256_castsi256_si128(b_u8),
        out_ptr,
      );
      write_rgb_16(
        _mm256_extracti128_si256::<1>(r_u8),
        _mm256_extracti128_si256::<1>(g_u8),
        _mm256_extracti128_si256::<1>(b_u8),
        out_ptr.add(48),
      );
      x += 32;
    }
    if x < width {
      scalar::bgra64_to_rgb_row(&bgra64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 Bgra64 → packed u8 RGBA. 32 pixels per SIMD iteration.
/// B↔R swap; source alpha passes through (narrowed via `>> 8`).
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_bgra64_to_rgba_row(bgra64: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = _mm512_loadu_si512(ptr.cast());
      let raw1 = _mm512_loadu_si512(ptr.add(32).cast());
      let raw2 = _mm512_loadu_si512(ptr.add(64).cast());
      let raw3 = _mm512_loadu_si512(ptr.add(96).cast());
      let (b_u16, g_u16, r_u16, a_u16) = deinterleave_rgba64_32px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x32_to_u8x32(r_u16);
      let g_u8 = narrow_u16x32_to_u8x32(g_u16);
      let b_u8 = narrow_u16x32_to_u8x32(b_u16);
      let a_u8 = narrow_u16x32_to_u8x32(a_u16);
      let out_ptr = rgba_out.as_mut_ptr().add(x * 4);
      write_rgba_16(
        _mm256_castsi256_si128(r_u8),
        _mm256_castsi256_si128(g_u8),
        _mm256_castsi256_si128(b_u8),
        _mm256_castsi256_si128(a_u8),
        out_ptr,
      );
      write_rgba_16(
        _mm256_extracti128_si256::<1>(r_u8),
        _mm256_extracti128_si256::<1>(g_u8),
        _mm256_extracti128_si256::<1>(b_u8),
        _mm256_extracti128_si256::<1>(a_u8),
        out_ptr.add(64),
      );
      x += 32;
    }
    if x < width {
      scalar::bgra64_to_rgba_row(&bgra64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// AVX-512 Bgra64 → native-depth u16 RGB. 32 pixels per SIMD iteration.
/// B↔R swap; alpha discarded.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_bgra64_to_rgb_u16_row(
  bgra64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = _mm512_loadu_si512(ptr.cast());
      let raw1 = _mm512_loadu_si512(ptr.add(32).cast());
      let raw2 = _mm512_loadu_si512(ptr.add(64).cast());
      let raw3 = _mm512_loadu_si512(ptr.add(96).cast());
      // Swap B↔R: store (R=ch2, G=ch1, B=ch0)
      let (b_u16, g_u16, r_u16, _a) = deinterleave_rgba64_32px(raw0, raw1, raw2, raw3);
      write_rgb_u16_32(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 32;
    }
    if x < width {
      scalar::bgra64_to_rgb_u16_row(&bgra64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// AVX-512 Bgra64 → native-depth u16 RGBA. 32 pixels per SIMD iteration.
/// B↔R swap; source alpha preserved at position 3.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_bgra64_to_rgba_u16_row(
  bgra64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 32 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = _mm512_loadu_si512(ptr.cast());
      let raw1 = _mm512_loadu_si512(ptr.add(32).cast());
      let raw2 = _mm512_loadu_si512(ptr.add(64).cast());
      let raw3 = _mm512_loadu_si512(ptr.add(96).cast());
      // Swap B↔R: (R=ch2, G=ch1, B=ch0, A=ch3)
      let (b_u16, g_u16, r_u16, a_u16) = deinterleave_rgba64_32px(raw0, raw1, raw2, raw3);
      let out_ptr = rgba_out.as_mut_ptr().add(x * 4);
      write_rgba_u16_8(
        _mm512_extracti32x4_epi32::<0>(r_u16),
        _mm512_extracti32x4_epi32::<0>(g_u16),
        _mm512_extracti32x4_epi32::<0>(b_u16),
        _mm512_extracti32x4_epi32::<0>(a_u16),
        out_ptr,
      );
      write_rgba_u16_8(
        _mm512_extracti32x4_epi32::<1>(r_u16),
        _mm512_extracti32x4_epi32::<1>(g_u16),
        _mm512_extracti32x4_epi32::<1>(b_u16),
        _mm512_extracti32x4_epi32::<1>(a_u16),
        out_ptr.add(32),
      );
      write_rgba_u16_8(
        _mm512_extracti32x4_epi32::<2>(r_u16),
        _mm512_extracti32x4_epi32::<2>(g_u16),
        _mm512_extracti32x4_epi32::<2>(b_u16),
        _mm512_extracti32x4_epi32::<2>(a_u16),
        out_ptr.add(64),
      );
      write_rgba_u16_8(
        _mm512_extracti32x4_epi32::<3>(r_u16),
        _mm512_extracti32x4_epi32::<3>(g_u16),
        _mm512_extracti32x4_epi32::<3>(b_u16),
        _mm512_extracti32x4_epi32::<3>(a_u16),
        out_ptr.add(96),
      );
      x += 32;
    }
    if x < width {
      scalar::bgra64_to_rgba_u16_row(&bgra64[x * 4..], &mut rgba_out[x * 4..], width - x);
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
