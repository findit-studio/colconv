//! AVX2 alpha-extract helpers — SIMD parity of `crate::row::scalar::alpha_extract`.
//!
//! Each fn matches its scalar counterpart byte-for-byte (verified by
//! `*_matches_scalar_widths` tests in this file).
//!
//! # Strategy
//!
//! AVX2 doubles the SSE4.1 lane width; we keep the same load + mask/blend
//! shape and bump the per-iteration block size where it is cheap to do so.
//!
//! **u8 helpers** (helpers 1 and 4): block = 8 px / iter (one `__m256i`
//! of RGBA = 32 bytes = 8 px x 4 ch). An α-slot mask repeats the SSE4.1
//! `[0, 0, 0, 0xFF, ...]` pattern across both 128-bit lanes, then
//! `_mm256_blendv_epi8` substitutes only the α byte of each pixel.
//!
//! **u16 / u16->u8 helpers** (helpers 2, 3, 5, 6): block = 8 px / iter.
//! Helper 2 packs 2 source `__m256i` of u16 down to one `__m256i` of u8
//! via `_mm256_packus_epi16`; that intrinsic is per-128-bit lane and
//! produces a lane-split byte order — we restore natural order with
//! `_mm256_permute4x64_epi64::<0xD8>` immediately after the pack so the
//! downstream shuffle / blend sees `[A0, Y0, U0, V0, A1, ..., A7, ...]`.
//! Helpers 3 and 6 stay in u16 throughout, using two `__m256i` of dest
//! per iter; `_mm256_shuffle_epi8` is per-lane but each 4-u16 pixel is
//! contained inside one 128-bit lane, so no cross-lane fixup is needed.
//!
//! **BITS shift** in helper 5: `_mm256_srl_epi16` accepts a runtime
//! `__m128i` shift count, so we build it once via
//! `_mm_cvtsi32_si128(BITS as i32 - 8)` and avoid per-`BITS`
//! monomorphization with `_mm256_srli_epi16` literals.
//!
//! # Cross-lane safety
//!
//! Helper 2 is the only place `_mm256_packus_epi16` (per-lane) is used,
//! and the immediately-following `_mm256_permute4x64_epi64::<0xD8>` is
//! exactly the lane-split fixup pattern documented in
//! `super::chroma_i16x16` / `super::narrow_u8x32`. All other helpers
//! confine each pixel inside a 128-bit lane and therefore use only
//! per-lane `_mm256_shuffle_epi8` / `_mm256_blendv_epi8`, which is safe.

#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::row::scalar::alpha_extract as scalar;

// ---------------------------------------------------------------------------
// Helper 1: VUYA u8 -> u8 RGBA  (α at packed slot 3)
// ---------------------------------------------------------------------------

/// VUYA -> u8 RGBA: gather α from `packed[3 + 4*n]` into `rgba_out[3 + 4*n]`.
///
/// Block: 8 px / iter via `_mm256_blendv_epi8` with an α-slot mask.
///
/// # Safety
///
/// AVX2 must be available. Both slices must be `>= width * 4` bytes.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn copy_alpha_packed_u8x4_at_3(packed: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  unsafe {
    // Mask: 0xFF (-1) at α byte (offset 3 of each 4-byte pixel), 0 elsewhere.
    // `_mm256_set_epi8` takes args high-to-low (byte 31 first, byte 0 last).
    // 0xFF positions align to RGBA α slots: bytes 31, 27, 23, 19, 15, 11, 7, 3.
    // Each 128-bit lane holds 4 pixels; both lanes mirror the same mask.
    let alpha_mask = _mm256_set_epi8(
      -1, 0, 0, 0, // bytes 31..28 (px7: α at byte 31)
      -1, 0, 0, 0, // bytes 27..24 (px6: α at byte 27)
      -1, 0, 0, 0, // bytes 23..20 (px5: α at byte 23)
      -1, 0, 0, 0, // bytes 19..16 (px4: α at byte 19)
      -1, 0, 0, 0, // bytes 15..12 (px3: α at byte 15)
      -1, 0, 0, 0, // bytes 11..8  (px2: α at byte 11)
      -1, 0, 0, 0, // bytes  7..4  (px1: α at byte  7)
      -1, 0, 0, 0, // bytes  3..0  (px0: α at byte  3)
    );

    let mut x = 0usize;
    while x + 8 <= width {
      let off = x * 4;
      let src = _mm256_loadu_si256(packed.as_ptr().add(off).cast());
      let dst = _mm256_loadu_si256(rgba_out.as_ptr().add(off).cast());
      // blendv: where mask byte has high bit set (0xFF), pick `src`; else `dst`.
      let merged = _mm256_blendv_epi8(dst, src, alpha_mask);
      _mm256_storeu_si256(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 8;
    }

    if x < width {
      scalar::copy_alpha_packed_u8x4_at_3(
        &packed[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Helper 2: AYUV64 u16 -> u8 RGBA  (α at packed slot 0, depth >> 8)
// ---------------------------------------------------------------------------

/// AYUV64 -> u8 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> 8`.
///
/// Block: 8 px / iter. We load 2 × `__m256i` of u16 (= 16 u16 = 8 px),
/// right-shift by 8, narrow via `_mm256_packus_epi16` and undo the
/// per-128-bit-lane pack split with `_mm256_permute4x64_epi64::<0xD8>`,
/// then shuffle into α slots and blend over the existing u8 RGBA.
///
/// # Safety
///
/// AVX2 must be available. `packed.len() >= width * 4`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn copy_alpha_packed_u16x4_to_u8_at_0(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  unsafe {
    // Mask for u8 output α slot (byte 3 of each 4-byte pixel), 8 px wide.
    let alpha_mask = _mm256_set_epi8(
      -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, // hi lane
      -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, // lo lane
    );

    // Shuffle that scatters the 8 u8 α bytes (already in natural order
    // after the lane-fix permute) into the α slot of each 4-byte pixel.
    // Per 128-bit lane we hold 4 pixels (= bytes 0..15 of the lane), and
    // the 4 α bytes for that lane sit in bytes 0..3 of the lane (because
    // the natural-order pack output is `[A0..A7, junk]` with the high
    // 16 bytes coming from the second pack input which we set to zero).
    //
    // Wait: that would place all 8 α bytes in the LOW 128-bit lane, not
    // 4 in each lane. We instead split the work: shuffle the LOW lane to
    // place α0..α3 into α slots of pixels 0..3 (low 128-bit lane), then
    // shift the high 4 α bytes into the low lane via a 4-byte right-
    // shift inside the upper 128-bit lane (or more simply, broadcast the
    // permuted vector with `_mm256_permute2x128_si256` so α4..α7 live in
    // the LOW lane of a separate vector). That's awkward.
    //
    // Simpler: keep the SSE4.1 shuffle exactly, which addresses bytes
    // 0..15 within EACH 128-bit lane independently. We arrange so that
    // after the lane-fix permute, the 8 α bytes are split 4-and-4 across
    // the two lanes already. To make that happen we feed `_mm256_packus
    // _epi16(lo, hi)` where `lo` = px0..px3 (4 u16 α + 12 u16 chroma)
    // and `hi` = px4..px7. After the pack (per-lane) the byte stream is
    // `[lo.lane0_bytes(=lo low 128 narrowed), hi.lane0_bytes, lo.lane1
    // _bytes, hi.lane1_bytes]`; the 0xD8 permute reorders 64-bit chunks
    // to `[0, 2, 1, 3]`, giving `[lo.low_narrowed, lo.high_narrowed,
    // hi.low_narrowed, hi.high_narrowed]` which IS natural order.
    //
    // After natural order we have `[A0, Y0, U0, V0, A1, ..., A3, Y3,
    // U3, V3, A4, Y4, U4, V4, ..., A7, ..., V7]`. That places the 4 α
    // bytes for pixels 0..3 in the low 128-bit lane (at byte offsets 0,
    // 4, 8, 12) and the 4 α bytes for pixels 4..7 in the high lane (at
    // byte offsets 16, 20, 24, 28 = lane-local 0, 4, 8, 12). The
    // shuffle below is identical for both lanes: take byte 0 / 4 / 8 /
    // 12 of THIS lane and place at byte 3 / 7 / 11 / 15 of THIS lane.
    //
    // `_mm256_set_epi8` takes args high-to-low (byte 31 first, byte 0
    // last). Same per-lane pattern as the SSE4.1 helper.
    let shuf_mask = _mm256_set_epi8(
      12, -1, -1, -1, // hi-lane bytes 31..28 (px7): src[12+16] -> out[15+16]
      8, -1, -1, -1, // hi-lane bytes 27..24 (px6): src[ 8+16] -> out[11+16]
      4, -1, -1, -1, // hi-lane bytes 23..20 (px5): src[ 4+16] -> out[ 7+16]
      0, -1, -1, -1, // hi-lane bytes 19..16 (px4): src[ 0+16] -> out[ 3+16]
      12, -1, -1, -1, // lo-lane bytes 15..12 (px3): src[12]    -> out[15]
      8, -1, -1, -1, // lo-lane bytes 11..8  (px2): src[ 8]    -> out[11]
      4, -1, -1, -1, // lo-lane bytes  7..4  (px1): src[ 4]    -> out[ 7]
      0, -1, -1, -1, // lo-lane bytes  3..0  (px0): src[ 0]    -> out[ 3]
    );

    // Shift count (>> 8) — built once outside the loop.
    let shr8 = _mm_cvtsi32_si128(8);

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 AYUV64 pixels = 32 u16 = 2 × __m256i.
      let src_off = x * 4;
      let lo = _mm256_loadu_si256(packed.as_ptr().add(src_off).cast()); // px 0..3 (16 u16)
      let hi = _mm256_loadu_si256(packed.as_ptr().add(src_off + 16).cast()); // px 4..7 (16 u16)

      // Right-shift each u16 by 8 to bring the high byte (= u8 α) to the low byte.
      let lo_shr = _mm256_srl_epi16(lo, shr8);
      let hi_shr = _mm256_srl_epi16(hi, shr8);

      // Narrow u16 -> u8 (values fit in [0, 255] after the >> 8). This
      // intrinsic is per-128-bit lane, so it produces a lane-split byte
      // stream `[lo_lane0_narrowed, hi_lane0_narrowed, lo_lane1_narrowed,
      // hi_lane1_narrowed]` — fix it with `_mm256_permute4x64_epi64::<0xD8>`
      // (selector 0xD8 = 0b11_01_10_00 reorders 64-bit chunks
      // [0, 1, 2, 3] -> [0, 2, 1, 3], giving natural order
      // `[lo_lane0_n, lo_lane1_n, hi_lane0_n, hi_lane1_n]`).
      let packed_u8 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(lo_shr, hi_shr));

      // Now `packed_u8` = `[A0,Y0,U0,V0, A1,..., A7, ..., V7]` (natural).
      // Shuffle (per-lane) scatters the α bytes into the α slot.
      let a_scattered = _mm256_shuffle_epi8(packed_u8, shuf_mask);

      // Load existing rgba_out for 8 px and blend.
      let dst_off = x * 4;
      let dst = _mm256_loadu_si256(rgba_out.as_ptr().add(dst_off).cast());
      let merged = _mm256_blendv_epi8(dst, a_scattered, alpha_mask);
      _mm256_storeu_si256(rgba_out.as_mut_ptr().add(dst_off).cast(), merged);
      x += 8;
    }

    if x < width {
      scalar::copy_alpha_packed_u16x4_to_u8_at_0(
        &packed[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Helper 3: AYUV64 u16 -> u16 RGBA  (α at packed slot 0, no depth conv)
// ---------------------------------------------------------------------------

/// AYUV64 -> u16 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// Block: 8 px / iter (two `__m256i` of u16 RGBA = 64 bytes). Each
/// 4-u16 pixel sits inside one 128-bit lane, so the per-lane
/// `_mm256_shuffle_epi8` moves α between slot 0 and slot 3 within the
/// pixel without cross-lane concerns.
///
/// # Safety
///
/// AVX2 must be available. Both slices `>= width * 4` elements.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn copy_alpha_packed_u16x4_at_0(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  unsafe {
    // u16 α-slot mask: 0xFFFF at slot 3 of each 4-u16 pixel.
    // Each __m256i holds 16 u16 = 4 pixels; α slots are 3, 7, 11, 15.
    // `_mm256_set_epi16` takes args high-to-low (lane 15 first, lane 0 last).
    let alpha_mask_u16 = _mm256_set_epi16(
      -1, 0, 0, 0, // u16 lanes 15..12 (px3 of high 128: α at lane 15)
      -1, 0, 0, 0, // u16 lanes 11..8  (px2 of high 128: α at lane 11)
      -1, 0, 0, 0, // u16 lanes  7..4  (px1 of low  128: α at lane  7)
      -1, 0, 0, 0, // u16 lanes  3..0  (px0 of low  128: α at lane  3)
    );

    // Shuffle (per-128-bit lane): each lane holds 2 pixels (8 u16 =
    // bytes 0..15 of the lane). Within the lane:
    //   src u16 slot 0 (bytes 0,1) -> dst u16 slot 3 (bytes 6,7)   px_local0
    //   src u16 slot 4 (bytes 8,9) -> dst u16 slot 7 (bytes 14,15) px_local1
    // Both 128-bit lanes use the same pattern, so we mirror it.
    // `_mm256_set_epi8` takes args high-to-low (byte 31 first, byte 0 last).
    let shuf_mask = _mm256_set_epi8(
      9, 8, -1, -1, -1, -1, -1, -1, // hi lane px_local1: src[8,9] -> dst bytes 14,15
      1, 0, -1, -1, -1, -1, -1, -1, // hi lane px_local0: src[0,1] -> dst bytes  6,7
      9, 8, -1, -1, -1, -1, -1, -1, // lo lane px_local1: src[8,9] -> dst bytes 14,15
      1, 0, -1, -1, -1, -1, -1, -1, // lo lane px_local0: src[0,1] -> dst bytes  6,7
    );

    let mut x = 0usize;
    while x + 8 <= width {
      let off = x * 4;
      // Two __m256i cover 8 pixels of u16 RGBA output (16 u16 each).
      let src_lo = _mm256_loadu_si256(packed.as_ptr().add(off).cast()); // px 0..3 of packed
      let src_hi = _mm256_loadu_si256(packed.as_ptr().add(off + 16).cast()); // px 4..7 of packed
      let dst_lo = _mm256_loadu_si256(rgba_out.as_ptr().add(off).cast());
      let dst_hi = _mm256_loadu_si256(rgba_out.as_ptr().add(off + 16).cast());

      // Extract α (slot 0) from packed and place at slot 3 via per-lane shuffle.
      let a_lo = _mm256_shuffle_epi8(src_lo, shuf_mask);
      let a_hi = _mm256_shuffle_epi8(src_hi, shuf_mask);

      // Blend: where alpha_mask_u16 byte has high bit (0xFF), pick `a_*`.
      let merged_lo = _mm256_blendv_epi8(dst_lo, a_lo, alpha_mask_u16);
      let merged_hi = _mm256_blendv_epi8(dst_hi, a_hi, alpha_mask_u16);

      _mm256_storeu_si256(rgba_out.as_mut_ptr().add(off).cast(), merged_lo);
      _mm256_storeu_si256(rgba_out.as_mut_ptr().add(off + 16).cast(), merged_hi);
      x += 8;
    }

    if x < width {
      scalar::copy_alpha_packed_u16x4_at_0(
        &packed[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Helper 4: α plane u8 -> u8 RGBA
// ---------------------------------------------------------------------------

/// Yuva420p / 422p / 444p u8 -> u8 RGBA: scatter α plane into
/// `rgba_out[3 + 4*n]`.
///
/// Block: 8 px / iter via blend with α-slot mask. The 8 contiguous α
/// bytes are loaded into the low 64 bits of a `__m128i` and broadcast
/// across both 128-bit lanes of a `__m256i` so that each lane's
/// per-lane `_mm256_shuffle_epi8` can address its 4 α bytes (low 4 of
/// the lane).
///
/// # Safety
///
/// AVX2 must be available. `alpha.len() >= width`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  unsafe {
    // α-slot mask for u8 RGBA (8 px wide).
    let alpha_mask = _mm256_set_epi8(
      -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, // hi lane
      -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, // lo lane
    );

    // Per-lane shuffle scatters 4 α bytes into α slots of 4 pixels.
    // Each 128-bit lane gets the SAME 4 α bytes addressable at byte
    // offsets 0..3 of that lane; we use `_mm256_broadcastsi128_si256`
    // to duplicate the low 16 bytes (which contain α0..α7 in bytes 0..7
    // and zeros in 8..15) into both 128-bit lanes. The low lane reads
    // bytes 0..3 (= α0..α3); the high lane needs to read bytes 4..7 (=
    // α4..α7) so its shuffle indices are 4, 5, 6, 7 instead of 0, 1, 2, 3.
    //
    // `_mm256_set_epi8` takes args high-to-low (byte 31 first, byte 0 last).
    let shuf_mask = _mm256_set_epi8(
      7, -1, -1, -1, // hi-lane bytes 31..28 (px7)
      6, -1, -1, -1, // hi-lane bytes 27..24 (px6)
      5, -1, -1, -1, // hi-lane bytes 23..20 (px5)
      4, -1, -1, -1, // hi-lane bytes 19..16 (px4)
      3, -1, -1, -1, // lo-lane bytes 15..12 (px3)
      2, -1, -1, -1, // lo-lane bytes 11..8  (px2)
      1, -1, -1, -1, // lo-lane bytes  7..4  (px1)
      0, -1, -1, -1, // lo-lane bytes  3..0  (px0)
    );

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 α bytes into the low 64 bits of a 128-bit register.
      let a_raw_128 = _mm_loadl_epi64(alpha.as_ptr().add(x).cast());
      // Broadcast that 128-bit lane into both lanes of a 256-bit vector.
      // After this, BOTH 128-bit lanes contain α0..α7 in their low 8 bytes.
      let a_raw_256 = _mm256_broadcastsi128_si256(a_raw_128);
      let a_scattered = _mm256_shuffle_epi8(a_raw_256, shuf_mask);

      let off = x * 4;
      let dst = _mm256_loadu_si256(rgba_out.as_ptr().add(off).cast());
      let merged = _mm256_blendv_epi8(dst, a_scattered, alpha_mask);
      _mm256_storeu_si256(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 8;
    }

    if x < width {
      scalar::copy_alpha_plane_u8(&alpha[x..width], &mut rgba_out[x * 4..width * 4], width - x);
    }
  }
}

// ---------------------------------------------------------------------------
// Helper 5: α plane u16 -> u8 RGBA  (depth-conv >> (BITS-8))
// ---------------------------------------------------------------------------

/// Yuva*p9/10/12/14 -> u8 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> (BITS - 8)`.
///
/// Uses `_mm256_srl_epi16` with a runtime count vector
/// `_mm_cvtsi32_si128(BITS - 8)` to avoid per-`BITS` monomorphization.
/// Block: 8 px / iter (load 8 u16 α = 16 bytes = one `__m128i` widened
/// to `__m256i`, shift, narrow via `_mm_packus_epi16`, broadcast to
/// both 128-bit lanes, scatter into u8 RGBA).
///
/// # Safety
///
/// AVX2 must be available. `alpha.len() >= width`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn copy_alpha_plane_u16_to_u8<const BITS: u32>(
  alpha: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(BITS >= 8 && BITS <= 16, "BITS must be in [8, 16]");
  }
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  unsafe {
    let shr_count = _mm_cvtsi32_si128((BITS as i32) - 8);
    // α-slot mask for u8 RGBA (8 px wide).
    let alpha_mask = _mm256_set_epi8(
      -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, // hi lane
      -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, // lo lane
    );
    // Per-lane scatter, identical to helper 4 (α0..α7 broadcast to both
    // 128-bit lanes, hi lane indexes bytes 4..7, lo lane indexes 0..3).
    let shuf_mask = _mm256_set_epi8(
      7, -1, -1, -1, // hi-lane bytes 31..28 (px7)
      6, -1, -1, -1, // hi-lane bytes 27..24 (px6)
      5, -1, -1, -1, // hi-lane bytes 23..20 (px5)
      4, -1, -1, -1, // hi-lane bytes 19..16 (px4)
      3, -1, -1, -1, // lo-lane bytes 15..12 (px3)
      2, -1, -1, -1, // lo-lane bytes 11..8  (px2)
      1, -1, -1, -1, // lo-lane bytes  7..4  (px1)
      0, -1, -1, -1, // lo-lane bytes  3..0  (px0)
    );

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 u16 α values (16 bytes) into a __m128i.
      let a_u16 = _mm_loadu_si128(alpha.as_ptr().add(x).cast());
      // Right-shift by (BITS - 8).
      let a_shifted = _mm_srl_epi16(a_u16, shr_count);
      // Narrow u16 -> u8 (values fit in [0, 255] after shift). One
      // `_mm_packus_epi16` collapses 8 u16 + 8 zeros into 8 u8 (low 8
      // bytes) and 8 zeros (high 8 bytes) of a single __m128i.
      let a_u8_128 = _mm_packus_epi16(a_shifted, _mm_setzero_si128());
      // Broadcast the 128-bit lane into both lanes of a 256-bit vector,
      // so the per-lane shuffle below can address α0..α7 from EITHER lane.
      let a_u8_256 = _mm256_broadcastsi128_si256(a_u8_128);
      let a_scattered = _mm256_shuffle_epi8(a_u8_256, shuf_mask);

      let off = x * 4;
      let dst = _mm256_loadu_si256(rgba_out.as_ptr().add(off).cast());
      let merged = _mm256_blendv_epi8(dst, a_scattered, alpha_mask);
      _mm256_storeu_si256(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 8;
    }

    if x < width {
      scalar::copy_alpha_plane_u16_to_u8::<BITS>(
        &alpha[x..width],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Helper 6: α plane u16 -> u16 RGBA  (no depth conv)
// ---------------------------------------------------------------------------

/// Yuva*p9/10/12/14/16 -> u16 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// Block: 8 px / iter (two `__m256i` of u16 RGBA = 64 bytes). The 8
/// source α u16 are loaded into a `__m128i` and broadcast across both
/// 128-bit lanes of a `__m256i`, so the per-lane shuffle can address
/// the appropriate 4 α u16 from either lane.
///
/// # Safety
///
/// AVX2 must be available. `alpha.len() >= width`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn copy_alpha_plane_u16<const BITS: u32>(
  alpha: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(BITS > 0 && BITS <= 16, "BITS must be in [1, 16]");
  }
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  // BITS is validated above for caller safety but is not used at runtime:
  // no depth conversion is performed (u16 -> u16 is a direct scatter).
  let _ = BITS;

  unsafe {
    // u16 α-slot mask (4 pixels per __m256i; α at u16 lanes 3, 7, 11, 15).
    // `_mm256_set_epi16` takes args high-to-low (lane 15 first, lane 0 last).
    let alpha_mask_u16 = _mm256_set_epi16(
      -1, 0, 0, 0, // u16 lanes 15..12 (px3 of high 128)
      -1, 0, 0, 0, // u16 lanes 11..8  (px2 of high 128)
      -1, 0, 0, 0, // u16 lanes  7..4  (px1 of low  128)
      -1, 0, 0, 0, // u16 lanes  3..0  (px0 of low  128)
    );

    // After broadcasting the 8 source α u16 into both 128-bit lanes of a
    // __m256i, each lane holds α0..α7 in bytes 0..15. We split the 8 α
    // values across two output __m256i (each holding 4 pixels = 16 u16):
    //   dst_first  (px0..px3): lo 128-bit lane = α0/α1, hi = α2/α3.
    //   dst_second (px4..px7): lo 128-bit lane = α4/α5, hi = α6/α7.
    // Within each 128-bit dst lane (2 pixels = 8 u16 = bytes 0..15) we
    // want each α u16 placed at the α slot — u16 lane 3 (bytes 6,7) for
    // px_local0 and u16 lane 7 (bytes 14,15) for px_local1. `_mm256_
    // shuffle_epi8` is per-128-bit-lane, so each lane reads its source
    // bytes from the SAME 16-byte broadcast image (α0..α7 in bytes 0..15).
    //
    // `_mm256_set_epi8` takes args high-to-low (byte 31 first, byte 0 last).

    // dst_first shuffle: hi lane covers px2, px3; lo lane covers px0, px1.
    let shuf_dst0 = _mm256_set_epi8(
      7, 6, -1, -1, -1, -1, -1, -1, // hi lane px_local1 (= px3): α3 -> bytes 14,15 + 16
      5, 4, -1, -1, -1, -1, -1, -1, // hi lane px_local0 (= px2): α2 -> bytes  6,7 + 16
      3, 2, -1, -1, -1, -1, -1, -1, // lo lane px_local1 (= px1): α1 -> bytes 14,15
      1, 0, -1, -1, -1, -1, -1, -1, // lo lane px_local0 (= px0): α0 -> bytes  6,7
    );
    // dst_second shuffle: hi lane covers px6, px7; lo lane covers px4, px5.
    let shuf_dst1 = _mm256_set_epi8(
      15, 14, -1, -1, -1, -1, -1, -1, // hi lane px_local1 (= px7): α7 -> bytes 14,15 + 16
      13, 12, -1, -1, -1, -1, -1, -1, // hi lane px_local0 (= px6): α6 -> bytes  6,7 + 16
      11, 10, -1, -1, -1, -1, -1, -1, // lo lane px_local1 (= px5): α5 -> bytes 14,15
      9, 8, -1, -1, -1, -1, -1, -1, // lo lane px_local0 (= px4): α4 -> bytes  6,7
    );

    let mut x = 0usize;
    while x + 8 <= width {
      // Load 8 u16 α values = 16 bytes into a __m128i, then broadcast.
      let a_raw_128 = _mm_loadu_si128(alpha.as_ptr().add(x).cast());
      let a_raw_256 = _mm256_broadcastsi128_si256(a_raw_128);

      // Two output __m256i cover 8 pixels of u16 RGBA (16 u16 each).
      let off = x * 4;
      let dst_lo = _mm256_loadu_si256(rgba_out.as_ptr().add(off).cast());
      let dst_hi = _mm256_loadu_si256(rgba_out.as_ptr().add(off + 16).cast());

      let a_for_lo = _mm256_shuffle_epi8(a_raw_256, shuf_dst0);
      let a_for_hi = _mm256_shuffle_epi8(a_raw_256, shuf_dst1);

      let merged_lo = _mm256_blendv_epi8(dst_lo, a_for_lo, alpha_mask_u16);
      let merged_hi = _mm256_blendv_epi8(dst_hi, a_for_hi, alpha_mask_u16);

      _mm256_storeu_si256(rgba_out.as_mut_ptr().add(off).cast(), merged_lo);
      _mm256_storeu_si256(rgba_out.as_mut_ptr().add(off + 16).cast(), merged_hi);
      x += 8;
    }

    if x < width {
      scalar::copy_alpha_plane_u16::<BITS>(
        &alpha[x..width],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use crate::row::scalar::alpha_extract as scalar;

  fn pseudo_random_u8(out: &mut [u8], seed: u32) {
    let mut state = seed;
    for v in out.iter_mut() {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (state >> 16) as u8;
    }
  }

  fn pseudo_random_u16(out: &mut [u16], seed: u32) {
    let mut state = seed;
    for v in out.iter_mut() {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (state >> 8) as u16;
    }
  }

  // Covers 8-px main-loop block + scalar tail for various widths,
  // including SSE-block-aligned (4) and AVX2-block-aligned (8) edges.
  const WIDTHS: &[usize] = &[
    1, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 64, 128, 130,
  ];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx2_copy_alpha_packed_u8x4_at_3_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut packed, 0xC0FFEE);
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0xDECAF);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_packed_u8x4_at_3(&packed, &mut rgba_simd, w) };
      scalar::copy_alpha_packed_u8x4_at_3(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx2_copy_alpha_packed_u16x4_to_u8_at_0_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut packed, 0xCAB00D);
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0xFEED);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_packed_u16x4_to_u8_at_0(&packed, &mut rgba_simd, w) };
      scalar::copy_alpha_packed_u16x4_to_u8_at_0(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx2_copy_alpha_packed_u16x4_at_0_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut packed, 0xBEEF11);
      let mut rgba_simd = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut rgba_simd, 0x1337);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_packed_u16x4_at_0(&packed, &mut rgba_simd, w) };
      scalar::copy_alpha_packed_u16x4_at_0(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx2_copy_alpha_plane_u8_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut alpha = std::vec![0u8; w];
      pseudo_random_u8(&mut alpha, 0xABCDEF);
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0x123456);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_plane_u8(&alpha, &mut rgba_simd, w) };
      scalar::copy_alpha_plane_u8(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx2_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits10() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xC0DE);
      for v in alpha.iter_mut() {
        *v &= 0x03FF;
      }
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0xBABE);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_plane_u16_to_u8::<10>(&alpha, &mut rgba_simd, w) };
      scalar::copy_alpha_plane_u16_to_u8::<10>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx2_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits12() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xF00BAA);
      for v in alpha.iter_mut() {
        *v &= 0x0FFF;
      }
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0x5EED);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_plane_u16_to_u8::<12>(&alpha, &mut rgba_simd, w) };
      scalar::copy_alpha_plane_u16_to_u8::<12>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx2_copy_alpha_plane_u16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx2") {
      return;
    }
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xDEADBE);
      let mut rgba_simd = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut rgba_simd, 0xFADE);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_plane_u16::<10>(&alpha, &mut rgba_simd, w) };
      scalar::copy_alpha_plane_u16::<10>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }
}
