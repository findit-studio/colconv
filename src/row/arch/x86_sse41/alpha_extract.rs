//! SSE4.1 α-extract helpers — SIMD parity of `crate::row::scalar::alpha_extract`.
//!
//! Each fn matches its scalar counterpart byte-for-byte (verified by
//! `*_matches_scalar_widths` tests in this file).
//!
//! # Strategy
//!
//! SSE4.1 lacks the structured-load/store intrinsics that NEON provides, so
//! we instead use load + mask/blend to touch only the α byte of each pixel.
//!
//! **u8 helpers** (helpers 1 and 4): block = 4 px / iter (one `__m128i` of
//! RGBA = 16 bytes = 4 px × 4 ch). An α-slot mask
//! `[0,0,0,0xFF, 0,0,0,0xFF, 0,0,0,0xFF, 0,0,0,0xFF]` + `_mm_blendv_epi8`
//! replaces only the α bytes. Helper 1 additionally needs to extract the α
//! byte from slot 3 of a VUYA packed buffer; helper 4 reads a flat plane.
//!
//! **u16→u8 helpers** (helpers 2 and 5): block = 4 px / iter for helper 2
//! (two `__m128i` of u16 RGBA = 32 bytes = 4 px × 4 ch × 2 bytes). Helper
//! 5 loads 8 u16 α lanes, shifts, narrows to 8 u8 bytes via
//! `_mm_packus_epi16`, then scatters 4 at a time into RGBA u8 output.
//!
//! **u16 helpers** (helpers 3 and 6): block = 4 px / iter (two `__m128i` of
//! u16 RGBA). An α-slot mask for u16 (`[0x00,0x00, 0x00,0x00, 0x00,0x00,
//! 0xFF,0xFF, ...]` per pixel) + `_mm_blendv_epi8` substitutes the α u16.
//!
//! **BITS shift** in helper 5: `_mm_srl_epi16` accepts a variable-count
//! `__m128i`; we build it once via `_mm_cvtsi32_si128(BITS as i32 - 8)`.
//! This avoids per-BITS monomorphization with match and a const-generic
//! `_mm_srli_epi16` literal.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::row::scalar::alpha_extract as scalar;

// ---------------------------------------------------------------------------
// Helper 1: VUYA u8 → u8 RGBA  (α at packed slot 3)
// ---------------------------------------------------------------------------

/// VUYA → u8 RGBA: gather α from `packed[3 + 4*n]` into `rgba_out[3 + 4*n]`.
///
/// Block: 4 px / iter via `_mm_blendv_epi8` with an α-slot mask.
///
/// # Safety
///
/// SSE4.1 must be available. Both slices must be `>= width * 4` bytes.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn copy_alpha_packed_u8x4_at_3(packed: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  unsafe {
    // Mask: 0xFF in α (byte 3 of every pixel quadruple), 0x00 elsewhere.
    // _mm_set_epi8 takes args high-to-low (byte 15 first, byte 0 last).
    // 0xFF positions align to RGBA α slots: bytes 15, 11, 7, 3.
    let alpha_mask = _mm_set_epi8(
      -1, 0, 0, 0, // bytes 15..12 (px3: α at byte 15)
      -1, 0, 0, 0, // bytes 11..8  (px2: α at byte 11)
      -1, 0, 0, 0, // bytes  7..4  (px1: α at byte  7)
      -1, 0, 0, 0, // bytes  3..0  (px0: α at byte  3)
    );

    let mut x = 0usize;
    while x + 4 <= width {
      let off = x * 4;
      let src = _mm_loadu_si128(packed.as_ptr().add(off).cast());
      let dst = _mm_loadu_si128(rgba_out.as_ptr().add(off).cast());
      // blendv: where mask byte has high bit set (0xFF), pick src; else dst.
      let merged = _mm_blendv_epi8(dst, src, alpha_mask);
      _mm_storeu_si128(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 4;
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
// Helper 2: AYUV64 u16 → u8 RGBA  (α at packed slot 0, depth >> 8)
// ---------------------------------------------------------------------------

/// AYUV64 → u8 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> 8`.
///
/// Block: 4 px / iter. Each u16 α is right-shifted 8 and narrowed to u8,
/// then blended into the α slot of the u8 RGBA output.
///
/// # Safety
///
/// SSE4.1 must be available. `packed.len() >= width * 4`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn copy_alpha_packed_u16x4_to_u8_at_0(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  unsafe {
    // Mask for u8 output α slot (byte 3 of each 4-byte pixel).
    let alpha_mask = _mm_set_epi8(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0);
    // Shift count: >> 8
    let shr8 = _mm_cvtsi32_si128(8);

    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 AYUV64 pixels = 16 u16 = 2 × __m128i.
      let src_off = x * 4;
      let lo = _mm_loadu_si128(packed.as_ptr().add(src_off).cast()); // px 0-1 (8 u16)
      let hi = _mm_loadu_si128(packed.as_ptr().add(src_off + 8).cast()); // px 2-3 (8 u16)

      // α is at u16 slot 0 of each pixel: lanes 0 and 4 of each __m128i.
      // Right-shift by 8 to get the high byte.
      let lo_shr = _mm_srl_epi16(lo, shr8); // u16: [A0>>8, Y0>>8, U0>>8, V0>>8, A1>>8, ...]
      let hi_shr = _mm_srl_epi16(hi, shr8); // u16: [A2>>8, ..., A3>>8, ...]

      // Pack u16→u8 (unsaturated within 0..=255 since we shifted right).
      // _mm_packus_epi16 takes two i16x8 and narrows to u8x16.
      let packed_u8 = _mm_packus_epi16(lo_shr, hi_shr);
      // Now packed_u8 byte layout: [A0,Y0,U0,V0, A1,Y1,U1,V1, A2,Y2,U2,V2, A3,Y3,U3,V3]
      // We want byte 0 of each 4-byte group → that's bytes 0,4,8,12.
      // Shuffle to place α at slot 3 of each output pixel:
      // out[3+4*n] = packed_u8[0+4*n]. Shuffle: for each 4-byte output px,
      // take byte 0 from packed_u8 and place at byte 3.
      // _mm_set_epi8 takes args high-to-low (byte 15 first, byte 0 last).
      // Each group places one α byte at the high position of a 4-byte output pixel.
      let shuf_mask = _mm_set_epi8(
        12, -1, -1, -1, // bytes 15..12 (px3): src[12] → out[15]
        8, -1, -1, -1, // bytes 11..8  (px2): src[ 8] → out[11]
        4, -1, -1, -1, // bytes  7..4  (px1): src[ 4] → out[ 7]
        0, -1, -1, -1, // bytes  3..0  (px0): src[ 0] → out[ 3]
      );
      let a_scattered = _mm_shuffle_epi8(packed_u8, shuf_mask);

      // Load existing rgba_out for 4 px and blend.
      let dst_off = x * 4;
      let dst = _mm_loadu_si128(rgba_out.as_ptr().add(dst_off).cast());
      let merged = _mm_blendv_epi8(dst, a_scattered, alpha_mask);
      _mm_storeu_si128(rgba_out.as_mut_ptr().add(dst_off).cast(), merged);
      x += 4;
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
// Helper 3: AYUV64 u16 → u16 RGBA  (α at packed slot 0, no depth conv)
// ---------------------------------------------------------------------------

/// AYUV64 → u16 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// Block: 4 px / iter (two `__m128i` of u16 RGBA = 32 bytes).
///
/// # Safety
///
/// SSE4.1 must be available. Both slices `>= width * 4` elements.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn copy_alpha_packed_u16x4_at_0(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  unsafe {
    // u16 α-slot mask: 0xFFFF at slot 3 of each 4-u16 pixel group.
    // In bytes: last 2 bytes of every 8-byte group are 0xFF.
    // __m128i holds 8 u16 = 2 pixels; α is at u16 slots 3 and 7.
    let alpha_mask_u16 = _mm_set_epi16(-1, 0, 0, 0, -1, 0, 0, 0);

    // Shuffle: extract α from slot 0 of each AYUV64 pixel and place at slot 3.
    // For 2 pixels per __m128i (u16 slots 0-7):
    //   src slot 0 → dst slot 3  (px0)
    //   src slot 4 → dst slot 7  (px1)
    // In byte indices (u16 slot k = bytes 2k, 2k+1):
    //   slot 0 = bytes 0,1 → slot 3 = bytes 6,7
    //   slot 4 = bytes 8,9 → slot 7 = bytes 14,15
    // _mm_set_epi8 takes args high-to-low (byte 15 first, byte 0 last).
    // Each __m128i holds 2 pixels (px0 = slots 0-3, px1 = slots 4-7).
    // α (slot 0) → slot 3: src[0,1] → dst[6,7]; src[8,9] → dst[14,15].
    let shuf_mask = _mm_set_epi8(
      9, 8, -1, -1, -1, -1, -1, -1, // bytes 15..8 (px1): src[8,9] → bytes 14,15
      1, 0, -1, -1, -1, -1, -1, -1, // bytes  7..0 (px0): src[0,1] → bytes  6,7
    );

    let mut x = 0usize;
    while x + 4 <= width {
      let off = x * 4;
      // Two __m128i cover 4 pixels of u16 RGBA output.
      let src_lo = _mm_loadu_si128(packed.as_ptr().add(off).cast()); // px0,px1 of packed
      let src_hi = _mm_loadu_si128(packed.as_ptr().add(off + 8).cast()); // px2,px3 of packed
      let dst_lo = _mm_loadu_si128(rgba_out.as_ptr().add(off).cast());
      let dst_hi = _mm_loadu_si128(rgba_out.as_ptr().add(off + 8).cast());

      // Extract α (slot 0) from packed and place at slot 3 via shuffle.
      let a_lo = _mm_shuffle_epi8(src_lo, shuf_mask);
      let a_hi = _mm_shuffle_epi8(src_hi, shuf_mask);

      // Blend: where alpha_mask_u16 has high bit (0xFF byte), use a_lo/a_hi.
      let merged_lo = _mm_blendv_epi8(dst_lo, a_lo, alpha_mask_u16);
      let merged_hi = _mm_blendv_epi8(dst_hi, a_hi, alpha_mask_u16);

      _mm_storeu_si128(rgba_out.as_mut_ptr().add(off).cast(), merged_lo);
      _mm_storeu_si128(rgba_out.as_mut_ptr().add(off + 8).cast(), merged_hi);
      x += 4;
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
// Helper 4: α plane u8 → u8 RGBA
// ---------------------------------------------------------------------------

/// Yuva420p / 422p / 444p u8 → u8 RGBA: scatter α plane into
/// `rgba_out[3 + 4*n]`.
///
/// Block: 4 px / iter via blend with α-slot mask.
///
/// # Safety
///
/// SSE4.1 must be available. `alpha.len() >= width`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  unsafe {
    // α-slot mask for u8 RGBA: 0xFF at byte 3 of each 4-byte pixel.
    let alpha_mask = _mm_set_epi8(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0);
    // Shuffle: scatter 4 contiguous α bytes into slot 3 of each 4-byte pixel.
    // Input: [a0, a1, a2, a3, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?]  (only low 4 bytes valid)
    // Output slot 3: a0→byte3, a1→byte7, a2→byte11, a3→byte15; others zero.
    let shuf_mask = _mm_set_epi8(
      3, -1, -1, -1, // px3
      2, -1, -1, -1, // px2
      1, -1, -1, -1, // px1
      0, -1, -1, -1, // px0
    );

    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 α bytes into low 4 bytes of a register.
      let a_raw = _mm_cvtsi32_si128(i32::from_le_bytes([
        *alpha.get_unchecked(x),
        *alpha.get_unchecked(x + 1),
        *alpha.get_unchecked(x + 2),
        *alpha.get_unchecked(x + 3),
      ]));
      let a_scattered = _mm_shuffle_epi8(a_raw, shuf_mask);

      let off = x * 4;
      let dst = _mm_loadu_si128(rgba_out.as_ptr().add(off).cast());
      let merged = _mm_blendv_epi8(dst, a_scattered, alpha_mask);
      _mm_storeu_si128(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 4;
    }

    if x < width {
      scalar::copy_alpha_plane_u8(&alpha[x..width], &mut rgba_out[x * 4..width * 4], width - x);
    }
  }
}

// ---------------------------------------------------------------------------
// Helper 5: α plane u16 → u8 RGBA  (depth-conv >> (BITS-8))
// ---------------------------------------------------------------------------

/// Yuva*p9/10/12/14 → u8 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> (BITS - 8)`.
///
/// Uses `_mm_srl_epi16` with a runtime count vector `_mm_cvtsi32_si128(BITS - 8)`
/// to avoid per-BITS monomorphization. Block: 4 px / iter (load 4 u16 α,
/// shift, narrow via `_mm_packus_epi16`, scatter into u8 RGBA).
///
/// # Safety
///
/// SSE4.1 must be available. `alpha.len() >= width`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
    // BITS-bit canonicalization mask: AND'd before shift so over-range
    // source α samples don't leak through (matches scalar parity).
    let bits_mask = _mm_set1_epi16(((1u32 << BITS) - 1) as i16);
    // α-slot mask for u8 RGBA output.
    let alpha_mask = _mm_set_epi8(-1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0);
    // Scatter 4 u8 α values (in low 4 bytes after pack) into α slots.
    // After _mm_packus_epi16(4_u16s, zero), we get [a0,a1,a2,a3,0,0,0,0,...].
    // Shuffle: a0→byte3, a1→byte7, a2→byte11, a3→byte15.
    let shuf_mask = _mm_set_epi8(
      3, -1, -1, -1, // px3
      2, -1, -1, -1, // px2
      1, -1, -1, -1, // px1
      0, -1, -1, -1, // px0
    );

    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 u16 α values (8 bytes).
      let a_u16 = _mm_loadl_epi64(alpha.as_ptr().add(x).cast()); // [a0,a1,a2,a3, 0,0,0,0]
      // Mask to low BITS before shift (over-range α canonicalization).
      let a_masked = _mm_and_si128(a_u16, bits_mask);
      // Right-shift by (BITS - 8).
      let a_shifted = _mm_srl_epi16(a_masked, shr_count);
      // Narrow u16→u8 (values are in [0,255] after shift).
      let a_u8_vec = _mm_packus_epi16(a_shifted, _mm_setzero_si128()); // [a0,a1,a2,a3, 0,...]
      // Scatter into α slots.
      let a_scattered = _mm_shuffle_epi8(a_u8_vec, shuf_mask);

      let off = x * 4;
      let dst = _mm_loadu_si128(rgba_out.as_ptr().add(off).cast());
      let merged = _mm_blendv_epi8(dst, a_scattered, alpha_mask);
      _mm_storeu_si128(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 4;
    }

    if x < width {
      // Scalar tail uses `BE = false`: this SSE4.1 helper does host-native
      // u16 loads (`_mm_loadl_epi64`), which match LE-on-disk only on LE
      // hosts. The dispatcher routes BE = true directly to scalar (see
      // `dispatch::alpha_extract`), so the SIMD path here is BE = false by
      // construction.
      scalar::copy_alpha_plane_u16_to_u8::<BITS, false>(
        &alpha[x..width],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Helper 6: α plane u16 → u16 RGBA  (no depth conv)
// ---------------------------------------------------------------------------

/// Yuva*p9/10/12/14/16 → u16 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// Block: 4 px / iter (two `__m128i` of u16 RGBA = 32 bytes). α u16
/// values are shuffled from a flat plane into slot 3 of each pixel tuple.
///
/// # Safety
///
/// SSE4.1 must be available. `alpha.len() >= width`; `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
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

  unsafe {
    // BITS-bit canonicalization mask: AND'd before scatter so over-range
    // source α samples don't leak through (matches scalar parity).
    let bits_mask = _mm_set1_epi16(((1u32 << BITS) - 1) as i16);
    // u16 α-slot mask: 0xFFFF at u16 slot 3 and 7 (= 2 pixels per __m128i).
    // In bytes: mask bytes 6,7 and 14,15.
    let alpha_mask_u16 = _mm_set_epi16(-1, 0, 0, 0, -1, 0, 0, 0);

    // Scatter α u16 values into slot 3 of each 4-u16 pixel within __m128i.
    // __m128i holds 2 pixels (u16 slots 0-7 where px0=slots 0-3, px1=slots 4-7).
    // We load 4 α u16 into low 8 bytes: [a0, a1, a2, a3, 0, 0, 0, 0].
    // Per __m128i we want: α placed at slots 3 (bytes 6,7) and 7 (bytes 14,15).
    // lo __m128i: a0 → slot3 (bytes 6,7), a1 → slot7 (bytes 14,15).
    // hi __m128i: a2 → slot3 (bytes 6,7), a3 → slot7 (bytes 14,15).
    let shuf_lo = _mm_set_epi8(
      3, 2, -1, -1, -1, -1, -1, -1, // px1: bytes 2,3 → bytes 14,15
      1, 0, -1, -1, -1, -1, -1, -1, // px0: bytes 0,1 → bytes 6,7
    );
    let shuf_hi = _mm_set_epi8(
      7, 6, -1, -1, -1, -1, -1, -1, // px3: bytes 6,7 → bytes 14,15
      5, 4, -1, -1, -1, -1, -1, -1, // px2: bytes 4,5 → bytes 6,7
    );

    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 α u16 = 8 bytes into low 64 bits of a register, then
      // canonicalize over-range bits (matches scalar parity).
      let a_raw = _mm_and_si128(
        _mm_loadl_epi64(alpha.as_ptr().add(x).cast()), // [a0,a1,a2,a3, 0,0,0,0]
        bits_mask,
      );

      // Scatter into the two __m128i blocks (lo covers px0,px1; hi covers px2,px3).
      let a_lo = _mm_shuffle_epi8(a_raw, shuf_lo);
      let a_hi = _mm_shuffle_epi8(a_raw, shuf_hi);

      let off = x * 4;
      let dst_lo = _mm_loadu_si128(rgba_out.as_ptr().add(off).cast());
      let dst_hi = _mm_loadu_si128(rgba_out.as_ptr().add(off + 8).cast());

      let merged_lo = _mm_blendv_epi8(dst_lo, a_lo, alpha_mask_u16);
      let merged_hi = _mm_blendv_epi8(dst_hi, a_hi, alpha_mask_u16);

      _mm_storeu_si128(rgba_out.as_mut_ptr().add(off).cast(), merged_lo);
      _mm_storeu_si128(rgba_out.as_mut_ptr().add(off + 8).cast(), merged_hi);
      x += 4;
    }

    if x < width {
      // Scalar tail uses `BE = false`: see `copy_alpha_plane_u16_to_u8` above.
      scalar::copy_alpha_plane_u16::<BITS, false>(
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

  // Covers 4-px main-loop block + scalar tail for various widths.
  const WIDTHS: &[usize] = &[
    1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 128, 130,
  ];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn sse41_copy_alpha_packed_u8x4_at_3_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
  fn sse41_copy_alpha_packed_u16x4_to_u8_at_0_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
  fn sse41_copy_alpha_packed_u16x4_at_0_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
  fn sse41_copy_alpha_plane_u8_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
  fn sse41_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits10() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
      // SIMD reads native u16; pair with scalar BE = false (LE-on-LE-host).
      scalar::copy_alpha_plane_u16_to_u8::<10, false>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn sse41_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits12() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
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
      // SIMD reads native u16; pair with scalar BE = false (LE-on-LE-host).
      scalar::copy_alpha_plane_u16_to_u8::<12, false>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn sse41_copy_alpha_plane_u16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
      return;
    }
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xDEADBE);
      let mut rgba_simd = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut rgba_simd, 0xFADE);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_plane_u16::<10>(&alpha, &mut rgba_simd, w) };
      // SIMD reads native u16; pair with scalar BE = false (LE-on-LE-host).
      scalar::copy_alpha_plane_u16::<10, false>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }
}
