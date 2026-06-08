//! AVX-512 (F+BW) α-extract helpers — SIMD parity of `crate::row::scalar::alpha_extract`.
//!
//! Each fn matches its scalar counterpart byte-for-byte (verified by
//! `*_matches_scalar_widths` tests in this file).
//!
//! # Strategy
//!
//! AVX-512 doubles the AVX2 register width (64 bytes / 32 u16 per
//! `__m512i`) and adds first-class mask registers (`__mmask64` /
//! `__mmask32`), making α-slot substitution especially clean: instead
//! of building a byte-vector mask and using `blendv`, we compute a
//! single 64-bit (or 32-bit) bitmask whose set bits mark α positions
//! and use `_mm512_mask_blend_epi8` / `_mm512_mask_blend_epi16`.
//!
//! **u8 helpers** (helpers 1, 4, 5): block = 16 px / iter (one
//! `__m512i` of RGBA = 64 bytes = 16 px × 4 ch). The α-slot bitmask is
//! `0x8888_8888_8888_8888u64` — every 4th bit set, starting at bit 3
//! (which is the α byte of pixel 0). Helper 1 supplies α from a packed
//! VUYA buffer; helpers 4 and 5 supply α from a separate plane. For
//! helpers 4 / 5 the α bytes start out contiguous (16 in a `__m128i`)
//! and we use `_mm512_cvtepu8_epi32` to widen each α byte into its own
//! u32 lane (so each u32 lane holds exactly one α byte), then shift
//! left by 24 to move that byte from lane offset 0 to lane offset 3 —
//! which IS the α slot of the corresponding 4-byte RGBA pixel. The
//! mask blend then writes only those positions into `rgba_out`.
//!
//! **u16 / u16->u8 helpers** (helpers 2, 3, 6): block = 16 px / iter.
//! Helper 2 loads 2 × `__m512i` of u16 (= 16 px × 4 ch), shifts each
//! u16 right by 8, narrows to one `__m512i` of u8 via
//! `_mm512_packus_epi16`. `packus_epi16` is per-128-bit-lane and
//! produces a lane-split byte stream, so we apply the standard
//! `pack_fixup` permute (`_mm512_permutexvar_epi64` with index
//! `[0, 2, 4, 6, 1, 3, 5, 7]`) to restore natural order. After fixup
//! the byte stream is `[A0, Y0, U0, V0, A1, ..., A15, Y15, U15, V15]`,
//! matching helper 1's layout — we then run the same per-lane
//! `_mm512_shuffle_epi8` + `_mm512_mask_blend_epi8` pattern as helper
//! 1 (with shuffle indices that copy the α byte from byte offset 0 of
//! each 4-byte group to byte offset 3). Helper 3 stays in u16
//! throughout (2 × `__m512i` of u16 RGBA per iter); each 4-u16 pixel
//! sits inside a single 128-bit lane so per-lane `_mm512_shuffle_epi8`
//! moves α from slot 0 to slot 3 with no cross-lane fixup. Helper 6
//! takes 16 contiguous α u16 (one `__m256i`) and widens each 8-u16
//! half via `_mm512_cvtepu16_epi64` so each u64 lane holds one α u16
//! in its low 16 bits; a left shift by 48 moves that u16 from u16
//! offset 0 of the lane to u16 offset 3 — IS the α slot — then a
//! u16-granularity mask blend (`__mmask32 = 0x8888_8888`) substitutes
//! into rgba_out.
//!
//! # Mask register vs blendv
//!
//! AVX-512's mask blend (`_mm512_mask_blend_epi8(k, a, b)`) picks
//! `b[i]` where bit `i` of `k` is set, `a[i]` otherwise. This is
//! semantically equivalent to AVX2's `_mm256_blendv_epi8` but uses an
//! 8× cheaper bitmask instead of a 64-byte vector mask, and avoids
//! the AVX2 byte-mask construction overhead.
//!
//! # F+BW baseline
//!
//! No AVX-512VBMI ops are used. Cross-lane permutes are limited to
//! `_mm512_permutexvar_epi64` (F) and `_mm512_cvtepu8_epi32` /
//! `_mm512_cvtepu16_epi64` (F sign/zero-extend). Per-lane byte ops use
//! `_mm512_shuffle_epi8` (BW). Mask blends use
//! `_mm512_mask_blend_epi8` (BW) and `_mm512_mask_blend_epi16` (BW).

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::row::scalar::alpha_extract as scalar;

// Helper 1: VUYA u8 -> u8 RGBA  (α at packed slot 3).
/// VUYA -> u8 RGBA: gather α from `packed[3 + 4*n]` into `rgba_out[3 + 4*n]`.
///
/// Block: 16 px / iter via `_mm512_mask_blend_epi8` with an α-slot bitmask.
///
/// # Safety
///
/// AVX-512F + AVX-512BW must be available. Both slices must be `>= width * 4` bytes.
#[cfg(feature = "yuv-444-packed")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn copy_alpha_packed_u8x4_at_3(packed: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  // SAFETY: caller obligation — AVX-512F + AVX-512BW are available; both
  // slices have at least `width * 4` bytes.
  unsafe {
    // α-slot bitmask: every 4th bit set starting at bit 3 — i.e. bits
    // 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63.
    // In hex: 0x8888_8888_8888_8888. Each set bit corresponds to the α
    // byte (offset 3 of each 4-byte pixel) of the 16 packed pixels.
    const ALPHA_MASK_U8: __mmask64 = 0x8888_8888_8888_8888u64;

    let mut x = 0usize;
    while x + 16 <= width {
      let off = x * 4;
      let src = _mm512_loadu_si512(packed.as_ptr().add(off).cast());
      let dst = _mm512_loadu_si512(rgba_out.as_ptr().add(off).cast());
      // mask_blend: where mask bit i is set, take `src[i]`; else `dst[i]`.
      let merged = _mm512_mask_blend_epi8(ALPHA_MASK_U8, dst, src);
      _mm512_storeu_si512(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 16;
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

// Helper 2: AYUV64 u16 -> u8 RGBA  (α at packed slot 0, depth >> 8).
/// AYUV64 -> u8 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> 8`.
///
/// Block: 16 px / iter. We load 2 × `__m512i` of u16 (= 32 u16 each,
/// totaling 64 u16 = 16 px × 4 ch), right-shift each by 8 and narrow
/// to one `__m512i` of u8 via `_mm512_packus_epi16` followed by the
/// standard `pack_fixup` permute. After fixup the byte stream is
/// `[A0, Y0, U0, V0, A1, ..., A15, Y15, U15, V15]`. A per-lane
/// `_mm512_shuffle_epi8` then copies α bytes from byte offset 0 of
/// each 4-byte group to byte offset 3, and a mask blend writes the
/// result into `rgba_out`.
///
/// # Safety
///
/// AVX-512F + AVX-512BW must be available. `packed.len() >= width * 4`
/// (u16 elements); `rgba_out.len() >= width * 4` (u8 bytes).
#[cfg(feature = "yuv-444-packed")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn copy_alpha_packed_u16x4_to_u8_at_0(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  // SAFETY: caller obligation — AVX-512F + AVX-512BW are available; both
  // slices have the required lengths.
  unsafe {
    // α-slot bitmask for u8 RGBA (16 px = 64 bytes): bits 3, 7, 11, ..., 63.
    const ALPHA_MASK_U8: __mmask64 = 0x8888_8888_8888_8888u64;

    // Pack-fixup permute for `_mm512_packus_epi16`: that intrinsic is
    // per-128-bit-lane and produces 64-bit chunks in lane order
    // `[lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3]`. Permuting via index
    // `[0, 2, 4, 6, 1, 3, 5, 7]` reorders to natural
    // `[lo0..3 contiguous, hi0..3 contiguous]`.
    let pack_fixup = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);

    // Per-lane shuffle: each 128-bit lane holds 4 pixels (= 16 bytes,
    // each pixel = `[A, Y, U, V]`). For each pixel within the lane,
    // copy byte 0 (= α) to byte 3 (= α slot of RGBA pixel). Other
    // bytes don't matter — mask_blend only reads the α-slot bytes.
    // `_mm512_set_epi8` takes args high-to-low (byte 63 first, byte 0 last);
    // we replicate the same 16-byte pattern across all 4 lanes.
    #[rustfmt::skip]
    let shuf_mask = _mm512_set_epi8(
      // lane 3 (bytes 63..48) — pixels 12..15 of the lane-fixed stream
      12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1, -1, -1,
      // lane 2 (bytes 47..32) — pixels  8..11
      12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1, -1, -1,
      // lane 1 (bytes 31..16) — pixels  4.. 7
      12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1, -1, -1,
      // lane 0 (bytes 15.. 0) — pixels  0.. 3
      12, -1, -1, -1,  8, -1, -1, -1,  4, -1, -1, -1,  0, -1, -1, -1,
    );

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 AYUV64 pixels = 64 u16 = 2 × __m512i.
      let src_off = x * 4;
      let lo = _mm512_loadu_si512(packed.as_ptr().add(src_off).cast()); // px 0..7  (32 u16)
      let hi = _mm512_loadu_si512(packed.as_ptr().add(src_off + 32).cast()); // px 8..15

      // Right-shift each u16 by 8 to bring the high byte (= u8 α) to the low byte.
      let lo_shr = _mm512_srli_epi16::<8>(lo);
      let hi_shr = _mm512_srli_epi16::<8>(hi);

      // Narrow u16 -> u8 (values fit in [0, 255] after >> 8). Per-128-bit
      // lane intrinsic; `pack_fixup` restores natural element order so
      // the byte stream becomes `[A0, Y0, U0, V0, A1, ..., V15]`.
      let packed_u8 = _mm512_permutexvar_epi64(pack_fixup, _mm512_packus_epi16(lo_shr, hi_shr));

      // Per-lane shuffle scatters the α bytes from byte offset 0 of each
      // 4-byte group into byte offset 3 (the α slot of RGBA).
      let a_scattered = _mm512_shuffle_epi8(packed_u8, shuf_mask);

      // Load existing rgba_out for 16 px and mask-blend.
      let dst_off = x * 4;
      let dst = _mm512_loadu_si512(rgba_out.as_ptr().add(dst_off).cast());
      let merged = _mm512_mask_blend_epi8(ALPHA_MASK_U8, dst, a_scattered);
      _mm512_storeu_si512(rgba_out.as_mut_ptr().add(dst_off).cast(), merged);
      x += 16;
    }

    if x < width {
      scalar::copy_alpha_packed_u16x4_to_u8_at_0::<false>(
        &packed[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// Helper 3: AYUV64 u16 -> u16 RGBA  (α at packed slot 0, no depth conv).
/// AYUV64 -> u16 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// Block: 16 px / iter (two `__m512i` of u16 RGBA = 128 bytes total).
/// Each 4-u16 pixel sits inside one 128-bit lane, so the per-lane
/// `_mm512_shuffle_epi8` moves α between u16 slot 0 and u16 slot 3
/// within the pixel without cross-lane concerns. A u16-granularity
/// mask blend (`_mm512_mask_blend_epi16` with `__mmask32`) substitutes
/// only the α u16 of each pixel into rgba_out.
///
/// # Safety
///
/// AVX-512F + AVX-512BW must be available. Both slices `>= width * 4` u16.
#[cfg(feature = "yuv-444-packed")]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn copy_alpha_packed_u16x4_at_0(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  // SAFETY: caller obligation — AVX-512F + AVX-512BW are available; both
  // slices have `width * 4` u16 elements available.
  unsafe {
    // u16-granularity α-slot bitmask: each `__m512i` holds 32 u16 = 8
    // pixels; α u16 is at u16 lane 3 of each 4-u16 pixel — i.e. lanes
    // 3, 7, 11, 15, 19, 23, 27, 31. As a 32-bit bitmask:
    // 0x8888_8888.
    const ALPHA_MASK_U16: __mmask32 = 0x8888_8888u32;

    // Per-128-bit-lane shuffle (operates on bytes within each lane).
    // Each lane holds 2 pixels (= 8 u16 = 16 bytes). For each pixel:
    //   src u16 slot 0 (bytes 0,1 of pixel) -> dst u16 slot 3 (bytes 6,7)
    // For pixel-local 0 in the lane: src bytes 0,1 -> dst bytes 6,7.
    // For pixel-local 1 in the lane: src bytes 8,9 -> dst bytes 14,15.
    // The four 128-bit lanes share the same per-lane byte pattern.
    // `_mm512_set_epi8` takes args high-to-low (byte 63 first, byte 0 last).
    #[rustfmt::skip]
    let shuf_mask = _mm512_set_epi8(
      // lane 3 (bytes 63..48): same pattern, two pixels per lane.
      9, 8, -1, -1, -1, -1, -1, -1,  1, 0, -1, -1, -1, -1, -1, -1,
      // lane 2 (bytes 47..32):
      9, 8, -1, -1, -1, -1, -1, -1,  1, 0, -1, -1, -1, -1, -1, -1,
      // lane 1 (bytes 31..16):
      9, 8, -1, -1, -1, -1, -1, -1,  1, 0, -1, -1, -1, -1, -1, -1,
      // lane 0 (bytes 15.. 0):
      9, 8, -1, -1, -1, -1, -1, -1,  1, 0, -1, -1, -1, -1, -1, -1,
    );

    let mut x = 0usize;
    while x + 16 <= width {
      let off = x * 4;
      // Two __m512i cover 16 pixels of u16 RGBA output (32 u16 each).
      let src_lo = _mm512_loadu_si512(packed.as_ptr().add(off).cast()); // px 0..7
      let src_hi = _mm512_loadu_si512(packed.as_ptr().add(off + 32).cast()); // px 8..15
      let dst_lo = _mm512_loadu_si512(rgba_out.as_ptr().add(off).cast());
      let dst_hi = _mm512_loadu_si512(rgba_out.as_ptr().add(off + 32).cast());

      // Move α from u16 slot 0 to u16 slot 3 within each pixel (per-lane).
      let a_lo = _mm512_shuffle_epi8(src_lo, shuf_mask);
      let a_hi = _mm512_shuffle_epi8(src_hi, shuf_mask);

      // u16-granularity mask blend: where mask bit i is set, take `a_*`
      // u16 lane i; else keep `dst_*` u16 lane i.
      let merged_lo = _mm512_mask_blend_epi16(ALPHA_MASK_U16, dst_lo, a_lo);
      let merged_hi = _mm512_mask_blend_epi16(ALPHA_MASK_U16, dst_hi, a_hi);

      _mm512_storeu_si512(rgba_out.as_mut_ptr().add(off).cast(), merged_lo);
      _mm512_storeu_si512(rgba_out.as_mut_ptr().add(off + 32).cast(), merged_hi);
      x += 16;
    }

    if x < width {
      scalar::copy_alpha_packed_u16x4_at_0::<false>(
        &packed[x * 4..width * 4],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

// Helper 4: α plane u8 -> u8 RGBA.
/// Yuva420p / 422p / 444p u8 -> u8 RGBA: scatter α plane into
/// `rgba_out[3 + 4*n]`.
///
/// Block: 16 px / iter. The 16 contiguous α bytes are widened into a
/// `__m512i` via `_mm512_cvtepu8_epi32` so each u32 lane holds one α
/// byte in its low byte (= byte offset 0 of the 4-byte pixel). A
/// `_mm512_slli_epi32::<24>` moves that byte from offset 0 to offset
/// 3 of each 4-byte pixel — which is the α slot of RGBA. A mask blend
/// writes only those α bytes into rgba_out.
///
/// # Safety
///
/// AVX-512F + AVX-512BW must be available. `alpha.len() >= width`;
/// `rgba_out.len() >= width * 4`.
#[cfg(any(feature = "gbr", feature = "yuva"))]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  // SAFETY: caller obligation — AVX-512F + AVX-512BW are available;
  // `alpha` has `width` bytes, `rgba_out` has `width * 4` bytes.
  unsafe {
    // α-slot bitmask for u8 RGBA (16 px / iter): bits 3, 7, ..., 63.
    const ALPHA_MASK_U8: __mmask64 = 0x8888_8888_8888_8888u64;

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 α bytes into a __m128i.
      let a_raw_128 = _mm_loadu_si128(alpha.as_ptr().add(x).cast());
      // Zero-extend each u8 into its own u32 lane: result is a __m512i
      // of 16 u32 lanes, each holding one α byte in the low byte
      // (byte offset 0 of each 4-byte u32 lane = byte offset 0 of the
      // corresponding RGBA pixel within the 64-byte vector).
      let a_widened = _mm512_cvtepu8_epi32(a_raw_128);
      // Shift left by 24 bits (3 bytes): moves the α byte from byte
      // offset 0 to byte offset 3 of each u32 lane = α slot of RGBA.
      let a_at_alpha_slot = _mm512_slli_epi32::<24>(a_widened);

      let off = x * 4;
      let dst = _mm512_loadu_si512(rgba_out.as_ptr().add(off).cast());
      let merged = _mm512_mask_blend_epi8(ALPHA_MASK_U8, dst, a_at_alpha_slot);
      _mm512_storeu_si512(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 16;
    }

    if x < width {
      scalar::copy_alpha_plane_u8(&alpha[x..width], &mut rgba_out[x * 4..width * 4], width - x);
    }
  }
}

// Helper 5: α plane u16 -> u8 RGBA  (depth-conv >> (BITS-8)).
/// Yuva*p9/10/12/14 -> u8 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> (BITS - 8)`.
///
/// Uses `_mm256_srl_epi16` with a runtime count `_mm_cvtsi32_si128(BITS - 8)`
/// to avoid per-`BITS` monomorphization. Block: 16 px / iter. We load
/// 16 u16 α (one `__m256i`), shift, then narrow via `_mm256_packus_epi16`
/// followed by `_mm256_permute4x64_epi64::<0xD8>` (per-128-bit-lane fixup)
/// to a `__m128i` of 16 u8. Final stage matches helper 4: widen via
/// `_mm512_cvtepu8_epi32`, shift left by 24, and mask blend into rgba_out.
///
/// # Safety
///
/// AVX-512F + AVX-512BW must be available. `alpha.len() >= width`;
/// `rgba_out.len() >= width * 4`.
#[cfg(any(feature = "gbr", feature = "yuva"))]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: caller obligation — AVX-512F + AVX-512BW are available;
  // `alpha` has `width` u16 elements, `rgba_out` has `width * 4` bytes.
  unsafe {
    let shr_count = _mm_cvtsi32_si128((BITS as i32) - 8);
    // BITS-bit canonicalization mask: AND'd before shift so over-range
    // source α samples don't leak through (matches scalar parity).
    let bits_mask = _mm256_set1_epi16(((1u32 << BITS) - 1) as i16);
    // α-slot bitmask for u8 RGBA (16 px / iter).
    const ALPHA_MASK_U8: __mmask64 = 0x8888_8888_8888_8888u64;

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 u16 α values (32 bytes) into a __m256i, then canonicalize
      // over-range bits (matches scalar parity).
      let a_u16 = _mm256_and_si256(_mm256_loadu_si256(alpha.as_ptr().add(x).cast()), bits_mask);
      // Right-shift by (BITS - 8). `_mm256_srl_epi16` uses the low 64
      // bits of the count vector as a single shift amount applied to
      // every 16-bit lane.
      let a_shifted = _mm256_srl_epi16(a_u16, shr_count);
      // Narrow u16 -> u8 (values fit in [0, 255] after the shift).
      // `_mm256_packus_epi16` is per-128-bit-lane and produces a
      // lane-split byte order; we restore natural order with
      // `_mm256_permute4x64_epi64::<0xD8>` (selector 0xD8 reorders
      // 64-bit chunks `[0, 1, 2, 3] -> [0, 2, 1, 3]`). The packus
      // input `_mm256_setzero_si256()` gives zeros for the high half,
      // so the final 32 bytes are: [α0..α15 in low 16, zeros in high 16].
      let a_u8_256 =
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(a_shifted, _mm256_setzero_si256()));
      // Extract the low 16 bytes (the valid α values).
      let a_u8_128 = _mm256_castsi256_si128(a_u8_256);
      // Same final stage as helper 4: widen each α byte into its own
      // u32 lane, shift left by 24 to land at byte offset 3 (α slot
      // of each 4-byte RGBA pixel), then mask blend.
      let a_widened = _mm512_cvtepu8_epi32(a_u8_128);
      let a_at_alpha_slot = _mm512_slli_epi32::<24>(a_widened);

      let off = x * 4;
      let dst = _mm512_loadu_si512(rgba_out.as_ptr().add(off).cast());
      let merged = _mm512_mask_blend_epi8(ALPHA_MASK_U8, dst, a_at_alpha_slot);
      _mm512_storeu_si512(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 16;
    }

    if x < width {
      // Scalar tail uses `BE = false`: this AVX-512 helper does host-native
      // u16 loads (`_mm256_loadu_si256`), which match LE-on-disk only on LE
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

// Helper 6: α plane u16 -> u16 RGBA  (no depth conv).
/// Yuva*p9/10/12/14/16 -> u16 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// Block: 16 px / iter. The 16 source α u16 (one `__m256i` = 32
/// bytes) split into two 8-u16 halves; each half is widened via
/// `_mm512_cvtepu16_epi64` so each u64 lane holds one α u16 in its
/// low 16 bits (= u16 offset 0 within each 4-u16 RGBA pixel of the
/// 32-u16 vector). A `_mm512_slli_epi64::<48>` moves each α u16 from
/// u16 offset 0 to u16 offset 3 of its u64 lane = the α slot of the
/// corresponding RGBA pixel. A u16-granularity mask blend
/// (`__mmask32 = 0x8888_8888`) substitutes only the α u16 of each
/// pixel.
///
/// # Safety
///
/// AVX-512F + AVX-512BW must be available. `alpha.len() >= width`;
/// `rgba_out.len() >= width * 4` (u16 elements).
#[cfg(any(feature = "gbr", feature = "yuva"))]
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
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

  // SAFETY: caller obligation — AVX-512F + AVX-512BW are available;
  // `alpha` has `width` u16, `rgba_out` has `width * 4` u16.
  unsafe {
    // BITS-bit canonicalization mask: AND'd before scatter so over-range
    // source α samples don't leak through (matches scalar parity).
    let bits_mask = _mm256_set1_epi16(((1u32 << BITS) - 1) as i16);
    // u16-granularity α-slot bitmask: each `__m512i` holds 32 u16 = 8
    // pixels; α u16 is at u16 lane 3 of each 4-u16 pixel — lanes 3,
    // 7, 11, 15, 19, 23, 27, 31. As a 32-bit bitmask: 0x8888_8888.
    const ALPHA_MASK_U16: __mmask32 = 0x8888_8888u32;

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 u16 α values (32 bytes) into a __m256i, then canonicalize
      // over-range bits (matches scalar parity).
      let a_raw_256 = _mm256_and_si256(_mm256_loadu_si256(alpha.as_ptr().add(x).cast()), bits_mask);
      // Split into two 8-u16 halves.
      let a_lo_128 = _mm256_castsi256_si128(a_raw_256); // α0..α7
      let a_hi_128 = _mm256_extracti128_si256::<1>(a_raw_256); // α8..α15

      // Each half: widen to 8 × u64 (one α u16 per u64 lane in the low
      // 16 bits), then shift left by 48 to move each α u16 from u16
      // offset 0 to u16 offset 3 of its u64 lane = α slot of RGBA.
      let a_lo_u64 = _mm512_cvtepu16_epi64(a_lo_128);
      let a_hi_u64 = _mm512_cvtepu16_epi64(a_hi_128);
      let a_lo_at_slot = _mm512_slli_epi64::<48>(a_lo_u64);
      let a_hi_at_slot = _mm512_slli_epi64::<48>(a_hi_u64);

      let off = x * 4;
      // Two output __m512i cover 16 pixels of u16 RGBA (32 u16 each).
      let dst_lo = _mm512_loadu_si512(rgba_out.as_ptr().add(off).cast());
      let dst_hi = _mm512_loadu_si512(rgba_out.as_ptr().add(off + 32).cast());

      let merged_lo = _mm512_mask_blend_epi16(ALPHA_MASK_U16, dst_lo, a_lo_at_slot);
      let merged_hi = _mm512_mask_blend_epi16(ALPHA_MASK_U16, dst_hi, a_hi_at_slot);

      _mm512_storeu_si512(rgba_out.as_mut_ptr().add(off).cast(), merged_lo);
      _mm512_storeu_si512(rgba_out.as_mut_ptr().add(off + 32).cast(), merged_hi);
      x += 16;
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

// Tests.
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

  // Covers 16-px main-loop block + scalar tail for various widths,
  // including SSE-block-aligned (4), AVX2-block-aligned (8), and
  // AVX-512-block-aligned (16) edges.
  const WIDTHS: &[usize] = &[1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 64, 128, 130];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx512_copy_alpha_packed_u8x4_at_3_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512bw")
    {
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
  fn avx512_copy_alpha_packed_u16x4_to_u8_at_0_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512bw")
    {
      return;
    }
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut packed, 0xCAB00D);
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0xFEED);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_packed_u16x4_to_u8_at_0(&packed, &mut rgba_simd, w) };
      scalar::copy_alpha_packed_u16x4_to_u8_at_0::<false>(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx512_copy_alpha_packed_u16x4_at_0_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512bw")
    {
      return;
    }
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut packed, 0xBEEF11);
      let mut rgba_simd = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut rgba_simd, 0x1337);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe { super::copy_alpha_packed_u16x4_at_0(&packed, &mut rgba_simd, w) };
      scalar::copy_alpha_packed_u16x4_at_0::<false>(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn avx512_copy_alpha_plane_u8_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512bw")
    {
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
  fn avx512_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits10() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512bw")
    {
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
  fn avx512_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits12() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512bw")
    {
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
  fn avx512_copy_alpha_plane_u16_matches_scalar_widths() {
    if !std::arch::is_x86_feature_detected!("avx512f")
      || !std::arch::is_x86_feature_detected!("avx512bw")
    {
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
