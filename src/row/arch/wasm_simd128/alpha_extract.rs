//! wasm-simd128 α-extract helpers — SIMD parity of `crate::row::scalar::alpha_extract`.
//!
//! Each fn matches its scalar counterpart byte-for-byte (verified by
//! `*_matches_scalar_widths` tests in this file).
//!
//! # Strategy
//!
//! wasm-simd128 has no structured-load/store intrinsics (unlike NEON), so
//! we use the same mask+bitselect approach as SSE4.1 but with wasm intrinsics.
//!
//! **u8 helpers** (helpers 1 and 4): block = 4 px / iter (one v128 of u8 RGBA
//! = 16 bytes = 4 px × 4 ch). An α-slot mask with 0xFF at byte 3 of each
//! pixel + `v128_bitselect` replaces only the α bytes.
//!
//! **u16→u8 helpers** (helpers 2 and 5): block = 4 px / iter. `u16x8_shr`
//! (LOGICAL right shift — NOT `i16x8_shr` arithmetic, which would corrupt
//! α values ≥ 0x8000) shifts the u16 α, then `u8x16_narrow_i16x8` narrows
//! to u8, and a scatter shuffle + bitselect writes the α slot.
//!
//! **u16 helpers** (helpers 3 and 6): block = 4 px / iter (two v128 of u16
//! RGBA). An α-slot mask for u16 (0xFFFF at u16 slot 3 of each pixel) +
//! `v128_bitselect` substitutes the α u16.
//!
//! **BITS shift** in helpers 2 and 5: `u16x8_shr(v, count)` accepts a
//! runtime `u32` count — same wasm-specific advantage used throughout
//! the y2xx / yuv_planar_high_bit kernels. No per-BITS monomorphization
//! or const-generic-literal workaround needed (unlike SSE4.1's
//! `_mm_srli_epi16::<IMM8>`).

#![cfg_attr(not(feature = "std"), allow(dead_code))]

use core::arch::wasm32::*;

use crate::row::scalar::alpha_extract as scalar;

// ---------------------------------------------------------------------------
// Helper 1: VUYA u8 → u8 RGBA  (α at packed slot 3)
// ---------------------------------------------------------------------------

/// VUYA → u8 RGBA: gather α from `packed[3 + 4*n]` into `rgba_out[3 + 4*n]`.
///
/// VUYA layout per pixel: `[V(8), U(8), Y(8), A(8)]` — α is at slot 3, which
/// is the same position as the RGBA α slot. A single `v128_bitselect` per 4-px
/// block copies only the α bytes, leaving R/G/B intact.
///
/// Block: 4 px / iter (one v128 = 16 bytes = 4 px × 4 ch).
///
/// # Safety
///
/// `simd128` must be enabled at compile time. Both slices must be `>= width * 4`.
#[cfg(feature = "yuv-444-packed")]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn copy_alpha_packed_u8x4_at_3(packed: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  // α-slot mask: 0xFF at byte 3 of every 4-byte pixel (slots 3, 7, 11, 15).
  // v128_bitselect(v1, v2, mask): selects v1 bits where mask=1, v2 bits where mask=0.
  // We want α from `packed` (v1) and RGB from `rgba_out` (v2).
  let alpha_mask = i8x16(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let off = x * 4;
      let src = v128_load(packed.as_ptr().add(off).cast());
      let dst = v128_load(rgba_out.as_ptr().add(off).cast());
      // bitselect: where mask=0xFF (α slot), take `src`; where mask=0x00, keep `dst`.
      let merged = v128_bitselect(src, dst, alpha_mask);
      v128_store(rgba_out.as_mut_ptr().add(off).cast(), merged);
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
/// AYUV64 layout: `[A(16), Y(16), U(16), V(16)]` per pixel — α at u16 slot 0.
/// Block: 4 px / iter (2 × v128 loads of u16 packed, covering 4 px × 4 ch × 2 B = 32 B).
/// Uses `u16x8_shr` (LOGICAL right shift) — NOT `i16x8_shr` (arithmetic,
/// which sign-extends and corrupts α ≥ 0x8000).
///
/// # Safety
///
/// `simd128` must be enabled at compile time.
/// `packed.len() >= width * 4`; `rgba_out.len() >= width * 4`.
#[cfg(feature = "yuv-444-packed")]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn copy_alpha_packed_u16x4_to_u8_at_0(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  // α-slot mask for u8 RGBA output: 0xFF at byte 3 of each 4-byte pixel.
  let alpha_mask = i8x16(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let src_off = x * 4;
      // Load 4 AYUV64 pixels = 16 u16 = 2 × v128.
      // lo = [A0, Y0, U0, V0, A1, Y1, U1, V1] (8 u16)
      // hi = [A2, Y2, U2, V2, A3, Y3, U3, V3] (8 u16)
      let lo = v128_load(packed.as_ptr().add(src_off).cast());
      let hi = v128_load(packed.as_ptr().add(src_off + 8).cast());

      // Logical right-shift by 8: u16 α → high byte becomes low byte.
      // u16x8_shr is LOGICAL (zero-filling) — correct for unsigned α.
      // i16x8_shr is ARITHMETIC (sign-extending) — MUST NOT use, corrupts α ≥ 0x8000.
      let lo_shr = u16x8_shr(lo, 8);
      let hi_shr = u16x8_shr(hi, 8);

      // Narrow u16 → u8 (values are in [0, 255] after >> 8).
      // u8x16_narrow_i16x8 packs two i16x8 into one u8x16 with saturation.
      // lo_shr layout after narrow: [A0, Y0, U0, V0, A1, Y1, U1, V1, A2, Y2, U2, V2, A3, ...]
      let packed_u8 = u8x16_narrow_i16x8(lo_shr, hi_shr);

      // Scatter α (byte 0 of each 4-byte pixel group) to output slot 3.
      // After narrow, the layout is: px0=[A0,Y0,U0,V0], px1=[A1,Y1,U1,V1], ...
      // We want bytes at positions 0,4,8,12 placed at positions 3,7,11,15.
      // u8x16_swizzle: index ≥ 16 zeroes the lane (like _mm_shuffle_epi8).
      let shuf_mask = i8x16(
        -1, -1, -1, 0, // px0: src[0] → byte 3; bytes 0..2 = 0
        -1, -1, -1, 4, // px1: src[4] → byte 7; bytes 4..6 = 0
        -1, -1, -1, 8, // px2: src[8] → byte 11; bytes 8..10 = 0
        -1, -1, -1, 12, // px3: src[12] → byte 15; bytes 12..14 = 0
      );
      let a_scattered = u8x16_swizzle(packed_u8, shuf_mask);

      // Load existing rgba_out 4 px and blend.
      let dst_off = x * 4;
      let dst = v128_load(rgba_out.as_ptr().add(dst_off).cast());
      let merged = v128_bitselect(a_scattered, dst, alpha_mask);
      v128_store(rgba_out.as_mut_ptr().add(dst_off).cast(), merged);
      x += 4;
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

// ---------------------------------------------------------------------------
// Helper 3: AYUV64 u16 → u16 RGBA  (α at packed slot 0, no depth conv)
// ---------------------------------------------------------------------------

/// AYUV64 → u16 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// Block: 4 px / iter (two v128 of u16 RGBA = 32 bytes per iteration).
///
/// # Safety
///
/// `simd128` must be enabled at compile time.
/// Both slices must be `>= width * 4` elements.
#[cfg(feature = "yuv-444-packed")]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn copy_alpha_packed_u16x4_at_0(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  // u16 α-slot mask: 0xFFFF at u16 slot 3 and 7 within each v128
  // (= 2 pixels per v128). In bytes: 0xFF at bytes 6,7 and 14,15.
  let alpha_mask = i8x16(0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1);

  // Shuffle: extract α from u16 slot 0 of each AYUV64 pixel, place at u16 slot 3.
  // Each v128 holds 2 pixels (u16 slots 0-7: px0=slots 0-3, px1=slots 4-7).
  // α (slot 0) = bytes [0,1] → slot 3 = bytes [6,7].
  // α (slot 4) = bytes [8,9] → slot 7 = bytes [14,15].
  // u8x16_swizzle indices ≥ 16 zero the lane.
  let shuf_lo = i8x16(
    -1, -1, -1, -1, -1, -1, 0, 1, // px0: bytes 0,1 → bytes 6,7; other bytes zero
    -1, -1, -1, -1, -1, -1, 8, 9, // px1: bytes 8,9 → bytes 14,15; other bytes zero
  );
  let shuf_hi = i8x16(
    -1, -1, -1, -1, -1, -1, 0, 1, // px2 (in hi block): bytes 0,1 → bytes 6,7
    -1, -1, -1, -1, -1, -1, 8, 9, // px3 (in hi block): bytes 8,9 → bytes 14,15
  );

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      let off = x * 4;
      // Two v128 cover 4 pixels of u16 RGBA output (and 4 pixels of packed).
      let src_lo = v128_load(packed.as_ptr().add(off).cast()); // px0, px1 of packed
      let src_hi = v128_load(packed.as_ptr().add(off + 8).cast()); // px2, px3 of packed
      let dst_lo = v128_load(rgba_out.as_ptr().add(off).cast());
      let dst_hi = v128_load(rgba_out.as_ptr().add(off + 8).cast());

      // Extract α (slot 0) from packed and place at slot 3 via byte-level swizzle.
      let a_lo = u8x16_swizzle(src_lo, shuf_lo);
      let a_hi = u8x16_swizzle(src_hi, shuf_hi);

      // Blend: where alpha_mask = 0xFF (α u16 bytes), take a_lo/a_hi; else keep dst.
      let merged_lo = v128_bitselect(a_lo, dst_lo, alpha_mask);
      let merged_hi = v128_bitselect(a_hi, dst_hi, alpha_mask);

      v128_store(rgba_out.as_mut_ptr().add(off).cast(), merged_lo);
      v128_store(rgba_out.as_mut_ptr().add(off + 8).cast(), merged_hi);
      x += 4;
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

// ---------------------------------------------------------------------------
// Helper 4: α plane u8 → u8 RGBA
// ---------------------------------------------------------------------------

/// Yuva420p / 422p / 444p u8 → u8 RGBA: scatter α plane into
/// `rgba_out[3 + 4*n]`.
///
/// Block: 4 px / iter. Loads 4 contiguous α bytes, scatters them to slot 3
/// of each 4-byte RGBA pixel via `u8x16_swizzle`, then `v128_bitselect`.
///
/// # Safety
///
/// `simd128` must be enabled at compile time.
/// `alpha.len() >= width`; `rgba_out.len() >= width * 4`.
#[cfg(any(feature = "gbr", feature = "yuva"))]
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  // α-slot mask for u8 RGBA: 0xFF at byte 3 of each 4-byte pixel.
  let alpha_mask = i8x16(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);

  // Scatter: 4 contiguous α bytes → byte 3 of each 4-byte pixel group.
  // Input (in low 4 bytes of v128): [a0, a1, a2, a3, ?, ?, ?, ?, ...]
  // Output: [0, 0, 0, a0, 0, 0, 0, a1, 0, 0, 0, a2, 0, 0, 0, a3]
  // u8x16_swizzle: -1 (= 0xFF ≥ 16) zeroes the lane.
  let shuf_mask = i8x16(-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3);

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 α bytes via a 4-byte aligned read packed into low 32 bits.
      // We read exactly 4 bytes by loading them into a u32 and broadcasting.
      // wasm has no _mm_cvtsi32_si128 equivalent, but i32x4_splat(scalar)
      // works if we can get a scalar — or we use v128_load32_zero which
      // loads a u32 and zeroes the upper 12 bytes.
      let a_raw = v128_load32_zero(alpha.as_ptr().add(x).cast());
      let a_scattered = u8x16_swizzle(a_raw, shuf_mask);

      let off = x * 4;
      let dst = v128_load(rgba_out.as_ptr().add(off).cast());
      let merged = v128_bitselect(a_scattered, dst, alpha_mask);
      v128_store(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 4;
    }

    if x < width {
      scalar::copy_alpha_plane_u8(&alpha[x..width], &mut rgba_out[x * 4..width * 4], width - x);
    }
  }
}

// ---------------------------------------------------------------------------
// Helper 5: α plane u16 → u8 RGBA  (depth-conv >> (BITS - 8))
// ---------------------------------------------------------------------------

/// Yuva*p9/10/12/14 → u8 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> (BITS - 8)`.
///
/// Uses `u16x8_shr(v, BITS - 8)` with a runtime `u32` count — wasm's
/// `u16x8_shr` is LOGICAL (zero-filling), so this works correctly for all
/// `BITS ∈ [8, 16]` without per-BITS monomorphization.
/// Block: 4 px / iter.
///
/// # Safety
///
/// `simd128` must be enabled at compile time.
/// `alpha.len() >= width`; `rgba_out.len() >= width * 4`.
#[cfg(any(feature = "gbr", feature = "yuva"))]
#[inline]
#[target_feature(enable = "simd128")]
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

  // α-slot mask for u8 RGBA output: 0xFF at byte 3 of each 4-byte pixel.
  let alpha_mask = i8x16(0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1);

  // Scatter 4 u8 α values (in low bytes of each u16 after >> (BITS-8) and
  // narrow) into slot 3 of each 4-byte pixel.
  // After u8x16_narrow_i16x8(4_u16s_vec, zero_vec), low 8 bytes = [a0,a1,a2,a3,0,0,0,0].
  // Scatter: a0→byte3, a1→byte7, a2→byte11, a3→byte15.
  let shuf_mask = i8x16(-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3);

  // Compile-time shift count as u32 for u16x8_shr (runtime variable accepted).
  let shr_count: u32 = BITS - 8;

  // BITS-bit canonicalization mask: AND'd before shift so over-range
  // source α samples don't leak through (matches scalar parity).
  let bits_mask = u16x8_splat(((1u32 << BITS) - 1) as u16);

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 u16 α values (8 bytes) via v128_load64_zero (loads u64, zeroes upper).
      let a_u16_raw = v128_load64_zero(alpha.as_ptr().add(x).cast());
      // Mask to low BITS before shift (over-range α canonicalization).
      let a_u16 = v128_and(a_u16_raw, bits_mask);
      // Logical right-shift — u16x8_shr, NOT i16x8_shr (arithmetic would corrupt
      // α values ≥ 0x8000 by sign-extending).
      let a_shifted = u16x8_shr(a_u16, shr_count);
      // Narrow to u8 (values are in [0, 255] after shift).
      let zero = i16x8_splat(0);
      let a_u8_vec = u8x16_narrow_i16x8(a_shifted, zero);
      // Scatter α bytes to slot 3 of each pixel.
      let a_scattered = u8x16_swizzle(a_u8_vec, shuf_mask);

      let off = x * 4;
      let dst = v128_load(rgba_out.as_ptr().add(off).cast());
      let merged = v128_bitselect(a_scattered, dst, alpha_mask);
      v128_store(rgba_out.as_mut_ptr().add(off).cast(), merged);
      x += 4;
    }

    if x < width {
      // Scalar tail uses `BE = false`: this wasm-simd128 helper does
      // host-native u16 loads (`v128_load64_zero`), which match LE-on-disk
      // only on LE hosts. The dispatcher routes BE = true directly to scalar
      // (see `dispatch::alpha_extract`), so the SIMD path here is BE = false
      // by construction.
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
/// `rgba_out[3 + 4*n]` (u16). No depth conversion. `BITS` is informational.
///
/// Block: 4 px / iter (two v128 of u16 RGBA = 32 bytes). α u16 values are
/// shuffled from a flat plane into u16 slot 3 of each pixel tuple.
///
/// # Safety
///
/// `simd128` must be enabled at compile time.
/// `alpha.len() >= width`; `rgba_out.len() >= width * 4` elements.
#[cfg(any(feature = "gbr", feature = "yuva"))]
#[inline]
#[target_feature(enable = "simd128")]
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

  // u16 α-slot mask: 0xFFFF at u16 slot 3 and 7 within each v128 (2 pixels).
  // Bytes 6,7 and 14,15 are 0xFF; all others 0x00.
  let alpha_mask = i8x16(0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1);

  // Scatter α u16 into slot 3 of each 4-u16 pixel within a v128.
  // Each v128 holds 2 pixels; a_raw (low 8 bytes) = [a0, a1, a2, a3] u16.
  // lo block (px0, px1): a0(bytes 0,1) → slot3(bytes 6,7); a1(bytes 2,3) → slot7(bytes 14,15).
  // hi block (px2, px3): a2(bytes 4,5) → slot3(bytes 6,7); a3(bytes 6,7) → slot7(bytes 14,15).
  let shuf_lo = i8x16(
    -1, -1, -1, -1, -1, -1, 0, 1, // px0: bytes 0,1 → bytes 6,7
    -1, -1, -1, -1, -1, -1, 2, 3, // px1: bytes 2,3 → bytes 14,15
  );
  let shuf_hi = i8x16(
    -1, -1, -1, -1, -1, -1, 4, 5, // px2: bytes 4,5 → bytes 6,7
    -1, -1, -1, -1, -1, -1, 6, 7, // px3: bytes 6,7 → bytes 14,15
  );

  // BITS-bit canonicalization mask: AND'd before scatter so over-range
  // source α samples don't leak through (matches scalar parity).
  let bits_mask = u16x8_splat(((1u32 << BITS) - 1) as u16);

  unsafe {
    let mut x = 0usize;
    while x + 4 <= width {
      // Load 4 α u16 = 8 bytes into low 64 bits; upper 64 bits zero,
      // then AND with bits_mask (over-range α canonicalization).
      let a_raw = v128_and(v128_load64_zero(alpha.as_ptr().add(x).cast()), bits_mask);

      // Scatter into the two v128 blocks (lo covers px0,px1; hi covers px2,px3).
      let a_lo = u8x16_swizzle(a_raw, shuf_lo);
      let a_hi = u8x16_swizzle(a_raw, shuf_hi);

      let off = x * 4;
      let dst_lo = v128_load(rgba_out.as_ptr().add(off).cast());
      let dst_hi = v128_load(rgba_out.as_ptr().add(off + 8).cast());

      let merged_lo = v128_bitselect(a_lo, dst_lo, alpha_mask);
      let merged_hi = v128_bitselect(a_hi, dst_hi, alpha_mask);

      v128_store(rgba_out.as_mut_ptr().add(off).cast(), merged_lo);
      v128_store(rgba_out.as_mut_ptr().add(off + 8).cast(), merged_hi);
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
  // Includes widths from 1 (pure scalar) through 130 (many full blocks + tail).
  const WIDTHS: &[usize] = &[
    1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 128, 130,
  ];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn wasm_simd128_copy_alpha_packed_u8x4_at_3_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut packed, 0xC0FFEE);
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0xDECAF);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe {
        super::copy_alpha_packed_u8x4_at_3(&packed, &mut rgba_simd, w);
      }
      scalar::copy_alpha_packed_u8x4_at_3(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn wasm_simd128_copy_alpha_packed_u16x4_to_u8_at_0_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut packed, 0xCAB00D);
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0xFEED);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe {
        super::copy_alpha_packed_u16x4_to_u8_at_0(&packed, &mut rgba_simd, w);
      }
      scalar::copy_alpha_packed_u16x4_to_u8_at_0::<false>(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn wasm_simd128_copy_alpha_packed_u16x4_at_0_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut packed, 0xBEEF11);
      let mut rgba_simd = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut rgba_simd, 0x1337);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe {
        super::copy_alpha_packed_u16x4_at_0(&packed, &mut rgba_simd, w);
      }
      scalar::copy_alpha_packed_u16x4_at_0::<false>(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn wasm_simd128_copy_alpha_plane_u8_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut alpha = std::vec![0u8; w];
      pseudo_random_u8(&mut alpha, 0xABCDEF);
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0x123456);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe {
        super::copy_alpha_plane_u8(&alpha, &mut rgba_simd, w);
      }
      scalar::copy_alpha_plane_u8(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn wasm_simd128_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits10() {
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xC0DE);
      for v in alpha.iter_mut() {
        *v &= 0x03FF;
      }
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0xBABE);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe {
        super::copy_alpha_plane_u16_to_u8::<10>(&alpha, &mut rgba_simd, w);
      }
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
  fn wasm_simd128_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits12() {
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xF00BAA);
      for v in alpha.iter_mut() {
        *v &= 0x0FFF;
      }
      let mut rgba_simd = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0x5EED);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe {
        super::copy_alpha_plane_u16_to_u8::<12>(&alpha, &mut rgba_simd, w);
      }
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
  fn wasm_simd128_copy_alpha_plane_u16_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xDEADBE);
      let mut rgba_simd = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut rgba_simd, 0xFADE);
      let mut rgba_scalar = rgba_simd.clone();
      unsafe {
        super::copy_alpha_plane_u16::<10>(&alpha, &mut rgba_simd, w);
      }
      // SIMD reads native u16; pair with scalar BE = false (LE-on-LE-host).
      scalar::copy_alpha_plane_u16::<10, false>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }
}
