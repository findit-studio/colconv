//! wasm-simd128 kernels for 16-bit packed RGB/BGR/RGBA/BGRA sources (Tier 8 finish).
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
//! ## Per-format SIMD strategy (8 pixels per SIMD iteration via v128)
//!
//! ### Rgb48 / Bgr48 (stride-3)
//!
//! 8 pixels = 24 u16 = 48 bytes = three `v128_load` calls.
//! Channel deinterleave via `u8x16_swizzle` + `i8x16_shuffle` (same mask shapes as the
//! SSE4.1 sibling). For Bgr48, the extracted `(ch0=B, ch1=G, ch2=R)` are output-swapped.
//!
//! ### Rgba64 / Bgra64 (stride-4)
//!
//! 8 pixels = 32 u16 = 64 bytes = four `v128_load` calls.
//! 3-level `i16x8_shuffle` cascade deinterleaves 4 channels, mirroring the AVX2 sibling
//! (`xv36.rs` pattern). Produces four `v128` channel vectors each holding 8 u16 samples.
//!
//! ## Depth conversion
//!
//! - **u16 → u8:** `u16x8_shr(v, 8)` + `u8x16_narrow_i16x8` (saturating narrow; values ≤ 255
//!   so no saturation occurs in practice).
//! - **u16 → u16:** write 8-pixel chunks via `write_rgb_u16_8` / `write_rgba_u16_8` helpers.
//!
//! ## Scalar tail
//!
//! All kernels handle `width % 8` remaining pixels via the scalar reference.
// Kernels are wired into the dispatcher in the dispatch-wiring step; suppress
// dead_code until then.
#![allow(dead_code)]

use core::arch::wasm32::*;

use super::*;

// =============================================================================
// Stride-3 deinterleave helper (8 pixels, 3 × v128 loads)
// =============================================================================

/// Deinterleave 8 pixels of stride-3 u16 from three `v128` loads into
/// `(ch0, ch1, ch2)` channel vectors, each holding 8 u16 values in natural order.
///
/// For Rgb48: `ch0=R`, `ch1=G`, `ch2=B`.
/// For Bgr48: `ch0=B`, `ch1=G`, `ch2=R`; caller swaps on output.
///
/// Input layout (u16 elements):
///   v0 = [C0_0, C1_0, C2_0, C0_1, C1_1, C2_1, C0_2, C1_2]  (pixels 0-2, partial)
///   v1 = [C2_2, C0_3, C1_3, C2_3, C0_4, C1_4, C2_4, C0_5]  (pixels 2-5, partial)
///   v2 = [C1_5, C2_5, C0_6, C1_6, C2_6, C0_7, C1_7, C2_7]  (pixels 5-7)
///
/// # Safety
///
/// Caller must hold `simd128` target_feature.
#[inline(always)]
unsafe fn deinterleave_rgb48_8px(v0: v128, v1: v128, v2: v128) -> (v128, v128, v128) {
  // ch0: C0 lanes at byte offsets (in the concatenated 3-register window):
  //   pixel 0: bytes 0-1 from v0
  //   pixel 1: bytes 6-7 from v0
  //   pixel 2: bytes 18-19 from v1 (offset 2 in v1 = byte 18 relative)
  //   pixel 3: bytes 8-9 from v1
  //   pixel 4: bytes 24-25 from v1 (offset 8... no, v1 lanes 4-5 = bytes 8-11 = C0_4,C1_4)
  //     Actually C0_4 is at v1 byte offsets 8-9.
  //   pixel 5: bytes 14-15 from v1 (C0_5)
  //   pixel 6: bytes 4-5 from v2
  //   pixel 7: bytes 10-11 from v2
  //
  // Build ch0 from v0, v1, v2 contributions:
  //   from v0: bytes [0,1] → out bytes [0,1]; bytes [6,7] → out bytes [2,3]
  //   from v1: bytes [2,3] → out bytes [4,5]; bytes [8,9] → out bytes [6,7]
  //            bytes [14,15] → out bytes [8,9]
  //   from v2: bytes [4,5] → out bytes [10,11]; bytes [10,11] → out bytes [12,13]
  //            bytes [--] → ...
  //
  // Actually let's derive systematically. 24 u16 laid flat as bytes:
  //   v0 bytes 0..15:  [R0lo,R0hi, G0lo,G0hi, B0lo,B0hi, R1lo,R1hi, G1lo,G1hi, B1lo,B1hi, R2lo,R2hi, G2lo,G2hi]
  //   v1 bytes 0..15:  [B2lo,B2hi, R3lo,R3hi, G3lo,G3hi, B3lo,B3hi, R4lo,R4hi, G4lo,G4hi, B4lo,B4hi, R5lo,R5hi]
  //   v2 bytes 0..15:  [G5lo,G5hi, B5lo,B5hi, R6lo,R6hi, G6lo,G6hi, B6lo,B6hi, R7lo,R7hi, G7lo,G7hi, B7lo,B7hi]
  //
  // ch0 (R for Rgb48) = [R0,R1,R2,R3,R4,R5,R6,R7]:
  //   R0: v0[0..2]   R1: v0[6..8]   R2: v0[12..14]
  //   R3: v1[2..4]   R4: v1[8..10]  R5: v1[14..16] (= v1[14,15])
  //   R6: v2[4..6]   R7: v2[10..12]
  //
  // ch1 (G) = [G0,G1,G2,G3,G4,G5,G6,G7]:
  //   G0: v0[2..4]   G1: v0[8..10]  G2: v0[14..16]
  //   G3: v1[4..6]   G4: v1[10..12] G5: v2[0..2]
  //   G6: v2[6..8]   G7: v2[12..14]
  //
  // ch2 (B) = [B0,B1,B2,B3,B4,B5,B6,B7]:
  //   B0: v0[4..6]   B1: v0[10..12] B2: v1[0..2]
  //   B3: v1[6..8]   B4: v1[12..14] B5: v2[2..4]
  //   B6: v2[8..10]  B7: v2[14..16]

  // ch0: R0=v0[0,1], R1=v0[6,7], R2=v0[12,13], R3=v1[2,3], R4=v1[8,9], R5=v1[14,15],
  //      R6=v2[4,5], R7=v2[10,11]
  let ch0_v0 = i8x16(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  let ch0_v1 = i8x16(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1);
  let ch0_v2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11);
  let ch0 = v128_or(
    v128_or(u8x16_swizzle(v0, ch0_v0), u8x16_swizzle(v1, ch0_v1)),
    u8x16_swizzle(v2, ch0_v2),
  );

  // ch1: G0=v0[2,3], G1=v0[8,9], G2=v0[14,15], G3=v1[4,5], G4=v1[10,11],
  //      G5=v2[0,1], G6=v2[6,7], G7=v2[12,13]
  let ch1_v0 = i8x16(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  let ch1_v1 = i8x16(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1);
  let ch1_v2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13);
  let ch1 = v128_or(
    v128_or(u8x16_swizzle(v0, ch1_v0), u8x16_swizzle(v1, ch1_v1)),
    u8x16_swizzle(v2, ch1_v2),
  );

  // ch2: B0=v0[4,5], B1=v0[10,11], B2=v1[0,1], B3=v1[6,7], B4=v1[12,13],
  //      B5=v2[2,3], B6=v2[8,9], B7=v2[14,15]
  let ch2_v0 = i8x16(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  let ch2_v1 = i8x16(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1);
  let ch2_v2 = i8x16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15);
  let ch2 = v128_or(
    v128_or(u8x16_swizzle(v0, ch2_v0), u8x16_swizzle(v1, ch2_v1)),
    u8x16_swizzle(v2, ch2_v2),
  );

  (ch0, ch1, ch2)
}

// =============================================================================
// Stride-4 deinterleave helper (8 pixels, 4 × v128 loads)
// =============================================================================

/// Deinterleave 8 pixels of stride-4 u16 from four `v128` loads into
/// `(ch0, ch1, ch2, ch3)` channel vectors, each holding 8 u16 values in natural order.
///
/// For Rgba64: `(R, G, B, A)`. For Bgra64: `(B, G, R, A)`.
///
/// Uses a 2-level `i16x8_shuffle` cascade. Each level-1 shuffle pairs two
/// raw registers and groups two adjacent channels for 4 pixels; each
/// level-2 shuffle splits the channel pair into the final per-channel
/// 8-u16 vector. Mirrors the SSE4.1 sibling's per-channel gather
/// (`src/row/arch/x86_sse41/packed_rgb_16bit.rs`) but adapted to the
/// 8-lane `i16x8_shuffle` primitive.
///
/// Input layout:
///   raw0 = [C0_0, C1_0, C2_0, C3_0, C0_1, C1_1, C2_1, C3_1]  (pixels 0-1)
///   raw1 = [C0_2, C1_2, C2_2, C3_2, C0_3, C1_3, C2_3, C3_3]  (pixels 2-3)
///   raw2 = [C0_4, C1_4, C2_4, C3_4, C0_5, C1_5, C2_5, C3_5]  (pixels 4-5)
///   raw3 = [C0_6, C1_6, C2_6, C3_6, C0_7, C1_7, C2_7, C3_7]  (pixels 6-7)
///
/// # Safety
///
/// Caller must hold `simd128` target_feature.
#[inline(always)]
unsafe fn deinterleave_rgba64_8px(
  raw0: v128,
  raw1: v128,
  raw2: v128,
  raw3: v128,
) -> (v128, v128, v128, v128) {
  // Level 1: combine each adjacent raw pair (raw0+raw1 = pixels 0-3,
  // raw2+raw3 = pixels 4-7) and group the 4 channels into pairs.
  //
  // For (raw0, raw1) the 4 channels of pixels 0-3 sit at u16 lane
  // positions:
  //   C0_0=raw0[0], C0_1=raw0[4], C0_2=raw1[0]=8, C0_3=raw1[4]=12,
  //   C1_0=raw0[1], C1_1=raw0[5], C1_2=raw1[1]=9, C1_3=raw1[5]=13,
  //   C2_0=raw0[2], C2_1=raw0[6], C2_2=raw1[2]=10, C2_3=raw1[6]=14,
  //   C3_0=raw0[3], C3_1=raw0[7], C3_2=raw1[3]=11, C3_3=raw1[7]=15.
  //
  // Each i16x8_shuffle output holds 8 u16 = exactly two channels worth
  // of data for the 4-pixel group; pair channels (C0,C1) and (C2,C3).
  //   pair_01_c01: [C0_0, C0_1, C0_2, C0_3, C1_0, C1_1, C1_2, C1_3]
  //   pair_01_c23: [C2_0, C2_1, C2_2, C2_3, C3_0, C3_1, C3_2, C3_3]
  //   pair_23_c01: [C0_4, C0_5, C0_6, C0_7, C1_4, C1_5, C1_6, C1_7]
  //   pair_23_c23: [C2_4, C2_5, C2_6, C2_7, C3_4, C3_5, C3_6, C3_7]
  let pair_01_c01 = i16x8_shuffle::<0, 4, 8, 12, 1, 5, 9, 13>(raw0, raw1);
  let pair_01_c23 = i16x8_shuffle::<2, 6, 10, 14, 3, 7, 11, 15>(raw0, raw1);
  let pair_23_c01 = i16x8_shuffle::<0, 4, 8, 12, 1, 5, 9, 13>(raw2, raw3);
  let pair_23_c23 = i16x8_shuffle::<2, 6, 10, 14, 3, 7, 11, 15>(raw2, raw3);

  // Level 2: concatenate the lo half of each pair_*_c01 (= channel 0 for
  // 4 pixels) with the lo half of the matching counterpart (4 more
  // pixels) to get a single 8-u16 channel vector in natural pixel order.
  // Lanes 0..3 from first source = pixels 0..3; lanes 8..11 from second
  // source = pixels 4..7 (treated as the second i16x8_shuffle source).
  // Hi halves of each pair_* hold the next channel (C1 or C3).
  let ch0 = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(pair_01_c01, pair_23_c01);
  let ch1 = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(pair_01_c01, pair_23_c01);
  let ch2 = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(pair_01_c23, pair_23_c23);
  let ch3 = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(pair_01_c23, pair_23_c23);

  (ch0, ch1, ch2, ch3)
}

// =============================================================================
// u16 → u8 narrowing helper
// =============================================================================

/// Narrow a u16×8 vector to u8×8 (in the low half) via logical right-shift by 8.
///
/// Equivalent to scalar `(v >> 8) as u8`. Packs the zero high-half via
/// `u8x16_narrow_i16x8` (saturating; values ≤ 255 so no saturation occurs).
///
/// # Safety
///
/// Caller must hold `simd128` target_feature.
#[inline(always)]
unsafe fn narrow_u16x8_to_u8x8(v: v128) -> v128 {
  let shr = u16x8_shr(v, 8);
  let zero = u16x8_splat(0);
  u8x16_narrow_i16x8(shr, zero)
}

// ---- endian byte-swap helper -------------------------------------------------

/// Compile-time host endianness. `true` on BE targets, `false` on LE.
///
/// Used by [`byteswap_if_be`] to gate the swap on `BE != HOST_NATIVE_BE`,
/// covering all four `wire × host` quadrants. Mirrors the gate established
/// in the canonical NEON `bswap_u16x8_if_be` helper.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Conditionally byte-swap every u16 lane in `v` so the returned value is in
/// **host-native** byte order regardless of the host endianness.
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
/// Uses `u8x16_swizzle` with a compile-time mask. The unused branch folds
/// at compile time since both `BE` and `HOST_NATIVE_BE` are constants.
#[inline(always)]
unsafe fn byteswap_if_be<const BE: bool>(v: v128) -> v128 {
  if BE != HOST_NATIVE_BE {
    // Swap bytes within each u16 lane: [1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 13,12, 15,14]
    u8x16_swizzle(
      v,
      i8x16(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14),
    )
  } else {
    v
  }
}

// =============================================================================
// Rgb48 (R, G, B — 3 u16 elements per pixel)
// =============================================================================

/// wasm-simd128 Rgb48 → packed u8 RGB. 8 pixels per iteration.
///
/// Three `v128_load` calls deinterleave into `(R, G, B)` u16×8 via
/// `deinterleave_rgb48_8px`, then `>> 8` + `u8x16_narrow_i16x8` narrows each
/// channel to u8×8. `write_rgb_16` interleaves back to packed RGB.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgb48_to_rgb_row<const BE: bool>(
  rgb48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let (r, g, b) = deinterleave_rgb48_8px(v0, v1, v2);
      let r_u8 = narrow_u16x8_to_u8x8(r);
      let g_u8 = narrow_u16x8_to_u8x8(g);
      let b_u8 = narrow_u16x8_to_u8x8(b);
      // write_rgb_16 writes 16 pixels; we only have 8, so write to a temp buffer
      // of 48 bytes and copy 24 bytes.
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgb_row::<BE>(&rgb48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm-simd128 Rgb48 → packed u8 RGBA. 8 pixels per iteration. Alpha forced to 0xFF.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgb48_to_rgba_row<const BE: bool>(
  rgb48: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque_u8 = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let (r, g, b) = deinterleave_rgb48_8px(v0, v1, v2);
      let r_u8 = narrow_u16x8_to_u8x8(r);
      let g_u8 = narrow_u16x8_to_u8x8(g);
      let b_u8 = narrow_u16x8_to_u8x8(b);
      // write_rgba_16 writes 16 pixels (64 bytes); copy 32 bytes for 8 pixels.
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgba_row::<BE>(&rgb48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// wasm-simd128 Rgb48 → native-depth u16 RGB (identity copy). 8 pixels per iteration.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgb48_to_rgb_u16_row<const BE: bool>(
  rgb48: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let (r, g, b) = deinterleave_rgb48_8px(v0, v1, v2);
      write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgb_u16_row::<BE>(&rgb48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm-simd128 Rgb48 → native-depth u16 RGBA. 8 pixels per iteration. Alpha forced to 0xFFFF.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgb48_to_rgba_u16_row<const BE: bool>(
  rgb48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = u16x8_splat(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let (r, g, b) = deinterleave_rgb48_8px(v0, v1, v2);
      write_rgba_u16_8(r, g, b, opaque, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgba_u16_row::<BE>(&rgb48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Bgr48 (B, G, R — 3 u16 elements per pixel)
// =============================================================================

/// wasm-simd128 Bgr48 → packed u8 RGB. 8 pixels per iteration.
/// B↔R swap via passing `(ch2=R, ch1=G, ch0=B)` to write helpers.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_bgr48_to_rgb_row<const BE: bool>(
  bgr48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      // ch0=B, ch1=G, ch2=R
      let (b, g, r) = deinterleave_rgb48_8px(v0, v1, v2);
      let r_u8 = narrow_u16x8_to_u8x8(r);
      let g_u8 = narrow_u16x8_to_u8x8(g);
      let b_u8 = narrow_u16x8_to_u8x8(b);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgb_row::<BE>(&bgr48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm-simd128 Bgr48 → packed u8 RGBA. 8 pixels per iteration.
/// B↔R swap; alpha forced to 0xFF.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_bgr48_to_rgba_row<const BE: bool>(
  bgr48: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque_u8 = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let (b, g, r) = deinterleave_rgb48_8px(v0, v1, v2);
      let r_u8 = narrow_u16x8_to_u8x8(r);
      let g_u8 = narrow_u16x8_to_u8x8(g);
      let b_u8 = narrow_u16x8_to_u8x8(b);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, opaque_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgba_row::<BE>(&bgr48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// wasm-simd128 Bgr48 → native-depth u16 RGB. 8 pixels per iteration.
/// B↔R swap; values unchanged.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_bgr48_to_rgb_u16_row<const BE: bool>(
  bgr48: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let (b, g, r) = deinterleave_rgb48_8px(v0, v1, v2);
      // Output R, G, B order
      write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgb_u16_row::<BE>(&bgr48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm-simd128 Bgr48 → native-depth u16 RGBA. 8 pixels per iteration.
/// B↔R swap; alpha forced to 0xFFFF.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_bgr48_to_rgba_u16_row<const BE: bool>(
  bgr48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = u16x8_splat(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let (b, g, r) = deinterleave_rgb48_8px(v0, v1, v2);
      // Output R, G, B, A order
      write_rgba_u16_8(r, g, b, opaque, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgba_u16_row::<BE>(&bgr48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Rgba64 (R, G, B, A — 4 u16 elements per pixel)
// =============================================================================

/// wasm-simd128 Rgba64 → packed u8 RGB. 8 pixels per SIMD iteration.
/// Alpha discarded.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgba64_to_rgb_row<const BE: bool>(
  rgba64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let raw1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let raw2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let raw3 = byteswap_if_be::<BE>(v128_load(ptr.add(24).cast()));
      let (r, g, b, _a) = deinterleave_rgba64_8px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x8_to_u8x8(r);
      let g_u8 = narrow_u16x8_to_u8x8(g);
      let b_u8 = narrow_u16x8_to_u8x8(b);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgb_row::<BE>(&rgba64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm-simd128 Rgba64 → packed u8 RGBA. 8 pixels per SIMD iteration.
/// Source alpha passes through (narrowed via `>> 8`).
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgba64_to_rgba_row<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let raw1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let raw2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let raw3 = byteswap_if_be::<BE>(v128_load(ptr.add(24).cast()));
      let (r, g, b, a) = deinterleave_rgba64_8px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x8_to_u8x8(r);
      let g_u8 = narrow_u16x8_to_u8x8(g);
      let b_u8 = narrow_u16x8_to_u8x8(b);
      let a_u8 = narrow_u16x8_to_u8x8(a);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, a_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgba_row::<BE>(&rgba64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// wasm-simd128 Rgba64 → native-depth u16 RGB. 8 pixels per SIMD iteration.
/// Alpha discarded.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgba64_to_rgb_u16_row<const BE: bool>(
  rgba64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let raw1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let raw2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let raw3 = byteswap_if_be::<BE>(v128_load(ptr.add(24).cast()));
      let (r, g, b, _a) = deinterleave_rgba64_8px(raw0, raw1, raw2, raw3);
      write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgb_u16_row::<BE>(&rgba64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm-simd128 Rgba64 → native-depth u16 RGBA (identity copy). 8 pixels per SIMD iteration.
/// Source alpha preserved.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_rgba64_to_rgba_u16_row<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let raw0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let raw1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let raw2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let raw3 = byteswap_if_be::<BE>(v128_load(ptr.add(24).cast()));
      let (r, g, b, a) = deinterleave_rgba64_8px(raw0, raw1, raw2, raw3);
      write_rgba_u16_8(r, g, b, a, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgba_u16_row::<BE>(&rgba64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Bgra64 (B, G, R, A — 4 u16 elements per pixel)
// =============================================================================

/// wasm-simd128 Bgra64 → packed u8 RGB. 8 pixels per SIMD iteration.
/// B↔R swap; alpha discarded.
///
/// `deinterleave_rgba64_8px` yields `(B, G, R, A)` in source memory order.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_bgra64_to_rgb_row<const BE: bool>(
  bgra64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let raw1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let raw2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let raw3 = byteswap_if_be::<BE>(v128_load(ptr.add(24).cast()));
      // ch0=B, ch1=G, ch2=R, ch3=A
      let (b, g, r, _a) = deinterleave_rgba64_8px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x8_to_u8x8(r);
      let g_u8 = narrow_u16x8_to_u8x8(g);
      let b_u8 = narrow_u16x8_to_u8x8(b);
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgb_row::<BE>(&bgra64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm-simd128 Bgra64 → packed u8 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap; source alpha passes through (narrowed via `>> 8`).
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_bgra64_to_rgba_row<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let raw1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let raw2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let raw3 = byteswap_if_be::<BE>(v128_load(ptr.add(24).cast()));
      let (b, g, r, a) = deinterleave_rgba64_8px(raw0, raw1, raw2, raw3);
      let r_u8 = narrow_u16x8_to_u8x8(r);
      let g_u8 = narrow_u16x8_to_u8x8(g);
      let b_u8 = narrow_u16x8_to_u8x8(b);
      let a_u8 = narrow_u16x8_to_u8x8(a);
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, a_u8, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgba_row::<BE>(&bgra64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// wasm-simd128 Bgra64 → native-depth u16 RGB. 8 pixels per SIMD iteration.
/// B↔R swap; alpha discarded.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_bgra64_to_rgb_u16_row<const BE: bool>(
  bgra64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let raw1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let raw2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let raw3 = byteswap_if_be::<BE>(v128_load(ptr.add(24).cast()));
      // Swap B↔R: output (R=ch2, G=ch1, B=ch0)
      let (b, g, r, _a) = deinterleave_rgba64_8px(raw0, raw1, raw2, raw3);
      write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgb_u16_row::<BE>(&bgra64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// wasm-simd128 Bgra64 → native-depth u16 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap; source alpha preserved at position 3.
///
/// # Safety
///
/// 1. **`simd128` must be enabled at compile time.**
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_bgra64_to_rgba_u16_row<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let raw0 = byteswap_if_be::<BE>(v128_load(ptr.cast()));
      let raw1 = byteswap_if_be::<BE>(v128_load(ptr.add(8).cast()));
      let raw2 = byteswap_if_be::<BE>(v128_load(ptr.add(16).cast()));
      let raw3 = byteswap_if_be::<BE>(v128_load(ptr.add(24).cast()));
      // Swap B↔R: output (R=ch2, G=ch1, B=ch0, A=ch3)
      let (b, g, r, a) = deinterleave_rgba64_8px(raw0, raw1, raw2, raw3);
      write_rgba_u16_8(r, g, b, a, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgba_u16_row::<BE>(&bgra64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}
