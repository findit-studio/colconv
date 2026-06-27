//! SSE4.1 kernels for 16-bit packed RGB/BGR/RGBA/BGRA sources (Tier 8 finish).
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
//! ## Per-format SIMD strategy (8 pixels per SIMD iteration)
//!
//! ### Rgb48 / Bgr48 (stride-3)
//!
//! 8 pixels = 24 u16 elements = 48 bytes = 3 × 128-bit loads.
//!
//! ```text
//! v0 = [R0, G0, B0, R1, G1, B1, R2, G2]   (u16 positions 0–7)
//! v1 = [B2, R3, G3, B3, R4, G4, B4, R5]   (u16 positions 0–7)
//! v2 = [G5, B5, R6, G6, B6, R7, G7, B7]   (u16 positions 0–7)
//! ```
//!
//! `_mm_shuffle_epi8` with zeroing masks extracts each channel into a
//! u16×8 vector; the three per-register partial results are OR-combined.
//!
//! ### Rgba64 / Bgra64 (stride-4)
//!
//! 8 pixels = 32 u16 elements = 64 bytes = 4 × 128-bit loads.
//!
//! ```text
//! v0 = [R0, G0, B0, A0, R1, G1, B1, A1]
//! v1 = [R2, G2, B2, A2, R3, G3, B3, A3]
//! v2 = [R4, G4, B4, A4, R5, G5, B5, A5]
//! v3 = [R6, G6, B6, A6, R7, G7, B7, A7]
//! ```
//!
//! Same shuffle-then-OR approach; channels land at periodic u16 offsets
//! (0, 2, 4, 6 within each register pair), so masks are simpler.
//!
//! ## Depth conversion
//!
//! - **u16 → u8:** `_mm_srli_epi16(v, 8)` then `_mm_packus_epi16(v, zero)`
//!   → 8 u8 in the low half of a 128-bit register, matching scalar `(v >> 8) as u8`.
//! - **u16 → u16:** `write_rgb_u16_8` / `write_rgba_u16_8` interleave 8 u16
//!   lanes per channel into packed RGB/RGBA u16 output.
//!
//! ## Output helpers
//!
//! - **u8 output, 3-channel:** `write_rgb_16` writes 16 pixels (48 bytes); only
//!   the first 8 pixels (24 bytes) are valid. Write to a 48-byte temp, then
//!   `copy_nonoverlapping` 24 bytes (same pattern as `planar_gbr_high_bit.rs`).
//! - **u8 output, 4-channel:** `write_rgba_16` writes 16 pixels (64 bytes); copy
//!   32 bytes.
//! - **u16 output:** `write_rgb_u16_8` / `write_rgba_u16_8` write exactly 8
//!   pixels directly to the destination.
//!
//! ## Scalar tail
//!
//! All kernels handle `width % 8` remaining pixels via the scalar reference.
// Kernels are wired into the dispatcher in the SIMD dispatch task; suppress
// dead_code until then.
#![allow(dead_code)]

use super::*;

// Deinterleave helpers — Rgb48 / Bgr48 (3 u16 per pixel, 3 loads per 8 px).
//
// After three 128-bit loads for 8 pixels of a stride-3 u16 source:
//
//   v0 byte layout (u16 indices → byte indices):
//     [R0,G0,B0,R1,G1,B1,R2,G2] = bytes [0..1, 2..3, 4..5, 6..7, 8..9, 10..11, 12..13, 14..15]
//   v1:
//     [B2,R3,G3,B3,R4,G4,B4,R5] = bytes [0..1, 2..3, 4..5, 6..7, 8..9, 10..11, 12..13, 14..15]
//   v2:
//     [G5,B5,R6,G6,B6,R7,G7,B7] = bytes [0..1, 2..3, 4..5, 6..7, 8..9, 10..11, 12..13, 14..15]
//
// Resulting channel u16×8 vectors:
//   R = [R0,R1,R2,R3,R4,R5,R6,R7]
//   G = [G0,G1,G2,G3,G4,G5,G6,G7]
//   B = [B0,B1,B2,B3,B4,B5,B6,B7]
//
// For R channel:
//   • R0 (v0 u16-pos 0, bytes 0,1)  → output u16-pos 0 (bytes 0,1)
//   • R1 (v0 u16-pos 3, bytes 6,7)  → output u16-pos 1 (bytes 2,3)
//   • R2 (v0 u16-pos 6, bytes 12,13)→ output u16-pos 2 (bytes 4,5)
//   • R3 (v1 u16-pos 1, bytes 2,3)  → output u16-pos 3 (bytes 6,7)
//   • R4 (v1 u16-pos 4, bytes 8,9)  → output u16-pos 4 (bytes 8,9)
//   • R5 (v1 u16-pos 7, bytes 14,15)→ output u16-pos 5 (bytes 10,11)
//   • R6 (v2 u16-pos 2, bytes 4,5)  → output u16-pos 6 (bytes 12,13)
//   • R7 (v2 u16-pos 5, bytes 10,11)→ output u16-pos 7 (bytes 14,15)
//
// For G channel:
//   • G0 (v0 pos 1, bytes 2,3)   → out pos 0
//   • G1 (v0 pos 4, bytes 8,9)   → out pos 1
//   • G2 (v0 pos 7, bytes 14,15) → out pos 2
//   • G3 (v1 pos 2, bytes 4,5)   → out pos 3
//   • G4 (v1 pos 5, bytes 10,11) → out pos 4
//   • G5 (v2 pos 0, bytes 0,1)   → out pos 5
//   • G6 (v2 pos 3, bytes 6,7)   → out pos 6
//   • G7 (v2 pos 6, bytes 12,13) → out pos 7
//
// For B channel:
//   • B0 (v0 pos 2, bytes 4,5)   → out pos 0
//   • B1 (v0 pos 5, bytes 10,11) → out pos 1
//   • B2 (v1 pos 0, bytes 0,1)   → out pos 2
//   • B3 (v1 pos 3, bytes 6,7)   → out pos 3
//   • B4 (v1 pos 6, bytes 12,13) → out pos 4
//   • B5 (v2 pos 1, bytes 2,3)   → out pos 5
//   • B6 (v2 pos 4, bytes 8,9)   → out pos 6
//   • B7 (v2 pos 7, bytes 14,15) → out pos 7

/// Deinterleave 8 pixels of stride-3 u16 (Rgb48 or Bgr48 layout) from three
/// 128-bit registers `(v0, v1, v2)` into three separate u16×8 channel vectors
/// `(ch0, ch1, ch2)` where `ch0` = the first channel in memory order.
///
/// For Rgb48: `ch0 = R`, `ch1 = G`, `ch2 = B`.
/// For Bgr48: `ch0 = B`, `ch1 = G`, `ch2 = R`; swap on output as needed.
///
/// # Safety
///
/// Caller must have verified SSE4.1 availability.
#[inline(always)]
unsafe fn deinterleave_rgb48_8px(
  v0: __m128i,
  v1: __m128i,
  v2: __m128i,
) -> (__m128i, __m128i, __m128i) {
  unsafe {
    // ---- ch0 (first channel: R for Rgb48, B for Bgr48) ---------------------
    // From v0: want u16-positions 0, 3, 6 → output positions 0, 1, 2.
    let ch0_v0 = _mm_setr_epi8(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From v1: want u16-positions 1, 4, 7 → output positions 3, 4, 5.
    let ch0_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1);
    // From v2: want u16-positions 2, 5 → output positions 6, 7.
    let ch0_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11);
    let ch0 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, ch0_v0), _mm_shuffle_epi8(v1, ch0_v1)),
      _mm_shuffle_epi8(v2, ch0_v2),
    );

    // ---- ch1 (middle channel: G for both Rgb48 and Bgr48) ------------------
    // From v0: want u16-positions 1, 4, 7 → output positions 0, 1, 2.
    let ch1_v0 = _mm_setr_epi8(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From v1: want u16-positions 2, 5 → output positions 3, 4.
    let ch1_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1);
    // From v2: want u16-positions 0, 3, 6 → output positions 5, 6, 7.
    let ch1_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13);
    let ch1 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, ch1_v0), _mm_shuffle_epi8(v1, ch1_v1)),
      _mm_shuffle_epi8(v2, ch1_v2),
    );

    // ---- ch2 (third channel: B for Rgb48, R for Bgr48) --------------------
    // From v0: want u16-positions 2, 5 → output positions 0, 1.
    let ch2_v0 = _mm_setr_epi8(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    // From v1: want u16-positions 0, 3, 6 → output positions 2, 3, 4.
    let ch2_v1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1);
    // From v2: want u16-positions 1, 4, 7 → output positions 5, 6, 7.
    let ch2_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15);
    let ch2 = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, ch2_v0), _mm_shuffle_epi8(v1, ch2_v1)),
      _mm_shuffle_epi8(v2, ch2_v2),
    );

    (ch0, ch1, ch2)
  }
}

// Deinterleave helpers — Rgba64 / Bgra64 (4 u16 per pixel, 4 loads per 8 px).
//
// After four 128-bit loads for 8 pixels of a stride-4 u16 source:
//
//   v0 = [C0_0, C1_0, C2_0, C3_0, C0_1, C1_1, C2_1, C3_1]  (pixels 0, 1)
//   v1 = [C0_2, C1_2, C2_2, C3_2, C0_3, C1_3, C2_3, C3_3]  (pixels 2, 3)
//   v2 = [C0_4, C1_4, C2_4, C3_4, C0_5, C1_5, C2_5, C3_5]  (pixels 4, 5)
//   v3 = [C0_6, C1_6, C2_6, C3_6, C0_7, C1_7, C2_7, C3_7]  (pixels 6, 7)
//
// For Rgba64: C0=R, C1=G, C2=B, C3=A.
// For Bgra64: C0=B, C1=G, C2=R, C3=A.
//
// Each channel Ci occurs at u16 positions {offset, offset+4} within each
// register (offset = channel index 0..3 for Ci).
//
// For ch0 (channel index 0, bytes 0,1 and 8,9 within each register):
//   • C0_0 (v0 bytes 0,1)  → output u16-pos 0 (bytes 0,1)
//   • C0_1 (v0 bytes 8,9)  → output u16-pos 1 (bytes 2,3)
//   • C0_2 (v1 bytes 0,1)  → output u16-pos 2 (bytes 4,5)
//   • C0_3 (v1 bytes 8,9)  → output u16-pos 3 (bytes 6,7)
//   • C0_4 (v2 bytes 0,1)  → output u16-pos 4 (bytes 8,9)
//   • C0_5 (v2 bytes 8,9)  → output u16-pos 5 (bytes 10,11)
//   • C0_6 (v3 bytes 0,1)  → output u16-pos 6 (bytes 12,13)
//   • C0_7 (v3 bytes 8,9)  → output u16-pos 7 (bytes 14,15)

/// Deinterleave 8 pixels of stride-4 u16 (Rgba64 or Bgra64 layout) from four
/// 128-bit registers into four separate u16×8 channel vectors.
///
/// Returns `(ch0, ch1, ch2, ch3)` in memory order.
/// For Rgba64: `(R, G, B, A)`. For Bgra64: `(B, G, R, A)`.
///
/// # Safety
///
/// Caller must have verified SSE4.1 availability.
#[inline(always)]
unsafe fn deinterleave_rgba64_8px(
  v0: __m128i,
  v1: __m128i,
  v2: __m128i,
  v3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
  unsafe {
    // Each channel Ci is extracted from byte offsets (2*i, 2*i+1) and
    // (2*i+8, 2*i+9) within each register. Generic helper:
    //   from_v0/v1: pick bytes at offsets (b, b+1, b+8, b+9) → out positions 0,1 / 2,3
    //   from_v2/v3: same → out positions 4,5 / 6,7
    // where b = 2 * channel_index (0, 2, 4, or 6).

    // ---- ch0 (channel index 0, byte offset b=0) ----------------------------
    // Output: [C0_0,C0_1,C0_2,C0_3,C0_4,C0_5,C0_6,C0_7] at bytes [0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15]
    // From v0: bytes 0,1 → out 0,1; bytes 8,9 → out 2,3
    // From v1: bytes 0,1 → out 4,5; bytes 8,9 → out 6,7
    // From v2: bytes 0,1 → out 8,9; bytes 8,9 → out 10,11
    // From v3: bytes 0,1 → out 12,13; bytes 8,9 → out 14,15
    let c0_from_v0 = _mm_setr_epi8(0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let c0_from_v1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1);
    let c0_from_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 8, 9, -1, -1, -1, -1);
    let c0_from_v3 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 8, 9);
    let ch0 = _mm_or_si128(
      _mm_or_si128(
        _mm_shuffle_epi8(v0, c0_from_v0),
        _mm_shuffle_epi8(v1, c0_from_v1),
      ),
      _mm_or_si128(
        _mm_shuffle_epi8(v2, c0_from_v2),
        _mm_shuffle_epi8(v3, c0_from_v3),
      ),
    );

    // ---- ch1 (channel index 1, byte offset b=2) ----------------------------
    let c1_from_v0 = _mm_setr_epi8(2, 3, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let c1_from_v1 = _mm_setr_epi8(-1, -1, -1, -1, 2, 3, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1);
    let c1_from_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 10, 11, -1, -1, -1, -1);
    let c1_from_v3 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 10, 11);
    let ch1 = _mm_or_si128(
      _mm_or_si128(
        _mm_shuffle_epi8(v0, c1_from_v0),
        _mm_shuffle_epi8(v1, c1_from_v1),
      ),
      _mm_or_si128(
        _mm_shuffle_epi8(v2, c1_from_v2),
        _mm_shuffle_epi8(v3, c1_from_v3),
      ),
    );

    // ---- ch2 (channel index 2, byte offset b=4) ----------------------------
    let c2_from_v0 = _mm_setr_epi8(4, 5, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let c2_from_v1 = _mm_setr_epi8(-1, -1, -1, -1, 4, 5, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1);
    let c2_from_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 12, 13, -1, -1, -1, -1);
    let c2_from_v3 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 12, 13);
    let ch2 = _mm_or_si128(
      _mm_or_si128(
        _mm_shuffle_epi8(v0, c2_from_v0),
        _mm_shuffle_epi8(v1, c2_from_v1),
      ),
      _mm_or_si128(
        _mm_shuffle_epi8(v2, c2_from_v2),
        _mm_shuffle_epi8(v3, c2_from_v3),
      ),
    );

    // ---- ch3 (channel index 3, byte offset b=6) ----------------------------
    let c3_from_v0 = _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let c3_from_v1 = _mm_setr_epi8(-1, -1, -1, -1, 6, 7, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let c3_from_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 6, 7, 14, 15, -1, -1, -1, -1);
    let c3_from_v3 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, 7, 14, 15);
    let ch3 = _mm_or_si128(
      _mm_or_si128(
        _mm_shuffle_epi8(v0, c3_from_v0),
        _mm_shuffle_epi8(v1, c3_from_v1),
      ),
      _mm_or_si128(
        _mm_shuffle_epi8(v2, c3_from_v2),
        _mm_shuffle_epi8(v3, c3_from_v3),
      ),
    );

    (ch0, ch1, ch2, ch3)
  }
}

// u16 → u8 narrowing: `>> 8` + `_mm_packus_epi16`.
/// Narrow a u16×8 vector to u8×8 (in the low half) via logical right-shift by 8.
///
/// Equivalent to scalar `(v >> 8) as u8`. Zero-packs the high half.
#[inline(always)]
unsafe fn narrow_u16x8_to_u8x8(v: __m128i, zero: __m128i) -> __m128i {
  unsafe { _mm_packus_epi16(_mm_srli_epi16::<8>(v), zero) }
}

// ---- endian byte-swap helper ------------------------------------------------

/// Compile-time host endianness. `true` on BE targets, `false` on LE.
///
/// Used by [`byteswap_if_be`] to gate the byte-swap on `BE != HOST_NATIVE_BE`
/// so the swap fires only when the wire endian differs from the host's
/// native byte order — covering all four `wire × host` quadrants. Mirrors
/// the gate established in `gray.rs` and the canonical NEON
/// `bswap_u16x8_if_be` helper.
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
/// Uses `_mm_shuffle_epi8` (SSSE3, a subset of SSE4.1) with the same mask as
/// `endian::BYTESWAP_MASK_U16`. The unused branch folds at compile time
/// since `BE` and `HOST_NATIVE_BE` are both compile-time constants.
#[inline(always)]
unsafe fn byteswap_if_be<const BE: bool>(v: __m128i) -> __m128i {
  if BE != HOST_NATIVE_BE {
    // Swap bytes within each u16 lane: [1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 13,12, 15,14]
    const MASK: __m128i =
      unsafe { core::mem::transmute([1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]) };
    unsafe { _mm_shuffle_epi8(v, MASK) }
  } else {
    v
  }
}

// Rgb48 (R, G, B — 3 u16 elements per pixel).
/// SSE4.1 Rgb48 → packed u8 RGB. 8 pixels per SIMD iteration.
///
/// Loads 3 × 128-bit chunks (24 u16), deinterleaves with shuffle masks,
/// narrows via `>> 8`, writes 8 pixels (24 bytes) of interleaved RGB.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgb48_to_rgb_row<const BE: bool>(
  rgb48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (r_u16, g_u16, b_u16) = deinterleave_rgb48_8px(v0, v1, v2);
      let r_u8 = narrow_u16x8_to_u8x8(r_u16, zero);
      let g_u8 = narrow_u16x8_to_u8x8(g_u16, zero);
      let b_u8 = narrow_u16x8_to_u8x8(b_u16, zero);
      // write_rgb_16 writes 16 px (48 bytes); only first 8 px (24 bytes) valid.
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

/// SSE4.1 Rgb48 → packed u8 RGBA. 8 pixels per SIMD iteration. Alpha forced to 0xFF.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgb48_to_rgba_row<const BE: bool>(
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
    while x + 8 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (r_u16, g_u16, b_u16) = deinterleave_rgb48_8px(v0, v1, v2);
      let r_u8 = narrow_u16x8_to_u8x8(r_u16, zero);
      let g_u8 = narrow_u16x8_to_u8x8(g_u16, zero);
      let b_u8 = narrow_u16x8_to_u8x8(b_u16, zero);
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

/// SSE4.1 Rgb48 → native-depth u16 RGB. 8 pixels per iteration.
///
/// Deinterleaves with shuffle masks, writes 8 pixels via `write_rgb_u16_8`.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgb48_to_rgb_u16_row<const BE: bool>(
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
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (r_u16, g_u16, b_u16) = deinterleave_rgb48_8px(v0, v1, v2);
      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgb_u16_row::<BE>(&rgb48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 Rgb48 → native-depth u16 RGBA. 8 pixels per iteration. Alpha forced to 0xFFFF.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgb48_to_rgba_u16_row<const BE: bool>(
  rgb48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = _mm_set1_epi16(0xFFFFu16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgb48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (r_u16, g_u16, b_u16) = deinterleave_rgb48_8px(v0, v1, v2);
      write_rgba_u16_8(
        r_u16,
        g_u16,
        b_u16,
        opaque,
        rgba_out.as_mut_ptr().add(x * 4),
      );
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgba_u16_row::<BE>(&rgb48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// Bgr48 (B, G, R — 3 u16 elements per pixel).
/// SSE4.1 Bgr48 → packed u8 RGB. 8 pixels per SIMD iteration.
///
/// `deinterleave_rgb48_8px` yields `(B, G, R)` in source memory order;
/// the B↔R swap is applied by passing them as `(R=ch2, G=ch1, B=ch0)`.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_bgr48_to_rgb_row<const BE: bool>(
  bgr48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      // ch0=B, ch1=G, ch2=R (source BGR order)
      let (b_u16, g_u16, r_u16) = deinterleave_rgb48_8px(v0, v1, v2);
      let r_u8 = narrow_u16x8_to_u8x8(r_u16, zero);
      let g_u8 = narrow_u16x8_to_u8x8(g_u16, zero);
      let b_u8 = narrow_u16x8_to_u8x8(b_u16, zero);
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

/// SSE4.1 Bgr48 → packed u8 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap; alpha forced to 0xFF.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_bgr48_to_rgba_row<const BE: bool>(
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
    while x + 8 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (b_u16, g_u16, r_u16) = deinterleave_rgb48_8px(v0, v1, v2);
      let r_u8 = narrow_u16x8_to_u8x8(r_u16, zero);
      let g_u8 = narrow_u16x8_to_u8x8(g_u16, zero);
      let b_u8 = narrow_u16x8_to_u8x8(b_u16, zero);
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

/// SSE4.1 Bgr48 → native-depth u16 RGB. 8 pixels per SIMD iteration.
/// B↔R swap; values unchanged.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_bgr48_to_rgb_u16_row<const BE: bool>(
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
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (b_u16, g_u16, r_u16) = deinterleave_rgb48_8px(v0, v1, v2);
      // Store as R, G, B (swap applied by argument order)
      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgb_u16_row::<BE>(&bgr48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 Bgr48 → native-depth u16 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap; alpha forced to 0xFFFF.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_bgr48_to_rgba_u16_row<const BE: bool>(
  bgr48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = _mm_set1_epi16(0xFFFFu16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgr48.as_ptr().add(x * 3);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let (b_u16, g_u16, r_u16) = deinterleave_rgb48_8px(v0, v1, v2);
      write_rgba_u16_8(
        r_u16,
        g_u16,
        b_u16,
        opaque,
        rgba_out.as_mut_ptr().add(x * 4),
      );
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgba_u16_row::<BE>(&bgr48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// Rgba64 (R, G, B, A — 4 u16 elements per pixel).
/// SSE4.1 Rgba64 → packed u8 RGB. 8 pixels per SIMD iteration. Alpha discarded.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgba64_to_rgb_row<const BE: bool>(
  rgba64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let v3 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(24).cast()));
      let (r_u16, g_u16, b_u16, _a) = deinterleave_rgba64_8px(v0, v1, v2, v3);
      let r_u8 = narrow_u16x8_to_u8x8(r_u16, zero);
      let g_u8 = narrow_u16x8_to_u8x8(g_u16, zero);
      let b_u8 = narrow_u16x8_to_u8x8(b_u16, zero);
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

/// SSE4.1 Rgba64 → packed u8 RGBA. 8 pixels per SIMD iteration. Source alpha passes through.
///
/// All 4 channels narrowed via `>> 8`.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgba64_to_rgba_row<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = rgba64.as_ptr().add(x * 4);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let v3 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(24).cast()));
      let (r_u16, g_u16, b_u16, a_u16) = deinterleave_rgba64_8px(v0, v1, v2, v3);
      let r_u8 = narrow_u16x8_to_u8x8(r_u16, zero);
      let g_u8 = narrow_u16x8_to_u8x8(g_u16, zero);
      let b_u8 = narrow_u16x8_to_u8x8(b_u16, zero);
      let a_u8 = narrow_u16x8_to_u8x8(a_u16, zero);
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

/// SSE4.1 Rgba64 → native-depth u16 RGB. 8 pixels per SIMD iteration. Alpha discarded.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgba64_to_rgb_u16_row<const BE: bool>(
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
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let v3 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(24).cast()));
      let (r_u16, g_u16, b_u16, _a) = deinterleave_rgba64_8px(v0, v1, v2, v3);
      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgb_u16_row::<BE>(&rgba64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 Rgba64 → native-depth u16 RGBA (identity copy). 8 pixels per iteration.
///
/// All 4 channels passed through at native depth; source alpha preserved.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgba64_to_rgba_u16_row<const BE: bool>(
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
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let v3 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(24).cast()));
      let (r_u16, g_u16, b_u16, a_u16) = deinterleave_rgba64_8px(v0, v1, v2, v3);
      write_rgba_u16_8(r_u16, g_u16, b_u16, a_u16, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgba_u16_row::<BE>(&rgba64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// Bgra64 (B, G, R, A — 4 u16 elements per pixel).
/// SSE4.1 Bgra64 → packed u8 RGB. 8 pixels per SIMD iteration.
/// B↔R swap; alpha discarded.
///
/// `deinterleave_rgba64_8px` yields `(B, G, R, A)` in source memory order.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_bgra64_to_rgb_row<const BE: bool>(
  bgra64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let v3 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(24).cast()));
      // ch0=B, ch1=G, ch2=R, ch3=A (source BGRA order)
      let (b_u16, g_u16, r_u16, _a) = deinterleave_rgba64_8px(v0, v1, v2, v3);
      let r_u8 = narrow_u16x8_to_u8x8(r_u16, zero);
      let g_u8 = narrow_u16x8_to_u8x8(g_u16, zero);
      let b_u8 = narrow_u16x8_to_u8x8(b_u16, zero);
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

/// SSE4.1 Bgra64 → packed u8 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap; source alpha passes through (narrowed via `>> 8`).
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_bgra64_to_rgba_row<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let ptr = bgra64.as_ptr().add(x * 4);
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let v3 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(24).cast()));
      let (b_u16, g_u16, r_u16, a_u16) = deinterleave_rgba64_8px(v0, v1, v2, v3);
      let r_u8 = narrow_u16x8_to_u8x8(r_u16, zero);
      let g_u8 = narrow_u16x8_to_u8x8(g_u16, zero);
      let b_u8 = narrow_u16x8_to_u8x8(b_u16, zero);
      let a_u8 = narrow_u16x8_to_u8x8(a_u16, zero);
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

/// SSE4.1 Bgra64 → native-depth u16 RGB. 8 pixels per SIMD iteration.
/// B↔R swap; alpha discarded.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_bgra64_to_rgb_u16_row<const BE: bool>(
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
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let v3 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(24).cast()));
      let (b_u16, g_u16, r_u16, _a) = deinterleave_rgba64_8px(v0, v1, v2, v3);
      // Swap B↔R: store (R, G, B)
      write_rgb_u16_8(r_u16, g_u16, b_u16, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgb_u16_row::<BE>(&bgra64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 Bgra64 → native-depth u16 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap; source alpha preserved at position 3.
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_bgra64_to_rgba_u16_row<const BE: bool>(
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
      let v0 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
      let v1 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
      let v2 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(16).cast()));
      let v3 = byteswap_if_be::<BE>(_mm_loadu_si128(ptr.add(24).cast()));
      // Swap B↔R: store (R=ch2, G=ch1, B=ch0, A=ch3)
      let (b_u16, g_u16, r_u16, a_u16) = deinterleave_rgba64_8px(v0, v1, v2, v3);
      write_rgba_u16_8(r_u16, g_u16, b_u16, a_u16, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgba_u16_row::<BE>(&bgra64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}
