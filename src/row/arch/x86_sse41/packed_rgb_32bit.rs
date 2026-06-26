//! SSE4.1 kernels for 32-bit packed RGB / RGBA sources (Rgb96 / Rgba128).
//!
//! ## Format layouts
//!
//! | Format  | Elements per pixel | Channel order in memory |
//! |---------|--------------------|------------------------|
//! | Rgb96   | 3 u32              | R, G, B                |
//! | Rgba128 | 4 u32              | R, G, B, A             |
//!
//! ## Per-format SIMD strategy (8 pixels per SIMD iteration)
//!
//! Each 4-pixel sub-group is loaded as three (Rgb96) / four (Rgba128) 128-bit
//! registers (4 u32 lanes each), byte-swapped per u32 lane when `BE = true`,
//! and deinterleaved into per-channel `u32x4` lane vectors:
//! - **Rgb96** uses `_mm_shuffle_epi8` gather masks (the u32 analogue of the
//!   16-bit `deinterleave_rgb48_8px`).
//! - **Rgba128** uses the SSE2 `unpacklo/​hi` 4x4 transpose ladder.
//!
//! Two sub-groups (8 pixels) feed the shared writer helpers
//! ([`super::write_rgb_16`] / [`super::write_rgba_16`] for u8,
//! [`super::write_rgb_u16_8`] / [`super::write_rgba_u16_8`] for u16).
//!
//! ## Depth conversion
//!
//! - **u32 → u8:**  `_mm_srli_epi32::<24>` then a two-stage
//!   `_mm_packus_epi32` / `_mm_packus_epi16` narrow — net `>> 24`, matching
//!   scalar `(v >> 24) as u8`.
//! - **u32 → u16:** `_mm_srli_epi32::<16>` then `_mm_packus_epi32` — `>> 16`,
//!   matching scalar `(v >> 16) as u16`. Values fit in their target width
//!   after the shift, so the saturating packs never clamp.
//!
//! ## Scalar tail
//!
//! All kernels handle `width % 8` remaining pixels via the scalar reference.
// Kernels are wired into the dispatcher in the SIMD dispatch task; suppress
// dead_code until then.
#![allow(dead_code)]

use super::*;

// ---- endian byte-swap helper ------------------------------------------------

/// Compile-time host endianness. `true` on BE targets, `false` on LE.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Conditionally byte-swap every u32 lane in `v` so the returned value is in
/// **host-native** byte order regardless of the host endianness. The gate is
/// `BE != HOST_NATIVE_BE` — the swap fires only when the wire endian differs
/// from the host's native byte order.
#[inline(always)]
unsafe fn byteswap32_if_be<const BE: bool>(v: __m128i) -> __m128i {
  if BE != HOST_NATIVE_BE {
    // Reverse the four bytes inside every 32-bit lane.
    const MASK: __m128i =
      unsafe { core::mem::transmute([3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]) };
    unsafe { _mm_shuffle_epi8(v, MASK) }
  } else {
    v
  }
}

// ---- Rgb96 deinterleave (3 u32 per pixel, 3 loads per 4 px) -----------------

/// Deinterleave 4 pixels of stride-3 u32 (Rgb96 layout) from three 128-bit
/// registers into three `u32x4` channel lane vectors `(R, G, B)`.
///
/// Input lane layout: `v0 = [R0,G0,B0,R1]`, `v1 = [G1,B1,R2,G2]`,
/// `v2 = [B2,R3,G3,B3]`.
///
/// # Safety
///
/// Caller must have verified SSE4.1 availability.
#[inline(always)]
unsafe fn deinterleave_rgb96_4px(
  v0: __m128i,
  v1: __m128i,
  v2: __m128i,
) -> (__m128i, __m128i, __m128i) {
  unsafe {
    let r_v0 = _mm_setr_epi8(0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    let r_v1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
    let r_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
    let r = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, r_v0), _mm_shuffle_epi8(v1, r_v1)),
      _mm_shuffle_epi8(v2, r_v2),
    );

    let g_v0 = _mm_setr_epi8(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let g_v1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1);
    let g_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
    let g = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, g_v0), _mm_shuffle_epi8(v1, g_v1)),
      _mm_shuffle_epi8(v2, g_v2),
    );

    let b_v0 = _mm_setr_epi8(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let b_v1 = _mm_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    let b_v2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15);
    let b = _mm_or_si128(
      _mm_or_si128(_mm_shuffle_epi8(v0, b_v0), _mm_shuffle_epi8(v1, b_v1)),
      _mm_shuffle_epi8(v2, b_v2),
    );

    (r, g, b)
  }
}

/// Loads, byte-swaps, and deinterleaves 4 pixels of Rgb96 into `(R, G, B)`
/// `u32x4` lane vectors.
///
/// # Safety
///
/// `ptr` must point to at least 12 readable u32; SSE4.1 must be available.
#[inline(always)]
unsafe fn load_deint_rgb96_4px<const BE: bool>(ptr: *const u32) -> (__m128i, __m128i, __m128i) {
  unsafe {
    let v0 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
    let v1 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.add(4).cast()));
    let v2 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
    deinterleave_rgb96_4px(v0, v1, v2)
  }
}

// ---- u32 narrowing ----------------------------------------------------------

/// Narrows four `u32x4` lane vectors (`>> 24` already applied, value in the
/// low byte of each lane) into one packed `u8x16` channel vector via the
/// two-stage `_mm_packus_epi32` / `_mm_packus_epi16` ladder.
#[inline(always)]
unsafe fn pack_u32x4_quad_to_u8x16(v0: __m128i, v1: __m128i, v2: __m128i, v3: __m128i) -> __m128i {
  unsafe { _mm_packus_epi16(_mm_packus_epi32(v0, v1), _mm_packus_epi32(v2, v3)) }
}

// Rgb96 (R, G, B — 3 u32 elements per pixel).

/// SSE4.1 Rgb96 → packed u8 RGB. 8 pixels per SIMD iteration.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available (caller obligation).
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgb96_to_rgb_row<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgb96.as_ptr().add(x * 3);
      let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(p);
      let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(p.add(12));
      let r = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(r0),
        _mm_srli_epi32::<24>(r1),
        zero,
        zero,
      );
      let g = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(g0),
        _mm_srli_epi32::<24>(g1),
        zero,
        zero,
      );
      let b = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(b0),
        _mm_srli_epi32::<24>(b1),
        zero,
        zero,
      );
      // write_rgb_16 writes 16 px (48 bytes); only first 8 px (24 bytes) valid.
      let mut tmp = [0u8; 48];
      write_rgb_16(r, g, b, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgb_row::<BE>(&rgb96[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 Rgb96 → packed u8 RGBA. 8 pixels per SIMD iteration. Alpha forced to 0xFF.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgb96_to_rgba_row<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let opaque = _mm_set1_epi8(-1i8);
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgb96.as_ptr().add(x * 3);
      let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(p);
      let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(p.add(12));
      let r = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(r0),
        _mm_srli_epi32::<24>(r1),
        zero,
        zero,
      );
      let g = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(g0),
        _mm_srli_epi32::<24>(g1),
        zero,
        zero,
      );
      let b = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(b0),
        _mm_srli_epi32::<24>(b1),
        zero,
        zero,
      );
      let mut tmp = [0u8; 64];
      write_rgba_16(r, g, b, opaque, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgba_row::<BE>(&rgb96[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// SSE4.1 Rgb96 → native-depth u16 RGB. 8 pixels per SIMD iteration.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgb96_to_rgb_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgb96.as_ptr().add(x * 3);
      let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(p);
      let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(p.add(12));
      let r = _mm_packus_epi32(_mm_srli_epi32::<16>(r0), _mm_srli_epi32::<16>(r1));
      let g = _mm_packus_epi32(_mm_srli_epi32::<16>(g0), _mm_srli_epi32::<16>(g1));
      let b = _mm_packus_epi32(_mm_srli_epi32::<16>(b0), _mm_srli_epi32::<16>(b1));
      write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgb_u16_row::<BE>(&rgb96[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 Rgb96 → native-depth u16 RGBA. 8 pixels per SIMD iteration. Alpha forced to 0xFFFF.
///
/// When `BE = true` each loaded register is byte-swapped before deinterleaving.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgb96_to_rgba_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let opaque = _mm_set1_epi16(0xFFFFu16 as i16);
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgb96.as_ptr().add(x * 3);
      let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(p);
      let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(p.add(12));
      let r = _mm_packus_epi32(_mm_srli_epi32::<16>(r0), _mm_srli_epi32::<16>(r1));
      let g = _mm_packus_epi32(_mm_srli_epi32::<16>(g0), _mm_srli_epi32::<16>(g1));
      let b = _mm_packus_epi32(_mm_srli_epi32::<16>(b0), _mm_srli_epi32::<16>(b1));
      write_rgba_u16_8(r, g, b, opaque, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::rgb96_to_rgba_u16_row::<BE>(&rgb96[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// ---- Rgba128 deinterleave (4 u32 per pixel, 4 loads per 4 px) ---------------

/// Deinterleave 4 pixels of stride-4 u32 (Rgba128 layout) from four 128-bit
/// registers `v_n = [Rn, Gn, Bn, An]` into four `u32x4` channel lane vectors
/// `(R, G, B, A)` via the SSE2 `unpacklo/hi` 4x4 transpose ladder.
///
/// # Safety
///
/// Caller must have verified SSE4.1 availability.
#[inline(always)]
unsafe fn deinterleave_rgba128_4px(
  v0: __m128i,
  v1: __m128i,
  v2: __m128i,
  v3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
  unsafe {
    let t0 = _mm_unpacklo_epi32(v0, v1); // [R0, R1, G0, G1]
    let t1 = _mm_unpackhi_epi32(v0, v1); // [B0, B1, A0, A1]
    let t2 = _mm_unpacklo_epi32(v2, v3); // [R2, R3, G2, G3]
    let t3 = _mm_unpackhi_epi32(v2, v3); // [B2, B3, A2, A3]
    let r = _mm_unpacklo_epi64(t0, t2); // [R0, R1, R2, R3]
    let g = _mm_unpackhi_epi64(t0, t2); // [G0, G1, G2, G3]
    let b = _mm_unpacklo_epi64(t1, t3); // [B0, B1, B2, B3]
    let a = _mm_unpackhi_epi64(t1, t3); // [A0, A1, A2, A3]
    (r, g, b, a)
  }
}

/// Loads, byte-swaps, and deinterleaves 4 pixels of Rgba128 into `(R, G, B, A)`
/// `u32x4` lane vectors.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable u32; SSE4.1 must be available.
#[inline(always)]
unsafe fn load_deint_rgba128_4px<const BE: bool>(
  ptr: *const u32,
) -> (__m128i, __m128i, __m128i, __m128i) {
  unsafe {
    let v0 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
    let v1 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.add(4).cast()));
    let v2 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
    let v3 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.add(12).cast()));
    deinterleave_rgba128_4px(v0, v1, v2, v3)
  }
}

// Rgba128 (R, G, B, A — 4 u32 elements per pixel).

/// SSE4.1 Rgba128 → packed u8 RGB. 8 pixels per SIMD iteration. Alpha discarded.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgba128_to_rgb_row<const BE: bool>(
  rgba128: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgba128.as_ptr().add(x * 4);
      let (r0, g0, b0, _) = load_deint_rgba128_4px::<BE>(p);
      let (r1, g1, b1, _) = load_deint_rgba128_4px::<BE>(p.add(16));
      let r = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(r0),
        _mm_srli_epi32::<24>(r1),
        zero,
        zero,
      );
      let g = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(g0),
        _mm_srli_epi32::<24>(g1),
        zero,
        zero,
      );
      let b = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(b0),
        _mm_srli_epi32::<24>(b1),
        zero,
        zero,
      );
      let mut tmp = [0u8; 48];
      write_rgb_16(r, g, b, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgb_row::<BE>(&rgba128[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 Rgba128 → packed u8 RGBA. 8 pixels per SIMD iteration. Source alpha
/// passes through (narrowed `>> 24`).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgba128_to_rgba_row<const BE: bool>(
  rgba128: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = _mm_setzero_si128();
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgba128.as_ptr().add(x * 4);
      let (r0, g0, b0, a0) = load_deint_rgba128_4px::<BE>(p);
      let (r1, g1, b1, a1) = load_deint_rgba128_4px::<BE>(p.add(16));
      let r = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(r0),
        _mm_srli_epi32::<24>(r1),
        zero,
        zero,
      );
      let g = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(g0),
        _mm_srli_epi32::<24>(g1),
        zero,
        zero,
      );
      let b = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(b0),
        _mm_srli_epi32::<24>(b1),
        zero,
        zero,
      );
      let a = pack_u32x4_quad_to_u8x16(
        _mm_srli_epi32::<24>(a0),
        _mm_srli_epi32::<24>(a1),
        zero,
        zero,
      );
      let mut tmp = [0u8; 64];
      write_rgba_16(r, g, b, a, tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgba_row::<BE>(&rgba128[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// SSE4.1 Rgba128 → native-depth u16 RGB. 8 pixels per SIMD iteration. Alpha discarded.
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgba128_to_rgb_u16_row<const BE: bool>(
  rgba128: &[u32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgba128.as_ptr().add(x * 4);
      let (r0, g0, b0, _) = load_deint_rgba128_4px::<BE>(p);
      let (r1, g1, b1, _) = load_deint_rgba128_4px::<BE>(p.add(16));
      let r = _mm_packus_epi32(_mm_srli_epi32::<16>(r0), _mm_srli_epi32::<16>(r1));
      let g = _mm_packus_epi32(_mm_srli_epi32::<16>(g0), _mm_srli_epi32::<16>(g1));
      let b = _mm_packus_epi32(_mm_srli_epi32::<16>(b0), _mm_srli_epi32::<16>(b1));
      write_rgb_u16_8(r, g, b, rgb_out.as_mut_ptr().add(x * 3));
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgb_u16_row::<BE>(&rgba128[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// SSE4.1 Rgba128 → native-depth u16 RGBA. 8 pixels per SIMD iteration. Source
/// alpha passes through (narrowed `>> 16`).
///
/// # Safety
///
/// 1. SSE4.1 must be available.
/// 2. `rgba128.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse41_rgba128_to_rgba_u16_row<const BE: bool>(
  rgba128: &[u32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba128.len() >= width * 4, "rgba128 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let p = rgba128.as_ptr().add(x * 4);
      let (r0, g0, b0, a0) = load_deint_rgba128_4px::<BE>(p);
      let (r1, g1, b1, a1) = load_deint_rgba128_4px::<BE>(p.add(16));
      let r = _mm_packus_epi32(_mm_srli_epi32::<16>(r0), _mm_srli_epi32::<16>(r1));
      let g = _mm_packus_epi32(_mm_srli_epi32::<16>(g0), _mm_srli_epi32::<16>(g1));
      let b = _mm_packus_epi32(_mm_srli_epi32::<16>(b0), _mm_srli_epi32::<16>(b1));
      let a = _mm_packus_epi32(_mm_srli_epi32::<16>(a0), _mm_srli_epi32::<16>(a1));
      write_rgba_u16_8(r, g, b, a, rgba_out.as_mut_ptr().add(x * 4));
      x += 8;
    }
    if x < width {
      scalar::rgba128_to_rgba_u16_row::<BE>(&rgba128[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}
