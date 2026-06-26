//! AVX-512 kernels for 32-bit packed RGB / RGBA sources (Rgb96 / Rgba128).
//!
//! ## Format layouts
//!
//! | Format  | Elements per pixel | Channel order in memory |
//! |---------|--------------------|------------------------|
//! | Rgb96   | 3 u32              | R, G, B                |
//! | Rgba128 | 4 u32              | R, G, B, A             |
//!
//! ## Per-format SIMD strategy (32 pixels per outer iteration)
//!
//! The stride-3 u32 deinterleave does not tile cleanly across 512-bit lanes,
//! so — exactly like the 16-bit AVX-512 sibling's 3-channel path — each
//! 32-pixel outer iteration is processed as **four** 8-pixel SSE4.1-style
//! halves (`_mm_shuffle_epi8` gather + `_mm_srli_epi32` narrow +
//! `_mm_packus_epi32` / `_mm_packus_epi16` pack), reusing the shared
//! [`super::write_rgb_16`] / [`super::write_rgba_16`] / [`super::write_rgb_u16_8`]
//! / [`super::write_rgba_u16_8`] writers. SSE4.1 / SSSE3 are subsets of
//! AVX-512, so these 128-bit intrinsics run unchanged in the AVX-512F+BW
//! `target_feature` context.
//!
//! ## Depth conversion
//!
//! - **u32 → u8:**  `_mm_srli_epi32::<24>` then two-stage saturating narrow
//!   (`>> 24`).
//! - **u32 → u16:** `_mm_srli_epi32::<16>` then `_mm_packus_epi32` (`>> 16`).
//!
//! ## Scalar tail
//!
//! 16-pixel and 8-pixel cleanup passes follow the 32-pixel main loop; the
//! final `width % 8` pixels use the scalar reference.
// Kernels are wired into the dispatcher in the SIMD dispatch task; suppress
// dead_code until then.
#![allow(dead_code)]

use super::*;

// ---- endian byte-swap helper ------------------------------------------------

/// Compile-time host endianness. `true` on BE targets, `false` on LE.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Conditionally byte-swap every u32 lane in `v` into host-native order.
#[inline(always)]
unsafe fn byteswap32_if_be<const BE: bool>(v: __m128i) -> __m128i {
  if BE != HOST_NATIVE_BE {
    const MASK: __m128i =
      unsafe { core::mem::transmute([3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]) };
    unsafe { _mm_shuffle_epi8(v, MASK) }
  } else {
    v
  }
}

/// Deinterleave 4 pixels of stride-3 u32 (Rgb96) into `(R, G, B)` `u32x4`
/// channel lane vectors. See the SSE4.1 sibling for the mask derivation.
///
/// # Safety
///
/// Caller must hold AVX-512F + AVX-512BW `target_feature` (SSSE3 subset).
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

/// Loads, byte-swaps, and deinterleaves 4 pixels of Rgb96.
///
/// # Safety
///
/// `ptr` must point to at least 12 readable u32; AVX-512F+BW must be available.
#[inline(always)]
unsafe fn load_deint_rgb96_4px<const BE: bool>(ptr: *const u32) -> (__m128i, __m128i, __m128i) {
  unsafe {
    let v0 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.cast()));
    let v1 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.add(4).cast()));
    let v2 = byteswap32_if_be::<BE>(_mm_loadu_si128(ptr.add(8).cast()));
    deinterleave_rgb96_4px(v0, v1, v2)
  }
}

/// Narrows four `u32x4` lane vectors (`>> 24` applied) into one `u8x16`.
#[inline(always)]
unsafe fn pack_u32x4_quad_to_u8x16(v0: __m128i, v1: __m128i, v2: __m128i, v3: __m128i) -> __m128i {
  unsafe { _mm_packus_epi16(_mm_packus_epi32(v0, v1), _mm_packus_epi32(v2, v3)) }
}

// ---- 8-pixel building blocks ------------------------------------------------

/// Emits 8 Rgb96 pixels of packed u8 RGB at `out_ptr` (24 bytes).
#[inline(always)]
unsafe fn block_rgb96_to_rgb_8px<const BE: bool>(src_ptr: *const u32, out_ptr: *mut u8) {
  unsafe {
    let zero = _mm_setzero_si128();
    let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(src_ptr);
    let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(src_ptr.add(12));
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
    core::ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, 24);
  }
}

/// Emits 8 Rgb96 pixels of packed u8 RGBA at `out_ptr` (32 bytes), alpha 0xFF.
#[inline(always)]
unsafe fn block_rgb96_to_rgba_8px<const BE: bool>(src_ptr: *const u32, out_ptr: *mut u8) {
  unsafe {
    let zero = _mm_setzero_si128();
    let opaque = _mm_set1_epi8(-1i8);
    let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(src_ptr);
    let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(src_ptr.add(12));
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
    core::ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, 32);
  }
}

/// Emits 8 Rgb96 pixels of native u16 RGB at `out_ptr` (48 bytes).
#[inline(always)]
unsafe fn block_rgb96_to_rgb_u16_8px<const BE: bool>(src_ptr: *const u32, out_ptr: *mut u16) {
  unsafe {
    let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(src_ptr);
    let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(src_ptr.add(12));
    let r = _mm_packus_epi32(_mm_srli_epi32::<16>(r0), _mm_srli_epi32::<16>(r1));
    let g = _mm_packus_epi32(_mm_srli_epi32::<16>(g0), _mm_srli_epi32::<16>(g1));
    let b = _mm_packus_epi32(_mm_srli_epi32::<16>(b0), _mm_srli_epi32::<16>(b1));
    write_rgb_u16_8(r, g, b, out_ptr);
  }
}

/// Emits 8 Rgb96 pixels of native u16 RGBA at `out_ptr` (64 bytes), alpha 0xFFFF.
#[inline(always)]
unsafe fn block_rgb96_to_rgba_u16_8px<const BE: bool>(src_ptr: *const u32, out_ptr: *mut u16) {
  unsafe {
    let opaque = _mm_set1_epi16(0xFFFFu16 as i16);
    let (r0, g0, b0) = load_deint_rgb96_4px::<BE>(src_ptr);
    let (r1, g1, b1) = load_deint_rgb96_4px::<BE>(src_ptr.add(12));
    let r = _mm_packus_epi32(_mm_srli_epi32::<16>(r0), _mm_srli_epi32::<16>(r1));
    let g = _mm_packus_epi32(_mm_srli_epi32::<16>(g0), _mm_srli_epi32::<16>(g1));
    let b = _mm_packus_epi32(_mm_srli_epi32::<16>(b0), _mm_srli_epi32::<16>(b1));
    write_rgba_u16_8(r, g, b, opaque, out_ptr);
  }
}

/// Runs `block` over `width` pixels in 32 → 16 → 8 pixel passes, falling to
/// `tail` for the final `width % 8` pixels. `IN_PER_PX` / `OUT_PER_PX` are the
/// source / destination element strides per pixel.
macro_rules! run_rgb96_blocks {
  ($block:ident, $tail:expr, $src:ident, $dst:ident, $width:ident, $in_pp:expr, $out_pp:expr) => {{
    let mut x = 0usize;
    while x + 32 <= $width {
      for k in 0..4 {
        let xk = x + k * 8;
        $block::<BE>(
          $src.as_ptr().add(xk * $in_pp),
          $dst.as_mut_ptr().add(xk * $out_pp),
        );
      }
      x += 32;
    }
    while x + 8 <= $width {
      $block::<BE>(
        $src.as_ptr().add(x * $in_pp),
        $dst.as_mut_ptr().add(x * $out_pp),
      );
      x += 8;
    }
    if x < $width {
      // `$tail` is the scalar reference already turbofished with `::<BE>` at
      // the call site (a `:path` metavar can't carry the turbofish here).
      $tail(&$src[x * $in_pp..], &mut $dst[x * $out_pp..], $width - x);
    }
  }};
}

// Rgb96 (R, G, B — 3 u32 elements per pixel).

/// AVX-512 Rgb96 → packed u8 RGB. 32 pixels per outer iteration.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available (caller obligation).
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgb96_to_rgb_row<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  unsafe {
    run_rgb96_blocks!(
      block_rgb96_to_rgb_8px,
      scalar::rgb96_to_rgb_row::<BE>,
      rgb96,
      rgb_out,
      width,
      3,
      3
    );
  }
}

/// AVX-512 Rgb96 → packed u8 RGBA. 32 pixels per outer iteration. Alpha 0xFF.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgb96_to_rgba_row<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  unsafe {
    run_rgb96_blocks!(
      block_rgb96_to_rgba_8px,
      scalar::rgb96_to_rgba_row::<BE>,
      rgb96,
      rgba_out,
      width,
      3,
      4
    );
  }
}

/// AVX-512 Rgb96 → native-depth u16 RGB. 32 pixels per outer iteration.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgb96_to_rgb_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  unsafe {
    run_rgb96_blocks!(
      block_rgb96_to_rgb_u16_8px,
      scalar::rgb96_to_rgb_u16_row::<BE>,
      rgb96,
      rgb_out,
      width,
      3,
      3
    );
  }
}

/// AVX-512 Rgb96 → native-depth u16 RGBA. 32 pixels per outer iteration. Alpha 0xFFFF.
///
/// # Safety
///
/// 1. AVX-512F + AVX-512BW must be available.
/// 2. `rgb96.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn avx512_rgb96_to_rgba_u16_row<const BE: bool>(
  rgb96: &[u32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb96.len() >= width * 3, "rgb96 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  unsafe {
    run_rgb96_blocks!(
      block_rgb96_to_rgba_u16_8px,
      scalar::rgb96_to_rgba_u16_row::<BE>,
      rgb96,
      rgba_out,
      width,
      3,
      4
    );
  }
}
