//! AVX-512 (F + BW) kernels for 32-bit planar GBR + alpha sources
//! (`AV_PIX_FMT_GBRAP32{LE,BE}`).
//!
//! Lane width: 16 pixels per main iteration. Each plane is loaded as one
//! `u32x16` (`load_endian_u32x16`, byte-swapped per `BE`), narrowed `>> 16`,
//! and its four 128-bit lanes (`_mm512_extracti32x4_epi32`) are packed pairwise
//! to two `u16x8` channel vectors via `_mm_packus_epi32` (inputs in
//! `[0, 65535]`, so the unsigned-saturating pack is exact). Each `u16x8` feeds
//! the shared 128-bit `write_rgb_u16_8` / `write_rgba_u16_8` / `write_rgb_16` /
//! `write_rgba_16` interleave helpers (8 pixels each) — the same tail the
//! `Gbrap16` u16 kernels use. An 8-pixel SSE-style tail (two `u32x4` loads,
//! `load_endian_u32x4` from `x86_sse41`) handles widths that are not a multiple
//! of 16; the scalar reference handles the final remainder.

use super::{endian::load_endian_u32x16, *};
use crate::row::{arch::x86_sse41::endian::load_endian_u32x4, scalar::planar_gbr_32bit as scalar};

/// Narrow 16 pixels of one `u32` plane `>> 16` into a pair of `u16x8`
/// (`(px 0..7, px 8..15)`).
///
/// # Safety
/// AVX-512F+BW available; `ptr` points to ≥ 16 readable `u32`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn narrow_u16x8_pair<const BE: bool>(ptr: *const u32) -> (__m128i, __m128i) {
  unsafe {
    let v = _mm512_srli_epi32::<16>(load_endian_u32x16::<BE>(ptr.cast()));
    let l0 = _mm512_extracti32x4_epi32::<0>(v);
    let l1 = _mm512_extracti32x4_epi32::<1>(v);
    let l2 = _mm512_extracti32x4_epi32::<2>(v);
    let l3 = _mm512_extracti32x4_epi32::<3>(v);
    (_mm_packus_epi32(l0, l1), _mm_packus_epi32(l2, l3))
  }
}

/// Narrow 8 pixels of one `u32` plane `>> 16` into a `u16x8` (SSE-style tail).
///
/// # Safety
/// AVX-512F+BW (implies SSE4.1) available; `ptr` points to ≥ 8 readable `u32`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn narrow_u16x8<const BE: bool>(ptr: *const u32) -> __m128i {
  unsafe {
    let lo = _mm_srli_epi32::<16>(load_endian_u32x4::<BE>(ptr.cast()));
    let hi = _mm_srli_epi32::<16>(load_endian_u32x4::<BE>(ptr.add(4).cast()));
    _mm_packus_epi32(lo, hi)
  }
}

/// AVX-512 `gbr32_to_rgb_row`: drop α, `>> 24` → packed `R, G, B` u8.
///
/// # Safety
/// 1. AVX-512F+BW available. 2. `g`/`b`/`r` ≥ `width`. 3. `rgb_out` ≥
///    `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbr32_to_rgb_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  unsafe {
    let zero = _mm_setzero_si128();
    let to_u8 = |v: __m128i| _mm_packus_epi16(_mm_srli_epi16::<8>(v), zero);
    let mut x = 0usize;
    while x + 16 <= width {
      let (rl, rh) = narrow_u16x8_pair::<BE>(r.as_ptr().add(x));
      let (gl, gh) = narrow_u16x8_pair::<BE>(g.as_ptr().add(x));
      let (bl, bh) = narrow_u16x8_pair::<BE>(b.as_ptr().add(x));
      let mut tmp = [0u8; 48];
      write_rgb_16(to_u8(rl), to_u8(gl), to_u8(bl), tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      write_rgb_16(to_u8(rh), to_u8(gh), to_u8(bh), tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add((x + 8) * 3), 24);
      x += 16;
    }
    while x + 8 <= width {
      let mut tmp = [0u8; 48];
      write_rgb_16(
        to_u8(narrow_u16x8::<BE>(r.as_ptr().add(x))),
        to_u8(narrow_u16x8::<BE>(g.as_ptr().add(x))),
        to_u8(narrow_u16x8::<BE>(b.as_ptr().add(x))),
        tmp.as_mut_ptr(),
      );
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgb_out.as_mut_ptr().add(x * 3), 24);
      x += 8;
    }
    if x < width {
      scalar::gbr32_to_rgb_row::<BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// AVX-512 `gbr32_to_rgb_u16_row`: drop α, `>> 16` → packed `R, G, B` u16.
///
/// # Safety
/// 1. AVX-512F+BW available. 2. `g`/`b`/`r` ≥ `width`. 3. `rgb_u16_out` ≥
///    `3 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbr32_to_rgb_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (rl, rh) = narrow_u16x8_pair::<BE>(r.as_ptr().add(x));
      let (gl, gh) = narrow_u16x8_pair::<BE>(g.as_ptr().add(x));
      let (bl, bh) = narrow_u16x8_pair::<BE>(b.as_ptr().add(x));
      write_rgb_u16_8(rl, gl, bl, rgb_u16_out.as_mut_ptr().add(x * 3));
      write_rgb_u16_8(rh, gh, bh, rgb_u16_out.as_mut_ptr().add((x + 8) * 3));
      x += 16;
    }
    while x + 8 <= width {
      write_rgb_u16_8(
        narrow_u16x8::<BE>(r.as_ptr().add(x)),
        narrow_u16x8::<BE>(g.as_ptr().add(x)),
        narrow_u16x8::<BE>(b.as_ptr().add(x)),
        rgb_u16_out.as_mut_ptr().add(x * 3),
      );
      x += 8;
    }
    if x < width {
      scalar::gbr32_to_rgb_u16_row::<BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgb_u16_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// AVX-512 `gbra32_to_rgba_row`: `>> 24` all 4 channels → packed
/// `R, G, B, A` u8.
///
/// # Safety
/// 1. AVX-512F+BW available. 2. `g`/`b`/`r`/`a` ≥ `width`. 3. `rgba_out` ≥
///    `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbra32_to_rgba_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  unsafe {
    let zero = _mm_setzero_si128();
    let to_u8 = |v: __m128i| _mm_packus_epi16(_mm_srli_epi16::<8>(v), zero);
    let mut x = 0usize;
    while x + 16 <= width {
      let (rl, rh) = narrow_u16x8_pair::<BE>(r.as_ptr().add(x));
      let (gl, gh) = narrow_u16x8_pair::<BE>(g.as_ptr().add(x));
      let (bl, bh) = narrow_u16x8_pair::<BE>(b.as_ptr().add(x));
      let (al, ah) = narrow_u16x8_pair::<BE>(a.as_ptr().add(x));
      let mut tmp = [0u8; 64];
      write_rgba_16(to_u8(rl), to_u8(gl), to_u8(bl), to_u8(al), tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      write_rgba_16(to_u8(rh), to_u8(gh), to_u8(bh), to_u8(ah), tmp.as_mut_ptr());
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add((x + 8) * 4), 32);
      x += 16;
    }
    while x + 8 <= width {
      let mut tmp = [0u8; 64];
      write_rgba_16(
        to_u8(narrow_u16x8::<BE>(r.as_ptr().add(x))),
        to_u8(narrow_u16x8::<BE>(g.as_ptr().add(x))),
        to_u8(narrow_u16x8::<BE>(b.as_ptr().add(x))),
        to_u8(narrow_u16x8::<BE>(a.as_ptr().add(x))),
        tmp.as_mut_ptr(),
      );
      core::ptr::copy_nonoverlapping(tmp.as_ptr(), rgba_out.as_mut_ptr().add(x * 4), 32);
      x += 8;
    }
    if x < width {
      scalar::gbra32_to_rgba_row::<BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &a[x..width],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// AVX-512 `gbra32_to_rgba_u16_row`: `>> 16` all 4 channels → packed
/// `R, G, B, A` u16.
///
/// # Safety
/// 1. AVX-512F+BW available. 2. `g`/`b`/`r`/`a` ≥ `width`. 3. `rgba_u16_out` ≥
///    `4 * width`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn gbra32_to_rgba_u16_row<const BE: bool>(
  g: &[u32],
  b: &[u32],
  r: &[u32],
  a: &[u32],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let (rl, rh) = narrow_u16x8_pair::<BE>(r.as_ptr().add(x));
      let (gl, gh) = narrow_u16x8_pair::<BE>(g.as_ptr().add(x));
      let (bl, bh) = narrow_u16x8_pair::<BE>(b.as_ptr().add(x));
      let (al, ah) = narrow_u16x8_pair::<BE>(a.as_ptr().add(x));
      write_rgba_u16_8(rl, gl, bl, al, rgba_u16_out.as_mut_ptr().add(x * 4));
      write_rgba_u16_8(rh, gh, bh, ah, rgba_u16_out.as_mut_ptr().add((x + 8) * 4));
      x += 16;
    }
    while x + 8 <= width {
      write_rgba_u16_8(
        narrow_u16x8::<BE>(r.as_ptr().add(x)),
        narrow_u16x8::<BE>(g.as_ptr().add(x)),
        narrow_u16x8::<BE>(b.as_ptr().add(x)),
        narrow_u16x8::<BE>(a.as_ptr().add(x)),
        rgba_u16_out.as_mut_ptr().add(x * 4),
      );
      x += 8;
    }
    if x < width {
      scalar::gbra32_to_rgba_u16_row::<BE>(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &a[x..width],
        &mut rgba_u16_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}
