//! AVX2 kernels for 32-bit planar GBR + alpha sources
//! (`AV_PIX_FMT_GBRAP32{LE,BE}`).
//!
//! Lane width: 8 pixels per iteration. Each plane is loaded as one `u32x8`
//! (`load_endian_u32x8`, byte-swapped per `BE`), narrowed `>> 16`, and the two
//! 128-bit halves are packed to a single `u16x8` channel vector via
//! `_mm_packus_epi32` (inputs in `[0, 65535]`, so the unsigned-saturating pack
//! is exact). The narrowed channel vectors feed the shared 128-bit
//! `write_rgb_u16_8` / `write_rgba_u16_8` / `write_rgb_16` / `write_rgba_16`
//! interleave helpers — the same tail the `Gbrap16` u16 kernels use. For u8
//! outputs the `u16x8` is shifted `>> 8` (net `>> 24`) and packed to `u8`.
//! Scalar tails handle the remainder.

use super::{endian::load_endian_u32x8, *};
use crate::row::scalar::planar_gbr_32bit as scalar;

/// Load 8 pixels of one `u32` plane, narrow `>> 16`, pack to a `u16x8`.
///
/// # Safety
/// AVX2 available; `ptr` points to ≥ 8 readable `u32`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn narrow_u16x8<const BE: bool>(ptr: *const u32) -> __m128i {
  unsafe {
    let v = _mm256_srli_epi32::<16>(load_endian_u32x8::<BE>(ptr.cast()));
    _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
  }
}

/// AVX2 `gbr32_to_rgb_row`: drop α, `>> 24` → packed `R, G, B` u8.
///
/// # Safety
/// 1. AVX2 available. 2. `g`/`b`/`r` ≥ `width`. 3. `rgb_out` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let mut x = 0usize;
    while x + 8 <= width {
      let r_u8 = _mm_packus_epi16(
        _mm_srli_epi16::<8>(narrow_u16x8::<BE>(r.as_ptr().add(x))),
        zero,
      );
      let g_u8 = _mm_packus_epi16(
        _mm_srli_epi16::<8>(narrow_u16x8::<BE>(g.as_ptr().add(x))),
        zero,
      );
      let b_u8 = _mm_packus_epi16(
        _mm_srli_epi16::<8>(narrow_u16x8::<BE>(b.as_ptr().add(x))),
        zero,
      );
      let mut tmp = [0u8; 48];
      write_rgb_16(r_u8, g_u8, b_u8, tmp.as_mut_ptr());
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

/// AVX2 `gbr32_to_rgb_u16_row`: drop α, `>> 16` → packed `R, G, B` u16.
///
/// # Safety
/// 1. AVX2 available. 2. `g`/`b`/`r` ≥ `width`. 3. `rgb_u16_out` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 8 <= width {
      let r_v = narrow_u16x8::<BE>(r.as_ptr().add(x));
      let g_v = narrow_u16x8::<BE>(g.as_ptr().add(x));
      let b_v = narrow_u16x8::<BE>(b.as_ptr().add(x));
      write_rgb_u16_8(r_v, g_v, b_v, rgb_u16_out.as_mut_ptr().add(x * 3));
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

/// AVX2 `gbra32_to_rgba_row`: `>> 24` all 4 channels → packed `R, G, B, A` u8.
///
/// # Safety
/// 1. AVX2 available. 2. `g`/`b`/`r`/`a` ≥ `width`. 3. `rgba_out` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let mut x = 0usize;
    while x + 8 <= width {
      let r_u8 = _mm_packus_epi16(
        _mm_srli_epi16::<8>(narrow_u16x8::<BE>(r.as_ptr().add(x))),
        zero,
      );
      let g_u8 = _mm_packus_epi16(
        _mm_srli_epi16::<8>(narrow_u16x8::<BE>(g.as_ptr().add(x))),
        zero,
      );
      let b_u8 = _mm_packus_epi16(
        _mm_srli_epi16::<8>(narrow_u16x8::<BE>(b.as_ptr().add(x))),
        zero,
      );
      let a_u8 = _mm_packus_epi16(
        _mm_srli_epi16::<8>(narrow_u16x8::<BE>(a.as_ptr().add(x))),
        zero,
      );
      let mut tmp = [0u8; 64];
      write_rgba_16(r_u8, g_u8, b_u8, a_u8, tmp.as_mut_ptr());
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

/// AVX2 `gbra32_to_rgba_u16_row`: `>> 16` all 4 channels → packed
/// `R, G, B, A` u16.
///
/// # Safety
/// 1. AVX2 available. 2. `g`/`b`/`r`/`a` ≥ `width`. 3. `rgba_u16_out` ≥
///    `4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    while x + 8 <= width {
      let r_v = narrow_u16x8::<BE>(r.as_ptr().add(x));
      let g_v = narrow_u16x8::<BE>(g.as_ptr().add(x));
      let b_v = narrow_u16x8::<BE>(b.as_ptr().add(x));
      let a_v = narrow_u16x8::<BE>(a.as_ptr().add(x));
      write_rgba_u16_8(r_v, g_v, b_v, a_v, rgba_u16_out.as_mut_ptr().add(x * 4));
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
