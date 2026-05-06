//! wasm-simd128 kernels for planar GBR sources (Tier 10).
//!
//! Mirrors the per-arch shape of the x86 / NEON kernels: 16 pixels per
//! iteration via three (or four) `u8x16_swizzle` shuffles per 16-byte
//! output block, OR'd together. Masks match the
//! `super::x86_common::write_rgb_16` derivation byte-for-byte.

use core::arch::wasm32::*;

use super::*;

/// Writes 16 pixels of packed `R, G, B` (48 bytes) from three u8x16
/// channel vectors, mirroring the x86 `write_rgb_16` helper.
///
/// # Safety
///
/// `ptr` must point to at least 48 writable bytes. `simd128` must be
/// enabled at compile time (caller obligation via `target_feature`).
#[inline(always)]
unsafe fn write_rgb_16(r: v128, g: v128, b: v128, ptr: *mut u8) {
  unsafe {
    // Block 0 (output bytes 0..16):
    //   R G B R G B R G B R G B R G B R
    //   sourced from R[0..6], G[0..5], B[0..5] respectively.
    let r0 = i8x16(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5);
    let g0 = i8x16(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1);
    let b0 = i8x16(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1);
    let out0 = v128_or(
      v128_or(u8x16_swizzle(r, r0), u8x16_swizzle(g, g0)),
      u8x16_swizzle(b, b0),
    );

    // Block 1 (output bytes 16..32).
    let r1 = i8x16(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1);
    let g1 = i8x16(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10);
    let b1 = i8x16(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1);
    let out1 = v128_or(
      v128_or(u8x16_swizzle(r, r1), u8x16_swizzle(g, g1)),
      u8x16_swizzle(b, b1),
    );

    // Block 2 (output bytes 32..48).
    let r2 = i8x16(
      -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1,
    );
    let g2 = i8x16(
      -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1,
    );
    let b2 = i8x16(
      10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15,
    );
    let out2 = v128_or(
      v128_or(u8x16_swizzle(r, r2), u8x16_swizzle(g, g2)),
      u8x16_swizzle(b, b2),
    );

    v128_store(ptr.cast(), out0);
    v128_store(ptr.add(16).cast(), out1);
    v128_store(ptr.add(32).cast(), out2);
  }
}

/// Writes 16 pixels of packed `R, G, B, A` (64 bytes) from four u8x16
/// channel vectors, mirroring the x86 `write_rgba_16` helper.
///
/// # Safety
///
/// `ptr` must point to at least 64 writable bytes.
#[inline(always)]
unsafe fn write_rgba_16(r: v128, g: v128, b: v128, a: v128, ptr: *mut u8) {
  unsafe {
    // Block 0 (output bytes 0..16): pixels 0..3.
    let r0 = i8x16(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);
    let g0 = i8x16(-1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1);
    let b0 = i8x16(-1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1);
    let a0 = i8x16(-1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3);
    let out0 = v128_or(
      v128_or(u8x16_swizzle(r, r0), u8x16_swizzle(g, g0)),
      v128_or(u8x16_swizzle(b, b0), u8x16_swizzle(a, a0)),
    );

    // Block 1 (output bytes 16..32): pixels 4..7.
    let r1 = i8x16(4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1, -1);
    let g1 = i8x16(-1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1, -1);
    let b1 = i8x16(-1, -1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7, -1);
    let a1 = i8x16(-1, -1, -1, 4, -1, -1, -1, 5, -1, -1, -1, 6, -1, -1, -1, 7);
    let out1 = v128_or(
      v128_or(u8x16_swizzle(r, r1), u8x16_swizzle(g, g1)),
      v128_or(u8x16_swizzle(b, b1), u8x16_swizzle(a, a1)),
    );

    // Block 2 (output bytes 32..48): pixels 8..11.
    let r2 = i8x16(8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1, -1);
    let g2 = i8x16(-1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1);
    let b2 = i8x16(-1, -1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11, -1);
    let a2 = i8x16(-1, -1, -1, 8, -1, -1, -1, 9, -1, -1, -1, 10, -1, -1, -1, 11);
    let out2 = v128_or(
      v128_or(u8x16_swizzle(r, r2), u8x16_swizzle(g, g2)),
      v128_or(u8x16_swizzle(b, b2), u8x16_swizzle(a, a2)),
    );

    // Block 3 (output bytes 48..64): pixels 12..15.
    let r3 = i8x16(
      12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1,
    );
    let g3 = i8x16(
      -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1,
    );
    let b3 = i8x16(
      -1, -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1,
    );
    let a3 = i8x16(
      -1, -1, -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15,
    );
    let out3 = v128_or(
      v128_or(u8x16_swizzle(r, r3), u8x16_swizzle(g, g3)),
      v128_or(u8x16_swizzle(b, b3), u8x16_swizzle(a, a3)),
    );

    v128_store(ptr.cast(), out0);
    v128_store(ptr.add(16).cast(), out1);
    v128_store(ptr.add(32).cast(), out2);
    v128_store(ptr.add(48).cast(), out3);
  }
}

/// wasm-simd128 G/B/R planar → packed `R, G, B`.
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `g.len()`, `b.len()`, `r.len()` ≥ `width`.
/// 3. `rgb_out.len()` ≥ `3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbr_to_rgb_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  // SAFETY: simd128 is compile-time enabled per caller obligation.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let g_v = v128_load(g.as_ptr().add(x).cast());
      let b_v = v128_load(b.as_ptr().add(x).cast());
      let r_v = v128_load(r.as_ptr().add(x).cast());
      write_rgb_16(r_v, g_v, b_v, rgb_out.as_mut_ptr().add(x * 3));
      x += 16;
    }
    if x < width {
      scalar::gbr_to_rgb_row(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgb_out[x * 3..width * 3],
        width - x,
      );
    }
  }
}

/// wasm-simd128 G/B/R/A planar → packed `R, G, B, A`.
///
/// # Safety
///
/// Same as [`gbr_to_rgb_row`] plus `a.len()` ≥ `width`,
/// `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbra_to_rgba_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  a: &[u8],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  // SAFETY: see `gbr_to_rgb_row`.
  unsafe {
    let mut x = 0usize;
    while x + 16 <= width {
      let g_v = v128_load(g.as_ptr().add(x).cast());
      let b_v = v128_load(b.as_ptr().add(x).cast());
      let r_v = v128_load(r.as_ptr().add(x).cast());
      let a_v = v128_load(a.as_ptr().add(x).cast());
      write_rgba_16(r_v, g_v, b_v, a_v, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::gbra_to_rgba_row(
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

/// wasm-simd128 G/B/R planar → packed `R, G, B, A` with constant
/// `α = 0xFF`.
///
/// # Safety
///
/// Same as [`gbr_to_rgb_row`] plus `rgba_out.len()` ≥ `4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn gbr_to_rgba_opaque_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  // SAFETY: see `gbr_to_rgb_row`.
  unsafe {
    let opaque = u8x16_splat(0xFF);
    let mut x = 0usize;
    while x + 16 <= width {
      let g_v = v128_load(g.as_ptr().add(x).cast());
      let b_v = v128_load(b.as_ptr().add(x).cast());
      let r_v = v128_load(r.as_ptr().add(x).cast());
      write_rgba_16(r_v, g_v, b_v, opaque, rgba_out.as_mut_ptr().add(x * 4));
      x += 16;
    }
    if x < width {
      scalar::gbr_to_rgba_opaque_row(
        &g[x..width],
        &b[x..width],
        &r[x..width],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}
