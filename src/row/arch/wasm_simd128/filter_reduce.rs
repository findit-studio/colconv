//! wasm-simd128 separable-filter H/V passes: per output span, a dot
//! product of **signed** `f32` coefficients over a zero-padded coefficient
//! arena.
//!
//! The signed twin of [`area_reduce`](super::area_reduce). The arena pads
//! every span's `f32` coefficients to a multiple of 8 with zeros, so the
//! hot loop runs pure wide loads (8 samples + 8 coefficients per chunk); a
//! zero coefficient annihilates its sample, so sample loads only stage
//! through a stack copy at the row-end boundary.
//!
//! Both passes accumulate in `f64` (`f64x2` lanes) — the per-lane product
//! is exact, so H parity is a small tolerance from the tap-sum order while
//! the V-pass (mul+add, **not** fma) stays bit-equal to the scalar
//! reference. Mirrors the wasm area f32 path (`widen`/`mask_pd`/`deint3`),
//! with the integer c3 gathers reused from the area u16 c3 swizzle masks.

#![cfg_attr(not(feature = "std"), allow(dead_code))]
#![cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]

use core::arch::wasm32::*;

/// Eight source samples widened to four `f64x2` pairs.
type F64x8 = (v128, v128, v128, v128);

/// One element type the filter H-pass widens to `f64x2` lanes. The c1/c3
/// kernel bodies are generic over the load; the public entry points pin it
/// to `u8` / `u16` / `f32`.
trait WasmElem: Copy + Default {
  /// Widen the 8 contiguous samples at `row[base..base + 8]`.
  ///
  /// # Safety
  ///
  /// `base + 8 <= row.len()`; simd128 enabled.
  unsafe fn load8(row: &[Self], base: usize) -> F64x8;

  /// Widen channel `ch` of the 8-pixel interleaved group at cell `cell`.
  ///
  /// # Safety
  ///
  /// `(cell + 8) * 3 <= row.len()`; `ch < 3`; simd128 enabled.
  unsafe fn load8_c3(row: &[Self], cell: usize, ch: usize) -> F64x8;
}

/// Widens an `f32x4` to two `f64x2` pairs `(lanes 0-1, 2-3)`. simd128's
/// `promote_low` widens lanes 0-1 only, so the high pair shuffles down.
#[inline]
#[target_feature(enable = "simd128")]
fn widen_f32x4(s: v128) -> (v128, v128) {
  (
    f64x2_promote_low_f32x4(s),
    f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(s, s)),
  )
}

#[inline]
#[target_feature(enable = "simd128")]
fn widen_f32x8(lo: v128, hi: v128) -> F64x8 {
  let (a, b) = widen_f32x4(lo);
  let (c, d) = widen_f32x4(hi);
  (a, b, c, d)
}

/// Widens 8 `u16` lanes to four `f64x2` pairs.
#[inline]
#[target_feature(enable = "simd128")]
fn widen_u16x8(s16: v128) -> F64x8 {
  let lo = f32x4_convert_i32x4(i32x4_extend_low_u16x8(s16));
  let hi = f32x4_convert_i32x4(i32x4_extend_high_u16x8(s16));
  widen_f32x8(lo, hi)
}

impl WasmElem for u8 {
  #[inline]
  #[target_feature(enable = "simd128")]
  unsafe fn load8(row: &[u8], base: usize) -> F64x8 {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe { widen_u16x8(i16x8_load_extend_u8x8(row.as_ptr().add(base))) }
  }
  #[inline]
  #[target_feature(enable = "simd128")]
  unsafe fn load8_c3(row: &[u8], cell: usize, ch: usize) -> F64x8 {
    // SAFETY: `(cell + 8) * 3 <= row.len()`; two 16-byte loads cover the
    // 24-byte group, the per-channel swizzle gathers 8 u8.
    unsafe {
      let base = cell * 3;
      let v0 = v128_load(row.as_ptr().add(base).cast());
      let v1 = v128_load(row.as_ptr().add(base + 8).cast());
      widen_u16x8(u16x8_extend_low_u8x16(gather_u8_c3(v0, v1, ch)))
    }
  }
}

impl WasmElem for u16 {
  #[inline]
  #[target_feature(enable = "simd128")]
  unsafe fn load8(row: &[u16], base: usize) -> F64x8 {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe { widen_u16x8(v128_load(row.as_ptr().add(base).cast())) }
  }
  #[inline]
  #[target_feature(enable = "simd128")]
  unsafe fn load8_c3(row: &[u16], cell: usize, ch: usize) -> F64x8 {
    // SAFETY: `(cell + 8) * 3 <= row.len()`; three 16-byte loads cover the
    // 48-byte group, the per-channel swizzle triple gathers 8 u16.
    unsafe {
      let base = cell * 3;
      let v0 = v128_load(row.as_ptr().add(base).cast());
      let v1 = v128_load(row.as_ptr().add(base + 8).cast());
      let v2 = v128_load(row.as_ptr().add(base + 16).cast());
      widen_u16x8(gather_u16_c3(v0, v1, v2, ch))
    }
  }
}

impl WasmElem for f32 {
  #[inline]
  #[target_feature(enable = "simd128")]
  unsafe fn load8(row: &[f32], base: usize) -> F64x8 {
    // SAFETY: `base + 8 <= row.len()`.
    unsafe {
      widen_f32x8(
        v128_load(row.as_ptr().add(base).cast()),
        v128_load(row.as_ptr().add(base + 4).cast()),
      )
    }
  }
  #[inline]
  #[target_feature(enable = "simd128")]
  unsafe fn load8_c3(row: &[f32], cell: usize, ch: usize) -> F64x8 {
    // SAFETY: `(cell + 8) * 3 <= row.len()`; two four-pixel deinterleaves
    // cover the group, `.ch` selects the channel.
    unsafe {
      let p = row.as_ptr().add(cell * 3);
      let (r0, g0, b0) = deint3_f32(
        v128_load(p.cast()),
        v128_load(p.add(4).cast()),
        v128_load(p.add(8).cast()),
      );
      let (r1, g1, b1) = deint3_f32(
        v128_load(p.add(12).cast()),
        v128_load(p.add(16).cast()),
        v128_load(p.add(20).cast()),
      );
      let (lo, hi) = match ch {
        0 => (r0, r1),
        1 => (g0, g1),
        _ => (b0, b1),
      };
      widen_f32x8(lo, hi)
    }
  }
}

/// Gathers channel `ch`'s 8 `u8` samples of a 24-byte RGB group split
/// across two overlapping 16-byte loads (area u8 c3 shuffle indices: byte
/// g of the chunk is lane g of v0 for g < 16, lane g+8 across the shuffle
/// boundary for g >= 16).
#[inline]
#[target_feature(enable = "simd128")]
fn gather_u8_c3(v0: v128, v1: v128, ch: usize) -> v128 {
  match ch {
    0 => i8x16_shuffle::<0, 3, 6, 9, 12, 15, 26, 29, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1),
    1 => i8x16_shuffle::<1, 4, 7, 10, 13, 24, 27, 30, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1),
    _ => i8x16_shuffle::<2, 5, 8, 11, 14, 25, 28, 31, 0, 0, 0, 0, 0, 0, 0, 0>(v0, v1),
  }
}

/// Gathers channel `ch`'s 8 `u16` samples of a 48-byte RGB group split
/// across three overlapping 16-byte loads (area u16 c3 swizzle masks; an
/// index `>= 16` zeroes the lane, so the per-channel OR reassembles the 8
/// samples in order).
#[inline]
#[target_feature(enable = "simd128")]
fn gather_u16_c3(v0: v128, v1: v128, v2: v128, ch: usize) -> v128 {
  const M0: [v128; 3] = [
    u8x16(
      0, 1, 6, 7, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ),
    u8x16(
      2, 3, 8, 9, 14, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ),
    u8x16(
      4, 5, 10, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ),
  ];
  const M1: [v128; 3] = [
    u8x16(
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 2, 3, 8, 9, 14, 15, 0x80, 0x80, 0x80, 0x80,
    ),
    u8x16(
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 4, 5, 10, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ),
    u8x16(
      0x80, 0x80, 0x80, 0x80, 0, 1, 6, 7, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ),
  ];
  const M2: [v128; 3] = [
    u8x16(
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 4, 5, 10, 11,
    ),
    u8x16(
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 6, 7, 12, 13,
    ),
    u8x16(
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 2, 3, 8, 9, 14, 15,
    ),
  ];
  v128_or(
    v128_or(i8x16_swizzle(v0, M0[ch]), i8x16_swizzle(v1, M1[ch])),
    i8x16_swizzle(v2, M2[ch]),
  )
}

/// Deinterleaves four interleaved RGB `f32` pixels into planar
/// `(R0..R3, G0..G3, B0..B3)` (area wasm `deint3_f32`).
#[inline]
#[target_feature(enable = "simd128")]
fn deint3_f32(x: v128, y: v128, z: v128) -> (v128, v128, v128) {
  let rx = i32x4_shuffle::<0, 3, 0, 3>(x, x);
  let ryz = i32x4_shuffle::<2, 5, 2, 5>(y, z);
  let r = i32x4_shuffle::<0, 1, 4, 5>(rx, ryz);
  let gx = i32x4_shuffle::<1, 4, 1, 4>(x, y);
  let gyz = i32x4_shuffle::<3, 6, 3, 6>(y, z);
  let g = i32x4_shuffle::<0, 1, 4, 5>(gx, gyz);
  let bx = i32x4_shuffle::<2, 5, 2, 5>(x, y);
  let bz = i32x4_shuffle::<0, 3, 0, 3>(z, z);
  let b = i32x4_shuffle::<0, 1, 4, 5>(bx, bz);
  (r, g, b)
}

/// Widens 8 signed `f32` coefficients to four `f64x2` pairs.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn widen_coeffs(c: &[f32]) -> F64x8 {
  // SAFETY: caller passes an 8-multiple arena chunk.
  unsafe {
    widen_f32x8(
      v128_load(c.as_ptr().cast()),
      v128_load(c.as_ptr().add(4).cast()),
    )
  }
}

/// Zeroes the `f64x2` sample lanes whose coefficient lane is zero (arena
/// padding) — `0.0 * NaN` would otherwise poison the span.
#[inline]
#[target_feature(enable = "simd128")]
fn mask_pd(sf: v128, cf: v128) -> v128 {
  v128_and(sf, f64x2_ne(cf, f64x2_splat(0.0)))
}

/// Accumulates 8 widened samples against 4 widened coefficient pairs into
/// a running `f64x2` (mul+add; the product is exact in `f64`).
#[inline]
#[target_feature(enable = "simd128")]
fn mac8(acc: v128, s: F64x8, c: F64x8) -> v128 {
  let a = f64x2_add(acc, f64x2_mul(mask_pd(s.0, c.0), c.0));
  let a = f64x2_add(a, f64x2_mul(mask_pd(s.1, c.1), c.1));
  let a = f64x2_add(a, f64x2_mul(mask_pd(s.2, c.2), c.2));
  f64x2_add(a, f64x2_mul(mask_pd(s.3, c.3), c.3))
}

/// Sums the two `f64x2` lanes.
#[inline]
#[target_feature(enable = "simd128")]
fn hsum_pd(v: v128) -> f64 {
  f64x2_extract_lane::<0>(v) + f64x2_extract_lane::<1>(v)
}

/// Loads + widens 8 contiguous samples at cell `base`, staging the row
/// end.
///
/// # Safety
///
/// `base < cells`; simd128 enabled; `row.len() >= cells`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn load8_staged_c1<S: WasmElem>(row: &[S], base: usize) -> F64x8 {
  // SAFETY: a full chunk loads directly; the row end stages a zero-filled
  // 8-element copy.
  unsafe {
    if base + 8 <= row.len() {
      S::load8(row, base)
    } else {
      let mut sbuf = [S::default(); 8];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      S::load8(&sbuf, 0)
    }
  }
}

/// Loads + widens channel `ch` of the group at cell `cell`, staging the
/// row end.
///
/// # Safety
///
/// `cell < cells`; `ch < 3`; simd128 enabled; `row.len() >= cells * 3`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn load8_staged_c3<S: WasmElem>(row: &[S], cell: usize, ch: usize) -> F64x8 {
  // SAFETY: a full group loads directly; the row end stages its 24
  // interleaved samples.
  unsafe {
    if (cell + 8) * 3 <= row.len() {
      S::load8_c3(row, cell, ch)
    } else {
      let mut sbuf = [S::default(); 24];
      let take = row.len() - cell * 3;
      sbuf[..take].copy_from_slice(&row[cell * 3..]);
      S::load8_c3(&sbuf, 0, ch)
    }
  }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn h_reduce_c1<S: WasmElem>(
  row: &[S],
  starts: &[usize],
  coeffs: &[f32],
  coff: &[usize],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &coeffs[coff[j]..coff[j + 1]];
    // SAFETY: each chunk loads in-bounds or stages the row end; coeffs
    // from the 8-multiple arena slice.
    unsafe {
      let mut acc = f64x2_splat(0.0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        acc = mac8(
          acc,
          load8_staged_c1(row, start + ci * 8),
          widen_coeffs(chunk),
        );
      }
      h_tmp[j] = hsum_pd(acc);
    }
  }
}

#[inline]
#[target_feature(enable = "simd128")]
unsafe fn h_reduce_c3<S: WasmElem>(
  row: &[S],
  starts: &[usize],
  coeffs: &[f32],
  coff: &[usize],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &coeffs[coff[j]..coff[j + 1]];
    // SAFETY: each group loads in-bounds or stages the row end; coeffs
    // from the 8-multiple arena slice.
    unsafe {
      let mut acc0 = f64x2_splat(0.0);
      let mut acc1 = f64x2_splat(0.0);
      let mut acc2 = f64x2_splat(0.0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let cell = start + ci * 8;
        let c = widen_coeffs(chunk);
        acc0 = mac8(acc0, load8_staged_c3(row, cell, 0), c);
        acc1 = mac8(acc1, load8_staged_c3(row, cell, 1), c);
        acc2 = mac8(acc2, load8_staged_c3(row, cell, 2), c);
      }
      h_tmp[j * 3] = hsum_pd(acc0);
      h_tmp[j * 3 + 1] = hsum_pd(acc1);
      h_tmp[j * 3 + 2] = hsum_pd(acc2);
    }
  }
}

// ---- Concrete per-element entry points (the dispatcher's targets) -----

macro_rules! wasm_h_entry {
  ($c1:ident, $c3:ident, $elem:ty, $doc:literal) => {
    #[doc = $doc]
    ///
    /// # Safety
    ///
    /// simd128 enabled; the arena binds to this row (see [`h_reduce_c1`]).
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn $c1(
      row: &[$elem],
      starts: &[usize],
      coeffs: &[f32],
      coff: &[usize],
      h_tmp: &mut [f64],
    ) {
      // SAFETY: forwarded under the caller's arena guarantees.
      unsafe { h_reduce_c1::<$elem>(row, starts, coeffs, coff, h_tmp) }
    }

    #[doc = $doc]
    ///
    /// # Safety
    ///
    /// simd128 enabled; the arena binds to this row (see [`h_reduce_c3`]).
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn $c3(
      row: &[$elem],
      starts: &[usize],
      coeffs: &[f32],
      coff: &[usize],
      h_tmp: &mut [f64],
    ) {
      // SAFETY: forwarded under the caller's arena guarantees.
      unsafe { h_reduce_c3::<$elem>(row, starts, coeffs, coff, h_tmp) }
    }
  };
}

wasm_h_entry!(
  filter_h_reduce_row_u8_c1,
  filter_h_reduce_row_u8_c3,
  u8,
  "Filter H-pass over `u8` samples (1 / 3 channel)."
);
wasm_h_entry!(
  filter_h_reduce_row_u16_c1,
  filter_h_reduce_row_u16_c3,
  u16,
  "Filter H-pass over `u16` samples (1 / 3 channel)."
);
wasm_h_entry!(
  filter_h_reduce_row_f32_c1,
  filter_h_reduce_row_f32_c3,
  f32,
  "Filter H-pass over `f32` samples (1 / 3 channel)."
);

/// Filter V-pass AXPY: `acc[i] += w * h_tmp[i]` in `f64` (mul+add, **not**
/// fma) so each lane matches the scalar reference bit-for-bit. Two
/// elements per iteration.
///
/// # Safety
///
/// simd128 enabled. `h_tmp.len() >= acc.len()`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn filter_v_accumulate(acc: &mut [f64], h_tmp: &[f64], w: f32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = f64x2_splat(f64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 2 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 2 <= n {
      let t = v128_load(h_tmp.as_ptr().add(i).cast());
      let a = v128_load(acc.as_ptr().add(i).cast());
      v128_store(
        acc.as_mut_ptr().add(i).cast(),
        f64x2_add(a, f64x2_mul(t, wv)),
      );
      i += 2;
    }
  }
  for k in i..n {
    acc[k] += f64::from(w) * h_tmp[k];
  }
}
