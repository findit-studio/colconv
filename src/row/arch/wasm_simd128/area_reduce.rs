//! wasm-simd128 fused-downscale H-pass: per output span, widening
//! multiply-accumulate over the plan-time zero-padded u16 weight
//! arena (see the NEON sibling for the arena contract).
//!
//! Per 8-tap chunk: 8 samples zero-extend through
//! `i16x8_load_extend_u8x8` and meet 8 arena weights in
//! `u32x4_extmul_low/high_u16x8` exact u32 lanes. Padding lanes
//! multiply by zero, so sample loads only stage through a stack copy
//! at the row-end boundary. 3-channel rows deinterleave with
//! two-source `i8x16_shuffle` over two overlapping 16-byte loads.
//! Bit-identical to the scalar reference by integer associativity.

#![cfg_attr(not(feature = "std"), allow(dead_code))]
#![cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]

use core::arch::wasm32::*;

/// Sums the four u32 lanes of `acc`.
#[inline]
#[target_feature(enable = "simd128")]
fn hsum_u32(acc: v128) -> u32 {
  (i32x4_extract_lane::<0>(acc) as u32)
    .wrapping_add(i32x4_extract_lane::<1>(acc) as u32)
    .wrapping_add(i32x4_extract_lane::<2>(acc) as u32)
    .wrapping_add(i32x4_extract_lane::<3>(acc) as u32)
}

/// Accumulates the eight exact u32 products of `s16 * w` into `acc`.
#[inline]
#[target_feature(enable = "simd128")]
fn mac_u16x8(acc: v128, s16: v128, w: v128) -> v128 {
  let acc = i32x4_add(acc, u32x4_extmul_low_u16x8(s16, w));
  i32x4_add(acc, u32x4_extmul_high_u16x8(s16, w))
}

/// # Safety
///
/// simd128 must be enabled at compile time. Caller guarantees the
/// padded-arena contract of the NEON sibling: `w16_off.len() ==
/// starts.len() + 1` monotonic and bounded by `w16.len()`, span
/// lengths multiples of 8, `row.len() >= cells`, `h_tmp.len() >=
/// starts.len()`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn area_h_reduce_row_c1(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 8-byte sample load is either fully in-bounds
    // (guarded against row.len()) or staged through a zero-filled
    // stack copy; weight loads come from the 8-multiple arena slice.
    unsafe {
      let mut acc = u32x4_splat(0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let s16 = if base + 8 <= row.len() {
          i16x8_load_extend_u8x8(row.as_ptr().add(base))
        } else {
          let mut sbuf = [0u8; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          i16x8_load_extend_u8x8(sbuf.as_ptr())
        };
        let w = v128_load(chunk.as_ptr().cast());
        acc = mac_u16x8(acc, s16, w);
      }
      h_tmp[j] = hsum_u32(acc);
    }
  }
}

/// 3-channel (interleaved RGB) variant: two overlapping 16-byte loads
/// cover each chunk's 24 bytes, and per-channel two-source
/// `i8x16_shuffle` indices gather the eight samples of each channel
/// (byte g of the chunk is lane g of the first load for g < 16, lane
/// g + 8 across the shuffle boundary for g >= 16).
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells * 3` and
/// `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn area_h_reduce_row_c3(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: the two 16-byte loads cover bytes 0..24 of the chunk and
    // are either fully in-bounds (guarded against row.len()) or staged
    // through a zero-filled 24-byte stack copy; weight loads come from
    // the 8-multiple arena slice.
    unsafe {
      let mut acc0 = u32x4_splat(0);
      let mut acc1 = u32x4_splat(0);
      let mut acc2 = u32x4_splat(0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = (start + ci * 8) * 3;
        let mut sbuf = [0u8; 24];
        let (v0, v1) = if base + 24 <= row.len() {
          (
            v128_load(row.as_ptr().add(base).cast()),
            v128_load(row.as_ptr().add(base + 8).cast()),
          )
        } else {
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          (
            v128_load(sbuf.as_ptr().cast()),
            v128_load(sbuf.as_ptr().add(8).cast()),
          )
        };
        let w = v128_load(chunk.as_ptr().cast());
        let r = u16x8_extend_low_u8x16(i8x16_shuffle::<
          0,
          3,
          6,
          9,
          12,
          15,
          26,
          29,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
        >(v0, v1));
        let g = u16x8_extend_low_u8x16(i8x16_shuffle::<
          1,
          4,
          7,
          10,
          13,
          24,
          27,
          30,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
        >(v0, v1));
        let b = u16x8_extend_low_u8x16(i8x16_shuffle::<
          2,
          5,
          8,
          11,
          14,
          25,
          28,
          31,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
        >(v0, v1));
        acc0 = mac_u16x8(acc0, r, w);
        acc1 = mac_u16x8(acc1, g, w);
        acc2 = mac_u16x8(acc2, b, w);
      }
      h_tmp[j * 3] = hsum_u32(acc0);
      h_tmp[j * 3 + 1] = hsum_u32(acc1);
      h_tmp[j * 3 + 2] = hsum_u32(acc2);
    }
  }
}

/// V-pass AXPY: `acc[i] += w * h_tmp[i]`, exact u64 lanes via
/// `u64x2_extmul_low/high_u32x4` (4 elements per iteration).
///
/// # Safety
///
/// simd128 must be enabled at compile time.
/// `h_tmp.len() >= acc.len()`; every product-sum stays within u64
/// (the engine's denominator bound).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn area_v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wq = u32x4_splat(w);
  let mut i = 0usize;
  // SAFETY: loop guard `i + 4 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 4 <= n {
      let t = v128_load(h_tmp.as_ptr().add(i).cast());
      let a01 = v128_load(acc.as_ptr().add(i).cast());
      let a23 = v128_load(acc.as_ptr().add(i + 2).cast());
      let p01 = u64x2_extmul_low_u32x4(t, wq);
      let p23 = u64x2_extmul_high_u32x4(t, wq);
      v128_store(acc.as_mut_ptr().add(i).cast(), i64x2_add(a01, p01));
      v128_store(acc.as_mut_ptr().add(i + 2).cast(), i64x2_add(a23, p23));
      i += 4;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * u64::from(h_tmp[k]);
  }
}

/// Sums the two u64 lanes of `acc`.
#[inline]
#[target_feature(enable = "simd128")]
fn hsum_u64(acc: v128) -> u64 {
  u64x2_extract_lane::<0>(acc).wrapping_add(u64x2_extract_lane::<1>(acc))
}

/// Accumulates the eight exact `u16 * u16 -> u32` products of `s16 * w`
/// into the two u64 lanes of `acc`. Unlike the u8 [`mac_u16x8`], a `u16`
/// span sum overflows `u32`, so each exact u32 product (from the
/// widening `u32x4_extmul_low/high_u16x8`) widens to u64 via
/// `u64x2_extend_low/high_u32x4` before accumulating.
#[inline]
#[target_feature(enable = "simd128")]
fn mac_u16x8_u64(acc: v128, s16: v128, w: v128) -> v128 {
  let p_lo = u32x4_extmul_low_u16x8(s16, w);
  let p_hi = u32x4_extmul_high_u16x8(s16, w);
  let acc = i64x2_add(acc, u64x2_extend_low_u32x4(p_lo));
  let acc = i64x2_add(acc, u64x2_extend_high_u32x4(p_lo));
  let acc = i64x2_add(acc, u64x2_extend_low_u32x4(p_hi));
  i64x2_add(acc, u64x2_extend_high_u32x4(p_hi))
}

/// 16-bit-element H-pass (1 channel): like [`area_h_reduce_row_c1`] but
/// the samples load directly as `u16` and the products accumulate in
/// `u64` lanes (a single span sum can exceed `u32`).
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells` `u16`
/// elements and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn area_h_reduce_row_u16_c1(
  row: &[u16],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 8-element u16 load is fully in-bounds (guarded) or
    // staged through a zero-filled stack copy; weights come from the
    // 8-multiple arena slice.
    unsafe {
      let mut acc = u64x2_splat(0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let s16 = if base + 8 <= row.len() {
          v128_load(row.as_ptr().add(base).cast())
        } else {
          let mut sbuf = [0u16; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          v128_load(sbuf.as_ptr().cast())
        };
        let w = v128_load(chunk.as_ptr().cast());
        acc = mac_u16x8_u64(acc, s16, w);
      }
      h_tmp[j] = hsum_u64(acc);
    }
  }
}

/// 16-bit-element H-pass (3-channel interleaved RGB): each 8-tap chunk
/// spans 24 `u16` (48 bytes), so three overlapping 16-byte loads cover
/// it and a per-channel triple of `i8x16_swizzle` masks gathers each
/// channel's eight samples (a `u16` lives in exactly one load, the other
/// two masks zeroing that lane with an out-of-range index).
///
/// # Safety
///
/// As [`area_h_reduce_row_u16_c1`], with `row.len() >= cells * 3` `u16`
/// elements and `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn area_h_reduce_row_u16_c3(
  row: &[u16],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u64],
) {
  // For channel ch the eight samples sit at u16 index ch + 3t (byte
  // 2*(ch + 3t)), split across the three 16-byte loads. Each swizzle
  // pulls its load's contributing u16 pairs into output lanes 0..7; an
  // index >= 16 (here 0x80) zeroes the rest, so the per-channel OR of
  // the three swizzles reassembles the eight samples in order.
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
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: the three 16-byte loads cover bytes 0..48 of the chunk and
    // are either fully in-bounds (guarded against row.len()) or staged
    // through a zero-filled 48-byte stack copy; weight loads come from
    // the 8-multiple arena slice.
    unsafe {
      let mut acc = [u64x2_splat(0); 3];
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = (start + ci * 8) * 3;
        let mut sbuf = [0u16; 24];
        let (v0, v1, v2) = if base + 24 <= row.len() {
          (
            v128_load(row.as_ptr().add(base).cast()),
            v128_load(row.as_ptr().add(base + 8).cast()),
            v128_load(row.as_ptr().add(base + 16).cast()),
          )
        } else {
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          (
            v128_load(sbuf.as_ptr().cast()),
            v128_load(sbuf.as_ptr().add(8).cast()),
            v128_load(sbuf.as_ptr().add(16).cast()),
          )
        };
        let w = v128_load(chunk.as_ptr().cast());
        for ch in 0..3 {
          let gathered = v128_or(
            v128_or(i8x16_swizzle(v0, M0[ch]), i8x16_swizzle(v1, M1[ch])),
            i8x16_swizzle(v2, M2[ch]),
          );
          acc[ch] = mac_u16x8_u64(acc[ch], gathered, w);
        }
      }
      for ch in 0..3 {
        h_tmp[j * 3 + ch] = hsum_u64(acc[ch]);
      }
    }
  }
}

/// 16-bit-element V-pass AXPY: `acc[i] += w * h_tmp[i]` with `h_tmp`
/// already `u64`. The `u32 * u64 -> u64` product splits each `h_tmp`
/// lane into 32-bit halves — `i8x16_shuffle` packs both lanes' low
/// halves (and, separately, high halves) into the low two u32 lanes, so
/// `u64x2_extmul_low_u32x4` against `w` gives `w * lo` and `w * hi`, the
/// latter shifted up 32 — summed mod 2^64 (exact by the engine bound).
/// Two elements per iteration.
///
/// # Safety
///
/// simd128 must be enabled at compile time. `h_tmp.len() >= acc.len()`;
/// every product-sum stays within u64.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn area_v_accumulate_u16(acc: &mut [u64], h_tmp: &[u64], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wq = u32x4_splat(w);
  let mut i = 0usize;
  // SAFETY: loop guard `i + 2 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 2 <= n {
      let t = v128_load(h_tmp.as_ptr().add(i).cast());
      // Low halves of both u64 lanes -> u32 lanes 0,1; high halves ->
      // u32 lanes 0,1 of a second vector. `u64x2_extmul_low_u32x4`
      // consumes exactly those low two lanes.
      let t_lo = i8x16_shuffle::<0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27>(t, t);
      let t_hi = i8x16_shuffle::<4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31>(t, t);
      let prod_lo = u64x2_extmul_low_u32x4(t_lo, wq);
      let prod_hi = u64x2_extmul_low_u32x4(t_hi, wq);
      let prod = i64x2_add(prod_lo, u64x2_shl(prod_hi, 32));
      let a = v128_load(acc.as_ptr().add(i).cast());
      v128_store(acc.as_mut_ptr().add(i).cast(), i64x2_add(a, prod));
      i += 2;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * h_tmp[k];
  }
}

/// Sums the two f64 lanes of `v`.
#[inline]
#[target_feature(enable = "simd128")]
fn hsum_pd(v: v128) -> f64 {
  f64x2_extract_lane::<0>(v) + f64x2_extract_lane::<1>(v)
}

/// Widens eight `u16` arena weights to four `f64` lane-pairs
/// `(w0w1, w2w3, w4w5, w6w7)`.
#[target_feature(enable = "simd128")]
fn widen_w16_f64(w: v128) -> (v128, v128, v128, v128) {
  let w_lo = i32x4_extend_low_u16x8(w);
  let w_hi = i32x4_extend_high_u16x8(w);
  (
    f64x2_convert_low_i32x4(w_lo),
    f64x2_convert_low_i32x4(i32x4_shuffle::<2, 3, 2, 3>(w_lo, w_lo)),
    f64x2_convert_low_i32x4(w_hi),
    f64x2_convert_low_i32x4(i32x4_shuffle::<2, 3, 2, 3>(w_hi, w_hi)),
  )
}

/// Accumulates eight `f32` samples (`s_lo` lanes 0-3, `s_hi` lanes 4-7)
/// against four widened weight pairs. A separate multiply then add —
/// since the integer-weight times f32-sample product is exact in f64,
/// this matches a fused multiply-add anyway.
#[target_feature(enable = "simd128")]
fn mac8_f32(acc: v128, s_lo: v128, s_hi: v128, wf0: v128, wf1: v128, wf2: v128, wf3: v128) -> v128 {
  // simd128's `promote_low` widens only lanes 0-1, so the high pair of
  // each f32x4 shuffles down before promoting.
  let s1 = mask_pd(
    f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(s_lo, s_lo)),
    wf1,
  );
  let s3 = mask_pd(
    f64x2_promote_low_f32x4(i32x4_shuffle::<2, 3, 2, 3>(s_hi, s_hi)),
    wf3,
  );
  let a = f64x2_add(
    acc,
    f64x2_mul(mask_pd(f64x2_promote_low_f32x4(s_lo), wf0), wf0),
  );
  let a = f64x2_add(a, f64x2_mul(s1, wf1));
  let a = f64x2_add(
    a,
    f64x2_mul(mask_pd(f64x2_promote_low_f32x4(s_hi), wf2), wf2),
  );
  f64x2_add(a, f64x2_mul(s3, wf3))
}

/// Zeroes the `f64` sample lanes whose weight lane is zero — the arena's
/// padding lanes. The integer kernels lean on `0 * sample == 0`, but
/// `0.0 * NaN` and `0.0 * inf` are NaN, so a direct-loaded padding lane
/// holding a non-finite neighbor would otherwise poison the span.
#[target_feature(enable = "simd128")]
fn mask_pd(sf: v128, wf: v128) -> v128 {
  v128_and(sf, f64x2_ne(wf, f64x2_splat(0.0)))
}

/// Deinterleaves four interleaved RGB `f32` pixels (`x = R0 G0 B0 R1`,
/// `y = G1 B1 R2 G2`, `z = B2 R3 G3 B3`) into planar
/// `(R0..R3, G0..G3, B0..B3)` via two-source `i32x4_shuffle` (lanes
/// 0..3 select from the first source, 4..7 from the second).
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

/// Float-element H-pass (1 channel): `f32` samples widen to `f64` and
/// meet the `u16` weights widened to `f64`; the per-span sums live in
/// `f64`. The products are exact, so the only departure from the scalar
/// reference is the tap-sum order — float addition does not associate,
/// so parity is a small tolerance, not bit-exactness.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells` `f32`
/// elements and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn area_h_reduce_row_f32_c1(
  row: &[f32],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 8-element f32 load is fully in-bounds (guarded) or
    // staged through a zero-filled stack copy; weights come from the
    // 8-multiple arena slice.
    unsafe {
      let mut acc = f64x2_splat(0.0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let (s_lo, s_hi) = if base + 8 <= row.len() {
          (
            v128_load(row.as_ptr().add(base).cast()),
            v128_load(row.as_ptr().add(base + 4).cast()),
          )
        } else {
          let mut sbuf = [0f32; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          (
            v128_load(sbuf.as_ptr().cast()),
            v128_load(sbuf.as_ptr().add(4).cast()),
          )
        };
        let w = v128_load(chunk.as_ptr().cast());
        let (wf0, wf1, wf2, wf3) = widen_w16_f64(w);
        acc = mac8_f32(acc, s_lo, s_hi, wf0, wf1, wf2, wf3);
      }
      h_tmp[j] = hsum_pd(acc);
    }
  }
}

/// Float-element H-pass (3-channel interleaved RGB): two overlapping
/// four-pixel `i32x4_shuffle` deinterleaves cover the eight-pixel chunk,
/// each channel sharing one widened weight set.
///
/// # Safety
///
/// As [`area_h_reduce_row_f32_c1`], with `row.len() >= cells * 3` `f32`
/// elements and `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn area_h_reduce_row_f32_c3(
  row: &[f32],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: the six 16-byte loads cover the chunk's 24 f32 and are
    // either fully in-bounds (guarded) or staged through a zero-filled
    // 24-element stack copy; weights come from the 8-multiple arena
    // slice.
    unsafe {
      let mut acc0 = f64x2_splat(0.0);
      let mut acc1 = f64x2_splat(0.0);
      let mut acc2 = f64x2_splat(0.0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = (start + ci * 8) * 3;
        let mut sbuf = [0f32; 24];
        let p = if base + 24 <= row.len() {
          row.as_ptr().add(base)
        } else {
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          sbuf.as_ptr()
        };
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
        let w = v128_load(chunk.as_ptr().cast());
        let (wf0, wf1, wf2, wf3) = widen_w16_f64(w);
        acc0 = mac8_f32(acc0, r0, r1, wf0, wf1, wf2, wf3);
        acc1 = mac8_f32(acc1, g0, g1, wf0, wf1, wf2, wf3);
        acc2 = mac8_f32(acc2, b0, b1, wf0, wf1, wf2, wf3);
      }
      h_tmp[j * 3] = hsum_pd(acc0);
      h_tmp[j * 3 + 1] = hsum_pd(acc1);
      h_tmp[j * 3 + 2] = hsum_pd(acc2);
    }
  }
}

/// Float-element V-pass AXPY: `acc[i] += w * h_tmp[i]` in `f64`. A
/// separate multiply then add (not a fused multiply-add) so each lane
/// matches the scalar reference bit-for-bit — the V-pass is
/// element-wise, with no reordering. Two elements per iteration.
///
/// # Safety
///
/// simd128 must be enabled at compile time. `h_tmp.len() >= acc.len()`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn area_v_accumulate_f32(acc: &mut [f64], h_tmp: &[f64], w: f64) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = f64x2_splat(w);
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
    acc[k] += w * h_tmp[k];
  }
}
