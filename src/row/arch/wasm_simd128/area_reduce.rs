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
