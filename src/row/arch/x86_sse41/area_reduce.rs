//! SSE4.1 fused-downscale H-pass: per output span, widening
//! multiply-accumulate over the plan-time zero-padded u16 weight
//! arena (see the NEON sibling for the arena contract).
//!
//! Per 8-tap chunk: 8 samples zero-extend `u8 -> u16`
//! (`_mm_cvtepu8_epi16`) and meet 8 arena weights through the
//! `_mm_mullo_epi16` / `_mm_mulhi_epu16` pair, whose halves
//! interleave (`_mm_unpacklo/hi_epi16`) into exact u32 products.
//! Padding lanes multiply by zero, so sample loads only stage through
//! a stack copy at the row-end boundary. 3-channel rows deinterleave
//! with paired `_mm_shuffle_epi8` masks over two overlapping 16-byte
//! loads. Bit-identical to the scalar reference by integer
//! associativity.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(target_arch = "x86_64")]
#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

/// Sums the four u32 lanes of `acc`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn hsum_u32(acc: __m128i) -> u32 {
  let hi = _mm_shuffle_epi32::<0b01_00_11_10>(acc);
  let s = _mm_add_epi32(acc, hi);
  let s2 = _mm_add_epi32(s, _mm_shuffle_epi32::<0b00_00_00_01>(s));
  _mm_cvtsi128_si32(s2) as u32
}

/// Accumulates the eight exact u32 products of `s16 * w` into `acc`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn mac_u16x8(acc: __m128i, s16: __m128i, w: __m128i) -> __m128i {
  let lo = _mm_mullo_epi16(s16, w);
  let hi = _mm_mulhi_epu16(s16, w);
  let acc = _mm_add_epi32(acc, _mm_unpacklo_epi16(lo, hi));
  _mm_add_epi32(acc, _mm_unpackhi_epi16(lo, hi))
}

/// # Safety
///
/// SSE4.1 must be available. Caller guarantees the padded-arena
/// contract of the NEON sibling: `w16_off.len() == starts.len() + 1`
/// monotonic and bounded by `w16.len()`, span lengths multiples of 8,
/// `row.len() >= cells`, `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "sse4.1")]
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
      let mut acc = _mm_setzero_si128();
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let s8 = if base + 8 <= row.len() {
          _mm_loadl_epi64(row.as_ptr().add(base).cast())
        } else {
          let mut sbuf = [0u8; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          _mm_loadl_epi64(sbuf.as_ptr().cast())
        };
        let s16 = _mm_cvtepu8_epi16(s8);
        let w = _mm_loadu_si128(chunk.as_ptr().cast());
        acc = mac_u16x8(acc, s16, w);
      }
      h_tmp[j] = hsum_u32(acc);
    }
  }
}

/// 3-channel (interleaved RGB) variant: two overlapping 16-byte loads
/// cover each chunk's 24 bytes, and per-channel `_mm_shuffle_epi8`
/// mask pairs gather the eight samples of each channel.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells * 3` and
/// `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn area_h_reduce_row_c3(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  // Byte g of a 24-byte chunk lives in lane g of the first load
  // (bytes 0..16) or lane g - 8 of the second (bytes 8..24); each
  // channel's eight samples (g = ch + 3t) gather from whichever load
  // holds them, with -1 zeroing the other mask's lane.
  let (m0, m1): ([__m128i; 3], [__m128i; 3]) = (
    [
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0),
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1),
      _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2),
    ],
    [
      _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, -1, -1, -1, -1, -1, -1,
      ),
      _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, -1, -1, -1, -1, -1,
      ),
      _mm_set_epi8(
        -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, -1, -1, -1, -1, -1,
      ),
    ],
  );
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: the two 16-byte loads cover bytes 0..24 of the chunk and
    // are either fully in-bounds (guarded against row.len()) or staged
    // through a zero-filled 24-byte stack copy; weight loads come from
    // the 8-multiple arena slice.
    unsafe {
      let mut acc = [_mm_setzero_si128(); 3];
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = (start + ci * 8) * 3;
        let mut sbuf = [0u8; 24];
        let (v0, v1) = if base + 24 <= row.len() {
          (
            _mm_loadu_si128(row.as_ptr().add(base).cast()),
            _mm_loadu_si128(row.as_ptr().add(base + 8).cast()),
          )
        } else {
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          (
            _mm_loadu_si128(sbuf.as_ptr().cast()),
            _mm_loadu_si128(sbuf.as_ptr().add(8).cast()),
          )
        };
        let w = _mm_loadu_si128(chunk.as_ptr().cast());
        for ch in 0..3 {
          let gathered = _mm_or_si128(_mm_shuffle_epi8(v0, m0[ch]), _mm_shuffle_epi8(v1, m1[ch]));
          acc[ch] = mac_u16x8(acc[ch], _mm_cvtepu8_epi16(gathered), w);
        }
      }
      for ch in 0..3 {
        h_tmp[j * 3 + ch] = hsum_u32(acc[ch]);
      }
    }
  }
}

/// V-pass AXPY: `acc[i] += w * h_tmp[i]`, exact u64 lanes via
/// `_mm_mul_epu32` over even/odd u32 lanes re-paired with
/// `_mm_unpacklo/hi_epi64` (4 elements per iteration).
///
/// # Safety
///
/// SSE4.1 must be available. `h_tmp.len() >= acc.len()`; every
/// product-sum stays within u64 (the engine's denominator bound).
#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn area_v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  // Each 64-bit lane's low half holds `w` (high half zero), the form
  // `_mm_mul_epu32` consumes.
  let wv = _mm_set1_epi64x(i64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 4 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 4 <= n {
      let t = _mm_loadu_si128(h_tmp.as_ptr().add(i).cast());
      let even = _mm_mul_epu32(t, wv);
      let odd = _mm_mul_epu32(_mm_srli_epi64::<32>(t), wv);
      let a01 = _mm_loadu_si128(acc.as_ptr().add(i).cast());
      let a23 = _mm_loadu_si128(acc.as_ptr().add(i + 2).cast());
      let p01 = _mm_unpacklo_epi64(even, odd);
      let p23 = _mm_unpackhi_epi64(even, odd);
      _mm_storeu_si128(acc.as_mut_ptr().add(i).cast(), _mm_add_epi64(a01, p01));
      _mm_storeu_si128(acc.as_mut_ptr().add(i + 2).cast(), _mm_add_epi64(a23, p23));
      i += 4;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * u64::from(h_tmp[k]);
  }
}
