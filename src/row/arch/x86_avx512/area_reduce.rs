//! AVX-512 fused-downscale H-pass and V-pass, widening the SSE4.1
//! kernels over the plan-time zero-padded u16 weight arena (see the
//! NEON sibling for the arena contract).
//!
//! The H-pass widens within a span to 32 taps per iteration
//! (`_mm512_mullo_epi16` / `_mm512_mulhi_epu16` into exact u32 lanes);
//! the remaining ≤24 padding-aligned taps and any chunk that would
//! read past the row end fall to the proven 128-bit SSE step, which
//! owns the row-end staging. The V-pass reaches 16 `u64` products per
//! iteration. Bit-identical to the scalar reference by integer
//! associativity.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

/// Sums the four u32 lanes of a 128-bit accumulator.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn hsum128_u32(acc: __m128i) -> u32 {
  let hi = _mm_shuffle_epi32::<0b01_00_11_10>(acc);
  let s = _mm_add_epi32(acc, hi);
  let s2 = _mm_add_epi32(s, _mm_shuffle_epi32::<0b00_00_00_01>(s));
  _mm_cvtsi128_si32(s2) as u32
}

/// Accumulates the eight exact u32 products of `s16 * w` (128-bit, 8
/// u16 lanes) into `acc`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac128_u16x8(acc: __m128i, s16: __m128i, w: __m128i) -> __m128i {
  let lo = _mm_mullo_epi16(s16, w);
  let hi = _mm_mulhi_epu16(s16, w);
  let acc = _mm_add_epi32(acc, _mm_unpacklo_epi16(lo, hi));
  _mm_add_epi32(acc, _mm_unpackhi_epi16(lo, hi))
}

/// Accumulates the thirty-two exact u32 products of `s16 * w`
/// (512-bit, 32 u16 lanes) into `acc`. The `unpack` interleaves run
/// per 128-bit lane, so a product lands in some lane of `acc` — which
/// lane is immaterial to the final reduction.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn mac512_u16x32(acc: __m512i, s16: __m512i, w: __m512i) -> __m512i {
  let lo = _mm512_mullo_epi16(s16, w);
  let hi = _mm512_mulhi_epu16(s16, w);
  let acc = _mm512_add_epi32(acc, _mm512_unpacklo_epi16(lo, hi));
  _mm512_add_epi32(acc, _mm512_unpackhi_epi16(lo, hi))
}

/// One proven 128-bit c1 step for an 8-tap group at `base`, with
/// row-end staging.
///
/// # Safety
///
/// AVX-512 (⊇ SSE4.1) available; `base < row.len()`; `w8.len() >= 8`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn c1_group(acc: __m128i, row: &[u8], base: usize, w8: &[u16]) -> __m128i {
  // SAFETY: the 8-byte load is guarded against row.len() or staged
  // through a zero-filled stack copy; the weight load reads 8 arena
  // u16.
  unsafe {
    let s8 = if base + 8 <= row.len() {
      _mm_loadl_epi64(row.as_ptr().add(base).cast())
    } else {
      let mut sbuf = [0u8; 8];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      _mm_loadl_epi64(sbuf.as_ptr().cast())
    };
    let w = _mm_loadu_si128(w8.as_ptr().cast());
    mac128_u16x8(acc, _mm_cvtepu8_epi16(s8), w)
  }
}

/// # Safety
///
/// AVX-512F+BW must be available. Caller guarantees the padded-arena
/// contract of the NEON sibling: `w16_off.len() == starts.len() + 1`
/// monotonic and bounded by `w16.len()`, span lengths multiples of 8,
/// `row.len() >= cells`, `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_h_reduce_row_c1(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    let len = span.len();
    // SAFETY: the wide 32-byte load runs only when fully in-bounds
    // (`start + t + 32 <= row.len()`); the remaining ≤24 taps and the
    // boundary chunk delegate to `c1_group`, which stages the row end.
    unsafe {
      let mut acc = _mm512_setzero_si512();
      let mut acc128 = _mm_setzero_si128();
      let mut t = 0usize;
      while t + 32 <= len && start + t + 32 <= row.len() {
        // 32 contiguous u8 -> 32 u16.
        let s8 = _mm256_loadu_si256(row.as_ptr().add(start + t).cast());
        let s16 = _mm512_cvtepu8_epi16(s8);
        let w = _mm512_loadu_si512(span[t..].as_ptr().cast());
        acc = mac512_u16x32(acc, s16, w);
        t += 32;
      }
      while t + 8 <= len {
        acc128 = c1_group(acc128, row, start + t, &span[t..]);
        t += 8;
      }
      let wide = _mm512_reduce_add_epi32(acc) as u32;
      h_tmp[j] = wide.wrapping_add(hsum128_u32(acc128));
    }
  }
}

/// The SSE4.1 per-128-lane deinterleave masks broadcast to all four
/// 512-bit lanes, so one shuffle deinterleaves four 8-pixel groups.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn c3_masks() -> ([__m512i; 3], [__m512i; 3]) {
  let m0 = [
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2),
  ];
  let m1 = [
    _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, -1, -1, -1, -1, -1, -1,
    ),
    _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, -1, -1, -1, -1, -1,
    ),
    _mm_set_epi8(
      -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, -1, -1, -1, -1, -1,
    ),
  ];
  (
    [
      _mm512_broadcast_i32x4(m0[0]),
      _mm512_broadcast_i32x4(m0[1]),
      _mm512_broadcast_i32x4(m0[2]),
    ],
    [
      _mm512_broadcast_i32x4(m1[0]),
      _mm512_broadcast_i32x4(m1[1]),
      _mm512_broadcast_i32x4(m1[2]),
    ],
  )
}

/// One proven 128-bit c3 step for an 8-pixel group at cell `cell`,
/// with row-end staging.
///
/// # Safety
///
/// AVX-512 (⊇ SSE4.1) available; `cell < cells`; `w8.len() >= 8`; `m0`
/// / `m1` carry the SSE masks in their low 128 bits.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn c3_group(
  acc: &mut [__m128i; 3],
  row: &[u8],
  cell: usize,
  w8: &[u16],
  m0: &[__m512i; 3],
  m1: &[__m512i; 3],
) {
  let base = cell * 3;
  // SAFETY: the two 16-byte loads cover bytes 0..24 of the chunk and
  // are guarded against row.len() or staged through a zero-filled
  // 24-byte copy.
  unsafe {
    let (v0, v1) = if base + 24 <= row.len() {
      (
        _mm_loadu_si128(row.as_ptr().add(base).cast()),
        _mm_loadu_si128(row.as_ptr().add(base + 8).cast()),
      )
    } else {
      let mut sbuf = [0u8; 24];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      (
        _mm_loadu_si128(sbuf.as_ptr().cast()),
        _mm_loadu_si128(sbuf.as_ptr().add(8).cast()),
      )
    };
    let w = _mm_loadu_si128(w8.as_ptr().cast());
    for ch in 0..3 {
      let m0c = _mm512_castsi512_si128(m0[ch]);
      let m1c = _mm512_castsi512_si128(m1[ch]);
      let gathered = _mm_or_si128(_mm_shuffle_epi8(v0, m0c), _mm_shuffle_epi8(v1, m1c));
      acc[ch] = mac128_u16x8(acc[ch], _mm_cvtepu8_epi16(gathered), w);
    }
  }
}

/// 3-channel variant. The wide path packs four 8-pixel groups into the
/// four 128-bit lanes and deinterleaves all with one per-lane
/// `_mm512_shuffle_epi8`; `_mm512_unpacklo_epi8` widens each lane's
/// low 8 samples `u8 -> u16`.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells * 3` and
/// `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_h_reduce_row_c3(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  let (m0, m1) = c3_masks();
  let zero = _mm512_setzero_si512();
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    let len = span.len();
    // SAFETY: the wide path runs only when all four groups' 24-byte
    // spans are fully in-bounds (`(start + t + 32) * 3 <= row.len()`);
    // boundary and trailing groups delegate to `c3_group`, which
    // stages the row end.
    unsafe {
      let mut acc = [_mm512_setzero_si512(); 3];
      let mut acc128 = [_mm_setzero_si128(); 3];
      let mut t = 0usize;
      while t + 32 <= len && (start + t + 32) * 3 <= row.len() {
        // Four 8-pixel groups -> the four 128-bit lanes. `inserti32x4`
        // takes a const lane index, so the four inserts are unrolled.
        let lo = |g: usize| _mm_loadu_si128(row.as_ptr().add((start + t + g * 8) * 3).cast());
        let hi = |g: usize| _mm_loadu_si128(row.as_ptr().add((start + t + g * 8) * 3 + 8).cast());
        let v0 = _mm512_inserti32x4::<3>(
          _mm512_inserti32x4::<2>(
            _mm512_inserti32x4::<1>(_mm512_castsi128_si512(lo(0)), lo(1)),
            lo(2),
          ),
          lo(3),
        );
        let v1 = _mm512_inserti32x4::<3>(
          _mm512_inserti32x4::<2>(
            _mm512_inserti32x4::<1>(_mm512_castsi128_si512(hi(0)), hi(1)),
            hi(2),
          ),
          hi(3),
        );
        // The four 128-bit lanes hold the four groups' 8 u16 weights.
        let w = _mm512_loadu_si512(span[t..].as_ptr().cast());
        for ch in 0..3 {
          let gathered = _mm512_or_si512(
            _mm512_shuffle_epi8(v0, m0[ch]),
            _mm512_shuffle_epi8(v1, m1[ch]),
          );
          let s16 = _mm512_unpacklo_epi8(gathered, zero);
          acc[ch] = mac512_u16x32(acc[ch], s16, w);
        }
        t += 32;
      }
      while t + 8 <= len {
        c3_group(&mut acc128, row, start + t, &span[t..], &m0, &m1);
        t += 8;
      }
      for ch in 0..3 {
        let wide = _mm512_reduce_add_epi32(acc[ch]) as u32;
        h_tmp[j * 3 + ch] = wide.wrapping_add(hsum128_u32(acc128[ch]));
      }
    }
  }
}

/// V-pass AXPY: `acc[i] += w * h_tmp[i]`, exact u64 lanes via
/// `_mm512_mul_epu32` over even/odd u32 lanes, reassembled with
/// `_mm512_permutex2var_epi64` (16 elements per iteration).
///
/// # Safety
///
/// AVX-512F must be available. `h_tmp.len() >= acc.len()`; every
/// product-sum stays within u64 (the engine's denominator bound).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn area_v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm512_set1_epi64(i64::from(w));
  // even product e -> contiguous lane 2e; odd product o -> lane 2o+1.
  let idx_lo = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
  let idx_hi = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
  let mut i = 0usize;
  // SAFETY: loop guard `i + 16 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 16 <= n {
      let t = _mm512_loadu_si512(h_tmp.as_ptr().add(i).cast());
      // even = [t0,t2,..,t14]*w, odd = [t1,t3,..,t15]*w (u64 lanes).
      let even = _mm512_mul_epu32(t, wv);
      let odd = _mm512_mul_epu32(_mm512_srli_epi64::<32>(t), wv);
      // Interleave back to contiguous [t0..t7]*w and [t8..t15]*w.
      let lo = _mm512_permutex2var_epi64(even, idx_lo, odd);
      let hi = _mm512_permutex2var_epi64(even, idx_hi, odd);
      let a_lo = _mm512_loadu_si512(acc.as_ptr().add(i).cast());
      let a_hi = _mm512_loadu_si512(acc.as_ptr().add(i + 8).cast());
      _mm512_storeu_si512(acc.as_mut_ptr().add(i).cast(), _mm512_add_epi64(a_lo, lo));
      _mm512_storeu_si512(
        acc.as_mut_ptr().add(i + 8).cast(),
        _mm512_add_epi64(a_hi, hi),
      );
      i += 16;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * u64::from(h_tmp[k]);
  }
}
