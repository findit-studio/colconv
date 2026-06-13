//! AVX2 fused-downscale H-pass and V-pass, widening the SSE4.1 kernels
//! over the plan-time zero-padded u16 weight arena (see the NEON
//! sibling for the arena contract).
//!
//! A span's taps read contiguous source cells and its weights are
//! padded to a multiple of 8, so the H-pass widens *within* a span:
//! 16 taps per iteration (`_mm256_mullo_epi16` / `_mm256_mulhi_epu16`
//! into exact u32 lanes). The boundary groups — a trailing 8-tap
//! group and any chunk that would read past the row end — fall to the
//! proven 128-bit SSE step, which owns all the row-end staging; the
//! wide path only runs fully in-bounds. The V-pass doubles the SSE
//! throughput to 8 `u64` products per iteration. Bit-identical to the
//! scalar reference by integer associativity.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

/// Sums the four u32 lanes of a 128-bit accumulator.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum128_u32(acc: __m128i) -> u32 {
  let hi = _mm_shuffle_epi32::<0b01_00_11_10>(acc);
  let s = _mm_add_epi32(acc, hi);
  let s2 = _mm_add_epi32(s, _mm_shuffle_epi32::<0b00_00_00_01>(s));
  _mm_cvtsi128_si32(s2) as u32
}

/// Sums the eight u32 lanes of a 256-bit accumulator.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum256_u32(acc: __m256i) -> u32 {
  let lo = _mm256_castsi256_si128(acc);
  let hi = _mm256_extracti128_si256::<1>(acc);
  hsum128_u32(_mm_add_epi32(lo, hi))
}

/// Accumulates the eight exact u32 products of `s16 * w` (128-bit, 8
/// u16 lanes) into `acc`.
#[inline]
#[target_feature(enable = "avx2")]
fn mac128_u16x8(acc: __m128i, s16: __m128i, w: __m128i) -> __m128i {
  let lo = _mm_mullo_epi16(s16, w);
  let hi = _mm_mulhi_epu16(s16, w);
  let acc = _mm_add_epi32(acc, _mm_unpacklo_epi16(lo, hi));
  _mm_add_epi32(acc, _mm_unpackhi_epi16(lo, hi))
}

/// Accumulates the sixteen exact u32 products of `s16 * w` (256-bit,
/// 16 u16 lanes) into `acc`. The `unpack` interleaves run per 128-bit
/// lane, so a product lands in some lane of `acc` — which lane is
/// immaterial to the final `hsum256_u32`.
#[inline]
#[target_feature(enable = "avx2")]
fn mac256_u16x16(acc: __m256i, s16: __m256i, w: __m256i) -> __m256i {
  let lo = _mm256_mullo_epi16(s16, w);
  let hi = _mm256_mulhi_epu16(s16, w);
  let acc = _mm256_add_epi32(acc, _mm256_unpacklo_epi16(lo, hi));
  _mm256_add_epi32(acc, _mm256_unpackhi_epi16(lo, hi))
}

/// One proven 128-bit c1 step for an 8-tap group at `base`, with
/// row-end staging. Returns the products folded into `acc`.
///
/// # Safety
///
/// AVX2 (⊇ SSE4.1) available; `base < row.len()`; `w8.len() >= 8`.
#[inline]
#[target_feature(enable = "avx2")]
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
/// AVX2 must be available. Caller guarantees the padded-arena contract
/// of the NEON sibling: `w16_off.len() == starts.len() + 1` monotonic
/// and bounded by `w16.len()`, span lengths multiples of 8,
/// `row.len() >= cells`, `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "avx2")]
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
    // SAFETY: wide loads run only fully in-bounds (the `* 3`-free
    // `base + 16 <= row.len()` guard); boundary and trailing groups
    // delegate to `c1_group`, which stages the row end.
    unsafe {
      let mut acc = _mm256_setzero_si256();
      let mut acc128 = _mm_setzero_si128();
      let mut t = 0usize;
      while t + 16 <= len && start + t + 16 <= row.len() {
        let s16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(row.as_ptr().add(start + t).cast()));
        let w = _mm256_loadu_si256(span[t..].as_ptr().cast());
        acc = mac256_u16x16(acc, s16, w);
        t += 16;
      }
      while t + 8 <= len {
        acc128 = c1_group(acc128, row, start + t, &span[t..]);
        t += 8;
      }
      h_tmp[j] = hsum256_u32(acc).wrapping_add(hsum128_u32(acc128));
    }
  }
}

/// The SSE4.1 per-128-lane deinterleave masks (byte g of a 24-byte
/// chunk → lane g of the first load or lane g-8 of the second), each
/// broadcast to both 256-bit lanes so one shuffle deinterleaves two
/// 8-pixel groups at once.
#[inline]
#[target_feature(enable = "avx2")]
fn c3_masks() -> ([__m256i; 3], [__m256i; 3]) {
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
      _mm256_broadcastsi128_si256(m0[0]),
      _mm256_broadcastsi128_si256(m0[1]),
      _mm256_broadcastsi128_si256(m0[2]),
    ],
    [
      _mm256_broadcastsi128_si256(m1[0]),
      _mm256_broadcastsi128_si256(m1[1]),
      _mm256_broadcastsi128_si256(m1[2]),
    ],
  )
}

/// One proven 128-bit c3 step for an 8-pixel group at cell `cell`
/// (byte base `cell * 3`), with row-end staging. Folds each channel's
/// products into `acc[ch]`.
///
/// # Safety
///
/// AVX2 (⊇ SSE4.1) available; `cell < cells`; `w8.len() >= 8`; `m0`,
/// `m1` are the SSE masks (low 128 of the broadcast pair).
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn c3_group(
  acc: &mut [__m128i; 3],
  row: &[u8],
  cell: usize,
  w8: &[u16],
  m0: &[__m256i; 3],
  m1: &[__m256i; 3],
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
      let m0c = _mm256_castsi256_si128(m0[ch]);
      let m1c = _mm256_castsi256_si128(m1[ch]);
      let gathered = _mm_or_si128(_mm_shuffle_epi8(v0, m0c), _mm_shuffle_epi8(v1, m1c));
      acc[ch] = mac128_u16x8(acc[ch], _mm_cvtepu8_epi16(gathered), w);
    }
  }
}

/// 3-channel variant. The wide path packs two 8-pixel groups into the
/// two 128-bit lanes (`group A` low, `group B` high) and deinterleaves
/// both with one per-lane `_mm256_shuffle_epi8`; `_mm256_unpacklo_epi8`
/// widens each lane's low 8 samples `u8 -> u16`.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells * 3` and
/// `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn area_h_reduce_row_c3(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  let (m0, m1) = c3_masks();
  let zero = _mm256_setzero_si256();
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    let len = span.len();
    // SAFETY: the wide path runs only when both groups' 24-byte spans
    // are fully in-bounds (`(start + t + 16) * 3 <= row.len()`);
    // boundary and trailing groups delegate to `c3_group`, which
    // stages the row end.
    unsafe {
      let mut acc = [_mm256_setzero_si256(); 3];
      let mut acc128 = [_mm_setzero_si128(); 3];
      let mut t = 0usize;
      while t + 16 <= len && (start + t + 16) * 3 <= row.len() {
        let base_a = (start + t) * 3;
        let base_b = (start + t + 8) * 3;
        let v0 = _mm256_inserti128_si256::<1>(
          _mm256_castsi128_si256(_mm_loadu_si128(row.as_ptr().add(base_a).cast())),
          _mm_loadu_si128(row.as_ptr().add(base_b).cast()),
        );
        let v1 = _mm256_inserti128_si256::<1>(
          _mm256_castsi128_si256(_mm_loadu_si128(row.as_ptr().add(base_a + 8).cast())),
          _mm_loadu_si128(row.as_ptr().add(base_b + 8).cast()),
        );
        // lane0 = group A's 8 u16 weights, lane1 = group B's.
        let w = _mm256_loadu_si256(span[t..].as_ptr().cast());
        for ch in 0..3 {
          let gathered = _mm256_or_si256(
            _mm256_shuffle_epi8(v0, m0[ch]),
            _mm256_shuffle_epi8(v1, m1[ch]),
          );
          let s16 = _mm256_unpacklo_epi8(gathered, zero);
          acc[ch] = mac256_u16x16(acc[ch], s16, w);
        }
        t += 16;
      }
      while t + 8 <= len {
        c3_group(&mut acc128, row, start + t, &span[t..], &m0, &m1);
        t += 8;
      }
      for ch in 0..3 {
        h_tmp[j * 3 + ch] = hsum256_u32(acc[ch]).wrapping_add(hsum128_u32(acc128[ch]));
      }
    }
  }
}

/// V-pass AXPY: `acc[i] += w * h_tmp[i]`, exact u64 lanes via
/// `_mm256_mul_epu32` over even/odd u32 lanes, reassembled with
/// `_mm256_permute2x128_si256` (8 elements per iteration).
///
/// # Safety
///
/// AVX2 must be available. `h_tmp.len() >= acc.len()`; every
/// product-sum stays within u64 (the engine's denominator bound).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn area_v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm256_set1_epi64x(i64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 8 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 8 <= n {
      let t = _mm256_loadu_si256(h_tmp.as_ptr().add(i).cast());
      // even = [t0,t2,t4,t6]*w, odd = [t1,t3,t5,t7]*w (u64 lanes).
      let even = _mm256_mul_epu32(t, wv);
      let odd = _mm256_mul_epu32(_mm256_srli_epi64::<32>(t), wv);
      // unpack (per-128-lane) -> [t0,t1,t4,t5] and [t2,t3,t6,t7];
      // permute reassembles the contiguous [t0..t3] and [t4..t7].
      let pl = _mm256_unpacklo_epi64(even, odd);
      let ph = _mm256_unpackhi_epi64(even, odd);
      let lo = _mm256_permute2x128_si256::<0x20>(pl, ph);
      let hi = _mm256_permute2x128_si256::<0x31>(pl, ph);
      let a_lo = _mm256_loadu_si256(acc.as_ptr().add(i).cast());
      let a_hi = _mm256_loadu_si256(acc.as_ptr().add(i + 4).cast());
      _mm256_storeu_si256(acc.as_mut_ptr().add(i).cast(), _mm256_add_epi64(a_lo, lo));
      _mm256_storeu_si256(
        acc.as_mut_ptr().add(i + 4).cast(),
        _mm256_add_epi64(a_hi, hi),
      );
      i += 8;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * u64::from(h_tmp[k]);
  }
}

/// Sums the two u64 lanes of a 128-bit accumulator.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum128_u64(acc: __m128i) -> u64 {
  let hi = _mm_unpackhi_epi64(acc, acc);
  _mm_cvtsi128_si64(_mm_add_epi64(acc, hi)) as u64
}

/// Sums the four u64 lanes of a 256-bit accumulator.
#[inline]
#[target_feature(enable = "avx2")]
fn hsum256_u64(acc: __m256i) -> u64 {
  let lo = _mm256_castsi256_si128(acc);
  let hi = _mm256_extracti128_si256::<1>(acc);
  hsum128_u64(_mm_add_epi64(lo, hi))
}

/// Accumulates the eight exact `u16 * u16 -> u32` products of `s16 * w`
/// (128-bit, 8 u16 lanes) into the two u64 lanes of `acc`. Unlike the
/// u8 [`mac128_u16x8`], a `u16` span sum overflows `u32`, so the
/// products widen to u64 before accumulating. Mirrors the SSE4.1
/// `mac_u16x8_u64`.
#[inline]
#[target_feature(enable = "avx2")]
fn mac128_u16x8_u64(acc: __m128i, s16: __m128i, w: __m128i) -> __m128i {
  let lo = _mm_mullo_epi16(s16, w);
  let hi = _mm_mulhi_epu16(s16, w);
  let p_lo = _mm_unpacklo_epi16(lo, hi);
  let p_hi = _mm_unpackhi_epi16(lo, hi);
  let acc = _mm_add_epi64(acc, _mm_cvtepu32_epi64(p_lo));
  let acc = _mm_add_epi64(acc, _mm_cvtepu32_epi64(_mm_srli_si128::<8>(p_lo)));
  let acc = _mm_add_epi64(acc, _mm_cvtepu32_epi64(p_hi));
  _mm_add_epi64(acc, _mm_cvtepu32_epi64(_mm_srli_si128::<8>(p_hi)))
}

/// Accumulates the sixteen exact `u16 * u16 -> u32` products of
/// `s16 * w` (256-bit, 16 u16 lanes) into the four u64 lanes of `acc`.
/// The `mullo`/`mulhi` pair gives exact u32 products; each
/// `unpack`-result's eight u32 (per-128-lane order) widen to u64 in
/// four `_mm256_cvtepu32_epi64` groups before adding. Which u64 lane a
/// product lands in is immaterial to the final `hsum256_u64`.
#[inline]
#[target_feature(enable = "avx2")]
fn mac256_u16x16_u64(acc: __m256i, s16: __m256i, w: __m256i) -> __m256i {
  let lo = _mm256_mullo_epi16(s16, w);
  let hi = _mm256_mulhi_epu16(s16, w);
  let p_lo = _mm256_unpacklo_epi16(lo, hi);
  let p_hi = _mm256_unpackhi_epi16(lo, hi);
  let acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(p_lo)));
  let acc = _mm256_add_epi64(
    acc,
    _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(p_lo)),
  );
  let acc = _mm256_add_epi64(acc, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(p_hi)));
  _mm256_add_epi64(
    acc,
    _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(p_hi)),
  )
}

/// One proven 128-bit u16 c1 step for an 8-tap group at `base`, with
/// row-end staging. Folds the products into the u64 lanes of `acc`.
/// Mirrors the inner logic of the SSE4.1 `area_h_reduce_row_u16_c1`.
///
/// # Safety
///
/// AVX2 (⊇ SSE4.1) available; `base < row.len()`; `w8.len() >= 8`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn c1_group_u16(acc: __m128i, row: &[u16], base: usize, w8: &[u16]) -> __m128i {
  // SAFETY: the 8-element u16 load is guarded against row.len() or
  // staged through a zero-filled stack copy; the weight load reads 8
  // arena u16.
  unsafe {
    let s16 = if base + 8 <= row.len() {
      _mm_loadu_si128(row.as_ptr().add(base).cast())
    } else {
      let mut sbuf = [0u16; 8];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      _mm_loadu_si128(sbuf.as_ptr().cast())
    };
    let w = _mm_loadu_si128(w8.as_ptr().cast());
    mac128_u16x8_u64(acc, s16, w)
  }
}

/// 16-bit-element H-pass (1 channel): like [`area_h_reduce_row_c1`] but
/// the samples load directly as `u16` (no `u8 -> u16` widening) and the
/// products accumulate in `u64` lanes — a single span sum can exceed
/// `u32`. The wide path widens 16 taps per iteration into the four u64
/// lanes of a 256-bit accumulator; boundary and trailing 8-tap groups
/// fall to the 128-bit `c1_group_u16`, which owns the row-end staging.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells` `u16`
/// elements and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn area_h_reduce_row_u16_c1(
  row: &[u16],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    let len = span.len();
    // SAFETY: wide loads run only fully in-bounds (`start + t + 16 <=
    // row.len()`, in u16 elements); boundary and trailing groups
    // delegate to `c1_group_u16`, which stages the row end.
    unsafe {
      let mut acc = _mm256_setzero_si256();
      let mut acc128 = _mm_setzero_si128();
      let mut t = 0usize;
      while t + 16 <= len && start + t + 16 <= row.len() {
        let s16 = _mm256_loadu_si256(row.as_ptr().add(start + t).cast());
        let w = _mm256_loadu_si256(span[t..].as_ptr().cast());
        acc = mac256_u16x16_u64(acc, s16, w);
        t += 16;
      }
      while t + 8 <= len {
        acc128 = c1_group_u16(acc128, row, start + t, &span[t..]);
        t += 8;
      }
      h_tmp[j] = hsum256_u64(acc).wrapping_add(hsum128_u64(acc128));
    }
  }
}

/// The SSE4.1 u16 per-channel deinterleave masks: for channel `ch` the
/// eight samples sit at u16 index `ch + 3t` (byte `2*(ch + 3t)`), split
/// across the three overlapping 16-byte loads of a 48-byte chunk. Each
/// mask pulls its load's contributing u16 pairs into output lanes 0..7
/// and -1 zeroes the rest. Copied verbatim from the SSE4.1
/// `area_h_reduce_row_u16_c3`.
#[inline]
#[target_feature(enable = "avx2")]
fn c3_u16_masks() -> ([__m128i; 3], [__m128i; 3], [__m128i; 3]) {
  let m0: [__m128i; 3] = [
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 7, 6, 1, 0),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 14, 9, 8, 3, 2),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 10, 5, 4),
  ];
  let m1: [__m128i; 3] = [
    _mm_set_epi8(-1, -1, -1, -1, 15, 14, 9, 8, 3, 2, -1, -1, -1, -1, -1, -1),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, 11, 10, 5, 4, -1, -1, -1, -1, -1, -1),
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 12, 7, 6, 1, 0, -1, -1, -1, -1),
  ];
  let m2: [__m128i; 3] = [
    _mm_set_epi8(11, 10, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    _mm_set_epi8(13, 12, 7, 6, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    _mm_set_epi8(15, 14, 9, 8, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
  ];
  (m0, m1, m2)
}

/// One proven 128-bit u16 c3 step for an 8-pixel group at cell `cell`
/// (u16 base `cell * 3`), with row-end staging. Folds each channel's
/// products into the u64 lanes of `acc[ch]`. Mirrors the inner logic of
/// the SSE4.1 `area_h_reduce_row_u16_c3`.
///
/// # Safety
///
/// AVX2 (⊇ SSE4.1) available; `cell < cells`; `w8.len() >= 8`; `m0`,
/// `m1`, `m2` are the SSE u16 c3 masks.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn c3_group_u16(
  acc: &mut [__m128i; 3],
  row: &[u16],
  cell: usize,
  w8: &[u16],
  m0: &[__m128i; 3],
  m1: &[__m128i; 3],
  m2: &[__m128i; 3],
) {
  let base = cell * 3;
  // SAFETY: the three 16-byte loads cover the chunk's 24 u16 (48 bytes)
  // and are guarded against row.len() or staged through a zero-filled
  // 24-u16 copy.
  unsafe {
    let (v0, v1, v2) = if base + 24 <= row.len() {
      (
        _mm_loadu_si128(row.as_ptr().add(base).cast()),
        _mm_loadu_si128(row.as_ptr().add(base + 8).cast()),
        _mm_loadu_si128(row.as_ptr().add(base + 16).cast()),
      )
    } else {
      let mut sbuf = [0u16; 24];
      let take = row.len() - base;
      sbuf[..take].copy_from_slice(&row[base..]);
      (
        _mm_loadu_si128(sbuf.as_ptr().cast()),
        _mm_loadu_si128(sbuf.as_ptr().add(8).cast()),
        _mm_loadu_si128(sbuf.as_ptr().add(16).cast()),
      )
    };
    let w = _mm_loadu_si128(w8.as_ptr().cast());
    for ch in 0..3 {
      let gathered = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(v0, m0[ch]), _mm_shuffle_epi8(v1, m1[ch])),
        _mm_shuffle_epi8(v2, m2[ch]),
      );
      acc[ch] = mac128_u16x8_u64(acc[ch], gathered, w);
    }
  }
}

/// 16-bit-element H-pass (3-channel interleaved RGB). A `u16` 8-pixel
/// group spans 24 `u16` (48 bytes), three 16-byte loads — not the
/// 24-byte two-128-lane pack the u8 c3 wide path uses — so the AVX2
/// wide path is impractical here and this mirrors the SSE4.1 u16 c3
/// 128-bit step over every 8-tap chunk (three overlapping loads + nine
/// `_mm_shuffle_epi8` masks + `mac128_u16x8_u64`). The wide path only
/// helps extreme downscales anyway.
///
/// # Safety
///
/// As [`area_h_reduce_row_u16_c1`], with `row.len() >= cells * 3` `u16`
/// elements and `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn area_h_reduce_row_u16_c3(
  row: &[u16],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u64],
) {
  let (m0, m1, m2) = c3_u16_masks();
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    let len = span.len();
    // SAFETY: each 8-tap chunk delegates to `c3_group_u16`, which
    // stages the chunk's 48 bytes against the row end; weights come
    // from the 8-multiple arena slice.
    unsafe {
      let mut acc = [_mm_setzero_si128(); 3];
      let mut t = 0usize;
      while t + 8 <= len {
        c3_group_u16(&mut acc, row, start + t, &span[t..], &m0, &m1, &m2);
        t += 8;
      }
      for ch in 0..3 {
        h_tmp[j * 3 + ch] = hsum128_u64(acc[ch]);
      }
    }
  }
}

/// 16-bit-element V-pass AXPY: `acc[i] += w * h_tmp[i]` with `h_tmp`
/// already `u64`. The `u32 * u64 -> u64` product splits `h_tmp` into
/// 32-bit halves — `_mm256_mul_epu32` gives `w * lo` and `w * hi`, the
/// latter shifted up 32 — summed mod 2^64 (exact by the engine bound).
/// Four elements per iteration; mirrors the SSE4.1
/// `area_v_accumulate_u16`.
///
/// # Safety
///
/// AVX2 must be available. `h_tmp.len() >= acc.len()`; every
/// product-sum stays within u64.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn area_v_accumulate_u16(acc: &mut [u64], h_tmp: &[u64], w: u32) {
  let n = acc.len();
  debug_assert!(h_tmp.len() >= n, "h_tmp too short");
  let wv = _mm256_set1_epi64x(i64::from(w));
  let mut i = 0usize;
  // SAFETY: loop guard `i + 4 <= n` with `h_tmp.len() >= n` keeps all
  // loads and stores in bounds.
  unsafe {
    while i + 4 <= n {
      let t = _mm256_loadu_si256(h_tmp.as_ptr().add(i).cast());
      let prod_lo = _mm256_mul_epu32(t, wv);
      let prod_hi = _mm256_mul_epu32(_mm256_srli_epi64::<32>(t), wv);
      let prod = _mm256_add_epi64(prod_lo, _mm256_slli_epi64::<32>(prod_hi));
      let a = _mm256_loadu_si256(acc.as_ptr().add(i).cast());
      _mm256_storeu_si256(acc.as_mut_ptr().add(i).cast(), _mm256_add_epi64(a, prod));
      i += 4;
    }
  }
  for k in i..n {
    acc[k] += u64::from(w) * h_tmp[k];
  }
}
