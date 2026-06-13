//! Scalar reference for the fused-downscale H-pass: per output span,
//! the weighted sum of its source samples. The exact-integer contract
//! (`u32` sums bounded by `src * 255`) makes every backend's results
//! bit-identical to this one.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

/// `h_tmp[j * channels + ch] = Σ weights[k] * row[(start_j + i) * channels + ch]`
/// for each output span `j` described by `(starts, offsets, weights)`.
pub(crate) fn area_h_reduce_row(
  row: &[u8],
  channels: usize,
  starts: &[usize],
  offsets: &[usize],
  weights: &[usize],
  h_tmp: &mut [u32],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &weights[offsets[j]..offsets[j + 1]];
    let base = j * channels;
    for ch in 0..channels {
      let mut sum = 0u32;
      for (i, &w) in span.iter().enumerate() {
        sum += w as u32 * u32::from(row[(start + i) * channels + ch]);
      }
      h_tmp[base + ch] = sum;
    }
  }
}

/// V-pass AXPY reference: `acc[i] += w * h_tmp[i]` over the H-reduced
/// row. Products and sums are exact in u64 by the engine's
/// denominator bound (`denom * 255 <= u64::MAX`).
pub(crate) fn area_v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u64) {
  for (a, t) in acc.iter_mut().zip(h_tmp.iter()) {
    *a += w * u64::from(*t);
  }
}

/// 16-bit-element H-pass reference: like [`area_h_reduce_row`] but the
/// samples are `u16` and the per-span sums live in `u64` — a `u16`
/// sample times a `u16`-bounded weight reaches `~2^32`, so even a
/// single span overflows `u32`. Bit-identical contract for every u16
/// backend.
pub(crate) fn area_h_reduce_row_u16(
  row: &[u16],
  channels: usize,
  starts: &[usize],
  offsets: &[usize],
  weights: &[usize],
  h_tmp: &mut [u64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &weights[offsets[j]..offsets[j + 1]];
    let base = j * channels;
    for ch in 0..channels {
      let mut sum = 0u64;
      for (i, &w) in span.iter().enumerate() {
        sum += w as u64 * u64::from(row[(start + i) * channels + ch]);
      }
      h_tmp[base + ch] = sum;
    }
  }
}

/// 16-bit-element V-pass AXPY reference: `acc[i] += w * h_tmp[i]` with
/// `h_tmp` already `u64`. Exact in `u64` by the engine's denominator
/// bound (`denom * 65535 <= u64::MAX`).
pub(crate) fn area_v_accumulate_u16(acc: &mut [u64], h_tmp: &[u64], w: u64) {
  for (a, t) in acc.iter_mut().zip(h_tmp.iter()) {
    *a += w * *t;
  }
}
