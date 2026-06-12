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
