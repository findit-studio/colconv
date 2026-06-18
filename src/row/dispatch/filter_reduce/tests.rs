use super::*;

fn arena(starts: &[usize], offsets: &[usize], coeffs: &[f32]) -> FilterPaddedSpans {
  FilterPaddedSpans::build(starts, offsets, coeffs).expect("valid spans")
}

#[test]
fn build_proves_padding_and_reach() {
  // A 4-tap and a 9-tap span pad to 8 and 16 entries; the reach is the
  // furthest real end. Non-monotonic offsets and non-8-multiple spans are
  // unrepresentable: only this builder populates the fields.
  let p = arena(&[0, 4], &[0, 4, 13], &[1.0f32; 13]);
  assert_eq!(p.off, std::vec![0, 8, 24]);
  assert_eq!(p.coeffs.len(), 24);
  assert_eq!(p.max_reach, 13);
  assert_eq!(p.starts, std::vec![0, 4]);
  assert_eq!(&p.coeffs[..4], [1.0, 1.0, 1.0, 1.0]);
  // The padding lanes are exactly +0.0, so they annihilate their sample.
  assert_eq!(&p.coeffs[4..8], [0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn build_preserves_signed_coefficients() {
  // Unlike the integer area arena there is no weight bound: negative lobes
  // (Catmull-Rom / Lanczos) must survive into the arena verbatim.
  let p = arena(&[0], &[0, 3], &[-0.25f32, 1.5, -0.25]);
  assert_eq!(&p.coeffs[..3], [-0.25, 1.5, -0.25]);
  assert_eq!(p.coeffs.len(), 8);
}

#[test]
fn build_rejects_mismatched_offsets() {
  // `offsets.len()` must be `out + 1`; a wrong shape returns None (scalar
  // fallback), never an out-of-bounds index.
  assert!(FilterPaddedSpans::build(&[0, 4], &[0, 4], &[1.0f32; 4]).is_none());
}

#[test]
#[should_panic(expected = "padded arena shape")]
fn dispatcher_panics_on_plan_mismatch() {
  // An arena built for a one-span plan must not bind to a two-span call:
  // kernel h_tmp writes would no longer be covered by the outer bound
  // assert.
  let p = arena(&[0], &[0, 4], &[1.0f32; 4]);
  let row = [0u8; 16];
  let coeffs = [1.0f32; 8];
  let mut h_tmp = [0.0f64; 2];
  filter_h_reduce_row(
    &row,
    1,
    &[0, 4],
    &[0, 4, 8],
    &coeffs,
    Some(&p),
    &mut h_tmp,
    true,
  );
}

#[test]
#[should_panic(expected = "padded arena exceeds row")]
fn dispatcher_panics_on_row_shorter_than_reach() {
  // A 4-tap span against a 3-cell row: rejected before any kernel index
  // arithmetic could wrap.
  let p = arena(&[0], &[0, 4], &[1.0f32; 4]);
  let row = [0u8; 3];
  let coeffs = [1.0f32; 4];
  let mut h_tmp = [0.0f64; 1];
  filter_h_reduce_row(&row, 1, &[0], &[0, 4], &coeffs, Some(&p), &mut h_tmp, true);
}

// ---- Per-backend SIMD-vs-scalar parity ------------------------------------
//
// The H-pass dispatcher routes to whichever SIMD tier the host exposes
// (NEON on aarch64; SSE4.1 / AVX2 / AVX-512 on x86_64; simd128 on wasm —
// each gated by its `*_available()` probe, with the x86 highest-tier-wins
// cascade). These tests pin its output against the scalar reference for
// every element type (u8 / u16 / f32) and both channel counts. The kernels
// widen each sample and coefficient to `f64` exactly, so the only departure
// from the sequential scalar reference is the tap-sum order — float addition
// does not associate — hence a tight *relative* tolerance, not bit-equality.
// The V-pass is element-wise (mul+add, no reorder), so it must match scalar
// bit-for-bit. The geometry ladder covers single-chunk spans with tails,
// multi-chunk spans, the row-end staging path (the last window hugging the
// row end), and signed (negative-lobe) coefficients.

/// Deterministic LCG byte stream (matches the resample suite's `lcg_bytes`).
fn lcg(n: usize, seed: u32) -> std::vec::Vec<u8> {
  let mut state = seed;
  let mut out = std::vec![0u8; n];
  for b in &mut out {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
  out
}

/// Builds a filter span set tiling a `cells`-wide source: `out` windows of
/// `taps` signed coefficients each, the last hugging the row end so the
/// kernels exercise their zero-filled staging copy. Returns
/// `(starts, offsets, coeffs)` in the dispatcher's arena-input shape.
fn spans(
  cells: usize,
  out: usize,
  taps: usize,
  seed: u32,
) -> (
  std::vec::Vec<usize>,
  std::vec::Vec<usize>,
  std::vec::Vec<f32>,
) {
  assert!(taps <= cells, "window cannot exceed the row");
  let raw = lcg(out * taps, seed);
  let mut starts = std::vec::Vec::with_capacity(out);
  let mut offsets = std::vec::Vec::with_capacity(out + 1);
  let mut coeffs = std::vec::Vec::with_capacity(out * taps);
  offsets.push(0);
  for j in 0..out {
    // Spread the windows across the row; clamp the last so `start + taps`
    // lands exactly on the row end (the staged-load boundary).
    let span = cells - taps;
    let start = if out > 1 { span * j / (out - 1) } else { 0 };
    starts.push(start);
    for k in 0..taps {
      // Signed coefficients with a negative lobe, roughly unit-sum so the
      // f64 magnitudes resemble a normalized reconstruction window.
      let v = f32::from(raw[j * taps + k]) / 255.0 - 0.4;
      coeffs.push(v);
    }
    offsets.push(coeffs.len());
  }
  (starts, offsets, coeffs)
}

/// Largest relative deviation between two `f64` slices (absolute floor for
/// near-zero outputs from cancelling signed lobes).
fn max_rel(a: &[f64], b: &[f64]) -> f64 {
  assert_eq!(a.len(), b.len());
  let mut worst = 0.0f64;
  for (&x, &y) in a.iter().zip(b.iter()) {
    let d = (x - y).abs() / (x.abs().max(y.abs()).max(1e-12));
    worst = worst.max(d);
  }
  worst
}

/// Drives the H-pass differential for one element type. `to_elem` maps the
/// LCG bytes into the element's value domain.
fn check_h_pass<S, F>(to_elem: F)
where
  S: FilterSimdElem,
  F: Fn(u8) -> S,
{
  // (cells, out, taps): single-chunk+tail (6/7 taps), multi-chunk (20),
  // row-end staging (taps == cells boundary cases), padded-8-multiple
  // remainders.
  let cases: &[(usize, usize, usize)] = &[
    (256, 40, 6),
    (256, 40, 7),
    (300, 17, 20),
    (64, 7, 9),
    (40, 5, 8),
    (40, 5, 16),
    (33, 4, 33),
    (9, 3, 9),
  ];
  for &(cells, out, taps) in cases {
    let (starts, offsets, coeffs) = spans(cells, out, taps, (cells * 7 + out * 13 + taps) as u32);
    let arena = arena(&starts, &offsets, &coeffs);
    for &channels in &[1usize, 3usize] {
      let raw = lcg(cells * channels, (cells + out + taps + channels) as u32 + 1);
      let row: std::vec::Vec<S> = raw.iter().map(|&b| to_elem(b)).collect();
      let mut h_simd = std::vec![0.0f64; out * channels];
      let mut h_scalar = std::vec![0.0f64; out * channels];
      filter_h_reduce_row(
        &row,
        channels,
        &starts,
        &offsets,
        &coeffs,
        Some(&arena),
        &mut h_simd,
        true,
      );
      // Scalar reference (use_simd == false routes to it on every host).
      filter_h_reduce_row(
        &row,
        channels,
        &starts,
        &offsets,
        &coeffs,
        Some(&arena),
        &mut h_scalar,
        false,
      );
      let rel = max_rel(&h_scalar, &h_simd);
      assert!(
        rel <= 1e-9,
        "H-pass reorder too large: cells={cells} out={out} taps={taps} c={channels} rel={rel}"
      );
    }
  }
}

#[test]
fn h_pass_u8_simd_matches_scalar() {
  check_h_pass::<u8, _>(|b| b);
}

#[test]
fn h_pass_u16_simd_matches_scalar() {
  // Spread across the full 16-bit range so the high bits a u8 path drops
  // are widened and summed.
  check_h_pass::<u16, _>(|b| (u16::from(b) << 8) | u16::from(b));
}

#[test]
fn h_pass_f32_simd_matches_scalar() {
  check_h_pass::<f32, _>(|b| f32::from(b) + f32::from(b) / 256.0);
}

#[test]
fn v_pass_simd_matches_scalar_bit_exact() {
  // Element-wise mul+add — no reorder — so the SIMD V-pass equals the
  // scalar reference bit-for-bit for every length (including the scalar
  // tail past the vector width) and signed weight.
  let raw = lcg(257, 0x5EED);
  let h_tmp: std::vec::Vec<f64> = raw
    .iter()
    .enumerate()
    .map(|(i, &b)| (f64::from(b) - 110.0) * (1.0 + i as f64 / 64.0))
    .collect();
  for &w in &[-0.37f32, 0.0, 0.51, 1.0, -1.83] {
    for n in [0usize, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 255, 257] {
      let n = n.min(h_tmp.len());
      let mut a_simd = std::vec![0.0f64; n];
      let mut a_scalar = std::vec![0.0f64; n];
      // Seed both accumulators identically so the AXPY adds onto real state.
      for (k, slot) in a_simd.iter_mut().enumerate() {
        *slot = f64::from(k as u32) * 0.25 - 3.0;
      }
      a_scalar.copy_from_slice(&a_simd);
      filter_v_accumulate(&mut a_simd, &h_tmp[..n], w, true);
      filter_v_accumulate(&mut a_scalar, &h_tmp[..n], w, false);
      assert_eq!(a_simd, a_scalar, "V-pass diverged: n={n} w={w}");
    }
  }
}
