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
  // The real tap count per span — the lane boundary the kernels mask
  // beyond (a 4-tap and a 9-tap span).
  assert_eq!(p.ksize, std::vec![4, 9]);
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

// ---- f32 non-finite samples under REAL zero coefficients ------------------
//
// The arena pads every span to an 8-multiple with zero coefficients, so the
// kernels run pure wide loads. The mask must neutralize only those PADDING
// lanes — not a REAL zero coefficient the plan legitimately produced (a
// normalized window can carry an exact 0.0 tap). With f32 input the scalar
// reference computes `0.0 * NaN == NaN` for such a tap; masking the sample
// to 0 there would yield a finite value, breaking scalar-vs-SIMD parity for
// non-finite inputs. These tests place NaN / ±inf samples directly under
// real zero coefficients and assert the SIMD H-pass matches scalar
// bit-for-bit (both NaN, or equal bits).

/// `true` iff `a` and `b` agree on finiteness AND value: both NaN, or both
/// the same infinity, or both finite within a tight tolerance. The crux is
/// the **NaN/inf propagation** — a masked-away real zero coefficient would
/// turn a scalar NaN into a finite SIMD value, which this rejects — while
/// finite lanes keep the H-pass's reorder tolerance (the tap-sum order
/// differs, so finite sums are not bit-identical).
fn same_finiteness_f64(a: f64, b: f64) -> bool {
  if a.is_nan() || b.is_nan() {
    return a.is_nan() && b.is_nan();
  }
  if a.is_infinite() || b.is_infinite() {
    // Same sign of infinity (an inf result must survive as the same inf).
    return a == b;
  }
  let tol = 1e-9 * a.abs().max(b.abs()).max(1.0);
  (a - b).abs() <= tol
}

/// Runs the SIMD and scalar H-pass over the given span set + row, asserting
/// every output lane agrees on finiteness and value (see
/// [`same_finiteness_f64`]). `channels` is 1 or 3.
fn assert_h_parity_f32(
  channels: usize,
  starts: &[usize],
  offsets: &[usize],
  coeffs: &[f32],
  row: &[f32],
  out: usize,
) {
  let p = arena(starts, offsets, coeffs);
  let mut h_simd = std::vec![0.0f64; out * channels];
  let mut h_scalar = std::vec![0.0f64; out * channels];
  filter_h_reduce_row(
    row,
    channels,
    starts,
    offsets,
    coeffs,
    Some(&p),
    &mut h_simd,
    true,
  );
  filter_h_reduce_row(
    row,
    channels,
    starts,
    offsets,
    coeffs,
    Some(&p),
    &mut h_scalar,
    false,
  );
  for (i, (&s, &d)) in h_scalar.iter().zip(h_simd.iter()).enumerate() {
    assert!(
      same_finiteness_f64(s, d),
      "non-finite parity lane {i}: scalar {s} ({:#018x}) simd {d} ({:#018x})",
      s.to_bits(),
      d.to_bits()
    );
  }
}

#[test]
fn f32_real_zero_coeff_over_nonfinite_matches_scalar_c1() {
  // Window of 4 real taps with TWO interior real zero coefficients; the
  // source samples under them are NaN and +inf. Scalar yields NaN (the zero
  // times a non-finite); the fixed SIMD must too (it must NOT mask a real
  // zero-coeff lane). The single span pads to 8 lanes, so lanes 4..8 (the
  // padding) load real row samples 4..8 here — kept finite so the padding
  // mask's job (zeroing them) is unambiguous.
  let starts = [0usize];
  let offsets = [0usize, 4];
  let coeffs = [0.5f32, 0.0, 0.0, 0.5];
  let row = [1.0f32, f32::NAN, f32::INFINITY, 2.0, 3.0, 4.0, 5.0, 6.0];
  assert_h_parity_f32(1, &starts, &offsets, &coeffs, &row, 1);
}

#[test]
fn f32_real_zero_coeff_over_neg_inf_and_padding_nonfinite_c1() {
  // Two windows: the first puts a real zero coefficient over -inf (scalar
  // and SIMD must both go NaN); the second is a finite window whose PADDING
  // lanes (past its 3 real taps) overlap a NaN / +inf just past the window —
  // those the mask MUST neutralize, so this window stays finite and equal.
  let starts = [0usize, 8];
  let offsets = [0usize, 4, 7];
  // Window 0: [0.5, 0.0, 0.25, 0.25] — real zero at tap 1 (over row[1]).
  // Window 1: [0.3, 0.3, 0.4] — 3 real taps (over row[8..11]), pads to 8 so
  // padding lanes cover row[11..16].
  let coeffs = [0.5f32, 0.0, 0.25, 0.25, 0.3, 0.3, 0.4];
  let row = [
    10.0f32,
    f32::NEG_INFINITY, // under window 0's real zero coeff -> poisons (NaN)
    1.0,
    2.0,
    0.0,
    0.0,
    0.0,
    0.0,
    8.0,           // window 1 real tap 0
    9.0,           // window 1 real tap 1
    1.5,           // window 1 real tap 2 (finite — window 1's result stays finite)
    f32::NAN,      // window 1 padding lane (index 3, past its 3 real taps)
    f32::INFINITY, // window 1 padding lane
    0.0,
    0.0,
    0.0,
  ];
  assert_h_parity_f32(1, &starts, &offsets, &coeffs, &row, 2);
}

#[test]
fn f32_real_zero_coeff_over_nonfinite_matches_scalar_c3() {
  // 3-channel interleaved: the shared window has a real zero coefficient at
  // tap 1; pixel 1's three channels are NaN / +inf / -inf, so each channel's
  // dot product poisons to NaN under scalar — the SIMD c3 kernel must agree.
  let starts = [0usize];
  let offsets = [0usize, 3];
  let coeffs = [0.5f32, 0.0, 0.5];
  // 8 pixels x 3 channels; pixel 1 (the zero-coeff tap) is non-finite.
  let mut row = std::vec![0.0f32; 8 * 3];
  for (px, chunk) in row.chunks_exact_mut(3).enumerate() {
    chunk[0] = px as f32 + 1.0;
    chunk[1] = px as f32 + 1.5;
    chunk[2] = px as f32 + 2.0;
  }
  row[3] = f32::NAN; // pixel 1, channel 0
  row[4] = f32::INFINITY; // pixel 1, channel 1
  row[5] = f32::NEG_INFINITY; // pixel 1, channel 2
  assert_h_parity_f32(3, &starts, &offsets, &coeffs, &row, 1);
}

#[test]
fn f32_real_zero_coeff_multichunk_matches_scalar_c1() {
  // A wide window (>8 real taps) so the real zero coefficient lands in a
  // FULL (non-trailing) chunk — exercising the unmasked full-chunk path —
  // plus a trailing partial chunk. The non-finite sample under the real zero
  // must still poison to NaN under both paths.
  let starts = [0usize];
  let offsets = [0usize, 11];
  // 11 taps: a real zero at index 5 (inside the first full 8-lane chunk).
  let coeffs = [0.1f32, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1];
  let mut row = std::vec![1.0f32; 16];
  row[5] = f32::INFINITY; // under the real zero coeff in the full chunk
  assert_h_parity_f32(1, &starts, &offsets, &coeffs, &row, 1);
}
