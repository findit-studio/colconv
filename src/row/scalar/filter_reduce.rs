//! Scalar reference for the separable **filter** resampler's H/V passes —
//! the signed twin of [`area_reduce`](super::area_reduce).
//!
//! Each output sample is a dot product of **signed** `f32` coefficients
//! over its source window. Unlike the integer area engine the sums are
//! not exact: every backend accumulates in `f64` and matches this
//! reference within a small tolerance — the integer streams end up `±1`
//! LSB of Pillow (the user-accepted contract), not bit-for-bit. The
//! H-pass leaves the **raw `f64` dot product** in `h_tmp`; the stream
//! quantizes it to PIL's two-pass intermediate per element type
//! afterwards (`quantize_intermediate`), so the H-pass itself only
//! differs per element by how a sample widens to `f64`.
//!
//! The coefficient window is the `FilterStream`'s `h.span(ox)` slice; the
//! SIMD arena pads each window to a lane multiple with zero coefficients
//! (zero annihilates), so the kernels run pure wide loads while still
//! summing exactly the real taps. The V-pass is element-wise over the
//! already-`f64` `h_tmp`, so it has a single form shared by every element.

#![cfg_attr(
  any(not(feature = "std"), not(any(feature = "rgb", feature = "gray"))),
  allow(dead_code)
)]

/// One element type the filter H-pass widens to its `f64` accumulation
/// domain. The integer variants widen losslessly; `f32` widens exactly.
/// Mirrors the per-element split of the area H-pass references, kept as a
/// trait so the c1/c3 bodies are written once.
pub(crate) trait FilterElem: Copy {
  /// Promote one source sample to the `f64` accumulation domain.
  fn widen(self) -> f64;
}

impl FilterElem for u8 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn widen(self) -> f64 {
    f64::from(self)
  }
}

impl FilterElem for u16 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn widen(self) -> f64 {
    f64::from(self)
  }
}

impl FilterElem for f32 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn widen(self) -> f64 {
    f64::from(self)
  }
}

/// `h_tmp[j] = Σ coeffs[k] * row[start_j + k]` (1 channel), the raw `f64`
/// dot product over output span `j`'s signed `f32` window. `starts[j]` is
/// the window's first source sample, `offsets` its prefix bounds into
/// `coeffs`. The sample widens to `f64` per [`FilterElem`].
pub(crate) fn filter_h_reduce_row_c1<S: FilterElem>(
  row: &[S],
  starts: &[usize],
  offsets: &[usize],
  coeffs: &[f32],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &coeffs[offsets[j]..offsets[j + 1]];
    let mut acc = 0.0f64;
    for (k, &w) in span.iter().enumerate() {
      acc += f64::from(w) * row[start + k].widen();
    }
    h_tmp[j] = acc;
  }
}

/// 3-channel (interleaved) variant: each window's signed `f32` coeffs are
/// shared across the three channels, with sample `(start + k)` read at
/// `(start + k) * 3 + ch`.
pub(crate) fn filter_h_reduce_row_c3<S: FilterElem>(
  row: &[S],
  starts: &[usize],
  offsets: &[usize],
  coeffs: &[f32],
  h_tmp: &mut [f64],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &coeffs[offsets[j]..offsets[j + 1]];
    let base = j * 3;
    let mut acc0 = 0.0f64;
    let mut acc1 = 0.0f64;
    let mut acc2 = 0.0f64;
    for (k, &w) in span.iter().enumerate() {
      let wf = f64::from(w);
      let cell = (start + k) * 3;
      acc0 += wf * row[cell].widen();
      acc1 += wf * row[cell + 1].widen();
      acc2 += wf * row[cell + 2].widen();
    }
    h_tmp[base] = acc0;
    h_tmp[base + 1] = acc1;
    h_tmp[base + 2] = acc2;
  }
}

/// Per-lane keep mask for the trailing 8-lane chunk of a padded span:
/// lane `k` is all-ones bits (`f64::from_bits(!0)`) when it is a **real**
/// tap (`k < real`) and `+0.0` when it is arena padding (`k >= real`).
/// The SIMD kernels AND this onto their sample lanes so a padding lane is
/// forced to `+0.0` (annihilating its `0.0` coefficient even for a
/// non-finite sample), while a real lane — **including a real zero
/// coefficient** — keeps its sample so `0.0 * NaN == NaN` survives exactly
/// as this reference computes it. Only the last chunk of a span is ever
/// partial (`real` in `1..=7`); full chunks skip the mask entirely.
#[cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn padding_keep_mask8(real: usize) -> [f64; 8] {
  let mut m = [0.0f64; 8];
  for (k, lane) in m.iter_mut().enumerate() {
    // `(k < real) as u64` is 1/0; negating spreads it to all-ones / zero.
    *lane = f64::from_bits((u64::from(k < real)).wrapping_neg());
  }
  m
}

/// V-pass AXPY: `acc[i] += w * h_tmp[i]` over the H-reduced (and
/// quantized) row, in `f64`. `w` is one signed `f32` vertical
/// coefficient. Element-wise — no reordering — so every backend matches
/// this bit-for-bit (only the H-pass carries the float tolerance).
pub(crate) fn filter_v_accumulate(acc: &mut [f64], h_tmp: &[f64], w: f32) {
  let wf = f64::from(w);
  for (a, t) in acc.iter_mut().zip(h_tmp.iter()) {
    *a += wf * *t;
  }
}
