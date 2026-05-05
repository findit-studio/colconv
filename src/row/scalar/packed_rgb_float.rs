// ---- Tier 9 packed-float-RGB helpers (Rgbf32) -------------------------
//
// Compact float→integer conversion kernels behind the [`Rgbf32`]
// source-side sinker family. Each pixel is `3 × f32` (linear R, G, B).
// HDR values > 1.0 saturate to the output range; values < 0.0 clamp
// to 0.
//
// This file provides the scalar reference / fallback implementations;
// SIMD dispatch lives in `row::dispatch::rgb_float_ops` and per-arch
// backends in `row::arch::*::packed_rgb_float`.
//
// Rounding convention: `(value * out_max).round()` then saturating cast.
// Matches the symmetric error bound the rest of the float→integer
// conversions in this crate use (see `scalar::mod` rounding docs).

/// Round-to-nearest-even (IEEE 754 default rounding mode) for a
/// non-negative `f32` known to fit in `u32` after the cast. This is
/// the reference rounding rule that the SIMD backends' saturating-
/// convert intrinsics use (`vcvtnq_u32_f32` on NEON,
/// `_mm{,256,512}_cvtps_epi32` with default MXCSR on x86,
/// `f32x4_nearest` + `i32x4_trunc_sat_f32x4` on wasm).
///
/// `f32::round` would round half-away-from-zero (e.g. `0.5 → 1`),
/// which diverges from SIMD on exact `.5` inputs. `f32::round_ties_even`
/// matches SIMD but is `std`-only on f32 — `colconv` is `no_std`-capable,
/// so we implement the rule manually using integer arithmetic.
#[cfg_attr(not(tarpaulin), inline(always))]
fn round_ties_even_nonneg(x: f32) -> f32 {
  // After `clamp(0.0, scale)` with `scale ≤ 65535`, `x as u32` is
  // lossless (truncates toward zero, no saturation). NaN→0 also matches
  // the SIMD backends' NaN-handling (NaN is not a valid sample for
  // colour-converted output; clamp would drop it via `min(NaN, 1) = NaN`
  // → `(NaN * 255).round_ties_even() = NaN` → `as u8 = 0` in scalar,
  // and `vcvtnq_u32_f32(NaN) = 0` likewise).
  if !x.is_finite() {
    return 0.0;
  }
  let i = x as u32;
  let frac = x - (i as f32);
  if frac < 0.5 {
    i as f32
  } else if frac > 0.5 {
    (i + 1) as f32
  } else {
    // Exact half — round to even.
    if i & 1 == 0 { i as f32 } else { (i + 1) as f32 }
  }
}

/// Clamps a single `f32` value to `[0, 1]` and scales to a `u8` using
/// round-to-nearest-even — matches every SIMD backend's saturating-
/// convert behavior. NaN → `0`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn f32_to_u8_clamped(v: f32) -> u8 {
  let clamped = v.clamp(0.0, 1.0);
  let scaled = clamped * 255.0;
  round_ties_even_nonneg(scaled) as u8
}

/// Clamps a single `f32` value to `[0, 1]` and scales to a `u16`.
/// See [`f32_to_u8_clamped`] for the rounding-mode rationale.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn f32_to_u16_clamped(v: f32) -> u16 {
  let clamped = v.clamp(0.0, 1.0);
  let scaled = clamped * 65535.0;
  round_ties_even_nonneg(scaled) as u16
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B` `u8`
/// output. Each `f32` is clamped to `[0, 1]` and scaled by 255.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgb_row(rgb_in: &[f32], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 3;
    rgb_out[i] = f32_to_u8_clamped(rgb_in[i]);
    rgb_out[i + 1] = f32_to_u8_clamped(rgb_in[i + 1]);
    rgb_out[i + 2] = f32_to_u8_clamped(rgb_in[i + 2]);
  }
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B, A` `u8`
/// output with `A = 0xFF` (the float source has no alpha).
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgba_row(rgb_in: &[f32], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let s = x * 3;
    let d = x * 4;
    rgba_out[d] = f32_to_u8_clamped(rgb_in[s]);
    rgba_out[d + 1] = f32_to_u8_clamped(rgb_in[s + 1]);
    rgba_out[d + 2] = f32_to_u8_clamped(rgb_in[s + 2]);
    rgba_out[d + 3] = 0xFF;
  }
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B` `u16`
/// output. Each `f32` is clamped to `[0, 1]` and scaled by 65535.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgb_u16_row(rgb_in: &[f32], rgb_out: &mut [u16], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let i = x * 3;
    rgb_out[i] = f32_to_u16_clamped(rgb_in[i]);
    rgb_out[i + 1] = f32_to_u16_clamped(rgb_in[i + 1]);
    rgb_out[i + 2] = f32_to_u16_clamped(rgb_in[i + 2]);
  }
}

/// Converts packed `R, G, B` `f32` input to packed `R, G, B, A` `u16`
/// output with `A = 0xFFFF`.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgba_out.len() < 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgba_u16_row(rgb_in: &[f32], rgba_out: &mut [u16], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let s = x * 3;
    let d = x * 4;
    rgba_out[d] = f32_to_u16_clamped(rgb_in[s]);
    rgba_out[d + 1] = f32_to_u16_clamped(rgb_in[s + 1]);
    rgba_out[d + 2] = f32_to_u16_clamped(rgb_in[s + 2]);
    rgba_out[d + 3] = 0xFFFF;
  }
}

/// **Lossless** float pass-through: copies the packed `R, G, B` `f32`
/// row into the output buffer without conversion. Source HDR values
/// (> 1.0) and negatives are preserved bit-exact.
///
/// # Panics
///
/// Panics (any build profile) if `rgb_in.len() < 3 * width` or
/// `rgb_out.len() < 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgbf32_to_rgb_f32_row(rgb_in: &[f32], rgb_out: &mut [f32], width: usize) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");
  rgb_out[..width * 3].copy_from_slice(&rgb_in[..width * 3]);
}
