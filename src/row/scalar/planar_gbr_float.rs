//! Scalar f32-source and f16-source kernels for planar GBR float formats.
//!
//! Source planes are `&[f32]` (or `&[half::f16]` widened to f32 at dispatch).
//! Nominal sample range `[0.0, 1.0]`; HDR values > 1.0 are permitted and
//! handled as follows:
//!
//! - **Integer-output paths** (`*_u8`, `*_u16`): clamped via
//!   `.clamp(0.0, 1.0)` before scaling. NaN clamps to 1.0 via IEEE
//!   `f32::min` fold.
//! - **Lossless float-output paths** (`*_f32`, f16 narrow paths): HDR, NaN,
//!   and Inf are preserved bit-exact (`gbrpf32_to_rgb_f32_row`,
//!   `gbrpf32_to_rgba_f32_row`).
//! - **f16-output paths** (`*_f16`): HDR values exceeding the f16 maximum
//!   (~65504) saturate to `f16::INFINITY` / `f16::NEG_INFINITY`. This is the
//!   documented caller-visible behaviour; callers needing full HDR range use
//!   the f32 pass-through accessors.
//!
//! # Rounding (float → integer)
//!
//! `(y.clamp(0.0, 1.0) * scale + 0.5) as T`
//!
//! Adding 0.5 before truncation gives round-to-nearest (ties round up),
//! MXCSR-independent. Matches the Grayf32 and Rgbf32 scalar contracts.
//!
//! # f32 → f16 rounding
//!
//! IEEE-754 round-to-nearest-even via `half::f16::from_f32` (the `half`
//! crate default). No override needed.
//!
//! # Channel reorder
//!
//! FFmpeg planar GBR stores planes in **G, B, R** order, but the packed
//! output convention is **R, G, B** (matching `AV_PIX_FMT_RGB24`). Every
//! kernel performs this reorder.

// Kernels are not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#![cfg_attr(not(test), allow(dead_code))]

use crate::ColorMatrix;

// ---- shared helpers --------------------------------------------------------

/// Round-to-nearest f32 → u8, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 255, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u8(y: f32) -> u8 {
  (y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Round-to-nearest f32 → u16, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 65535, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u16(y: f32) -> u16 {
  (y.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// f32 → half::f16 via IEEE-754 round-to-nearest-even.
/// HDR values exceeding f16 max (~65504) saturate to ±Inf.
#[inline(always)]
fn f32_to_f16(y: f32) -> half::f16 {
  half::f16::from_f32(y)
}

// ---- Gbrpf32 → u8 RGB ------------------------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **bytes**.
///
/// Each f32 sample is clamped to `[0.0, 1.0]` and scaled to `[0, 255]`
/// with round-half-up. Output order is **R, G, B** per pixel.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = f32_to_u8(r[x]);
    rgb_out[dst + 1] = f32_to_u8(g[x]);
    rgb_out[dst + 2] = f32_to_u8(b[x]);
  }
}

// ---- Gbrpf32 → u8 RGBA (opaque α) -----------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **bytes**
/// with constant opaque α = `0xFF`. Used for `Gbrpf32` sources (no α plane).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u8(r[x]);
    rgba_out[dst + 1] = f32_to_u8(g[x]);
    rgba_out[dst + 2] = f32_to_u8(b[x]);
    rgba_out[dst + 3] = 0xFF;
  }
}

// ---- Gbrpf32 → u16 RGB -----------------------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **`u16`**.
///
/// Each f32 sample is clamped to `[0.0, 1.0]` and scaled to `[0, 65535]`
/// with round-half-up (full-range).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_u16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = f32_to_u16(r[x]);
    rgb_out[dst + 1] = f32_to_u16(g[x]);
    rgb_out[dst + 2] = f32_to_u16(b[x]);
  }
}

// ---- Gbrpf32 → u16 RGBA (opaque α) ----------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **`u16`**
/// with constant opaque α = `0xFFFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_u16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u16(r[x]);
    rgba_out[dst + 1] = f32_to_u16(g[x]);
    rgba_out[dst + 2] = f32_to_u16(b[x]);
    rgba_out[dst + 3] = 0xFFFF;
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) ------------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **`f32`**.
///
/// Lossless interleave — no clamping, no rounding. HDR values > 1.0,
/// NaN, and Inf are preserved bit-exact.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f32_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = r[x];
    rgb_out[dst + 1] = g[x];
    rgb_out[dst + 2] = b[x];
  }
}

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) --------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **`f32`**
/// with α = `1.0` (opaque). Lossless — HDR, NaN, and Inf preserved.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f32_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = r[x];
    rgba_out[dst + 1] = g[x];
    rgba_out[dst + 2] = b[x];
    rgba_out[dst + 3] = 1.0;
  }
}

// ---- Gbrpf32 → f16 RGB (fused narrow + interleave) -------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **`half::f16`**.
///
/// Fused planar-gather, IEEE-754 round-to-nearest-even f32→f16 narrow, and
/// interleave in a single pass. HDR values exceeding the f16 maximum (~65504)
/// saturate to `half::f16::INFINITY`. Callers needing full HDR range use
/// `gbrpf32_to_rgb_f32_row` instead.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = f32_to_f16(r[x]);
    rgb_out[dst + 1] = f32_to_f16(g[x]);
    rgb_out[dst + 2] = f32_to_f16(b[x]);
  }
}

// ---- Gbrpf32 → f16 RGBA (fused narrow, α = f16(1.0)) ----------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **`half::f16`**
/// with α = `half::f16::from_f32(1.0)`. HDR > ~65504 saturates to f16 ±Inf.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let one_f16 = half::f16::from_f32(1.0);
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_f16(r[x]);
    rgba_out[dst + 1] = f32_to_f16(g[x]);
    rgba_out[dst + 2] = f32_to_f16(b[x]);
    rgba_out[dst + 3] = one_f16;
  }
}

// ---- Gbrpf32 → u8 luma (staged via RGB scratch) ----------------------------

/// Derives luma (Y') from planar G/B/R `f32` rows by staging through an
/// 8-bit packed-RGB scratch buffer in chunks of up to 64 pixels.
///
/// The intermediate u8 RGB uses round-half-up clamping; luma is then computed
/// by `rgb_to_luma_row`. `matrix` and `full_range` control the luma weighting.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  luma_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    gbrpf32_to_rgb_row(
      &g[offset..],
      &b[offset..],
      &r[offset..],
      &mut scratch[..n * 3],
      n,
    );
    super::rgb_to_luma_row(
      &scratch[..n * 3],
      &mut luma_out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → u16 luma (staged via RGB scratch) ---------------------------

/// Derives luma (Y') in `u16` from planar G/B/R `f32` rows by staging through
/// an 8-bit packed-RGB scratch buffer in chunks of up to 64 pixels.
///
/// The u16 luma value has the same dynamic range as the u8 path (0–255), zero-
/// extended into the u16 carrier — matching the convention of packed-YUV
/// `*_to_luma_u16_row` kernels for 8-bit-equivalent sources.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_u16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  luma_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    gbrpf32_to_rgb_row(
      &g[offset..],
      &b[offset..],
      &r[offset..],
      &mut scratch[..n * 3],
      n,
    );
    super::rgb_to_luma_u16_row(
      &scratch[..n * 3],
      &mut luma_out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → HSV (staged via RGB scratch) --------------------------------

/// Converts planar G/B/R `f32` rows to planar HSV **bytes** by staging
/// through an 8-bit packed-RGB scratch buffer in chunks of up to 64 pixels.
///
/// Matches OpenCV `cv2.COLOR_RGB2HSV` semantics: `H ∈ [0, 179]`, `S, V ∈
/// [0, 255]`. f32 values are clamped via `f32_to_u8` before the RGB→HSV step.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_hsv_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    gbrpf32_to_rgb_row(
      &g[offset..],
      &b[offset..],
      &r[offset..],
      &mut scratch[..n * 3],
      n,
    );
    super::rgb_to_hsv_row(
      &scratch[..n * 3],
      &mut h_out[offset..offset + n],
      &mut s_out[offset..offset + n],
      &mut v_out[offset..offset + n],
      n,
    );
    offset += n;
  }
}

// ---- Gbrapf32 → u8 RGBA (source α) ----------------------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A` **bytes**.
///
/// α is sourced from the `a` plane: clamped to `[0.0, 1.0]` and scaled by 255
/// with round-half-up.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u8(r[x]);
    rgba_out[dst + 1] = f32_to_u8(g[x]);
    rgba_out[dst + 2] = f32_to_u8(b[x]);
    rgba_out[dst + 3] = f32_to_u8(a[x]);
  }
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A` **`u16`**.
///
/// α is sourced from the `a` plane: clamped to `[0.0, 1.0]` and scaled by
/// 65535 with round-half-up.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_u16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u16(r[x]);
    rgba_out[dst + 1] = f32_to_u16(g[x]);
    rgba_out[dst + 2] = f32_to_u16(b[x]);
    rgba_out[dst + 3] = f32_to_u16(a[x]);
  }
}

// ---- Gbrapf32 → f32 RGBA (lossless source α) --------------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A` **`f32`**.
///
/// Lossless — HDR, NaN, and Inf are preserved bit-exact in all four channels
/// including α.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f32_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = r[x];
    rgba_out[dst + 1] = g[x];
    rgba_out[dst + 2] = b[x];
    rgba_out[dst + 3] = a[x];
  }
}

// ---- Gbrapf32 → f16 RGBA (fused narrow, source α) ---------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A`
/// **`half::f16`** with source α.
///
/// Fused narrow: all four channels converted via IEEE-754 round-to-nearest-even
/// in a single pass. HDR > ~65504 saturates to f16 ±Inf.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_f16(r[x]);
    rgba_out[dst + 1] = f32_to_f16(g[x]);
    rgba_out[dst + 2] = f32_to_f16(b[x]);
    rgba_out[dst + 3] = f32_to_f16(a[x]);
  }
}

// ---- Unit tests ------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // ---- gbrpf32_to_rgb_row --------------------------------------------------

  #[test]
  fn gbrpf32_to_rgb_clamps_and_scales() {
    // Values: 0.0, 0.5, 1.0, 1.5, -0.1 → 0, 128, 255, 255, 0
    // All three channels use the same value for simplicity.
    let vals = [0.0f32, 0.5, 1.0, 1.5, -0.1];
    let expected = [0u8, 128, 255, 255, 0];
    for (v, e) in vals.iter().zip(expected.iter()) {
      let g = [*v; 1];
      let b = [*v; 1];
      let r = [*v; 1];
      let mut out = [0u8; 3];
      gbrpf32_to_rgb_row(&g, &b, &r, &mut out, 1);
      assert_eq!(out[0], *e, "R: v={v}, expected={e}");
      assert_eq!(out[1], *e, "G: v={v}, expected={e}");
      assert_eq!(out[2], *e, "B: v={v}, expected={e}");
    }
  }

  #[test]
  fn gbrpf32_to_rgb_channel_reorder() {
    // G=0.0, B=0.5, R=1.0 → packed R=255, G=0, B=128
    let g = [0.0f32];
    let b = [0.5f32];
    let r = [1.0f32];
    let mut out = [0u8; 3];
    gbrpf32_to_rgb_row(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], 255, "R");
    assert_eq!(out[1], 0, "G");
    assert_eq!(out[2], 128, "B");
  }

  // ---- gbrpf32_to_rgba_row -------------------------------------------------

  #[test]
  fn gbrpf32_to_rgba_fills_alpha_max() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let mut out = [0u8; 4];
    gbrpf32_to_rgba_row(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], 0xFF, "alpha must be 0xFF");
  }

  #[test]
  fn gbrpf32_to_rgba_clamps_and_scales() {
    let vals = [0.0f32, 0.5, 1.0, 1.5, -0.1];
    let expected = [0u8, 128, 255, 255, 0];
    for (v, e) in vals.iter().zip(expected.iter()) {
      let g = [*v; 1];
      let b = [*v; 1];
      let r = [*v; 1];
      let mut out = [0u8; 4];
      gbrpf32_to_rgba_row(&g, &b, &r, &mut out, 1);
      assert_eq!(out[0], *e, "R: v={v}");
      assert_eq!(out[3], 0xFF, "alpha must remain 0xFF");
    }
  }

  // ---- gbrpf32_to_rgb_u16_row ----------------------------------------------

  #[test]
  fn gbrpf32_to_rgb_u16_clamps_and_scales() {
    let vals = [0.0f32, 0.5, 1.0, 1.5, -0.1];
    // 0.5 → (0.5 * 65535 + 0.5) as u16 = 32768
    let expected = [0u16, 32768, 65535, 65535, 0];
    for (v, e) in vals.iter().zip(expected.iter()) {
      let g = [*v; 1];
      let b = [*v; 1];
      let r = [*v; 1];
      let mut out = [0u16; 3];
      gbrpf32_to_rgb_u16_row(&g, &b, &r, &mut out, 1);
      assert_eq!(out[0], *e, "R u16: v={v}");
      assert_eq!(out[1], *e, "G u16: v={v}");
      assert_eq!(out[2], *e, "B u16: v={v}");
    }
  }

  // ---- gbrpf32_to_rgba_u16_row ---------------------------------------------

  #[test]
  fn gbrpf32_to_rgba_u16_fills_alpha_max() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let mut out = [0u16; 4];
    gbrpf32_to_rgba_u16_row(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }

  // ---- gbrpf32_to_rgb_f32_row (lossless) ------------------------------------

  #[test]
  fn gbrpf32_to_rgb_f32_lossless_passthrough() {
    // HDR 2.5, NaN, Inf, negative all preserved bit-exact.
    let g = [2.5f32, f32::NAN, f32::INFINITY, -1.0];
    let b = [0.1f32, 0.2, 0.3, 0.4];
    let r = [0.5f32, 0.6, 0.7, 0.8];
    let mut out = [0.0f32; 12];
    gbrpf32_to_rgb_f32_row(&g, &b, &r, &mut out, 4);
    // Check R channel (index 0, 3, 6, 9 in RGBA interleave = index 0, 3, 6, 9)
    assert_eq!(out[0], r[0]);
    assert_eq!(out[3], r[1]);
    assert_eq!(out[6], r[2]);
    assert_eq!(out[9], r[3]);
    // Check G channel (index 1, 4, 7, 10)
    assert_eq!(out[1], g[0], "G HDR preserved");
    assert!(out[4].is_nan(), "G NaN preserved");
    assert!(out[7].is_infinite() && out[7] > 0.0, "G +Inf preserved");
    assert_eq!(out[10], g[3], "G negative preserved");
  }

  // ---- gbrpf32_to_rgba_f32_row (lossless, α = 1.0) -------------------------

  #[test]
  fn gbrpf32_to_rgba_f32_alpha_is_one() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let mut out = [0.0f32; 4];
    gbrpf32_to_rgba_f32_row(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], 1.0, "alpha must be 1.0");
  }

  #[test]
  fn gbrpf32_to_rgba_f32_lossless_passthrough() {
    let r = [2.5f32];
    let g = [f32::NAN];
    let b = [f32::NEG_INFINITY];
    let mut out = [0.0f32; 4];
    gbrpf32_to_rgba_f32_row(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], 2.5, "R HDR preserved");
    assert!(out[1].is_nan(), "G NaN preserved");
    assert!(out[2].is_infinite() && out[2] < 0.0, "B -Inf preserved");
    assert_eq!(out[3], 1.0, "alpha = 1.0");
  }

  // ---- gbrpf32_to_rgb_f16_row ----------------------------------------------

  #[test]
  fn gbrpf32_to_rgb_f16_normal_values() {
    let g = [0.0f32, 0.5, 1.0];
    let b = [0.25f32, 0.75, 0.0];
    let r = [1.0f32, 0.0, 0.5];
    let mut out = vec![half::f16::ZERO; 9];
    gbrpf32_to_rgb_f16_row(&g, &b, &r, &mut out, 3);
    assert_eq!(out[0], half::f16::from_f32(1.0), "R[0]");
    assert_eq!(out[1], half::f16::from_f32(0.0), "G[0]");
    assert_eq!(out[2], half::f16::from_f32(0.25), "B[0]");
  }

  #[test]
  fn gbrpf32_to_rgb_f16_hdr_saturates_to_inf() {
    // Input 70000.0 > f16 max (~65504) → +Inf
    let g = [70_000.0f32];
    let b = [-70_000.0f32];
    let r = [0.5f32];
    let mut out = vec![half::f16::ZERO; 3];
    gbrpf32_to_rgb_f16_row(&g, &b, &r, &mut out, 1);
    // G maps to index 1
    assert!(out[1].is_infinite() && out[1].to_f32() > 0.0, "G +Inf");
    // B maps to index 2
    assert!(out[2].is_infinite() && out[2].to_f32() < 0.0, "B -Inf");
  }

  // ---- gbrpf32_to_rgba_f16_row ---------------------------------------------

  #[test]
  fn gbrpf32_to_rgba_f16_alpha_is_one() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let mut out = vec![half::f16::ZERO; 4];
    gbrpf32_to_rgba_f16_row(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], half::f16::from_f32(1.0), "alpha must be f16(1.0)");
  }

  // ---- gbrpf32_to_luma_row -------------------------------------------------

  #[test]
  fn gbrpf32_to_luma_zero_gives_zero() {
    let g = [0.0f32];
    let b = [0.0f32];
    let r = [0.0f32];
    let mut out = [0xFFu8; 1];
    gbrpf32_to_luma_row(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert_eq!(out[0], 0);
  }

  #[test]
  fn gbrpf32_to_luma_max_gives_255() {
    let g = [1.0f32];
    let b = [1.0f32];
    let r = [1.0f32];
    let mut out = [0u8; 1];
    gbrpf32_to_luma_row(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert_eq!(out[0], 255);
  }

  // ---- gbrpf32_to_luma_u16_row ---------------------------------------------

  #[test]
  fn gbrpf32_to_luma_u16_zero_gives_zero() {
    let g = [0.0f32];
    let b = [0.0f32];
    let r = [0.0f32];
    let mut out = [0xFFFFu16; 1];
    gbrpf32_to_luma_u16_row(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert_eq!(out[0], 0);
  }

  #[test]
  fn gbrpf32_to_luma_u16_max_gives_255_zero_extended() {
    let g = [1.0f32];
    let b = [1.0f32];
    let r = [1.0f32];
    let mut out = [0u16; 1];
    gbrpf32_to_luma_u16_row(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert_eq!(out[0], 255, "luma_u16 is zero-extended u8 luma");
  }

  // ---- gbrpf32_to_hsv_row --------------------------------------------------

  #[test]
  fn gbrpf32_to_hsv_achromatic_black() {
    let g = [0.0f32];
    let b = [0.0f32];
    let r = [0.0f32];
    let mut h = [0xFFu8; 1];
    let mut s = [0xFFu8; 1];
    let mut v = [0xFFu8; 1];
    gbrpf32_to_hsv_row(&g, &b, &r, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 0, "V must be 0 for black");
    assert_eq!(s[0], 0, "S must be 0 for achromatic");
  }

  #[test]
  fn gbrpf32_to_hsv_achromatic_white() {
    let g = [1.0f32];
    let b = [1.0f32];
    let r = [1.0f32];
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    gbrpf32_to_hsv_row(&g, &b, &r, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 255, "V must be 255 for white");
    assert_eq!(s[0], 0, "S must be 0 for achromatic");
  }

  // ---- gbrapf32_to_rgba_row ------------------------------------------------

  #[test]
  fn gbrapf32_to_rgba_source_alpha_passthrough() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let a = [0.5f32];
    let mut out = [0u8; 4];
    gbrapf32_to_rgba_row(&g, &b, &r, &a, &mut out, 1);
    // 0.5 → (0.5 * 255 + 0.5) as u8 = 128
    assert_eq!(out[3], 128, "alpha from source plane");
  }

  #[test]
  fn gbrapf32_to_rgba_source_alpha_clamps() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    // Test α > 1.0 → 255 and α < 0.0 → 0
    let a_high = [1.5f32];
    let a_low = [-0.1f32];
    let mut out_high = [0u8; 4];
    let mut out_low = [0u8; 4];
    gbrapf32_to_rgba_row(&g, &b, &r, &a_high, &mut out_high, 1);
    gbrapf32_to_rgba_row(&g, &b, &r, &a_low, &mut out_low, 1);
    assert_eq!(out_high[3], 255, "alpha HDR clamps to 255");
    assert_eq!(out_low[3], 0, "alpha negative clamps to 0");
  }

  // ---- gbrapf32_to_rgba_u16_row --------------------------------------------

  #[test]
  fn gbrapf32_to_rgba_u16_source_alpha_passthrough() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let a = [0.5f32];
    let mut out = [0u16; 4];
    gbrapf32_to_rgba_u16_row(&g, &b, &r, &a, &mut out, 1);
    // 0.5 → (0.5 * 65535 + 0.5) as u16 = 32768
    assert_eq!(out[3], 32768, "u16 alpha from source plane");
  }

  #[test]
  fn gbrapf32_to_rgba_u16_source_alpha_clamps() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let a_high = [1.5f32];
    let a_low = [-0.1f32];
    let mut out_high = [0u16; 4];
    let mut out_low = [0u16; 4];
    gbrapf32_to_rgba_u16_row(&g, &b, &r, &a_high, &mut out_high, 1);
    gbrapf32_to_rgba_u16_row(&g, &b, &r, &a_low, &mut out_low, 1);
    assert_eq!(out_high[3], 65535, "u16 alpha HDR clamps to 65535");
    assert_eq!(out_low[3], 0, "u16 alpha negative clamps to 0");
  }

  // ---- gbrapf32_to_rgba_f32_row (lossless source α) -------------------------

  #[test]
  fn gbrapf32_to_rgba_f32_lossless_passthrough() {
    // HDR 2.5, NaN, Inf, negative all preserved — including in α
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let a = [2.5f32];
    let mut out = [0.0f32; 4];
    gbrapf32_to_rgba_f32_row(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[3], 2.5, "HDR alpha preserved bit-exact");
  }

  #[test]
  fn gbrapf32_to_rgba_f32_nan_alpha_preserved() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let a = [f32::NAN];
    let mut out = [0.0f32; 4];
    gbrapf32_to_rgba_f32_row(&g, &b, &r, &a, &mut out, 1);
    assert!(out[3].is_nan(), "NaN alpha preserved");
  }

  // ---- gbrapf32_to_rgba_f16_row --------------------------------------------

  #[test]
  fn gbrapf32_to_rgba_f16_source_alpha_passthrough() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let a = [0.75f32];
    let mut out = vec![half::f16::ZERO; 4];
    gbrapf32_to_rgba_f16_row(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[3], half::f16::from_f32(0.75), "f16 alpha from source");
  }

  #[test]
  fn gbrapf32_to_rgba_f16_hdr_alpha_saturates() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let a = [70_000.0f32];
    let mut out = vec![half::f16::ZERO; 4];
    gbrapf32_to_rgba_f16_row(&g, &b, &r, &a, &mut out, 1);
    assert!(
      out[3].is_infinite() && out[3].to_f32() > 0.0,
      "HDR alpha saturates to +Inf"
    );
  }
}
