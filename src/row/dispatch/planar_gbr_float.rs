//! Runtime SIMD dispatchers for planar GBR float sources.
//!
//! Covers `Gbrpf32` / `Gbrapf32` (f32 element type) and `Gbrpf16` /
//! `Gbrapf16` (half::f16 element type). SIMD backends will be wired in
//! Tasks 3–7; for now every entry calls the scalar kernel directly.
//!
//! `use_simd = false` bypasses any future SIMD cascade and calls scalar
//! directly. Lossless f32-output paths take `_use_simd` (ignored) because
//! they have no SIMD acceleration.
//!
//! # Overflow guards
//!
//! Output-buffer length checks use `rgb_row_bytes` / `rgba_row_bytes` /
//! `rgb_row_elems` / `rgba_row_elems` — the same checked-multiply helpers
//! used throughout the crate. These are hoisted BEFORE plane-bound assertions
//! so a 32-bit overflow surfaces as the documented "overflows usize" panic
//! rather than a passing plane-len check followed by a write past the end of
//! the buffer.
//!
//! # f16-source paths
//!
//! For f16-source → integer / luma / HSV outputs the dispatcher widens each
//! f16 plane to f32 in per-call stack scratch (up to 64 elements/plane,
//! chunked), then calls the corresponding `gbrpf32_to_*` scalar kernel.
//! For f16-source → f16 output the f16-native kernels in
//! [`super::scalar::planar_gbr_f16`] are called directly.

// Dispatchers in this module are not yet consumed by any sinker (Task 8 wires
// the MixedSinker impls). Allow dead_code until then.
#![allow(dead_code)]

use crate::{
  ColorMatrix,
  row::{
    rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems,
    scalar::{planar_gbr_f16 as scalar_f16, planar_gbr_float as scalar},
  },
};

// ---- Gbrpf32 → u8 RGB -------------------------------------------------------

/// Dispatch `gbrpf32_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  let _ = use_simd;
  scalar::gbrpf32_to_rgb_row(g, b, r, out, width);
}

// ---- Gbrpf32 → u8 RGBA ------------------------------------------------------

/// Dispatch `gbrpf32_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  let _ = use_simd;
  scalar::gbrpf32_to_rgba_row(g, b, r, out, width);
}

// ---- Gbrpf32 → u16 RGB ------------------------------------------------------

/// Dispatch `gbrpf32_to_rgb_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_u16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  let _ = use_simd;
  scalar::gbrpf32_to_rgb_u16_row(g, b, r, out, width);
}

// ---- Gbrpf32 → u16 RGBA -----------------------------------------------------

/// Dispatch `gbrpf32_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_u16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  let _ = use_simd;
  scalar::gbrpf32_to_rgba_u16_row(g, b, r, out, width);
}

// ---- Gbrpf32 → f32 RGB (lossless) -------------------------------------------

/// Dispatch `gbrpf32_to_rgb_f32_row` (lossless interleave; no SIMD needed).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f32_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  scalar::gbrpf32_to_rgb_f32_row(g, b, r, out, width);
}

// ---- Gbrpf32 → f32 RGBA (lossless) ------------------------------------------

/// Dispatch `gbrpf32_to_rgba_f32_row` (lossless; no SIMD needed).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f32_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  scalar::gbrpf32_to_rgba_f32_row(g, b, r, out, width);
}

// ---- Gbrpf32 → f16 RGB (fused narrow) ----------------------------------------

/// Dispatch `gbrpf32_to_rgb_f16_row` (fused f32→f16 narrow + interleave).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  let _ = use_simd;
  scalar::gbrpf32_to_rgb_f16_row(g, b, r, out, width);
}

// ---- Gbrpf32 → f16 RGBA (fused narrow) ---------------------------------------

/// Dispatch `gbrpf32_to_rgba_f16_row` (fused f32→f16 narrow + interleave).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  let _ = use_simd;
  scalar::gbrpf32_to_rgba_f16_row(g, b, r, out, width);
}

// ---- Gbrpf32 → u8 luma ------------------------------------------------------

/// Dispatch `gbrpf32_to_luma_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= width, "out too short");
  let _ = use_simd;
  scalar::gbrpf32_to_luma_row(g, b, r, out, width, matrix, full_range);
}

// ---- Gbrpf32 → u16 luma -----------------------------------------------------

/// Dispatch `gbrpf32_to_luma_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_u16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= width, "out too short");
  let _ = use_simd;
  scalar::gbrpf32_to_luma_u16_row(g, b, r, out, width, matrix, full_range);
}

// ---- Gbrpf32 → HSV ----------------------------------------------------------

/// Dispatch `gbrpf32_to_hsv_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_hsv_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(h_out.len() >= width, "h_out too short");
  assert!(s_out.len() >= width, "s_out too short");
  assert!(v_out.len() >= width, "v_out too short");
  let _ = use_simd;
  scalar::gbrpf32_to_hsv_row(g, b, r, h_out, s_out, v_out, width);
}

// ---- Gbrapf32 → u8 RGBA (source α) -----------------------------------------

/// Dispatch `gbrapf32_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  let _ = use_simd;
  scalar::gbrapf32_to_rgba_row(g, b, r, a, out, width);
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

/// Dispatch `gbrapf32_to_rgba_u16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_u16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  let _ = use_simd;
  scalar::gbrapf32_to_rgba_u16_row(g, b, r, a, out, width);
}

// ---- Gbrapf32 → f32 RGBA (lossless source α) --------------------------------

/// Dispatch `gbrapf32_to_rgba_f32_row` (lossless; no SIMD needed).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f32_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  scalar::gbrapf32_to_rgba_f32_row(g, b, r, a, out, width);
}

// ---- Gbrapf32 → f16 RGBA (fused narrow, source α) ---------------------------

/// Dispatch `gbrapf32_to_rgba_f16_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f16_row(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  out: &mut [half::f16],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  let _ = use_simd;
  scalar::gbrapf32_to_rgba_f16_row(g, b, r, a, out, width);
}

// ---- Gbrpf16 → f16 RGB (lossless, f16-native) --------------------------------

/// Dispatch `gbrpf16_to_rgb_f16_row` (lossless f16 interleave).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgb_f16_row(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [half::f16],
  width: usize,
  _use_simd: bool,
) {
  let out_min = rgb_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  scalar_f16::gbrpf16_to_rgb_f16_row(g, b, r, out, width);
}

// ---- Gbrpf16 → f16 RGBA (lossless, f16-native) ------------------------------

/// Dispatch `gbrpf16_to_rgba_f16_row` (lossless f16 interleave + α = f16(1.0)).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgba_f16_row(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [half::f16],
  width: usize,
  _use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  scalar_f16::gbrpf16_to_rgba_f16_row(g, b, r, out, width);
}

// ---- Gbrapf16 → f16 RGBA (lossless, source α) --------------------------------

/// Dispatch `gbrapf16_to_rgba_f16_row` (lossless f16 interleave + source α).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf16_to_rgba_f16_row(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  out: &mut [half::f16],
  width: usize,
  _use_simd: bool,
) {
  let out_min = rgba_row_elems(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(a.len() >= width, "a row too short");
  assert!(out.len() >= out_min, "out too short");
  scalar_f16::gbrapf16_to_rgba_f16_row(g, b, r, a, out, width);
}

// ---- Gbrpf16 → u8 RGB (widen f16 → f32, then scalar) -----------------------

/// Dispatch `gbrpf16_to_rgb_row`: widen f16 planes to f32, then call
/// `gbrpf32_to_rgb_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgb_row(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgb_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  const CHUNK: usize = 64;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    for i in 0..n {
      gf[i] = g[offset + i].to_f32();
      bf[i] = b[offset + i].to_f32();
      rf[i] = r[offset + i].to_f32();
    }
    scalar::gbrpf32_to_rgb_row(&gf[..n], &bf[..n], &rf[..n], &mut out[offset * 3..], n);
    offset += n;
    let _ = use_simd;
  }
}

// ---- Gbrpf16 → u8 RGBA (widen f16 → f32) ------------------------------------

/// Dispatch `gbrpf16_to_rgba_row`: widen f16 planes to f32, then call
/// `gbrpf32_to_rgba_row`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgba_row(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  let out_min = rgba_row_bytes(width);
  assert!(g.len() >= width, "g row too short");
  assert!(b.len() >= width, "b row too short");
  assert!(r.len() >= width, "r row too short");
  assert!(out.len() >= out_min, "out too short");
  const CHUNK: usize = 64;
  let mut gf = [0.0f32; CHUNK];
  let mut bf = [0.0f32; CHUNK];
  let mut rf = [0.0f32; CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    for i in 0..n {
      gf[i] = g[offset + i].to_f32();
      bf[i] = b[offset + i].to_f32();
      rf[i] = r[offset + i].to_f32();
    }
    scalar::gbrpf32_to_rgba_row(&gf[..n], &bf[..n], &rf[..n], &mut out[offset * 4..], n);
    offset += n;
    let _ = use_simd;
  }
}

// ---- 32-bit overflow guard tests --------------------------------------------

#[cfg(all(test, feature = "std", target_pointer_width = "32"))]
mod tests {
  use super::*;

  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbrpf32_to_rgb_panics_on_width_overflow() {
    let w = usize::MAX / 2 + 1;
    let g = vec![0.0f32; w];
    let b = vec![0.0f32; w];
    let r = vec![0.0f32; w];
    let mut out = vec![0u8; 3];
    gbrpf32_to_rgb_row(&g, &b, &r, &mut out, w, false);
  }

  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbrpf32_to_rgba_panics_on_width_overflow() {
    let w = usize::MAX / 2 + 1;
    let g = vec![0.0f32; w];
    let b = vec![0.0f32; w];
    let r = vec![0.0f32; w];
    let mut out = vec![0u8; 4];
    gbrpf32_to_rgba_row(&g, &b, &r, &mut out, w, false);
  }

  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbrpf32_to_rgb_u16_panics_on_width_overflow() {
    let w = usize::MAX / 2 + 1;
    let g = vec![0.0f32; w];
    let b = vec![0.0f32; w];
    let r = vec![0.0f32; w];
    let mut out = vec![0u16; 3];
    gbrpf32_to_rgb_u16_row(&g, &b, &r, &mut out, w, false);
  }

  #[test]
  #[should_panic(expected = "overflows usize")]
  fn gbrpf32_to_rgba_u16_panics_on_width_overflow() {
    let w = usize::MAX / 2 + 1;
    let g = vec![0.0f32; w];
    let b = vec![0.0f32; w];
    let r = vec![0.0f32; w];
    let mut out = vec![0u16; 4];
    gbrpf32_to_rgba_u16_row(&g, &b, &r, &mut out, w, false);
  }
}
