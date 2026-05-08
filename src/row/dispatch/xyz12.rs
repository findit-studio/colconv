//! Tier 12 — packed CIE XYZ 12-bit (`Xyz12`) source-side row
//! dispatchers.
//!
//! Each entry point converts one row of packed `X, Y, Z` `u16` input
//! (low-12-bit-active samples) to the requested output format. Every
//! kernel takes:
//!
//! - `BE: const bool` — wire-format endianness of the source `u16`s.
//! - `target_gamut: DcpTargetGamut` — runtime choice of XYZ → RGB
//!   matrix (DCI-P3 / Rec.709 / Rec.2020).
//!
//! Pipeline (per pixel): SMPTE ST 428-1 §8 inverse-OETF → 3×3 matmul
//! → sRGB-shape OETF (skipped for f32 outputs) → range scale + integer
//! narrow (only for u8 / u16 outputs).
//!
//! SIMD backends ship in tranches 7–11; only the scalar fallback is
//! wired right now. The signature is forward-compatible: backends will
//! plug into the same `cfg_select!` block as the dispatcher grows.

use crate::{
  DcpTargetGamut,
  row::{
    rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar::xyz12 as scalar_xyz12,
  },
};

/// XYZ12 → packed `R, G, B` `u8` row dispatcher.
///
/// `use_simd = false` forces scalar; SIMD backends are wired in
/// tranches 7–11.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgb_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  target_gamut: DcpTargetGamut,
  _use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_bytes(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  scalar_xyz12::xyz12_to_rgb_row::<BE>(xyz, rgb_out, width, target_gamut);
}

/// XYZ12 → packed `R, G, B, A` `u8` row dispatcher (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgba_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  target_gamut: DcpTargetGamut,
  _use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_bytes(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  scalar_xyz12::xyz12_to_rgba_row::<BE>(xyz, rgba_out, width, target_gamut);
}

/// XYZ12 → packed `R, G, B` `u16` row dispatcher (full-range scaling).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgb_u16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  target_gamut: DcpTargetGamut,
  _use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  scalar_xyz12::xyz12_to_rgb_u16_row::<BE>(xyz, rgb_out, width, target_gamut);
}

/// XYZ12 → packed `R, G, B, A` `u16` row dispatcher (alpha = `0xFFFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgba_u16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  target_gamut: DcpTargetGamut,
  _use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  scalar_xyz12::xyz12_to_rgba_u16_row::<BE>(xyz, rgba_out, width, target_gamut);
}

/// XYZ12 → packed linear `R, G, B` `f32` row dispatcher.
///
/// **Lossless** linear-RGB output — no OETF, no clamp. Out-of-gamut
/// negative R/G/B and HDR > 1 values are emitted bit-exact.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgb_f32_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [f32],
  width: usize,
  target_gamut: DcpTargetGamut,
  _use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  scalar_xyz12::xyz12_to_rgb_f32_row::<BE>(xyz, rgb_out, width, target_gamut);
}

/// XYZ12 → packed linear `X, Y, Z` `f32` row dispatcher (lossless XYZ
/// pass-through after step-1 inverse-OETF).
///
/// No matrix, no gamma, no clamp — useful for callers that do their
/// own gamut conversion downstream.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_xyz_f32_row<const BE: bool>(
  xyz: &[u16],
  xyz_out: &mut [f32],
  width: usize,
  _use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let xyz_out_min = rgb_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(xyz_out.len() >= xyz_out_min, "xyz_out row too short");

  scalar_xyz12::xyz12_to_xyz_f32_row::<BE>(xyz, xyz_out, width);
}

/// XYZ12 → packed `R, G, B` `f16` row dispatcher (gamma-encoded,
/// clamped to `[0, 1]`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgb_f16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [half::f16],
  width: usize,
  target_gamut: DcpTargetGamut,
  _use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgb_out_min = rgb_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgb_out.len() >= rgb_out_min, "rgb_out row too short");

  scalar_xyz12::xyz12_to_rgb_f16_row::<BE>(xyz, rgb_out, width, target_gamut);
}

/// XYZ12 → packed `R, G, B, A` `f16` row dispatcher (alpha = `1.0`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn xyz12_to_rgba_f16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [half::f16],
  width: usize,
  target_gamut: DcpTargetGamut,
  _use_simd: bool,
) {
  let xyz_in_min = rgb_row_elems(width);
  let rgba_out_min = rgba_row_elems(width);
  assert!(xyz.len() >= xyz_in_min, "xyz row too short");
  assert!(rgba_out.len() >= rgba_out_min, "rgba_out row too short");

  scalar_xyz12::xyz12_to_rgba_f16_row::<BE>(xyz, rgba_out, width, target_gamut);
}
