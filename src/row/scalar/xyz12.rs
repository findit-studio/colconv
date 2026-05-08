//! Scalar reference kernels for the Tier 12 (DCP / Xyz12) source.
//!
//! Pipeline (per-pixel):
//!
//! ```text
//! xyz_u12  →  xyz_linear (f32)  →  rgb_linear (f32) via M_xyz_to_rgb
//!         →  rgb_gamma (f32) via OETF  →  bgr_u8 / rgb_u8 / etc
//! ```
//!
//! Steps:
//!
//! 1. SMPTE ST 428-1 §8 inverse-OETF:
//!    `xyz_lin = (x_u12 / 4095)^2.6 / 0.91653`. Applied to each X/Y/Z
//!    sample independently.
//! 2. 3×3 matmul against the active gamut's `M_xyz_to_rgb` constant.
//! 3. sRGB-shape OETF (12.92 linear segment + `1.055 * c^(1/2.4) -
//!    0.055` upper segment). Skipped for f32-output paths
//!    (`xyz12_to_rgb_f32_row` / `xyz12_to_xyz_f32_row`).
//! 4. Range scale + integer narrow with round-half-up — only for u8 /
//!    u16 outputs.
//!
//! All kernels are const-generic over `BE: bool` for source endianness;
//! the `BE = false` branch is a compile-time no-op.

use crate::DcpTargetGamut;

use super::xyz12_constants::{INV_4095, SAMPLE_MASK, SMPTE428_INV_NORM, xyz_to_rgb_matrix};

/// `f32` `powf` portable across `std` and `no_std + alloc` builds.
/// `std` provides `f32::powf` directly via libm; `no_std` builds opt
/// into the same routine via the `libm` crate (gated by the `alloc`
/// feature in the crate's `Cargo.toml`).
#[cfg_attr(not(tarpaulin), inline(always))]
fn powf32(x: f32, y: f32) -> f32 {
  #[cfg(feature = "std")]
  {
    f32::powf(x, y)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::powf(x, y)
  }
}

// ---------------------------------------------------------------------------
// Helpers — kept `pub(crate)` so SIMD backends can re-use the OETF
// formula in their scalar tail / scalar-`powf` lanes.
// ---------------------------------------------------------------------------

/// Reads a packed XYZ12 sample with byte-swap if `BE` is set; masks
/// the upper 4 bits per the SMPTE ST 428-1 12-bit-active convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn read_xyz12_sample<const BE: bool>(s: u16) -> u16 {
  let raw = if BE { u16::from_be(s) } else { u16::from_le(s) };
  raw & SAMPLE_MASK
}

/// SMPTE ST 428-1 §8 inverse OETF: u12 → linear XYZ value in `f32`.
/// `xyz_lin = (x_u12 / 4095)^2.6 / 0.91653`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn smpte428_inverse_oetf(x_u12: u16) -> f32 {
  let normalised = (x_u12 & SAMPLE_MASK) as f32 * INV_4095;
  powf32(normalised, 2.6_f32) * SMPTE428_INV_NORM
}

/// Applies a 3×3 matrix to a linear XYZ vector, returning linear RGB.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn matmul3_xyz_rgb(m: &[[f32; 3]; 3], xyz: [f32; 3]) -> [f32; 3] {
  let [x, y, z] = xyz;
  [
    m[0][0] * x + m[0][1] * y + m[0][2] * z,
    m[1][0] * x + m[1][1] * y + m[1][2] * z,
    m[2][0] * x + m[2][1] * y + m[2][2] * z,
  ]
}

/// sRGB-shape OETF (Rec.709 form is identical to ~3-decimal precision
/// for the [0,1] range). Used by every integer-output sinker path.
///
/// `c < 0.0031308`: `12.92 * c` (linear toe).
/// `c >= 0.0031308`: `1.055 * c^(1/2.4) - 0.055`.
///
/// Inputs `c < 0` are clamped to zero before the upper segment to
/// avoid `c^(1/2.4)` returning NaN; inputs `c > 1` are returned with
/// the upper-segment formula applied (callers clamp at the integer
/// narrow).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn oetf_srgb(c: f32) -> f32 {
  if c < 0.0031308_f32 {
    12.92_f32 * c
  } else {
    1.055_f32 * powf32(c, 1.0_f32 / 2.4_f32) - 0.055_f32
  }
}

/// Round-half-up f32 → u8 narrow with `[0, 1]` clamp.
/// `(c.clamp(0, 1) * 255 + 0.5) as u8`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn narrow_unit_to_u8(c: f32) -> u8 {
  let scaled = c.clamp(0.0_f32, 1.0_f32) * 255.0_f32 + 0.5_f32;
  scaled.clamp(0.0_f32, 255.0_f32) as u8
}

/// Round-half-up f32 → u16 narrow with `[0, 1]` clamp.
/// `(c.clamp(0, 1) * 65535 + 0.5) as u16`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn narrow_unit_to_u16(c: f32) -> u16 {
  let scaled = c.clamp(0.0_f32, 1.0_f32) * 65535.0_f32 + 0.5_f32;
  scaled.clamp(0.0_f32, 65535.0_f32) as u16
}

/// Computes a single pixel's linear RGB from packed XYZ12 input.
/// Steps 1 + 2 of the pipeline (inverse-OETF + matmul). Used by every
/// downstream output kernel.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_pixel_to_rgb_linear<const BE: bool>(
  m: &[[f32; 3]; 3],
  triple: &[u16; 3],
) -> [f32; 3] {
  let x = smpte428_inverse_oetf(read_xyz12_sample::<BE>(triple[0]));
  let y = smpte428_inverse_oetf(read_xyz12_sample::<BE>(triple[1]));
  let z = smpte428_inverse_oetf(read_xyz12_sample::<BE>(triple[2]));
  matmul3_xyz_rgb(m, [x, y, z])
}

/// Computes a single pixel's linear XYZ (steps 1 only). Used by
/// `xyz12_to_xyz_f32_row` for lossless XYZ pass-through.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_pixel_to_xyz_linear<const BE: bool>(triple: &[u16; 3]) -> [f32; 3] {
  [
    smpte428_inverse_oetf(read_xyz12_sample::<BE>(triple[0])),
    smpte428_inverse_oetf(read_xyz12_sample::<BE>(triple[1])),
    smpte428_inverse_oetf(read_xyz12_sample::<BE>(triple[2])),
  ]
}

// ---------------------------------------------------------------------------
// Per-output kernels.
// ---------------------------------------------------------------------------

/// XYZ12 → packed RGB (u8). Full pipeline: inverse-OETF + matmul +
/// sRGB OETF + clamp + ×255 + round-half-up.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_to_rgb_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u8],
  width: usize,
  target_gamut: DcpTargetGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  for x in 0..width {
    let i = x * 3;
    let triple = [xyz[i], xyz[i + 1], xyz[i + 2]];
    let rgb_lin = xyz12_pixel_to_rgb_linear::<BE>(&m, &triple);
    rgb_out[i] = narrow_unit_to_u8(oetf_srgb(rgb_lin[0]));
    rgb_out[i + 1] = narrow_unit_to_u8(oetf_srgb(rgb_lin[1]));
    rgb_out[i + 2] = narrow_unit_to_u8(oetf_srgb(rgb_lin[2]));
  }
}

/// XYZ12 → packed RGBA (u8). Same as RGB; alpha forced to `0xFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_to_rgba_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  target_gamut: DcpTargetGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  for x in 0..width {
    let xi = x * 3;
    let oi = x * 4;
    let triple = [xyz[xi], xyz[xi + 1], xyz[xi + 2]];
    let rgb_lin = xyz12_pixel_to_rgb_linear::<BE>(&m, &triple);
    rgba_out[oi] = narrow_unit_to_u8(oetf_srgb(rgb_lin[0]));
    rgba_out[oi + 1] = narrow_unit_to_u8(oetf_srgb(rgb_lin[1]));
    rgba_out[oi + 2] = narrow_unit_to_u8(oetf_srgb(rgb_lin[2]));
    rgba_out[oi + 3] = 0xFF;
  }
}

/// XYZ12 → packed RGB (u16). Full pipeline; full-range scaling
/// `[0, 1] × 65535 + round-half-up`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_to_rgb_u16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [u16],
  width: usize,
  target_gamut: DcpTargetGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  for x in 0..width {
    let i = x * 3;
    let triple = [xyz[i], xyz[i + 1], xyz[i + 2]];
    let rgb_lin = xyz12_pixel_to_rgb_linear::<BE>(&m, &triple);
    rgb_out[i] = narrow_unit_to_u16(oetf_srgb(rgb_lin[0]));
    rgb_out[i + 1] = narrow_unit_to_u16(oetf_srgb(rgb_lin[1]));
    rgb_out[i + 2] = narrow_unit_to_u16(oetf_srgb(rgb_lin[2]));
  }
}

/// XYZ12 → packed RGBA (u16). Same as RGB-u16; alpha forced to
/// `0xFFFF`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_to_rgba_u16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  target_gamut: DcpTargetGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  for x in 0..width {
    let xi = x * 3;
    let oi = x * 4;
    let triple = [xyz[xi], xyz[xi + 1], xyz[xi + 2]];
    let rgb_lin = xyz12_pixel_to_rgb_linear::<BE>(&m, &triple);
    rgba_out[oi] = narrow_unit_to_u16(oetf_srgb(rgb_lin[0]));
    rgba_out[oi + 1] = narrow_unit_to_u16(oetf_srgb(rgb_lin[1]));
    rgba_out[oi + 2] = narrow_unit_to_u16(oetf_srgb(rgb_lin[2]));
    rgba_out[oi + 3] = 0xFFFF;
  }
}

/// XYZ12 → packed linear RGB (f32). Lossless after the matrix; **no
/// OETF, no clamp** — out-of-gamut negative R/G/B and HDR > 1 values
/// are emitted bit-exact.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_to_rgb_f32_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [f32],
  width: usize,
  target_gamut: DcpTargetGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  for x in 0..width {
    let i = x * 3;
    let triple = [xyz[i], xyz[i + 1], xyz[i + 2]];
    let rgb_lin = xyz12_pixel_to_rgb_linear::<BE>(&m, &triple);
    rgb_out[i] = rgb_lin[0];
    rgb_out[i + 1] = rgb_lin[1];
    rgb_out[i + 2] = rgb_lin[2];
  }
}

/// XYZ12 → packed linear XYZ (f32). Lossless XYZ pass-through — only
/// step 1 of the pipeline (SMPTE ST 428-1 inverse OETF). No matrix, no
/// gamma, no clamp. Useful for callers that want to do their own gamut
/// conversion downstream.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_to_xyz_f32_row<const BE: bool>(xyz: &[u16], xyz_out: &mut [f32], width: usize) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(xyz_out.len() >= width * 3, "xyz_out row too short");
  for x in 0..width {
    let i = x * 3;
    let triple = [xyz[i], xyz[i + 1], xyz[i + 2]];
    let xyz_lin = xyz12_pixel_to_xyz_linear::<BE>(&triple);
    xyz_out[i] = xyz_lin[0];
    xyz_out[i + 1] = xyz_lin[1];
    xyz_out[i + 2] = xyz_lin[2];
  }
}

/// XYZ12 → packed RGB (f16). Full pipeline like u8 but f16 narrow at
/// the end (IEEE-754 RNE via `f16::from_f32`). Clamp `[0, 1]` before
/// narrowing per integer-output convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_to_rgb_f16_row<const BE: bool>(
  xyz: &[u16],
  rgb_out: &mut [half::f16],
  width: usize,
  target_gamut: DcpTargetGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  for x in 0..width {
    let i = x * 3;
    let triple = [xyz[i], xyz[i + 1], xyz[i + 2]];
    let rgb_lin = xyz12_pixel_to_rgb_linear::<BE>(&m, &triple);
    rgb_out[i] = half::f16::from_f32(oetf_srgb(rgb_lin[0]).clamp(0.0, 1.0));
    rgb_out[i + 1] = half::f16::from_f32(oetf_srgb(rgb_lin[1]).clamp(0.0, 1.0));
    rgb_out[i + 2] = half::f16::from_f32(oetf_srgb(rgb_lin[2]).clamp(0.0, 1.0));
  }
}

/// XYZ12 → packed RGBA (f16). Same as f16 RGB; alpha forced to
/// `1.0_f16`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn xyz12_to_rgba_f16_row<const BE: bool>(
  xyz: &[u16],
  rgba_out: &mut [half::f16],
  width: usize,
  target_gamut: DcpTargetGamut,
) {
  debug_assert!(xyz.len() >= width * 3, "xyz row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let m = xyz_to_rgb_matrix(target_gamut);
  let one_f16 = half::f16::from_f32(1.0);
  for x in 0..width {
    let xi = x * 3;
    let oi = x * 4;
    let triple = [xyz[xi], xyz[xi + 1], xyz[xi + 2]];
    let rgb_lin = xyz12_pixel_to_rgb_linear::<BE>(&m, &triple);
    rgba_out[oi] = half::f16::from_f32(oetf_srgb(rgb_lin[0]).clamp(0.0, 1.0));
    rgba_out[oi + 1] = half::f16::from_f32(oetf_srgb(rgb_lin[1]).clamp(0.0, 1.0));
    rgba_out[oi + 2] = half::f16::from_f32(oetf_srgb(rgb_lin[2]).clamp(0.0, 1.0));
    rgba_out[oi + 3] = one_f16;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  /// Tolerance for comparing f32 values derived via the same algorithm
  /// the kernel implements. Round-trip through `f32::powf` is platform
  /// stable but the f64 -> f32 narrow at our derived-fixture step
  /// introduces a ~4e-7 noise floor, so `4e-6` is comfortably above
  /// any platform variation.
  const EPSILON_F32: f32 = 4e-6;

  fn assert_close(a: f32, b: f32, tag: &str) {
    let diff = (a - b).abs();
    assert!(diff <= EPSILON_F32, "{tag}: {a} vs {b} (diff {diff})");
  }

  // ---- OETF / inverse-OETF spot checks ----

  #[test]
  fn smpte428_inverse_oetf_zero_is_zero() {
    assert_eq!(smpte428_inverse_oetf(0), 0.0);
  }

  #[test]
  fn smpte428_inverse_oetf_max_is_normalised() {
    // (4095/4095)^2.6 / 0.91653 = 1.0 / 0.91653 ≈ 1.0911
    let actual = smpte428_inverse_oetf(4095);
    assert!((actual - 1.0_f32 / 0.91653_f32).abs() < EPSILON_F32);
  }

  #[test]
  fn smpte428_inverse_oetf_masks_upper_bits() {
    // Setting bit 13 should not change the result (upper 4 bits masked).
    let clean = smpte428_inverse_oetf(0x0800);
    let dirty = smpte428_inverse_oetf(0xF800);
    assert_eq!(clean, dirty);
  }

  #[test]
  fn oetf_srgb_zero_is_zero() {
    assert_eq!(oetf_srgb(0.0), 0.0);
  }

  #[test]
  fn oetf_srgb_uses_linear_below_threshold() {
    let c = 0.001_f32;
    let expected = 12.92_f32 * c;
    assert_eq!(oetf_srgb(c), expected);
  }

  #[test]
  fn oetf_srgb_one_is_one() {
    let v = oetf_srgb(1.0);
    assert!((v - 1.0).abs() < EPSILON_F32);
  }

  #[test]
  fn oetf_srgb_continuous_at_threshold() {
    let lo = oetf_srgb(0.0031307);
    let hi = oetf_srgb(0.0031309);
    // Should be close — function is continuous at the segment boundary.
    assert!((hi - lo).abs() < 1e-5);
  }

  #[test]
  fn narrow_unit_to_u8_round_half_up() {
    assert_eq!(narrow_unit_to_u8(0.0), 0);
    assert_eq!(narrow_unit_to_u8(1.0), 255);
    // 0.5 / 255 = 0.00196… → narrow_unit_to_u8(0.00196) ≈ 1.
    assert_eq!(narrow_unit_to_u8(0.5_f32), 128);
    assert_eq!(narrow_unit_to_u8(-1.0), 0);
    assert_eq!(narrow_unit_to_u8(2.0), 255);
  }

  #[test]
  fn narrow_unit_to_u16_round_half_up() {
    assert_eq!(narrow_unit_to_u16(0.0), 0);
    assert_eq!(narrow_unit_to_u16(1.0), 65535);
    assert_eq!(narrow_unit_to_u16(-1.0), 0);
    assert_eq!(narrow_unit_to_u16(2.0), 65535);
  }

  // ---- Derived-fixture parity tests (per gamut) ----
  //
  // Expected values produced by `examples/derive_xyz_matrices.rs` (run
  // 2026-05-08). Hardcoded as f32 literals below; the same algorithm
  // is implemented in the kernel, so this test is a regression lock
  // against accidental drift in the per-pixel math.

  #[test]
  fn xyz12_to_rgb_f32_rec709_zero_input() {
    let xyz = [0_u16; 3];
    let mut out = [0.0_f32; 3];
    xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
    assert_eq!(out, [0.0; 3]);
  }

  #[test]
  fn xyz12_to_rgb_f32_dci_p3_mid_gray() {
    let xyz: [u16; 3] = [0x800, 0x800, 0x800];
    let mut out = [0.0_f32; 3];
    xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
    // DCI-P3 expected: [0.2087783, 0.1722948, 0.1650483].
    assert_close(out[0], 0.208_778_3, "R");
    assert_close(out[1], 0.172_294_8, "G");
    assert_close(out[2], 0.165_048_3, "B");
  }

  #[test]
  fn xyz12_to_rgb_f32_rec709_mid_gray() {
    let xyz: [u16; 3] = [0x800, 0x800, 0x800];
    let mut out = [0.0_f32; 3];
    xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
    assert_close(out[0], 0.216_984_87, "R");
    assert_close(out[1], 0.170_760_4, "G");
    assert_close(out[2], 0.163_619_68, "B");
  }

  #[test]
  fn xyz12_to_rgb_f32_rec2020_three_quarter() {
    let xyz: [u16; 3] = [0xC00, 0xC00, 0xC00];
    let mut out = [0.0_f32; 3];
    xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec2020);
    assert_close(out[0], 0.572_369_93, "R");
    assert_close(out[1], 0.498_964_94, "G");
    assert_close(out[2], 0.473_854_f32, "B");
  }

  #[test]
  fn xyz12_to_rgb_f32_preserves_negative_after_matrix() {
    // y_only_max under Rec.709 → R = -1.677, G = +2.05, B = -0.222.
    let xyz: [u16; 3] = [0, 0xFFF, 0];
    let mut out = [0.0_f32; 3];
    xyz12_to_rgb_f32_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
    assert!(out[0] < 0.0, "expected negative R, got {}", out[0]);
    assert!(out[2] < 0.0, "expected negative B, got {}", out[2]);
    assert_close(out[0], -1.677_395_3, "R");
    assert_close(out[1], 2.046_815_2, "G");
    assert_close(out[2], -0.222_553_5, "B");
  }

  #[test]
  fn xyz12_to_rgb_clamps_at_u8() {
    // x_only_max under Rec.709 → R = +3.5, G = -1.0 → after OETF +
    // clamp + ×255 → R = 255, G = 0.
    let xyz: [u16; 3] = [0xFFF, 0, 0];
    let mut out = [0_u8; 3];
    xyz12_to_rgb_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
    assert_eq!(out[0], 255);
    assert_eq!(out[1], 0);
  }

  #[test]
  fn xyz12_to_rgba_fills_alpha_max() {
    let xyz: [u16; 3] = [0x800, 0x800, 0x800];
    let mut out = [0_u8; 4];
    xyz12_to_rgba_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
    assert_eq!(out[3], 0xFF);
  }

  #[test]
  fn xyz12_to_rgba_u16_fills_alpha_max() {
    let xyz: [u16; 3] = [0x800, 0x800, 0x800];
    let mut out = [0_u16; 4];
    xyz12_to_rgba_u16_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
    assert_eq!(out[3], 0xFFFF);
  }

  #[test]
  fn xyz12_to_xyz_f32_lossless_round_trip() {
    // Pass-through: input -> step-1 inverse-OETF -> output. For u12 =
    // (0x800, 0x800, 0x800) the linear value is the same in all three
    // channels.
    let xyz: [u16; 3] = [0x800, 0x800, 0x800];
    let mut out = [0.0_f32; 3];
    xyz12_to_xyz_f32_row::<false>(&xyz, &mut out, 1);
    let expected = powf32(0x800_u16 as f32 * INV_4095, 2.6_f32) * SMPTE428_INV_NORM;
    assert_close(out[0], expected, "X");
    assert_close(out[1], expected, "Y");
    assert_close(out[2], expected, "Z");
  }

  #[test]
  fn xyz12_be_byte_swap_matches_le() {
    // BE = byte-swap of LE input: the kernel should produce identical
    // output for byte-swapped vs native input.
    let raw: [u16; 3] = [0x0800, 0x0800, 0x0800];
    let mut out_le = [0.0_f32; 3];
    xyz12_to_rgb_f32_row::<false>(&raw, &mut out_le, 1, DcpTargetGamut::DciP3);

    let swapped: [u16; 3] = [
      raw[0].swap_bytes(),
      raw[1].swap_bytes(),
      raw[2].swap_bytes(),
    ];
    let mut out_be = [0.0_f32; 3];
    xyz12_to_rgb_f32_row::<true>(&swapped, &mut out_be, 1, DcpTargetGamut::DciP3);
    assert_eq!(out_le, out_be);
  }

  #[test]
  fn xyz12_to_rgb_u16_full_range_scaling() {
    let xyz: [u16; 3] = [0xFFF, 0xFFF, 0xFFF];
    let mut out = [0_u16; 3];
    xyz12_to_rgb_u16_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
    // Per derivation: rgb_linear = (1.265, 1.044, 1.0) → after OETF +
    // clamp [0,1] × 65535 → (65535, 65535, 65535).
    assert_eq!(out[0], 65535);
    assert_eq!(out[1], 65535);
    assert_eq!(out[2], 65535);
  }

  #[test]
  fn xyz12_to_rgb_f16_clamps_to_unit_range() {
    let xyz: [u16; 3] = [0xFFF, 0, 0];
    let mut out = [half::f16::from_f32(0.0); 3];
    xyz12_to_rgb_f16_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::Rec709);
    assert_eq!(out[0].to_f32(), 1.0);
    assert_eq!(out[1].to_f32(), 0.0);
  }

  #[test]
  fn xyz12_to_rgba_f16_alpha_one() {
    let xyz: [u16; 3] = [0x800, 0x800, 0x800];
    let mut out = [half::f16::from_f32(0.0); 4];
    xyz12_to_rgba_f16_row::<false>(&xyz, &mut out, 1, DcpTargetGamut::DciP3);
    assert_eq!(out[3].to_f32(), 1.0);
  }

  #[test]
  fn xyz12_to_rgb_target_gamut_changes_output() {
    let xyz: [u16; 3] = [0xC00, 0xC00, 0xC00];
    let mut out_p3 = [0.0_f32; 3];
    let mut out_709 = [0.0_f32; 3];
    let mut out_2020 = [0.0_f32; 3];
    xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_p3, 1, DcpTargetGamut::DciP3);
    xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_709, 1, DcpTargetGamut::Rec709);
    xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_2020, 1, DcpTargetGamut::Rec2020);
    // All three should differ on R (different matrix scales).
    assert!(
      (out_p3[0] - out_709[0]).abs() > 1e-3,
      "DCI-P3 vs Rec.709 R: {} vs {}",
      out_p3[0],
      out_709[0],
    );
    assert!(
      (out_p3[0] - out_2020[0]).abs() > 1e-3,
      "DCI-P3 vs Rec.2020 R: {} vs {}",
      out_p3[0],
      out_2020[0],
    );
  }

  #[test]
  fn xyz12_to_rgb_high_12bit_value_clipped_to_0fff_mask() {
    // Setting bit 13 should be equivalent to the clean 12-bit input.
    let xyz_clean: [u16; 3] = [0x0800, 0x0800, 0x0800];
    let xyz_dirty: [u16; 3] = [0x2800, 0xA800, 0xF800];
    let mut out_clean = [0_u8; 3];
    let mut out_dirty = [0_u8; 3];
    xyz12_to_rgb_row::<false>(&xyz_clean, &mut out_clean, 1, DcpTargetGamut::DciP3);
    xyz12_to_rgb_row::<false>(&xyz_dirty, &mut out_dirty, 1, DcpTargetGamut::DciP3);
    assert_eq!(out_clean, out_dirty);
  }

  #[test]
  fn xyz12_to_rgb_multi_pixel_independence() {
    let xyz: [u16; 6] = [
      0x800, 0x800, 0x800, // pixel 0
      0xFFF, 0, 0, // pixel 1
    ];
    let mut out = [0_u8; 6];
    xyz12_to_rgb_row::<false>(&xyz, &mut out, 2, DcpTargetGamut::Rec709);

    let mut single = [0_u8; 3];
    xyz12_to_rgb_row::<false>(&xyz[..3], &mut single, 1, DcpTargetGamut::Rec709);
    assert_eq!(&out[..3], &single);

    let mut single1 = [0_u8; 3];
    xyz12_to_rgb_row::<false>(&xyz[3..], &mut single1, 1, DcpTargetGamut::Rec709);
    assert_eq!(&out[3..], &single1);
  }
}
