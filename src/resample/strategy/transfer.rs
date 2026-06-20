//! RFC #238 — caller-configurable opto-electronic transfer functions for
//! the [`AveragingDomain::Linear`](super::AveragingDomain::Linear) domain.
//!
//! The Linear domain averages an area downscale in *linear light*: each
//! source pixel is decoded to RGB, the inverse transfer function (EOTF)
//! lifts the gamma-encoded RGB to linear light, the linear samples are
//! box-averaged, and the forward transfer function (OETF) re-encodes the
//! result. A [`TransferFunction`] is the EOTF/OETF pair that brackets that
//! average; the two are exact analytic inverses, so a decode→re-encode
//! round-trip with no averaging is (modulo float rounding) the identity.
//!
//! In this release the Linear tail decodes through the production
//! `yuv_*_to_rgb_row` kernels, which clamp + quantize to 8-bit `[0, 255]`
//! before the EOTF — so the average is **display-referred**: it mixes the
//! converted in-gamut 8-bit RGB in linear light (a gamma-correct resize),
//! not a scene-linear average of the unclamped affine decode. The curves
//! here are written to also cover the unclamped (out-of-`[0, 1]`) case a
//! future scene-linear consumer will feed; see the EOTF/OETF extrapolation
//! notes and the `sinker::mixed::linear_light` module header.
//!
//! # Why this exists separately from the matrix
//!
//! colconv's [`ColorMatrix`] (the mediaframe `Matrix`) is an H.273
//! *MatrixCoefficients* value — it fixes the YCbCr→RGB matrix, not the
//! transfer characteristics. The transfer (H.273 *TransferCharacteristics*)
//! is an independent axis colconv's YUV row stage does not carry, because
//! the convert is purely affine. The Linear domain is the first consumer
//! that needs it, so this module supplies the curves and a per-matrix
//! [default resolution](TransferFunction::for_matrix).
//!
//! # The per-`ColorMatrix` default
//!
//! When the caller selects the Linear domain without an explicit override,
//! the transfer is resolved from the sink's [`ColorMatrix`] by the
//! established colorimetric convention:
//!
//! - The [`ColorMatrix::Rgb`] identity (sRGB / ST 428-1 primaries) resolves
//!   to [`TransferFunction::Srgb`] — the matching sRGB curve.
//! - Every YCbCr video matrix (BT.601 / BT.709 / BT.2020 / SMPTE-170M /
//!   240M / FCC / BT.470BG / YCgCo / …) resolves to
//!   [`TransferFunction::Bt1886`] — the ITU-R BT.1886 reference *display*
//!   EOTF (pure 2.4 gamma) that SDR video is mastered against, regardless
//!   of which matrix coefficients carry the chroma. This is the standard
//!   video convention: the matrix selects the YCbCr basis, the BT.1886
//!   display curve selects how the encoded RGB maps to light.
//!
//! Callers who carry a known transfer out of band override it explicitly
//! with [`MixedSinker::with_transfer_function`](crate::sinker::MixedSinker::with_transfer_function).
//!
//! [`ColorMatrix`]: crate::ColorMatrix

use crate::ColorMatrix;

/// `f32` `powf` portable across `std` and `no_std + alloc` builds. `std`
/// provides `f32::powf`; `no_std` builds opt into the same routine via the
/// `libm` crate (gated by the `alloc` feature in `Cargo.toml`). Mirrors the
/// helpers of the same name in `row::scalar::xyz12` and the RFC #238 PoC —
/// duplicated rather than shared because those are `xyz`-gated and the
/// Linear domain does not imply `xyz`.
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

/// The opto-electronic transfer function the
/// [`AveragingDomain::Linear`](super::AveragingDomain::Linear) domain
/// linearises and re-encodes RGB through.
///
/// Each variant exposes the inverse transfer [`eotf`](Self::eotf)
/// (encoded → linear, the *decode*) and the forward transfer
/// [`oetf`](Self::oetf) (linear → encoded, the *encode*) as exact analytic
/// inverses. The default ([`Self::Bt1886`]) is the SDR-video display curve;
/// see [`Self::for_matrix`] for how a sink with no caller override resolves
/// the curve from its [`ColorMatrix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TransferFunction {
  /// Identity / no-op transfer: `eotf(c) == c` and `oetf(c) == c`. The
  /// Linear domain then averages the already-"linear" encoded RGB — i.e.
  /// it models content whose codes are already light-linear. Cheap and
  /// occasionally useful as a baseline; rarely the physically-correct
  /// choice for real video.
  LinearPassthrough,
  /// The sRGB transfer pair (IEC 61966-2-1): a 12.92 linear toe below the
  /// breakpoint, `1.055 * c^(1/2.4) - 0.055` above. The companion of the
  /// [`ColorMatrix::Rgb`] identity matrix.
  Srgb,
  /// The ITU-R BT.1886 reference display EOTF — pure 2.4 gamma
  /// (`linear = encoded^2.4`, `encoded = linear^(1/2.4)`), no toe. The SDR
  /// video standard and the **default**: BT.601 / BT.709 / BT.2020 content
  /// is mastered against this display curve.
  #[default]
  Bt1886,
  /// Pure 2.2 gamma (`linear = encoded^2.2`). A common display-gamma
  /// approximation; provided as a cheap caller option distinct from the
  /// 2.4 BT.1886 curve.
  Gamma22,
}

impl TransferFunction {
  /// Lowercase identifier for the curve
  /// (`"linear"` / `"srgb"` / `"bt1886"` / `"gamma22"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::LinearPassthrough => "linear",
      Self::Srgb => "srgb",
      Self::Bt1886 => "bt1886",
      Self::Gamma22 => "gamma22",
    }
  }

  /// Resolves the default transfer for a sink whose caller did **not**
  /// override it, from the sink's [`ColorMatrix`].
  ///
  /// [`ColorMatrix::Rgb`] (the sRGB / ST 428-1 identity) maps to
  /// [`Self::Srgb`]; every YCbCr video matrix — and the
  /// unknown/unspecified codes, which default to the video assumption —
  /// maps to [`Self::Bt1886`], the ITU-R BT.1886 reference display EOTF.
  /// See the module-level documentation for the colorimetric rationale.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn for_matrix(matrix: ColorMatrix) -> Self {
    match matrix {
      // The sRGB / ST 428-1 identity (GBR) pairs with the sRGB curve.
      ColorMatrix::Rgb => Self::Srgb,
      // Every YCbCr matrix is SDR video → the BT.1886 display EOTF. The
      // matrix only selects the chroma basis; the display curve is
      // BT.1886 for all of them. Unknown / Unspecified inherit the video
      // assumption (the same fallback FFmpeg's height-based inference
      // resolves a matrix to).
      _ => Self::Bt1886,
    }
  }

  /// Inverse transfer (EOTF): encoded `[0, 1]` → linear light — the
  /// *decode* the Linear domain applies per source pixel before binning.
  ///
  /// Inputs outside `[0, 1]` extrapolate analytically; the negative side
  /// of every curve is mirrored through the origin
  /// (`eotf(-c) == -eotf(c)`) so super-black excursions linearise
  /// symmetrically rather than folding. The integer narrow downstream
  /// clamps the re-encoded result.
  ///
  /// The current colconv Linear path feeds only clamped 8-bit RGB in
  /// `[0, 1]`, so this out-of-range extrapolation is dormant there; it is
  /// part of the public contract for any (future) scene-linear consumer
  /// that decodes the unclamped affine `YUV→RGB`, where super-black /
  /// super-white codes do appear. Do not narrow it to `[0, 1]`-only.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn eotf(self, c: f32) -> f32 {
    match self {
      Self::LinearPassthrough => c,
      Self::Srgb => {
        // 12.92 toe below 0.04045, `((c + 0.055) / 1.055)^2.4` above.
        if c.abs() <= 0.040_45 {
          c / 12.92
        } else {
          c.signum() * powf32((c.abs() + 0.055) / 1.055, 2.4)
        }
      }
      Self::Bt1886 => c.signum() * powf32(c.abs(), 2.4),
      Self::Gamma22 => c.signum() * powf32(c.abs(), 2.2),
    }
  }

  /// Forward transfer (OETF): linear light → encoded `[0, 1]` — the
  /// *encode* the Linear domain applies per output pixel after binning.
  /// The exact analytic inverse of [`Self::eotf`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn oetf(self, c: f32) -> f32 {
    match self {
      Self::LinearPassthrough => c,
      Self::Srgb => {
        // 12.92 toe below 0.0031308, `1.055 * c^(1/2.4) - 0.055` above.
        if c.abs() <= 0.003_130_8 {
          12.92 * c
        } else {
          c.signum() * (1.055 * powf32(c.abs(), 1.0 / 2.4) - 0.055)
        }
      }
      Self::Bt1886 => c.signum() * powf32(c.abs(), 1.0 / 2.4),
      Self::Gamma22 => c.signum() * powf32(c.abs(), 1.0 / 2.2),
    }
  }
}
