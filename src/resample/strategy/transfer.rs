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

/// Verified ITU-R BT.2100 PQ / HLG per-channel inverse-EOTF / OETF math.
///
/// These are the net-new, reference-critical transfer stage of the BT.2100
/// non-affine colour decode (ICtCp — H.273 `MatrixCoefficients = 14` — and
/// SMPTE 2085): a source decodes `I,Ct,Cp → L'M'S'` (inverse ICtCp matrix)
/// `→ LMS` via the inverse-EOTF here `→ RGB` (LMS→RGB, BT.2020 primaries).
/// The math is kept here, private, until the deferred ICtCp matrix-wiring
/// (#303) routes a `ColorMatrix::Ictcp` source through it; it is
/// intentionally **not** placed on the public [`TransferFunction`] enum,
/// which is the RFC #238 *linear-light averaging* abstraction — a different
/// consumer — and which is `pub` without `#[non_exhaustive]` (adding
/// variants there would be a breaking change for downstream exhaustive
/// matches).
///
/// All constants are the published values of ITU-R BT.2100-2 (Tables 4 / 5)
/// and SMPTE ST 2084:2014, cross-checked against the `colour-science`
/// Python library; the reference-anchor tests in `tests.rs`
/// (`transfer_function_pq_matches_st2084_reference` /
/// `transfer_function_hlg_matches_bt2100_reference`) pin every constant.
// Consumed by the ICtCp non-affine decode (`row::scalar::ictcp`, #303) via
// the `crate::resample::pq_hlg` re-export; dead only in feature configs that
// build neither `yuv-planar` nor the transcendental tier it needs.
#[allow(dead_code)]
pub(crate) mod pq_hlg {
  use super::powf32;

  /// `f32` natural logarithm portable across `std` and `no_std + alloc`
  /// builds (companion of [`super::powf32`] for the HLG log segment).
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn lnf32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
      f32::ln(x)
    }
    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    {
      libm::logf(x)
    }
  }

  /// `f32` `exp` portable across `std` and `no_std + alloc` builds (for the
  /// HLG inverse-OETF log segment).
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn expf32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
      f32::exp(x)
    }
    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    {
      libm::expf(x)
    }
  }

  /// `f32` square root portable across `std` and `no_std + alloc` builds.
  /// `f32::sqrt` is a `std`-only intrinsic, so `no_std` routes through
  /// `libm::sqrtf` (HLG OETF lower / gamma segment).
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn sqrtf32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
      f32::sqrt(x)
    }
    #[cfg(all(not(feature = "std"), feature = "alloc"))]
    {
      libm::sqrtf(x)
    }
  }

  /// PQ exponent `m1 = 2610 / 16384` (BT.2100 Table 4) = `0.159301758`.
  const PQ_M1: f32 = 2610.0 / 16384.0;
  /// PQ exponent `m2 = 2523 / 4096 × 128` (BT.2100 Table 4) = `78.84375`.
  const PQ_M2: f32 = 2523.0 / 4096.0 * 128.0;
  /// PQ coefficient `c1 = 3424 / 4096` (BT.2100 Table 4) = `0.8359375`;
  /// equals `c3 − c2 + 1`, so PQ maps signal `1.0` to linear `1.0`.
  const PQ_C1: f32 = 3424.0 / 4096.0;
  /// PQ coefficient `c2 = 2413 / 4096 × 32` (BT.2100 Table 4) = `18.8515625`.
  const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0;
  /// PQ coefficient `c3 = 2392 / 4096 × 32` (BT.2100 Table 4) = `18.6875`.
  const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0;

  /// HLG coefficient `a = 0.17883277` (BT.2100 Table 5 / ARIB STD-B67).
  const HLG_A: f32 = 0.178_832_77;
  /// HLG coefficient `b = 1 − 4a = 0.28466892` (BT.2100 Table 5).
  const HLG_B: f32 = 0.284_668_92;
  /// HLG coefficient `c = 0.5 − a·ln(4a) = 0.55991073` (BT.2100 Table 5);
  /// the literal is the f32-nearest value (the trailing digit is below f32
  /// precision).
  const HLG_C: f32 = 0.559_910_7;

  /// SMPTE ST 2084 / BT.2100 PQ EOTF: signal `E'` → display-linear `Y`
  /// normalised so `1.0` = 10 000 cd/m².
  ///   `Y = (max(E'^(1/m2) − c1, 0) / (c2 − c3·E'^(1/m2)))^(1/m1)`.
  /// The negative side is mirrored through the origin (odd extension).
  /// PQ signal `E'` is defined on `[0, 1]` (`1.0` = the 10 000 cd/m² peak),
  /// so the magnitude is clamped to `1.0`: a super-white input saturates at
  /// the peak — a defined, monotonic policy — rather than crossing the
  /// `den = c2 − c3·vp = 0` pole at `|c| ≈ 1.99` (which overflows toward
  /// `+inf` just below it, and folds to black via the trailing `.max(0.0)`
  /// just above it where `num / den` goes negative).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn pq_eotf(c: f32) -> f32 {
    let vp = powf32(c.abs().min(1.0), 1.0 / PQ_M2);
    let num = (vp - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * vp;
    c.signum() * powf32((num / den).max(0.0), 1.0 / PQ_M1)
  }

  /// SMPTE ST 2084 / BT.2100 PQ inverse-EOTF: display-linear `Y`
  /// (normalised, `1.0` = 10 000 cd/m²) → signal `E'`.
  ///   `E' = ((c1 + c2·Y^m1) / (1 + c3·Y^m1))^m2`.
  /// For `Y ≥ 0` the base is in `(0, c2/c3) ⊂ (0, 2)`, so the power needs
  /// no NaN guard; the negative side is mirrored through the origin.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn pq_oetf(c: f32) -> f32 {
    let yp = powf32(c.abs(), PQ_M1);
    c.signum() * powf32((PQ_C1 + PQ_C2 * yp) / (1.0 + PQ_C3 * yp), PQ_M2)
  }

  /// BT.2100 / ARIB STD-B67 HLG inverse-OETF: signal `E'` → scene-linear
  /// `E` (per-channel scene light, **not** the full display EOTF whose OOTF
  /// system-gamma is luminance-dependent across channels).
  ///   `E = E'^2 / 3`                      for `|E'| ≤ 1/2`
  ///   `E = (exp((E' − c) / a) + b) / 12`  for `|E'| > 1/2`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn hlg_eotf(c: f32) -> f32 {
    // HLG signal `E'` is defined on `[0, 1]`; clamp the magnitude so a
    // super-white input saturates at the peak (the log segment otherwise
    // grows unbounded for `E' > 1`), matching `pq_eotf`'s defined domain.
    let a = c.abs().min(1.0);
    let e = if a <= 0.5 {
      a * a / 3.0
    } else {
      (expf32((a - HLG_C) / HLG_A) + HLG_B) / 12.0
    };
    c.signum() * e
  }

  /// BT.2100 / ARIB STD-B67 HLG OETF: scene-linear `E` → signal `E'` (the
  /// per-channel inverse of [`hlg_eotf`]).
  ///   `E' = sqrt(3·E)`             for `|E| ≤ 1/12`
  ///   `E' = a·ln(12·E − b) + c`    for `|E| > 1/12`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn hlg_oetf(c: f32) -> f32 {
    let e = c.abs();
    let v = if e <= 1.0 / 12.0 {
      sqrtf32(3.0 * e)
    } else {
      HLG_A * lnf32(12.0 * e - HLG_B) + HLG_C
    };
    c.signum() * v
  }
}

/// Verified ITU-R BT.2020 opto-electronic transfer (the SDR "camera" gamma,
/// H.273 `TransferCharacteristics ∈ {1, 14, 15}`) and its analytic inverse.
///
/// This is the reference-critical transfer stage of the **constant-luminance**
/// (`YcCbcCrc`) colour decode — H.273 `MatrixCoefficients = 13`,
/// [`ColorMatrix::ChromaDerivedCl`](crate::ColorMatrix::ChromaDerivedCl). The
/// CL system, unlike every non-constant-luminance (affine) matrix, derives its
/// luma from **linear** light and bends the chroma normalisers through *this*
/// curve, so the decode genuinely needs the BT.2020 OETF in the pixel loop
/// (`row::scalar::cl`). It is kept here, private, beside the [`pq_hlg`] BT.2100
/// math the ICtCp decode uses — both are the non-affine transfer foundation,
/// deliberately **not** on the public [`TransferFunction`] enum (that is the
/// RFC #238 linear-light averaging abstraction, a different consumer, and is
/// `pub` without `#[non_exhaustive]`).
///
/// All constants are the published values of ITU-R BT.2020-2 (Table 4),
/// cross-checked against the `colour-science` `oetf_BT2020` /
/// `oetf_inverse_BT2020` reference; the anchor tests in `row::scalar::cl`
/// pin them.
// Consumed by the constant-luminance decode (`row::scalar::cl`, #303) via the
// `crate::resample::bt2020_oetf` re-export; dead only in feature configs that
// build neither `yuv-planar` nor the transcendental tier it needs.
#[allow(dead_code)]
pub(crate) mod bt2020_oetf {
  use super::powf32;

  /// BT.2020 toe slope below the breakpoint (`E' = 4.5·E`).
  const SLOPE: f32 = 4.5;
  /// BT.2020 OETF exponent (`0.45`) and its reciprocal for the inverse.
  const GAMMA: f32 = 0.45;

  /// The `(α, β)` pair selected by the system bit depth, per BT.2020-2 Table 4
  /// (mirrors `colour-science`'s `CONSTANTS_BT2020`):
  ///
  /// - 10-bit system (the default): `α = 1.099`, `β = 0.018`.
  /// - 12-bit system: `α = 1.0993`, `β = 0.0181`.
  ///
  /// The constant-luminance decode picks the depth from the signalled
  /// [`Transfer`](crate::Transfer) (`Bt2020_12Bit` → 12-bit, else 10-bit).
  #[cfg_attr(not(tarpaulin), inline(always))]
  const fn alpha_beta(is_12_bit: bool) -> (f32, f32) {
    if is_12_bit {
      (1.0993, 0.0181)
    } else {
      (1.099, 0.018)
    }
  }

  /// BT.2020 OETF: linear light `E` → encoded signal `E'`.
  ///   `E' = 4.5·E`                     for `E < β`
  ///   `E' = α·E^0.45 − (α − 1)`        for `E ≥ β`.
  ///
  /// The published BT.2020-2 (Table 4) constants are **not** a smooth join:
  /// the toe reaches `4.5·β` as `E → β⁻`, while the power branch starts at
  /// `α·β^0.45 − (α − 1) = oetf(β)`, and these differ (10-bit `oetf(β) =
  /// 0.081248 > 4.5·β = 0.081`; 12-bit `oetf(β) = 0.0814472 < 4.5·β =
  /// 0.081450`). The curve is therefore **discontinuous at `β`**, and for the
  /// 12-bit constants it steps **down** there (locally non-monotonic). This is
  /// faithfully reproduced (matching `colour-science`'s `oetf_BT2020`). The
  /// consequence for invertibility — a precise four-interval partition of the
  /// 12-bit encoded domain, with every endpoint's inclusivity — is documented
  /// in full on [`oetf_inverse`]; do not duplicate or paraphrase it here.
  ///
  /// The negative side is mirrored through the origin (odd extension) so an
  /// out-of-gamut excursion re-encodes symmetrically rather than producing a
  /// NaN from a fractional power of a negative base.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn oetf(e: f32, is_12_bit: bool) -> f32 {
    let (a, b) = alpha_beta(is_12_bit);
    let m = e.abs();
    let v = if m < b {
      SLOPE * m
    } else {
      a * powf32(m, GAMMA) - (a - 1.0)
    };
    e.signum() * v
  }

  /// BT.2020 inverse-OETF: encoded signal `E'` → linear light `E`.
  ///   `E = E' / 4.5`                                     for `E' < T`
  ///   `E = ((E' + α − 1) / α)^(1/0.45)`                  otherwise,
  /// where the toe↔power breakpoint `T` is the **10-bit** forward OETF
  /// evaluated at this depth's `β`: `T = 1.099·β^0.45 − 0.099`.
  ///
  /// `T` reproduces `colour-science`'s `oetf_inverse_BT2020` branch condition
  /// **exactly**, including a quirk: colour's inverse tests `E_p <
  /// oetf_BT2020(b)` but calls `oetf_BT2020(b)` **without** forwarding
  /// `is_12_bits_system`, so the threshold always uses the 10-bit `α = 1.099`
  /// — even when decoding a 12-bit signal (whose `β = 0.0181`). The
  /// constant-luminance decode is verified against `colour.YcCbcCrc_to_RGB`,
  /// which routes through that inverse, so matching it bit-for-bit (rather than
  /// the "mathematically cleaner" per-depth `α`) is what keeps the decode
  /// reference-exact. For the 10-bit constants `T = oetf(β)` either way (the
  /// threshold `α` and the curve `α` coincide), so only the 12-bit path is
  /// affected.
  ///
  /// # Complete 12-bit encoded-domain partition
  ///
  /// The single threshold `T = 0.0816979` partitions the non-negative encoded
  /// domain into **four intervals**. Because the forward [`oetf`] is
  /// discontinuous and (for 12-bit) non-monotonic at `β`, two of them do not
  /// round-trip — one by the BT.2020 standard, one by the deliberate
  /// colour-compat choice. The boundary values are
  /// `oetf(β) = 0.0814472 < 4.5·β = 0.0814500 < T = 0.0816979`. Every endpoint
  /// inclusivity below was confirmed against `colour.oetf_inverse_BT2020`
  /// (which branch colour takes *at* the endpoint); the per-interval reference
  /// tests in `row::scalar::cl` pin one point in each.
  ///
  /// | Interval (`E'`)            | Branch | Round-trip                                  |
  /// |---------------------------|--------|---------------------------------------------|
  /// | `[0, oetf(β))`            | toe    | **exact** — `E < β` toe-encodes here, toe recovers it |
  /// | `[oetf(β), 4.5·β)`        | toe    | **no — standard** (BT.2020 spec)            |
  /// | `[4.5·β, T)`             | toe    | **no — colour quirk** (deliberate)          |
  /// | `[T, ∞)`                 | power  | **exact**                                   |
  ///
  /// - **`[oetf(β), 4.5·β)` — the standard discontinuity overlap** (~2.8e-6
  ///   wide). It is exactly the intersection of the toe image `[0, 4.5·β)` and
  ///   the power image `[oetf(β), ∞)`: both forward branches reach it, so **no**
  ///   threshold can recover the original `E` — a genuine BT.2020 property. The
  ///   **lower endpoint is inclusive** (`β` itself forward-maps to `oetf(β)`,
  ///   which decodes to the toe `oetf(β)/4.5 ≈ 0.0180994 ≠ β`, so `β` is the
  ///   first non-round-tripping point). The **upper endpoint is exclusive**:
  ///   the toe image never attains `4.5·β` (it is the open sup as `E → β⁻`).
  /// - **`[4.5·β, T)` — the colour-quirk band** (~2.5e-4 wide). Every value
  ///   here **has a unique, valid power preimage** (`E ≥ β`; the standard
  ///   round-trip *exists*), yet because `T` uses the 10-bit `α` (the colour
  ///   non-propagation quirk above), `E' < T` holds and the inverse
  ///   **knowingly returns the toe value `E'/4.5` instead** — diverging from the
  ///   round-trip by the toe↔power gap (~6.5e-7 in `E`). This is a **deliberate
  ///   reference-compat choice**, NOT a standard property: it is exactly what
  ///   `colour.oetf_inverse_BT2020` returns, and the decode is verified against
  ///   `colour.YcCbcCrc_to_RGB`. The **lower endpoint `4.5·β` is inclusive**
  ///   here (it is power-reachable only, not in the toe image, and `4.5·β < T`
  ///   so it takes the toe); the **upper endpoint `T` is exclusive** (`E' < T`
  ///   is strict, so a value *at* `T` takes the power branch — the next
  ///   interval). A future change that "corrected" this band to round-trip
  ///   would break the bit-exact colour agreement (guarded by the reference
  ///   test).
  ///
  /// **10-bit has no overlap or quirk band**: there `oetf(β) > 4.5·β`
  /// (`0.081248 > 0.081`), the curve is monotonic across `β`, and `T = oetf(β)`
  /// (the threshold `α` and the curve `α` coincide). The partition collapses to
  /// the simple two intervals `[0, oetf(β))` (toe, exact) and `[oetf(β), ∞)`
  /// (power, exact) — `β` round-trips. The residual in either 12-bit
  /// non-round-tripping interval is far below the output quantization.
  /// Odd-extended so `inverse(-c) == -inverse(c)`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn oetf_inverse(ep: f32, is_12_bit: bool) -> f32 {
    let (a, b) = alpha_beta(is_12_bit);
    // `colour-science`'s threshold is `oetf_BT2020(β)` with `is_12_bits_system`
    // left at its `false` default, so the breakpoint always uses the 10-bit
    // `α = 1.099` (with this depth's `β`); replicate that exactly. For 10-bit
    // this equals `oetf(β)`; for 12-bit it is the documented colour quirk.
    let (a10, _) = alpha_beta(false);
    let threshold = a10 * powf32(b, GAMMA) - (a10 - 1.0);
    let m = ep.abs();
    let v = if m < threshold {
      m / SLOPE
    } else {
      powf32((m + (a - 1.0)) / a, 1.0 / GAMMA)
    };
    ep.signum() * v
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
