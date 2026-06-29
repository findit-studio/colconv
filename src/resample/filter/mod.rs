//! Separable **filter** resampling — the signed-coefficient twin of the
//! integer area engine in the parent module.
//!
//! Where [`AreaResampler`](super::AreaResampler) plans exact integer
//! box-coverage spans, [`FilteredResampler`] plans *windowed
//! reconstruction-filter* spans: per output sample, a contiguous run of
//! source samples each multiplied by a **signed** floating-point weight
//! drawn from a [`FilterKernel`], normalized so the run sums to one. This
//! is the Pillow (PIL) `Image.resize` convention — [`Triangle`] matches
//! PIL `BILINEAR`, [`CatmullRom`] matches PIL `BICUBIC`, and [`Lanczos3`]
//! matches PIL `LANCZOS`. The `u8` output path reproduces PIL **byte-exact**
//! by resampling on PIL's 8bpc fixed-point coefficient grid (`coeffs_q8`,
//! [`PRECISION_BITS`] = 22); the 32bpc `u16` / `f32` paths use the
//! full-precision `f32` coefficients, matching PIL's double-coefficient
//! resampler within +-1 LSB (`u16`) or `f32` precision.
//!
//! The half-pixel sampling convention ([`FilterAxis::build`]) mirrors
//! PIL's `precompute_coeffs` exactly: that center/support arithmetic is
//! the correctness crux, so it is ported verbatim (in `f64`) rather than
//! re-derived.
//!
//! [`FilterKernel`] is **public and unsealed** — it is the crate's
//! extension point for custom kernels. A hostile kernel cannot make the
//! window unsafe: [`FilterAxis::build`] validates [`FilterKernel::support`]
//! is finite, strictly positive, and bounded before sizing any window.
//!
//! Downscale, upscale, and mixed per-axis ratios are all supported, on the
//! scalar path and every SIMD backend.
//!
//! # Support is dynamic — no per-kernel support policy
//!
//! A kernel declares only its *intrinsic* truncation radius
//! ([`FilterKernel::support`], the half-width at unit scale); the engine
//! supplies the dynamic sizing. [`FilterAxis::build`] computes `scale =
//! in_size / out_size`, `filterscale = max(scale, 1)`, and widens the
//! footprint to `support() * filterscale` — so **every** kernel gets
//! scale-relative anti-alias widening on downscale and its native support on
//! upscale, for free. An earlier design sketch hypothesized a per-kernel
//! `SupportPolicy { Intrinsic, Truncated, ScaleRelative }`; it is
//! unnecessary, because the engine's `filterscale` already applies the
//! scale-relative policy universally. A new kernel therefore only defines a
//! finite `support()` (its truncation radius) and a `weight()` profile — it
//! never selects a sizing policy.

use std::vec::Vec;

use super::{PlanGeometry, ResampleError};

// Test-only allocation failpoint for the first `FilterAxis` plan-table
// reservation (`starts`) in `FilterAxis::build`. Armed, that reservation
// returns the crate's recoverable `AllocationFailed` WITHOUT reserving —
// and is consulted only AFTER the no-allocation zero-tap dry pass — so the
// regression tests can prove an invalid sub-ULP support is rejected
// (`InvalidFilterSupport`) BEFORE any plan table is sized, while a valid
// kernel still reaches and trips the armed failpoint (`AllocationFailed`).
// `Cell<bool>` is plenty (single-threaded, take-on-read). Strictly
// test-only — the non-test build compiles this away entirely. Mirrors the
// `#178` failpoint convention (`FORCE_*_ALLOC_FAILURE` / `arm_*`).
#[cfg(all(test, feature = "std"))]
std::thread_local! {
  static FORCE_FILTER_AXIS_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the `FilterAxis` plan-table allocation failpoint for the **next**
/// [`FilterAxis::build`] on the current thread. The flag is consumed
/// (take-on-read) by that build's first table reservation, so it fires
/// exactly once and cannot leak into a later test. Test-only.
#[cfg(all(test, feature = "std"))]
pub(crate) fn arm_filter_axis_alloc_failure() {
  FORCE_FILTER_AXIS_ALLOC_FAILURE.with(|f| f.set(true));
}

/// `next_up(x)` — the next representable `f64` above a finite, positive `x`,
/// by incrementing its bit pattern (finite positives are monotone in their
/// `u64` representation). Used only to size one ULP via subtraction; never
/// called on `NaN`, infinities, or non-positive values (the caller guards
/// `center > 0` and finite).
#[cfg_attr(not(tarpaulin), inline(always))]
fn next_up_f64(x: f64) -> f64 {
  f64::from_bits(x.to_bits() + 1)
}

/// Whether the scaled `support` is below the `f64` grid spacing at the output
/// extent — the `O(1)`, no-allocation predicate by which [`FilterAxis::build`]
/// rejects a support too small to be faithfully evaluated at this geometry.
///
/// A window degenerates to zero taps (`xmin == xmax`) only where `support` is
/// small enough that `center - support == center` (and `center + support ==
/// center`) in `f64` — i.e. where `support` falls below the spacing (ULP) at
/// that center. The projected centers are `(xx + 0.5) * scale` for
/// `xx in 0..out_size`, all in `(0, c_max]` with `c_max = (out_size - 0.5) *
/// scale` the largest. ULP grows with the binade, so `c_max` carries the
/// **largest** ULP of any center: `ULP_above(c) <= ULP_above(c_max)` for every
/// center `c`. Hence if `support >= ULP_above(c_max)` then `support >=
/// ULP_above(c)` at *every* center, so `center + support` rounds strictly above
/// `center` and `center - support` strictly below — `support` is absorbed
/// *nowhere*, no window degenerates, and `build` proceeds with no per-output
/// scan. Only when `support < ULP_above(c_max)` is absorption (hence a zero
/// tap) even possible; such a support cannot resolve across the output extent,
/// so `build` rejects it as [`ResampleError::InvalidFilterSupport`].
///
/// The ULP is sized by the actual `f64` subtraction `next_up(c_max) - c_max`
/// (not a hand-derived constant), so the comparison is exact. `c_max` is finite
/// and positive here (`scale > 0`, `out_size >= 1`), guarding [`next_up_f64`].
///
/// Comparing against the ULP *magnitude* — rather than absorption at the single
/// point `c_max` (`c_max - support == c_max`) — is what makes this sound: a
/// cleaner intermediate center (e.g. an exact integer) sharing `c_max`'s ULP
/// can absorb a `support` that `c_max`'s own low mantissa bits would keep
/// distinct, so testing one specific center could miss a real zero tap.
#[cfg_attr(not(tarpaulin), inline(always))]
fn support_absorbable_at_max_center(scale: f64, support: f64, out_size: usize) -> bool {
  let c_max = (out_size as f64 - 0.5) * scale;
  let ulp = next_up_f64(c_max) - c_max;
  support < ulp
}

/// `f64` `floor` portable across `std` and `no_std + alloc` builds,
/// mirroring the crate's `powf32` float-math gating: `std` uses the
/// inherent method, `no_std` opts into `libm` (gated by `alloc`).
#[cfg_attr(not(tarpaulin), inline(always))]
fn floor_f64(x: f64) -> f64 {
  #[cfg(feature = "std")]
  {
    f64::floor(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::floor(x)
  }
}

/// `f64` `ceil` portable across `std` and `no_std + alloc` builds. See
/// [`floor_f64`].
#[cfg_attr(not(tarpaulin), inline(always))]
fn ceil_f64(x: f64) -> f64 {
  #[cfg(feature = "std")]
  {
    f64::ceil(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::ceil(x)
  }
}

/// `f64` `round` (round-half-away-from-zero) portable across `std` and
/// `no_std + alloc` builds. See [`floor_f64`]. Snaps a normalized weight to
/// PIL's fixed-point grid in [`FilterAxis::build`]; `f64::round` is
/// round-half-away-from-zero, matching PIL's `int(0.5 + x)` for `x >= 0` and
/// `int(-0.5 + x)` for `x < 0`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn round_f64(x: f64) -> f64 {
  #[cfg(feature = "std")]
  {
    f64::round(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::round(x)
  }
}

/// `f64` `sin` portable across `std` and `no_std + alloc` builds. See
/// [`floor_f64`]. The Lanczos and windowed-sinc kernels need it.
#[cfg_attr(not(tarpaulin), inline(always))]
fn sin_f64(x: f64) -> f64 {
  #[cfg(feature = "std")]
  {
    f64::sin(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::sin(x)
  }
}

/// `f64` `cos` portable across `std` and `no_std + alloc` builds. See
/// [`floor_f64`]. Only the [`BlackmanSinc`] window needs it.
#[cfg_attr(not(tarpaulin), inline(always))]
fn cos_f64(x: f64) -> f64 {
  #[cfg(feature = "std")]
  {
    f64::cos(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::cos(x)
  }
}

/// `f64` `exp2` (base-2 exponential, `2^x`) portable across `std` and
/// `no_std + alloc` builds. See [`floor_f64`]. Only the [`Gaussian`] kernel
/// needs it.
#[cfg_attr(not(tarpaulin), inline(always))]
fn exp2_f64(x: f64) -> f64 {
  #[cfg(feature = "std")]
  {
    f64::exp2(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::exp2(x)
  }
}

/// A separable reconstruction-filter kernel: its `support` radius and its
/// `weight` profile. The window for an output sample spans the source
/// samples within `support` (scaled by the downscale ratio) of the
/// sample's projected center; [`FilterAxis::build`] evaluates `weight`
/// at each and normalizes.
///
/// **Public and unsealed** — implement this for a custom kernel. The
/// engine treats `weight` as an arbitrary (possibly negative) profile and
/// normalizes the resulting window to sum to one; it does **not** assume
/// non-negativity, so a kernel with negative lobes (Catmull-Rom, Lanczos)
/// is fully supported. [`support`](Self::support) must return a finite,
/// strictly positive, bounded radius — [`FilterAxis::build`] validates
/// this and rejects a hostile kernel with
/// [`ResampleError::InvalidFilterSupport`] rather than sizing an unsafe
/// window. All evaluation is in `f64`.
pub trait FilterKernel {
  /// Half-width of the kernel's nonzero region, in source-sample units
  /// at unit scale: `weight(x) == 0` for `|x| >= support()`. Must be
  /// finite, `> 0`, and bounded (see the trait docs).
  fn support(&self) -> f64;

  /// The (possibly negative) filter weight at signed offset `x` from the
  /// window center, in source-sample units at unit scale.
  fn weight(&self, x: f64) -> f64;
}

/// PIL `BILINEAR` — the triangle (linear / "tent") filter. Support 1; the
/// weight falls linearly from 1 at the center to 0 at `|x| = 1`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Triangle;

impl FilterKernel for Triangle {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    1.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
  }
}

/// PIL `BICUBIC` — the Catmull-Rom cubic (Keys, `a = -0.5`). Support 2,
/// with the standard two-segment piecewise-cubic profile and a negative
/// outer lobe on `1 <= |x| < 2`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CatmullRom;

impl CatmullRom {
  /// Keys parameter — `-0.5` reproduces PIL `BICUBIC`.
  const A: f64 = -0.5;
}

impl FilterKernel for CatmullRom {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    2.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    // PIL `bicubic_filter`: `(a+2)|x|^3 - (a+3)|x|^2 + 1` for |x| < 1;
    // `a|x|^3 - 5a|x|^2 + 8a|x| - 4a` for 1 <= |x| < 2; 0 beyond.
    let a = Self::A;
    let t = x.abs();
    if t < 1.0 {
      ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
    } else if t < 2.0 {
      (((t - 5.0) * t + 8.0) * t - 4.0) * a
    } else {
      0.0
    }
  }
}

/// PIL `LANCZOS` — the Lanczos filter with `a = 3`. Support 3;
/// `weight(x) = sinc(x) * sinc(x / 3)` for `|x| < 3`, zero beyond.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lanczos3;

impl Lanczos3 {
  /// Normalized sinc, `sin(pi t) / (pi t)`, with the removable
  /// singularity at `t == 0` defined as 1 (PIL's `sinc_filter`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn sinc(t: f64) -> f64 {
    if t == 0.0 {
      1.0
    } else {
      let pt = core::f64::consts::PI * t;
      sin_f64(pt) / pt
    }
  }
}

impl FilterKernel for Lanczos3 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    3.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    // PIL evaluates the windowed sinc on the half-open `[-3, 3)` and
    // zeroes the rest; the `< 3.0` guard matches that boundary exactly.
    if x > -3.0 && x < 3.0 {
      Self::sinc(x) * Self::sinc(x / 3.0)
    } else {
      0.0
    }
  }
}

/// The Mitchell-Netravali cubic (`B = C = 1/3`) — the high-quality general
/// cubic recommended by Mitchell & Netravali (1988) as the best subjective
/// trade-off between blurring and ringing. Support 2, with a small negative
/// outer lobe on `1 <= |x| < 2`. Not a PIL filter (PIL exposes no Mitchell);
/// validated against the closed-form Mitchell-Netravali weights.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Mitchell;

impl Mitchell {
  /// Mitchell-Netravali `B` (blur) parameter; `1/3` is the recommended value.
  const B: f64 = 1.0 / 3.0;
  /// Mitchell-Netravali `C` (ring) parameter; `1/3` is the recommended value.
  const C: f64 = 1.0 / 3.0;
}

impl FilterKernel for Mitchell {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    2.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    // Mitchell-Netravali piecewise cubic (the standard form, divided by 6):
    //   |x| < 1:       (12 - 9B - 6C)|x|^3 + (-18 + 12B + 6C)|x|^2 + (6 - 2B)
    //   1 <= |x| < 2:  (-B - 6C)|x|^3 + (6B + 30C)|x|^2 + (-12B - 48C)|x|
    //                  + (8B + 24C)
    // At B = C = 1/3 this is 8/9 at x = 0, 1/18 at |x| = 1, 0 at |x| = 2.
    let (b, c) = (Self::B, Self::C);
    let t = x.abs();
    if t < 1.0 {
      (((12.0 - 9.0 * b - 6.0 * c) * t + (-18.0 + 12.0 * b + 6.0 * c)) * t * t + (6.0 - 2.0 * b))
        / 6.0
    } else if t < 2.0 {
      ((((-b - 6.0 * c) * t + (6.0 * b + 30.0 * c)) * t + (-12.0 * b - 48.0 * c)) * t
        + (8.0 * b + 24.0 * c))
        / 6.0
    } else {
      0.0
    }
  }
}

/// OpenCV `INTER_CUBIC` — the Keys cubic with `a = -0.75` (the parameter
/// OpenCV/`cv2` uses, vs the `a = -0.5` of [`CatmullRom`]/PIL `BICUBIC`).
/// Support 2, the same two-segment piecewise-cubic profile as
/// [`CatmullRom`] with a deeper negative outer lobe. The kernel weights
/// match `cv2.resize(..., interpolation=cv2.INTER_CUBIC)`; exact pixel
/// parity additionally depends on the caller's coordinate convention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OpenCvCubic;

impl OpenCvCubic {
  /// Keys parameter — `-0.75` reproduces OpenCV `INTER_CUBIC`.
  const A: f64 = -0.75;
}

impl FilterKernel for OpenCvCubic {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    2.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    // Keys cubic (same form as `CatmullRom`, `a = -0.75`):
    // `(a+2)|x|^3 - (a+3)|x|^2 + 1` for |x| < 1;
    // `a|x|^3 - 5a|x|^2 + 8a|x| - 4a` for 1 <= |x| < 2; 0 beyond.
    let a = Self::A;
    let t = x.abs();
    if t < 1.0 {
      ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
    } else if t < 2.0 {
      (((t - 5.0) * t + 8.0) * t - 4.0) * a
    } else {
      0.0
    }
  }
}

/// FFmpeg `swscale` `SWS_BICUBIC` (the default scaler) — the Keys cubic
/// with `a = -0.6`, equivalently the Mitchell-Netravali cubic with
/// `B = 0`, `C = 0.6`. Support 2, the same two-segment piecewise-cubic
/// profile as [`CatmullRom`]/[`OpenCvCubic`], between their `a` values:
/// shallower outer lobe than `cv2`'s `a = -0.75` ([`OpenCvCubic`]),
/// deeper than PIL's `a = -0.5` ([`CatmullRom`]). The kernel weights match
/// swscale's default bicubic coefficient table; exact pixel parity
/// additionally depends on swscale's own coordinate/scaling pipeline.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SwscaleBicubic;

impl SwscaleBicubic {
  /// Keys parameter — `-0.6` reproduces swscale's default `SWS_BICUBIC`.
  /// Equivalent to the Mitchell-Netravali `B = 0`, `C = 0.6` form swscale
  /// builds its coefficients from: for `|x| < 1` that table is
  /// `((12 - 9B - 6C)|x|^3 + (-18 + 12B + 6C)|x|^2 + (6 - 2B)) / 6`, which
  /// at `B = 0`, `C = 0.6` is `1.4|x|^3 - 2.4|x|^2 + 1` — exactly the Keys
  /// `a = -0.6` inner segment `(a+2)|x|^3 - (a+3)|x|^2 + 1`.
  const A: f64 = -0.6;
}

impl FilterKernel for SwscaleBicubic {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    2.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    // Keys cubic (same form as `CatmullRom`/`OpenCvCubic`, `a = -0.6`):
    // `(a+2)|x|^3 - (a+3)|x|^2 + 1` for |x| < 1   (= 1.4|x|^3 - 2.4|x|^2 + 1);
    // `a|x|^3 - 5a|x|^2 + 8a|x| - 4a` for 1 <= |x| < 2
    //                              (= -0.6|x|^3 + 3|x|^2 - 4.8|x| + 2.4);
    // 0 beyond.
    let a = Self::A;
    let t = x.abs();
    if t < 1.0 {
      ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
    } else if t < 2.0 {
      (((t - 5.0) * t + 8.0) * t - 4.0) * a
    } else {
      0.0
    }
  }
}

/// OpenCV `INTER_LANCZOS4` — the Lanczos filter with `a = 4` (an 8-tap
/// window, vs the 6-tap `a = 3` of [`Lanczos3`]/PIL `LANCZOS`). Support 4;
/// `weight(x) = sinc(x) * sinc(x / 4)` for `|x| < 4`, zero beyond. The
/// kernel weights match `cv2.resize(..., interpolation=cv2.INTER_LANCZOS4)`;
/// exact pixel parity additionally depends on the caller's coordinate
/// convention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Lanczos4;

impl Lanczos4 {
  /// Normalized sinc, `sin(pi t) / (pi t)`, with the removable singularity
  /// at `t == 0` defined as 1 (the same `sinc_filter` as [`Lanczos3`]).
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn sinc(t: f64) -> f64 {
    if t == 0.0 {
      1.0
    } else {
      let pt = core::f64::consts::PI * t;
      sin_f64(pt) / pt
    }
  }
}

impl FilterKernel for Lanczos4 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    4.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    if x > -4.0 && x < 4.0 {
      Self::sinc(x) * Self::sinc(x / 4.0)
    } else {
      0.0
    }
  }
}

/// FFmpeg `swscale` `SWS_GAUSS` — the Gaussian filter `weight(x) =
/// 2^(-p * x^2)` with `p = 3` (swscale's default Gauss parameter). It has a
/// rounded passband with no ringing (no negative lobes, no zero crossings),
/// so unlike the cubic and Lanczos kernels it is **not interpolating** —
/// `weight` is nonzero at every nonzero integer — and is not a partition of
/// unity, but [`FilterAxis::build`] renormalizes every window so each output
/// still sums to one (preserving average brightness).
///
/// A Gaussian never reaches zero, so its support is a **truncation** choice.
/// swscale itself does not use a fixed support — it sizes the filter
/// dynamically from the scaling ratio — so this is a fixed-support
/// realization of the swscale Gauss *shape* (`exp2(-p d^2)`, `p = 3`), not
/// byte-exact to any particular swscale output size. The radius is fixed at
/// 3: at `|x| = 3` the weight is `2^(-27) ~ 7.5e-9`, far below `f32`
/// precision, so the truncation is numerically negligible (and the window
/// renormalization absorbs it regardless).
///
/// Because it is non-interpolating, a resize that changes only **one** axis
/// still convolves the other (scale-1) axis with the kernel — `weight(±1)`
/// is nonzero, so an unscaled axis is softened rather than passed through
/// untouched. This is inherent to every non-interpolating separable kernel
/// (e.g. [`Mitchell`]) and matches swscale, which likewise applies its filter
/// per axis; an interpolating kernel ([`CatmullRom`], the splines) instead
/// reproduces a scale-1 axis exactly because its integer taps are zero. Pick
/// an interpolating kernel if a per-axis resize must leave the other axis
/// bit-exact.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Gaussian;

impl Gaussian {
  /// swscale's default Gauss exponent parameter `p` in `2^(-p * x^2)`.
  const P: f64 = 3.0;
  /// Fixed truncation radius — see the type docs. At `|x| = 3` the weight
  /// is `2^(-P * 9) = 2^(-27)`, negligible at `f32` precision.
  const SUPPORT: f64 = 3.0;
}

impl FilterKernel for Gaussian {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    Self::SUPPORT
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    // swscale Gauss shape `2^(-p x^2)`, truncated to the half-open
    // `[-3, 3)` (the same boundary convention as the windowed-sinc
    // kernels). `exp2_f64` keeps the `no_std + alloc` build portable.
    if x.abs() < Self::SUPPORT {
      exp2_f64(-Self::P * x * x)
    } else {
      0.0
    }
  }
}

/// A **Blackman-windowed sinc** filter with radius `a = 3` — a windowed-sinc
/// reconstruction kernel distinct from the [`Lanczos3`]/[`Lanczos4`] family
/// (which window the sinc with a *sinc*). Support 3;
/// `weight(x) = sinc(x) * w_B(x)` for `|x| < 3`, zero beyond, where the
/// Blackman window is `w_B(x) = 0.42 + 0.5 cos(pi x / 3) + 0.08 cos(2 pi x /
/// 3)` over `|x| <= 3` (the conventional `0.42 / 0.5 / 0.08` Blackman
/// coefficients — not the "exact" `7938/9240/1430` variant). At `x = 0` the
/// window is `0.42 + 0.5 + 0.08 = 1`, so `weight(0) = 1`; at `|x| = 3` it is
/// `0.42 - 0.5 + 0.08 = 0`, so the window tapers to zero at the support
/// boundary.
///
/// Like the Lanczos kernels it is interpolating at the integer nodes (the
/// `sinc(x)` factor is zero at every nonzero integer), but the Blackman
/// window suppresses the side lobes more aggressively than Lanczos's sinc
/// window — a shallower outer lobe (`weight(1.5) ~ -0.072` vs Lanczos3's
/// `~ -0.135`) for less ringing, at a slightly softer passband. This is a
/// **reference-formula** windowed sinc; it is *not* byte-exact to FFmpeg
/// swscale's `SWS_SINC` (swscale sizes its sinc dynamically and uses its own
/// windowing/blur pipeline), which is an explicit non-goal — the bar is
/// exactness to the cited Blackman-sinc formula, in `f64`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlackmanSinc;

impl BlackmanSinc {
  /// Window / support radius `a` — a 6-tap window at unit scale (`a = 3`).
  const A: f64 = 3.0;

  /// Normalized sinc, `sin(pi t) / (pi t)`, with the removable singularity
  /// at `t == 0` defined as 1 (the same `sinc_filter` as [`Lanczos3`]).
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn sinc(t: f64) -> f64 {
    if t == 0.0 {
      1.0
    } else {
      let pt = core::f64::consts::PI * t;
      sin_f64(pt) / pt
    }
  }

  /// Conventional Blackman window `0.42 + 0.5 cos(pi x / a) + 0.08 cos(2 pi
  /// x / a)` on `|x| <= a`; the caller guards the `|x| < a` support, so this
  /// is only evaluated inside the window.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn window(x: f64) -> f64 {
    let u = core::f64::consts::PI * x / Self::A;
    0.42 + 0.5 * cos_f64(u) + 0.08 * cos_f64(2.0 * u)
  }
}

impl FilterKernel for BlackmanSinc {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    Self::A
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    // Windowed sinc on the half-open `[-3, 3)` (the same boundary convention
    // as the Lanczos and Gauss kernels); the window itself is zero at the
    // boundary, so the half-open guard is consistent with continuity.
    if x > -Self::A && x < Self::A {
      Self::sinc(x) * Self::window(x)
    } else {
      0.0
    }
  }
}

/// The **cubic B-spline** basis function `B3` (support 2) — the classic
/// *smoothing* cubic spline. It is **non-interpolating**: it blurs rather
/// than passing samples through (`weight(0) = 2/3`, `weight(±1) = 1/6`, so an
/// interior sample is spread across its neighbours), which makes it distinct
/// from the *interpolating* cubics ([`CatmullRom`], [`OpenCvCubic`],
/// [`SwscaleBicubic`]) and from the interpolating zimg [`Spline16`] /
/// [`Spline36`] / [`Spline64`] (those are different polynomial splines whose
/// integer taps are zero). It is, however, a **partition of unity** — every
/// shifted set of taps sums to one — so it preserves DC / average brightness,
/// and the engine renormalizes each window regardless.
///
/// Reference (the uniform cubic B-spline `B3`, the order-4 cardinal
/// B-spline):
///
/// ```text
/// |x| < 1:       (4 - 6|x|^2 + 3|x|^3) / 6
/// 1 <= |x| < 2:  (2 - |x|)^3 / 6
/// else:          0
/// ```
///
/// the smoothing cubic used by many imaging libraries' "cubic B-spline"
/// option (e.g. the `B = 1`, `C = 0` Mitchell-Netravali point). Validated
/// against the closed-form `B3` weights, in `f64`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubicBSpline;

impl FilterKernel for CubicBSpline {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    2.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    // Uniform cubic B-spline B3, evaluated in Horner form:
    //   |x| < 1:      (4 - 6 t^2 + 3 t^3) / 6
    //   1 <= |x| < 2: (2 - t)^3 / 6
    let t = x.abs();
    if t < 1.0 {
      (((3.0 * t - 6.0) * t) * t + 4.0) / 6.0
    } else if t < 2.0 {
      let u = 2.0 - t;
      u * u * u / 6.0
    } else {
      0.0
    }
  }
}

/// Evaluates the cubic `c0 + t*(c1 + t*(c2 + t*c3))` (Horner form) — the
/// per-segment polynomial of the zimg/Avisynth Spline kernels below.
#[cfg_attr(not(tarpaulin), inline(always))]
fn spline_poly3(t: f64, c0: f64, c1: f64, c2: f64, c3: f64) -> f64 {
  c0 + t * (c1 + t * (c2 + t * c3))
}

/// zimg / Avisynth **Spline16** — the 2-tap interpolating spline (support 2).
/// Interpolating (`weight(0) = 1`, zero at every other integer) and a
/// partition of unity, so it preserves DC; piecewise-cubic and continuous at
/// the knots (it is not globally `C1`, by construction). Sharper than a
/// bicubic at this support. The weights match zimg's `Spline16Filter`; exact
/// pixel parity additionally depends on the caller's coordinate convention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Spline16;

impl FilterKernel for Spline16 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    2.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    let t = x.abs();
    if t < 1.0 {
      spline_poly3(t, 1.0, -1.0 / 5.0, -9.0 / 5.0, 1.0)
    } else if t < 2.0 {
      spline_poly3(t - 1.0, 0.0, -7.0 / 15.0, 4.0 / 5.0, -1.0 / 3.0)
    } else {
      0.0
    }
  }
}

/// zimg / Avisynth **Spline36** — the 3-tap interpolating spline (support 3).
/// Like [`Spline16`] but with a wider support and a third lobe, trading a
/// little ringing for a flatter passband; a popular high-quality downscaler.
/// Interpolating + partition of unity. The weights match zimg's
/// `Spline36Filter`; exact pixel parity also depends on the caller's
/// coordinate convention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Spline36;

impl FilterKernel for Spline36 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    3.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    let t = x.abs();
    if t < 1.0 {
      spline_poly3(t, 1.0, -3.0 / 209.0, -453.0 / 209.0, 13.0 / 11.0)
    } else if t < 2.0 {
      spline_poly3(t - 1.0, 0.0, -156.0 / 209.0, 270.0 / 209.0, -6.0 / 11.0)
    } else if t < 3.0 {
      spline_poly3(t - 2.0, 0.0, 26.0 / 209.0, -45.0 / 209.0, 1.0 / 11.0)
    } else {
      0.0
    }
  }
}

/// zimg / Avisynth **Spline64** — the 4-tap interpolating spline (support 4),
/// the widest of the family: the flattest passband and sharpest cutoff, at
/// the cost of the most ringing. Interpolating + partition of unity. The
/// weights match zimg's `Spline64Filter`; exact pixel parity also depends on
/// the caller's coordinate convention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Spline64;

impl FilterKernel for Spline64 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn support(&self) -> f64 {
    4.0
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn weight(&self, x: f64) -> f64 {
    let t = x.abs();
    if t < 1.0 {
      spline_poly3(t, 1.0, -3.0 / 2911.0, -6387.0 / 2911.0, 49.0 / 41.0)
    } else if t < 2.0 {
      spline_poly3(
        t - 1.0,
        0.0,
        -2328.0 / 2911.0,
        4032.0 / 2911.0,
        -24.0 / 41.0,
      )
    } else if t < 3.0 {
      spline_poly3(t - 2.0, 0.0, 582.0 / 2911.0, -1008.0 / 2911.0, 6.0 / 41.0)
    } else if t < 4.0 {
      spline_poly3(t - 3.0, 0.0, -97.0 / 2911.0, 168.0 / 2911.0, -1.0 / 41.0)
    } else {
      0.0
    }
  }
}

/// Per-axis signed-coefficient spans of a filter
/// [`ResamplePlan`](super::ResamplePlan): for each output index, the first
/// contributing source sample plus the normalized (row-sums-to-one)
/// floating-point window. The signed twin of
/// [`AxisSpans`](super::AxisSpans) — a separate type so the integer area
/// arena is never perturbed.
///
/// Windows are stored row-major in `coeffs`, sliced for output index `j`
/// by `starts[j]` and the prefix `offsets`. Coefficients are `f32`
/// (normalized in `f64`, then narrowed) — the +-1-LSB parity budget
/// absorbs the narrowing.
///
/// A parallel `coeffs_q8` holds the same windows snapped to PIL's 8bpc
/// fixed-point grid ([`PRECISION_BITS`] = 22): the `u8` stream reads it for
/// **byte-exact** Pillow parity (see [`FilterAxis::build`]), while `u16`
/// and `f32` keep the full-precision float set. The two sets share
/// `starts` / `offsets`, so a window slices identically out of either.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FilterAxis {
  /// First contributing source sample per output index.
  starts: Vec<usize>,
  /// Prefix offsets into `coeffs` / `coeffs_q8`; `out_len() + 1` entries.
  /// Window `j`'s length is `offsets[j+1] - offsets[j]`.
  offsets: Vec<usize>,
  /// Concatenated per-window normalized weights (full `f32` precision).
  coeffs: Vec<f32>,
  /// Concatenated per-window weights snapped to PIL's 8bpc fixed-point
  /// grid: `round_half_away(w_norm * 2^PRECISION_BITS) / 2^PRECISION_BITS`,
  /// exactly representable in `f32` (`PRECISION_BITS <= 23`). The `u8`
  /// stream reads this set so both passes' `clip8` quantize identically to
  /// PIL's integer pipeline. Snapped from the full-precision `f64`
  /// normalized weight — never from the narrowed `coeffs` entry, whose
  /// `f32` cast would reintroduce an off-by-one in the snap.
  coeffs_q8: Vec<f32>,
  /// Maximum number of windows that contain any single source index — the
  /// peak count of output rows whose vertical window is open at one source
  /// row. Sizes [`FilterStream`]'s accumulator ring so no two open rows
  /// alias the same slot. (Vertical use only; harmless for the H axis.)
  max_overlap: usize,
}

/// PIL's 8bpc fixed-point coefficient scale exponent (`ImagingResample.c`
/// `PRECISION_BITS`): an 8bpc coefficient is `round(coeff * 2^PRECISION_BITS)`
/// as an integer, the pass accumulates those integers, and finalizes with
/// `(ss + 2^(PRECISION_BITS-1)) >> PRECISION_BITS`. `coeffs_q8` snaps to this
/// grid so the `u8` stream's `f64` arithmetic reproduces that pipeline
/// byte-for-byte. `22 <= 23` mantissa bits, so `K / 2^22` is exact in `f32`.
const PRECISION_BITS: u32 = 22;

impl FilterAxis {
  /// Builds the signed-coefficient spans for one `in_size -> out_size`
  /// axis under `kernel`, mirroring PIL `precompute_coeffs` exactly:
  ///
  /// ```text
  /// scale       = in_size / out_size                 (downscale: >= 1)
  /// filterscale = max(scale, 1)                       (==1 only at unit scale)
  /// support     = kernel.support() * filterscale
  /// for each output xx:
  ///   center = (xx + 0.5) * scale + phase
  ///   xmin   = max(0,       floor(center - support))
  ///   xmax   = min(in_size, ceil (center + support))  // exclusive
  ///   w[k]   = kernel.weight((xmin + k + 0.5 - center) / filterscale)
  ///   normalize so sum(w) == 1
  /// ```
  ///
  /// `filterscale` widens the kernel footprint for downscale (the
  /// anti-alias low-pass), and the `/ filterscale` argument scaling keeps
  /// the kernel's unit profile. Every window is renormalized after
  /// clamping to `[0, in_size)`, so edge windows (clipped on one side)
  /// still sum to one — preserving average brightness.
  ///
  /// `phase` is an additive sub-sample shift of every window center (in
  /// source-sample units): `center = (xx + 0.5) * scale + phase`. It carries a
  /// chroma axis's sampling position (siting); at `phase == 0.0` (co-sited, the
  /// only value this foundation passes) it is a literal no-op — adding `0.0` to
  /// the strictly-positive `center` leaves every window byte-identical, so the
  /// `u8` (`coeffs_q8`) and `u16` (`coeffs`) streams emit the same bytes.
  ///
  /// Alongside the full-precision `f32` set, each window is also snapped to
  /// PIL's 8bpc fixed-point grid into `coeffs_q8`: the un-normalized `f64`
  /// weights are retained for the window, and once the window sum `inv =
  /// 1/ww` is known each normalized weight `w_norm = w_f64 * inv` becomes
  /// `round_half_away(w_norm * 2^PRECISION_BITS) / 2^PRECISION_BITS`. The
  /// snap is taken from the full-precision `f64` weight — not the narrowed
  /// `coeffs` entry — so the `u8` stream's `f64` accumulation of
  /// `K_i/2^22 * pixel_i` equals PIL's `ss/2^22`, and its `clip8` finalize
  /// equals PIL's `(ss + 2^21) >> 22` clipped, byte-for-byte on both passes.
  ///
  /// # Errors
  ///
  /// - [`ResampleError::InvalidFilterSupport`] if `kernel.support()` is
  ///   not finite, not `> 0`, or so large the window cannot be sized
  ///   safely — a hostile kernel is rejected before any allocation.
  /// - [`ResampleError::Overflow`] if a window-arena length is
  ///   unrepresentable, [`ResampleError::AllocationFailed`] if an arena
  ///   reservation is refused (the planner's recoverable-allocation
  ///   contract).
  pub(crate) fn build(
    in_size: usize,
    out_size: usize,
    kernel: &dyn FilterKernel,
    phase: f64,
  ) -> Result<Self, ResampleError> {
    debug_assert!(in_size > 0 && out_size > 0);
    let support_unit = kernel.support();
    // A hostile `support` cannot size an unsafe window: reject only the
    // genuinely degenerate cases — non-finite or non-positive. A support
    // wider than the source is NOT rejected: that is the ordinary
    // narrow-source enlarge case (e.g. a `1x1 -> 7x7` Lanczos upscale), where
    // every window clamps to `[0, in_size)` and normalizes over the available
    // samples exactly as PIL does. The clamp bounds each window to at most
    // `in_size` samples regardless of the support's magnitude, so no finite
    // support can size an unbounded window.
    if !support_unit.is_finite() || support_unit <= 0.0 {
      return Err(ResampleError::InvalidFilterSupport(
        InvalidFilterSupport::new(support_unit, in_size),
      ));
    }

    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = support_unit * filterscale;
    // The clamp to `[0, in_size)` bounds every window to at most `in_size`
    // samples, whatever the support — so a wide support on a narrow source
    // collapses to the available samples rather than overrunning.
    let geometry = || PlanGeometry::new(in_size, 1, out_size, 1);

    // Overflow / capacity preflight, BEFORE any scan or allocation: reject an
    // `out_size` whose plan tables could never be allocated arithmetically.
    // `offsets` is the largest table at `out_size + 1` usizes; if even its
    // byte size overflows `usize` or exceeds the `isize::MAX` allocation cap,
    // no reservation could succeed, so fail fast here. This catches a hostile
    // `out_size == usize::MAX` in `O(1)` — without it the geometry validation
    // below could scan an astronomical index range first.
    let offsets_len = out_size
      .checked_add(1)
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;
    // The byte size of the largest table; its representability is the gate,
    // the value itself is unused beyond rejecting an unallocatable geometry.
    offsets_len
      .checked_mul(core::mem::size_of::<usize>())
      .filter(|&b| b <= isize::MAX as usize)
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;

    // Geometry validation against zero-tap windows, in `O(1)` — no per-output
    // scan. A sub-ULP support survives the finite / `> 0` / `<= in_size` checks
    // above yet collapses a window to zero taps where it falls below the `f64`
    // grid spacing at that center; the largest center `c_max` carries the
    // coarsest spacing, so a support below `c_max`'s ULP cannot be faithfully
    // evaluated across the output extent and is invalid for this geometry.
    // Reject it here, BEFORE sizing any plan table, so an invalid support never
    // allocates and the fill loop's `n > 0` (hence the overlap sweep's
    // `lo <= j`) is guaranteed. When the support is NOT absorbable even at
    // `c_max`, no window can degenerate, so we proceed; a hostile huge
    // `out_size` then fails fast at the bounded table reservation below.
    if support_absorbable_at_max_center(scale, support, out_size) {
      return Err(ResampleError::InvalidFilterSupport(
        InvalidFilterSupport::new(support_unit, in_size),
      ));
    }

    // Endpoint guard for the right-edge clamp. `scale` is the rounded f64
    // `in_size / out_size`, so the last center `((out_size - 1) + 0.5) * scale`
    // can round just past `in_size`; then `floor(center - support)` can exceed
    // the `min(in_size, .)`-clamped `xmax`, inverting the window. The last
    // output carries the largest center — with absorption ruled out above, the
    // only window the right clamp can invert — so validating it with the exact
    // fill-loop math proves every window is non-empty, in O(1) and before any
    // reservation.
    let last_center = ((out_size - 1) as f64 + 0.5) * scale;
    let last_lo = floor_f64(last_center - support);
    let last_xmin = if last_lo < 0.0 { 0 } else { last_lo as usize };
    let last_xmax = (ceil_f64(last_center + support) as usize).min(in_size);
    if last_xmax <= last_xmin {
      return Err(ResampleError::InvalidFilterSupport(
        InvalidFilterSupport::new(support_unit, in_size),
      ));
    }

    let mut starts = Vec::new();
    // The first plan-table reservation consults the test-only failpoint
    // (after the O(1) support validation, so a sub-grid support is rejected
    // before it can fire). On the non-test build the whole branch compiles away.
    #[cfg(all(test, feature = "std"))]
    if FORCE_FILTER_AXIS_ALLOC_FAILURE.with(|f| f.take()) {
      return Err(ResampleError::AllocationFailed(geometry()));
    }
    starts
      .try_reserve_exact(out_size)
      .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
    let mut ksize = Vec::new();
    ksize
      .try_reserve_exact(out_size)
      .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
    let mut offsets = Vec::new();
    offsets
      .try_reserve_exact(offsets_len)
      .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
    let mut coeffs: Vec<f32> = Vec::new();
    // Parallel 8bpc fixed-point set (`u8` stream): same windows snapped to
    // PIL's `PRECISION_BITS` grid. Grown per window in lockstep with
    // `coeffs`, so it shares `starts` / `offsets`.
    let mut coeffs_q8: Vec<f32> = Vec::new();
    // Scratch holding one window's un-normalized `f64` weights, reused
    // across outputs (cleared, capacity retained). The `q8` snap reads
    // these full-precision weights — not the narrowed `coeffs` entries —
    // so the fixed-point grid lands on PIL's exact integer.
    let mut w_f64: Vec<f64> = Vec::new();
    offsets.push(0);
    // `2^PRECISION_BITS`: the `f64` snap scale and the exact `f32` divisor.
    // `22 <= 23` mantissa bits, so `2^22` is exact in `f32` and `K / 2^22`
    // (any integer `K` in range) is exactly representable — no rounding the
    // snapped coefficient introduces.
    let q8_scale_f64 = f64::from(1u32 << PRECISION_BITS);
    let q8_scale_f32 = (1u32 << PRECISION_BITS) as f32;

    for xx in 0..out_size {
      let center = (xx as f64 + 0.5) * scale + phase;
      // `floor(center - support)` is >= some value >= -1 for the first
      // output, so the `max(0, .)` clamp is the only lower guard needed;
      // the cast is taken after the clamp so it never sees a negative.
      let lo = floor_f64(center - support);
      let xmin = if lo < 0.0 { 0 } else { lo as usize };
      let hi = ceil_f64(center + support);
      // `hi` is positive here (center > 0, support > 0), so the cast is
      // lossless before the min clamp to the exclusive source bound.
      let xmax = (hi as usize).min(in_size);
      // The support + endpoint validation above guarantees `xmax > xmin`;
      // `checked_sub` keeps that a recoverable error rather than an underflow
      // if hostile f64 rounding ever evaded those O(1) predicates.
      let Some(n) = xmax.checked_sub(xmin).filter(|&n| n > 0) else {
        return Err(ResampleError::InvalidFilterSupport(
          InvalidFilterSupport::new(support_unit, in_size),
        ));
      };

      // Grow both coeff arenas (and the scratch) one window at a time under
      // the recoverable contract; `n <= in_size` so each reservation is
      // bounded. `w_f64` is cleared first, so the reserve is a no-op once
      // its capacity has grown to the widest window seen.
      coeffs
        .try_reserve(n)
        .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
      coeffs_q8
        .try_reserve(n)
        .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
      w_f64.clear();
      w_f64
        .try_reserve(n)
        .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
      let mut ww = 0.0f64;
      let base = coeffs.len();
      for k in 0..n {
        let x = (xmin + k) as f64 + 0.5 - center;
        let w = kernel.weight(x / filterscale);
        coeffs.push(w as f32);
        w_f64.push(w);
        ww += w;
      }
      // PIL normalizes by the window sum; it is positive for every kernel
      // here (the central lobe dominates the negative tails). Guard the
      // degenerate `ww == 0` so a pathological custom kernel cannot divide
      // by zero — leave the window unnormalized rather than emit NaNs.
      let inv = if ww != 0.0 { 1.0 / ww } else { 1.0 };
      if ww != 0.0 {
        for c in &mut coeffs[base..base + n] {
          *c = (f64::from(*c) * inv) as f32;
        }
      }
      // 8bpc fixed-point snap from the full-precision normalized weight:
      // `K = round_half_away(w_norm * 2^PRECISION_BITS)`, coefficient
      // `K / 2^PRECISION_BITS` (exact in `f32`). When `ww == 0` the window
      // is left unnormalized (`inv == 1`), mirroring the float set.
      for &w in &w_f64 {
        let k = round_f64(w * inv * q8_scale_f64);
        coeffs_q8.push((k as f32) / q8_scale_f32);
      }

      starts.push(xmin);
      ksize.push(n);
      offsets.push(coeffs.len());
    }

    // Peak window overlap, by a two-pointer sweep: `starts` is
    // non-decreasing, so the windows open at the moment window `j` starts
    // are exactly those `i <= j` whose exclusive end `starts[i]+ksize[i]`
    // is still `> starts[j]`. Advancing `lo` past closed windows makes the
    // sweep linear. This is the tight ring capacity for the V axis.
    //
    // The `lo < j` guard bounds the lower pointer: window `j` is always open
    // at its own start (every window has `ksize >= 1` — a sub-grid support
    // that would degenerate a window is rejected up front), so `lo` never needs to
    // pass `j`. The guard makes the sweep robust even if that invariant were
    // ever weakened, so `lo` can never index past `starts`.
    let mut max_overlap = 0usize;
    let mut lo = 0usize;
    for j in 0..starts.len() {
      while lo < j && starts[lo] + ksize[lo] <= starts[j] {
        lo += 1;
      }
      let open = j - lo + 1;
      if open > max_overlap {
        max_overlap = open;
      }
    }

    Ok(Self {
      starts,
      offsets,
      coeffs,
      coeffs_q8,
      max_overlap,
    })
  }

  /// Number of output samples on this axis.
  // Consumed by the filter streaming engine (gated to routed families);
  // idle in the in-between cascade combos.
  #[cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn out_len(&self) -> usize {
    self.starts.len()
  }

  /// Peak count of windows open at any one source index — the
  /// [`FilterStream`] accumulator-ring capacity for this axis.
  #[cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn max_overlap(&self) -> usize {
    self.max_overlap
  }

  /// `(first source sample, full-precision window)` for output index `j`;
  /// `j` must be below [`Self::out_len`]. Read by the `u16` / `f32` V-pass.
  #[cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn span(&self, j: usize) -> (usize, &[f32]) {
    (
      self.starts[j],
      &self.coeffs[self.offsets[j]..self.offsets[j + 1]],
    )
  }

  /// `(first source sample, 8bpc fixed-point window)` for output index `j`;
  /// `j` must be below [`Self::out_len`]. Read by the `u8` V-pass so its
  /// finalize quantizes byte-for-byte with PIL. Shares `starts` / `offsets`
  /// with [`Self::span`], differing only in the snapped coefficient values.
  #[cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn span_q8(&self, j: usize) -> (usize, &[f32]) {
    (
      self.starts[j],
      &self.coeffs_q8[self.offsets[j]..self.offsets[j + 1]],
    )
  }

  /// Fallible deep copy following the planner's recoverable-allocation
  /// contract — used by [`FilterStream`] to own its geometry for the
  /// frame.
  #[cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]
  fn try_clone(&self) -> Result<Self, ResampleError> {
    fn copy<T: Copy>(src: &[T]) -> Result<Vec<T>, ResampleError> {
      let mut v = Vec::new();
      v.try_reserve_exact(src.len())
        .map_err(|_| ResampleError::AllocationFailed(PlanGeometry::new(0, 0, 0, 0)))?;
      v.extend_from_slice(src);
      Ok(v)
    }
    Ok(Self {
      starts: copy(&self.starts)?,
      offsets: copy(&self.offsets)?,
      coeffs: copy(&self.coeffs)?,
      coeffs_q8: copy(&self.coeffs_q8)?,
      max_overlap: self.max_overlap,
    })
  }
}

/// Payload for [`ResampleError::InvalidFilterSupport`] — the rejected
/// [`FilterKernel::support`] value and the source extent it was checked
/// against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidFilterSupport {
  /// The kernel-reported support radius that failed validation.
  support: f64,
  /// Source axis extent the support was bounded against.
  in_size: usize,
}

impl InvalidFilterSupport {
  /// Constructs a new `InvalidFilterSupport` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(support: f64, in_size: usize) -> Self {
    Self { support, in_size }
  }

  /// The kernel-reported support radius that failed validation.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn support(&self) -> f64 {
    self.support
  }

  /// Source axis extent the support was bounded against.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn in_size(&self) -> usize {
    self.in_size
  }
}

// The filter stream engine compiles for every family whose engine gate
// is widened (the same 14-feature cascade the area stream uses), ready to
// route as formats wire in; only `rgb` / `gray` actually consume it in
// this stage (Rgb24 / Rgb48 / Grayf32), so its shared accessors carry a
// targeted dead-code allow for the in-between combos.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
mod stream;
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
))]
pub(crate) use stream::{FilterSample, FilterStream};

#[cfg(all(test, feature = "std"))]
mod tests;
