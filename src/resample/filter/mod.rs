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
//! matches PIL `LANCZOS` — reproduced here within +-1 LSB using `f64`
//! coefficients rather than PIL's integer fixed-point.
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
//! Downscale only in this stage; upscaling is rejected
//! ([`ResampleError::UpscaleUnsupported`]). SIMD backends and the upscale
//! direction land in later increments — this is the scalar foundation.

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

/// Whether any output window over an `in_size -> out_size` axis under this
/// `scale` / `support` would degenerate to zero taps — the geometry-only
/// (no-allocation) twin of the per-output window math in
/// [`FilterAxis::build`]. A sub-ULP positive support survives build's
/// finite / `> 0` / `<= in_size` checks yet can round
/// `floor(center - support)` and `ceil(center + support)` to the same
/// integer when `center` is integral, leaving `xmin == xmax` and an empty
/// window that covers no source sample. `build` runs this BEFORE sizing any
/// plan table so such a kernel is rejected ([`ResampleError::InvalidFilterSupport`])
/// ahead of allocation, and the window fill loop relies on its guarantee
/// (`n > 0`) to keep the overlap sweep's `lo <= j` invariant.
#[cfg_attr(not(tarpaulin), inline(always))]
fn first_zero_tap_window(scale: f64, support: f64, in_size: usize, out_size: usize) -> bool {
  for xx in 0..out_size {
    let center = (xx as f64 + 0.5) * scale;
    let lo = floor_f64(center - support);
    let xmin = if lo < 0.0 { 0 } else { lo as usize };
    let hi = ceil_f64(center + support);
    let xmax = (hi as usize).min(in_size);
    if xmax == xmin {
      return true;
    }
  }
  false
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

/// `f64` `sin` portable across `std` and `no_std + alloc` builds. See
/// [`floor_f64`]. Only the Lanczos kernel needs it.
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
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FilterAxis {
  /// First contributing source sample per output index.
  starts: Vec<usize>,
  /// Prefix offsets into `coeffs`; `out_len() + 1` entries. Window `j`'s
  /// length is `offsets[j+1] - offsets[j]`.
  offsets: Vec<usize>,
  /// Concatenated per-window normalized weights.
  coeffs: Vec<f32>,
  /// Maximum number of windows that contain any single source index — the
  /// peak count of output rows whose vertical window is open at one source
  /// row. Sizes [`FilterStream`]'s accumulator ring so no two open rows
  /// alias the same slot. (Vertical use only; harmless for the H axis.)
  max_overlap: usize,
}

impl FilterAxis {
  /// Builds the signed-coefficient spans for one `in_size -> out_size`
  /// axis under `kernel`, mirroring PIL `precompute_coeffs` exactly:
  ///
  /// ```text
  /// scale       = in_size / out_size                 (downscale: >= 1)
  /// filterscale = max(scale, 1)                       (==1 only at unit scale)
  /// support     = kernel.support() * filterscale
  /// for each output xx:
  ///   center = (xx + 0.5) * scale
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
  ) -> Result<Self, ResampleError> {
    debug_assert!(in_size > 0 && out_size > 0);
    let support_unit = kernel.support();
    // A hostile `support` cannot size an unsafe window: reject anything
    // non-finite, non-positive, or large enough that `ceil(support)`
    // would not fit a sane window bound. `in_size` is the natural cap —
    // no window can be wider than the source — so a support past it is
    // both pointless and a red flag.
    if !support_unit.is_finite() || support_unit <= 0.0 || support_unit > in_size as f64 {
      return Err(ResampleError::InvalidFilterSupport(
        InvalidFilterSupport::new(support_unit, in_size),
      ));
    }

    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = support_unit * filterscale;
    // Window width is bounded by `2*support + 2` (the floor/ceil span);
    // with `support <= in_size * filterscale` and the clamp to
    // `[0, in_size)`, no window exceeds `in_size` samples.
    let geometry = || PlanGeometry::new(in_size, 1, out_size, 1);

    // No-allocation dry pass over the output window geometry: a sub-ULP
    // support can survive the validation above yet collapse a window to
    // zero taps (see `first_zero_tap_window`). Reject such a kernel here,
    // BEFORE sizing any plan table, so an invalid support never allocates —
    // and so the fill loop's `n > 0` (hence the overlap sweep's `lo <= j`)
    // is guaranteed.
    if first_zero_tap_window(scale, support, in_size, out_size) {
      return Err(ResampleError::InvalidFilterSupport(
        InvalidFilterSupport::new(support_unit, in_size),
      ));
    }

    let mut starts = Vec::new();
    // The first plan-table reservation consults the test-only failpoint
    // (after the dry pass, so a zero-tap kernel is rejected before it can
    // fire). On the non-test build the whole branch compiles away.
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
    let offsets_len = out_size
      .checked_add(1)
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;
    let mut offsets = Vec::new();
    offsets
      .try_reserve_exact(offsets_len)
      .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
    let mut coeffs: Vec<f32> = Vec::new();
    offsets.push(0);

    for xx in 0..out_size {
      let center = (xx as f64 + 0.5) * scale;
      // `floor(center - support)` is >= some value >= -1 for the first
      // output, so the `max(0, .)` clamp is the only lower guard needed;
      // the cast is taken after the clamp so it never sees a negative.
      let lo = floor_f64(center - support);
      let xmin = if lo < 0.0 { 0 } else { lo as usize };
      let hi = ceil_f64(center + support);
      // `hi` is positive here (center > 0, support > 0), so the cast is
      // lossless before the min clamp to the exclusive source bound.
      let xmax = (hi as usize).min(in_size);
      let n = xmax - xmin;
      // The dry pass above rejected any zero-tap window, so every window
      // here covers at least one source sample.
      debug_assert!(n > 0);

      // Grow the coeff arena one window at a time under the recoverable
      // contract; `n <= in_size` so each reservation is bounded.
      coeffs
        .try_reserve(n)
        .map_err(|_| ResampleError::AllocationFailed(geometry()))?;
      let mut ww = 0.0f64;
      let base = coeffs.len();
      for k in 0..n {
        let x = (xmin + k) as f64 + 0.5 - center;
        let w = kernel.weight(x / filterscale);
        coeffs.push(w as f32);
        ww += w;
      }
      // PIL normalizes by the window sum; it is positive for every kernel
      // here (the central lobe dominates the negative tails). Guard the
      // degenerate `ww == 0` so a pathological custom kernel cannot divide
      // by zero — leave the window unnormalized rather than emit NaNs.
      if ww != 0.0 {
        let inv = 1.0 / ww;
        for c in &mut coeffs[base..base + n] {
          *c = (f64::from(*c) * inv) as f32;
        }
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
    // at its own start (every window has `ksize >= 1` — zero-tap windows are
    // rejected by the dry pass before this point), so `lo` never needs to
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

  /// `(first source sample, normalized window)` for output index `j`;
  /// `j` must be below [`Self::out_len`].
  #[cfg_attr(not(any(feature = "rgb", feature = "gray")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn span(&self, j: usize) -> (usize, &[f32]) {
    (
      self.starts[j],
      &self.coeffs[self.offsets[j]..self.offsets[j + 1]],
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
