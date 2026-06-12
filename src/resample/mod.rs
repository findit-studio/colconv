//! Resampling strategies for [`MixedSinker`](crate::sinker::MixedSinker).
//!
//! A [`Resampler`] is injected at sinker construction and decides the
//! sinker's **output geometry** once, before any output buffer
//! attaches: [`Resampler::plan`] returns `Ok(None)` for the identity
//! (output geometry == source geometry — the sinker takes the direct
//! conversion path) or `Ok(Some(plan))` carrying the output geometry
//! that every output buffer is then validated against. The walker-side
//! contract is unchanged either way:
//! [`PixelSink::begin_frame`](crate::PixelSink::begin_frame) keeps
//! validating frames against the **source** geometry.
//!
//! The trait is sealed — resampling strategies ship with this crate.
//! [`NoopResampler`], the default `R` of
//! [`MixedSinker`](crate::sinker::MixedSinker), is the identity
//! strategy; [`AreaResampler`] plans area (box-coverage) downscales.
//! The streaming engine that executes non-identity plans lands with
//! the per-format route dispatch.

use std::vec::Vec;

use derive_more::{IsVariant, TryUnwrap, Unwrap};
use thiserror::Error;

/// Decides a [`MixedSinker`](crate::sinker::MixedSinker)'s output
/// geometry from its source geometry, once, at construction.
///
/// Sealed — only strategies shipped by this crate implement it.
pub trait Resampler: sealed::Sealed {
  /// Builds the resampling plan for a `src_w x src_h` source frame.
  ///
  /// `Ok(None)` means the resampling is the identity: output geometry
  /// equals source geometry and the sinker takes the direct conversion
  /// path with no resampling state at all.
  ///
  /// # Errors
  ///
  /// Strategy-specific validation of the requested output geometry —
  /// see [`ResampleError`] for the variants.
  fn plan(&self, src_w: usize, src_h: usize) -> Result<Option<ResamplePlan>, ResampleError>;
}

mod sealed {
  pub trait Sealed {}
}

/// Identity strategy and the default `R` of
/// [`MixedSinker`](crate::sinker::MixedSinker): [`Resampler::plan`]
/// always returns `Ok(None)`, so output geometry equals source
/// geometry and the sinker behaves exactly like a non-resampling sink.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoopResampler;

impl sealed::Sealed for NoopResampler {}

impl Resampler for NoopResampler {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn plan(&self, _src_w: usize, _src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
    Ok(None)
  }
}

/// Area (box-coverage) downscale strategy — the `cv2.INTER_AREA`
/// convention that analysis pipelines are calibrated against. Plans
/// exact integer coverage spans on both axes, fractional ratios
/// included (1920 -> 336 is a x40/7 scale). Requesting the source
/// geometry plans the identity (`Ok(None)`); upscaling on either axis
/// is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AreaResampler {
  out_w: usize,
  out_h: usize,
}

impl AreaResampler {
  /// Strategy producing an `out_w x out_h` output frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn to(out_w: usize, out_h: usize) -> Self {
    Self { out_w, out_h }
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

impl sealed::Sealed for AreaResampler {}

impl Resampler for AreaResampler {
  fn plan(&self, src_w: usize, src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
    if self.out_w == 0 || self.out_h == 0 {
      return Err(ResampleError::ZeroOutputDimension(
        ZeroOutputDimension::new(self.out_w, self.out_h),
      ));
    }
    if self.out_w > src_w || self.out_h > src_h {
      return Err(ResampleError::UpscaleUnsupported(UpscaleUnsupported::new(
        src_w, src_h, self.out_w, self.out_h,
      )));
    }
    if self.out_w == src_w && self.out_h == src_h {
      return Ok(None);
    }
    ResamplePlan::area(src_w, src_h, self.out_w, self.out_h).map(Some)
  }
}

/// Per-axis area-coverage spans of a [`ResamplePlan`]: for each output
/// index, the first contributing source cell plus the integer overlap
/// weight of every contributing cell.
///
/// Geometry lives on the axis's `x out` integer grid — output pixel
/// `j` covers `[j * src, (j + 1) * src)` and source cell `i` covers
/// `[i * out, (i + 1) * out)` — so weights are exact for fractional
/// ratios and every span sums to `src`, the normalization denominator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AxisSpans {
  /// First contributing source cell per output index.
  starts: Vec<usize>,
  /// Prefix offsets into `weights`; `out_len() + 1` entries.
  offsets: Vec<usize>,
  /// Concatenated per-span overlap weights.
  weights: Vec<usize>,
}

impl AxisSpans {
  /// Builds the exact area spans for one `src -> out` axis. `None`
  /// when `src * out` overflows `usize`; that product bounds every
  /// intermediate term, so checking it once up front (before any
  /// allocation) keeps the loop in plain arithmetic.
  fn area(src: usize, out: usize) -> Option<Self> {
    src.checked_mul(out)?;
    let mut starts = Vec::with_capacity(out);
    let mut offsets = Vec::with_capacity(out + 1);
    let mut weights = Vec::new();
    offsets.push(0);
    for j in 0..out {
      let lo = j * src;
      let hi = lo + src;
      let start = lo / out;
      starts.push(start);
      for i in start..hi.div_ceil(out) {
        let cell_lo = i * out;
        let cell_hi = cell_lo + out;
        weights.push(cell_hi.min(hi) - cell_lo.max(lo));
      }
      offsets.push(weights.len());
    }
    Some(Self {
      starts,
      offsets,
      weights,
    })
  }

  /// Number of output samples on this axis.
  // Consumed by the std-gated tests until the streaming engine becomes
  // the first non-test caller; the allowance disappears with it.
  #[cfg_attr(not(all(test, feature = "std")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn out_len(&self) -> usize {
    self.starts.len()
  }

  /// `(first source cell, overlap weights)` for output index `j`;
  // Consumed by the std-gated tests until the streaming engine becomes
  // the first non-test caller; the allowance disappears with it.
  #[cfg_attr(not(all(test, feature = "std")), allow(dead_code))]
  /// `j` must be below [`Self::out_len`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn span(&self, j: usize) -> (usize, &[usize]) {
    (
      self.starts[j],
      &self.weights[self.offsets[j]..self.offsets[j + 1]],
    )
  }
}

/// Output-geometry product of [`Resampler::plan`], built once at
/// sinker construction. Carries the per-axis area spans the streaming
/// engine consumes; the source dimensions double as the spans'
/// normalization denominators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResamplePlan {
  src_w: usize,
  src_h: usize,
  out_w: usize,
  out_h: usize,
  h: AxisSpans,
  v: AxisSpans,
}

impl ResamplePlan {
  /// Builds the exact area plan for `src -> out`. The strategy has
  /// already validated zero, upscale, and identity geometry.
  fn area(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Result<Self, ResampleError> {
    match (AxisSpans::area(src_w, out_w), AxisSpans::area(src_h, out_h)) {
      (Some(h), Some(v)) => Ok(Self {
        src_w,
        src_h,
        out_w,
        out_h,
        h,
        v,
      }),
      _ => Err(ResampleError::Overflow(PlanOverflow::new(
        src_w, src_h, out_w, out_h,
      ))),
    }
  }

  /// Source width in pixels — the horizontal spans' normalization
  /// denominator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_w(&self) -> usize {
    self.src_w
  }

  /// Source height in pixels — the vertical spans' normalization
  /// denominator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_h(&self) -> usize {
    self.src_h
  }

  /// Horizontal-axis spans.
  // Consumed by the std-gated tests until the streaming engine becomes
  // the first non-test caller; the allowance disappears with it.
  #[cfg_attr(not(all(test, feature = "std")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn h(&self) -> &AxisSpans {
    &self.h
  }

  /// Vertical-axis spans.
  // Consumed by the std-gated tests until the streaming engine becomes
  // the first non-test caller; the allowance disappears with it.
  #[cfg_attr(not(all(test, feature = "std")), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn v(&self) -> &AxisSpans {
    &self.v
  }

  /// Output width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Output height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }

  /// Output `(width, height)` pair.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_dims(&self) -> (usize, usize) {
    (self.out_w, self.out_h)
  }
}

/// Source vs requested-output geometry payload for
/// [`ResampleError::UpscaleUnsupported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpscaleUnsupported {
  /// Source width.
  src_w: usize,
  /// Source height.
  src_h: usize,
  /// Requested output width.
  out_w: usize,
  /// Requested output height.
  out_h: usize,
}

impl UpscaleUnsupported {
  /// Constructs a new `UpscaleUnsupported` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Self {
    Self {
      src_w,
      src_h,
      out_w,
      out_h,
    }
  }

  /// Source width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_w(&self) -> usize {
    self.src_w
  }

  /// Source height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_h(&self) -> usize {
    self.src_h
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

/// Requested-output geometry payload for
/// [`ResampleError::ZeroOutputDimension`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroOutputDimension {
  /// Requested output width.
  out_w: usize,
  /// Requested output height.
  out_h: usize,
}

impl ZeroOutputDimension {
  /// Constructs a new `ZeroOutputDimension` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(out_w: usize, out_h: usize) -> Self {
    Self { out_w, out_h }
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

/// Geometry payload for [`ResampleError::Overflow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanOverflow {
  /// Source width.
  src_w: usize,
  /// Source height.
  src_h: usize,
  /// Requested output width.
  out_w: usize,
  /// Requested output height.
  out_h: usize,
}

impl PlanOverflow {
  /// Constructs a new `PlanOverflow` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Self {
    Self {
      src_w,
      src_h,
      out_w,
      out_h,
    }
  }

  /// Source width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_w(&self) -> usize {
    self.src_w
  }

  /// Source height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_h(&self) -> usize {
    self.src_h
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

/// Errors returned by [`Resampler::plan`] while validating the
/// requested output geometry at sinker construction.
///
/// All variants surface before the sinker exists and before any output
/// buffer attaches, so they are always recoverable: fix the requested
/// geometry and construct again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, IsVariant, TryUnwrap, Unwrap, Error)]
#[non_exhaustive]
pub enum ResampleError {
  /// The requested output geometry exceeds the source geometry on at
  /// least one axis and the strategy only downscales.
  #[error(
    "resample output {}x{} exceeds source {}x{} (upscaling is unsupported)",
    .0.out_w(), .0.out_h(), .0.src_w(), .0.src_h()
  )]
  UpscaleUnsupported(UpscaleUnsupported),

  /// The requested output geometry has a zero side.
  #[error(
    "resample output dimensions must be nonzero, got {}x{}",
    .0.out_w(), .0.out_h()
  )]
  ZeroOutputDimension(ZeroOutputDimension),

  /// Building the span tables would overflow `usize`: a per-axis
  /// `source x output` product is unrepresentable. Only reachable
  /// with extreme dimensions (32-bit targets foremost).
  #[error(
    "resample plan geometry overflows usize: source {}x{}, output {}x{}",
    .0.src_w(), .0.src_h(), .0.out_w(), .0.out_h()
  )]
  Overflow(PlanOverflow),
}

#[cfg(all(test, feature = "std"))]
mod tests;
