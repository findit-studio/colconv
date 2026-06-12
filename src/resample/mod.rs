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
//! strategy.

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

/// Output-geometry product of [`Resampler::plan`], built once at
/// sinker construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResamplePlan {
  /// Output width in pixels.
  out_w: usize,
  /// Output height in pixels.
  out_h: usize,
}

impl ResamplePlan {
  /// Constructs a plan with the given output geometry.
  ///
  /// Gated to `std` test builds alongside [`test_support`] — its only
  /// callers — so feature-powerset test builds without `std` don't
  /// carry a dead constructor under `-D warnings`.
  #[cfg(all(test, feature = "std"))]
  pub(crate) const fn new(out_w: usize, out_h: usize) -> Self {
    Self { out_w, out_h }
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
}

/// Test-only strategies exercising the geometry split without a real
/// resampling engine. Gated to `std` test builds: every consumer
/// (this module's own tests and the sinker geometry tests) is
/// `std`-gated, and a plain `cfg(test)` gate would leave these
/// fixtures dead — and denied — in feature-powerset test builds
/// without `std`.
#[cfg(all(test, feature = "std"))]
pub(crate) mod test_support {
  use super::{ResampleError, ResamplePlan, Resampler, ZeroOutputDimension, sealed::Sealed};

  /// Plans a fixed output geometry regardless of source geometry.
  pub(crate) struct FixedDownscale {
    out_w: usize,
    out_h: usize,
  }

  impl FixedDownscale {
    pub(crate) const fn new(out_w: usize, out_h: usize) -> Self {
      Self { out_w, out_h }
    }
  }

  impl Sealed for FixedDownscale {}

  impl Resampler for FixedDownscale {
    fn plan(&self, _src_w: usize, _src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
      Ok(Some(ResamplePlan::new(self.out_w, self.out_h)))
    }
  }

  /// Always rejects the plan, for error-propagation tests.
  pub(crate) struct AlwaysFails;

  impl Sealed for AlwaysFails {}

  impl Resampler for AlwaysFails {
    fn plan(&self, _src_w: usize, _src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
      Err(ResampleError::ZeroOutputDimension(
        ZeroOutputDimension::new(0, 0),
      ))
    }
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
