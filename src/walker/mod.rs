//! A uniform `Walker` layer over the per-format frame walkers.
//!
//! Every source pixel format already ships a free walker fn
//! (`xyz12_to`, `yuv420p_to`, `bayer_to`, …) that iterates a
//! `crate::frame::*Frame` row-by-row and dispatches each row to a
//! [`PixelSink`]. Those fns each take their own bespoke value
//! parameters — `xyz12_to` takes a [`DcpTargetGamut`], the YUV walkers
//! take a `full_range` flag plus a [`ColorMatrix`], the Bayer walkers
//! take a pattern / demosaic / white-balance / colour-correction
//! bundle. [`Walker`] unifies them behind one associated-fn surface:
//! the per-format conversion knobs move into a format-specific
//! `Options` value type ([`Xyz12Options`], [`YuvOptions`],
//! [`BayerOptions`]), and [`Walker::walk`] forwards them to the
//! underlying free fn.
//!
//! This module is **purely additive** — it sits on top of the existing
//! walkers and sinks and changes none of their behaviour. The marker
//! types it implements [`Walker`] for are mediaframe's foreign
//! `crate::source::*` ZSTs; [`Walker`] is colconv's own local trait, so
//! the impls satisfy the orphan rule (a local trait for a foreign
//! type).

#[cfg(all(test, feature = "std"))]
mod tests;

#[cfg(feature = "xyz")]
use crate::frame::Xyz12Frame;
#[cfg(feature = "xyz")]
use crate::source::{Xyz12, Xyz12Sink, xyz12_to};
use crate::{ColorMatrix, DcpTargetGamut, PixelSink};

/// A uniform entry point over a source format's frame walker.
///
/// `S` is the [`PixelSink`] implementation the rows are dispatched to.
/// Implementors are the per-format marker ZSTs from
/// [`crate::source`]; each names the matching frame borrow as
/// [`Frame`](Self::Frame) and its conversion knobs as
/// [`Options`](Self::Options).
///
/// [`walk`](Self::walk) is an associated fn (no `&self`) — the marker
/// is a ZST and carries no state, so the walk is fully described by the
/// frame, the options, and the sink.
pub trait Walker<S> {
  /// The validated source frame borrow this walker iterates — e.g.
  /// [`Xyz12Frame`] for the XYZ12 source.
  type Frame<'a>;

  /// The per-format conversion options forwarded to the underlying
  /// walker fn — e.g. [`Xyz12Options`] for the XYZ12 source.
  type Options;

  /// Walks `src` row by row, applying `opts`, dispatching each row to
  /// `sink`.
  fn walk(src: &Self::Frame<'_>, opts: &Self::Options, sink: &mut S) -> Result<(), S::Error>
  where
    S: PixelSink;
}

/// Conversion options for the XYZ12 ([`Xyz12`]) source — the target RGB
/// gamut its inverse-OETF + 3×3 matrix converts into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Xyz12Options {
  target_gamut: DcpTargetGamut,
}

impl Xyz12Options {
  /// Creates options with the default target gamut
  /// ([`DcpTargetGamut`]'s own default — `DciP3`, the SMPTE ST 428-1
  /// D-Cinema decode target).
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      target_gamut: DcpTargetGamut::DciP3,
    }
  }

  /// The target RGB gamut the XYZ → RGB matrix converts into.
  #[inline(always)]
  pub const fn target_gamut(&self) -> DcpTargetGamut {
    self.target_gamut
  }

  /// Sets the target RGB gamut (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_target_gamut(mut self, target_gamut: DcpTargetGamut) -> Self {
    self.target_gamut = target_gamut;
    self
  }
}

impl Default for Xyz12Options {
  #[inline(always)]
  fn default() -> Self {
    Self::new()
  }
}

/// Conversion options shared by the YUV-family sources — the
/// quantisation range (`full_range`) and the YCbCr [`ColorMatrix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct YuvOptions {
  full_range: bool,
  matrix: ColorMatrix,
}

impl YuvOptions {
  /// Creates options for limited-range [`ColorMatrix::Bt709`] — the
  /// implicit default of the common HD YUV pipeline.
  #[inline(always)]
  pub const fn new() -> Self {
    Self {
      full_range: false,
      matrix: ColorMatrix::Bt709,
    }
  }

  /// Whether the source samples are full-range (`true`) or
  /// limited/studio-range (`false`).
  #[inline(always)]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }

  /// The YCbCr matrix the source was encoded with.
  #[inline(always)]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }

  /// Marks the source as full-range (`true`) in place.
  #[inline(always)]
  pub const fn set_full_range(&mut self) -> &mut Self {
    self.full_range = true;
    self
  }

  /// Marks the source as full-range (`true`), consuming builder.
  #[must_use]
  #[inline(always)]
  pub const fn with_full_range(mut self) -> Self {
    self.full_range = true;
    self
  }

  /// Assigns the raw `full_range` flag in place.
  #[inline(always)]
  pub const fn update_full_range(&mut self, full_range: bool) -> &mut Self {
    self.full_range = full_range;
    self
  }

  /// Assigns the raw `full_range` flag, consuming builder.
  #[must_use]
  #[inline(always)]
  pub const fn maybe_full_range(mut self, full_range: bool) -> Self {
    self.full_range = full_range;
    self
  }

  /// Marks the source as limited/studio-range (`false`) in place.
  #[inline(always)]
  pub const fn clear_full_range(&mut self) -> &mut Self {
    self.full_range = false;
    self
  }

  /// Sets the YCbCr matrix (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_matrix(mut self, matrix: ColorMatrix) -> Self {
    self.matrix = matrix;
    self
  }
}

impl Default for YuvOptions {
  #[inline(always)]
  fn default() -> Self {
    Self::new()
  }
}

/// Conversion options for the Bayer ([`raw::Bayer`](crate::raw::Bayer))
/// sources — the mosaic `pattern`, the `demosaic` algorithm, the
/// white-balance `wb` gains, and the colour-correction matrix `ccm`.
///
/// There is **no `Default`**: the [`BayerPattern`](crate::raw::BayerPattern)
/// is frame-intrinsic (it describes the sensor's mosaic and cannot be
/// guessed), so callers must name it via [`new`](Self::new). This is
/// why [`Walker`] does not bound `Options: Default`.
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BayerOptions {
  pattern: crate::raw::BayerPattern,
  demosaic: crate::raw::BayerDemosaic,
  wb: crate::raw::WhiteBalance,
  ccm: crate::raw::ColorCorrectionMatrix,
}

#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
impl BayerOptions {
  /// Creates options for the given mosaic `pattern`, defaulting the
  /// demosaic to [`BayerDemosaic::Bilinear`](crate::raw::BayerDemosaic),
  /// the white balance to
  /// [`WhiteBalance::neutral`](crate::raw::WhiteBalance::neutral), and
  /// the colour-correction matrix to
  /// [`ColorCorrectionMatrix::identity`](crate::raw::ColorCorrectionMatrix::identity).
  #[inline(always)]
  pub const fn new(pattern: crate::raw::BayerPattern) -> Self {
    Self {
      pattern,
      demosaic: crate::raw::BayerDemosaic::Bilinear,
      wb: crate::raw::WhiteBalance::neutral(),
      ccm: crate::raw::ColorCorrectionMatrix::identity(),
    }
  }

  /// The sensor's Bayer mosaic pattern.
  #[inline(always)]
  pub const fn pattern(&self) -> crate::raw::BayerPattern {
    self.pattern
  }

  /// The demosaic reconstruction algorithm.
  #[inline(always)]
  pub const fn demosaic(&self) -> crate::raw::BayerDemosaic {
    self.demosaic
  }

  /// The per-channel white-balance gains.
  #[inline(always)]
  pub const fn wb(&self) -> crate::raw::WhiteBalance {
    self.wb
  }

  /// The 3×3 colour-correction matrix applied after white balance.
  #[inline(always)]
  pub const fn ccm(&self) -> crate::raw::ColorCorrectionMatrix {
    self.ccm
  }

  /// Sets the demosaic algorithm (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_demosaic(mut self, demosaic: crate::raw::BayerDemosaic) -> Self {
    self.demosaic = demosaic;
    self
  }

  /// Sets the white-balance gains (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_wb(mut self, wb: crate::raw::WhiteBalance) -> Self {
    self.wb = wb;
    self
  }

  /// Sets the colour-correction matrix (consuming builder).
  #[must_use]
  #[inline(always)]
  pub const fn with_ccm(mut self, ccm: crate::raw::ColorCorrectionMatrix) -> Self {
    self.ccm = ccm;
    self
  }
}

// `Walker` is colconv's local trait and `Xyz12<BE>` is mediaframe's
// foreign marker, so this impl is allowed under the orphan rule. The
// single per-impl `where S: Xyz12Sink<BE>` is the bound `xyz12_to`
// requires; the trait's method-scoped `where S: PixelSink` is implied
// by it (`Xyz12Sink<BE>: PixelSink`).
#[cfg(feature = "xyz")]
#[cfg_attr(docsrs, doc(cfg(feature = "xyz")))]
impl<const BE: bool, S> Walker<S> for Xyz12<BE>
where
  S: Xyz12Sink<BE>,
{
  type Frame<'a> = Xyz12Frame<'a, BE>;
  type Options = Xyz12Options;

  #[inline(always)]
  fn walk(src: &Self::Frame<'_>, opts: &Self::Options, sink: &mut S) -> Result<(), S::Error>
  where
    S: PixelSink,
  {
    xyz12_to::<BE, _>(src, opts.target_gamut(), sink)
  }
}
