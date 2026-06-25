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

use crate::{ColorMatrix, DcpTargetGamut, PixelSink};
#[cfg(feature = "xyz")]
use crate::{
  frame::Xyz12Frame,
  source::{Xyz12, Xyz12Sink, xyz12_to},
};
#[cfg(feature = "bayer")]
use crate::{
  frame::{BayerFrame, BayerFrame16, BayerSink, BayerSink16, bayer_to, bayer16_to},
  source::{Bayer, Bayer16},
};
#[cfg(feature = "mono")]
use crate::{
  frame::{MonoblackFrame, MonowhiteFrame, Pal8Frame},
  source::{
    Monoblack, MonoblackSink, Monowhite, MonowhiteSink, Pal8, Pal8Sink, monoblack_to, monowhite_to,
    pal8_to,
  },
};

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

/// Generates a [`Walker`] impl for one source marker, forwarding
/// [`walk`](Walker::walk) to that format's free `{fmt}_to` walker fn.
///
/// `$marker` is the foreign `crate::source::*` ZST, `$sink` the marker's
/// [`PixelSink`] subtrait (the single per-impl bound the `{fmt}_to` fn
/// requires — the trait's method-scoped `where S: PixelSink` is implied
/// by it), `$frame` the per-format frame borrow's base type (the macro
/// appends the GAT lifetime), `$opts` the [`Options`](Walker::Options)
/// value type, and the closure-shaped tail names the `src` / `opts` /
/// `sink` bindings the `$body` expression delegates with.
///
/// The second arm carries a leading `@const $c: $cty;` for the
/// const-generic source families (the XYZ12 `BE` byte-order bool, the
/// Bayer16 `BITS` depth): it threads the const through the impl header,
/// the marker, the sink bound, and the frame's generic list. (The `@`
/// sentinel avoids the `<const …>` matcher mis-parse —
/// rust-lang/rust#143874.)
// Gated to the union of the source families it generates impls for —
// otherwise a build with none of them active (e.g. `--features yuva`)
// sees the macro as dead and `-D unused-macros` rejects it.
#[cfg(any(feature = "xyz", feature = "bayer", feature = "mono"))]
macro_rules! walker {
  ($marker:ty, $sink:path, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    impl<S> Walker<S> for $marker
    where
      S: $sink,
    {
      type Frame<'a> = $frame<'a>;
      type Options = $opts;

      #[inline(always)]
      fn walk($s: &Self::Frame<'_>, $o: &Self::Options, $k: &mut S) -> Result<(), S::Error>
      where
        S: PixelSink,
      {
        $body
      }
    }
  };
  (@const $c:ident: $cty:ty; $marker:ty, $sink:ident, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    impl<const $c: $cty, S> Walker<S> for $marker
    where
      S: $sink<$c>,
    {
      type Frame<'a> = $frame<'a, $c>;
      type Options = $opts;

      #[inline(always)]
      fn walk($s: &Self::Frame<'_>, $o: &Self::Options, $k: &mut S) -> Result<(), S::Error>
      where
        S: PixelSink,
      {
        $body
      }
    }
  };
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

// Every impl below pairs colconv's local [`Walker`] trait with a
// foreign `crate::source::*` marker, so each satisfies the orphan rule
// (local trait, foreign type). The single per-impl `where S: …Sink`
// bound is the one its `{fmt}_to` fn requires; the trait's
// method-scoped `where S: PixelSink` is implied by it (every `…Sink`
// supertraits `PixelSink`).

// XYZ12 — the target RGB gamut its inverse-OETF + 3×3 matrix decodes
// into rides on the [`Xyz12Options`]; `BE` is the wire byte order.
#[cfg(feature = "xyz")]
#[cfg_attr(docsrs, doc(cfg(feature = "xyz")))]
walker!(@const BE: bool; Xyz12<BE>, Xyz12Sink, Xyz12Frame, Xyz12Options,
  |src, opts, sink| xyz12_to::<BE, _>(src, opts.target_gamut(), sink));

// Bayer (8-bit) — the mosaic pattern, demosaic, white balance, and
// colour-correction matrix all ride on the [`BayerOptions`].
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
walker!(
  Bayer,
  BayerSink,
  BayerFrame,
  BayerOptions,
  |src, opts, sink| bayer_to(
    src,
    opts.pattern(),
    opts.demosaic(),
    opts.wb(),
    opts.ccm(),
    sink
  )
);

// Bayer16 (10/12/14/16-bit) — same parameter bundle as 8-bit Bayer, so
// it reuses [`BayerOptions`]; `BITS` is the active sample depth.
#[cfg(feature = "bayer")]
#[cfg_attr(docsrs, doc(cfg(feature = "bayer")))]
walker!(@const BITS: u32; Bayer16<BITS>, BayerSink16, BayerFrame16, BayerOptions,
  |src, opts, sink| bayer16_to::<BITS, _>(src, opts.pattern(), opts.demosaic(), opts.wb(), opts.ccm(), sink));

// Pal8 — the BGRA palette is frame-intrinsic (carried by the
// [`Pal8Frame`], not the caller), so there are no conversion knobs and
// [`Options`](Walker::Options) is the unit type.
#[cfg(feature = "mono")]
#[cfg_attr(docsrs, doc(cfg(feature = "mono")))]
walker!(Pal8, Pal8Sink, Pal8Frame, (), |src, _opts, sink| pal8_to(
  src, sink
));

// Monoblack — 1-bit-per-pixel, bit 0 → black. Its `full_range` /
// `matrix` knobs match the YUV shape, so it reuses [`YuvOptions`]
// (`YuvOptions::matrix()` is the same `mediaframe::color::Matrix` the
// `monoblack_to` walker takes).
#[cfg(feature = "mono")]
#[cfg_attr(docsrs, doc(cfg(feature = "mono")))]
walker!(
  Monoblack,
  MonoblackSink,
  MonoblackFrame,
  YuvOptions,
  |src, opts, sink| monoblack_to(src, opts.full_range(), opts.matrix(), sink)
);

// Monowhite — inverted-polarity sibling of Monoblack (bit 0 → white);
// same `full_range` / `matrix` knobs, so it reuses [`YuvOptions`] too.
#[cfg(feature = "mono")]
#[cfg_attr(docsrs, doc(cfg(feature = "mono")))]
walker!(
  Monowhite,
  MonowhiteSink,
  MonowhiteFrame,
  YuvOptions,
  |src, opts, sink| monowhite_to(src, opts.full_range(), opts.matrix(), sink)
);
