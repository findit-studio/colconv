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
// Planar YUV — 8-bit (`Yuv*pFrame`) on the plain arm, plus the high-bit
// families on the `@const_bits` arm. The 8-bit walkers are the uniform
// `(src, full_range, matrix, sink)` fns. The high-bit families are
// endian-generic: their marker is `Yuv*pN<const BE>`, the underlying
// frame struct `Yuv*pFrame16<'a, BITS, BE>` carries the depth as a
// leading const, and the const-generic `{fmt}_to_endian::<S, BE>` walker
// covers both LE and BE (the LE `{fmt}_to` is its `BE = false` wrapper).
#[cfg(feature = "yuv-planar")]
use crate::{
  frame::{
    Yuv410pFrame, Yuv411pFrame, Yuv420pFrame, Yuv420pFrame16, Yuv422pFrame, Yuv422pFrame16,
    Yuv440pFrame, Yuv440pFrame16, Yuv444pFrame, Yuv444pFrame16,
  },
  source::{
    Yuv410p, Yuv410pSink, Yuv411p, Yuv411pSink, Yuv420p, Yuv420p9, Yuv420p9Sink, Yuv420p10,
    Yuv420p10Sink, Yuv420p12, Yuv420p12Sink, Yuv420p14, Yuv420p14Sink, Yuv420p16, Yuv420p16Sink,
    Yuv420pSink, Yuv422p, Yuv422p9, Yuv422p9Sink, Yuv422p10, Yuv422p10Sink, Yuv422p12,
    Yuv422p12Sink, Yuv422p14, Yuv422p14Sink, Yuv422p16, Yuv422p16Sink, Yuv422pSink, Yuv440p,
    Yuv440p10, Yuv440p10Sink, Yuv440p12, Yuv440p12Sink, Yuv440pSink, Yuv444p, Yuv444p9,
    Yuv444p9Sink, Yuv444p10, Yuv444p10Sink, Yuv444p12, Yuv444p12Sink, Yuv444p14, Yuv444p14Sink,
    Yuv444p16, Yuv444p16Sink, Yuv444pSink, yuv410p_to, yuv411p_to, yuv420p_to, yuv420p9_to_endian,
    yuv420p10_to_endian, yuv420p12_to_endian, yuv420p14_to_endian, yuv420p16_to_endian, yuv422p_to,
    yuv422p9_to_endian, yuv422p10_to_endian, yuv422p12_to_endian, yuv422p14_to_endian,
    yuv422p16_to_endian, yuv440p_to, yuv440p10_to_endian, yuv440p12_to_endian, yuv444p_to,
    yuv444p9_to_endian, yuv444p10_to_endian, yuv444p12_to_endian, yuv444p14_to_endian,
    yuv444p16_to_endian,
  },
};
// Semi-planar YUV — Nv* (8-bit) on the plain arm + P0xx/P2xx/P4xx
// (high-bit) on the `@const_bits` arm. The high-bit markers are
// endian-generic (`P010<const BE>` …) over the shared `PnFrame` /
// `PnFrame422` / `PnFrame444` structs (`<'a, BITS, BE>`), and their
// const-generic `{fmt}_to_endian::<S, BE>` covers LE + BE.
#[cfg(feature = "yuv-semi-planar")]
use crate::{
  frame::{Nv12Frame, Nv16Frame, Nv21Frame, Nv24Frame, Nv42Frame, PnFrame, PnFrame422, PnFrame444},
  source::{
    Nv12, Nv12Sink, Nv16, Nv16Sink, Nv21, Nv21Sink, Nv24, Nv24Sink, Nv42, Nv42Sink, P010, P010Sink,
    P012, P012Sink, P016, P016Sink, P210, P210Sink, P212, P212Sink, P216, P216Sink, P410, P410Sink,
    P412, P412Sink, P416, P416Sink, nv12_to, nv16_to, nv21_to, nv24_to, nv42_to, p010_to_endian,
    p012_to_endian, p016_to_endian, p210_to_endian, p212_to_endian, p216_to_endian, p410_to_endian,
    p412_to_endian, p416_to_endian,
  },
};
// Packed YUV 4:2:2 / 4:1:1 — single-buffer `(src, full_range, matrix,
// sink)` walkers.
#[cfg(feature = "yuv-packed")]
use crate::{
  frame::{Uyvy422Frame, Uyyvyy411Frame, Yuyv422Frame, Yvyu422Frame},
  source::{
    Uyvy422, Uyvy422Sink, Uyyvyy411, Uyyvyy411Sink, Yuyv422, Yuyv422Sink, Yvyu422, Yvyu422Sink,
    uyvy422_to, uyyvyy411_to, yuyv422_to, yvyu422_to,
  },
};
// Packed YUV 4:2:2 high-bit (Y2xx) — endian-generic markers
// (`Y210<const BE>` …) over the shared `Y2xxFrame<'a, BITS, BE>` struct;
// the const-generic `{fmt}_to_endian::<S, BE>` covers LE + BE.
#[cfg(feature = "y2xx")]
use crate::{
  frame::Y2xxFrame,
  source::{
    Y210, Y210Sink, Y212, Y212Sink, Y216, Y216Sink, y210_to_endian, y212_to_endian, y216_to_endian,
  },
};
// Planar YUVA — uniform `(full_range, matrix)` sources; the alpha plane
// is read inside the walker from the frame (never an `Options` knob), so
// they reuse `YuvOptions`. 8-bit `Yuva*pFrame` on the plain arm; the
// high-bit families on the `@const_bits` arm — endian-generic markers
// (`Yuva420p10<const BE>` …) over the shared `Yuva*pFrame16<'a, BITS, BE>`
// structs, with const-generic `{fmt}_to_endian::<S, BE>` covering LE + BE.
#[cfg(feature = "yuva")]
use crate::{
  frame::{
    Yuva420pFrame, Yuva420pFrame16, Yuva422pFrame, Yuva422pFrame16, Yuva444pFrame, Yuva444pFrame16,
  },
  source::{
    Yuva420p, Yuva420p9, Yuva420p9Sink, Yuva420p10, Yuva420p10Sink, Yuva420p16, Yuva420p16Sink,
    Yuva420pSink, Yuva422p, Yuva422p9, Yuva422p9Sink, Yuva422p10, Yuva422p10Sink, Yuva422p12,
    Yuva422p12Sink, Yuva422p16, Yuva422p16Sink, Yuva422pSink, Yuva444p, Yuva444p9, Yuva444p9Sink,
    Yuva444p10, Yuva444p10Sink, Yuva444p12, Yuva444p12Sink, Yuva444p14, Yuva444p14Sink, Yuva444p16,
    Yuva444p16Sink, Yuva444pSink, yuva420p_to, yuva420p9_to_endian, yuva420p10_to_endian,
    yuva420p16_to_endian, yuva422p_to, yuva422p9_to_endian, yuva422p10_to_endian,
    yuva422p12_to_endian, yuva422p16_to_endian, yuva444p_to, yuva444p9_to_endian,
    yuva444p10_to_endian, yuva444p12_to_endian, yuva444p14_to_endian, yuva444p16_to_endian,
  },
};
// Packed RGB — already-RGB sources (no chroma matrix). The 8-bit packed
// families (`Rgb24`/`Bgr24`/`Rgba`/…/`Bgrx`) ride the plain arm; the
// 16-bit families (`Rgb48`/`Bgr48`/`Rgba64`/`Bgra64`) are endian-generic
// — marker `Rgb48<const BE>` over the trailing-`BE` frame
// `Rgb48Frame<'a, BE>` (no leading bit-depth const), so they ride the
// `@const BE` arm and delegate to the const-generic
// `{fmt}_to_endian::<_, BE>` (the LE `{fmt}_to` is its `BE = false`
// wrapper). The free `{fmt}_to` / `{fmt}_to_endian` walkers still take
// `(full_range, matrix)` — the RGB-input row carries them for the
// `with_luma` / `with_hsv` outputs — so every RGB family reuses
// [`YuvOptions`]; the RGB-only outputs (`with_rgb`/`with_rgba`/`…`)
// ignore them.
#[cfg(feature = "rgb")]
use crate::{
  frame::{
    AbgrFrame, ArgbFrame, Bgr24Frame, Bgr48Frame, Bgra64Frame, BgraFrame, BgrxFrame, Rgb24Frame,
    Rgb48Frame, Rgba64Frame, RgbaFrame, RgbxFrame, XbgrFrame, XrgbFrame,
  },
  source::{
    Abgr, AbgrSink, Argb, ArgbSink, Bgr24, Bgr24Sink, Bgr48, Bgr48Sink, Bgra, Bgra64, Bgra64Sink,
    BgraSink, Bgrx, BgrxSink, Rgb24, Rgb24Sink, Rgb48, Rgb48Sink, Rgba, Rgba64, Rgba64Sink,
    RgbaSink, Rgbx, RgbxSink, Xbgr, XbgrSink, Xrgb, XrgbSink, abgr_to, argb_to, bgr24_to,
    bgr48_to_endian, bgra_to, bgra64_to_endian, bgrx_to, rgb24_to, rgb48_to_endian, rgba_to,
    rgba64_to_endian, rgbx_to, xbgr_to, xrgb_to,
  },
};
// Legacy packed RGB (5/5/6/5/5/5/4/4/4-bit, `AV_PIX_FMT_*565/555/444LE`).
// Byte-order-fixed LE (no `_to_endian` walker), so they ride the plain
// arm exactly like the 8-bit packed families and reuse [`YuvOptions`].
#[cfg(feature = "rgb-legacy")]
use crate::{
  frame::{Bgr444Frame, Bgr555Frame, Bgr565Frame, Rgb444Frame, Rgb555Frame, Rgb565Frame},
  source::{
    Bgr444, Bgr444Sink, Bgr555, Bgr555Sink, Bgr565, Bgr565Sink, Rgb444, Rgb444Sink, Rgb555,
    Rgb555Sink, Rgb565, Rgb565Sink, bgr444_to, bgr555_to, bgr565_to, rgb444_to, rgb555_to,
    rgb565_to,
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
/// The `@const $c: $cty;` arm handles the source families whose marker
/// carries a *single* const parameter that is also the last item in the
/// frame's generic list (the XYZ12 `BE` byte-order bool over
/// `Xyz12Frame<'a, BE>`, the Bayer16 `BITS` depth over
/// `BayerFrame16<'a, BITS>`): it threads the const through the impl
/// header, the marker, the sink bound, and the frame's generic list.
///
/// The `@const_bits $bits, BE;` arm handles the high-bit YUV / YUVA /
/// Y2xx families. Their marker is endian-generic (`Yuv420p10<const BE>`,
/// the `marker!` macro's endian-aware arm) but the *bit depth is baked
/// into the marker name* and lives as a separate leading const on the
/// underlying frame struct (`Yuv420pFrame16<'a, BITS, BE>`,
/// `PnFrame<'a, BITS, BE>`, `Y2xxFrame<'a, BITS, BE>`, …). So only `BE`
/// is generic in the impl, while `$bits` is a literal spliced between the
/// frame lifetime and `BE`. The walk delegates to the const-generic
/// `{fmt}_to_endian::<S, BE>` (the public `{fmt}_to` is just its
/// `BE = false` wrapper), so a single impl covers **both** the LE
/// (`BE = false`) and BE (`BE = true`) high-bit sources.
///
/// (The `@` sentinel avoids the `<const …>` matcher mis-parse —
/// rust-lang/rust#143874.)
// Gated to the union of the source families it generates impls for —
// otherwise a build with none of them active sees the macro as dead and
// `-D unused-macros` rejects it.
#[cfg(any(
  feature = "xyz",
  feature = "bayer",
  feature = "mono",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "y2xx",
  feature = "yuva",
  feature = "rgb",
  feature = "rgb-legacy",
))]
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
  // High-bit YUV / YUVA / Y2xx: BE-generic marker, BITS literal baked
  // into the marker name and spliced as the *leading* const on the
  // underlying frame struct (`$frame<'a, $bits, BE>`). `$marker` /
  // `$sink` here are the bare identifiers (the macro appends `<BE>`); the
  // `$body` delegates to the const-generic `{fmt}_to_endian`, so the one
  // impl serves LE (`BE = false`) and BE (`BE = true`).
  (@const_bits $bits:literal, $be:ident; $marker:ident, $sink:ident, $frame:ident, $opts:ty, |$s:ident, $o:ident, $k:ident| $body:expr) => {
    impl<const $be: bool, S> Walker<S> for $marker<$be>
    where
      S: $sink<$be>,
    {
      type Frame<'a> = $frame<'a, $bits, $be>;
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

// ===== Uniform YUV families =============================================
//
// Every source below is a *uniform* YUV format: its only conversion
// knobs are the quantisation range (`full_range`) and the YCbCr
// [`ColorMatrix`], so they all reuse [`YuvOptions`].
//
// The 8-bit families (planar `Yuv*p`, semi-planar `Nv*`, packed
// `Yuyv422`/`Uyvy422`/…, 8-bit `Yuva*p`) have no byte-order axis: they
// ride the **plain** arm and forward to their uniform
// `{fmt}_to(src, full_range, matrix, sink)` walker.
//
// The high-bit families (9/10/12/14/16-bit planar, P0xx/P2xx/P4xx
// semi-planar, Y2xx, and the high-bit YUVA families) are const-generic
// in the wire byte order (`<const BE: bool>`), and mediaframe exposes a
// matching const-generic walker for each — `{fmt}_to_endian::<S, BE>`
// (macro-generated by the `walker!`/`marker!` `*_be` arms, alongside the
// LE-only `{fmt}_to` wrapper which is just its `BE = false` shim). So
// every high-bit family rides the **`@const_bits`** arm: the marker is
// `Fmt<const BE>`, the [`Frame`](Walker::Frame) GAT is the underlying
// `*Frame16` / `PnFrame*` / `Y2xxFrame` struct with the depth literal
// spliced before `BE` (`Yuv420pFrame16<'a, 10, BE>`, `PnFrame<'a, 10,
// BE>`, `Y2xxFrame<'a, 10, BE>`, …), and [`Walker::walk`] delegates to
// `{fmt}_to_endian::<_, BE>`. A single impl per family therefore covers
// **both** the LE source (`BE = false`, the impl at the marker's default)
// and the BE source (`BE = true`) — byte-identical to calling
// `{fmt}_to_endian` directly. The module stays additive: the existing
// `{fmt}_to` / `{fmt}_to_endian` walkers, sinks, and kernels are
// untouched.
//
// The YUVA families are uniform too: the alpha plane is read *inside*
// the walker straight from the frame, never threaded as an `Options`
// knob, so they share [`YuvOptions`] with the alpha-less YUV families.

// ---- Planar YUV, 8-bit -------------------------------------------------
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(
  Yuv420p,
  Yuv420pSink,
  Yuv420pFrame,
  YuvOptions,
  |src, opts, sink| yuv420p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(
  Yuv422p,
  Yuv422pSink,
  Yuv422pFrame,
  YuvOptions,
  |src, opts, sink| yuv422p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(
  Yuv444p,
  Yuv444pSink,
  Yuv444pFrame,
  YuvOptions,
  |src, opts, sink| yuv444p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(
  Yuv440p,
  Yuv440pSink,
  Yuv440pFrame,
  YuvOptions,
  |src, opts, sink| yuv440p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(
  Yuv410p,
  Yuv410pSink,
  Yuv410pFrame,
  YuvOptions,
  |src, opts, sink| yuv410p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(
  Yuv411p,
  Yuv411pSink,
  Yuv411pFrame,
  YuvOptions,
  |src, opts, sink| yuv411p_to(src, opts.full_range(), opts.matrix(), sink)
);

// ---- Planar YUV, high-bit (BE-generic marker; LE + BE via `_to_endian`) -
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 9, BE; Yuv420p9, Yuv420p9Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 10, BE; Yuv420p10, Yuv420p10Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 12, BE; Yuv420p12, Yuv420p12Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 14, BE; Yuv420p14, Yuv420p14Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 16, BE; Yuv420p16, Yuv420p16Sink, Yuv420pFrame16, YuvOptions,
  |src, opts, sink| yuv420p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 9, BE; Yuv422p9, Yuv422p9Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 10, BE; Yuv422p10, Yuv422p10Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 12, BE; Yuv422p12, Yuv422p12Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 14, BE; Yuv422p14, Yuv422p14Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 16, BE; Yuv422p16, Yuv422p16Sink, Yuv422pFrame16, YuvOptions,
  |src, opts, sink| yuv422p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 9, BE; Yuv444p9, Yuv444p9Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 10, BE; Yuv444p10, Yuv444p10Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 12, BE; Yuv444p12, Yuv444p12Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 14, BE; Yuv444p14, Yuv444p14Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 16, BE; Yuv444p16, Yuv444p16Sink, Yuv444pFrame16, YuvOptions,
  |src, opts, sink| yuv444p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 10, BE; Yuv440p10, Yuv440p10Sink, Yuv440pFrame16, YuvOptions,
  |src, opts, sink| yuv440p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-planar")))]
walker!(@const_bits 12, BE; Yuv440p12, Yuv440p12Sink, Yuv440pFrame16, YuvOptions,
  |src, opts, sink| yuv440p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Semi-planar YUV, 8-bit (Nv*) --------------------------------------
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(Nv12, Nv12Sink, Nv12Frame, YuvOptions, |src, opts, sink| {
  nv12_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(Nv16, Nv16Sink, Nv16Frame, YuvOptions, |src, opts, sink| {
  nv16_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(Nv21, Nv21Sink, Nv21Frame, YuvOptions, |src, opts, sink| {
  nv21_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(Nv24, Nv24Sink, Nv24Frame, YuvOptions, |src, opts, sink| {
  nv24_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(Nv42, Nv42Sink, Nv42Frame, YuvOptions, |src, opts, sink| {
  nv42_to(src, opts.full_range(), opts.matrix(), sink)
});

// ---- Semi-planar YUV, high-bit (P0xx/P2xx/P4xx; LE + BE via `_to_endian`)
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@const_bits 10, BE; P010, P010Sink, PnFrame, YuvOptions,
  |src, opts, sink| p010_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@const_bits 12, BE; P012, P012Sink, PnFrame, YuvOptions,
  |src, opts, sink| p012_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@const_bits 16, BE; P016, P016Sink, PnFrame, YuvOptions,
  |src, opts, sink| p016_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@const_bits 10, BE; P210, P210Sink, PnFrame422, YuvOptions,
  |src, opts, sink| p210_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@const_bits 12, BE; P212, P212Sink, PnFrame422, YuvOptions,
  |src, opts, sink| p212_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@const_bits 16, BE; P216, P216Sink, PnFrame422, YuvOptions,
  |src, opts, sink| p216_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@const_bits 10, BE; P410, P410Sink, PnFrame444, YuvOptions,
  |src, opts, sink| p410_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@const_bits 12, BE; P412, P412Sink, PnFrame444, YuvOptions,
  |src, opts, sink| p412_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuv-semi-planar")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-semi-planar")))]
walker!(@const_bits 16, BE; P416, P416Sink, PnFrame444, YuvOptions,
  |src, opts, sink| p416_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Packed YUV 4:2:2 / 4:1:1 ------------------------------------------
#[cfg(feature = "yuv-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-packed")))]
walker!(
  Yuyv422,
  Yuyv422Sink,
  Yuyv422Frame,
  YuvOptions,
  |src, opts, sink| yuyv422_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-packed")))]
walker!(
  Uyvy422,
  Uyvy422Sink,
  Uyvy422Frame,
  YuvOptions,
  |src, opts, sink| uyvy422_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-packed")))]
walker!(
  Yvyu422,
  Yvyu422Sink,
  Yvyu422Frame,
  YuvOptions,
  |src, opts, sink| yvyu422_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuv-packed")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuv-packed")))]
walker!(
  Uyyvyy411,
  Uyyvyy411Sink,
  Uyyvyy411Frame,
  YuvOptions,
  |src, opts, sink| uyyvyy411_to(src, opts.full_range(), opts.matrix(), sink)
);

// ---- Packed YUV 4:2:2 high-bit (Y2xx; LE + BE via `_to_endian`) --------
#[cfg(feature = "y2xx")]
#[cfg_attr(docsrs, doc(cfg(feature = "y2xx")))]
walker!(@const_bits 10, BE; Y210, Y210Sink, Y2xxFrame, YuvOptions,
  |src, opts, sink| y210_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "y2xx")]
#[cfg_attr(docsrs, doc(cfg(feature = "y2xx")))]
walker!(@const_bits 12, BE; Y212, Y212Sink, Y2xxFrame, YuvOptions,
  |src, opts, sink| y212_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "y2xx")]
#[cfg_attr(docsrs, doc(cfg(feature = "y2xx")))]
walker!(@const_bits 16, BE; Y216, Y216Sink, Y2xxFrame, YuvOptions,
  |src, opts, sink| y216_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Planar YUVA, 8-bit (alpha read inside `{fmt}_to`, not an Option) --
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(
  Yuva420p,
  Yuva420pSink,
  Yuva420pFrame,
  YuvOptions,
  |src, opts, sink| yuva420p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(
  Yuva422p,
  Yuva422pSink,
  Yuva422pFrame,
  YuvOptions,
  |src, opts, sink| yuva422p_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(
  Yuva444p,
  Yuva444pSink,
  Yuva444pFrame,
  YuvOptions,
  |src, opts, sink| yuva444p_to(src, opts.full_range(), opts.matrix(), sink)
);

// ---- Planar YUVA, high-bit (BE-generic marker; LE + BE via `_to_endian`)
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 9, BE; Yuva420p9, Yuva420p9Sink, Yuva420pFrame16, YuvOptions,
  |src, opts, sink| yuva420p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 10, BE; Yuva420p10, Yuva420p10Sink, Yuva420pFrame16, YuvOptions,
  |src, opts, sink| yuva420p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 16, BE; Yuva420p16, Yuva420p16Sink, Yuva420pFrame16, YuvOptions,
  |src, opts, sink| yuva420p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 9, BE; Yuva422p9, Yuva422p9Sink, Yuva422pFrame16, YuvOptions,
  |src, opts, sink| yuva422p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 10, BE; Yuva422p10, Yuva422p10Sink, Yuva422pFrame16, YuvOptions,
  |src, opts, sink| yuva422p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 12, BE; Yuva422p12, Yuva422p12Sink, Yuva422pFrame16, YuvOptions,
  |src, opts, sink| yuva422p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 16, BE; Yuva422p16, Yuva422p16Sink, Yuva422pFrame16, YuvOptions,
  |src, opts, sink| yuva422p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 9, BE; Yuva444p9, Yuva444p9Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p9_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 10, BE; Yuva444p10, Yuva444p10Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p10_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 12, BE; Yuva444p12, Yuva444p12Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p12_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 14, BE; Yuva444p14, Yuva444p14Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p14_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "yuva")]
#[cfg_attr(docsrs, doc(cfg(feature = "yuva")))]
walker!(@const_bits 16, BE; Yuva444p16, Yuva444p16Sink, Yuva444pFrame16, YuvOptions,
  |src, opts, sink| yuva444p16_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ===== Packed RGB families =============================================
//
// These sources are *already RGB* — there is no chroma matrix. The
// underlying `{fmt}_to` / `{fmt}_to_endian` walkers nonetheless take
// `(full_range, matrix)` because the RGB-input row carries them through
// to the `with_luma` / `with_hsv` outputs (the `with_rgb` / `with_rgba`
// / `with_rgb_u16` outputs ignore them). So every RGB family reuses
// [`YuvOptions`] and forwards `opts.full_range()` / `opts.matrix()`,
// byte-identical to a direct walker call. The module stays additive: the
// existing walkers, sinks, and kernels are untouched.

// ---- Packed RGB, 8-bit (plain arm; no byte-order axis) -----------------
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(
  Rgb24,
  Rgb24Sink,
  Rgb24Frame,
  YuvOptions,
  |src, opts, sink| rgb24_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(
  Bgr24,
  Bgr24Sink,
  Bgr24Frame,
  YuvOptions,
  |src, opts, sink| bgr24_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(Rgba, RgbaSink, RgbaFrame, YuvOptions, |src, opts, sink| {
  rgba_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(Bgra, BgraSink, BgraFrame, YuvOptions, |src, opts, sink| {
  bgra_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(Argb, ArgbSink, ArgbFrame, YuvOptions, |src, opts, sink| {
  argb_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(Abgr, AbgrSink, AbgrFrame, YuvOptions, |src, opts, sink| {
  abgr_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(Xrgb, XrgbSink, XrgbFrame, YuvOptions, |src, opts, sink| {
  xrgb_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(Rgbx, RgbxSink, RgbxFrame, YuvOptions, |src, opts, sink| {
  rgbx_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(Xbgr, XbgrSink, XbgrFrame, YuvOptions, |src, opts, sink| {
  xbgr_to(src, opts.full_range(), opts.matrix(), sink)
});
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(Bgrx, BgrxSink, BgrxFrame, YuvOptions, |src, opts, sink| {
  bgrx_to(src, opts.full_range(), opts.matrix(), sink)
});

// ---- Packed RGB, 16-bit (BE-generic marker; LE + BE via `_to_endian`) --
// Marker `Fmt<const BE>` over the trailing-`BE` frame `FmtFrame<'a, BE>`
// (no leading bit-depth const), so these ride the `@const BE` arm (same
// shape as XYZ12) and delegate to `{fmt}_to_endian::<_, BE>`; one impl
// covers both LE (`BE = false`) and BE (`BE = true`).
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@const BE: bool; Rgb48<BE>, Rgb48Sink, Rgb48Frame, YuvOptions,
  |src, opts, sink| rgb48_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@const BE: bool; Bgr48<BE>, Bgr48Sink, Bgr48Frame, YuvOptions,
  |src, opts, sink| bgr48_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@const BE: bool; Rgba64<BE>, Rgba64Sink, Rgba64Frame, YuvOptions,
  |src, opts, sink| rgba64_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));
#[cfg(feature = "rgb")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb")))]
walker!(@const BE: bool; Bgra64<BE>, Bgra64Sink, Bgra64Frame, YuvOptions,
  |src, opts, sink| bgra64_to_endian::<_, BE>(src, opts.full_range(), opts.matrix(), sink));

// ---- Legacy packed RGB (byte-order-fixed LE; plain arm) ----------------
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(
  Rgb565,
  Rgb565Sink,
  Rgb565Frame,
  YuvOptions,
  |src, opts, sink| rgb565_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(
  Bgr565,
  Bgr565Sink,
  Bgr565Frame,
  YuvOptions,
  |src, opts, sink| bgr565_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(
  Rgb555,
  Rgb555Sink,
  Rgb555Frame,
  YuvOptions,
  |src, opts, sink| rgb555_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(
  Bgr555,
  Bgr555Sink,
  Bgr555Frame,
  YuvOptions,
  |src, opts, sink| bgr555_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(
  Rgb444,
  Rgb444Sink,
  Rgb444Frame,
  YuvOptions,
  |src, opts, sink| rgb444_to(src, opts.full_range(), opts.matrix(), sink)
);
#[cfg(feature = "rgb-legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "rgb-legacy")))]
walker!(
  Bgr444,
  Bgr444Sink,
  Bgr444Frame,
  YuvOptions,
  |src, opts, sink| bgr444_to(src, opts.full_range(), opts.matrix(), sink)
);
