//! [`MixedSinker`] — the common "I want some subset of {RGB, Luma, HSV}
//! written into my own buffers" consumer.
//!
//! Generic over the source format via an `F: SourceFormat` type
//! parameter. One `PixelSink` impl per supported format. Currently
//! ships impls for:
//!
//! - **8‑bit planar**: [`Yuv411p`](crate::source::Yuv411p),
//!   [`Yuv420p`](crate::source::Yuv420p),
//!   [`Yuv422p`](crate::source::Yuv422p),
//!   [`Yuv440p`](crate::source::Yuv440p),
//!   [`Yuv444p`](crate::source::Yuv444p).
//! - **8‑bit semi‑planar**: [`Nv12`](crate::source::Nv12),
//!   [`Nv21`](crate::source::Nv21), [`Nv16`](crate::source::Nv16),
//!   [`Nv24`](crate::source::Nv24), [`Nv42`](crate::source::Nv42).
//! - **9/10/12/14/16‑bit planar 4:2:0**:
//!   [`Yuv420p9`](crate::source::Yuv420p9),
//!   [`Yuv420p10`](crate::source::Yuv420p10),
//!   [`Yuv420p12`](crate::source::Yuv420p12),
//!   [`Yuv420p14`](crate::source::Yuv420p14),
//!   [`Yuv420p16`](crate::source::Yuv420p16).
//! - **9/10/12/14/16‑bit planar 4:2:2**:
//!   [`Yuv422p9`](crate::source::Yuv422p9),
//!   [`Yuv422p10`](crate::source::Yuv422p10),
//!   [`Yuv422p12`](crate::source::Yuv422p12),
//!   [`Yuv422p14`](crate::source::Yuv422p14),
//!   [`Yuv422p16`](crate::source::Yuv422p16).
//! - **10/12‑bit planar 4:4:0**:
//!   [`Yuv440p10`](crate::source::Yuv440p10),
//!   [`Yuv440p12`](crate::source::Yuv440p12).
//! - **9/10/12/14/16‑bit planar 4:4:4**:
//!   [`Yuv444p9`](crate::source::Yuv444p9),
//!   [`Yuv444p10`](crate::source::Yuv444p10),
//!   [`Yuv444p12`](crate::source::Yuv444p12),
//!   [`Yuv444p14`](crate::source::Yuv444p14),
//!   [`Yuv444p16`](crate::source::Yuv444p16).
//! - **10/12/16‑bit semi‑planar high‑bit‑packed 4:2:0**:
//!   [`P010`](crate::source::P010), [`P012`](crate::source::P012),
//!   [`P016`](crate::source::P016).
//! - **10/12/16‑bit semi‑planar high‑bit‑packed 4:2:2**:
//!   [`P210`](crate::source::P210), [`P212`](crate::source::P212),
//!   [`P216`](crate::source::P216).
//! - **10/12/16‑bit semi‑planar high‑bit‑packed 4:4:4**:
//!   [`P410`](crate::source::P410), [`P412`](crate::source::P412),
//!   [`P416`](crate::source::P416).
//! - **YUVA (alpha-bearing planar)**: the entire FFmpeg-shipped
//!   YUVA family — `Yuva420p` / `Yuva420p9/10/16`, `Yuva422p` /
//!   `Yuva422p9/10/12/16`, `Yuva444p` / `Yuva444p9/10/12/14/16`.
//!   Source-side alpha pass-through to `with_rgba` /
//!   `with_rgba_u16`, with native SIMD on every backend.
//! - **8‑bit packed RGB sources** (Tier 6):
//!   [`Rgb24`](crate::source::Rgb24) (`R, G, B` bytes),
//!   [`Bgr24`](crate::source::Bgr24) (`B, G, R` bytes),
//!   [`Rgba`](crate::source::Rgba) (`R, G, B, A` bytes),
//!   [`Bgra`](crate::source::Bgra) (`B, G, R, A` bytes),
//!   [`Argb`](crate::source::Argb) (`A, R, G, B` bytes — leading alpha),
//!   [`Abgr`](crate::source::Abgr) (`A, B, G, R` bytes — leading alpha),
//!   [`Xrgb`](crate::source::Xrgb) / [`Rgbx`](crate::source::Rgbx) /
//!   [`Xbgr`](crate::source::Xbgr) / [`Bgrx`](crate::source::Bgrx)
//!   (4-byte packed RGB with one ignored padding byte at the leading
//!   or trailing position).
//!   The source row is already 8‑bit RGB at the byte level —
//!   `with_rgb` is an identity copy / channel swap /
//!   drop-alpha-or-padding, `with_rgba` is a memcpy / channel
//!   reorder (alpha passed through for the alpha-bearing 4-byte
//!   sources, forced to `0xFF` for the 3-byte sources and the
//!   padding-byte family), `with_luma` derives Y' from R/G/B,
//!   `with_hsv` reuses the existing kernel.
//! - **8‑bit planar GBR sources** (Tier 10):
//!   [`Gbrp`](crate::source::Gbrp) (three planes: G, B, R) and
//!   [`Gbrap`](crate::source::Gbrap) (four planes: G, B, R, A — real
//!   per-pixel α). Both reuse the standard `with_rgb` / `with_rgba` /
//!   `with_luma` / `with_luma_u16` / `with_hsv` channels via dedicated
//!   `gbr_to_rgb_row` / `gbra_to_rgba_row` / `gbr_to_rgba_opaque_row`
//!   SIMD kernels (no chroma matrix — the source is already component
//!   RGB). `Gbrap`'s `with_rgb + with_rgba` combo uses Strategy A+
//!   (expand RGB → RGBA, then α-overwrite from the source plane).
//! - **10‑bit packed RGB sources** (Tier 6 — Ship 9e):
//!   [`X2Rgb10`](crate::source::X2Rgb10) and
//!   [`X2Bgr10`](crate::source::X2Bgr10). Each pixel is a 32-bit LE word
//!   with `(MSB) 2X | 10c2 | 10c1 | 10c0 (LSB)` (R/G/B for X2RGB10,
//!   B/G/R for X2BGR10). Unlike the 8‑bit byte-shuffle family above,
//!   the source is **not** byte-aligned RGB — every output path
//!   starts with bit-level extraction of the three 10‑bit channels:
//!   `with_rgb` extracts and down-shifts each channel from 10→8 bits,
//!   `with_rgba` does the same and forces alpha to `0xFF` (the 2‑bit
//!   field is padding, not real alpha), `with_rgb_u16` preserves
//!   native 10‑bit precision (low-bit aligned in `u16`, value range
//!   `[0, 1023]`), and `with_luma` / `with_hsv` reuse the staged u8
//!   RGB scratch path.
//!
//! High‑bit‑depth source impls expose both `with_rgb` (u8 output) and
//! `with_rgb_u16` (native‑depth u16 output). Calling `with_rgb_u16` on
//! an 8‑bit source format is a compile error.
//!
//! All configuration and processing methods are fallible — no panics
//! under normal contract violations — so the sink is usable on
//! `panic = "abort"` targets.

use core::marker::PhantomData;

// `Vec<u8>` is only used by the `rgb_scratch` lazy scratch buffer
// (gated on the same 15-feature any as the per-format `process`
// impls). The import is left unconditional because gating it would
// also leave `extern crate alloc as std` unused under
// `--features "alloc"` alone, which is harder to express.
#[allow(unused_imports)]
use std::vec::Vec;

use derive_more::{Display, IsVariant, TryUnwrap, Unwrap};
use thiserror::Error;

use crate::{
  SourceFormat,
  resample::{NoopResampler, ResampleError, ResamplePlan, Resampler},
};
// PixelSink is referenced only via intra-doc links (`[`PixelSink::*`]`)
// in this file; the rustc lint can't see those uses, so silence it.
#[allow(unused_imports)]
use crate::PixelSink;

pub use mediaframe::{
  frame::{WidthAlignment, WidthAlignmentRequirement},
  source::{HsvFrame, HsvFrameMut, HsvPlane},
};

/// Frame dimensions handed to `begin_frame` don't match the sinker's
/// configured size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DimensionMismatch {
  /// Width declared at sinker construction.
  configured_w: usize,
  /// Height declared at sinker construction.
  configured_h: usize,
  /// Width of the frame handed to the walker.
  frame_w: u32,
  /// Height of the frame handed to the walker.
  frame_h: u32,
}

impl DimensionMismatch {
  /// Constructs a new `DimensionMismatch` payload.
  #[inline]
  pub const fn new(configured_w: usize, configured_h: usize, frame_w: u32, frame_h: u32) -> Self {
    Self {
      configured_w,
      configured_h,
      frame_w,
      frame_h,
    }
  }

  /// Width declared at sinker construction.
  #[inline]
  pub const fn configured_w(&self) -> usize {
    self.configured_w
  }

  /// Height declared at sinker construction.
  #[inline]
  pub const fn configured_h(&self) -> usize {
    self.configured_h
  }

  /// Width of the frame handed to the walker.
  #[inline]
  pub const fn frame_w(&self) -> u32 {
    self.frame_w
  }

  /// Height of the frame handed to the walker.
  #[inline]
  pub const fn frame_h(&self) -> u32 {
    self.frame_h
  }
}

/// Generic "insufficient buffer" payload, shared across every
/// `MixedSinkerError::Insufficient*Buffer` variant. `expected` / `actual`
/// are expressed in the unit reported by each variant's Display
/// impl (`bytes` for the byte buffers, `elements` for the typed
/// `u16` / `f32` / `f16` buffers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsufficientBuffer {
  /// Minimum elements (bytes or typed elements) required.
  expected: usize,
  /// Elements supplied.
  actual: usize,
}

impl InsufficientBuffer {
  /// Constructs a new `InsufficientBuffer` payload.
  #[inline]
  pub const fn new(expected: usize, actual: usize) -> Self {
    Self { expected, actual }
  }

  /// Minimum elements (bytes or typed elements) required.
  #[inline]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Elements supplied.
  #[inline]
  pub const fn actual(&self) -> usize {
    self.actual
  }
}

/// HSV plane identification and size mismatch payload for
/// [`MixedSinkerError::InsufficientHsvPlane`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InsufficientHsvPlane {
  /// Which HSV plane was short (H, S, or V).
  which: HsvPlane,
  /// Minimum bytes required (`width x height`).
  expected: usize,
  /// Bytes supplied.
  actual: usize,
}

impl InsufficientHsvPlane {
  /// Constructs a new `InsufficientHsvPlane` payload.
  #[inline]
  pub const fn new(which: HsvPlane, expected: usize, actual: usize) -> Self {
    Self {
      which,
      expected,
      actual,
    }
  }

  /// Which HSV plane was short (H, S, or V).
  #[inline]
  pub const fn which(&self) -> HsvPlane {
    self.which
  }

  /// Minimum bytes required (`width x height`).
  #[inline]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Bytes supplied.
  #[inline]
  pub const fn actual(&self) -> usize {
    self.actual
  }
}

/// Frame-size overflow payload for [`MixedSinkerError::GeometryOverflow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryOverflow {
  /// Configured width.
  width: usize,
  /// Configured height.
  height: usize,
  /// Channel count the overflowing product was computed with.
  channels: usize,
}

impl GeometryOverflow {
  /// Constructs a new `GeometryOverflow` payload.
  #[inline]
  pub const fn new(width: usize, height: usize, channels: usize) -> Self {
    Self {
      width,
      height,
      channels,
    }
  }

  /// Configured width.
  #[inline]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// Configured height.
  #[inline]
  pub const fn height(&self) -> usize {
    self.height
  }

  /// Channel count the overflowing product was computed with.
  #[inline]
  pub const fn channels(&self) -> usize {
    self.channels
  }
}

/// Row shape mismatch payload for [`MixedSinkerError::RowShapeMismatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowShapeMismatch {
  /// Which slice mismatched. See [`RowSlice`] for variants.
  which: RowSlice,
  /// Row index reported by the offending row.
  row: usize,
  /// Expected slice length in elements of the slice's element type.
  expected: usize,
  /// Actual slice length in the same unit as `expected`.
  actual: usize,
}

impl RowShapeMismatch {
  /// Constructs a new `RowShapeMismatch` payload.
  #[inline]
  pub const fn new(which: RowSlice, row: usize, expected: usize, actual: usize) -> Self {
    Self {
      which,
      row,
      expected,
      actual,
    }
  }

  /// Which slice mismatched. See [`RowSlice`] for variants.
  #[inline]
  pub const fn which(&self) -> RowSlice {
    self.which
  }

  /// Row index reported by the offending row.
  #[inline]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Expected slice length in elements of the slice's element type.
  #[inline]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Actual slice length in the same unit as `expected`.
  #[inline]
  pub const fn actual(&self) -> usize {
    self.actual
  }
}

/// Row-index-out-of-range payload for [`MixedSinkerError::RowIndexOutOfRange`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowIndexOutOfRange {
  /// Row index reported by the offending row.
  row: usize,
  /// Sink's configured height.
  configured_height: usize,
}

impl RowIndexOutOfRange {
  /// Constructs a new `RowIndexOutOfRange` payload.
  #[inline]
  pub const fn new(row: usize, configured_height: usize) -> Self {
    Self {
      row,
      configured_height,
    }
  }

  /// Row index reported by the offending row.
  #[inline]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Sink's configured height.
  #[inline]
  pub const fn configured_height(&self) -> usize {
    self.configured_height
  }
}

/// Snapshot of a resampled frame's complete output configuration:
/// presence plus attachment identity (data pointer and length) for
/// luma, luma_u16, rgb, rgba, rgb_u16, rgba_u16, the `rgb_f32`
/// float-RGB output, and the three HSV planes. Equality is the
/// per-frame immutability check — in safe code a mid-frame `set_*`
/// necessarily supplies a different borrow, so an identity change is
/// exactly a reattachment. The `*_u16` / `rgb_f32` slots are `(0, 0)`
/// for every format that attaches no such output, so adding them
/// leaves those formats' snapshots unchanged.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono"
))]
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FrozenOutputs {
  idents: [(usize, usize); 10],
}

#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono"
))]
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
impl FrozenOutputs {
  /// Identity of one attached buffer: `(data pointer, length)`, or
  /// `(0, 0)` for an absent output (a slice pointer is never null).
  fn ident<T>(buf: Option<&[T]>) -> (usize, usize) {
    buf.map_or((0, 0), |b| (b.as_ptr() as usize, b.len()))
  }

  /// Builds the snapshot from the currently attached outputs.
  #[allow(clippy::too_many_arguments)]
  pub(super) fn snapshot(
    luma: Option<&[u8]>,
    luma_u16: Option<&[u16]>,
    rgb: Option<&[u8]>,
    rgba: Option<&[u8]>,
    rgb_u16: Option<&[u16]>,
    rgba_u16: Option<&[u16]>,
    rgb_f32: Option<&[f32]>,
    hsv: Option<(&[u8], &[u8], &[u8])>,
  ) -> Self {
    let (h, s, v) = match hsv {
      Some((h, s, v)) => (
        Self::ident(Some(h)),
        Self::ident(Some(s)),
        Self::ident(Some(v)),
      ),
      None => ((0, 0), (0, 0), (0, 0)),
    };
    Self {
      idents: [
        Self::ident(luma),
        Self::ident(luma_u16),
        Self::ident(rgb),
        Self::ident(rgba),
        Self::ident(rgb_u16),
        Self::ident(rgba_u16),
        Self::ident(rgb_f32),
        h,
        s,
        v,
      ],
    }
  }
}

/// Enforces the per-frame frozen output configuration for resampling
/// sinkers — presence AND buffer identity of every output the emit
/// closures consult. Shared by every routed format's resampled paths.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono"
))]
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(super) fn frozen_outputs_check(
  resample_outputs: &mut Option<FrozenOutputs>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  rgb_u16: &Option<&mut [u16]>,
  rgba_u16: &Option<&mut [u16]>,
  rgb_f32: &Option<&mut [f32]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
) -> Result<(), MixedSinkerError> {
  let snapshot = FrozenOutputs::snapshot(
    luma.as_deref(),
    luma_u16.as_deref(),
    rgb.as_deref(),
    rgba.as_deref(),
    rgb_u16.as_deref(),
    rgba_u16.as_deref(),
    rgb_f32.as_deref(),
    hsv.as_mut().map(|f| {
      let (h, s, v) = f.hsv();
      (&h[..], &s[..], &v[..])
    }),
  );
  match resample_outputs {
    None => *resample_outputs = Some(snapshot),
    Some(frozen) if *frozen != snapshot => {
      return Err(MixedSinkerError::ResampleOutputsChanged(
        ResampleOutputsChanged::new(idx),
      ));
    }
    Some(_) => {}
  }
  Ok(())
}

/// Mid-frame output-set change payload for
/// [`MixedSinkerError::ResampleOutputsChanged`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResampleOutputsChanged {
  /// Source row whose `process` call observed the changed output set.
  row: usize,
}

impl ResampleOutputsChanged {
  /// Constructs a new `ResampleOutputsChanged` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(row: usize) -> Self {
    Self { row }
  }

  /// Source row whose `process` call observed the changed output set.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Errors returned by [`MixedSinker`] configuration and per-frame
/// preflight.
///
/// All variants are recoverable: the sinker never mutates caller
/// buffers on an error return, so the caller can inspect the variant,
/// rebuild or resize buffers, and retry.
///
/// **Note (API change):** the former `*BufferTooShort` / `HsvPlaneTooShort`
/// variants were renamed to `Insufficient*Buffer` / `InsufficientHsvPlane`
/// and their payload structs renamed from `BufferTooShort` /
/// `HsvPlaneTooShort` to `InsufficientBuffer` / `InsufficientHsvPlane`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, IsVariant, TryUnwrap, Unwrap, Error)]
#[non_exhaustive]
pub enum MixedSinkerError {
  /// The frame handed to the walker does not match the dimensions
  /// declared at [`MixedSinker::new`]. Returned from
  /// [`PixelSink::begin_frame`] before any row is processed.
  #[error(
    "MixedSinker frame dimensions mismatch: configured {}x{} but got {}x{}",
    .0.configured_w(), .0.configured_h(), .0.frame_w(), .0.frame_h()
  )]
  DimensionMismatch(DimensionMismatch),

  /// RGB buffer attached via [`MixedSinker::with_rgb`] /
  /// [`MixedSinker::set_rgb`] is shorter than `width x height x 3`.
  #[error("MixedSinker insufficient rgb buffer: expected >= {} bytes, got {}", .0.expected(), .0.actual())]
  InsufficientRgbBuffer(InsufficientBuffer),

  /// `u16` RGB buffer attached via [`MixedSinker::with_rgb_u16`] /
  /// [`MixedSinker::set_rgb_u16`] is shorter than `width x height x 3`
  /// `u16` elements. Only the high‑bit‑depth source impls
  /// (currently [`Yuv420p10`](crate::source::Yuv420p10)) write into this
  /// buffer.
  #[error("MixedSinker insufficient rgb_u16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbU16Buffer(InsufficientBuffer),

  /// Native-depth `u16` luma buffer attached via per-format
  /// `with_luma_u16` is shorter than `width x height` `u16`
  /// elements. Tier 4 sources (V210 / Y210 / Y212 / Y216) are the
  /// first consumers of this API.
  #[error("MixedSinker insufficient luma_u16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientLumaU16Buffer(InsufficientBuffer),

  /// RGBA buffer attached via [`MixedSinker::with_rgba`] /
  /// [`MixedSinker::set_rgba`] is shorter than `width x height x 4`.
  /// The fourth byte per pixel is alpha — opaque (`0xFF`) by default
  /// when the source has no alpha plane.
  #[error("MixedSinker insufficient rgba buffer: expected >= {} bytes, got {}", .0.expected(), .0.actual())]
  InsufficientRgbaBuffer(InsufficientBuffer),

  /// `u16` RGBA buffer attached via `with_rgba_u16` / `set_rgba_u16`
  /// (per-format impl, not yet shipped on any sink) is shorter than
  /// `width x height x 4` `u16` elements. Only high‑bit‑depth source
  /// impls write into this buffer; the fourth `u16` per pixel is
  /// alpha — opaque (`(1 << BITS) - 1`) by default when the source
  /// has no alpha plane.
  #[error("MixedSinker insufficient rgba_u16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbaU16Buffer(InsufficientBuffer),

  /// `f32` RGB buffer attached via per-format `with_rgb_f32` /
  /// `set_rgb_f32` is shorter than `width x height x 3` `f32` elements.
  /// Only float-source impls (currently
  /// [`Rgbf32`](crate::source::Rgbf32)) write into this buffer.
  #[error("MixedSinker insufficient rgb_f32 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbF32Buffer(InsufficientBuffer),

  /// `half::f16` RGB buffer attached via per-format `with_rgb_f16` /
  /// `set_rgb_f16` is shorter than `width x height x 3` `f16` elements.
  /// Only half-float-source impls (currently
  /// [`Rgbf16`](crate::source::Rgbf16)) write into this buffer.
  #[error("MixedSinker insufficient rgb_f16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbF16Buffer(InsufficientBuffer),

  /// `f32` RGBA buffer attached via per-format `with_rgba_f32` /
  /// `set_rgba_f32` is shorter than `width x height x 4` `f32` elements.
  /// Only float-planar-GBR source impls write into this buffer.
  #[error("MixedSinker insufficient rgba_f32 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbaF32Buffer(InsufficientBuffer),

  /// `half::f16` RGBA buffer attached via per-format `with_rgba_f16` /
  /// `set_rgba_f16` is shorter than `width x height x 4` `f16` elements.
  /// Only float-planar-GBR source impls write into this buffer.
  #[error("MixedSinker insufficient rgba_f16 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientRgbaF16Buffer(InsufficientBuffer),

  /// `f32` XYZ buffer attached via `with_xyz_f32` / `set_xyz_f32` is
  /// shorter than `width x height x 3` `f32` elements. Only the
  /// [`Xyz12`](crate::source::Xyz12) source impl writes into this buffer.
  #[error("MixedSinker insufficient xyz_f32 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientXyzF32Buffer(InsufficientBuffer),

  /// `f32` luma buffer attached via `with_luma_f32` / `set_luma_f32` is
  /// shorter than `width x height` `f32` elements.
  #[error("MixedSinker insufficient luma_f32 buffer: expected >= {} elements, got {}", .0.expected(), .0.actual())]
  InsufficientLumaF32Buffer(InsufficientBuffer),

  /// Luma buffer is shorter than `width x height`.
  #[error("MixedSinker insufficient luma buffer: expected >= {} bytes, got {}", .0.expected(), .0.actual())]
  InsufficientLumaBuffer(InsufficientBuffer),

  /// One of the three HSV planes is shorter than `width x height`.
  #[error("MixedSinker insufficient hsv {:?} plane: expected >= {} bytes, got {}", .0.which(), .0.expected(), .0.actual())]
  InsufficientHsvPlane(InsufficientHsvPlane),

  /// Declared frame geometry does not fit in `usize`. Only reachable
  /// on 32‑bit targets (wasm32, i686) with extreme dimensions.
  #[error("MixedSinker frame size overflows usize: {} x {} x channels={}", .0.width(), .0.height(), .0.channels())]
  GeometryOverflow(GeometryOverflow),

  /// A row handed directly to [`PixelSink::process`] has a slice
  /// length that doesn't match the sink's configured width. Returned
  /// by `process` as a defense-in-depth check — [`PixelSink::begin_frame`]
  /// already validates frame-level dimensions, but this catches
  /// direct `process` callers that bypass the walker (hand-crafted
  /// rows, replayed rows, etc.) before a wrong-shaped slice reaches
  /// an unsafe SIMD kernel.
  ///
  /// Lengths are expressed in **slice elements** — `u8` bytes for
  /// the 8‑bit source rows (Y, U/V half, UV/VU half) and `u16`
  /// elements for the 10‑bit source rows (Y10, U/V half 10). The
  /// message deliberately says "elements" rather than "bytes" so the
  /// same variant can serve both the `u8` and `u16` row families.
  #[error(
    "MixedSinker row shape mismatch at row {}: {} slice has {} elements, expected {}",
    .0.row(), .0.which(), .0.actual(), .0.expected()
  )]
  RowShapeMismatch(RowShapeMismatch),

  /// A row handed to [`PixelSink::process`] has `row.row() >=
  /// configured_height`. The walker bounds `idx < height` via its
  /// `for row in 0..h` loop combined with the `begin_frame`
  /// dimension check, but a direct caller could pass any value.
  /// Returning an error instead of slice-indexing past the end keeps
  /// the no-panic contract intact.
  #[error(
    "MixedSinker row index {} is out of range for configured height {}",
    .0.row(), .0.configured_height()
  )]
  RowIndexOutOfRange(RowIndexOutOfRange),

  /// The sinker's configured `width` violates the format's chroma-group
  /// stride requirement. For 4:2:0 / 4:2:2 formats the width must be
  /// even (`WidthAlignmentRequirement::Even`); for planar 4:1:0
  /// ([`Yuv410p`](crate::source::Yuv410p)) and packed 4:1:1
  /// ([`Uyyvyy411`](crate::source::Uyyvyy411)) the width must be a
  /// multiple of 4 (`WidthAlignmentRequirement::MultipleOfFour`).
  /// Planar 4:1:1 ([`Yuv411p`](crate::source::Yuv411p)) accepts
  /// non-4-aligned widths via `width.div_ceil(4)` and does not produce
  /// this error. Supersedes the former `OddWidth` (even-only) and
  /// `WidthNotMultipleOf4` variants.
  ///
  /// `MixedSinker::new` is infallible and accepts any width, so this error
  /// surfaces the misconfiguration at the first use site
  /// ([`PixelSink::begin_frame`] or [`PixelSink::process`]) before any row
  /// primitive is invoked, preserving the no-panic contract.
  #[error("MixedSinker configured width {} {}", .0.width(), .0.required())]
  WidthAlignment(WidthAlignment),

  /// Building the resampling plan failed in
  /// [`MixedSinker::with_resampler`]: the strategy rejected the
  /// requested output geometry. Surfaces before the sinker exists, so
  /// no buffer state is touched.
  #[error(transparent)]
  Resample(#[from] ResampleError),

  /// On a resampling sinker the attached-output configuration is
  /// frozen per frame: streams carry frame progress, so an output
  /// attached, detached, or **replaced with a different buffer**
  /// after the first processed row would silently miss (or split)
  /// the rows already finalized. The offending `process` call fails
  /// before any stream mutates caller output; re-attach and call
  /// [`PixelSink::begin_frame`] to restart the frame with the new
  /// configuration. The direct (identity) path is unaffected.
  #[error(
    "MixedSinker resampled output set changed mid-frame at source row {}; \
     restart the frame via begin_frame",
    .0.row()
  )]
  ResampleOutputsChanged(ResampleOutputsChanged),
}

/// Identifies which slice of a multi‑plane source row mismatched in
/// [`MixedSinkerError::RowShapeMismatch`].
///
/// `#[non_exhaustive]` because each new source format the crate grows
/// support for — YUV422p / YUV444p (full‑width chroma), P010 / P016
/// (10/16‑bit planes), etc. — will add its own variant. Pattern
/// matches from downstream code should include a `_ => …` arm.
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash, IsVariant)]
#[non_exhaustive]
pub enum RowSlice {
  /// Y (luma) plane — every 4:2:0 / 4:2:2 / 4:4:4 source.
  #[display("Y")]
  Y,
  /// Half‑width U (Cb) plane in a planar 4:2:0 source ([`Yuv420p`]).
  #[display("U Half")]
  UHalf,
  /// Half‑width V (Cr) plane in a planar 4:2:0 source ([`Yuv420p`]).
  #[display("V Half")]
  VHalf,
  /// Quarter‑width U (Cb) plane in a planar 4:1:1 / 4:1:0 source
  /// ([`Yuv411p`](crate::source::Yuv411p) — DV-NTSC legacy;
  /// [`Yuv410p`](crate::source::Yuv410p) — Cinepak / extreme-old codecs).
  /// `width.div_ceil(4)` bytes per row — each chroma sample covers
  /// four Y columns horizontally. Yuv410p enforces `width % 4 == 0`
  /// at the frame layer (so `width.div_ceil(4) == width / 4`); Yuv411p
  /// accepts arbitrary widths via FFmpeg ceiling chroma. In 4:1:0 the
  /// same chroma row also covers four consecutive Y rows vertically;
  /// in 4:1:1 chroma is full-height.
  #[display("U Quarter")]
  UQuarter,
  /// Quarter‑width V (Cr) plane in a planar 4:1:1 / 4:1:0 source
  /// ([`Yuv411p`](crate::source::Yuv411p) /
  /// [`Yuv410p`](crate::source::Yuv410p)). `width.div_ceil(4)` bytes per
  /// row (see [`Self::UQuarter`] for the Yuv410p-vs-Yuv411p
  /// width-rounding distinction).
  #[display("V Quarter")]
  VQuarter,
  /// Half‑width interleaved UV plane in a semi‑planar 4:2:0 source
  /// ([`Nv12`]). Each row is `U0, V0, U1, V1, …` for `width / 2` pairs.
  #[display("UV Half")]
  UvHalf,
  /// Half‑width interleaved VU plane in a semi‑planar 4:2:0 source
  /// ([`Nv21`]). Each row is `V0, U0, V1, U1, …` for `width / 2`
  /// pairs — byte order swapped relative to [`Self::UvHalf`].
  #[display("VU Half")]
  VuHalf,
  /// Full-width U (Cb) plane in a planar 4:4:4 source
  /// ([`Yuv444p`](crate::source::Yuv444p)). `width` bytes per row.
  #[display("U Full")]
  UFull,
  /// Full-width V (Cr) plane in a planar 4:4:4 source
  /// ([`Yuv444p`](crate::source::Yuv444p)). `width` bytes per row.
  #[display("V Full")]
  VFull,
  /// Full-width alpha plane in an 8‑bit YUVA source
  /// ([`Yuva420p`](crate::source::Yuva420p)). `width` bytes per row
  /// (1:1 with Y).
  #[display("A Full")]
  AFull,
  /// Full-width U row of a **10-bit** 4:4:4 planar source. `u16`
  /// samples, `width` elements.
  #[display("U Full 10")]
  UFull10,
  /// Full-width V row of a **10-bit** 4:4:4 planar source.
  #[display("V Full 10")]
  VFull10,
  /// Full-width alpha row of a **10-bit** 4:4:4 planar source with an
  /// alpha plane ([`Yuva444p10`](crate::source::Yuva444p10)). `u16`
  /// samples, `width` elements, low-bit-packed.
  #[display("A Full 10")]
  AFull10,
  /// Full-width U row of a **12-bit** 4:4:4 planar source.
  #[display("U Full 12")]
  UFull12,
  /// Full-width V row of a **12-bit** 4:4:4 planar source.
  #[display("V Full 12")]
  VFull12,
  /// Full-width alpha row of a **12-bit** YUVA planar source
  /// ([`Yuva422p12`](crate::source::Yuva422p12) /
  /// [`Yuva444p12`](crate::source::Yuva444p12)). `u16` samples, `width`
  /// elements, low-bit-packed.
  #[display("A Full 12")]
  AFull12,
  /// Full-width U row of a **14-bit** 4:4:4 planar source.
  #[display("U Full 14")]
  UFull14,
  /// Full-width V row of a **14-bit** 4:4:4 planar source.
  #[display("V Full 14")]
  VFull14,
  /// Full-width alpha row of a **14-bit** YUVA planar source
  /// ([`Yuva444p14`](crate::source::Yuva444p14)). `u16` samples, `width`
  /// elements, low-bit-packed.
  #[display("A Full 14")]
  AFull14,
  /// Full‑width interleaved UV plane in a semi‑planar **4:4:4** source
  /// ([`Nv24`](crate::source::Nv24)). Each row is `U0, V0, U1, V1, …` for
  /// `width` pairs (`2 * width` bytes). One UV pair per Y pixel — no
  /// chroma subsampling.
  #[display("UV Full")]
  UvFull,
  /// Full‑width interleaved VU plane in a semi‑planar **4:4:4** source
  /// ([`Nv42`](crate::source::Nv42)). Each row is `V0, U0, V1, U1, …` for
  /// `width` pairs — byte order swapped relative to [`Self::UvFull`].
  #[display("VU Full")]
  VuFull,
  /// Full‑width Y row of a **9‑bit** planar source
  /// ([`Yuv420p9`](crate::source::Yuv420p9) /
  /// [`Yuv422p9`](crate::source::Yuv422p9) /
  /// [`Yuv444p9`](crate::source::Yuv444p9)). `u16` samples, `width`
  /// elements (low 9 bits active).
  #[display("Y9")]
  Y9,
  /// Half‑width U row of a **9‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 9")]
  UHalf9,
  /// Half‑width V row of a **9‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 9")]
  VHalf9,
  /// Full‑width U row of a **9‑bit** 4:4:4 planar source.
  #[display("U Full 9")]
  UFull9,
  /// Full‑width V row of a **9‑bit** 4:4:4 planar source.
  #[display("V Full 9")]
  VFull9,
  /// Full-width alpha row of a **9-bit** YUVA planar source
  /// ([`Yuva420p9`](crate::source::Yuva420p9)). `u16` samples, `width`
  /// elements, low-bit-packed.
  #[display("A Full 9")]
  AFull9,
  /// Full‑width Y row of a **10‑bit** planar source ([`Yuv420p10`]).
  /// `u16` samples, `width` elements.
  #[display("Y10")]
  Y10,
  /// Half‑width U row of a **10‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 10")]
  UHalf10,
  /// Half‑width V row of a **10‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 10")]
  VHalf10,
  /// Half‑width interleaved UV row of a **10‑bit semi‑planar** source
  /// ([`P010`]). `u16` samples, `width` elements laid out as
  /// `U0, V0, U1, V1, …` (high‑bit‑packed: each element's 10 active
  /// bits sit in the high 10 of its `u16`).
  #[display("UV Half 10")]
  UvHalf10,
  /// Full‑width Y row of a **12‑bit** source — used for both the
  /// planar ([`Yuv420p12`], low‑bit‑packed) and semi‑planar
  /// ([`P012`], high‑bit‑packed) families. `u16` samples, `width`
  /// elements. The packing direction depends on the source format;
  /// the row‑shape check only verifies length, so a single variant
  /// covers both.
  #[display("Y12")]
  Y12,
  /// Half‑width U row of a **12‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 12")]
  UHalf12,
  /// Half‑width V row of a **12‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 12")]
  VHalf12,
  /// Half‑width interleaved UV row of a **12‑bit semi‑planar** source
  /// ([`P012`]). `u16` samples, `width` elements (high‑bit‑packed: 12
  /// active bits in the high 12 of each `u16`).
  #[display("UV Half 12")]
  UvHalf12,
  /// Full‑width Y row of a **14‑bit** planar source ([`Yuv420p14`]).
  /// `u16` samples, `width` elements, low‑bit‑packed.
  #[display("Y14")]
  Y14,
  /// Half‑width U row of a **14‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 14")]
  UHalf14,
  /// Half‑width V row of a **14‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 14")]
  VHalf14,
  /// Full‑width Y row of a **16‑bit** source — used for both the
  /// planar ([`Yuv420p16`](crate::source::Yuv420p16)) and semi‑planar
  /// ([`P016`](crate::source::P016)) families. At 16 bits there is no
  /// high‑vs‑low packing distinction.
  #[display("Y16")]
  Y16,
  /// Half‑width U row of a **16‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("U Half 16")]
  UHalf16,
  /// Half‑width V row of a **16‑bit** planar source. `u16` samples,
  /// `width / 2` elements.
  #[display("V Half 16")]
  VHalf16,
  /// Half‑width interleaved UV row of a **16‑bit semi‑planar** source
  /// ([`P016`](crate::source::P016)). `u16` samples, `width` elements.
  #[display("UV Half 16")]
  UvHalf16,
  /// Full-width alpha row of a **16-bit** YUVA planar source
  /// ([`Yuva420p16`](crate::source::Yuva420p16) /
  /// [`Yuva444p16`](crate::source::Yuva444p16)). `u16` samples,
  /// `width` elements (full u16 range).
  #[display("A Full 16")]
  AFull16,
  /// Full-width U row of a **16-bit** 4:4:4 planar source
  /// ([`Yuv444p16`](crate::source::Yuv444p16) /
  /// [`Yuva444p16`](crate::source::Yuva444p16)). `u16` samples,
  /// `width` elements (full u16 range).
  #[display("U Full 16")]
  UFull16,
  /// Full-width V row of a **16-bit** 4:4:4 planar source. `u16`
  /// samples, `width` elements (full u16 range).
  #[display("V Full 16")]
  VFull16,
  /// Full‑width interleaved UV row of a **10‑bit semi‑planar 4:4:4**
  /// source ([`P410`](crate::source::P410)). `u16` samples, `2 * width`
  /// elements, high‑bit‑packed.
  #[display("UV Full 10")]
  UvFull10,
  /// Full‑width interleaved UV row of a **12‑bit semi‑planar 4:4:4**
  /// source ([`P412`](crate::source::P412)). `u16` samples, `2 * width`
  /// elements, high‑bit‑packed.
  #[display("UV Full 12")]
  UvFull12,
  /// Full‑width interleaved UV row of a **16‑bit semi‑planar 4:4:4**
  /// source ([`P416`](crate::source::P416)). `u16` samples, `2 * width`
  /// elements (no high/low packing distinction at 16 bits).
  #[display("UV Full 16")]
  UvFull16,
  /// `above` row of an **8-bit Bayer** source
  /// ([`Bayer`](crate::raw::Bayer)). `u8` samples, `width` elements;
  /// supplied by the walker via the **mirror-by-2** boundary
  /// contract — see [`crate::raw::BayerRow::above`] — so at the
  /// top edge this is `mid_row(1)`, not `mid` itself. Replicate
  /// fallback (`above == mid`) only when `height < 2` (no mirror
  /// partner exists).
  #[display("Bayer Above")]
  BayerAbove,
  /// `mid` row of an **8-bit Bayer** source. `u8` samples, `width`
  /// elements — the row currently being produced.
  #[display("Bayer Mid")]
  BayerMid,
  /// `below` row of an **8-bit Bayer** source. `u8` samples, `width`
  /// elements; mirror-by-2 supplies `mid_row(h - 2)` at the bottom
  /// edge — see [`crate::raw::BayerRow::below`]. Replicate fallback
  /// (`below == mid`) only when `height < 2`.
  #[display("Bayer Below")]
  BayerBelow,
  /// `above` row of a **high-bit-depth Bayer** source
  /// ([`Bayer16<BITS>`](crate::raw::Bayer16)). `u16` samples,
  /// `width` elements; mirror-by-2 supplies `mid_row(1)` at the
  /// top edge. Replicate fallback (`above == mid`) only when
  /// `height < 2`.
  #[display("Bayer16 Above")]
  Bayer16Above,
  /// `mid` row of a **high-bit-depth Bayer** source. `u16` samples,
  /// `width` elements.
  #[display("Bayer16 Mid")]
  Bayer16Mid,
  /// `below` row of a **high-bit-depth Bayer** source. `u16`
  /// samples, `width` elements; mirror-by-2 supplies
  /// `mid_row(h - 2)` at the bottom edge. Replicate fallback
  /// (`below == mid`) only when `height < 2`.
  #[display("Bayer16 Below")]
  Bayer16Below,
  /// Pixel-index row of a [`Pal8`](crate::raw::Pal8) source.
  /// `u8` samples, `width` elements — each value is an index into
  /// the 256-entry BGRA palette carried alongside in `Pal8Row`.
  #[display("Pal8 index row")]
  Pal8IndexRow,
  /// Packed **RGB565** LE row of an [`Rgb565`](crate::source::Rgb565) source.
  /// `2 * width` `u8` bytes — one `u16` LE word per pixel.
  #[display("RGB565 packed")]
  Rgb565Packed,
  /// Packed **BGR565** LE row of a [`Bgr565`](crate::source::Bgr565) source.
  /// Same `2 * width` byte shape as [`Rgb565Packed`](Self::Rgb565Packed)
  /// with R↔B channel positions swapped.
  #[display("BGR565 packed")]
  Bgr565Packed,
  /// Packed **RGB555** LE row of an [`Rgb555`](crate::source::Rgb555) source.
  /// `2 * width` `u8` bytes — one `u16` LE word per pixel (bit 15 unused).
  #[display("RGB555 packed")]
  Rgb555Packed,
  /// Packed **BGR555** LE row of a [`Bgr555`](crate::source::Bgr555) source.
  /// Same shape as [`Rgb555Packed`](Self::Rgb555Packed) with R↔B swapped.
  #[display("BGR555 packed")]
  Bgr555Packed,
  /// Packed **RGB444** LE row of an [`Rgb444`](crate::source::Rgb444) source.
  /// `2 * width` `u8` bytes — one `u16` LE word per pixel (bits [15:12]
  /// unused).
  #[display("RGB444 packed")]
  Rgb444Packed,
  /// Packed **BGR444** LE row of a [`Bgr444`](crate::source::Bgr444) source.
  /// Same shape as [`Rgb444Packed`](Self::Rgb444Packed) with R↔B swapped.
  #[display("BGR444 packed")]
  Bgr444Packed,
  /// Packed `R, G, B` row of an [`Rgb24`](crate::source::Rgb24) source.
  /// `3 * width` `u8` bytes.
  #[display("RGB packed")]
  RgbPacked,
  /// Packed `B, G, R` row of a [`Bgr24`](crate::source::Bgr24) source.
  /// `3 * width` `u8` bytes (channel-order swapped relative to
  /// [`RgbPacked`](Self::RgbPacked)).
  #[display("BGR packed")]
  BgrPacked,
  /// Packed `R, G, B, A` row of an [`Rgba`](crate::source::Rgba) source.
  /// `4 * width` `u8` bytes — alpha is real (not padding).
  #[display("RGBA packed")]
  RgbaPacked,
  /// Packed `B, G, R, A` row of a [`Bgra`](crate::source::Bgra) source.
  /// `4 * width` `u8` bytes — alpha lane preserved, channel order
  /// swapped on the first three bytes relative to
  /// [`RgbaPacked`](Self::RgbaPacked).
  #[display("BGRA packed")]
  BgraPacked,
  /// Packed `A, R, G, B` row of an [`Argb`](crate::source::Argb) source.
  /// `4 * width` `u8` bytes — alpha at the **leading** position vs
  /// [`RgbaPacked`](Self::RgbaPacked).
  #[display("ARGB packed")]
  ArgbPacked,
  /// Packed `A, B, G, R` row of an [`Abgr`](crate::source::Abgr) source.
  /// `4 * width` `u8` bytes — leading alpha + reversed RGB order vs
  /// [`ArgbPacked`](Self::ArgbPacked).
  #[display("ABGR packed")]
  AbgrPacked,
  /// Packed `X, R, G, B` row of an [`Xrgb`](crate::source::Xrgb) source
  /// (FFmpeg `0rgb`). `4 * width` `u8` bytes — leading **padding**
  /// byte (not alpha).
  #[display("XRGB packed")]
  XrgbPacked,
  /// Packed `R, G, B, X` row of an [`Rgbx`](crate::source::Rgbx) source
  /// (FFmpeg `rgb0`). `4 * width` `u8` bytes — trailing padding byte.
  #[display("RGBX packed")]
  RgbxPacked,
  /// Packed `X, B, G, R` row of an [`Xbgr`](crate::source::Xbgr) source
  /// (FFmpeg `0bgr`). `4 * width` `u8` bytes — leading padding byte
  /// + reversed RGB order vs [`XrgbPacked`](Self::XrgbPacked).
  #[display("XBGR packed")]
  XbgrPacked,
  /// Packed `B, G, R, X` row of a [`Bgrx`](crate::source::Bgrx) source
  /// (FFmpeg `bgr0`). `4 * width` `u8` bytes — trailing padding byte
  /// + reversed RGB order vs [`RgbxPacked`](Self::RgbxPacked).
  #[display("BGRX packed")]
  BgrxPacked,
  /// Packed `X2RGB10` LE row of an
  /// [`X2Rgb10`](crate::source::X2Rgb10) source. `4 * width` `u8` bytes
  /// (one little-endian `u32` per pixel with `(MSB) 2X | 10R | 10G |
  /// 10B (LSB)` packing).
  #[display("X2RGB10 packed")]
  X2Rgb10Packed,
  /// Packed `X2BGR10` LE row of an
  /// [`X2Bgr10`](crate::source::X2Bgr10) source. `4 * width` `u8` bytes
  /// — channel positions reversed relative to
  /// [`X2Rgb10Packed`](Self::X2Rgb10Packed).
  #[display("X2BGR10 packed")]
  X2Bgr10Packed,
  /// Packed `Y0, U0, Y1, V0, …` row of a
  /// [`Yuyv422`](crate::source::Yuyv422) source (FFmpeg `yuyv422` /
  /// YUY2). `2 * width` `u8` bytes — Y in even byte positions, U/V
  /// in odd positions with U preceding V.
  #[display("YUYV422 packed")]
  Yuyv422Packed,
  /// Packed `U0, Y0, V0, Y1, …` row of a
  /// [`Uyvy422`](crate::source::Uyvy422) source (FFmpeg `uyvy422` /
  /// UYVY). `2 * width` `u8` bytes — Y in odd byte positions, U/V
  /// in even positions with U preceding V.
  #[display("UYVY422 packed")]
  Uyvy422Packed,
  /// Packed `Y0, V0, Y1, U0, …` row of a
  /// [`Yvyu422`](crate::source::Yvyu422) source (FFmpeg `yvyu422` /
  /// YVYU). `2 * width` `u8` bytes — Y in even byte positions, V/U
  /// in odd positions with V preceding U (chroma order swapped vs
  /// [`Yuyv422Packed`](Self::Yuyv422Packed)).
  #[display("YVYU422 packed")]
  Yvyu422Packed,
  /// Packed `U0, Y0, Y1, V0, Y2, Y3, …` row of a
  /// [`Uyyvyy411`](crate::source::Uyyvyy411) source (FFmpeg
  /// `uyyvyy411`). `width * 3 / 2` `u8` bytes — one (U, V) chroma
  /// pair shared across 4 luma samples (4:1:1 horizontal
  /// subsampling, 12 bpp, DV legacy).
  #[display("UYYVYY411 packed")]
  Uyyvyy411Packed,
  /// Packed `v210` row of a [`V210`](crate::source::V210) source —
  /// Tier 4 10-bit pro-broadcast SDI capture format. Each 16-byte
  /// word holds 12 x 10-bit samples = 6 pixels (4:2:2: 6 Y +
  /// 3 Cb + 3 Cr). Row length: `(width / 6) * 16` `u8` bytes.
  #[display("V210 packed")]
  V210Packed,
  /// Packed `y210` row of a [`Y210`](crate::source::Y210) source —
  /// Tier 4 10-bit MSB-aligned in u16 with YUYV422 byte order.
  /// Row length: `2 * width` `u16` elements (= `4 * width` bytes).
  #[display("Y210 packed")]
  Y210Packed,
  /// Packed `y212` row — same shape as Y210 with BITS=12.
  #[display("Y212 packed")]
  Y212Packed,
  /// Packed `y216` row — same shape as Y210 with BITS=16.
  #[display("Y216 packed")]
  Y216Packed,
  /// Packed `v410` row of a `V410` source — Tier 5 10-bit 4:4:4
  /// packed format. One `u32` word per pixel; row length: `width`
  /// `u32` elements (= `4 * width` bytes).
  #[display("V410 packed")]
  V410Packed,
  /// Packed `v30x` row of a `V30X` source — Tier 5 10-bit 4:4:4
  /// packed format, sibling of V410 with 2-bit padding at the
  /// **low** end. One `u32` word per pixel; row length: `width`
  /// `u32` elements (= `4 * width` bytes).
  #[display("V30X packed")]
  V30XPacked,
  /// Packed `xv36` row of an `Xv36` source — Tier 5 16-bit 4:4:4
  /// packed format. Four `u16` elements per pixel (one per channel);
  /// row length: `4 * width` `u16` elements (= `8 * width` bytes).
  #[display("XV36 packed")]
  Xv36Packed,
  /// Packed `vuya` row of a `Vuya` source — Tier 5 8-bit 4:4:4
  /// packed format. Four bytes per pixel in V/U/Y/A order; row
  /// length: `4 * width` bytes.
  #[display("VUYA packed")]
  VuyaPacked,
  /// Packed `vuyx` row of a `Vuyx` source — Tier 5 8-bit 4:4:4
  /// packed format. Four bytes per pixel in V/U/Y/X order (X is
  /// padding); row length: `4 * width` bytes.
  #[display("VUYX packed")]
  VuyxPacked,
  /// Packed `ayuv64` row of an `Ayuv64` source — Tier 5 16-bit
  /// 4:4:4 packed format. Four `u16` elements per pixel in A/Y/U/V
  /// order; row length: `4 * width` `u16` elements (= `8 * width`
  /// bytes).
  #[display("AYUV64 packed")]
  Ayuv64Packed,
  /// Packed `R, G, B` row of an [`Rgbf32`](crate::source::Rgbf32) source —
  /// Tier 9 32-bit float per channel. Row length: `3 * width` `f32`
  /// elements (= `12 * width` bytes).
  #[display("RGBF32 packed")]
  RgbF32Packed,
  /// Packed `R, G, B` row of an [`Rgbf16`](crate::source::Rgbf16) source —
  /// Tier 9 16-bit half-precision float per channel. Row length:
  /// `3 * width` `half::f16` elements (= `6 * width` bytes).
  #[display("RGBF16 packed")]
  RgbF16Packed,
  /// Packed `X, Y, Z` row of an [`Xyz12`](crate::source::Xyz12) source —
  /// Tier 12 12-bit CIE XYZ packed in u16 triples — active 12 bits
  /// in `[15:4]`, low 4 bits zero (per FFmpeg
  /// `AV_PIX_FMT_XYZ12LE/BE`). Row length: `3 * width` `u16` elements
  /// (= `6 * width` bytes).
  #[display("XYZ12 packed")]
  Xyz12Packed,
  /// Green plane row of an 8-bit planar GBR source
  /// ([`Gbrp`](crate::source::Gbrp) /
  /// [`Gbrap`](crate::source::Gbrap)). `u8` samples, `width` elements.
  #[display("G plane")]
  GPlane,
  /// Blue plane row of an 8-bit planar GBR source. `u8` samples,
  /// `width` elements.
  #[display("B plane")]
  BPlane,
  /// Red plane row of an 8-bit planar GBR source. `u8` samples,
  /// `width` elements.
  #[display("R plane")]
  RPlane,
  /// Plane row of a float-32 planar GBR source (`Gbrpf32` /
  /// `Gbrapf32`). `f32` samples, `width` elements per plane.
  #[display("GBR f32 plane")]
  GbrF32Plane,
  /// Plane row of a float-16 planar GBR source (`Gbrpf16` /
  /// `Gbrapf16`). `half::f16` samples, `width` elements per plane.
  #[display("GBR f16 plane")]
  GbrF16Plane,
  /// Packed `R, G, B` row of an [`Rgb48`](crate::source::Rgb48) source —
  /// `width * 3` u16 elements (each channel 16 bits, R, G, B order).
  #[display("RGB48 packed")]
  Rgb48Packed,
  /// Packed `B, G, R` row of a [`Bgr48`](crate::source::Bgr48) source —
  /// `width * 3` u16 elements (channel order reversed vs
  /// [`Rgb48Packed`](Self::Rgb48Packed)).
  #[display("BGR48 packed")]
  Bgr48Packed,
  /// Packed `R, G, B, A` row of an [`Rgba64`](crate::source::Rgba64) source —
  /// `width * 4` u16 elements (each channel 16 bits; alpha is real).
  #[display("RGBA64 packed")]
  Rgba64Packed,
  /// Packed `B, G, R, A` row of a [`Bgra64`](crate::source::Bgra64) source —
  /// `width * 4` u16 elements (channel order reversed on RGB vs
  /// [`Rgba64Packed`](Self::Rgba64Packed); alpha at slot 3 is real).
  #[display("BGRA64 packed")]
  Bgra64Packed,
}

/// A sink that writes any subset of `{RGB, Luma, HSV}` into
/// caller-provided buffers.
///
/// Each output is optional — provide `Some(buffer)` to have that
/// channel written, leave it `None` to skip. Providing no outputs is
/// legal (the kernel still walks the source and calls `process`
/// for each row, but nothing is written).
///
/// When HSV is requested **without** RGB, `MixedSinker` keeps a single
/// row of intermediate RGB in an internal scratch buffer (allocated
/// lazily on first use). If RGB output is also requested, the user's
/// RGB buffer serves as the intermediate for HSV and no scratch is
/// allocated.
///
/// # Type parameters
///
/// `F` identifies the source format — `Yuv420p`, `Nv12`, `Nv21`,
/// `Yuv420p10`, `Yuv420p12`, `Yuv420p14`, `P010`, `P012`, etc. Each
/// format provides its own `impl PixelSink for MixedSinker<'_, F>`.
/// See the module‑level docs for the full list of shipped impls.
///
/// `R` is the resampling strategy deciding the sinker's **output**
/// geometry, injected via [`Self::with_resampler`]. The default
/// [`NoopResampler`] is the identity:
/// output geometry == source geometry, i.e. exactly the historical
/// behavior of [`Self::new`]. Output buffers always validate against
/// the output geometry; [`PixelSink::begin_frame`] always validates
/// the walker against the source geometry.
///
/// Formats route non-identity plans as they wire into the streaming
/// engine (currently [`Yuv420p`](crate::source::Yuv420p)). Every other
/// per-format [`PixelSink`] impl stays pinned to the default strategy:
/// a sinker built with a non-identity strategy can attach
/// (output-validated) buffers but does not implement [`PixelSink`], so
/// routing it through a walker is a compile error rather than a
/// geometry-mismatch panic.
pub struct MixedSinker<'a, F: SourceFormat, R = NoopResampler> {
  rgb: Option<&'a mut [u8]>,
  rgb_u16: Option<&'a mut [u16]>,
  rgb_f32: Option<&'a mut [f32]>,
  rgb_f16: Option<&'a mut [half::f16]>,
  rgba: Option<&'a mut [u8]>,
  rgba_u16: Option<&'a mut [u16]>,
  rgba_f32: Option<&'a mut [f32]>,
  rgba_f16: Option<&'a mut [half::f16]>,
  luma: Option<&'a mut [u8]>,
  luma_u16: Option<&'a mut [u16]>,
  luma_f32: Option<&'a mut [f32]>,
  // `HsvFrameMut` is cfg-gated to the same 15-feature any as the
  // per-format `process` impls that read it.
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  hsv: Option<HsvFrameMut<'a>>,
  /// Lossless linear-XYZ pass-through buffer used by the
  /// [`Xyz12`](crate::source::Xyz12) source's `with_xyz_f32` accessor.
  /// `None` for every other source format.
  #[cfg(feature = "xyz")]
  xyz_f32: Option<&'a mut [f32]>,
  width: usize,
  height: usize,
  /// Output geometry from the resampler's plan; equals
  /// `(width, height)` under the identity plan. Every output-buffer
  /// length validation sizes against these, never against the source
  /// geometry.
  out_width: usize,
  out_height: usize,
  /// The non-identity plan fixed by [`MixedSinker::with_resampler`];
  /// `None` for [`MixedSinker::new`] and identity plans (the sinker
  /// then takes the direct conversion path).
  plan: Option<ResamplePlan>,
  /// Row-stage area streams (color group / luma group) for formats
  /// that route non-identity plans. Lazily created in `process`,
  /// reset in `begin_frame`. Gated like the engine itself, widening
  /// as families wire in.
  #[cfg(any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono"
  ))]
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  rgb_stream: Option<crate::resample::AreaStream<u8>>,
  /// Row-stage area stream for high-bit packed-RGB sources (`u16`
  /// elements binned at native depth). Lazily created in `process`,
  /// reset in `begin_frame`. Gated to `rgb`; widens as high-bit
  /// families wire in.
  #[cfg(feature = "rgb")]
  rgb_stream_u16: Option<crate::resample::AreaStream<u16>>,
  /// Row-stage area stream for packed-float-RGB sources (`f32`
  /// elements binned in float). Lazily created in `process`, reset in
  /// `begin_frame`. Gated to the float family **and** the engine (the
  /// `AreaStream` machinery is gated to `yuv-planar` / `rgb`, which
  /// `rgb-float` does not imply).
  #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
  rgb_stream_f32: Option<crate::resample::AreaStream<f32>>,
  #[cfg(feature = "yuv-planar")]
  luma_stream: Option<crate::resample::AreaStream<u8>>,
  /// Output configuration frozen at a resampled frame's first
  /// processed row; `None` between frames. Captures presence AND
  /// attachment identity (pointer/length) of every output the emit
  /// closures consult, so both membership changes and same-channel
  /// buffer replacement trip
  /// [`MixedSinkerError::ResampleOutputsChanged`].
  #[cfg(any(
    feature = "yuv-planar",
    feature = "rgb",
    feature = "gbr",
    feature = "gray",
    feature = "xyz",
    feature = "bayer",
    feature = "mono"
  ))]
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  resample_outputs: Option<FrozenOutputs>,
  /// Whether resampled processing may take the native decimation tier
  /// (bin native planes, convert once at output resolution). Defaults
  /// to `true`; benchmarks and differential tests flip it to force the
  /// row-stage tier — the [`Self::with_simd`] pattern. Gated like the
  /// engine; widens per routed family.
  #[cfg(feature = "yuv-planar")]
  native: bool,
  /// Native-tier join state for the 4:2:0 planar family; lazily
  /// created in `process`, reset in `begin_frame`.
  #[cfg(feature = "yuv-planar")]
  native_420: Option<planar_8bit::NativeYuv420>,
  /// Lazily grown to `3 * width` bytes when HSV is requested without a
  /// user RGB buffer. Empty otherwise.
  ///
  /// Consumed by per-format `process` impls that derive HSV from RGB
  /// via the lazy scratch path. Under `--features "alloc"` alone (no
  /// per-format family), no `process` impl reads this field, so the
  /// cfg enumerates every source family.
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  rgb_scratch: Vec<u8>,
  /// Source-width `u16` RGB staging for high-bit packed-RGB resampling:
  /// the wire row converts here before feeding [`Self::rgb_stream_u16`].
  /// Lazily grown to `3 * width` `u16`; empty otherwise. Gated to `rgb`.
  #[cfg(feature = "rgb")]
  rgb_scratch_u16: Vec<u16>,
  /// Source-width `f32` RGB staging for packed-float-RGB resampling:
  /// the wire row converts here (host-native, lossless) before feeding
  /// [`Self::rgb_stream_f32`]. Lazily grown to `3 * width` `f32`; empty
  /// otherwise. Gated to the float family **and** the engine.
  #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
  rgb_scratch_f32: Vec<f32>,
  /// Whether row primitives dispatch to their SIMD backend. Defaults
  /// to `true`; benchmarks flip this with [`Self::with_simd`] /
  /// [`Self::set_simd`] to A/B test scalar vs SIMD on the same frame.
  simd: bool,
  /// Q8 fixed-point luma coefficients `(cr, cg, cb)` such that
  /// `luma = ((cr * R + cg * G + cb * B + 128) >> 8) as u8`. Only
  /// consulted by source impls that *derive* luma from RGB
  /// (currently the `Bayer` / `Bayer16<BITS>` family and the `Pal8`
  /// mono palette path — YUV impls memcpy from the native Y plane
  /// and ignore this field). Default: BT.709 `(54, 183, 19)`.
  #[cfg(any(feature = "bayer", feature = "mono"))]
  luma_coefficients_q8: (u32, u32, u32),
  _fmt: PhantomData<F>,
  _resampler: PhantomData<R>,
}

/// Luma coefficient set for sources that derive luma from RGB.
///
/// Only consulted by `MixedSinker` impls whose source is *not* YUV
/// (currently the Bayer / Bayer16 family — YUV impls memcpy from
/// the native Y plane). For Bayer the choice should match the
/// gamut your [`crate::raw::ColorCorrectionMatrix`] targets:
///
/// - CCM target = Rec.709 / sRGB → use [`Self::Bt709`] (the default)
/// - CCM target = Rec.2020 (UHDTV / HDR10) → use [`Self::Bt2020`]
/// - CCM target = DCI-P3 (cinema) → use [`Self::DciP3`]
/// - CCM target = ACEScg / ACES AP1 → use [`Self::AcesAp1`]
/// - CCM target = SDTV (rare for RAW) → use [`Self::Bt601`]
/// - CCM target = something else, or you've measured your own
///   weights → use [`Self::Custom`] (constructed via
///   [`Self::try_custom`] or [`Self::custom`])
///
/// Picking the wrong set still produces a **valid** luma plane,
/// but its numeric values won't match what a downstream
/// luma-driven analysis (scene-cut detection, brightness
/// thresholding, perceptual diff) expects for non-grayscale
/// content. Uniform-gray content is unaffected — every coefficient
/// set agrees on gray.
///
/// Each variant resolves to a Q8 `(cr, cg, cb)` triple summing to
/// `256` so `(cr * R + cg * G + cb * B + 128) >> 8` produces
/// `u8` luma without bias. The triples come from each standard's
/// published coefficients rounded to nearest u32.
#[derive(Debug, Clone, Copy, PartialEq, IsVariant)]
#[non_exhaustive]
pub enum LumaCoefficients {
  /// **BT.709 / sRGB** (`R=0.2126, G=0.7152, B=0.0722`) → Q8
  /// `(54, 183, 19)`. The default; most common output gamut and
  /// the implicit weights every YUV→RGB→luma video pipeline uses.
  Bt709,
  /// **BT.2020 / Rec.2020** (`R=0.2627, G=0.6780, B=0.0593`) → Q8
  /// `(67, 174, 15)`. UHDTV / HDR10 / Rec.2100 (HLG, PQ).
  Bt2020,
  /// **BT.601 / SMPTE 170M** (`R=0.2990, G=0.5870, B=0.1140`) →
  /// Q8 `(77, 150, 29)`. Legacy SDTV / NTSC / PAL. Rare for RAW
  /// pipelines but included for completeness.
  Bt601,
  /// **DCI-P3** (`R=0.228975, G=0.691739, B=0.079287`) → Q8
  /// `(59, 177, 20)`. Theatrical / cinema P3 displays. Note the
  /// **D65 white point** is the same as Rec.709, so for
  /// luma-only purposes this is close to `Bt709` (within ~1 LSB
  /// for most content).
  DciP3,
  /// **ACES AP1 / ACEScg** (`R=0.2722287, G=0.6740818,
  /// B=0.0536895`) → Q8 `(70, 172, 14)`. Cinema grading working
  /// space. Numerically very close to BT.2020. (Naïve nearest
  /// rounding gives `(70, 173, 14)` which sums to 257; the `cg`
  /// term is rounded down by 1 LSB so the triple sums to 256
  /// without biasing the `>> 8` divisor.)
  AcesAp1,
  /// Caller-supplied coefficients. Use [`Self::try_custom`] or
  /// [`Self::custom`] to construct — the inner
  /// [`CustomLumaCoefficients`] keeps fields private so every
  /// `Custom` value is guaranteed finite, non-negative, and
  /// magnitude-bounded.
  Custom(CustomLumaCoefficients),
}

/// Validated red / green / blue luma weights, accessible only through
/// [`LumaCoefficients::Custom`] (or [`Self::try_new`] /
/// [`Self::new`]).
///
/// Each weight is a finite, non-negative `f32` ≤
/// [`Self::MAX_COEFFICIENT`]. The bound is much tighter than
/// [`crate::raw::WhiteBalance::MAX_GAIN`] (`1e6`) because the luma
/// kernel multiplies these into a `u32` accumulator — see
/// [`Self::MAX_COEFFICIENT`] for the overflow analysis.
///
/// The struct intentionally has no public fields. Use
/// [`Self::r`] / [`Self::g`] / [`Self::b`] to read components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CustomLumaCoefficients {
  r: f32,
  g: f32,
  b: f32,
}

impl CustomLumaCoefficients {
  /// Maximum permitted per-channel weight. `10.0` is far above any
  /// realistic published luma coefficient (the standard sets all
  /// individual weights are ≤ `1.0`) and far below the value at
  /// which the per-pixel `u32` accumulator could overflow:
  /// `(coef * 256 + 0.5) as u32 ≤ 10 * 256 + 1 = 2_561`, so the
  /// largest per-row term is `2_561 * 255 = 653_055`, and the
  /// three-channel sum + bias `3 * 653_055 + 128 = 1_959_293` —
  /// six orders of magnitude below `u32::MAX`.
  ///
  /// `1e6` (the
  /// [`crate::raw::WhiteBalance::MAX_GAIN`] bound) **would not be
  /// safe here** — `1e6 * 256 = 256_000_000`, and `256_000_000 *
  /// 255 ≈ 6.5e10` overflows `u32`.
  pub const MAX_COEFFICIENT: f32 = 10.0;

  /// Constructs a [`CustomLumaCoefficients`] from explicit R / G / B
  /// weights, validating that each is **finite, non-negative, and
  /// ≤ [`Self::MAX_COEFFICIENT`]**.
  ///
  /// Returns [`LumaCoefficientsError`] for the first failing
  /// channel. A weight of `0` is permitted (the channel doesn't
  /// contribute to luma — degenerate but well-defined).
  ///
  /// The weights are *not* required to sum to `1.0`; sums far from
  /// `1.0` produce a brightness-scaled luma plane (the doc on
  /// [`LumaCoefficients`] flags this), which is sometimes
  /// intentional (matte / key extraction). Only NaN / ±∞ /
  /// negative / out-of-range weights are rejected because those
  /// would silently corrupt the luma plane via the `f32 → u32`
  /// saturating cast or overflow the accumulator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(r: f32, g: f32, b: f32) -> Result<Self, LumaCoefficientsError> {
    if !r.is_finite() {
      return Err(LumaCoefficientsError::NonFinite {
        channel: LumaChannel::R,
        value: r,
      });
    }
    if !g.is_finite() {
      return Err(LumaCoefficientsError::NonFinite {
        channel: LumaChannel::G,
        value: g,
      });
    }
    if !b.is_finite() {
      return Err(LumaCoefficientsError::NonFinite {
        channel: LumaChannel::B,
        value: b,
      });
    }
    if r < 0.0 {
      return Err(LumaCoefficientsError::Negative {
        channel: LumaChannel::R,
        value: r,
      });
    }
    if g < 0.0 {
      return Err(LumaCoefficientsError::Negative {
        channel: LumaChannel::G,
        value: g,
      });
    }
    if b < 0.0 {
      return Err(LumaCoefficientsError::Negative {
        channel: LumaChannel::B,
        value: b,
      });
    }
    if r > Self::MAX_COEFFICIENT {
      return Err(LumaCoefficientsError::OutOfBounds {
        channel: LumaChannel::R,
        value: r,
        max: Self::MAX_COEFFICIENT,
      });
    }
    if g > Self::MAX_COEFFICIENT {
      return Err(LumaCoefficientsError::OutOfBounds {
        channel: LumaChannel::G,
        value: g,
        max: Self::MAX_COEFFICIENT,
      });
    }
    if b > Self::MAX_COEFFICIENT {
      return Err(LumaCoefficientsError::OutOfBounds {
        channel: LumaChannel::B,
        value: b,
        max: Self::MAX_COEFFICIENT,
      });
    }
    Ok(Self { r, g, b })
  }

  /// Constructs a [`CustomLumaCoefficients`], panicking on invalid
  /// input. Prefer [`Self::try_new`] for caller-supplied values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(r: f32, g: f32, b: f32) -> Self {
    match Self::try_new(r, g, b) {
      Ok(c) => c,
      Err(_) => panic!("invalid CustomLumaCoefficients (non-finite, negative, or out of range)"),
    }
  }

  /// Red weight.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> f32 {
    self.r
  }

  /// Green weight.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> f32 {
    self.g
  }

  /// Blue weight.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> f32 {
    self.b
  }
}

/// Identifies which luma weight failed validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant)]
#[non_exhaustive]
pub enum LumaChannel {
  /// Red weight.
  R,
  /// Green weight.
  G,
  /// Blue weight.
  B,
}

/// Errors returned by [`CustomLumaCoefficients::try_new`] (and the
/// convenience [`LumaCoefficients::try_custom`]).
#[derive(Debug, Clone, Copy, PartialEq, IsVariant, Error)]
#[non_exhaustive]
pub enum LumaCoefficientsError {
  /// A weight is non-finite (NaN, +∞, or -∞).
  #[error("CustomLumaCoefficients.{channel:?} is non-finite (got {value})")]
  NonFinite {
    /// Which channel failed validation.
    channel: LumaChannel,
    /// The offending weight value.
    value: f32,
  },
  /// A weight is negative. Zero is allowed (zeroes the channel).
  #[error("CustomLumaCoefficients.{channel:?} is negative (got {value})")]
  Negative {
    /// Which channel failed validation.
    channel: LumaChannel,
    /// The offending weight value.
    value: f32,
  },
  /// A weight exceeds [`CustomLumaCoefficients::MAX_COEFFICIENT`]
  /// (`10.0`). The bound is far above any realistic luma weight
  /// but closes the door on values that would saturate the
  /// `f32 → u32` cast in [`LumaCoefficients::to_q8`] or overflow
  /// the per-row `u32` accumulator.
  #[error("CustomLumaCoefficients.{channel:?} = {value} exceeds the magnitude bound ({max})")]
  OutOfBounds {
    /// Which channel failed validation.
    channel: LumaChannel,
    /// The offending weight value.
    value: f32,
    /// The bound that was exceeded
    /// ([`CustomLumaCoefficients::MAX_COEFFICIENT`]).
    max: f32,
  },
}

impl LumaCoefficients {
  /// Resolves the coefficient set to its Q8 fixed-point triple
  /// `(cr, cg, cb)` such that
  /// `luma = ((cr * R + cg * G + cb * B + 128) >> 8) as u8`. The
  /// preset triples come from each standard's published weights
  /// rounded to nearest u32 and (for the published presets) sum
  /// to exactly `256`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn to_q8(self) -> (u32, u32, u32) {
    match self {
      Self::Bt709 => (54, 183, 19),
      Self::Bt2020 => (67, 174, 15),
      Self::Bt601 => (77, 150, 29),
      Self::DciP3 => (59, 177, 20),
      // Naïve nearest rounding gives `(70, 173, 14)` which sums
      // to 257; the `>> 8` divisor implicitly assumes 256, so we
      // shave 1 LSB off `cg` (the largest, smallest-relative-
      // -error coefficient). Resulting (R, G, B) error vs. the
      // published weights is `(+0.0012, -0.0022, +0.0010)`.
      Self::AcesAp1 => (70, 172, 14),
      // Custom values are guaranteed finite + non-negative +
      // ≤ `MAX_COEFFICIENT` (= 10.0) by `CustomLumaCoefficients::
      // try_new`, so the `as u32` cast cannot saturate to
      // `u32::MAX` and the downstream accumulator cannot overflow.
      Self::Custom(c) => (
        (c.r * 256.0 + 0.5) as u32,
        (c.g * 256.0 + 0.5) as u32,
        (c.b * 256.0 + 0.5) as u32,
      ),
    }
  }

  /// Constructs [`Self::Custom`] from explicit R / G / B weights,
  /// validating each via [`CustomLumaCoefficients::try_new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_custom(r: f32, g: f32, b: f32) -> Result<Self, LumaCoefficientsError> {
    match CustomLumaCoefficients::try_new(r, g, b) {
      Ok(c) => Ok(Self::Custom(c)),
      Err(e) => Err(e),
    }
  }

  /// Constructs [`Self::Custom`] from explicit R / G / B weights,
  /// panicking on invalid input. Prefer [`Self::try_custom`] for
  /// caller-supplied values.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn custom(r: f32, g: f32, b: f32) -> Self {
    Self::Custom(CustomLumaCoefficients::new(r, g, b))
  }
}

impl Default for LumaCoefficients {
  /// Default is [`Self::Bt709`] — matches the implicit weights
  /// every YUV-source → RGB → luma video pipeline uses.
  fn default() -> Self {
    Self::Bt709
  }
}

impl<F: SourceFormat> MixedSinker<'_, F, NoopResampler> {
  /// Creates an empty [`MixedSinker`] for the given dimensions, with
  /// the identity resampler (output geometry == source geometry).
  /// Attach output buffers with `with_rgb` / `with_luma` / `with_hsv`;
  /// each attachment validates that the buffer is at least
  /// `width * height * bytes_per_pixel` so short-buffer bugs surface
  /// *before* any rows are written — not after half the frame has
  /// been mutated. For a sinker whose outputs land at a smaller
  /// geometry, construct via [`MixedSinker::with_resampler`] instead.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn new(width: usize, height: usize) -> Self {
    Self::with_geometry(width, height, width, height)
  }
}

impl<F: SourceFormat, R> MixedSinker<'_, F, R> {
  /// Field initializer shared by [`MixedSinker::new`] and
  /// [`MixedSinker::with_resampler`]: source geometry plus the output
  /// geometry that buffer validation sizes against.
  fn with_geometry(width: usize, height: usize, out_width: usize, out_height: usize) -> Self {
    Self {
      rgb: None,
      rgb_u16: None,
      rgb_f32: None,
      rgb_f16: None,
      rgba: None,
      rgba_u16: None,
      rgba_f32: None,
      rgba_f16: None,
      luma: None,
      luma_u16: None,
      luma_f32: None,
      #[cfg(any(
        feature = "bayer",
        feature = "gbr",
        feature = "gray",
        feature = "mono",
        feature = "rgb",
        feature = "rgb-float",
        feature = "rgb-legacy",
        feature = "v210",
        feature = "xyz",
        feature = "y2xx",
        feature = "yuv-444-packed",
        feature = "yuv-packed",
        feature = "yuv-planar",
        feature = "yuv-semi-planar",
        feature = "yuva",
      ))]
      hsv: None,
      #[cfg(feature = "xyz")]
      xyz_f32: None,
      width,
      height,
      out_width,
      out_height,
      plan: None,
      #[cfg(any(
        feature = "yuv-planar",
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "xyz",
        feature = "bayer",
        feature = "mono"
      ))]
      rgb_stream: None,
      #[cfg(feature = "rgb")]
      rgb_stream_u16: None,
      #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
      rgb_stream_f32: None,
      #[cfg(feature = "yuv-planar")]
      luma_stream: None,
      #[cfg(any(
        feature = "yuv-planar",
        feature = "rgb",
        feature = "gbr",
        feature = "gray",
        feature = "xyz",
        feature = "bayer",
        feature = "mono"
      ))]
      resample_outputs: None,
      #[cfg(feature = "yuv-planar")]
      native: true,
      #[cfg(feature = "yuv-planar")]
      native_420: None,
      #[cfg(any(
        feature = "bayer",
        feature = "gbr",
        feature = "gray",
        feature = "mono",
        feature = "rgb",
        feature = "rgb-float",
        feature = "rgb-legacy",
        feature = "v210",
        feature = "xyz",
        feature = "y2xx",
        feature = "yuv-444-packed",
        feature = "yuv-packed",
        feature = "yuv-planar",
        feature = "yuv-semi-planar",
        feature = "yuva",
      ))]
      rgb_scratch: Vec::new(),
      #[cfg(feature = "rgb")]
      rgb_scratch_u16: Vec::new(),
      #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
      rgb_scratch_f32: Vec::new(),
      simd: true,
      // BT.709 by default — matches the implicit weights every
      // YUV→RGB→luma pipeline uses, and is the most common Bayer
      // CCM target. Per-format impls (`MixedSinker<Bayer>` etc.)
      // expose `with_luma_coefficients` for callers whose CCM
      // targets a different gamut.
      #[cfg(any(feature = "bayer", feature = "mono"))]
      luma_coefficients_q8: (54, 183, 19),
      _fmt: PhantomData,
      _resampler: PhantomData,
    }
  }

  /// Returns `true` iff the sinker will write 8‑bit RGB.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb(&self) -> bool {
    self.rgb.is_some()
  }

  /// Returns `true` iff the sinker will write `u16` RGB at the
  /// source's native bit depth. Only high‑bit‑depth source impls
  /// (currently [`Yuv420p10`](crate::source::Yuv420p10)) honor this
  /// buffer — attaching it on an 8‑bit source format is legal but
  /// no writes occur.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb_u16(&self) -> bool {
    self.rgb_u16.is_some()
  }

  /// Returns `true` iff the sinker will write `f32` RGB. Only
  /// float-source impls (currently [`Rgbf32`](crate::source::Rgbf32))
  /// honor this buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb_f32(&self) -> bool {
    self.rgb_f32.is_some()
  }

  /// Returns `true` iff the sinker will write `half::f16` RGB. Only
  /// half-float-source impls (currently [`Rgbf16`](crate::source::Rgbf16))
  /// honor this buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgb_f16(&self) -> bool {
    self.rgb_f16.is_some()
  }

  /// Returns `true` iff the sinker will write `f32` RGBA. Only
  /// float-planar-GBR source impls (`Gbrpf32` / `Gbrapf32` / `Gbrpf16` /
  /// `Gbrapf16`) honor this buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgba_f32(&self) -> bool {
    self.rgba_f32.is_some()
  }

  /// Returns `true` iff the sinker will write `half::f16` RGBA. Only
  /// float-planar-GBR source impls (`Gbrpf32` / `Gbrapf32` / `Gbrpf16` /
  /// `Gbrapf16`) honor this buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgba_f16(&self) -> bool {
    self.rgba_f16.is_some()
  }

  /// Returns `true` iff the sinker will write 8‑bit RGBA. The
  /// fourth byte per pixel is alpha — opaque (`0xFF`) by default
  /// when the source has no alpha plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgba(&self) -> bool {
    self.rgba.is_some()
  }

  /// Returns `true` iff the sinker will write `u16` RGBA at the
  /// source's native bit depth. The fourth `u16` per pixel is alpha
  /// — opaque (`(1 << BITS) - 1`) by default when the source has no
  /// alpha plane. Only high‑bit‑depth source impls honor this
  /// buffer.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_rgba_u16(&self) -> bool {
    self.rgba_u16.is_some()
  }

  /// Returns `true` iff the sinker will write luma.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_luma(&self) -> bool {
    self.luma.is_some()
  }

  /// Returns `true` iff the sinker will write native-depth `u16`
  /// luma. Only honored by per-format impls that wire the
  /// `with_luma_u16` accessor (Tier 4 source families).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_luma_u16(&self) -> bool {
    self.luma_u16.is_some()
  }

  /// Returns `true` iff the sinker will write `f32` luma.
  /// Only honored by `Grayf32` source impls.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_luma_f32(&self) -> bool {
    self.luma_f32.is_some()
  }

  /// Returns `true` iff the sinker will write HSV.
  ///
  /// Gated on the same 15-feature any as the `hsv` field — under
  /// `--features "alloc"` alone, no per-format `process` impl
  /// compiles, the field doesn't exist, and this getter is also gone.
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn produces_hsv(&self) -> bool {
    self.hsv.is_some()
  }

  /// Frame width in pixels. Declared at construction.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// Frame height in pixels. Declared at construction.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> usize {
    self.height
  }

  /// Output width in pixels — what output buffers validate against.
  /// Equals [`Self::width`] unless constructed via
  /// [`MixedSinker::with_resampler`] with a non-identity plan.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_width(&self) -> usize {
    self.out_width
  }

  /// Output height in pixels — what output buffers validate against.
  /// Equals [`Self::height`] unless constructed via
  /// [`MixedSinker::with_resampler`] with a non-identity plan.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_height(&self) -> usize {
    self.out_height
  }

  /// The resampling plan fixed at construction — `Some` only for
  /// sinkers built via [`MixedSinker::with_resampler`] with a
  /// non-identity strategy.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn resample_plan(&self) -> Option<&ResamplePlan> {
    self.plan.as_ref()
  }

  /// Capacity of the source-row staging scratch — a white-box probe
  /// for the resample ordering tests (a rejected row must not have
  /// grown the scratch). Gated on `std` like the tests that consume it,
  /// so it is not dead code in the alloc-only test build.
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "rgb",
      all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
    )
  ))]
  pub(crate) fn rgb_scratch_capacity(&self) -> usize {
    self.rgb_scratch.capacity()
  }

  /// Whether the high-bit packed-RGB `u16` area stream has been
  /// created — a white-box probe for the resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `std` like the tests that consume it.
  #[cfg(all(test, feature = "rgb", feature = "std"))]
  pub(crate) fn rgb_stream_u16_allocated(&self) -> bool {
    self.rgb_stream_u16.is_some()
  }

  /// Whether the packed-float-RGB `f32` area stream has been created —
  /// a white-box probe for the resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on the float family, the engine, and `std` like
  /// the tests that consume it.
  #[cfg(all(
    test,
    feature = "rgb-float",
    any(feature = "yuv-planar", feature = "rgb"),
    feature = "std"
  ))]
  pub(crate) fn rgb_stream_f32_allocated(&self) -> bool {
    self.rgb_stream_f32.is_some()
  }

  /// Returns `true` iff row primitives dispatch to their SIMD backend.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn simd(&self) -> bool {
    self.simd
  }

  /// Returns `true` iff resampled processing may take the native
  /// decimation tier. See [`Self::with_native`].
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn native(&self) -> bool {
    self.native
  }

  /// Toggles the native decimation tier in place. See
  /// [`Self::with_native`] for the consuming builder variant.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_native(&mut self, native: bool) -> &mut Self {
    self.native = native;
    self
  }

  /// Sets whether resampled processing may take the native decimation
  /// tier (bin native planes, convert once at output resolution).
  /// Defaults to `true`, mirroring [`Self::with_simd`].
  ///
  /// The tiers differ in color SEMANTICS, not just speed: native
  /// averages in the source (YUV) domain and converts once — the
  /// fused semantics video pipelines (libswscale-class) produce —
  /// while the row-stage tier converts every source pixel first and
  /// averages in RGB, matching `cv2.INTER_AREA` applied to decoded
  /// RGB. Luma is bit-identical either way (both tiers bin the same Y
  /// plane). In-gamut color differs only by per-pixel rounding;
  /// OUT-OF-GAMUT content (super-blacks/whites, illegal chroma
  /// excursions) diverges as far as the content sits outside the
  /// gamut — unbounded in principle, with measured examples of
  /// 34/255 on a mild extreme checkerboard and 117/255 on a crafted
  /// Bt2020 limited-range case (both pinned by regression). Pass
  /// `false` for strict RGB-domain `INTER_AREA` semantics at
  /// source-resolution conversion cost.
  #[cfg(feature = "yuv-planar")]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_native(mut self, native: bool) -> Self {
    self.set_native(native);
    self
  }

  /// Toggles the SIMD dispatch in place. See [`Self::with_simd`] for the
  /// consuming builder variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_simd(&mut self, simd: bool) -> &mut Self {
    self.simd = simd;
    self
  }

  /// Sets whether row primitives dispatch to their SIMD backend.
  /// Defaults to `true` — pass `false` to force the scalar reference
  /// path (intended for benchmarks, fuzzing, and differential
  /// testing). See [`Self::set_simd`] for the in‑place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_simd(mut self, simd: bool) -> Self {
    self.set_simd(simd);
    self
  }

  /// Full-frame slot count (`out_width x out_height x channels`) with
  /// overflow checking — **output** geometry, since this sizes the
  /// caller's output buffers (`out == source` under the identity
  /// plan). The result is the minimum required `buf.len()` for any
  /// `&[T]` buffer holding `channels` slots per pixel — bytes for
  /// `&[u8]`, `u16` elements for `&[u16]`, `f32` elements for `&[f32]`,
  /// `f16` elements for `&[half::f16]`. The function does NOT scale by
  /// element size; callers compare against `buf.len()` (which Rust
  /// reports in elements of the slice's element type).
  ///
  /// Returns `Err(GeometryOverflow)` if the product cannot fit in
  /// `usize` — only reachable on 32‑bit targets with extreme dimensions.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn frame_elems(&self, channels: usize) -> Result<usize, MixedSinkerError> {
    self
      .out_width
      .checked_mul(self.out_height)
      .and_then(|n| n.checked_mul(channels))
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        self.out_width,
        self.out_height,
        channels,
      )))
  }

  /// Full-frame element count (`out_width x out_height`) for a
  /// single-channel `&[T]` buffer, with overflow checking. Equivalent
  /// to [`frame_elems(1)`](Self::frame_elems) numerically, but the
  /// dedicated name documents "one slot per pixel" at the call site
  /// (e.g. luma planes) without the channels=1 magic number.
  ///
  /// Returns `Err(GeometryOverflow { channels: 1 })` on overflow.
  ///
  /// Consumed by every non-Bayer sinker family; Bayer is RGB-only and
  /// has no single-channel pixel-count sizing.
  #[cfg(any(
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn frame_pixels(&self) -> Result<usize, MixedSinkerError> {
    self
      .out_width
      .checked_mul(self.out_height)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        self.out_width,
        self.out_height,
        1,
      )))
  }
}

impl<'a, F: SourceFormat, R> MixedSinker<'a, F, R> {
  /// Creates an empty [`MixedSinker`] whose **output geometry** is
  /// decided by `resampler`: [`Resampler::plan`] runs once, here, and
  /// every buffer attached afterwards validates against the resulting
  /// output geometry. [`PixelSink::begin_frame`] keeps validating the
  /// walker against the `width x height` **source** geometry, so the
  /// existing frame-mismatch protection is unchanged.
  ///
  /// With [`NoopResampler`] this is equivalent to [`MixedSinker::new`]
  /// (identity plan, infallible in practice).
  ///
  /// Formats route non-identity plans as they wire into the streaming
  /// engine (currently [`Yuv420p`](crate::source::Yuv420p)); the
  /// remaining per-format [`PixelSink`] impls stay pinned to the
  /// default strategy, so a sinker built here for those formats
  /// validates buffers against its output geometry but cannot yet
  /// process frames (see the type-level docs).
  ///
  /// # Errors
  ///
  /// [`MixedSinkerError::Resample`] when the strategy rejects the
  /// requested output geometry — see
  /// [`ResampleError`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_resampler(width: usize, height: usize, resampler: R) -> Result<Self, MixedSinkerError>
  where
    R: Resampler,
  {
    let plan = resampler.plan(width, height)?;
    let (out_width, out_height) = match plan.as_ref() {
      Some(plan) => plan.out_dims(),
      None => (width, height),
    };
    let mut sink = Self::with_geometry(width, height, out_width, out_height);
    sink.plan = plan;
    Ok(sink)
  }

  /// Attaches a packed 24-bit RGB output buffer.
  /// Returns `Err(InsufficientRgbBuffer)` if
  /// `buf.len() < out_width x out_height x 3` (output geometry; equals
  /// `width x height x 3` under the default identity resampler), or
  /// `Err(GeometryOverflow)` on 32‑bit targets when the product
  /// overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgb(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb`](Self::with_rgb).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbBuffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb = Some(buf);
    Ok(self)
  }

  // NOTE: `with_rgb_f32` / `set_rgb_f32` and `with_luma_f32` /
  // `set_luma_f32` are **not** declared here. Same rationale as
  // `with_rgb_u16` below: only the float-output formats actually
  // write these buffers, so the setters live on format-specific
  // impl blocks (`Grayf32` writes both; `Rgbf32` and `Rgbf16` only
  // write `rgb_f32`). Attaching an f32 buffer to a sink whose
  // `process` doesn't write it would leave the caller buffer
  // silently stale — the format-specific scoping turns that into a
  // compile error.

  // NOTE: `with_rgb_u16` / `set_rgb_u16` are **not** declared here.
  // They live on a format‑specific impl block further down (currently
  // [`MixedSinker<Yuv420p10>`]) so the buffer can only be attached to
  // sink types whose `PixelSink` impl actually writes it. Attaching a
  // `u16` RGB buffer to a [`Yuv420p`] / [`Nv12`] / [`Nv21`] sink is a
  // compile error, not a silent stale‑state bug. Future high‑bit‑depth
  // markers (12‑bit, 14‑bit, P010) will add their own impl blocks.

  // NOTE: `with_rgba` / `set_rgba` are **not** declared here either —
  // same rationale as `with_rgb_u16` above. The Ship 8 RGBA path is
  // currently wired only on [`MixedSinker<Yuv420p>`]; attaching an
  // RGBA buffer to a sink whose `PixelSink::process` doesn't write
  // it would silently leave the caller buffer untouched while
  // `produces_rgba()` returned `true`. Each format that writes RGBA
  // gets its own format‑specific impl block exposing the accessors.
  // Future formats (NV12 / NV21 / Yuv422p / Yuv444p / P010 / etc.)
  // add their own impl blocks as RGBA support lands.

  // NOTE: `with_rgba_u16` / `set_rgba_u16` are **not** declared here
  // for the same reason — they live on the format‑specific impl
  // blocks for high‑bit‑depth sources that actually write
  // native‑depth RGBA.

  /// Attaches a single-plane luma output buffer.
  /// Returns `Err(InsufficientLumaBuffer)` if
  /// `buf.len() < out_width x out_height` (output geometry), or
  /// `Err(GeometryOverflow)` on 32‑bit overflow.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_luma(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_luma`](Self::with_luma).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaBuffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma = Some(buf);
    Ok(self)
  }

  /// Attaches three HSV output planes. Returns
  /// `Err(MixedSinkerError::InsufficientHsvPlane(e))` (inspect via
  /// `e.which()` / `e.expected()` / `e.actual()`) naming the first
  /// short plane, or `Err(MixedSinkerError::GeometryOverflow(_))` on
  /// 32-bit overflow.
  ///
  /// HSV is only meaningful when at least one source family is
  /// compiled, so this method is gated on the same 15-feature any as
  /// the per-format `process` impls that consume the `hsv` field.
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_hsv(
    mut self,
    h: &'a mut [u8],
    s: &'a mut [u8],
    v: &'a mut [u8],
  ) -> Result<Self, MixedSinkerError> {
    self.set_hsv(h, s, v)?;
    Ok(self)
  }

  /// In-place variant of [`with_hsv`](Self::with_hsv).
  #[cfg(any(
    feature = "bayer",
    feature = "gbr",
    feature = "gray",
    feature = "mono",
    feature = "rgb",
    feature = "rgb-float",
    feature = "rgb-legacy",
    feature = "v210",
    feature = "xyz",
    feature = "y2xx",
    feature = "yuv-444-packed",
    feature = "yuv-packed",
    feature = "yuv-planar",
    feature = "yuv-semi-planar",
    feature = "yuva",
  ))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_hsv(
    &mut self,
    h: &'a mut [u8],
    s: &'a mut [u8],
    v: &'a mut [u8],
  ) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(1)?;
    if h.len() < expected {
      return Err(MixedSinkerError::InsufficientHsvPlane(
        InsufficientHsvPlane::new(HsvPlane::H, expected, h.len()),
      ));
    }
    if s.len() < expected {
      return Err(MixedSinkerError::InsufficientHsvPlane(
        InsufficientHsvPlane::new(HsvPlane::S, expected, s.len()),
      ));
    }
    if v.len() < expected {
      return Err(MixedSinkerError::InsufficientHsvPlane(
        InsufficientHsvPlane::new(HsvPlane::V, expected, v.len()),
      ));
    }
    self.hsv = Some(HsvFrameMut::new(h, s, v));
    Ok(self)
  }
}

/// Returns `Ok(())` iff the walker's frame dimensions exactly match
/// the sinker's configured dimensions. Called from
/// [`PixelSink::begin_frame`] in every `MixedSinker<F>` impl.
///
/// The sinker's RGB / luma / HSV buffers were sized for
/// `configured_w x configured_h`. A shorter frame would silently
/// leave the bottom rows of those buffers stale from the previous
/// frame; a taller frame would overrun them. Either is a real
/// failure mode, but neither is a panic-worthy bug — the caller can
/// recover by rebuilding the sinker. Returning `Err` before any row
/// is processed guarantees no partial output.
///
/// Consumed by every per-format `MixedSinker<F>::process` impl.
/// Under `--features "alloc"` alone (no per-format family), no
/// `process` impl compiles and this helper would be flagged unused.
#[cfg(any(
  feature = "bayer",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn check_dimensions_match(
  configured_w: usize,
  configured_h: usize,
  frame_w: u32,
  frame_h: u32,
) -> Result<(), MixedSinkerError> {
  let fw = frame_w as usize;
  let fh = frame_h as usize;
  if fw != configured_w || fh != configured_h {
    return Err(MixedSinkerError::DimensionMismatch(DimensionMismatch::new(
      configured_w,
      configured_h,
      frame_w,
      frame_h,
    )));
  }
  Ok(())
}

/// Slice the RGBA row out of an attached RGBA plane buffer. Returns
/// `Err(GeometryOverflow)` if `one_plane_end x 4` wraps `usize` (only
/// reachable on 32-bit targets at extreme dimensions).
///
/// Centralises the duplicated overflow/bounds-check pattern that every
/// `MixedSinker<F>::process` impl runs in both the standalone-RGBA
/// branch and the Strategy-A expand branch.
///
/// Consumed by every non-Bayer sinker family (Bayer is RGB-only, no
/// RGBA path).
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgba_plane_row_slice(
  buf: &mut [u8],
  one_plane_start: usize,
  one_plane_end: usize,
  width: usize,
  height: usize,
) -> Result<&mut [u8], MixedSinkerError> {
  let end = one_plane_end
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width, height, 4,
    )))?;
  let start = one_plane_start * 4; // ≤ end, fits.
  Ok(&mut buf[start..end])
}

/// `u16` analogue of [`rgba_plane_row_slice`] — slices the RGBA row out
/// of an attached `u16` RGBA plane buffer. This helper indexes in `u16`
/// elements, not bytes: like the `u8` variant, RGBA rows use `x 4`
/// elements per pixel, so the overflow check is the same, but the byte
/// offsets differ because each element is 2 bytes. Used by the
/// high-bit-depth 4:2:0 sinkers that fan `u16` RGB out to `u16` RGBA.
///
/// Bayer is RGB-only and packed YUV 4:2:2 / 4:1:1 (`yuv-packed`) emits
/// u8 only; semi-planar 8-bit NV is also u8-only and never reaches a
/// u16 RGBA fan-out path, so this helper is unused under those
/// families.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgba_u16_plane_row_slice(
  buf: &mut [u16],
  one_plane_start: usize,
  one_plane_end: usize,
  width: usize,
  height: usize,
) -> Result<&mut [u16], MixedSinkerError> {
  let end = one_plane_end
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width, height, 4,
    )))?;
  let start = one_plane_start * 4; // ≤ end, fits.
  Ok(&mut buf[start..end])
}

/// Pick an RGB row buffer for the kernel to write into: caller's RGB
/// plane slice when attached, or the growing scratch buffer otherwise
/// (HSV-only callers don't allocate an RGB plane). Returns
/// `Err(GeometryOverflow)` if `width x 3` or `one_plane_end x 3` wraps
/// `usize` — see [`rgba_plane_row_slice`] for the rationale.
///
/// `rgb_scratch` is grown via `Vec::resize` only when too small; the
/// caller keeps the existing capacity across rows so steady-state
/// processing allocates zero times.
///
/// Consumed by per-format `process` impls that need a stable RGB row
/// buffer (either user-attached or scratch-backed). Under
/// `--features "alloc"` alone (no per-format family), no impl
/// compiles and this helper would be flagged unused.
#[cfg(any(
  feature = "bayer",
  feature = "gbr",
  feature = "gray",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgb_row_buf_or_scratch<'a>(
  rgb: Option<&'a mut [u8]>,
  rgb_scratch: &'a mut Vec<u8>,
  one_plane_start: usize,
  one_plane_end: usize,
  width: usize,
  height: usize,
) -> Result<&'a mut [u8], MixedSinkerError> {
  match rgb {
    Some(buf) => {
      let end = one_plane_end
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          width, height, 3,
        )))?;
      let start = one_plane_start * 3;
      Ok(&mut buf[start..end])
    }
    None => {
      let row_bytes = width
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          width, height, 3,
        )))?;
      if rgb_scratch.len() < row_bytes {
        rgb_scratch.resize(row_bytes, 0);
      }
      Ok(&mut rgb_scratch[..row_bytes])
    }
  }
}

/// Grows `rgb_scratch` to a **source-width** RGB row (`width * 3`
/// bytes) and returns the slice, following the planner's recoverable-
/// allocation contract (the exact reserve makes the resize incapable
/// of reallocating; refusal surfaces as `AllocationFailed` in the
/// preflight phase, not an abort in infallible growth).
///
/// The shared staging point for packed-RGB-canonical resampled
/// sources whose row must be channel-swapped or converted to RGB
/// before feeding the area stream. [`MixedSinker<Rgb24>`] skips it —
/// its source is already RGB and feeds the stream with zero copy.
#[cfg(any(
  feature = "yuv-planar",
  feature = "rgb",
  feature = "gbr",
  feature = "gray",
  feature = "xyz",
  feature = "bayer",
  feature = "mono"
))]
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn source_rgb_scratch<'s>(
  rgb_scratch: &'s mut Vec<u8>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u8], MixedSinkerError> {
  let row_bytes = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      3,
    )))?;
  if rgb_scratch.len() < row_bytes {
    rgb_scratch
      .try_reserve_exact(row_bytes - rgb_scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    rgb_scratch.resize(row_bytes, 0);
  }
  Ok(&mut rgb_scratch[..row_bytes])
}

/// Freezes the output configuration for a resampled packed-RGB frame
/// and reports whether any output is attached. Run before the
/// source-row conversion and the stream so a sink with no attached
/// outputs stays the documented legal no-op (it neither allocates nor
/// enforces sequencing) while a mid-frame output-set change is still
/// caught. Mirrors the YUV resample path's freeze-then-conditional
/// ordering.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[cfg_attr(not(any(feature = "rgb", feature = "gbr")), allow(dead_code))]
pub(super) fn packed_rgb_resample_preflight(
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
) -> Result<bool, MixedSinkerError> {
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    &None,
    &None,
    &None,
    hsv,
    idx,
  )?;
  Ok(rgb.is_some() || rgba.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some())
}

/// Fused downscale for [`MixedSinker<Rgb24, R>`]: the packed source
/// row feeds the 3-channel area stream with no conversion step; RGB
/// copies, and luma / luma_u16 / HSV / RGBA derive from each finalized
/// output row.
///
/// `src_rgb` is the **source-width** canonical RGB row — `Rgb24` hands
/// in its packed source directly (zero copy); channel-swapped or
/// converting formats (the `Bgr24` / padding-byte family, planar
/// `Gbrp`) stage their row into a source-width scratch first, so this
/// one tail serves every packed-RGB-canonical source. The caller runs
/// [`packed_rgb_resample_preflight`] first and skips the rest when no
/// output is attached.
///
/// Lazily creates the 3-channel area stream and checks strict row
/// sequencing — run **before** a converting format stages its source
/// row, so an out-of-sequence row is rejected without the scratch
/// allocation/conversion (matching the `Rgb24` / YUV ordering).
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[cfg_attr(not(any(feature = "rgb", feature = "gbr")), allow(dead_code))]
pub(super) fn packed_rgb_resample_stream<'s>(
  rgb_stream: &'s mut Option<crate::resample::AreaStream<u8>>,
  plan: &ResamplePlan,
  idx: usize,
) -> Result<&'s mut crate::resample::AreaStream<u8>, MixedSinkerError> {
  // Sequence-check before allocating: a fresh stream expects row 0, so
  // an out-of-sequence first row is rejected without creating the
  // output-width buffers — keeping freeze, then sequence-check, then
  // stage, and never letting AllocationFailed mask OutOfSequenceRow.
  let expected = rgb_stream.as_ref().map_or(0, |stream| stream.next_y());
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  let stream = match rgb_stream {
    Some(stream) => stream,
    None => rgb_stream.insert(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      3,
    )?),
  };
  Ok(stream)
}

/// Feeds the prepared source-width canonical RGB row into the (already
/// sequence-checked) stream and derives every attached output (rgb,
/// rgba, luma, luma_u16, hsv) from each finalized output row.
#[cfg(any(feature = "rgb", feature = "gbr"))]
#[cfg_attr(not(any(feature = "rgb", feature = "gbr")), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_resample_emit(
  stream: &mut crate::resample::AreaStream<u8>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_rgb: &[u8],
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  stream.feed_row(idx, src_rgb, use_simd, |oy, out_row| {
    if let Some(buf) = rgb.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_row);
    }
    if let Some(buf) = luma.as_deref_mut() {
      crate::row::rgb_to_luma_row(
        out_row,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        use_simd,
      );
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::rgb_to_luma_u16_row(
        out_row,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        use_simd,
      );
    }
    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      crate::row::rgb_to_hsv_row(
        out_row,
        &mut h[oy * ow..(oy + 1) * ow],
        &mut s[oy * ow..(oy + 1) * ow],
        &mut v[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = rgba.as_deref_mut() {
      crate::row::expand_rgb_to_rgba_row(out_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
    }
  })?;

  Ok(())
}

/// Source-width `u16` RGB staging for high-bit packed-RGB resampling:
/// the wire row converts here before feeding [`AreaStream<u16>`]. Grows
/// `scratch` to `3 * width` `u16` under the planner's
/// recoverable-allocation contract. Mirrors [`source_rgb_scratch`] for
/// the 16-bit element path.
#[cfg(feature = "rgb")]
pub(super) fn source_rgb_u16_scratch<'s>(
  scratch: &'s mut Vec<u16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u16], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0);
  }
  Ok(&mut scratch[..row])
}

/// Freezes the output configuration for a resampled high-bit
/// packed-RGB frame — the full u8 **and** u16 output set — and reports
/// whether any output is attached. Mirrors
/// [`packed_rgb_resample_preflight`], extended with the native-depth
/// `rgb_u16` / `rgba_u16` / `luma_u16` channels.
#[cfg(feature = "rgb")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_u16_resample_preflight(
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  rgb_u16: &Option<&mut [u16]>,
  rgba_u16: &Option<&mut [u16]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
) -> Result<bool, MixedSinkerError> {
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    hsv,
    idx,
  )?;
  Ok(
    rgb.is_some()
      || rgba.is_some()
      || luma.is_some()
      || rgb_u16.is_some()
      || rgba_u16.is_some()
      || luma_u16.is_some()
      || hsv.is_some(),
  )
}

/// Lazily creates the 3-channel `u16` area stream and checks strict row
/// sequencing — run before the source conversion so an out-of-sequence
/// row is rejected without the staging work. Mirrors
/// [`packed_rgb_resample_stream`] for the 16-bit element path.
#[cfg(feature = "rgb")]
pub(super) fn packed_rgb_u16_resample_stream<'s>(
  rgb_stream_u16: &'s mut Option<crate::resample::AreaStream<u16>>,
  plan: &ResamplePlan,
  idx: usize,
) -> Result<&'s mut crate::resample::AreaStream<u16>, MixedSinkerError> {
  // Sequence-check before allocating (see packed_rgb_resample_stream):
  // an out-of-sequence first row is rejected without creating the u16
  // output-width buffers, so AllocationFailed never masks
  // OutOfSequenceRow.
  let expected = rgb_stream_u16.as_ref().map_or(0, |stream| stream.next_y());
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  let stream = match rgb_stream_u16 {
    Some(stream) => stream,
    None => rgb_stream_u16.insert(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      3,
    )?),
  };
  Ok(stream)
}

/// Feeds the prepared source-width `u16` RGB row into the (already
/// sequence-checked) stream and derives every attached output from each
/// finalized output row. Binning runs at native 16-bit depth; the
/// `rgb_u16` / `rgba_u16` outputs copy it directly, while the u8 and
/// `luma_u16` outputs derive from a single `>> 8` narrowing — the same
/// source-of-truth ordering the direct Rgb48 path uses (luma /
/// luma_u16 / hsv all read the narrowed u8 RGB). `narrow_scratch` is
/// sized to the out-width u8 RGB row only when one of those narrowed
/// outputs is attached, so a native-u16-only sink neither grows it nor
/// risks its allocation failure.
#[cfg(feature = "rgb")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_u16_resample_emit(
  stream: &mut crate::resample::AreaStream<u16>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_u16: &[u16],
  narrow_scratch: &mut Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  // The u8 / luma_u16 outputs derive from a `>> 8` narrowing of the
  // binned row; a native-u16-only sink (only rgb_u16 / rgba_u16) never
  // touches it, so the out-width u8 scratch is sized — and its
  // allocation failure risked — only when one of those outputs is
  // attached. The predicate gates both the sizing here and the use in
  // the closure, so they cannot drift.
  let need_narrow =
    rgb.is_some() || rgba.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  stream.feed_row(idx, src_u16, use_simd, |oy, binned| {
    // Native-depth u16 outputs copy the binned row directly.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(binned);
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      crate::row::expand_rgb_u16_to_rgba_u16_row::<16>(
        binned,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
      );
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      for (d, &s) in nrow.iter_mut().zip(binned.iter()) {
        *d = (s >> 8) as u8;
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::expand_rgb_to_rgba_row(nrow, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Source-width `f32` RGB staging for packed-float-RGB resampling: the
/// wire row converts here (host-native f32, lossless) before feeding
/// [`AreaStream<f32>`]. Grows `scratch` to `3 * width` `f32` under the
/// planner's recoverable-allocation contract. Mirrors
/// [`source_rgb_u16_scratch`] for the float element path.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
pub(super) fn source_rgb_f32_scratch<'s>(
  scratch: &'s mut Vec<f32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [f32], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row, 0.0);
  }
  Ok(&mut scratch[..row])
}

/// Freezes the output configuration for a resampled packed-float-RGB
/// frame — the full u8 / u16 / `rgb_f32` output set — and reports
/// whether any output is attached. Mirrors
/// [`packed_rgb_u16_resample_preflight`], extended with the lossless
/// `rgb_f32` channel.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_f32_resample_preflight(
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  rgb_u16: &Option<&mut [u16]>,
  rgba_u16: &Option<&mut [u16]>,
  luma_u16: &Option<&mut [u16]>,
  rgb_f32: &Option<&mut [f32]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
) -> Result<bool, MixedSinkerError> {
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    rgb_f32,
    hsv,
    idx,
  )?;
  Ok(
    rgb.is_some()
      || rgba.is_some()
      || luma.is_some()
      || rgb_u16.is_some()
      || rgba_u16.is_some()
      || luma_u16.is_some()
      || rgb_f32.is_some()
      || hsv.is_some(),
  )
}

/// Lazily creates the 3-channel `f32` area stream and checks strict row
/// sequencing — run before the source conversion so an out-of-sequence
/// row is rejected without the staging work. Mirrors
/// [`packed_rgb_u16_resample_stream`] for the float element path.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
pub(super) fn packed_rgb_f32_resample_stream<'s>(
  rgb_stream_f32: &'s mut Option<crate::resample::AreaStream<f32>>,
  plan: &ResamplePlan,
  idx: usize,
) -> Result<&'s mut crate::resample::AreaStream<f32>, MixedSinkerError> {
  // Sequence-check before allocating (see packed_rgb_u16_resample_stream):
  // an out-of-sequence first row is rejected without creating the f32
  // output-width buffers, so AllocationFailed never masks
  // OutOfSequenceRow.
  let expected = rgb_stream_f32.as_ref().map_or(0, |stream| stream.next_y());
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  let stream = match rgb_stream_f32 {
    Some(stream) => stream,
    None => rgb_stream_f32.insert(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      3,
    )?),
  };
  Ok(stream)
}

/// Feeds the prepared source-width `f32` RGB row into the (already
/// sequence-checked) stream and derives every attached output from each
/// finalized output row. Binning runs in float; the `rgb_f32` output
/// copies it losslessly, and every integer output mirrors the direct
/// [`Rgbf32`](crate::source::Rgbf32) path's clamp+scale kernels run over
/// the binned row. The binned row is already host-native f32 (the wire
/// converted via `rgbf32_to_rgb_f32_row::<BE>` before feeding), so the
/// emit kernels use `::<false>` — no further byte swap. `narrow_scratch`
/// is sized to the out-width u8 RGB row only when one of the outputs
/// that stage through it (`rgb` / `luma` / `luma_u16` / `hsv`) is
/// attached, so an f32-only or native-u16-only sink neither grows it nor
/// risks its allocation failure.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_f32_resample_emit(
  stream: &mut crate::resample::AreaStream<f32>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_f32: &[f32],
  narrow_scratch: &mut Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  // The u8 RGB / luma / luma_u16 / hsv outputs stage through a u8 RGB
  // narrowing of the binned float row (exactly the direct path's
  // `rgbf32_to_rgb_row` scratch); an f32-only or native-u16-only sink
  // never touches it, so the out-width u8 scratch is sized — and its
  // allocation failure risked — only when one of those outputs is
  // attached. The predicate gates both the sizing here and the use in
  // the closure, so they cannot drift. `rgba` (u8) derives directly
  // from the float source via `rgbf32_to_rgba_row`, mirroring the
  // direct path, so it does not need the narrow row.
  let need_narrow = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  stream.feed_row(idx, src_f32, use_simd, |oy, binned| {
    // Lossless float pass-through — copy the binned row verbatim
    // (mirrors the direct path's `rgbf32_to_rgb_f32_row`; the binned
    // row is already host-native, so this is a plain copy).
    if let Some(buf) = rgb_f32.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(binned);
    }
    // u16 outputs — direct float→u16 clamp+scale (no narrowing stage),
    // exactly as the direct Rgbf32 path derives them from the source.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      crate::row::rgbf32_to_rgb_u16_row::<false>(
        binned,
        &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      crate::row::rgbf32_to_rgba_u16_row::<false>(
        binned,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
        use_simd,
      );
    }
    // u8 RGBA — direct float→u8 clamp+scale, alpha 0xFF (the direct
    // path emits RGBA straight from the float source, not via an
    // expand of the u8 RGB row).
    if let Some(buf) = rgba.as_deref_mut() {
      crate::row::rgbf32_to_rgba_row::<false>(
        binned,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
        use_simd,
      );
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      // Stage the u8 RGB row once via the direct path's float→u8
      // clamp+scale; rgb / luma / luma_u16 / hsv all read it, matching
      // the direct Rgbf32 source-of-truth ordering exactly.
      crate::row::rgbf32_to_rgb_row::<false>(binned, nrow, ow, use_simd);
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::rgb_to_hsv_row(
          nrow,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

/// Configurable-coefficients luma derivation from packed
/// `R, G, B` u8 row.
///
/// Q8 fixed-point: `Y ≈ (cr·R + cg·G + cb·B + 128) >> 8`, where
/// `(cr, cg, cb)` is the caller's [`LumaCoefficients`] resolved
/// via [`LumaCoefficients::to_q8`]. The presets all sum to `256`
/// so the divisor is implicit in the `>> 8`. `rgb` carries
/// `3 * luma.len()` packed bytes; the loop writes one luma
/// sample per pixel.
///
/// Used by Bayer / Bayer16 / Pal8 [`MixedSinker`] paths whose source
/// has no native luma plane to memcpy from. YUV source impls take
/// their luma directly off the Y plane and don't go through this
/// helper, so they don't need a configurable coefficient set —
/// the source's `ColorMatrix` already fixed it at encode time.
#[cfg(any(feature = "bayer", feature = "mono"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgb_row_to_luma_row(rgb: &[u8], luma: &mut [u8], coeffs_q8: (u32, u32, u32)) {
  // Caller's contract: `rgb` packs `3 * luma.len()` bytes. The
  // current callers (`MixedSinker<Bayer>` and
  // `MixedSinker<Bayer16<BITS>>`) both slice their `luma` and
  // `rgb_row` from the same `width`, so the relationship holds
  // structurally — but the `debug_assert` makes that obvious to
  // any future caller and turns silent OOB indexing into a clear
  // failure under tests.
  //
  // `checked_mul` instead of `3 * luma.len()` because, while the
  // existing `frame_elems` validation in caller paths makes the
  // product fit, a future caller passing a raw slice with no such
  // upstream check could trigger a `usize` overflow inside the
  // assert message itself (panic before the assertion runs).
  // Failing the assert on overflow yields a clean diagnostic.
  debug_assert!(
    luma
      .len()
      .checked_mul(3)
      .is_some_and(|need| rgb.len() >= need),
    "rgb_row_to_luma_row: rgb.len()={} but need {} (= 3 x luma.len()={})",
    rgb.len(),
    luma.len().saturating_mul(3),
    luma.len(),
  );
  let (cr, cg, cb) = coeffs_q8;
  for (i, d) in luma.iter_mut().enumerate() {
    let r = rgb[3 * i] as u32;
    let g = rgb[3 * i + 1] as u32;
    let b = rgb[3 * i + 2] as u32;
    *d = ((cr * r + cg * g + cb * b + 128) >> 8).min(255) as u8;
  }
}

/// Same as [`rgb_row_to_luma_row`] but widens the luma byte to `u16` via
/// `(y << 8) | y` (`0 → 0x0000`, `255 → 0xFFFF`).
///
/// Used by format sinker paths that expose a `with_luma_u16` output channel
/// (e.g. `MixedSinker<Pal8>`).
#[cfg(feature = "mono")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn rgb_row_to_luma_u16_row(
  rgb: &[u8],
  luma_u16: &mut [u16],
  coeffs_q8: (u32, u32, u32),
) {
  debug_assert!(
    luma_u16
      .len()
      .checked_mul(3)
      .is_some_and(|need| rgb.len() >= need),
    "rgb_row_to_luma_u16_row: rgb.len()={} but need {} (= 3 x luma_u16.len()={})",
    rgb.len(),
    luma_u16.len().saturating_mul(3),
    luma_u16.len(),
  );
  let (cr, cg, cb) = coeffs_q8;
  for (i, dst) in luma_u16.iter_mut().enumerate() {
    let r = rgb[3 * i] as u32;
    let g = rgb[3 * i + 1] as u32;
    let b = rgb[3 * i + 2] as u32;
    let y = ((cr * r + cg * g + cb * b + 128) >> 8).min(255) as u16;
    *dst = (y << 8) | y;
  }
}

// ---- Format-specific impl blocks (split out of mod.rs) ------------------
//
// Each child module hosts the `MixedSinker<'_, F>` impl blocks for a
// related family of source formats. mod.rs keeps only the shared
// prelude (errors, types, struct, generic impls, helpers) and the
// `LumaCoefficients` API. Per-format `with_rgba` / `set_rgba` builders
// and `PixelSink` impls live in the child modules below.

#[cfg(feature = "yuv-444-packed")]
mod ayuv64;
#[cfg(feature = "bayer")]
mod bayer;
#[cfg(feature = "gray")]
mod gray;
#[cfg(feature = "rgb-legacy")]
mod legacy_rgb;
#[cfg(feature = "mono")]
mod mono1bit;
#[cfg(feature = "rgb")]
mod packed_rgb_10bit;
#[cfg(feature = "rgb")]
mod packed_rgb_16bit;
#[cfg(feature = "rgb")]
mod packed_rgb_8bit;
#[cfg(feature = "rgb-float")]
mod packed_rgb_f16;
#[cfg(feature = "rgb-float")]
mod packed_rgb_float;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_4_1_1;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_8bit;
#[cfg(feature = "mono")]
mod pal8;
#[cfg(feature = "yuv-planar")]
mod planar_8bit;
#[cfg(feature = "gbr")]
mod planar_gbr_8bit;
#[cfg(feature = "gbr")]
mod planar_gbr_f16;
#[cfg(feature = "gbr")]
mod planar_gbr_float;
#[cfg(feature = "gbr")]
mod planar_gbr_high_bit;
#[cfg(feature = "yuv-semi-planar")]
mod semi_planar_8bit;
#[cfg(feature = "yuv-planar")]
mod subsampled_4_2_0_high_bit;
#[cfg(feature = "yuv-planar")]
mod subsampled_4_2_2_high_bit;
#[cfg(feature = "yuv-planar")]
mod subsampled_4_4_4_high_bit;
#[cfg(feature = "v210")]
mod v210;
#[cfg(feature = "yuv-444-packed")]
mod v30x;
#[cfg(feature = "yuv-444-packed")]
mod v410;
#[cfg(feature = "yuv-444-packed")]
mod vuya;
#[cfg(feature = "yuv-444-packed")]
mod vuyx;
#[cfg(feature = "yuv-444-packed")]
mod xv36;
#[cfg(feature = "xyz")]
mod xyz12;
#[cfg(feature = "y2xx")]
mod y210;
#[cfg(feature = "y2xx")]
mod y212;
#[cfg(feature = "y2xx")]
mod y216;
#[cfg(feature = "yuva")]
mod yuva_4_2_0;
#[cfg(feature = "yuva")]
mod yuva_4_2_2;
#[cfg(feature = "yuva")]
mod yuva_4_4_4;

#[cfg(all(test, feature = "std"))]
mod tests;

#[cfg(all(test, feature = "std"))]
mod api_smoke_tests {
  use super::*;

  #[cfg(feature = "v210")]
  #[test]
  fn mixed_sinker_default_does_not_produce_luma_u16() {
    // Use the currently available V210 source format marker for this smoke test.
    let sink: MixedSinker<'_, crate::source::V210> = MixedSinker::new(6, 1);
    assert!(!sink.produces_luma_u16());
  }

  #[test]
  fn luma_u16_buffer_too_short_error_displays() {
    let err = MixedSinkerError::InsufficientLumaU16Buffer(InsufficientBuffer::new(100, 50));
    let msg = format!("{err}");
    assert!(msg.contains("100"));
    assert!(msg.contains("50"));
  }
}
