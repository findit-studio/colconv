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
/// luma, luma_u16, rgb, rgba, rgb_u16, rgba_u16, the `rgb_f32` /
/// `rgba_f32` float-RGB(A) outputs, the `xyz_f32` linear-XYZ output,
/// the `rgb_f16` / `rgba_f16` half-float outputs, the `luma_f32`
/// float-luma output, and the three HSV planes. Equality is the
/// per-frame immutability check — in safe code a mid-frame `set_*`
/// necessarily supplies a different borrow, so an identity change is
/// exactly a reattachment. The `*_u16` / `rgb_f32` / `rgba_f32` /
/// `xyz_f32` / `*_f16` / `luma_f32` slots are `(0, 0)` for every format
/// that attaches no such output, so adding them leaves those formats'
/// snapshots unchanged.
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
#[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FrozenOutputs {
  idents: [(usize, usize); 15],
}

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
    rgba_f32: Option<&[f32]>,
    xyz_f32: Option<&[f32]>,
    rgb_f16: Option<&[half::f16]>,
    rgba_f16: Option<&[half::f16]>,
    hsv: Option<(&[u8], &[u8], &[u8])>,
    luma_f32: Option<&[f32]>,
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
        Self::ident(rgba_f32),
        Self::ident(xyz_f32),
        Self::ident(rgb_f16),
        Self::ident(rgba_f16),
        h,
        s,
        v,
        Self::ident(luma_f32),
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
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
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
  rgba_f32: &Option<&mut [f32]>,
  xyz_f32: &Option<&mut [f32]>,
  rgb_f16: &Option<&mut [half::f16]>,
  rgba_f16: &Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_f32: &Option<&mut [f32]>,
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
    rgba_f32.as_deref(),
    xyz_f32.as_deref(),
    rgb_f16.as_deref(),
    rgba_f16.as_deref(),
    hsv.as_mut().map(|f| {
      let (h, s, v) = f.hsv();
      (&h[..], &s[..], &v[..])
    }),
    luma_f32.as_deref(),
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

/// How a packed-alpha source's color channels relate to its alpha when
/// the frame is **area-resampled** — the only place the distinction is
/// observable, because area-binning averages color and alpha together.
///
/// In [`Self::Straight`] (a.k.a. *unassociated* / *non-premultiplied*)
/// alpha, the RGB triple stores the surface's own color and α is an
/// independent coverage term; averaging the channels independently is
/// correct. In [`Self::Premultiplied`] (a.k.a. *associated*) alpha, RGB
/// has already been multiplied by α, so a correct area-average must bin
/// the premultiplied channels and un-premultiply afterwards — averaging
/// straight RGB of a premultiplied source would let fully-transparent
/// pixels (whose stored RGB is arbitrary) bleed into the result.
///
/// Every packed-RGBA source format colconv ships today is straight (see
/// [`DefaultAlphaMode`]); the mode only matters on the resample path and
/// is a no-op for the direct (identity-plan) conversions, which copy
/// alpha through untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, IsVariant)]
pub enum AlphaMode {
  /// Unassociated alpha: RGB is the surface color, α an independent
  /// coverage term. Channels area-average independently.
  #[default]
  Straight,
  /// Associated alpha: RGB is already premultiplied by α. The resample
  /// path bins the premultiplied channels and un-premultiplies per
  /// finalized output row.
  Premultiplied,
}

impl AlphaMode {
  /// Returns the lowercase string name of the mode (`"straight"` /
  /// `"premultiplied"`), matching the variant's conventional spelling.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Straight => "straight",
      Self::Premultiplied => "premultiplied",
    }
  }
}

/// Per-format default [`AlphaMode`], consulted by [`MixedSinker::new`]
/// to seed the sink's alpha mode before any
/// [`MixedSinker::with_alpha_mode`] override.
///
/// The blanket impl below makes every [`SourceFormat`] default to
/// [`AlphaMode::Straight`] — true of every packed-RGBA source colconv
/// ships today (`Rgba` / `Bgra` / `Argb` / `Abgr` / `Rgba64` /
/// `Bgra64`). A future source format whose wire alpha is associated
/// would carry its premultiplied default here (replacing the blanket
/// with per-format impls), so callers get correct area-resampling
/// without having to pass [`MixedSinker::with_alpha_mode`] by hand.
pub trait DefaultAlphaMode: SourceFormat {
  /// The alpha interpretation a freshly built [`MixedSinker`] over this
  /// format starts in.
  const DEFAULT_ALPHA_MODE: AlphaMode = AlphaMode::Straight;
}

impl<F: SourceFormat> DefaultAlphaMode for F {}

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
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  ))]
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  rgb_stream: Option<crate::resample::AreaStream<u8>>,
  /// Row-stage area stream for high-bit packed-RGB sources (`u16`
  /// elements binned at native depth). Lazily created in `process`,
  /// reset in `begin_frame`. Gated to `rgb` (high-bit packed RGB),
  /// `gbr` (high-bit planar GBR scatters into the same u16 tail), and
  /// `yuv-444-packed` / `y2xx` / `v210` (the high-bit packed YUV color
  /// groups bin their converted native-u16 RGB row here); widens as
  /// high-bit families wire in.
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210"
  ))]
  rgb_stream_u16: Option<crate::resample::AreaStream<u16>>,
  /// Row-stage **4-channel** area stream for packed straight/premult
  /// RGBA sources (`Rgba` / `Bgra` / `Argb` / `Abgr`). Bins the staged
  /// canonical `R, G, B, A` row so resampled alpha is a real area mean
  /// (not forced opaque) and — under [`AlphaMode::Premultiplied`] —
  /// color is binned premultiplied. Lazily created in `process`, reset
  /// in `begin_frame`. Gated to `rgb`; the 3-channel [`Self::rgb_stream`]
  /// still serves the rgb-only straight path with no regression.
  #[cfg(feature = "rgb")]
  rgba_stream: Option<crate::resample::AreaStream<u8>>,
  /// Row-stage **4-channel** `u16` area stream for the high-bit packed
  /// RGBA sources (`Rgba64` / `Bgra64`). Bins the staged canonical
  /// `R, G, B, A` row at native 16-bit depth; the native-depth
  /// `rgba_u16` output copies it, the u8 / narrowed outputs derive via
  /// `>> 8`. Lazily created in `process`, reset in `begin_frame`. Gated
  /// to `rgb`.
  #[cfg(feature = "rgb")]
  rgba_stream_u16: Option<crate::resample::AreaStream<u16>>,
  /// Alpha mode frozen at a resampled frame's first row. A mid-frame
  /// [`Self::set_alpha_mode`] change is then rejected before any stream
  /// is fed, since a stream mixing straight and premultiplied rows would
  /// match neither all-straight nor all-premultiplied output. `None`
  /// between frames; re-armed on each frame's first resampled row (so a
  /// stale value never leaks across frames). Gated to `rgb`.
  #[cfg(feature = "rgb")]
  frozen_alpha_mode: Option<AlphaMode>,
  /// Row-stage area stream for packed-float-RGB sources (`f32`
  /// elements binned in float). Lazily created in `process`, reset in
  /// `begin_frame`. The `rgb-float` family needs the engine fenced in
  /// (`AreaStream` is gated to `yuv-planar` / `rgb`, which `rgb-float`
  /// does not imply); the `gbr` family already pulls the engine via the
  /// #146 cascade, so its float-GBR scatter reaches this same tail with
  /// no separate fence.
  #[cfg(any(
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
    feature = "gbr"
  ))]
  rgb_stream_f32: Option<crate::resample::AreaStream<f32>>,
  /// Row-stage area stream for the packed-CIE-XYZ-12-bit source
  /// ([`Xyz12`](crate::source::Xyz12)). The wire row converts to
  /// **linear XYZ** `f32` (post-OETF, pre-matrix) and bins in float so
  /// the area mean is taken in linear light — the matrix and gamma are
  /// applied per finalized output row, after the bin. Gated to `xyz`;
  /// the engine is already pulled in for `xyz` by the shared
  /// [`AreaStream`] gate (the `#145`/`#146` cascade widened it to
  /// `xyz`), so no separate engine feature is required.
  #[cfg(feature = "xyz")]
  xyz_stream_f32: Option<crate::resample::AreaStream<f32>>,
  /// Row-stage area stream for single-plane luma binning. Used by the
  /// planar YUV family (Y-plane luma), the [`Gray8`](crate::source::Gray8)
  /// source (Gray *is* a luma plane), and `mono` (bin the expanded
  /// 0/255 luma plane). Lazily created in `process`, reset in
  /// `begin_frame`. Gated like the engine; widens as families wire in.
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
  #[cfg_attr(
    not(any(feature = "yuv-planar", feature = "gray", feature = "mono")),
    allow(dead_code)
  )]
  luma_stream: Option<crate::resample::AreaStream<u8>>,
  /// Row-stage area stream for single-plane **u16** luma binning. Used
  /// by the [`Gray16`](crate::source::Gray16) source, whose luma plane
  /// is a native `u16` and so bins at u16 precision (the `u8`
  /// [`Self::luma_stream`] would lose the low byte), and by the high-bit
  /// packed YUV families (`yuv-444-packed` / `y2xx` / `v210`), which bin
  /// their native Y here so resampled luma stays the area-downscaled Y at
  /// native depth. Lazily created in `process`, reset in `begin_frame`.
  /// Gated to `gray` / `yuv-444-packed` / `y2xx` / `v210`; widens as u16
  /// luma families wire in.
  #[cfg(any(
    feature = "gray",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210"
  ))]
  luma_stream_u16: Option<crate::resample::AreaStream<u16>>,
  /// Row-stage area stream for single-plane **f32** luma binning. Used
  /// by the [`Grayf32`](crate::source::Grayf32) source, whose luma plane
  /// is a native `f32` and so bins at f32 precision (the `u8` / `u16`
  /// luma streams would quantize every sample before averaging). Lazily
  /// created in `process`, reset in `begin_frame`. Gated to `gray`;
  /// widens as f32 luma families wire in.
  #[cfg(feature = "gray")]
  luma_stream_f32: Option<crate::resample::AreaStream<f32>>,
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
    feature = "mono",
    feature = "yuv-semi-planar",
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210",
    feature = "rgb-legacy"
  ))]
  #[cfg_attr(not(any(feature = "yuv-planar", feature = "rgb")), allow(dead_code))]
  resample_outputs: Option<FrozenOutputs>,
  /// Whether resampled processing may take the native decimation tier
  /// (bin native planes, convert once at output resolution). Defaults
  /// to `true`; benchmarks and differential tests flip it to force the
  /// row-stage tier — the [`Self::with_simd`] pattern. The native tier
  /// exists only for the 8-bit planar 4:2:0
  /// ([`Yuv420p`](crate::source::Yuv420p)); every other routed family
  /// (including the semi-planar formats) always takes the row-stage
  /// tier and ignores this flag.
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
  /// Source-width `u8` luma staging for the **packed YUV 4:2:2**
  /// resample path: the interleaved Y bytes are de-interleaved here (via
  /// the format's own `*_to_luma_row` kernel — the exact Y→luma
  /// derivation the direct path uses) before feeding the single-channel
  /// [`Self::luma_stream`]. The packed colour stream simultaneously
  /// stages its RGB row in [`Self::rgb_scratch`], so the Y row needs its
  /// own buffer rather than sharing that scratch. Lazily grown to
  /// `width` `u8`; empty otherwise. Gated to `yuv-packed`.
  #[cfg(feature = "yuv-packed")]
  luma_scratch: Vec<u8>,
  /// Source-width **native-channel** `u8` staging for the legacy 16-bit
  /// packed-RGB ([`Rgb565`](crate::source::Rgb565) and family) resample
  /// path: each packed source row is unpacked to its 3 **native** R/G/B
  /// channels (5/6/5, 5/5/5 or 4/4/4 values, each `<= 63`, NOT expanded
  /// to 8-bit) here before feeding the shared u8 [`Self::rgb_stream`], so
  /// the area mean is taken at native depth. Lazily grown to `3 * width`
  /// `u8`; empty otherwise. Gated to `rgb-legacy`.
  #[cfg(feature = "rgb-legacy")]
  legacy_rgb_native_scratch: Vec<u8>,
  /// Out-width **re-packed source-format** `u8` staging for the legacy
  /// 16-bit packed-RGB resample tail. Per finalized output row the binned
  /// native R/G/B channels are re-packed back into the source's packed
  /// `u16` word (LE bytes, `2 * out_width`) here, then the **exact**
  /// direct `*_to_*` kernels run over it — so every output is
  /// byte-identical to a direct conversion of the area-downscaled
  /// source-format frame. The integer twin of the `gbr` family's
  /// [`Self::rgb_plane_scratch_f32`] de-interleave staging. Lazily grown
  /// to `2 * out_width` `u8`; empty for a no-output sink. Gated to
  /// `rgb-legacy`.
  #[cfg(feature = "rgb-legacy")]
  legacy_rgb_packed_scratch: Vec<u8>,
  /// Source-width `u16` RGB staging for high-bit packed-RGB resampling:
  /// the wire row converts here before feeding [`Self::rgb_stream_u16`].
  /// Lazily grown to `3 * width` `u16`; empty otherwise. Gated to `rgb`
  /// (high-bit packed RGB), `gbr` (high-bit planar GBR scatters its
  /// G/B/R planes here before the same u16 tail), and the high-bit packed
  /// YUV color groups (`yuv-444-packed` / `y2xx` / `v210`) stage their
  /// native-u16 RGB row here.
  #[cfg(any(
    feature = "rgb",
    feature = "gbr",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210"
  ))]
  rgb_scratch_u16: Vec<u16>,
  /// Source-width canonical `R, G, B, A` `u8` staging for the packed
  /// straight/premult RGBA resample tail: each source row is converted
  /// to canonical RGBA (`Rgba` identity, `Bgra` swap, `Argb` / `Abgr`
  /// rotate α to slot 3) here — and, under [`AlphaMode::Premultiplied`],
  /// premultiplied in place — before feeding the 4-channel
  /// [`Self::rgba_stream`]. Lazily grown to `4 * width` `u8`; empty
  /// otherwise. Gated to `rgb`.
  #[cfg(feature = "rgb")]
  rgba_scratch: Vec<u8>,
  /// Source-width canonical `R, G, B, A` host-native `u16` staging for
  /// the high-bit packed RGBA resample tail (`Rgba64` / `Bgra64`): the
  /// wire row converts to host-native u16 RGBA here (and is premultiplied
  /// in place under [`AlphaMode::Premultiplied`]) before feeding the
  /// 4-channel [`Self::rgba_stream_u16`]. Lazily grown to `4 * width`
  /// `u16`; empty otherwise. Gated to `rgb`.
  #[cfg(feature = "rgb")]
  rgba_scratch_u16: Vec<u16>,
  /// Out-width host-native straight `R, G, B, A` `u16` staging for the
  /// high-bit packed RGBA resample tail: per finalized output row the
  /// binned native RGBA is resolved to its straight form here (a copy in
  /// [`AlphaMode::Straight`], an un-premultiply in
  /// [`AlphaMode::Premultiplied`]), then every native and narrowed
  /// output derives from this single straight row. Lazily grown to
  /// `4 * out_width` `u16`; empty for an output-less sink. Gated to
  /// `rgb`.
  #[cfg(feature = "rgb")]
  rgba_color_scratch_u16: Vec<u16>,
  /// Source-width host-native `u16` luma staging for the
  /// [`Gray16`](crate::source::Gray16) resample path: the wire `Gray16`
  /// row converts here (source wire `BE` → host-native u16, the same
  /// kernel the direct `luma_u16` path uses) before feeding
  /// [`Self::luma_stream_u16`]. The high-bit packed YUV families
  /// (`yuv-444-packed` / `y2xx` / `v210`) reuse it to stage their
  /// de-interleaved native Y row before the same u16 luma stream. Lazily
  /// grown to `width` `u16`; empty otherwise. Gated to `gray` /
  /// `yuv-444-packed` / `y2xx` / `v210`.
  #[cfg(any(
    feature = "gray",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210"
  ))]
  luma_scratch_u16: Vec<u16>,
  /// Source-width host-native `f32` luma staging for the
  /// [`Grayf32`](crate::source::Grayf32) resample path: the wire
  /// `Grayf32` row converts here (source wire `BE` → host-native f32 via
  /// the same kernel the direct `luma_f32` path uses) before feeding
  /// [`Self::luma_stream_f32`]. Lazily grown to `width` `f32`; empty
  /// otherwise. Gated to `gray`.
  #[cfg(feature = "gray")]
  luma_scratch_f32: Vec<f32>,
  /// Source-width `f32` RGB staging for packed-float-RGB resampling:
  /// the wire row converts here (host-native, lossless) before feeding
  /// [`Self::rgb_stream_f32`]. Lazily grown to `3 * width` `f32`; empty
  /// otherwise. Gated like [`Self::rgb_stream_f32`]: the `rgb-float`
  /// family fences in the engine, `gbr` already carries it.
  #[cfg(any(
    all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
    feature = "gbr"
  ))]
  rgb_scratch_f32: Vec<f32>,
  /// Out-width G/B/R `f32` plane staging for the float planar-GBR
  /// ([`Gbrpf32`](crate::source::Gbrpf32)) resample tail. That tail
  /// de-interleaves each binned packed `R, G, B` row back into G/B/R
  /// planes (`[0..w]` = G, `[w..2w]` = B, `[2w..3w]` = R) so it can run
  /// the exact direct `gbrpf32_*` kernels — the `rgb-float`
  /// ([`Rgbf32`](crate::source::Rgbf32)) tail's packed `rgbf32_*` kernels
  /// are not compiled in a `gbr` build. Lazily grown to `3 * out_width`
  /// `f32`; empty for an `rgb_f32`-only sink (which copies the binned row
  /// directly). Gated to `gbr`.
  #[cfg(feature = "gbr")]
  rgb_plane_scratch_f32: Vec<f32>,
  /// Out-width G/B/R `half::f16` plane staging for the half-float
  /// planar-GBR ([`Gbrpf16`](crate::source::Gbrpf16)) resample tail.
  /// There is no `AreaStream<f16>`, so that tail bins in `f32` (the
  /// shared [`Self::rgb_stream_f32`]) and, per finalized output row,
  /// de-interleaves the binned packed `R, G, B` `f32` row into the
  /// `f32` planes ([`Self::rgb_plane_scratch_f32`]), **rounds each
  /// element to `half::f16`** into these planes (`[0..w]` = G,
  /// `[w..2w]` = B, `[2w..3w]` = R), then runs the exact direct
  /// `gbrpf16_*` kernels — so every output is byte-identical to a
  /// direct `Gbrpf16` conversion of the `f32` block-mean rounded to
  /// f16. Lazily grown to `3 * out_width` `half::f16`; empty for a
  /// sink with no f16-plane-derived output. Gated to `gbr`.
  #[cfg(feature = "gbr")]
  rgb_plane_scratch_f16: Vec<half::f16>,
  /// Out-width **packed** `R, G, B` `half::f16` staging for the
  /// half-float packed-RGB ([`Rgbf16`](crate::source::Rgbf16)) resample
  /// tail. There is no `AreaStream<f16>`, so that tail bins in `f32` (the
  /// shared [`Self::rgb_stream_f32`]) and, per finalized output row,
  /// **rounds each binned packed `f32` element to `half::f16`** into this
  /// packed row, then runs the exact direct `rgbf16_*` kernels — so every
  /// output is byte-identical to a direct `Rgbf16` conversion of the `f32`
  /// block-mean rounded to f16. Unlike the planar
  /// [`Gbrpf16`](crate::source::Gbrpf16) tail this row stays **packed** (no
  /// de-interleave into planes), because the `rgbf16_*` kernels consume
  /// packed input. Lazily grown to `3 * out_width` `half::f16`; empty for a
  /// sink with no f16-derived output. Gated like [`Self::rgb_stream_f32`]:
  /// the `rgb-float` family fences in the engine.
  #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
  rgb_packed_scratch_f16: Vec<half::f16>,
  /// Source-width **linear-XYZ** `f32` staging for the
  /// [`Xyz12`](crate::source::Xyz12) resample path: the wire row
  /// converts here (inverse-OETF only, no matrix) before feeding
  /// [`Self::xyz_stream_f32`]. Lazily grown to `3 * width` `f32`; empty
  /// otherwise. Gated to `xyz`.
  #[cfg(feature = "xyz")]
  xyz_scratch_f32: Vec<f32>,
  /// Whether row primitives dispatch to their SIMD backend. Defaults
  /// to `true`; benchmarks flip this with [`Self::with_simd`] /
  /// [`Self::set_simd`] to A/B test scalar vs SIMD on the same frame.
  simd: bool,
  /// How the source's packed alpha relates to its color when the frame
  /// is area-resampled. Seeded from `F::DEFAULT_ALPHA_MODE`
  /// ([`DefaultAlphaMode`]) at construction; overridden per call by
  /// [`Self::with_alpha_mode`] / [`Self::set_alpha_mode`], the
  /// [`Self::with_simd`] flag pattern. Only the packed-RGBA resample
  /// tail consults it; every direct (identity-plan) path and every
  /// non-RGBA source ignores it.
  alpha_mode: AlphaMode,
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
        feature = "mono",
        feature = "yuv-semi-planar",
        feature = "yuv-packed",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210",
        feature = "rgb-legacy"
      ))]
      rgb_stream: None,
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210"
      ))]
      rgb_stream_u16: None,
      #[cfg(feature = "rgb")]
      rgba_stream: None,
      #[cfg(feature = "rgb")]
      rgba_stream_u16: None,
      #[cfg(feature = "rgb")]
      frozen_alpha_mode: None,
      #[cfg(any(
        all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
        feature = "gbr"
      ))]
      rgb_stream_f32: None,
      #[cfg(feature = "xyz")]
      xyz_stream_f32: None,
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
      luma_stream: None,
      #[cfg(any(
        feature = "gray",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210"
      ))]
      luma_stream_u16: None,
      #[cfg(feature = "gray")]
      luma_stream_f32: None,
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
      #[cfg(feature = "yuv-packed")]
      luma_scratch: Vec::new(),
      #[cfg(feature = "rgb-legacy")]
      legacy_rgb_native_scratch: Vec::new(),
      #[cfg(feature = "rgb-legacy")]
      legacy_rgb_packed_scratch: Vec::new(),
      #[cfg(any(
        feature = "rgb",
        feature = "gbr",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210"
      ))]
      rgb_scratch_u16: Vec::new(),
      #[cfg(feature = "rgb")]
      rgba_scratch: Vec::new(),
      #[cfg(feature = "rgb")]
      rgba_scratch_u16: Vec::new(),
      #[cfg(feature = "rgb")]
      rgba_color_scratch_u16: Vec::new(),
      #[cfg(any(
        feature = "gray",
        feature = "yuv-444-packed",
        feature = "y2xx",
        feature = "v210"
      ))]
      luma_scratch_u16: Vec::new(),
      #[cfg(feature = "gray")]
      luma_scratch_f32: Vec::new(),
      #[cfg(any(
        all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
        feature = "gbr"
      ))]
      rgb_scratch_f32: Vec::new(),
      #[cfg(feature = "gbr")]
      rgb_plane_scratch_f32: Vec::new(),
      #[cfg(feature = "gbr")]
      rgb_plane_scratch_f16: Vec::new(),
      #[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
      rgb_packed_scratch_f16: Vec::new(),
      #[cfg(feature = "xyz")]
      xyz_scratch_f32: Vec::new(),
      simd: true,
      alpha_mode: F::DEFAULT_ALPHA_MODE,
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
      feature = "xyz",
      feature = "bayer",
      feature = "gbr",
      feature = "gray",
      feature = "yuv-packed",
      feature = "yuv-444-packed",
      feature = "rgb-legacy",
      all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb"))
    )
  ))]
  pub(crate) fn rgb_scratch_capacity(&self) -> usize {
    self.rgb_scratch.capacity()
  }

  /// Capacity of the legacy 16-bit packed-RGB source-row native-channel
  /// staging scratch — a white-box probe for the resample ordering tests
  /// (a rejected row must not have grown the scratch). Gated on
  /// `rgb-legacy` + `std` like the tests that consume it.
  #[cfg(all(test, feature = "rgb-legacy", feature = "std"))]
  pub(crate) fn legacy_rgb_native_scratch_capacity(&self) -> usize {
    self.legacy_rgb_native_scratch.capacity()
  }

  /// Capacity of the legacy 16-bit packed-RGB re-packed-source-row
  /// staging scratch — a white-box probe for the resample tests (a
  /// native-`u16`-only sink, which copies the binned row at native depth,
  /// must still size it because the re-pack feeds the `rgb_u16` kernel;
  /// a no-output sink must not). Gated on `rgb-legacy` + `std`.
  #[cfg(all(test, feature = "rgb-legacy", feature = "std"))]
  pub(crate) fn legacy_rgb_packed_scratch_capacity(&self) -> usize {
    self.legacy_rgb_packed_scratch.capacity()
  }

  /// Whether the high-bit packed-RGB `u16` area stream has been
  /// created — a white-box probe for the resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `std` + the families that bin into the u16
  /// tail (`rgb` high-bit packed RGB, `yuv-444-packed` high-bit packed
  /// 4:4:4 YUV color group, `y2xx` / `v210` high-bit packed 4:2:2 YUV).
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "rgb",
      feature = "yuv-444-packed",
      feature = "y2xx",
      feature = "v210"
    )
  ))]
  pub(crate) fn rgb_stream_u16_allocated(&self) -> bool {
    self.rgb_stream_u16.is_some()
  }

  /// Whether the packed-float-RGB `f32` area stream has been created —
  /// a white-box probe for the resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `std` and the families that drive the float
  /// tail: the `rgb-float` family (fenced to the engine) or `gbr` (which
  /// already carries the engine).
  #[cfg(all(
    test,
    feature = "std",
    any(
      all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
      feature = "gbr"
    )
  ))]
  pub(crate) fn rgb_stream_f32_allocated(&self) -> bool {
    self.rgb_stream_f32.is_some()
  }

  /// Capacity of the float planar-GBR G/B/R plane scratch — a white-box
  /// probe for the resample tests (an `rgb_f32`-only sink must not grow
  /// it). Gated on `gbr` + `std` like the test that consumes it.
  #[cfg(all(test, feature = "gbr", feature = "std"))]
  pub(crate) fn rgb_plane_scratch_capacity(&self) -> usize {
    self.rgb_plane_scratch_f32.capacity()
  }

  /// Capacity of the half-float planar-GBR ([`Gbrpf16`](crate::source::Gbrpf16))
  /// G/B/R f16 plane scratch — a white-box probe for the resample tests (a
  /// no-output sink must not grow it). Gated on `gbr` + `std` like the test
  /// that consumes it.
  #[cfg(all(test, feature = "gbr", feature = "std"))]
  pub(crate) fn rgb_plane_scratch_f16_capacity(&self) -> usize {
    self.rgb_plane_scratch_f16.capacity()
  }

  /// Capacity of the half-float packed-RGB ([`Rgbf16`](crate::source::Rgbf16))
  /// packed f16 scratch row — a white-box probe for the resample tests (a
  /// no-output sink must not grow it). Gated on the `rgb-float` engine fence
  /// + `std` like the test that consumes it.
  #[cfg(all(
    test,
    feature = "std",
    feature = "rgb-float",
    any(feature = "yuv-planar", feature = "rgb")
  ))]
  pub(crate) fn rgb_packed_scratch_f16_capacity(&self) -> usize {
    self.rgb_packed_scratch_f16.capacity()
  }

  /// Whether the [`Xyz12`](crate::source::Xyz12) linear-XYZ `f32` area
  /// stream has been created — a white-box probe for the resample
  /// ordering tests (an out-of-sequence first row must be rejected
  /// before the stream is allocated). Gated on `xyz` and `std` like the
  /// tests that consume it.
  #[cfg(all(test, feature = "xyz", feature = "std"))]
  pub(crate) fn xyz_stream_f32_allocated(&self) -> bool {
    self.xyz_stream_f32.is_some()
  }

  /// Capacity of the [`Xyz12`](crate::source::Xyz12) source-row
  /// linear-XYZ staging scratch — a white-box probe for the resample
  /// ordering tests (a rejected row must not have grown the scratch).
  /// Gated on `xyz` and `std` like the tests that consume it.
  #[cfg(all(test, feature = "xyz", feature = "std"))]
  pub(crate) fn xyz_scratch_f32_capacity(&self) -> usize {
    self.xyz_scratch_f32.capacity()
  }

  /// Whether the single-channel luma `u8` area stream has been created
  /// — a white-box probe for the [`Gray8`](crate::source::Gray8),
  /// `mono`, and packed-YUV-4:2:2 resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `gray`/`mono`/`yuv-packed` and `std` like the
  /// tests that consume it.
  #[cfg(all(
    test,
    feature = "std",
    any(feature = "gray", feature = "mono", feature = "yuv-packed")
  ))]
  pub(crate) fn luma_stream_allocated(&self) -> bool {
    self.luma_stream.is_some()
  }

  /// Whether the 3-channel packed-RGB `u8` area stream has been created
  /// — a white-box probe for the packed-YUV-4:2:2, high-bit packed 4:4:4
  /// / 4:2:2 YUV, and legacy packed-RGB resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `std` + `any(yuv-packed, yuv-444-packed, y2xx,
  /// v210, rgb-legacy)` — those routes all bin their converted u8 RGB
  /// through the shared `rgb_stream`.
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "yuv-packed",
      feature = "yuv-444-packed",
      feature = "y2xx",
      feature = "v210",
      feature = "rgb-legacy"
    )
  ))]
  pub(crate) fn rgb_stream_allocated(&self) -> bool {
    self.rgb_stream.is_some()
  }

  /// Capacity of the packed-YUV-4:2:2 source-row Y de-interleave staging
  /// scratch — a white-box probe for the resample ordering tests (a
  /// rejected row must not have grown the scratch). Gated on
  /// `yuv-packed` and `std` like the tests that consume it.
  #[cfg(all(test, feature = "std", feature = "yuv-packed"))]
  pub(crate) fn luma_scratch_capacity(&self) -> usize {
    self.luma_scratch.capacity()
  }

  /// Capacity of the source-row `u16` RGB staging scratch — a white-box
  /// probe for the high-bit packed 4:4:4 YUV resample ordering tests (a
  /// rejected row must not have grown the scratch). Gated on
  /// `yuv-444-packed` + `std` like the tests that consume it.
  #[cfg(all(test, feature = "std", feature = "yuv-444-packed"))]
  pub(crate) fn rgb_scratch_u16_capacity(&self) -> usize {
    self.rgb_scratch_u16.capacity()
  }

  /// Capacity of the source-row `u16` luma (native Y) staging scratch —
  /// a white-box probe for the high-bit packed 4:4:4 YUV resample
  /// ordering tests (a rejected row must not have grown the scratch).
  /// Gated on `yuv-444-packed` + `std` like the tests that consume it.
  #[cfg(all(test, feature = "std", feature = "yuv-444-packed"))]
  pub(crate) fn luma_scratch_u16_capacity(&self) -> usize {
    self.luma_scratch_u16.capacity()
  }

  /// Whether the single-channel **u16** luma area stream has been
  /// created — a white-box probe for the
  /// [`Gray16`](crate::source::Gray16) and high-bit packed 4:4:4 YUV
  /// (`yuv-444-packed`) / high-bit packed 4:2:2 YUV (`y2xx` / `v210`)
  /// resample ordering tests (an out-of-sequence first row must be
  /// rejected before the stream is allocated). Gated on `gray` /
  /// `yuv-444-packed` / `y2xx` / `v210` and `std` like the tests that
  /// consume it.
  #[cfg(all(
    test,
    feature = "std",
    any(
      feature = "gray",
      feature = "yuv-444-packed",
      feature = "y2xx",
      feature = "v210"
    )
  ))]
  pub(crate) fn luma_stream_u16_allocated(&self) -> bool {
    self.luma_stream_u16.is_some()
  }

  /// Whether the single-channel **f32** luma area stream has been
  /// created — a white-box probe for the
  /// [`Grayf32`](crate::source::Grayf32) resample ordering tests (an
  /// out-of-sequence first row must be rejected before the stream is
  /// allocated). Gated on `gray` and `std` like the tests that consume
  /// it.
  #[cfg(all(test, feature = "std", feature = "gray"))]
  pub(crate) fn luma_stream_f32_allocated(&self) -> bool {
    self.luma_stream_f32.is_some()
  }

  /// Capacity of the [`Grayf32`](crate::source::Grayf32) source-row
  /// host-native `f32` luma staging scratch — a white-box probe for the
  /// resample tests (a rejected row must not have grown the scratch).
  /// Gated on `gray` and `std` like the tests that consume it.
  #[cfg(all(test, feature = "std", feature = "gray"))]
  pub(crate) fn luma_scratch_f32_capacity(&self) -> usize {
    self.luma_scratch_f32.capacity()
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

  /// Returns how the source's packed alpha is interpreted when the
  /// frame is area-resampled. See [`Self::with_alpha_mode`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn alpha_mode(&self) -> AlphaMode {
    self.alpha_mode
  }

  /// Sets the alpha interpretation in place. See
  /// [`Self::with_alpha_mode`] for the consuming builder variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_alpha_mode(&mut self, mode: AlphaMode) -> &mut Self {
    self.alpha_mode = mode;
    self
  }

  /// Sets how the source's packed alpha relates to its color channels
  /// when the frame is **area-resampled**, overriding the per-format
  /// default ([`DefaultAlphaMode`], [`AlphaMode::Straight`] for every
  /// packed-RGBA source colconv ships). Mirrors the [`Self::with_simd`]
  /// builder pattern.
  ///
  /// [`AlphaMode::Premultiplied`] makes the packed-RGBA resample tail
  /// bin premultiplied color and un-premultiply per finalized output
  /// row, so fully-transparent pixels never bleed their stored color
  /// into a downscaled result. The mode is a no-op for the direct
  /// (identity-plan) conversions — which copy alpha through untouched —
  /// and for every non-RGBA source. See [`Self::set_alpha_mode`] for
  /// the in-place variant.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_alpha_mode(mut self, mode: AlphaMode) -> Self {
    self.set_alpha_mode(mode);
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
        rgb_scratch
          .try_reserve_exact(row_bytes - rgb_scratch.len())
          .map_err(|_| {
            MixedSinkerError::Resample(ResampleError::AllocationFailed(
              crate::resample::PlanGeometry::new(width, height, width, height),
            ))
          })?;
        rgb_scratch.resize(row_bytes, 0);
      }
      Ok(&mut rgb_scratch[..row_bytes])
    }
  }
}

/// Grows `scratch` to a single source-width `u16` RGB row
/// (`width * 3` elements) for the **direct** (non-resample) path and
/// returns the slice. Follows the recoverable-allocation contract —
/// `try_reserve_exact` before the resize, mapping allocator refusal to a
/// recoverable [`MixedSinkerError`] instead of aborting in `process` — for
/// the 10-bit packed-RGB `rgba_u16` fan-out, which has no native α kernel
/// and so stages the native RGB row before
/// [`expand_rgb_u16_to_rgba_u16_row`].
#[cfg(feature = "rgb")]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn direct_rgb_u16_scratch(
  scratch: &mut Vec<u16>,
  width: usize,
  height: usize,
) -> Result<&mut [u16], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width, height, 3,
    )))?;
  if scratch.len() < row {
    scratch
      .try_reserve_exact(row - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(width, height, width, height),
        ))
      })?;
    scratch.resize(row, 0);
  }
  Ok(&mut scratch[..row])
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
  feature = "mono",
  feature = "yuv-semi-planar",
  feature = "yuv-packed",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210",
  feature = "rgb-legacy"
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

/// Grows `scratch` to a **source-width** `u8` luma row (`width`
/// bytes) and returns the slice, following the planner's recoverable-
/// allocation contract (the exact reserve makes the resize incapable
/// of reallocating; refusal surfaces as `AllocationFailed` in the
/// preflight phase, not an abort in infallible growth).
///
/// The staging point for `mono` resampling (each 1-bit source row is
/// expanded to source-width 0/255 luma here) and for **packed YUV
/// 4:2:2** resampling (the interleaved Y bytes are de-interleaved here)
/// before feeding the single-channel area stream.
#[cfg(any(feature = "mono", feature = "yuv-packed"))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn source_luma_scratch<'s>(
  scratch: &'s mut Vec<u8>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u8], MixedSinkerError> {
  if scratch.len() < width {
    scratch
      .try_reserve_exact(width - scratch.len())
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
    scratch.resize(width, 0);
  }
  Ok(&mut scratch[..width])
}

/// Grows `scratch` to a **source-width** host-native `u16` luma row
/// (`width` elements) and returns the slice, following the planner's
/// recoverable-allocation contract (the exact reserve makes the resize
/// incapable of reallocating; refusal surfaces as `AllocationFailed` in
/// the preflight phase, not an abort in infallible growth).
///
/// The staging point for the [`Gray16`](crate::source::Gray16) resample
/// path (the wire row converts to host-native u16 luma here) and for the
/// high-bit packed YUV families (`yuv-444-packed` / `y2xx` / `v210`),
/// whose native Y is de-interleaved here (each format's own
/// `*_to_luma_u16_row` kernel) before feeding the single-channel u16 luma
/// stream. The u16 twin of [`source_luma_scratch`].
#[cfg(any(
  feature = "gray",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210"
))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(super) fn source_luma_u16_scratch<'s>(
  scratch: &'s mut Vec<u16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u16], MixedSinkerError> {
  if scratch.len() < width {
    scratch
      .try_reserve_exact(width - scratch.len())
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
    scratch.resize(width, 0);
  }
  Ok(&mut scratch[..width])
}

/// Freezes the output configuration for a resampled packed-RGB frame
/// and reports whether any output is attached. Run before the
/// source-row conversion and the stream so a sink with no attached
/// outputs stays the documented legal no-op (it neither allocates nor
/// enforces sequencing) while a mid-frame output-set change is still
/// caught. Mirrors the YUV resample path's freeze-then-conditional
/// ordering.
#[cfg(any(feature = "rgb", feature = "gbr", feature = "bayer"))]
#[cfg_attr(
  not(any(feature = "rgb", feature = "gbr", feature = "bayer")),
  allow(dead_code)
)]
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
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
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
///
/// `rgb-legacy` reuses this u8 stream to bin its **native** R/G/B
/// channels (5/6/5, 5/5/5 or 4/4/4 values — each fits in a `u8`); the
/// per-format emit re-packs the binned native channels and runs the
/// direct kernels, so the RGB888 [`packed_rgb_resample_emit`] is not
/// shared with that family.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "bayer",
  feature = "rgb-legacy"
))]
#[cfg_attr(
  not(any(
    feature = "rgb",
    feature = "gbr",
    feature = "bayer",
    feature = "rgb-legacy"
  )),
  allow(dead_code)
)]
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
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210"
))]
#[cfg_attr(
  not(any(
    feature = "rgb",
    feature = "gbr",
    feature = "yuv-444-packed",
    feature = "y2xx",
    feature = "v210"
  )),
  allow(dead_code)
)]
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
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210"
))]
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
///
/// The legacy 16-bit packed-RGB family (`rgb-legacy`) shares this
/// freeze: its output set is exactly `rgb` / `rgba` / `rgb_u16` /
/// `rgba_u16` / `luma` / `luma_u16` / `hsv`, the same one the high-bit
/// path freezes.
#[cfg(any(feature = "rgb", feature = "gbr", feature = "rgb-legacy"))]
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
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
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
#[cfg(any(feature = "rgb", feature = "gbr"))]
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
/// finalized output row. Binning runs at the source's native depth
/// (`SRC_BITS` active bits per `u16` element); the `rgb_u16` /
/// `rgba_u16` outputs copy it directly, while the u8 and `luma_u16`
/// outputs derive from a single `>> (SRC_BITS - 8)` narrowing — the same
/// source-of-truth ordering the direct path uses (luma / luma_u16 / hsv
/// all read the narrowed u8 RGB). `SRC_BITS` is `16` for the packed
/// `Rgb48` / `Bgr48` sources (whose elements are full-range u16) and the
/// source bit depth for the high-bit planar GBR sources (`Gbrp9` … 14
/// carry fewer than 16 active bits, so their narrowing shift and opaque
/// `rgba_u16` alpha both track `SRC_BITS`, not a hard-coded 16).
/// `narrow_scratch` is sized to the out-width u8 RGB row only when one of
/// those narrowed outputs is attached, so a native-u16-only sink neither
/// grows it nor risks its allocation failure.
#[cfg(any(
  feature = "rgb",
  feature = "gbr",
  feature = "yuv-444-packed",
  feature = "y2xx",
  feature = "v210"
))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_u16_resample_emit<const SRC_BITS: u32, const NATIVE_LUMA16: bool>(
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
  const {
    assert!(
      SRC_BITS >= 8 && SRC_BITS <= 16,
      "SRC_BITS must be in [8, 16]"
    )
  };
  let ow = plan.out_w();
  // The u8 / luma_u16 outputs derive from a `>> (SRC_BITS - 8)`
  // narrowing of the binned row; a native-u16-only sink (only rgb_u16 /
  // rgba_u16) never touches it, so the out-width u8 scratch is sized —
  // and its allocation failure risked — only when one of those outputs
  // is attached. The predicate gates both the sizing here and the use in
  // the closure, so they cannot drift.
  let need_narrow = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || hsv.is_some()
    || (!NATIVE_LUMA16 && luma_u16.is_some());
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
      crate::row::expand_rgb_u16_to_rgba_u16_row::<SRC_BITS>(
        binned,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
      );
    }
    // Native-precision `luma_u16`: derive directly from the native-depth
    // binned RGB, byte-identical to the direct
    // `gbr_to_luma_u16_high_bit_row` path. Only the high-bit-GBR callers
    // set `NATIVE_LUMA16`; the `Rgb48` / `Bgr48` callers leave it false
    // and take the narrowed `luma_u16` in the `need_narrow` block below.
    if NATIVE_LUMA16 && let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::rgb_to_luma_u16_native_row(
        binned,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        matrix,
        full_range,
        SRC_BITS,
      );
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      for (d, &s) in nrow.iter_mut().zip(binned.iter()) {
        *d = (s >> (SRC_BITS - 8)) as u8;
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
      if !NATIVE_LUMA16 && let Some(buf) = luma_u16.as_deref_mut() {
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

/// Source-width canonical `R, G, B, A` `u8` staging for the packed
/// straight/premult RGBA resample tail. Grows `scratch` to `4 * width`
/// `u8` under the planner's recoverable-allocation contract. Mirrors
/// [`source_rgb_scratch`] for the 4-channel RGBA row.
#[cfg(feature = "rgb")]
pub(super) fn source_rgba_scratch<'s>(
  scratch: &'s mut Vec<u8>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u8], MixedSinkerError> {
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      4,
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

/// `u16` analogue of [`source_rgba_scratch`] — source-width canonical
/// `R, G, B, A` host-native `u16` staging for the high-bit packed RGBA
/// resample tail. Grows `scratch` to `4 * width` `u16`.
#[cfg(feature = "rgb")]
pub(super) fn source_rgba_u16_scratch<'s>(
  scratch: &'s mut Vec<u16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u16], MixedSinkerError> {
  let row = width
    .checked_mul(4)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.src_h(),
      4,
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

/// Premultiplies one canonical `R, G, B, A` row in place: each color
/// channel becomes `round(c * α / MAX)`; α is left unchanged. The exact
/// integer op the [`AlphaMode::Premultiplied`] oracle mirrors, so the
/// binned-then-un-premultiplied output is byte-exact. `MAX` is `255` for
/// the u8 path and the source's native max for the u16 path.
#[cfg(feature = "rgb")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn premultiply_rgba_row_in_place<T>(row: &mut [T], width: usize, max: u32)
where
  T: Copy + TryFrom<u32> + Into<u32>,
{
  let half = max / 2;
  for px in row[..width * 4].chunks_exact_mut(4) {
    let a: u32 = px[3].into();
    for c in &mut px[..3] {
      let v: u32 = (*c).into();
      let pm = (v * a + half) / max;
      // `pm <= max` (since `v, a <= max`), so the cast back never fails.
      *c = T::try_from(pm).unwrap_or_else(|_| unreachable!("premultiplied value <= max"));
    }
  }
}

/// Un-premultiplied straight color channel for one premultiplied binned
/// value: `round(pm * MAX / α)` clamped to `MAX`, or `0` when `α == 0`
/// (a fully-transparent binned pixel exposes no color, so it cannot
/// bleed). The exact integer inverse of [`premultiply_rgba_row_in_place`].
#[cfg(feature = "rgb")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn unpremultiply_channel(pm: u32, a: u32, max: u32) -> u32 {
  // `checked_div` yields `None` exactly when `α == 0`, which maps to a
  // zero straight channel (a fully-transparent binned pixel exposes no
  // color); otherwise round-half-up and clamp to `max`.
  (pm * max + a / 2).checked_div(a).map_or(0, |q| q.min(max))
}

/// Un-premultiplies one binned canonical `R, G, B, A` row into the
/// caller's straight-RGBA destination (α copied through). Applied per
/// finalized output row when binning premultiplied.
#[cfg(feature = "rgb")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn unpremultiply_binned_rgba_into<T>(binned: &[T], dst: &mut [T], width: usize, max: u32)
where
  T: Copy + TryFrom<u32> + Into<u32>,
{
  for (out_px, in_px) in dst[..width * 4]
    .chunks_exact_mut(4)
    .zip(binned[..width * 4].chunks_exact(4))
  {
    let a: u32 = in_px[3].into();
    for c in 0..3 {
      let straight = unpremultiply_channel(in_px[c].into(), a, max);
      out_px[c] =
        T::try_from(straight).unwrap_or_else(|_| unreachable!("un-premultiplied value <= max"));
    }
    out_px[3] = in_px[3];
  }
}

/// Un-premultiplies one binned canonical `R, G, B, A` row into a
/// straight **RGB** destination (α dropped) — the packed RGB the
/// luma / hsv kernels consume in premultiplied mode.
#[cfg(feature = "rgb")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn unpremultiply_binned_rgb_into<T>(binned: &[T], dst: &mut [T], width: usize, max: u32)
where
  T: Copy + TryFrom<u32> + Into<u32>,
{
  for (out_px, in_px) in dst[..width * 3]
    .chunks_exact_mut(3)
    .zip(binned[..width * 4].chunks_exact(4))
  {
    let a: u32 = in_px[3].into();
    for c in 0..3 {
      let straight = unpremultiply_channel(in_px[c].into(), a, max);
      out_px[c] =
        T::try_from(straight).unwrap_or_else(|_| unreachable!("un-premultiplied value <= max"));
    }
  }
}

/// Drops α from one canonical `R, G, B, A` row into a packed `R, G, B`
/// destination — the straight-mode RGB the luma / hsv kernels consume.
#[cfg(feature = "rgb")]
#[cfg_attr(not(tarpaulin), inline(always))]
fn drop_alpha_rgba_to_rgb_row<T: Copy>(rgba: &[T], dst: &mut [T], width: usize) {
  for (out_px, in_px) in dst[..width * 3]
    .chunks_exact_mut(3)
    .zip(rgba[..width * 4].chunks_exact(4))
  {
    out_px.copy_from_slice(&in_px[..3]);
  }
}

/// Rejects a mid-frame [`AlphaMode`] change for a resampled packed-RGBA
/// frame. The mode is snapshotted in `begin_frame` (`frozen` is the mode
/// at frame start); the resample route and binning use it, so each sink
/// calls this **before** route selection in `process` and a later
/// differing live mode trips `ResampleOutputsChanged` — no row is fed
/// under a changed mode, and the snapshot is immune to out-of-sequence
/// rows since it is taken at the frame boundary, not the first row.
/// (`frozen` is `None` only before the first `begin_frame`, a contract
/// violation, which is likewise rejected.)
#[cfg(feature = "rgb")]
pub(super) fn check_frozen_alpha_mode(
  frozen: Option<AlphaMode>,
  current: AlphaMode,
  idx: usize,
) -> Result<(), MixedSinkerError> {
  if frozen != Some(current) {
    return Err(MixedSinkerError::ResampleOutputsChanged(
      ResampleOutputsChanged::new(idx),
    ));
  }
  Ok(())
}

/// Row-stage fused downscale for the packed straight/premultiplied RGBA
/// 8-bit family (`Rgba` / `Bgra` / `Argb` / `Abgr`) — the alpha-aware
/// 4-channel analogue of the 3-channel [`packed_rgb_resample_emit`]
/// path. `convert_rgba` stages the source row as a canonical
/// source-width `R, G, B, A` u8 row (identity / swap / α-rotate per
/// format); this tail bins all four channels so resampled alpha is a
/// real area mean (the forced-opaque-`0xFF` bug the 3-channel path hit),
/// then per finalized output row emits rgba (the binned row),
/// rgb (drop α), luma / luma_u16 / hsv (from the binned RGB).
///
/// Under [`AlphaMode::Premultiplied`] the staged row is premultiplied
/// in place before binning and un-premultiplied per output row, so the
/// color outputs are alpha-weighted and transparent pixels never bleed.
///
/// Atomic preflight with conditional ordering: a no-output call returns
/// before any freeze; an out-of-sequence FIRST row is rejected before
/// the freeze (so a rejected row stores no snapshot to poison a retry);
/// on a later row the freeze runs first (so a mid-frame output change
/// trips `ResampleOutputsChanged` rather than being masked by a
/// freshly-attached stream's row-0 mismatch). The stream and scratch are
/// created only after the sequence check, all before the single feed, so
/// a failure mutates no caller output.
#[cfg(feature = "rgb")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_resample(
  rgba_stream: &mut Option<crate::resample::AreaStream<u8>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch: &mut Vec<u8>,
  rgb_drop_scratch: &mut Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  alpha_mode: AlphaMode,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_any =
    rgb.is_some() || rgba.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  // No-output call: nothing to sequence, stays a no-op (no freeze, no
  // allocation) regardless of the row index.
  if !need_any {
    return Ok(());
  }
  let expected = rgba_stream.as_ref().map_or(0, |s| s.next_y());
  let first_row = resample_outputs.is_none();
  // First row: reject an out-of-sequence row before the freeze so a
  // rejected first row stores no snapshot that would poison a retry.
  if first_row && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Later row: a mid-frame output change is reported above; an
  // out-of-sequence later row is rejected here.
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  let premult = alpha_mode.is_premultiplied();
  // The luma / luma_u16 / hsv outputs (and rgb) need a packed RGB row.
  // It is the per-mode binned color with α dropped — sized to the
  // out-width RGB row only when one of those is attached, so an
  // rgba-only sink neither grows it nor risks its allocation failure.
  let need_rgb_drop = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  if rgba_stream.is_none() {
    *rgba_stream = Some(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      4,
    )?);
  }
  let rgb_drop: &mut [u8] = if need_rgb_drop {
    source_rgb_scratch(rgb_drop_scratch, ow, plan)?
  } else {
    &mut []
  };
  let src_rgba = source_rgba_scratch(rgba_scratch, w, plan)?;
  convert_rgba(src_rgba);
  if premult {
    premultiply_rgba_row_in_place::<u8>(src_rgba, w, 255);
  }
  let stream = rgba_stream.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    // RGBA output is the per-mode straight color: straight mode copies
    // the binned row; premult mode un-premultiplies it into the dst.
    if let Some(buf) = rgba.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      if premult {
        unpremultiply_binned_rgba_into::<u8>(binned, dst, ow, 255);
      } else {
        dst.copy_from_slice(binned);
      }
    }
    if need_rgb_drop {
      let nrow = &mut rgb_drop[..3 * ow];
      if premult {
        unpremultiply_binned_rgb_into::<u8>(binned, nrow, ow, 255);
      } else {
        drop_alpha_rgba_to_rgb_row(binned, nrow, ow);
      }
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

/// Row-stage fused downscale for the high-bit packed straight/premult
/// RGBA family (`Rgba64` / `Bgra64`) — the alpha-aware 4-channel
/// analogue of the 3-channel [`packed_rgb_u16_resample_emit`] path.
/// `convert_rgba` stages the wire row as a canonical host-native
/// source-width `R, G, B, A` u16 row (the format's `*_to_rgba_u16`
/// kernel, source wire `BE`); this tail bins all four channels at native
/// 16-bit depth so resampled alpha is a real area mean (not the
/// forced-opaque-`0xFFFF` the 3-channel u16 path emitted), then per
/// finalized output row resolves the binned native RGBA to its straight
/// form (a copy in straight mode, an un-premultiply in premult mode) and
/// emits: rgba_u16 / rgb_u16 at native depth, and rgba / rgb / luma /
/// luma_u16 / hsv from a single `>> 8` narrowing — the source-of-truth
/// ordering the 3-channel u16 path uses.
///
/// `MAX` is the source's native maximum (`65535` for the full-16-bit
/// `Rgba64` / `Bgra64`), governing both the premultiply rounding and the
/// un-premultiply clamp.
///
/// Same atomic conditional-ordering preflight as [`packed_rgba_resample`]:
/// a no-output call returns before any freeze; an out-of-sequence first
/// row is rejected before the freeze; a later-row output change trips
/// `ResampleOutputsChanged`; the stream and every scratch are created
/// after the sequence check and before the single feed.
#[cfg(feature = "rgb")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgba_u16_resample(
  rgba_stream_u16: &mut Option<crate::resample::AreaStream<u16>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgba_scratch_u16: &mut Vec<u16>,
  color_scratch_u16: &mut Vec<u16>,
  narrow_scratch: &mut Vec<u8>,
  max: u32,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  alpha_mode: AlphaMode,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgba_u16: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_any = rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma.is_some()
    || luma_u16.is_some()
    || hsv.is_some();
  if !need_any {
    return Ok(());
  }
  let expected = rgba_stream_u16.as_ref().map_or(0, |s| s.next_y());
  let first_row = resample_outputs.is_none();
  if first_row && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  // The u8 / narrowed outputs (incl. luma_u16, which these full-range
  // u16 sources derive from narrowed RGB at 8-bit precision — the same
  // source-of-truth the 3-channel u16 path's non-native-luma branch
  // uses) come from a `>> 8` narrowing of the straight RGB. A
  // native-u16-only sink (rgb_u16 / rgba_u16 only) never touches it, so
  // the out-width u8 RGB scratch is sized — and its allocation failure
  // risked — only when one of those outputs is attached.
  let need_narrow =
    rgb.is_some() || rgba.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  if rgba_stream_u16.is_none() {
    *rgba_stream_u16 = Some(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      4,
    )?);
  }
  // Out-width straight RGBA color row (resolved per output row). Always
  // sized when any output is attached, so every native and narrowed
  // output reads one canonical straight row.
  let color = source_rgba_u16_scratch(color_scratch_u16, ow, plan)?;
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  let premult = alpha_mode.is_premultiplied();
  let src_rgba = source_rgba_u16_scratch(rgba_scratch_u16, w, plan)?;
  convert_rgba_u16(src_rgba);
  if premult {
    premultiply_rgba_row_in_place::<u16>(src_rgba, w, max);
  }
  let stream = rgba_stream_u16.as_mut().expect("created above");
  stream.feed_row(idx, src_rgba, use_simd, |oy, binned| {
    // Resolve the per-mode straight native RGBA once.
    let color = &mut color[..4 * ow];
    if premult {
      unpremultiply_binned_rgba_into::<u16>(binned, color, ow, max);
    } else {
      color.copy_from_slice(binned);
    }
    // Native-depth u16 outputs copy from the straight color row.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      buf[oy * 4 * ow..(oy + 1) * 4 * ow].copy_from_slice(color);
    }
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      drop_alpha_rgba_to_rgb_row(color, dst, ow);
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      for (d, px) in nrow.chunks_exact_mut(3).zip(color.chunks_exact(4)) {
        d[0] = (px[0] >> 8) as u8;
        d[1] = (px[1] >> 8) as u8;
        d[2] = (px[2] >> 8) as u8;
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = rgba.as_deref_mut() {
        // Narrow all four straight channels (α `>> 8` too) into the dst.
        let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
        for (d, px) in dst.chunks_exact_mut(4).zip(color.chunks_exact(4)) {
          d[0] = (px[0] >> 8) as u8;
          d[1] = (px[1] >> 8) as u8;
          d[2] = (px[2] >> 8) as u8;
          d[3] = (px[3] >> 8) as u8;
        }
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
      // luma_u16: 8-bit-precision Y' derived from the narrowed straight
      // RGB and zero-extended — byte-identical to the direct full-range
      // u16 path's `luma_u16` (which narrows to u8 before luma).
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

/// Row-stage fused downscale for the high-bit packed 4:4:4 YUV family
/// (`V30X` / `V410` / `Xv36`). Unlike the 8-bit packed-YUV-4:2:2 path,
/// which carries a single u8 colour binning, this routes **three
/// independent area streams** because the direct path's per-output
/// conversions round and scale independently and luma is taken from the
/// native Y — so no single binning can reproduce every output:
///
/// 1. **u8 colour (rgb / rgba / hsv):** `convert_rgb_u8` stages the
///    direct u8 YUV→RGB row into a source-width u8 scratch; that row
///    bins through the shared u8 packed-RGB tail
///    ([`packed_rgb_resample_emit`]), which emits rgb / rgba / hsv.
///    Luma is *not* derived here (it is taken from Y), so `&mut None`
///    is passed for the tail's luma / luma_u16 outputs.
/// 2. **u16 colour (rgb_u16 / rgba_u16):** `convert_rgb_u16` stages the
///    native-depth u16 YUV→RGB row into a source-width u16 scratch; that
///    row bins through the shared u16 packed-RGB tail
///    ([`packed_rgb_u16_resample_emit`]) at `SRC_BITS`, emitting only
///    rgb_u16 / rgba_u16 (every narrowed output is `&mut None`, so the
///    tail's narrow scratch is never sized).
/// 3. **luma (luma / luma_u16):** `deinterleave_y` stages the native Y
///    into a source-width u16 scratch; a 1-channel `AreaStream<u16>`
///    bins it at native depth. `luma_u16` is the host-native binned Y;
///    `luma` is `binned_y >> (SRC_BITS - 8)`.
///
/// Colour outputs are byte-identical to the area-bin of the direct
/// full-resolution conversion (convert-then-bin — the fused form of
/// converting at full resolution then area-downscaling the RGB); luma is
/// the area-mean of the native Y. A uniform-gray downscale leaves every
/// colour output unchanged — the regression a single narrowed binning
/// would silently break.
///
/// Atomic preflight: a single [`frozen_outputs_check`] over the full
/// output set, then a single sequence check **before any allocation**
/// (so an out-of-sequence row is rejected without allocating and
/// `AllocationFailed` never masks `OutOfSequenceRow`), then every stream
/// and every source-width scratch are created — all before the first
/// feed — so a failure mutates no caller output. A no-output call has no
/// stream to sequence and stays a no-op regardless of the row index.
#[cfg(feature = "yuv-444-packed")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv444_triple_resample<const SRC_BITS: u32>(
  rgb_stream: &mut Option<crate::resample::AreaStream<u8>>,
  rgb_stream_u16: &mut Option<crate::resample::AreaStream<u16>>,
  luma_stream_u16: &mut Option<crate::resample::AreaStream<u16>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  luma_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  convert_rgb_u8: impl FnOnce(&mut [u8]),
  convert_rgb_u16: impl FnOnce(&mut [u16]),
  deinterleave_y: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      SRC_BITS >= 8 && SRC_BITS <= 16,
      "SRC_BITS must be in [8, 16]"
    )
  };
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_u8_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let need_u16_color = rgb_u16.is_some() || rgba_u16.is_some();

  // Single sequence check before any allocation. The canonical sequence
  // counter is whichever attached stream is fed every row; all attached
  // streams advance in lockstep, so checking one rejects an
  // out-of-sequence row for all without allocating any stream or scratch.
  let expected = if need_luma {
    luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if need_u8_color {
    rgb_stream.as_ref().map_or(0, |s| s.next_y())
  } else if need_u16_color {
    rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else {
    return Ok(());
  };
  // On the first row of a frame nothing is frozen yet, so reject an
  // out-of-sequence row here — before the freeze — so a rejected first row
  // never stores a snapshot that would poison a retry. On a later row the
  // freeze runs first (below) so a mid-frame output-set change is reported
  // as ResampleOutputsChanged rather than masked by a freshly-attached
  // stream's row-0 sequence mismatch.
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  // Single freeze over the full output set (luma_f32 is never produced
  // by this family, so it is frozen as absent). A mid-frame output change
  // trips ResampleOutputsChanged.
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }

  // Create the streams (post-sequence-check). Each plane runs the full
  // output grid against its own source grid (width `w`, height
  // `plan.src_h()`); the colour streams carry 3 interleaved channels, the
  // luma stream 1.
  if need_u8_color && rgb_stream.is_none() {
    *rgb_stream = Some(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      3,
    )?);
  }
  if need_u16_color && rgb_stream_u16.is_none() {
    *rgb_stream_u16 = Some(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      3,
    )?);
  }
  if need_luma && luma_stream_u16.is_none() {
    *luma_stream_u16 = Some(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      1,
    )?);
  }

  // Stage every source-width row (all fallible growths run before the
  // first feed, keeping the call atomic). The three scratches are
  // distinct fields and never alias.
  let u8_color_row = if need_u8_color {
    let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb_u8(scratch);
    Some(scratch)
  } else {
    None
  };
  let u16_color_row = if need_u16_color {
    let scratch = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
    convert_rgb_u16(scratch);
    Some(scratch)
  } else {
    None
  };
  let y_row = if need_luma {
    let scratch = source_luma_u16_scratch(luma_scratch_u16, w, plan)?;
    deinterleave_y(scratch);
    Some(scratch)
  } else {
    None
  };

  // Feed each stream and emit. The u8 tail emits rgb / rgba / hsv only
  // (luma comes from Y, so its luma / luma_u16 are `&mut None`); the u16
  // tail emits rgb_u16 / rgba_u16 only (every narrowed output is
  // `&mut None`, so its narrow scratch is never sized — `rgb_scratch`,
  // already consumed by the u8 feed above, is passed as the unused
  // placeholder).
  if let Some(scratch) = u8_color_row {
    let stream = rgb_stream.as_mut().expect("created in the preflight");
    packed_rgb_resample_emit(
      stream, plan, rgb, rgba, &mut None, &mut None, hsv, scratch, matrix, full_range, idx,
      use_simd,
    )?;
  }
  if let Some(scratch) = u16_color_row {
    let stream = rgb_stream_u16.as_mut().expect("created in the preflight");
    packed_rgb_u16_resample_emit::<SRC_BITS, false>(
      stream,
      plan,
      &mut None,
      &mut None,
      &mut None,
      rgb_u16,
      rgba_u16,
      &mut None,
      &mut None,
      scratch,
      rgb_scratch,
      matrix,
      full_range,
      idx,
      use_simd,
    )?;
  }
  if let Some(scratch) = y_row {
    let stream = luma_stream_u16.as_mut().expect("created in the preflight");
    stream.feed_row(idx, scratch, use_simd, |oy, binned_y| {
      // luma_u16: host-native pass-through of the binned native Y.
      if let Some(buf) = luma_u16.as_deref_mut() {
        buf[oy * ow..(oy + 1) * ow].copy_from_slice(binned_y);
      }
      // luma: narrow the binned native Y to u8 (`>> (SRC_BITS - 8)`).
      if let Some(buf) = luma.as_deref_mut() {
        for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(binned_y) {
          *dst = (src >> (SRC_BITS - 8)) as u8;
        }
      }
    })?;
  }

  Ok(())
}

/// Row-stage fused downscale for the **high-bit packed 4:2:2 YUV**
/// family (`Y210` / `Y212` / `Y216`, plus the exotic 10-bit `V210` word
/// packing) — the 4:2:2 analogue of the high-bit 4:4:4 route, with
/// **three** independent native-precision binnings.
///
/// High-bit packed YUV needs three binnings because the u8 and u16
/// YUV→RGB kernels (`range_params_n::<BITS, 8>` vs `::<BITS, BITS>`)
/// round and scale *independently*, and luma is native Y. Narrowing the
/// u16 bin to u8 would change a uniform-gray downscale's colour — a real
/// parity bug — so each output group bins its own native-precision
/// conversion:
/// 1. **u8 colour (rgb / rgba / hsv):** `convert_rgb_u8` fills a
///    source-width u8 RGB row (the format's `*_to_rgb_row` kernel —
///    chroma de-interleave + 4:2:2 upsample in-register), binned through
///    the u8 packed-RGB tail and fanned out to rgb / rgba / hsv.
/// 2. **u16 colour (rgb_u16 / rgba_u16):** `convert_rgb_u16` fills a
///    source-width native u16 RGB row (`*_to_rgb_u16_row`, source wire
///    `BE`), binned at native depth through the u16 tail with
///    `NATIVE_LUMA16 = false` and **only** rgb_u16 / rgba_u16 attached
///    (every narrowed u8 output passed as `&mut None`, so the tail's
///    narrow scratch is never sized).
/// 3. **luma (luma / luma_u16):** `deinterleave_y_u16` fills a
///    source-width host-native u16 Y row (`*_to_luma_u16_row`), binned
///    through the 1-channel u16 luma stream; luma_u16 is the host-native
///    binned Y, luma is `binned_Y >> (SRC_BITS - 8)`.
///
/// Colour outputs are byte-identical to the area-bin of the direct
/// full-resolution conversion (convert-then-bin — the fused form of
/// converting at full resolution then area-downscaling the RGB); luma is
/// the area-mean of the native Y. Atomic preflight: a single
/// [`frozen_outputs_check`], then a sequence check before any allocation
/// (so an out-of-sequence row is rejected without allocating and
/// `AllocationFailed` never masks `OutOfSequenceRow`), then the three
/// distinct, non-aliasing scratches grow and the three source rows stage
/// — all before the first feed, so a failure mutates no caller output. A
/// no-output call is a true no-op regardless of the row index.
#[cfg(any(feature = "y2xx", feature = "v210"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv422_triple_resample<const SRC_BITS: u32>(
  luma_stream_u16: &mut Option<crate::resample::AreaStream<u16>>,
  rgb_stream: &mut Option<crate::resample::AreaStream<u8>>,
  rgb_stream_u16: &mut Option<crate::resample::AreaStream<u16>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_scratch_u16: &mut Vec<u16>,
  rgb_scratch: &mut Vec<u8>,
  rgb_scratch_u16: &mut Vec<u16>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  matrix: crate::ColorMatrix,
  full_range: bool,
  deinterleave_y_u16: impl FnOnce(&mut [u16]),
  convert_rgb_u8: impl FnOnce(&mut [u8]),
  convert_rgb_u16: impl FnOnce(&mut [u16]),
) -> Result<(), MixedSinkerError> {
  const {
    assert!(
      SRC_BITS > 8 && SRC_BITS <= 16,
      "SRC_BITS must be in (8, 16] for high-bit packed 4:2:2 YUV"
    )
  };
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color_u8 = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let need_color_u16 = rgb_u16.is_some() || rgba_u16.is_some();

  // Sequence-check before allocating: every active stream started at row
  // 0 and the frozen output set keeps the active group fixed for the
  // frame, so they advance in lockstep — any active stream gives the
  // expected row. A no-output call has no stream to sequence and stays a
  // no-op regardless of the row index.
  let expected = if need_luma {
    luma_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else if need_color_u8 {
    rgb_stream.as_ref().map_or(0, |s| s.next_y())
  } else if need_color_u16 {
    rgb_stream_u16.as_ref().map_or(0, |s| s.next_y())
  } else {
    return Ok(());
  };
  // On the first row of a frame nothing is frozen yet, so reject an
  // out-of-sequence row here — before the freeze — so a rejected first row
  // never stores a snapshot that would poison a retry. On a later row the
  // freeze runs first (below) so a mid-frame output-set change is reported
  // as ResampleOutputsChanged rather than masked by a freshly-attached
  // stream's row-0 sequence mismatch.
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  if need_luma && luma_stream_u16.is_none() {
    *luma_stream_u16 = Some(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      1,
    )?);
  }
  if need_color_u8 && rgb_stream.is_none() {
    *rgb_stream = Some(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      3,
    )?);
  }
  if need_color_u16 && rgb_stream_u16.is_none() {
    *rgb_stream_u16 = Some(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      3,
    )?);
  }
  // Stage the three source-width rows into their own distinct,
  // non-aliasing scratches (all fallible growths precede the feeds,
  // keeping the call atomic).
  let luma_row = if need_luma {
    let scratch = source_luma_u16_scratch(luma_scratch_u16, w, plan)?;
    deinterleave_y_u16(scratch);
    Some(scratch)
  } else {
    None
  };
  let color_u8_row = if need_color_u8 {
    let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb_u8(scratch);
    Some(scratch)
  } else {
    None
  };
  let color_u16_row = if need_color_u16 {
    let scratch = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
    convert_rgb_u16(scratch);
    Some(scratch)
  } else {
    None
  };

  // Binning 3 — native Y through the 1-channel u16 luma stream. The
  // binned row is host-native; luma_u16 is its pass-through, luma the
  // `>> (SRC_BITS - 8)` narrowing.
  if let Some(y_row) = luma_row {
    let stream = luma_stream_u16.as_mut().expect("created in the preflight");
    stream.feed_row(idx, y_row, use_simd, |oy, binned_y| {
      if let Some(buf) = luma_u16.as_deref_mut() {
        buf[oy * ow..(oy + 1) * ow].copy_from_slice(binned_y);
      }
      if let Some(buf) = luma.as_deref_mut() {
        for (dst, &y) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(binned_y) {
          *dst = (y >> (SRC_BITS - 8)) as u8;
        }
      }
    })?;
  }

  // Binning 1 — u8 colour through the shared u8 packed-RGB tail (luma /
  // luma_u16 handled by binning 3, so they are `&mut None` here).
  if let Some(scratch) = color_u8_row {
    let stream = rgb_stream.as_mut().expect("created in the preflight");
    packed_rgb_resample_emit(
      stream, plan, rgb, rgba, &mut None, &mut None, hsv, scratch, matrix, full_range, idx,
      use_simd,
    )?;
  }

  // Binning 2 — u16 colour through the shared u16 packed-RGB tail at
  // native depth. Only rgb_u16 / rgba_u16 are emitted; every narrowed u8
  // output is `&mut None`, so the tail's narrow scratch is never sized.
  if let Some(scratch) = color_u16_row {
    let stream = rgb_stream_u16.as_mut().expect("created in the preflight");
    packed_rgb_u16_resample_emit::<SRC_BITS, false>(
      stream,
      plan,
      &mut None,
      &mut None,
      &mut None,
      rgb_u16,
      rgba_u16,
      &mut None,
      &mut None,
      scratch,
      rgb_scratch,
      matrix,
      full_range,
      idx,
      use_simd,
    )?;
  }

  Ok(())
}

/// Source-width `f32` RGB staging for packed-float-RGB resampling: the
/// wire row converts here (host-native f32, lossless) before feeding
/// [`AreaStream<f32>`]. Grows `scratch` to `3 * width` `f32` under the
/// planner's recoverable-allocation contract. Mirrors
/// [`source_rgb_u16_scratch`] for the float element path.
#[cfg(any(
  all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
  feature = "gbr"
))]
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

/// Out-width G/B/R `f32` plane staging for the [`Gbrpf32`](crate::source::Gbrpf32)
/// arm of the float packed-RGB tail: the dedicated `gbr` emit de-interleaves
/// each binned packed row back into three planes so it can run the exact
/// direct `gbrpf32_*` kernels for every output. Grows `scratch` to
/// `3 * width` `f32` — three contiguous planes — under the planner's
/// recoverable-allocation contract. Only the `gbr` emit consumes it; the
/// `rgb-float` ([`Rgbf32`](crate::source::Rgbf32)) caller never allocates it.
#[cfg(feature = "gbr")]
pub(super) fn rgb_plane_f32_scratch<'s>(
  scratch: &'s mut Vec<f32>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [f32], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
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

/// Out-width G/B/R `half::f16` plane staging for the
/// [`Gbrpf16`](crate::source::Gbrpf16) arm of the float packed-RGB
/// tail. There is no `AreaStream<f16>`, so binning runs in `f32`; this
/// emit de-interleaves each binned packed row into `f32` planes, rounds
/// them to `half::f16` here, and runs the exact direct `gbrpf16_*`
/// kernels for every output. Grows `scratch` to `3 * width`
/// `half::f16` — three contiguous planes — under the planner's
/// recoverable-allocation contract. Only the f16 emit consumes it.
#[cfg(feature = "gbr")]
pub(super) fn rgb_plane_f16_scratch<'s>(
  scratch: &'s mut Vec<half::f16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [half::f16], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
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
    scratch.resize(row, half::f16::ZERO);
  }
  Ok(&mut scratch[..row])
}

/// Freezes the output configuration for a resampled packed-float-RGB
/// frame — the full u8 / u16 / `rgb_f32` output set, plus the
/// `rgba_f32` / `rgb_f16` / `rgba_f16` outputs the planar-GBR
/// ([`Gbrpf32`](crate::source::Gbrpf32)) tail derives — and reports
/// whether any output is attached. Mirrors
/// [`packed_rgb_u16_resample_preflight`], extended with the lossless
/// `rgb_f32` / `rgba_f32` and the half-float `rgb_f16` / `rgba_f16`
/// channels. The [`Rgbf32`](crate::source::Rgbf32) caller passes `&None`
/// for `rgba_f32` / `rgb_f16` / `rgba_f16` (its tail emits none of them);
/// the `Gbrpf32` caller threads all three so every output its emit
/// writes participates in the freeze.
#[cfg(any(
  all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
  feature = "gbr"
))]
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
  rgba_f32: &Option<&mut [f32]>,
  rgb_f16: &Option<&mut [half::f16]>,
  rgba_f16: &Option<&mut [half::f16]>,
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
    rgba_f32,
    &None,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
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
      || rgba_f32.is_some()
      || rgb_f16.is_some()
      || rgba_f16.is_some()
      || hsv.is_some(),
  )
}

/// Lazily creates the 3-channel `f32` area stream and checks strict row
/// sequencing — run before the source conversion so an out-of-sequence
/// row is rejected without the staging work. Mirrors
/// [`packed_rgb_u16_resample_stream`] for the float element path.
#[cfg(any(
  all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")),
  feature = "gbr"
))]
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
  // The binned row is host-native f32 (the scatter decoded the source to
  // host order before binning), so the `rgbf32_*` kernels — which take a
  // wire-endian const and `load_f32` accordingly — must be told the data
  // is already host-order, else a big-endian target byte-swaps it and
  // corrupts every derived output.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
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
      crate::row::rgbf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
        binned,
        &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      crate::row::rgbf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
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
      crate::row::rgbf32_to_rgba_row::<HOST_NATIVE_BE>(
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
      crate::row::rgbf32_to_rgb_row::<HOST_NATIVE_BE>(binned, nrow, ow, use_simd);
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

/// Out-width **packed** `R, G, B` `half::f16` staging for the
/// [`Rgbf16`](crate::source::Rgbf16) arm of the float packed-RGB tail.
/// There is no `AreaStream<f16>`, so binning runs in `f32`; this emit
/// rounds each binned packed `f32` element to `half::f16` into this row
/// and runs the exact direct `rgbf16_*` kernels for every output. Unlike
/// the planar [`Gbrpf16`](crate::source::Gbrpf16) scratch this row stays
/// **packed** (no de-interleave). Grows `scratch` to `3 * width`
/// `half::f16` under the planner's recoverable-allocation contract. Only
/// the packed f16 emit consumes it.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
pub(super) fn rgb_packed_f16_scratch<'s>(
  scratch: &'s mut Vec<half::f16>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [half::f16], MixedSinkerError> {
  let row = width
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
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
    scratch.resize(row, half::f16::ZERO);
  }
  Ok(&mut scratch[..row])
}

/// Feeds the prepared source-width packed `R, G, B` `f32` row (the
/// [`Rgbf16`](crate::source::Rgbf16) wire widened f16 -> host-native f32)
/// into the float area stream and derives every attached output from each
/// finalized output row.
///
/// There is no `AreaStream<f16>`, so binning runs in `f32` for precision.
/// Per finalized output row this tail **rounds the binned packed `f32` row
/// to `half::f16`** ([`rgb_packed_f16_scratch`]) and runs the **exact
/// direct `rgbf16_*` kernels** over that packed f16 row. The result is
/// therefore byte-identical to a direct full-resolution `Rgbf16`
/// conversion of the frame whose per-pixel f16 `R, G, B` is the `f32` area
/// mean rounded to f16 (the parity oracle) — because the emit performs the
/// identical round-then-`rgbf16_*`. The f16-native kernels
/// (`rgbf16_to_rgb_f16_row` / `..._u16_row` / `..._rgba_u16_row` /
/// `..._row` / `..._rgba_row`) consume the rounded packed f16 row directly;
/// the lossless `rgb_f32` output — which the direct path derives by
/// *widening* the f16 source to f32 — is reproduced by widening the
/// **rounded** packed f16 row back to f32 (`rgbf16_to_rgb_f32_row`), so it
/// too matches the f16-rounded oracle, not the raw f32 bin. The u8 RGB /
/// luma / luma_u16 / hsv outputs stage through a u8 RGB narrowing of the
/// rounded packed f16 row (`rgbf16_to_rgb_row`, exactly the direct path's
/// scratch); `rgba` (u8) derives directly from the rounded packed f16 row
/// via `rgbf16_to_rgba_row`, mirroring the direct path.
///
/// The rounded packed f16 row holds **host-native** `half::f16` (rounded
/// from host-native binned f32), so every `rgbf16_*` kernel — which takes a
/// wire-endian const and byte-swaps when it differs from the host — is
/// invoked with `HOST_NATIVE_BE` to make its load a no-op on every host.
/// `packed_scratch_f16` (the rounded packed row) is sized — and its
/// allocation failure risked — only when an output is attached;
/// `narrow_scratch` (the u8 RGB row) only when an output that stages
/// through it (`rgb` / `luma` / `luma_u16` / `hsv`) is attached. So an
/// `rgb_f32`-only sink sizes only the packed f16 row, and a no-output sink
/// sizes neither.
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rgb_f16_resample_emit(
  stream: &mut crate::resample::AreaStream<f32>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_f32: &[f32],
  packed_scratch_f16: &mut Vec<half::f16>,
  narrow_scratch: &mut Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  // The rounded packed f16 row holds host-native data — the binned row was
  // decoded to host order during scatter, then rounded with `from_f32`,
  // which yields host-native `half::f16`. The `rgbf16_*` kernels take a
  // wire-endian const and byte-swap when it differs from the host, so pass
  // the host's own endianness to make every load a no-op; otherwise a
  // big-endian target would corrupt every output.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

  let ow = plan.out_w();
  // Every output derives from the rounded packed f16 row; even `rgb_f32`
  // does, because the direct `Rgbf16` path widens the f16 source to f32 (so
  // the oracle's `rgb_f32` is the f32 bin rounded to f16, then widened —
  // not the raw f32 bin). The predicate gates both the sizing here and the
  // round in the closure, so they cannot drift; a sink with no output sizes
  // nothing.
  let need_round = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgb_f16.is_some()
    || hsv.is_some();
  // The u8 RGB / luma / luma_u16 / hsv outputs stage through a u8 RGB
  // narrowing of the rounded packed f16 row (exactly the direct path's
  // `rgbf16_to_rgb_row` scratch); an f32-/f16-/native-u16-only sink never
  // touches it, so the out-width u8 scratch is sized — and its allocation
  // failure risked — only when one of those outputs is attached. `rgba`
  // (u8) derives directly from the rounded f16 row, so it does not need the
  // narrow row.
  let need_narrow = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  // Allocate both scratch rows up front (recoverable) before the stream's
  // closure writes any caller buffer, so an allocation refusal never leaves
  // a partially written output.
  let packed_f16: &mut [half::f16] = if need_round {
    rgb_packed_f16_scratch(packed_scratch_f16, ow, plan)?
  } else {
    &mut []
  };
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  stream.feed_row(idx, src_f32, use_simd, |oy, binned| {
    if need_round {
      // Round the binned packed `R, G, B` `f32` row to the packed f16 row
      // — the exact layout the direct `rgbf16_*` kernels consume, holding
      // the f32 block mean rounded to f16. (The f32-derived `rgb_f32`
      // output widens this rounded row back, never the raw bin.)
      let prow = &mut packed_f16[..3 * ow];
      for (dst, &src) in prow.iter_mut().zip(binned.iter()) {
        *dst = half::f16::from_f32(src);
      }
      let prow = &prow[..3 * ow];

      // ---- f16-native kernels: consume the rounded packed f16 row
      // directly, exactly as the direct `Rgbf16` path does ---------------
      if let Some(buf) = rgb_f16.as_deref_mut() {
        crate::row::rgbf16_to_rgb_f16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      // Lossless f32 widen of the **rounded** f16 row — the direct path
      // widens its f16 source to f32, so the oracle's `rgb_f32` is the
      // bin rounded to f16 then widened (NOT the raw f32 bin).
      if let Some(buf) = rgb_f32.as_deref_mut() {
        crate::row::rgbf16_to_rgb_f32_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_u16.as_deref_mut() {
        crate::row::rgbf16_to_rgb_u16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_u16.as_deref_mut() {
        crate::row::rgbf16_to_rgba_u16_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      // u8 RGBA — direct f16->u8 clamp+scale, alpha 0xFF (the direct path
      // emits RGBA straight from the f16 source, not via an expand of the
      // u8 RGB row).
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::rgbf16_to_rgba_row::<HOST_NATIVE_BE>(
          prow,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if need_narrow {
        let nrow = &mut narrow[..3 * ow];
        // Stage the u8 RGB row once via the direct path's f16->u8
        // clamp+scale; rgb / luma / luma_u16 / hsv all read it, matching
        // the direct Rgbf16 source-of-truth ordering exactly.
        crate::row::rgbf16_to_rgb_row::<HOST_NATIVE_BE>(prow, nrow, ow, use_simd);
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
    }
  })?;
  Ok(())
}

/// Feeds the prepared source-width packed `R, G, B` `f32` row (the
/// [`Gbrpf32`](crate::source::Gbrpf32) planes scattered into RGB order)
/// into the float area stream and derives every attached output from
/// each finalized output row. The `rgb-float` ([`Rgbf32`]) tail's
/// per-row `rgbf32_*` clamp/scale kernels are not compiled in a `gbr`
/// build, so this tail de-interleaves each binned packed row back into
/// G/B/R planes once and runs the **exact direct `gbrpf32_*` kernels** —
/// every output, `luma_u16` included, is therefore byte-identical to a
/// direct full-resolution `Gbrpf32` conversion of the binned frame (the
/// parity oracle). The binned row is host-native f32, so the kernels run
/// `::<false>`. `plane_scratch` holds the out-width G/B/R planes
/// (`[0..ow]` = G, `[ow..2ow]` = B, `[2ow..3ow]` = R); it is sized (and
/// its allocation failure risked) only when an output that reads the
/// planes is attached, so an `rgb_f32`-only sink neither grows it nor
/// risks its allocation. `rgb_f32` copies the binned row directly,
/// bypassing the planes. The lossless `rgba_f32` and the half-float
/// `rgb_f16` / `rgba_f16` outputs derive from the same de-interleaved
/// G/B/R planes via the direct `gbrpf32_to_rgba_f32_row` /
/// `gbrpf32_to_rgb_f16_row` / `gbrpf32_to_rgba_f16_row` kernels, so they
/// too are byte-identical to a direct `Gbrpf32` conversion of the binned
/// frame.
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
pub(super) fn planar_gbr_f32_resample_emit(
  stream: &mut crate::resample::AreaStream<f32>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_f32: &[f32],
  plane_scratch: &mut Vec<f32>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  // Every output but `rgb_f32` derives from the de-interleaved G/B/R
  // planes; an `rgb_f32`-only sink copies the binned row and never sizes
  // the plane scratch. The predicate gates both the sizing here and the
  // de-interleave in the closure, so they cannot drift. `rgba_f32` and
  // the f16 outputs run their direct `gbrpf32_*` kernels over the same
  // planes (byte-identical to the direct path), so they join the gate.
  let need_planes = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  let planes: &mut [f32] = if need_planes {
    rgb_plane_f32_scratch(plane_scratch, ow, plan)?
  } else {
    &mut []
  };
  stream.feed_row(idx, src_f32, use_simd, |oy, binned| {
    // Lossless float pass-through — copy the binned packed row verbatim
    // (the direct path's `gbrpf32_to_rgb_f32_row` over host-native data
    // is a plain interleave; the binned row is already that interleave).
    if let Some(buf) = rgb_f32.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(binned);
    }
    if need_planes {
      // De-interleave the binned packed `R, G, B` row into G/B/R planes
      // — the exact plane layout the direct `gbrpf32_*` kernels consume.
      let (g, rest) = planes.split_at_mut(ow);
      let (b, r) = rest.split_at_mut(ow);
      for x in 0..ow {
        r[x] = binned[x * 3];
        g[x] = binned[x * 3 + 1];
        b[x] = binned[x * 3 + 2];
      }
      let g = &g[..ow];
      let b = &b[..ow];
      let r = &r[..ow];
      // The de-interleaved planes hold host-native f32 (the binned row was
      // decoded to host order during scatter). The `gbrpf32_*` kernels take
      // a wire-endian const and byte-swap when it differs from the host, so
      // pass the host's own endianness to make the load a no-op — otherwise
      // a big-endian target would corrupt every plane-derived output.
      const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
      if let Some(buf) = rgb_u16.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_u16.as_deref_mut() {
        crate::row::gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      // Lossless packed `f32` RGBA — alpha forced to 1.0 (the direct
      // `gbrpf32_to_rgba_f32_row`, which the binned planes feed verbatim).
      if let Some(buf) = rgba_f32.as_deref_mut() {
        crate::row::gbrpf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      // f16 RGB / RGBA — fused f32->f16 narrow + interleave, exactly the
      // direct `gbrpf32_to_rgb_f16_row` / `gbrpf32_to_rgba_f16_row`.
      if let Some(buf) = rgb_f16.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_f16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_f16.as_deref_mut() {
        crate::row::gbrpf32_to_rgba_f16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb.as_deref_mut() {
        crate::row::gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        crate::row::gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
          g,
          b,
          r,
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

/// Feeds the prepared source-width packed `R, G, B` `f32` row (the
/// [`Gbrpf16`](crate::source::Gbrpf16) planes widened f16 -> host-native
/// f32 and scattered into RGB order) into the float area stream and
/// derives every attached output from each finalized output row.
///
/// There is no `AreaStream<f16>`, so binning runs in `f32` for
/// precision. Per finalized output row this tail de-interleaves the
/// binned packed row into `f32` G/B/R planes
/// ([`rgb_plane_f32_scratch`]), **rounds each element to `half::f16`**
/// (`half::f16::from_f32`) into the f16 planes ([`rgb_plane_f16_scratch`]:
/// `[0..ow]` = G, `[ow..2ow]` = B, `[2ow..3ow]` = R), then runs the
/// **exact direct `gbrpf16_*` kernels** over those f16 planes. The
/// result is therefore byte-identical to a direct full-resolution
/// `Gbrpf16` conversion of the frame whose per-pixel f16 G/B/R is the
/// `f32` area mean rounded to f16 (the parity oracle) — because the emit
/// performs the identical round-then-`gbrpf16_*`. The f16-native kernels
/// (`gbrpf16_to_rgb_f16_row` / `..._u16_row` / `..._row`) consume the
/// f16 planes directly; the outputs the direct path derives by widening
/// f16 -> f32 (`rgb_f32` / `rgba_f32` / `luma` / `luma_u16` / `hsv`) are
/// reproduced by widening the **rounded** f16 planes back to f32 (the
/// same lossless widen the direct path applies) and running the direct
/// `gbrpf32_*` kernels — so they too match the f16-rounded oracle, not
/// the raw f32 bin.
///
/// The rounded f16 planes hold **host-native** `half::f16` (rounded from
/// host-native binned f32), so every `gbrpf16_*` kernel — which takes a
/// wire-endian const and byte-swaps when it differs from the host — is
/// invoked with `HOST_NATIVE_BE` to make its load a no-op on every host;
/// the widen of those planes back to f32 likewise uses `HOST_NATIVE_BE`.
/// Only the f16 planes are staged: the f32-derived outputs must consume
/// the **rounded** values (widened back from f16), not the raw f32 bin,
/// so no f32 plane scratch is needed — the round writes directly from the
/// binned packed row. `plane_scratch_f16` is sized (and its allocation
/// failure risked) only when an output is attached, so a no-output sink
/// never grows it.
#[cfg(feature = "gbr")]
#[allow(clippy::too_many_arguments)]
pub(super) fn planar_gbr_f16_resample_emit(
  stream: &mut crate::resample::AreaStream<f32>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  rgba_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_f32: &[f32],
  plane_scratch_f16: &mut Vec<half::f16>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  use crate::row::scalar::planar_gbr_f16::widen_f16_be_to_host_f32;

  // The rounded f16 planes (and the f32 planes they round from) hold
  // host-native data — the binned row was decoded to host order during
  // scatter, then rounded with `from_f32`, which yields host-native
  // `half::f16`. The `gbrpf16_*` kernels take a wire-endian const and
  // byte-swap when it differs from the host, so pass the host's own
  // endianness to make every plane load a no-op; otherwise a big-endian
  // target would corrupt every output. The widen-back to f32 for the
  // f32-derived outputs uses the same const for the same reason.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  // Chunk size for the f16 -> f32 widen-back of the rounded planes, used
  // only by the outputs whose direct path widens f16 -> f32 (rgb_f32 /
  // rgba_f32 / luma / luma_u16 / hsv). Matches the dispatch layer's
  // widening chunk so the f32 staging is stack-resident.
  const WIDEN_CHUNK: usize = 64;

  let ow = plan.out_w();
  // Every output derives from the rounded f16 planes; even `rgb_f32`
  // does, because the direct `Gbrpf16` path widens the f16 source to f32
  // (so the oracle's `rgb_f32` is the f32 bin rounded to f16, then
  // widened — not the raw f32 bin). The predicate gates both the sizing
  // here and the de-interleave/round in the closure, so they cannot
  // drift; a sink with no output sizes nothing.
  let need_planes = rgb.is_some()
    || rgba.is_some()
    || luma.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || luma_u16.is_some()
    || rgb_f32.is_some()
    || rgba_f32.is_some()
    || rgb_f16.is_some()
    || rgba_f16.is_some()
    || hsv.is_some();
  // Allocate the f16 plane scratch up front (recoverable) before the
  // stream's closure writes any caller buffer, so an allocation refusal
  // never leaves a partially written output.
  let planes_f16: &mut [half::f16] = if need_planes {
    rgb_plane_f16_scratch(plane_scratch_f16, ow, plan)?
  } else {
    &mut []
  };
  stream.feed_row(idx, src_f32, use_simd, |oy, binned| {
    if need_planes {
      // De-interleave the binned packed `R, G, B` row directly into the
      // G/B/R f16 planes, **rounding** each element to `half::f16` — the
      // exact plane layout the direct `gbrpf16_*` kernels consume,
      // holding the f32 block mean rounded to f16. (No f32 plane stage:
      // the f32-derived outputs must consume the rounded values, so they
      // widen these f16 planes back, never the raw bin.)
      let (g16, rest_f16) = planes_f16.split_at_mut(ow);
      let (b16, r16) = rest_f16.split_at_mut(ow);
      for x in 0..ow {
        r16[x] = half::f16::from_f32(binned[x * 3]);
        g16[x] = half::f16::from_f32(binned[x * 3 + 1]);
        b16[x] = half::f16::from_f32(binned[x * 3 + 2]);
      }
      let g16 = &g16[..ow];
      let b16 = &b16[..ow];
      let r16 = &r16[..ow];

      // ---- f16-native kernels: consume the rounded f16 planes directly,
      // exactly as the direct `Gbrpf16` path does ------------------------
      if let Some(buf) = rgb_f16.as_deref_mut() {
        crate::row::gbrpf16_to_rgb_f16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_f16.as_deref_mut() {
        crate::row::gbrpf16_to_rgba_f16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb_u16.as_deref_mut() {
        crate::row::gbrpf16_to_rgb_u16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba_u16.as_deref_mut() {
        crate::row::gbrpf16_to_rgba_u16_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgb.as_deref_mut() {
        crate::row::gbrpf16_to_rgb_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        crate::row::gbrpf16_to_rgba_row::<HOST_NATIVE_BE>(
          g16,
          b16,
          r16,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
        );
      }

      // ---- f32-derived outputs: the direct `Gbrpf16` path widens the
      // f16 source planes to f32 and runs the `gbrpf32_*` kernels, so
      // reproduce that exactly by widening the **rounded** f16 planes
      // back to f32 (chunked, stack-resident) and running the same
      // kernels — byte-identical to the f16-rounded oracle ---------------
      let need_wide_back = rgb_f32.is_some()
        || rgba_f32.is_some()
        || luma.is_some()
        || luma_u16.is_some()
        || hsv.is_some();
      if need_wide_back {
        let mut gw = [0.0f32; WIDEN_CHUNK];
        let mut bw = [0.0f32; WIDEN_CHUNK];
        let mut rw = [0.0f32; WIDEN_CHUNK];
        let mut off = 0;
        while off < ow {
          let n = (ow - off).min(WIDEN_CHUNK);
          // The rounded f16 planes are host-native; widen with the
          // host's own endianness so the bit-normalize is a no-op, then
          // run the `gbrpf32_*` kernels with `HOST_NATIVE_BE` (the same
          // post-widen routing the direct path uses).
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(g16, off, &mut gw, n);
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(b16, off, &mut bw, n);
          widen_f16_be_to_host_f32::<HOST_NATIVE_BE>(r16, off, &mut rw, n);
          let gwn = &gw[..n];
          let bwn = &bw[..n];
          let rwn = &rw[..n];
          let cps = oy * ow + off;
          let cpe = cps + n;
          if let Some(buf) = rgb_f32.as_deref_mut() {
            crate::row::gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps * 3..cpe * 3],
              n,
              use_simd,
            );
          }
          if let Some(buf) = rgba_f32.as_deref_mut() {
            crate::row::gbrpf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps * 4..cpe * 4],
              n,
              use_simd,
            );
          }
          if let Some(buf) = luma.as_deref_mut() {
            crate::row::gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps..cpe],
              n,
              matrix,
              full_range,
              use_simd,
            );
          }
          if let Some(buf) = luma_u16.as_deref_mut() {
            crate::row::gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut buf[cps..cpe],
              n,
              matrix,
              full_range,
              use_simd,
            );
          }
          if let Some(hsv) = hsv.as_mut() {
            let (h, s, v) = hsv.hsv();
            crate::row::gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
              gwn,
              bwn,
              rwn,
              &mut h[cps..cpe],
              &mut s[cps..cpe],
              &mut v[cps..cpe],
              n,
              use_simd,
            );
          }
          off += n;
        }
      }
    }
  })?;
  Ok(())
}

// ---- Xyz12 (linear-light area mean) resample tail ----------------------
//
// The `Xyz12` source decodes a 2.6-gamma-encoded CIE-XYZ wire sample
// through SMPTE ST 428-1 §8 inverse-OETF -> linear XYZ -> 3x3 gamut
// matrix -> sRGB OETF -> narrow. Area-resampling must average in LINEAR
// light, so the wire row is converted to **linear XYZ** (`xyz12_to_
// xyz_f32_row`, post-OETF / pre-matrix), binned in float, and the
// non-linear tail (matrix + gamma + clamp/scale) is applied per
// finalized output row. Because the bin is a linear combination and the
// matrix is linear, `M . mean(xyz) == mean(M . xyz)` exactly — the
// matrix commutes with the bin — so every derived output equals the
// direct DCP pipeline applied to a frame whose per-pixel linear XYZ is
// the area mean of the source linear XYZ (the linear-light oracle). The
// OETF / narrow are per-pixel and run AFTER the matrix, exactly as the
// direct path does, so they need not commute with the bin.

/// Source-width **linear-XYZ** `f32` staging for the `Xyz12` resample
/// path: the wire row converts here (inverse-OETF only, no matrix)
/// before feeding [`AreaStream<f32>`]. Grows `scratch` to `3 * width`
/// `f32` under the planner's recoverable-allocation contract. Mirrors
/// [`source_rgb_f32_scratch`] for the linear-XYZ element path.
#[cfg(feature = "xyz")]
pub(super) fn source_xyz_f32_scratch<'s>(
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

/// Freezes the output configuration for a resampled `Xyz12` frame — the
/// full `Xyz12` output set — and reports whether any output is attached.
/// Mirrors [`packed_rgb_f32_resample_preflight`], with the lossless
/// `xyz_f32` channel added (and the `rgb_f32` slot reused for the
/// linear-RGB output). The `rgb_f16` / `rgba_f16` outputs are not
/// identity-tracked by [`FrozenOutputs`] (it carries no f16 slot), but
/// the emit still derives them, so they participate in the
/// "any output attached" predicate that keeps a no-output sink a no-op.
#[cfg(feature = "xyz")]
#[allow(clippy::too_many_arguments)]
pub(super) fn xyz12_resample_preflight(
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  rgb_u16: &Option<&mut [u16]>,
  rgba_u16: &Option<&mut [u16]>,
  rgb_f32: &Option<&mut [f32]>,
  xyz_f32: &Option<&mut [f32]>,
  rgb_f16: &Option<&mut [half::f16]>,
  rgba_f16: &Option<&mut [half::f16]>,
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
    &None,
    xyz_f32,
    rgb_f16,
    rgba_f16,
    hsv,
    &None,
    idx,
  )?;
  Ok(
    rgb.is_some()
      || rgba.is_some()
      || luma.is_some()
      || luma_u16.is_some()
      || rgb_u16.is_some()
      || rgba_u16.is_some()
      || rgb_f32.is_some()
      || xyz_f32.is_some()
      || rgb_f16.is_some()
      || rgba_f16.is_some()
      || hsv.is_some(),
  )
}

/// Lazily creates the 3-channel linear-XYZ `f32` area stream and checks
/// strict row sequencing — run before the source conversion so an
/// out-of-sequence row is rejected without the staging work. Mirrors
/// [`packed_rgb_f32_resample_stream`] for the `Xyz12` path.
#[cfg(feature = "xyz")]
pub(super) fn xyz12_resample_stream<'s>(
  xyz_stream_f32: &'s mut Option<crate::resample::AreaStream<f32>>,
  plan: &ResamplePlan,
  idx: usize,
) -> Result<&'s mut crate::resample::AreaStream<f32>, MixedSinkerError> {
  // Sequence-check before allocating (see packed_rgb_f32_resample_stream):
  // an out-of-sequence first row is rejected without creating the f32
  // output-width buffers, so AllocationFailed never masks
  // OutOfSequenceRow.
  let expected = xyz_stream_f32.as_ref().map_or(0, |stream| stream.next_y());
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      crate::resample::OutOfSequenceRow::new(expected, idx),
    )));
  }
  let stream = match xyz_stream_f32 {
    Some(stream) => stream,
    None => xyz_stream_f32.insert(crate::resample::AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      3,
    )?),
  };
  Ok(stream)
}

/// Feeds the prepared source-width **linear-XYZ** `f32` row into the
/// (already sequence-checked) stream and derives every attached output
/// from each finalized binned linear-XYZ output row.
///
/// `xyz_f32` copies the binned linear XYZ verbatim. Every other output
/// applies the direct DCP path's math to the binned XYZ: the gamut
/// matrix yields linear RGB (`rgb_f32`); the sRGB OETF + clamp/scale +
/// narrow yield the integer / f16 outputs (`rgb` / `rgba` / `rgb_u16` /
/// `rgba_u16` / `rgb_f16` / `rgba_f16`); and `luma` / `luma_u16` / `hsv`
/// derive from the staged u8 RGB row — exactly the direct path's
/// source-of-truth ordering. The u8 RGB staging row is sized only when
/// one of the outputs that reads it (`rgb` / `rgba` / `luma` /
/// `luma_u16` / `hsv`) is attached, so an f32-/f16-only sink neither
/// grows it nor risks its allocation failure.
///
/// `target_gamut` selects the XYZ -> RGB matrix; `luma_q15` carries the
/// gamut-matched Q15 luma weights (both ride [`Xyz12Row`] on the direct
/// path). These bind the entire frame via [`FrozenOutputs`] +
/// `xyz12_to`'s per-frame constants, so they cannot drift mid-frame.
#[cfg(feature = "xyz")]
#[allow(clippy::too_many_arguments)]
pub(super) fn xyz12_resample_emit(
  stream: &mut crate::resample::AreaStream<f32>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  rgb_f32: &mut Option<&mut [f32]>,
  xyz_f32: &mut Option<&mut [f32]>,
  rgb_f16: &mut Option<&mut [half::f16]>,
  rgba_f16: &mut Option<&mut [half::f16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_xyz: &[f32],
  narrow_scratch: &mut Vec<u8>,
  target_gamut: crate::DcpTargetGamut,
  luma_q15: (i32, i32, i32),
  idx: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  use crate::row::scalar::xyz12::{
    matmul3_xyz_rgb, narrow_unit_to_u8, narrow_unit_to_u16, oetf_srgb,
  };
  use crate::row::scalar::xyz12_constants::xyz_to_rgb_matrix;
  let ow = plan.out_w();
  let m = xyz_to_rgb_matrix(target_gamut);
  let one_f16 = half::f16::from_f32(1.0);
  // The u8 RGB / luma / luma_u16 / hsv outputs stage through a u8 RGB
  // narrowing of the binned linear XYZ (matrix + OETF + clamp/x255);
  // an f32-/f16-/native-u16-only sink never touches it, so the
  // out-width u8 scratch is sized — and its allocation failure risked —
  // only when one of those outputs is attached. The predicate gates
  // both the sizing here and the use in the closure, so they cannot
  // drift. `rgba` (u8) derives directly from the binned XYZ (matrix +
  // OETF + narrow + alpha, exactly the direct `xyz12_to_rgba_row`), so
  // it does not need the narrow row.
  let need_narrow = rgb.is_some() || luma.is_some() || luma_u16.is_some() || hsv.is_some();
  let narrow: &mut [u8] = if need_narrow {
    source_rgb_scratch(narrow_scratch, ow, plan)?
  } else {
    &mut []
  };
  stream.feed_row(idx, src_xyz, use_simd, |oy, binned| {
    // Lossless linear-XYZ pass-through — copy the binned row verbatim.
    if let Some(buf) = xyz_f32.as_deref_mut() {
      buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(binned);
    }
    // Linear RGB (matrix only, no OETF / clamp) — out-of-gamut negatives
    // and HDR > 1 preserved bit-exact, mirroring `xyz12_to_rgb_f32_row`
    // but over the already-inverse-OETF'd binned XYZ.
    if let Some(buf) = rgb_f32.as_deref_mut() {
      let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      for x in 0..ow {
        let i = x * 3;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[i] = rgb_lin[0];
        dst[i + 1] = rgb_lin[1];
        dst[i + 2] = rgb_lin[2];
      }
    }
    // f16 RGB / RGBA — matrix + OETF + clamp [0, 1] + IEEE-754 RNE
    // narrow, exactly as `xyz12_to_rgb_f16_row` / `xyz12_to_rgba_f16_row`.
    if let Some(buf) = rgb_f16.as_deref_mut() {
      let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      for x in 0..ow {
        let i = x * 3;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[i] = half::f16::from_f32(oetf_srgb(rgb_lin[0]).clamp(0.0, 1.0));
        dst[i + 1] = half::f16::from_f32(oetf_srgb(rgb_lin[1]).clamp(0.0, 1.0));
        dst[i + 2] = half::f16::from_f32(oetf_srgb(rgb_lin[2]).clamp(0.0, 1.0));
      }
    }
    if let Some(buf) = rgba_f16.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      for x in 0..ow {
        let i = x * 3;
        let o = x * 4;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[o] = half::f16::from_f32(oetf_srgb(rgb_lin[0]).clamp(0.0, 1.0));
        dst[o + 1] = half::f16::from_f32(oetf_srgb(rgb_lin[1]).clamp(0.0, 1.0));
        dst[o + 2] = half::f16::from_f32(oetf_srgb(rgb_lin[2]).clamp(0.0, 1.0));
        dst[o + 3] = one_f16;
      }
    }
    // u16 RGB / RGBA — matrix + OETF + clamp + x65535 + round-half-up,
    // exactly as `xyz12_to_rgb_u16_row` / `xyz12_to_rgba_u16_row`.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let dst = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      for x in 0..ow {
        let i = x * 3;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[i] = narrow_unit_to_u16(oetf_srgb(rgb_lin[0]));
        dst[i + 1] = narrow_unit_to_u16(oetf_srgb(rgb_lin[1]));
        dst[i + 2] = narrow_unit_to_u16(oetf_srgb(rgb_lin[2]));
      }
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      for x in 0..ow {
        let i = x * 3;
        let o = x * 4;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[o] = narrow_unit_to_u16(oetf_srgb(rgb_lin[0]));
        dst[o + 1] = narrow_unit_to_u16(oetf_srgb(rgb_lin[1]));
        dst[o + 2] = narrow_unit_to_u16(oetf_srgb(rgb_lin[2]));
        dst[o + 3] = 0xFFFF;
      }
    }
    // u8 RGBA — matrix + OETF + clamp + x255 + round-half-up, alpha
    // 0xFF (exactly `xyz12_to_rgba_row`), derived directly from the
    // binned XYZ rather than expanded from the staged u8 RGB row.
    if let Some(buf) = rgba.as_deref_mut() {
      let dst = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
      for x in 0..ow {
        let i = x * 3;
        let o = x * 4;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        dst[o] = narrow_unit_to_u8(oetf_srgb(rgb_lin[0]));
        dst[o + 1] = narrow_unit_to_u8(oetf_srgb(rgb_lin[1]));
        dst[o + 2] = narrow_unit_to_u8(oetf_srgb(rgb_lin[2]));
        dst[o + 3] = 0xFF;
      }
    }
    if need_narrow {
      let nrow = &mut narrow[..3 * ow];
      // Stage the u8 RGB row once via the direct path's matrix + OETF +
      // clamp + x255; rgb / rgba / luma / luma_u16 / hsv all read it.
      for x in 0..ow {
        let i = x * 3;
        let rgb_lin = matmul3_xyz_rgb(&m, [binned[i], binned[i + 1], binned[i + 2]]);
        nrow[i] = narrow_unit_to_u8(oetf_srgb(rgb_lin[0]));
        nrow[i + 1] = narrow_unit_to_u8(oetf_srgb(rgb_lin[1]));
        nrow[i + 2] = narrow_unit_to_u8(oetf_srgb(rgb_lin[2]));
      }
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(nrow);
      }
      if let Some(buf) = luma.as_deref_mut() {
        crate::row::xyz12_rgb_to_luma_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          luma_q15,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        crate::row::xyz12_rgb_to_luma_u16_row(
          nrow,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          luma_q15,
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
/// Format-agnostic planar-YUV resample helpers (the 4:2:0 native /
/// row-stage join and the shared row-stage path), reused by the 8-bit
/// planar family, the semi-planar family, and the packed YUV family
/// (which de-interleaves Y into a scratch first via
/// [`planar_resample::packed_yuv_dual_resample`]).
#[cfg(any(
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuv-packed"
))]
mod planar_resample;
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
