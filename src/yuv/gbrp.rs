//! Planar GBR 8-bit (`AV_PIX_FMT_GBRP`) — three full-resolution `u8`
//! planes in **G, B, R** order (FFmpeg convention).
//!
//! Unlike every YUV source in this crate, the input is already
//! component RGB — there's no chroma matrix work. The walker follows
//! the same `planar3` shape used by [`super::Yuv444p`] (full-width
//! planes, no chroma subsampling, no width parity constraint), but the
//! per-row kernels reorder G/B/R into packed RGB rather than running
//! the YUV → RGB matrix.
//!
//! # API naming
//!
//! The public row API exposes only the externally-correct `g()` / `b()`
//! / `r()` accessors. The internal `y()` / `u()` / `v()` aliases (used
//! by the walker function body and sinker) are `pub(crate)` only, so
//! they do not appear in the public API surface.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::GbrpFrame, sealed::Sealed};

/// Zero-sized marker for the planar GBR 8-bit source format
/// (`AV_PIX_FMT_GBRP`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gbrp;

impl Sealed for Gbrp {}
impl SourceFormat for Gbrp {}

/// One output row of a [`Gbrp`] source — three full-width planes in
/// G / B / R order. Use the [`Self::g`] / [`Self::b`] / [`Self::r`]
/// accessors.
#[derive(Debug, Clone, Copy)]
pub struct GbrpRow<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> GbrpRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      u,
      v,
      row,
      matrix,
      full_range,
    }
  }

  /// Green plane row — full width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn g(&self) -> &'a [u8] {
    self.y
  }
  /// Blue plane row — full width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn b(&self) -> &'a [u8] {
    self.u
  }
  /// Red plane row — full width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn r(&self) -> &'a [u8] {
    self.v
  }

  /// Output row index within the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
  /// YUV/RGB conversion matrix carried through from the kernel call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }
  /// Full-range vs limited-range flag carried through from the kernel call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }
}

/// Sinks that consume rows of this source format.
pub trait GbrpSink: for<'a> PixelSink<Input<'a> = GbrpRow<'a>> {}

/// Walks a [`GbrpFrame`] row by row into the sink.
pub fn gbrp_to<S: GbrpSink>(
  src: &GbrpFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let y_stride = src.y_stride() as usize;
  let u_stride = src.u_stride() as usize;
  let v_stride = src.v_stride() as usize;

  let y_plane = src.y();
  let u_plane = src.u();
  let v_plane = src.v();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let u = &u_plane[u_start..u_start + w];
    let v = &v_plane[v_start..v_start + w];

    sink.process(GbrpRow::new(y, u, v, row, matrix, full_range))?;
  }
  Ok(())
}
