//! Planar GBR + A 8-bit (`AV_PIX_FMT_GBRAP`) — four full-resolution
//! `u8` planes in **G, B, R, A** order.
//!
//! Same structure as [`super::Gbrp`] with an additional alpha plane
//! (1:1 with the colour planes — real per-pixel α, not padding).
//!
//! # API naming
//!
//! The public row API exposes only the externally-correct `g()` / `b()`
//! / `r()` / `a()` accessors. The internal `y()` / `u()` / `v()` aliases
//! (used by the walker function body) are `pub(crate)` only, so they do
//! not appear in the public API surface.

use crate::{ColorMatrix, PixelSink, SourceFormat, frame::GbrapFrame, sealed::Sealed};

/// Zero-sized marker for the planar GBRAP 8-bit source format
/// (`AV_PIX_FMT_GBRAP`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gbrap;

impl Sealed for Gbrap {}
impl SourceFormat for Gbrap {}

/// One output row of a [`Gbrap`] source — four full-width planes in
/// G / B / R / A order. Alpha is real (not padding) and is passed
/// through to RGBA output. Use the [`Self::g`] / [`Self::b`] /
/// [`Self::r`] / [`Self::a`] accessors.
#[derive(Debug, Clone, Copy)]
pub struct GbrapRow<'a> {
  y: &'a [u8],
  u: &'a [u8],
  v: &'a [u8],
  a: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> GbrapRow<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn new(
    y: &'a [u8],
    u: &'a [u8],
    v: &'a [u8],
    a: &'a [u8],
    row: usize,
    matrix: ColorMatrix,
    full_range: bool,
  ) -> Self {
    Self {
      y,
      u,
      v,
      a,
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
  /// Alpha plane row — full width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn a(&self) -> &'a [u8] {
    self.a
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
pub trait GbrapSink: for<'a> PixelSink<Input<'a> = GbrapRow<'a>> {}

/// Walks a [`GbrapFrame`] row by row into the sink.
pub fn gbrap_to<S: GbrapSink>(
  src: &GbrapFrame<'_>,
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
  let a_stride = src.a_stride() as usize;

  let y_plane = src.y();
  let u_plane = src.u();
  let v_plane = src.v();
  let a_plane = src.a();

  for row in 0..h {
    let y_start = row * y_stride;
    let y = &y_plane[y_start..y_start + w];

    let u_start = row * u_stride;
    let v_start = row * v_stride;
    let u = &u_plane[u_start..u_start + w];
    let v = &v_plane[v_start..v_start + w];

    let a_start = row * a_stride;
    let a = &a_plane[a_start..a_start + w];

    sink.process(GbrapRow::new(y, u, v, a, row, matrix, full_range))?;
  }
  Ok(())
}
