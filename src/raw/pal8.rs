//! 8-bit indexed-color (`AV_PIX_FMT_PAL8`) — single-plane mosaic with a
//! 256-entry BGRA palette.
//!
//! Walker hands each output row to a [`Pal8Sink`] together with the
//! palette so the sink can perform one lookup per pixel without
//! storing the palette on the sinker itself.

use crate::{PixelSink, SourceFormat, frame::Pal8Frame};

/// Zero-sized marker for the 8-bit indexed-color (`AV_PIX_FMT_PAL8`) source
/// format. Used as the `F` type parameter on
/// [`crate::sinker::MixedSinker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Pal8;

impl SourceFormat for Pal8 {}

/// One output row of a `PAL8` source handed to a [`Pal8Sink`].
///
/// Carries the pixel-index slice for this row, the 256-entry BGRA palette
/// shared across all rows, and the row index.
#[derive(Debug, Clone, Copy)]
pub struct Pal8Row<'a> {
  row: &'a [u8],
  palette: &'a [[u8; 4]; 256],
  idx: usize,
}

impl<'a> Pal8Row<'a> {
  /// Constructs a [`Pal8Row`].
  pub(crate) fn new(row: &'a [u8], palette: &'a [[u8; 4]; 256], idx: usize) -> Self {
    Self { row, palette, idx }
  }

  /// The pixel-index slice for this row. Length equals the frame width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn row(&self) -> &'a [u8] {
    self.row
  }

  /// The 256-entry BGRA palette shared across all rows of the frame.
  /// Each entry is `[B, G, R, A]` per FFmpeg's `AV_PIX_FMT_PAL8` convention.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn palette(&self) -> &'a [[u8; 4]; 256] {
    self.palette
  }

  /// Row index within the frame (0-based).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn idx(&self) -> usize {
    self.idx
  }
}

/// Sinks that consume 8-bit indexed-color (`PAL8`) rows.
///
/// A subtrait of [`PixelSink`] that pins the row shape to [`Pal8Row`].
pub trait Pal8Sink: for<'a> PixelSink<Input<'a> = Pal8Row<'a>> {}

/// Walks a [`Pal8Frame`] row by row, handing each row to the sink along with
/// the palette.
///
/// For each row, slices `data[row * stride .. row * stride + width]` from the
/// source plane and constructs a [`Pal8Row`] with the frame's palette. The
/// sink receives one row at a time and performs the palette lookup.
///
/// **Allocation profile.** Zero per-row and zero per-frame heap allocation.
/// The walker slices a row borrow into the source plane and hands it to the
/// sink. The sink owns the output buffer.
pub fn pal8_to<S: Pal8Sink>(src: &Pal8Frame<'_>, sink: &mut S) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;
  let w = src.width() as usize;
  let stride = src.stride() as usize;
  let h = src.height() as usize;
  let data = src.data();
  let palette = src.palette();
  for row in 0..h {
    let start = row * stride;
    let row_slice = &data[start..start + w];
    sink.process(Pal8Row::new(row_slice, palette, row))?;
  }
  Ok(())
}
