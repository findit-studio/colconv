//! [`MonowhiteFrame`] walker — 1-bit-per-pixel, MSB-first encoding,
//! bit=0 → white (Y=255), bit=1 → black (Y=0). Inverted polarity from
//! Monoblack.

use crate::{ColorMatrix, PixelSink, frame::MonowhiteFrame};

/// Marker type for the `Monowhite` source format (FFmpeg
/// `AV_PIX_FMT_MONOWHITE`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Monowhite;

impl crate::sealed::Sealed for Monowhite {}
impl crate::SourceFormat for Monowhite {}

/// A single row from a [`MonowhiteFrame`] — byte buffer (8 pixels per
/// byte, MSB first, inverted polarity).
#[derive(Debug, Clone, Copy)]
pub struct MonowhiteRow<'a> {
  data: &'a [u8],
  row: usize,
  matrix: ColorMatrix,
  full_range: bool,
}

impl<'a> MonowhiteRow<'a> {
  /// Constructs a new row slice.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(data: &'a [u8], row: usize, matrix: ColorMatrix, full_range: bool) -> Self {
    Self {
      data,
      row,
      matrix,
      full_range,
    }
  }

  /// Byte data for this row.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn data(&self) -> &'a [u8] {
    self.data
  }

  /// Output row index within the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Color matrix carried through from the kernel call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn matrix(&self) -> ColorMatrix {
    self.matrix
  }

  /// Full-range flag carried through from the kernel call.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    self.full_range
  }

  /// Frame width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn len(&self) -> usize {
    self.data.len() * 8
  }

  /// True if the row is empty.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn is_empty(&self) -> bool {
    self.data.is_empty()
  }
}

/// Sinks that consume rows of the Monowhite source format.
pub trait MonowhiteSink: for<'a> PixelSink<Input<'a> = MonowhiteRow<'a>> {}

/// Walks a [`MonowhiteFrame`] row by row, dispatching each row to the
/// sink.
pub fn monowhite_to<S: MonowhiteSink>(
  src: &MonowhiteFrame<'_>,
  full_range: bool,
  matrix: ColorMatrix,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let data = src.data();

  for row in 0..h {
    let start = row * stride;
    let end = start + stride.min(data.len() - start);
    let row_data = &data[start..end];
    sink.process(MonowhiteRow::new(row_data, row, matrix, full_range))?;
  }
  Ok(())
}
