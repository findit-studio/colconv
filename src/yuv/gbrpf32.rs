//! Walker for the `Gbrpf32` source format (`AV_PIX_FMT_GBRPF32LE`) — three
//! full-resolution `f32` planes in **G, B, R** order.
//!
//! Nominal range `[0.0, 1.0]`; HDR values > 1.0 are permitted. Integer
//! outputs clamp to `[0.0, 1.0]` before scaling; float outputs are
//! lossless pass-through.

use crate::{PixelSink, SourceFormat, frame::Gbrpf32Frame, sealed::Sealed};

/// Zero-sized marker for the planar GBR float-32 source format
/// (`AV_PIX_FMT_GBRPF32LE`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gbrpf32;

impl Sealed for Gbrpf32 {}
impl SourceFormat for Gbrpf32 {}

/// One output row from a [`Gbrpf32Frame`] — three full-width `f32` slices
/// in G / B / R order. Use [`Self::g`] / [`Self::b`] / [`Self::r`].
#[derive(Debug, Clone, Copy)]
pub struct Gbrpf32Row<'a> {
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  row: usize,
}

impl<'a> Gbrpf32Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(g: &'a [f32], b: &'a [f32], r: &'a [f32], row: usize) -> Self {
    Self { g, b, r, row }
  }

  /// Green plane row — `width` `f32` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn g(&self) -> &'a [f32] {
    self.g
  }
  /// Blue plane row — `width` `f32` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn b(&self) -> &'a [f32] {
    self.b
  }
  /// Red plane row — `width` `f32` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn r(&self) -> &'a [f32] {
    self.r
  }
  /// Output row index within the frame (0-based).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Sinks that consume rows of a [`Gbrpf32`] source.
pub trait Gbrpf32Sink: for<'a> PixelSink<Input<'a> = Gbrpf32Row<'a>> {}

/// Walks a [`Gbrpf32Frame`] row by row, dispatching each row to the sink.
pub fn gbrpf32_to<S: Gbrpf32Sink>(src: &Gbrpf32Frame<'_>, sink: &mut S) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let g_plane = src.g();
  let b_plane = src.b();
  let r_plane = src.r();
  let g_stride = src.g_stride() as usize;
  let b_stride = src.b_stride() as usize;
  let r_stride = src.r_stride() as usize;

  for row in 0..h {
    let g = &g_plane[row * g_stride..row * g_stride + w];
    let b = &b_plane[row * b_stride..row * b_stride + w];
    let r = &r_plane[row * r_stride..row * r_stride + w];
    sink.process(Gbrpf32Row::new(g, b, r, row))?;
  }
  Ok(())
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{PixelSink, frame::Gbrpf32Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_g_len: usize,
    last_row_idx: usize,
  }

  impl PixelSink for CountingSink {
    type Input<'r> = Gbrpf32Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Gbrpf32Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_g_len = row.g().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }

  impl Gbrpf32Sink for CountingSink {}

  #[test]
  fn gbrpf32_walker_visits_every_row_once() {
    // 4 px × 4 rows, tight stride
    let buf = std::vec![0.5f32; 4 * 4];
    let frame = Gbrpf32Frame::try_new(&buf, &buf, &buf, 4, 4, 4, 4, 4).unwrap();
    let mut sink = CountingSink {
      rows_seen: 0,
      last_g_len: 0,
      last_row_idx: 0,
    };
    gbrpf32_to(&frame, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_g_len, 4);
    assert_eq!(sink.last_row_idx, 3);
  }
}
