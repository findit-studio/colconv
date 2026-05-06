//! Walker for the `Gbrapf32` source format (`AV_PIX_FMT_GBRAPF32LE`) — four
//! full-resolution `f32` planes in **G, B, R, A** order.
//!
//! Alpha is real per-pixel; nominal range `[0.0, 1.0]` (opaque = 1.0).
//! Integer outputs clamp colour channels to `[0.0, 1.0]` before scaling;
//! float outputs are lossless pass-through.

use crate::{PixelSink, SourceFormat, frame::Gbrapf32Frame, sealed::Sealed};

/// Zero-sized marker for the planar GBRAP float-32 source format
/// (`AV_PIX_FMT_GBRAPF32LE`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gbrapf32;

impl Sealed for Gbrapf32 {}
impl SourceFormat for Gbrapf32 {}

/// One output row from a [`Gbrapf32Frame`] — four full-width `f32` slices
/// in G / B / R / A order. Use [`Self::g`] / [`Self::b`] / [`Self::r`] /
/// [`Self::a`].
#[derive(Debug, Clone, Copy)]
pub struct Gbrapf32Row<'a> {
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  a: &'a [f32],
  row: usize,
}

impl<'a> Gbrapf32Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(g: &'a [f32], b: &'a [f32], r: &'a [f32], a: &'a [f32], row: usize) -> Self {
    Self { g, b, r, a, row }
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
  /// Alpha plane row — `width` `f32` elements (opaque = 1.0).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn a(&self) -> &'a [f32] {
    self.a
  }
  /// Output row index within the frame (0-based).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Sinks that consume rows of a [`Gbrapf32`] source.
pub trait Gbrapf32Sink: for<'a> PixelSink<Input<'a> = Gbrapf32Row<'a>> {}

/// Walks a [`Gbrapf32Frame`] row by row, dispatching each row to the sink.
pub fn gbrapf32_to<S: Gbrapf32Sink>(src: &Gbrapf32Frame<'_>, sink: &mut S) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let g_plane = src.g();
  let b_plane = src.b();
  let r_plane = src.r();
  let a_plane = src.a();
  let g_stride = src.g_stride() as usize;
  let b_stride = src.b_stride() as usize;
  let r_stride = src.r_stride() as usize;
  let a_stride = src.a_stride() as usize;

  for row in 0..h {
    let g = &g_plane[row * g_stride..row * g_stride + w];
    let b = &b_plane[row * b_stride..row * b_stride + w];
    let r = &r_plane[row * r_stride..row * r_stride + w];
    let a = &a_plane[row * a_stride..row * a_stride + w];
    sink.process(Gbrapf32Row::new(g, b, r, a, row))?;
  }
  Ok(())
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{PixelSink, frame::Gbrapf32Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_a_len: usize,
    last_row_idx: usize,
  }

  impl PixelSink for CountingSink {
    type Input<'r> = Gbrapf32Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Gbrapf32Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_a_len = row.a().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }

  impl Gbrapf32Sink for CountingSink {}

  #[test]
  fn gbrapf32_walker_visits_every_row_once() {
    let buf = std::vec![1.0f32; 4 * 4];
    let frame = Gbrapf32Frame::try_new(&buf, &buf, &buf, &buf, 4, 4, 4, 4, 4, 4).unwrap();
    let mut sink = CountingSink {
      rows_seen: 0,
      last_a_len: 0,
      last_row_idx: 0,
    };
    gbrapf32_to(&frame, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_a_len, 4);
    assert_eq!(sink.last_row_idx, 3);
  }
}
