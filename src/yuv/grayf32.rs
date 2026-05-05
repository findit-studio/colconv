//! Walker spec for the `Grayf32` source format (FFmpeg `grayf32le`).
//!
//! Single `f32` luma plane. Nominal range `[0.0, 1.0]`; HDR > 1.0 is permitted.
//! Stride is in f32 elements. No chroma planes exist.

use crate::frame::Grayf32Frame;

walker! {
  planar1 {
    /// Marker type for the `Grayf32` source format (32-bit float luma).
    ///
    /// Nominal luma range `[0.0, 1.0]`; HDR values > 1.0 are permitted.
    /// Out-of-range values are clamped during output conversion, not at frame
    /// construction time.
    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    marker: Grayf32,
    frame: Grayf32Frame<'_>,
    row: Grayf32Row,
    sink: Grayf32Sink,
    walker: grayf32_to,
    elem_type: f32,
    row_doc: "A single row from a [`Grayf32Frame`] — `width` f32 luma samples.",
    walker_doc: "Walks a [`Grayf32Frame`] row by row, dispatching each row to the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Grayf32Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_y_len: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Grayf32Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Grayf32Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_y_len = row.y().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Grayf32Sink for CountingSink {}

  #[test]
  fn grayf32_walker_visits_every_row_once() {
    // 4 px × 4 rows = 16 f32 elements (tight stride)
    let buf = std::vec![0.5f32; 16];
    let frame = Grayf32Frame::new(&buf, 4, 4, 4);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_y_len: 0,
      last_row_idx: 0,
    };
    grayf32_to(&frame, false, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_y_len, 4); // width f32 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
