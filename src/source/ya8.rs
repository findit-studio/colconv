//! Walker spec for the `Ya8` source format (FFmpeg `ya8` / `AV_PIX_FMT_YA8`).
//!
//! Single `u8` plane packed as `[Y0, A0, Y1, A1, ...]`. Each pixel occupies
//! 2 bytes; stride covers `width × 2` bytes. Alpha is real source α at slot 1
//! of every pixel pair.

use crate::frame::Ya8Frame;

walker! {
  packed {
    /// Marker type for the `Ya8` source format (8-bit gray + alpha, 2 bytes/pixel).
    ///
    /// Packed layout per pixel: `[Y(8), A(8)]`. Alpha is real source transparency
    /// and is passed through to RGBA outputs.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Ya8,
    frame: Ya8Frame<'_>,
    row: Ya8Row,
    sink: Ya8Sink,
    walker: ya8_to,
    buf_field: packed,
    elem_type: u8,
    row_elems: |w| w * 2,
    row_doc: concat!(
      "One row of a [`Ya8`] source — `width × 2` bytes (2 bytes per pixel:\n",
      "Y then A).\n",
      "\n",
      "Byte layout per pixel:\n",
      "\n",
      "| Byte offset | Field |\n",
      "|-------------|-------|\n",
      "| 0           | Y (luma)      |\n",
      "| 1           | A (real α)    |\n",
      "\n",
      "The walker does not interpret the bytes — it passes the raw packed\n",
      "slice to the sink. Byte-level channel extraction happens in the\n",
      "row-kernel layer.",
    ),
    walker_doc: "Walks a [`Ya8Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Ya8Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_packed_len: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Ya8Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Ya8Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_packed_len = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Ya8Sink for CountingSink {}

  #[test]
  fn ya8_walker_visits_every_row_once() {
    // 4 px × 2 bytes × 4 rows = 32 bytes (tight stride)
    let buf = std::vec![0u8; 32];
    let frame = Ya8Frame::new(&buf, 4, 4, 8);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_packed_len: 0,
      last_row_idx: 0,
    };
    ya8_to(&frame, false, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_packed_len, 8); // width × 2 bytes per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
