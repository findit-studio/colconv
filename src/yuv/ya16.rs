//! Walker spec for the `Ya16` source format (FFmpeg `ya16le` / `AV_PIX_FMT_YA16LE`).
//!
//! Single `u16` plane packed as `[Y0, A0, Y1, A1, ...]`. Each pixel occupies
//! 2 u16 elements; stride covers `width × 2` u16 elements. Alpha is real source
//! α at element slot 1 of every pixel pair (little-endian u16).

use crate::frame::Ya16Frame;

walker! {
  packed {
    /// Marker type for the `Ya16` source format (16-bit gray + alpha, 2 u16/pixel).
    ///
    /// Packed layout per pixel: `[Y(16), A(16)]`, little-endian. Alpha is real
    /// source transparency and is passed through to RGBA outputs (depth-converted
    /// to u8 via `>> 8` for 8-bit RGBA output).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Ya16,
    frame: Ya16Frame<'_>,
    row: Ya16Row,
    sink: Ya16Sink,
    walker: ya16_to,
    buf_field: packed,
    elem_type: u16,
    row_elems: |w| w * 2,
    row_doc: concat!(
      "One row of a [`Ya16`] source — `width × 2` u16 elements (2 u16 per pixel:\n",
      "Y then A).\n",
      "\n",
      "u16 slot layout per pixel:\n",
      "\n",
      "| u16 slot | Field |\n",
      "|----------|-------|\n",
      "| 0        | Y (luma, 16-bit native)   |\n",
      "| 1        | A (real α, 16-bit native) |\n",
      "\n",
      "The walker does not interpret the u16 elements — it passes the raw packed\n",
      "slice to the sink. Channel extraction happens in the row-kernel layer.",
    ),
    walker_doc: "Walks a [`Ya16Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Ya16Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_packed_len: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Ya16Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Ya16Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_packed_len = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Ya16Sink for CountingSink {}

  #[test]
  fn ya16_walker_visits_every_row_once() {
    // 4 px × 2 u16 × 4 rows = 32 u16 elements (tight stride)
    let buf = std::vec![0u16; 32];
    let frame = Ya16Frame::new(&buf, 4, 4, 8);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_packed_len: 0,
      last_row_idx: 0,
    };
    ya16_to(&frame, false, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_packed_len, 8); // width × 2 u16 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
