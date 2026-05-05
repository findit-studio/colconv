//! Packed YUV 4:4:4 8-bit `VUYX` source — display-compose capture
//! format (FFmpeg `AV_PIX_FMT_VUYX`). Each pixel is a u8 quadruple
//! `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)` where the A byte is **padding**
//! (not real alpha). The A byte is read but discarded; RGBA outputs
//! always force α=`0xFF`. See [`crate::frame::VuyxFrame`] for layout
//! details. For the source-α sibling, see [`crate::yuv::Vuya`].
//!
//! Outputs are produced via:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; padding discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; α ignored on
//!   read; α forced to `0xFF`.
//! - `with_luma` — extracts the Y byte (byte 2 of each pixel)
//!   directly.
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.
//!
//! VUYX has no u16 output paths — it is an 8-bit source.

use crate::frame::VuyxFrame;

walker! {
  packed {
    /// Zero-sized marker for the packed **VUYX** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Vuyx,
    frame: VuyxFrame<'_>,
    row: VuyxRow,
    sink: VuyxSink,
    walker: vuyx_to,
    buf_field: packed,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: concat!(
      "One row of a [`Vuyx`] source — `width × 4` bytes (4 channels per\n",
      "pixel: V, U, Y, A; the A byte is padding and is ignored on read).\n",
      "\n",
      "Byte layout per pixel:\n",
      "\n",
      "| Byte offset | Field |\n",
      "|-------------|-------|\n",
      "| 0           | V     |\n",
      "| 1           | U     |\n",
      "| 2           | Y     |\n",
      "| 3           | A (padding — ignored on read; RGBA outputs force α=`0xFF`) |\n",
      "\n",
      "The walker does not interpret the bytes — it passes the raw packed\n",
      "slice to the sink. Byte-level channel extraction happens in the\n",
      "row-kernel layer.\n",
      "\n",
      "Full range: `[0, 255]` (8-bit). Limited range Y: `[16, 235]`,\n",
      "limited range chroma: `[16, 240]`.",
    ),
    walker_doc: "Walks a [`VuyxFrame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::VuyxFrame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_packed_len: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = VuyxRow<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: VuyxRow<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_packed_len = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl VuyxSink for CountingSink {}

  #[test]
  fn vuyx_walker_visits_every_row_once() {
    // 4 px × 4 channels × 4 rows = 64 bytes
    let buf = std::vec![0u8; 4 * 4 * 4];
    let frame = VuyxFrame::new(&buf, 4, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_packed_len: 0,
      last_row_idx: 0,
    };
    vuyx_to(&frame, false, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_packed_len, 16); // width × 4 bytes per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
