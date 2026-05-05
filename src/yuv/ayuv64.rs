//! Packed YUV 4:4:4 16-bit + source α `AYUV64` source format
//! (FFmpeg `AV_PIX_FMT_AYUV64LE`). Each pixel is a u16 quadruple
//! `A(16) ‖ Y(16) ‖ U(16) ‖ V(16)` = 8 bytes.
//!
//! | u16 slot | Field | Notes                            |
//! |----------|-------|----------------------------------|
//! | 0        | A     | Source α — real, 16-bit native   |
//! | 1        | Y     | Luma, 16-bit native              |
//! | 2        | U     | Cb chroma, 16-bit native         |
//! | 3        | V     | Cr chroma, 16-bit native         |
//!
//! 16-bit native means all 16 bits are active (no padding bits).
//!
//! Source α is real (not padding):
//! - For u8 RGBA output (`with_rgba`) it is depth-converted to u8
//!   via `>> 8`.
//! - For u16 RGBA output (`with_rgba_u16`) it is written direct as
//!   u16 without modification.
//!
//! Outputs are produced via:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; alpha discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; source α
//!   depth-converted (`>> 8`) to u8.
//! - `with_rgb_u16` — packed YUV → RGB u16 pipeline; alpha discarded.
//! - `with_rgba_u16` — packed YUV → RGBA u16 pipeline; source α
//!   passed through as u16.
//! - `with_luma` — extracts the Y value and downshifts to u8.
//! - `with_luma_u16` — extracts the Y value as native u16.
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.
//!
//! AYUV64 is type-distinct: it has real alpha at slot 0. There is no
//! α-as-padding sibling in scope.

use crate::frame::Ayuv64Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **AYUV64** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Ayuv64,
    frame: Ayuv64Frame<'_>,
    row: Ayuv64Row,
    sink: Ayuv64Sink,
    walker: ayuv64_to,
    buf_field: packed,
    elem_type: u16,
    row_elems: |w| w * 4,
    row_doc: concat!(
      "One row of an [`Ayuv64`] source — `width × 4` u16 elements (4\n",
      "channels per pixel: A, Y, U, V; the A slot is real source alpha).\n",
      "\n",
      "Each u16 channel holds a 16-bit native sample (all bits active).\n",
      "Channel layout per pixel:\n",
      "\n",
      "| u16 slot | Field | Notes                         |\n",
      "|----------|-------|-------------------------------|\n",
      "| 0        | A     | Source α — real, 16-bit native|\n",
      "| 1        | Y     | Luma                          |\n",
      "| 2        | U     | Cb chroma                     |\n",
      "| 3        | V     | Cr chroma                     |\n",
      "\n",
      "The walker does not interpret the u16 elements — it passes the raw\n",
      "packed slice to the sink. Channel extraction happens in the\n",
      "row-kernel layer.\n",
      "\n",
      "Full range: `[0, 65535]` (16-bit). Limited range Y: `[4096, 60160]`,\n",
      "limited range chroma: `[4096, 61440]`.",
    ),
    walker_doc: "Walks an [`Ayuv64Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Ayuv64Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_packed_len: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Ayuv64Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Ayuv64Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_packed_len = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Ayuv64Sink for CountingSink {}

  #[test]
  fn ayuv64_walker_visits_every_row_once() {
    // 4 px × 4 channels × 4 rows = 64 u16 elements
    let buf = std::vec![0u16; 4 * 4 * 4];
    let frame = Ayuv64Frame::new(&buf, 4, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_packed_len: 0,
      last_row_idx: 0,
    };
    ayuv64_to(&frame, false, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_packed_len, 16); // width × 4 u16 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
