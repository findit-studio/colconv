//! Packed RGB48 source (`AV_PIX_FMT_RGB48LE`) — 16 bits per channel,
//! `u16` element order `R, G, B`. Stride in u16 elements (≥ `3 * width`).
//!
//! Outputs (Tier 8 finish):
//! - `with_rgb`      — narrow each channel `>> 8`, pack as R, G, B.
//! - `with_rgba`     — same narrow + alpha = `0xFF`.
//! - `with_rgb_u16`  — native u16 passthrough (R, G, B order preserved).
//! - `with_rgba_u16` — native u16 passthrough + alpha = `0xFFFF`.
//! - `with_luma`     — Y′ from R/G/B after narrowing to u8.
//! - `with_luma_u16` — Y′ computed at u8 precision (matching `with_luma`'s
//!   output) and zero-extended to u16. Same convention as the 8-bit-source
//!   family; not native 16-bit luma precision.
//! - `with_hsv`      — HSV via u8 RGB staging.

use crate::frame::Rgb48Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **RGB48** (`AV_PIX_FMT_RGB48LE`) source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgb48,
    frame: Rgb48Frame<'_>,
    row: Rgb48Row,
    sink: Rgb48Sink,
    walker: rgb48_to,
    buf_field: rgb48,
    elem_type: u16,
    row_elems: |w| w * 3,
    row_doc: "One row of an [`Rgb48`] source — `width * 3` u16 elements \
              (`R, G, B` per pixel, each channel 16 bits).",
    walker_doc: "Walks an [`Rgb48Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Rgb48Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Rgb48Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Rgb48Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.rgb48().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Rgb48Sink for CountingSink {}

  #[test]
  fn rgb48_walker_visits_every_row_once() {
    // width=4, stride=12 (3*4), height=4 → plane needs 48 u16 elements
    let buf = std::vec![0u16; 12 * 4];
    let frame = Rgb48Frame::new(&buf, 4, 4, 12);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    rgb48_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 12); // width * 3 u16 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
