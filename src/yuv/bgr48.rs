//! Packed BGR48 source (`AV_PIX_FMT_BGR48LE`) — 16 bits per channel,
//! `u16` element order `B, G, R`. Stride in u16 elements (≥ `3 * width`).
//!
//! Outputs (Tier 8 finish):
//! - `with_rgb`      — swap B↔R, narrow each channel `>> 8`, pack as R, G, B.
//! - `with_rgba`     — same swap + narrow + alpha = `0xFF`.
//! - `with_rgb_u16`  — swap B↔R, native u16 passthrough (R, G, B output order).
//! - `with_rgba_u16` — swap B↔R, native u16 passthrough + alpha = `0xFFFF`.
//! - `with_luma`     — Y′ from R/G/B after channel swap and narrowing to u8.
//! - `with_luma_u16` — Y′ computed at u8 precision (matching `with_luma`'s
//!   output, with the same B↔R swap applied first) and zero-extended to
//!   u16. Same convention as the 8-bit-source family; not native 16-bit
//!   luma precision.
//! - `with_hsv`      — HSV via u8 RGB staging.

use crate::frame::Bgr48Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **BGR48** (`AV_PIX_FMT_BGR48LE`) source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Bgr48,
    frame: Bgr48Frame<'_>,
    row: Bgr48Row,
    sink: Bgr48Sink,
    walker: bgr48_to,
    buf_field: bgr48,
    elem_type: u16,
    row_elems: |w| w * 3,
    row_doc: "One row of a [`Bgr48`] source — `width * 3` u16 elements \
              (`B, G, R` per pixel, each channel 16 bits).",
    walker_doc: "Walks a [`Bgr48Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Bgr48Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Bgr48Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Bgr48Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.bgr48().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Bgr48Sink for CountingSink {}

  #[test]
  fn bgr48_walker_visits_every_row_once() {
    let buf = std::vec![0u16; 12 * 4];
    let frame = Bgr48Frame::new(&buf, 4, 4, 12);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    bgr48_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 12);
    assert_eq!(sink.last_row_idx, 3);
  }
}
