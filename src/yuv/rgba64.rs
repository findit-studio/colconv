//! Packed RGBA64 source (`AV_PIX_FMT_RGBA64LE`) — 16 bits per channel,
//! `u16` element order `R, G, B, A`. Stride in u16 elements (≥ `4 * width`).
//!
//! Outputs (Tier 8 finish):
//! - `with_rgb`      — drop alpha, narrow each R/G/B channel `>> 8`, pack as R, G, B.
//! - `with_rgba`     — all four channels narrowed `>> 8`; source alpha passed through.
//! - `with_rgb_u16`  — drop alpha, native u16 passthrough (R, G, B order).
//! - `with_rgba_u16` — all four channels passed through as-is; source alpha preserved.
//! - `with_luma`     — Y′ from R/G/B after narrowing to u8 (alpha ignored).
//! - `with_luma_u16` — Y′ from R/G/B at native 16-bit depth (alpha ignored).
//! - `with_hsv`      — HSV via u8 RGB staging (alpha ignored).

use crate::frame::Rgba64Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **RGBA64** (`AV_PIX_FMT_RGBA64LE`) source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgba64,
    frame: Rgba64Frame<'_>,
    row: Rgba64Row,
    sink: Rgba64Sink,
    walker: rgba64_to,
    buf_field: rgba64,
    elem_type: u16,
    row_elems: |w| w * 4,
    row_doc: "One row of an [`Rgba64`] source — `width * 4` u16 elements \
              (`R, G, B, A` per pixel, each channel 16 bits; alpha is real).",
    walker_doc: "Walks an [`Rgba64Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Rgba64Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Rgba64Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Rgba64Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.rgba64().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Rgba64Sink for CountingSink {}

  #[test]
  fn rgba64_walker_visits_every_row_once() {
    // width=4, stride=16 (4*4), height=4 → plane needs 64 u16 elements
    let buf = std::vec![0u16; 16 * 4];
    let frame = Rgba64Frame::new(&buf, 4, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    rgba64_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 16); // width * 4 u16 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
