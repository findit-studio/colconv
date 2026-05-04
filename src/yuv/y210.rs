//! Packed YUV 4:2:2 10-bit `Y210` source — high-bit-depth packed
//! capture format (Microsoft Media Foundation / DXVA HEVC 10-bit
//! 4:2:2 hardware decode). Each row is a sequence of YUYV-shaped
//! u16 quadruples (`Y₀, U, Y₁, V`); active 10 bits are MSB-aligned
//! in each u16 (low 6 bits = 0). See [`crate::frame::Y210Frame`]
//! for layout details.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=10, downshifted to u8.
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   10-bit depth, low-bit-packed in `u16`.
//! - `with_luma` — extracts the Y values from each Y210 quadruple
//!   and downshifts via `>> 8` (10-bit MSB-aligned → u8).
//! - `with_luma_u16` — extracts the 10-bit Y values into u16
//!   (low-bit-packed).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.

use crate::frame::Y210Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **Y210** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Y210,
    frame: Y210Frame<'_>,
    row: Y210Row,
    sink: Y210Sink,
    walker: y210_to,
    buf_field: packed,
    elem_type: u16,
    row_elems: |w| w * 2,
    row_doc: "One row of a [`Y210`] source — `width × 2` u16 elements\n\
              (`Y₀, U, Y₁, V` quadruples per 2-pixel block).",
    walker_doc: "Walks a [`Y210Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Y210Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Y210Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Y210Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Y210Sink for CountingSink {}

  #[test]
  fn y210_walker_visits_every_row_once() {
    let buf = std::vec![0u16; 8 * 4];
    let frame = Y210Frame::new(&buf, 4, 4, 8);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    y210_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 8);
    assert_eq!(sink.last_row_idx, 3);
  }
}
