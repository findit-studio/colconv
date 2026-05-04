//! Packed YUV 4:2:2 12-bit `Y212` source — high-bit-depth packed
//! capture format (Microsoft Media Foundation / DXVA HEVC 12-bit
//! 4:2:2 hardware decode). Each row is a sequence of YUYV-shaped
//! u16 quadruples (`Y₀, U, Y₁, V`); active 12 bits are MSB-aligned
//! in each u16 (low 4 bits = 0). See [`crate::frame::Y212Frame`]
//! for layout details.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=12, downshifted to u8.
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   12-bit depth, low-bit-packed in `u16`.
//! - `with_luma` — extracts the Y values from each Y212 quadruple
//!   and downshifts via `>> 8` (12-bit MSB-aligned → u8).
//! - `with_luma_u16` — extracts the 12-bit Y values into u16
//!   (low-bit-packed).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.

use crate::frame::Y212Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **Y212** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Y212,
    frame: Y212Frame<'_>,
    row: Y212Row,
    sink: Y212Sink,
    walker: y212_to,
    buf_field: packed,
    elem_type: u16,
    row_elems: |w| w * 2,
    row_doc: "One row of a [`Y212`] source — `width × 2` u16 elements\n\
              (`Y₀, U, Y₁, V` quadruples per 2-pixel block).",
    walker_doc: "Walks a [`Y212Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Y212Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Y212Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Y212Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Y212Sink for CountingSink {}

  #[test]
  fn y212_walker_visits_every_row_once() {
    let buf = std::vec![0u16; 8 * 4];
    let frame = Y212Frame::new(&buf, 4, 4, 8);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    y212_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 8);
    assert_eq!(sink.last_row_idx, 3);
  }
}
