//! Packed YUV 4:4:4 12-bit `XV36` source — high-bit-depth packed
//! capture format (FFmpeg `AV_PIX_FMT_XV36LE`). Each pixel is a u16
//! quadruple `U(16) ‖ Y(16) ‖ V(16) ‖ A(16)` with each channel using
//! the high 12 bits (low 4 bits zero, MSB-aligned at 12-bit). The
//! `X` prefix means the A slot is padding — read but discarded;
//! RGBA outputs force α = max. See [`crate::frame::Xv36Frame`] for
//! layout details.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=12, downshifted to u8; RGBA α = `0xFF` (XV36 has no alpha
//!   channel — A slot is padding).
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   12-bit depth, low-bit-packed in `u16` (high 4 bits zero); RGBA
//!   α = `0x0FFF` (12-bit max).
//! - `with_luma` — extracts Y values from each XV36 quadruple and
//!   downshifts via `>> 8` (12-bit MSB-aligned → u8 — equivalent to
//!   `>> 4` to drop padding then `>> 4` to bring 12-bit to 8-bit).
//! - `with_luma_u16` — extracts the 12-bit Y values via `>> 4`
//!   (drops padding) into u16 (low-bit-packed at 12-bit).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.

use crate::frame::Xv36Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **XV36** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Xv36,
    frame: Xv36Frame<'_>,
    row: Xv36Row,
    sink: Xv36Sink,
    walker: xv36_to,
    buf_field: packed,
    elem_type: u16,
    row_elems: |w| w * 4,
    row_doc: "One row of an [`Xv36`] source — `width × 4` u16 elements (4\n\
              channels per pixel: U, Y, V, A; the A slot is padding).",
    walker_doc: "Walks an [`Xv36Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Xv36Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Xv36Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Xv36Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Xv36Sink for CountingSink {}

  #[test]
  fn xv36_walker_visits_every_row_once() {
    let buf = std::vec![0u16; 4 * 4 * 4]; // 4 px × 4 channels × 4 rows = 64 u16 elements
    let frame = Xv36Frame::new(&buf, 4, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    xv36_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 16); // width × 4 u16 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
