//! Packed YUV 4:4:4 8-bit `VUYA` source — display-compose capture
//! format (FFmpeg `AV_PIX_FMT_VUYA`). Each pixel is a u8 quadruple
//! `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)` where the A byte is **real alpha**
//! (source transparency). See [`crate::frame::VuyaFrame`] for layout
//! details.
//!
//! Outputs are produced via:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; alpha discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; source α
//!   passed through from byte 3 of each pixel.
//! - `with_luma` — extracts the Y byte (byte 2 of each pixel)
//!   directly.
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.
//!
//! VUYA has no u16 output paths — it is an 8-bit source.

use crate::frame::VuyaFrame;

walker! {
  packed {
    /// Zero-sized marker for the packed **VUYA** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Vuya,
    frame: VuyaFrame<'_>,
    row: VuyaRow,
    sink: VuyaSink,
    walker: vuya_to,
    buf_field: packed,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One row of a [`Vuya`] source — `width × 4` bytes (4 channels per\n\
              pixel: V, U, Y, A; the A byte is real source alpha).",
    walker_doc: "Walks a [`VuyaFrame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::VuyaFrame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_packed_len: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = VuyaRow<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: VuyaRow<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_packed_len = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl VuyaSink for CountingSink {}

  #[test]
  fn vuya_walker_visits_every_row_once() {
    // 4 px × 4 channels × 4 rows = 64 bytes
    let buf = std::vec![0u8; 4 * 4 * 4];
    let frame = VuyaFrame::new(&buf, 4, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_packed_len: 0,
      last_row_idx: 0,
    };
    vuya_to(&frame, false, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_packed_len, 16); // width × 4 bytes per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
