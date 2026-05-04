//! Packed YUV 4:4:4 10-bit `V30X` source — sibling of [`crate::yuv::V410`]
//! with opposite padding position (FFmpeg `AV_PIX_FMT_V30XLE`). Each row is a
//! sequence of u32 words; one word per pixel. The 10-bit V / Y / U
//! channels are bit-packed per word with 2 bits of padding at the LSB (see
//! [`crate::frame::V30XFrame`] for the layout table).
//!
//! Bit layout per 32-bit word:
//!
//! ```text
//! (msb) 10V | 10Y | 10U | 2X (lsb)
//! ```
//!
//! V30X is a sibling of V410 with the padding at the **LSB** instead of
//! V410's MSB padding. The walker body is structurally identical to V410's.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=10, downshifted to u8.
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   10-bit depth, low-bit-packed in `u16`.
//! - `with_luma` — extracts the Y values from each V30X word and
//!   downshifts via `>> 2` (10-bit → u8).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.
//!
//! `with_luma_u16` is intentionally **not** exposed on `V30X` —
//! deferred until a real consumer surfaces (Spec § 11).

use crate::frame::V30XFrame;

walker! {
  packed {
    /// Zero-sized marker for the packed **V30X** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: V30X,
    frame: V30XFrame<'_>,
    row: V30XRow,
    sink: V30XSink,
    walker: v30x_to,
    buf_field: packed,
    elem_type: u32,
    row_elems: |w| w,
    row_doc: "One row of a [`V30X`] source — `width` u32 elements (one pixel\n\
              per word; 32-bit word with 10-bit V / Y / U channels and 2-bit\n\
              padding at the LSB).",
    walker_doc: "Walks a [`V30XFrame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::V30XFrame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = V30XRow<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: V30XRow<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl V30XSink for CountingSink {}

  #[test]
  fn v30x_walker_visits_every_row_once() {
    let buf = std::vec![0u32; 4 * 4]; // 4 px × 4 rows = 16 u32 words
    let frame = V30XFrame::new(&buf, 4, 4, 4);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    v30x_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 4); // width u32 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
