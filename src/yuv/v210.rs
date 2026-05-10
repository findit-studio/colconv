//! Packed YUV 4:2:2 10-bit `v210` source — pro-broadcast 10-bit SDI
//! capture format. Each 16-byte word holds 6 pixels (12 × 10-bit
//! samples). See [`crate::frame::V210Frame`] for layout details.
//!
//! The marker carries `<const BE: bool = false>`: `V210` (= `V210<false>`)
//! is the LE source (the canonical SMPTE-272M wire layout); `V210<true>`
//! is the BE source. The walker [`v210_to::<BE>`] propagates `BE` from
//! [`V210Frame<'_, BE>`] into the sinker dispatch.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=10, downshifted to u8.
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   10-bit depth, low-bit-packed in `u16`.
//! - `with_luma` — extracts the 6 Y values from each v210 word and
//!   downshifts via `>> 2`.
//! - `with_luma_u16` — extracts the 10-bit Y values into u16
//!   (low-bit-packed).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.

use crate::frame::V210Frame;

walker! {
  packed_be {
    /// Zero-sized marker for the packed **v210** source format.
    /// `<const BE: bool>` defaults to `false` (LE — the canonical
    /// SMPTE-272M layout); `V210` resolves to `V210<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: V210,
    frame: V210Frame,
    row: V210Row,
    sink: V210Sink,
    walker: v210_to,
    buf_field: v210,
    elem_type: u8,
    row_elems: |w| w.div_ceil(6) * 16,
    row_doc: "One row of a [`V210`] source — `(width / 6) * 16` packed bytes. \
              Endianness is recorded on the parent [`V210Frame<'_, BE>`] / \
              sinker, not on the Row itself — the kernel receives `BE` as \
              the runtime `big_endian` argument from the sinker dispatch.",
    walker_doc: "Walks a [`V210Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`V210Sink<BE>`].",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::V210Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = V210Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: V210Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.v210().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl V210Sink for CountingSink {}

  #[test]
  fn v210_walker_visits_every_row_once() {
    let buf = std::vec![0u8; 16 * 4];
    let frame = V210Frame::new(&buf, 6, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    v210_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 16);
    assert_eq!(sink.last_row_idx, 3);
  }
}
