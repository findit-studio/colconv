//! Packed YUV 4:4:4 10-bit `V410` source — DCI / SDI capture format
//! (FFmpeg `AV_PIX_FMT_V410`, also known as `XV30`). Each row is a
//! sequence of u32 words; one word per pixel. The 10-bit U / Y / V
//! channels are bit-packed per word with 2 bits of padding (see
//! [`crate::frame::V410Frame`] for the layout table).
//!
//! The marker carries `<const BE: bool = false>`: `V410` (=
//! `V410<false>`) is the LE wire variant; `V410<true>` is the BE wire
//! variant (each u32 word byte-swapped before unpacking). The walker
//! [`v410_to::<BE>`] propagates `BE` from [`V410Frame<'_, BE>`] into
//! the sinker dispatch.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=10, downshifted to u8.
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   10-bit depth, low-bit-packed in `u16`.
//! - `with_luma` — extracts the Y values from each V410 word and
//!   downshifts via `>> 2` (10-bit → u8).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.
//!
//! `with_luma_u16` is intentionally **not** exposed on `V410` —
//! deferred until a real consumer surfaces (Spec § 11).

use crate::frame::V410Frame;

walker! {
  packed_be {
    /// Zero-sized marker for the packed **V410** source format
    /// (`AV_PIX_FMT_V410{LE,BE}`). `<const BE: bool>` defaults to
    /// `false` (LE); the alias `V410` resolves to `V410<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: V410,
    frame: V410Frame,
    row: V410Row,
    sink: V410Sink,
    walker: v410_to,
    buf_field: packed,
    elem_type: u32,
    row_elems: |w| w,
    row_doc: concat!(
      "One row of a [`V410`] source — `width` u32 elements (one pixel\n",
      "per word; 32-bit word with 10-bit U / Y / V channels and 2-bit\n",
      "padding at the MSB).\n",
      "\n",
      "Bit layout per 32-bit word (LE):\n",
      "\n",
      "```text\n",
      "(msb) 2X | 10V | 10Y | 10U (lsb)\n",
      "```\n",
      "\n",
      "Full range: `[0, 1023]` (10-bit). Limited range Y: `[64, 940]`,\n",
      "limited range chroma: `[64, 960]`. Endianness is recorded on the\n",
      "parent [`V410Frame<'_, BE>`] / sinker, not on the Row itself —\n",
      "the kernel monomorphizes on `BE` at the sinker dispatch.",
    ),
    walker_doc: "Walks a [`V410Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`V410Sink<BE>`].",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::V410Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = V410Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: V410Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl V410Sink for CountingSink {}

  #[test]
  fn v410_walker_visits_every_row_once() {
    let buf = std::vec![0u32; 4 * 4]; // 4 px × 4 rows = 16 u32 words
    let frame = V410Frame::new(&buf, 4, 4, 4);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    v410_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 4); // width u32 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
