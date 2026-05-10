//! Packed YUV 4:2:2 16-bit `Y216` source — high-bit-depth packed
//! capture format. Each row is a sequence of YUYV-shaped u16
//! quadruples (`Y₀, U, Y₁, V`); all 16 bits per sample are active.
//! See [`crate::frame::Y216Frame`] for layout details.
//!
//! The marker carries `<const BE: bool = false>`: `Y216` (= `Y216<false>`)
//! is the LE source; `Y216<true>` is the BE source. The walker
//! [`y216_to::<BE>`] propagates `BE` from [`Y216Frame<'_, BE>`] into the
//! sinker dispatch.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   BITS=16, downshifted to u8.
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   16-bit depth, full-range u16.
//! - `with_luma` — extracts the Y values from each Y216 quadruple
//!   and downshifts via `>> 8` (16-bit → u8).
//! - `with_luma_u16` — extracts the 16-bit Y values into u16
//!   (direct memcpy of the Y values; full 16-bit fidelity).
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.

// `Y216Frame` is referenced through `$crate::frame::Y2xxFrame<'_, 16, BE>` by
// the `packed_be_y2xx` walker arm; no outer import needed.

walker! {
  packed_be_y2xx {
    /// Zero-sized marker for the packed **Y216** source format
    /// (`AV_PIX_FMT_Y216{LE,BE}`). `<const BE: bool>` defaults to `false`
    /// (LE); `Y216` resolves to `Y216<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Y216,
    frame_inner: Y2xxFrame,
    bits: 16,
    row: Y216Row,
    sink: Y216Sink,
    walker: y216_to,
    buf_field: packed,
    elem_type: u16,
    row_elems: |w| w * 2,
    row_doc: concat!(
      "One row of a [`Y216`] source — `width × 2` u16 elements\n",
      "(`Y₀, U, Y₁, V` quadruples per 2-pixel block).\n",
      "\n",
      "Each u16 sample is 16-bit native — all 16 bits active (no\n",
      "padding). Per 2-pixel block layout (4 u16 elements):\n",
      "\n",
      "| u16 slot | Field | Active bits           |\n",
      "|----------|-------|-----------------------|\n",
      "| 0        | Y₀    | bits `15:0` (16-bit) |\n",
      "| 1        | U     | bits `15:0` (16-bit) |\n",
      "| 2        | Y₁    | bits `15:0` (16-bit) |\n",
      "| 3        | V     | bits `15:0` (16-bit) |\n",
      "\n",
      "Full range Y: `[0, 65535]` (16-bit). Limited range Y: `[4096,\n",
      "60160]`, limited range chroma: `[4096, 61440]`.\n",
      "\n",
      "Endianness is recorded on the parent [`Y216Frame<'_, BE>`] / sinker,\n",
      "not on the Row itself — the kernel receives `BE` as the runtime\n",
      "`big_endian` argument from the sinker dispatch.",
    ),
    walker_doc: "Walks a [`Y216Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`Y216Sink<BE>`].",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Y216Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Y216Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Y216Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.packed().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Y216Sink for CountingSink {}

  #[test]
  fn y216_walker_visits_every_row_once() {
    let buf = std::vec![0u16; 8 * 4];
    let frame = Y216Frame::new(&buf, 4, 4, 8);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    y216_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 8);
    assert_eq!(sink.last_row_idx, 3);
  }
}
