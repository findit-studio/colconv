//! Packed YUV 4:2:2 10-bit `Y210` source — high-bit-depth packed
//! capture format (Microsoft Media Foundation / DXVA HEVC 10-bit
//! 4:2:2 hardware decode). Each row is a sequence of YUYV-shaped
//! u16 quadruples (`Y₀, U, Y₁, V`); active 10 bits are MSB-aligned
//! in each u16 (low 6 bits = 0). See [`crate::frame::Y210Frame`]
//! for layout details.
//!
//! The marker carries `<const BE: bool = false>`: `Y210` (= `Y210<false>`)
//! is the LE source; `Y210<true>` is the BE source. The walker
//! [`y210_to::<BE>`] propagates `BE` from
//! [`Y2xxFrame<'_, 10, BE>`](crate::frame::Y2xxFrame) into the
//! sinker dispatch.
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

// `Y210Frame` is referenced through `$crate::frame::Y2xxFrame<'_, 10, BE>` by
// the `packed_be_y2xx` walker arm; no outer import needed.

walker! {
  packed_be_y2xx {
    /// Zero-sized marker for the packed **Y210** source format
    /// (`AV_PIX_FMT_Y210{LE,BE}`). `<const BE: bool>` defaults to `false`
    /// (LE); `Y210` resolves to `Y210<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Y210,
    frame_inner: Y2xxFrame,
    bits: 10,
    row: Y210Row,
    sink: Y210Sink,
    walker: y210_to,
    buf_field: packed,
    elem_type: u16,
    row_elems: |w| w * 2,
    row_doc: concat!(
      "One row of a [`Y210`] source — `width × 2` u16 elements\n",
      "(`Y₀, U, Y₁, V` quadruples per 2-pixel block).\n",
      "\n",
      "Each u16 sample carries an active 10-bit value MSB-aligned (the\n",
      "low 6 bits are zero). Per 2-pixel block layout (4 u16 elements):\n",
      "\n",
      "| u16 slot | Field | Active bits           |\n",
      "|----------|-------|-----------------------|\n",
      "| 0        | Y₀    | bits `15:6` (10-bit) |\n",
      "| 1        | U     | bits `15:6` (10-bit) |\n",
      "| 2        | Y₁    | bits `15:6` (10-bit) |\n",
      "| 3        | V     | bits `15:6` (10-bit) |\n",
      "\n",
      "Full range Y: `[0, 1023]` (10-bit MSB-aligned in u16). Limited\n",
      "range Y: `[64, 940]`, limited range chroma: `[64, 960]`.\n",
      "\n",
      "Endianness is recorded on the parent \
       [`Y2xxFrame<'_, 10, BE>`](crate::frame::Y2xxFrame) / sinker,\n",
      "not on the Row itself — the kernel receives `BE` as the runtime\n",
      "`big_endian` argument from the sinker dispatch.",
    ),
    walker_doc: "Walks a [`Y2xxFrame<'_, 10, BE>`](crate::frame::Y2xxFrame) row \
                 by row into the sink. Propagates `<const BE: bool>` from the \
                 frame into [`Y210Sink<BE>`].",
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
