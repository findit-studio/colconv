//! Packed YUV 4:4:4 12-bit `XV36` source — high-bit-depth packed
//! capture format (FFmpeg `AV_PIX_FMT_XV36{LE,BE}`). Each pixel is a
//! u16 quadruple `U(16) ‖ Y(16) ‖ V(16) ‖ A(16)` with each channel
//! using the high 12 bits (low 4 bits zero, MSB-aligned at 12-bit).
//! The `X` prefix means the A slot is padding — read but discarded;
//! RGBA outputs force α = max. See [`crate::frame::Xv36Frame`] for
//! layout details.
//!
//! The marker carries `<const BE: bool = false>`: `Xv36` (=
//! `Xv36<false>`) is the LE source; `Xv36<true>` is the BE source.
//! The walker [`xv36_to::<BE>`] propagates `BE` from
//! [`Xv36Frame<'_, BE>`] into the sinker dispatch.
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
  packed_be {
    /// Zero-sized marker for the packed **XV36** source format
    /// (`AV_PIX_FMT_XV36{LE,BE}`). `<const BE: bool>` defaults to
    /// `false` (LE); the alias `Xv36` resolves to `Xv36<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Xv36,
    frame: Xv36Frame,
    row: Xv36Row,
    sink: Xv36Sink,
    walker: xv36_to,
    walker_endian: xv36_to_endian,
    buf_field: packed,
    elem_type: u16,
    row_elems: |w| w * 4,
    row_doc: concat!(
      "One row of an [`Xv36`] source — `width × 4` u16 elements (4\n",
      "channels per pixel: U, Y, V, A; the A slot is padding).\n",
      "\n",
      "Each u16 channel holds a 12-bit MSB-aligned sample with the low 4\n",
      "bits zero. Channel layout per pixel:\n",
      "\n",
      "| u16 slot | Field | Active bits           |\n",
      "|----------|-------|-----------------------|\n",
      "| 0        | U     | bits `15:4` (12-bit) |\n",
      "| 1        | Y     | bits `15:4` (12-bit) |\n",
      "| 2        | V     | bits `15:4` (12-bit) |\n",
      "| 3        | A     | bits `15:4` (padding)|\n",
      "\n",
      "Full range: `[0, 4095]` (12-bit). Limited range Y: `[256, 3760]`,\n",
      "limited range chroma: `[256, 3840]`. Endianness is recorded on\n",
      "the parent [`Xv36Frame<'_, BE>`] / sinker, not on the Row itself —\n",
      "the kernel monomorphizes on `BE` at the sinker dispatch.",
    ),
    walker_doc: "Walks an [`Xv36Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`Xv36Sink<BE>`].",
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
