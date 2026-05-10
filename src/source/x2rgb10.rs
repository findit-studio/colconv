//! Packed **X2RGB10** source (`AV_PIX_FMT_X2RGB10{LE,BE}`) — 10 bits per
//! channel, 32-bit word with `(MSB) 2X | 10R | 10G | 10B (LSB)`. The 2
//! leading bits are **ignored padding**.
//!
//! The marker carries `<const BE: bool = false>`: `X2Rgb10` (= `X2Rgb10<false>`)
//! is the LE source; `X2Rgb10<true>` is the BE source. The walker
//! [`x2rgb10_to::<BE>`] propagates `BE` from [`X2Rgb10Frame<'_, BE>`] into the
//! sinker dispatch.
//!
//! Outputs (Ship 9e):
//! - `with_rgb` — `x2rgb10_to_rgb_row` (down-shift each 10-bit
//!   channel to 8 bits and pack as `R, G, B`).
//! - `with_rgba` — `x2rgb10_to_rgba_row` (same down-shift + force
//!   alpha to `0xFF`).
//! - `with_rgb_u16` — `x2rgb10_to_rgb_u16_row` (native 10-bit
//!   precision, low-bit aligned in `u16`; max value `1023`).
//! - `with_luma` — drop padding into the u8 RGB scratch, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::frame::X2Rgb10Frame;

walker! {
  packed_be {
    /// Zero‑sized marker for the packed **X2RGB10** source format
    /// (`AV_PIX_FMT_X2RGB10{LE,BE}`). `<const BE: bool>` defaults to `false`
    /// (LE); the alias `X2Rgb10` resolves to `X2Rgb10<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: X2Rgb10,
    frame: X2Rgb10Frame,
    row: X2Rgb10Row,
    sink: X2Rgb10Sink,
    walker: x2rgb10_to,
    walker_endian: x2rgb10_to_endian,
    buf_field: x2rgb10,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: concat!(
      "One output row of an [`X2Rgb10`] source — `width * 4` bytes\n",
      "laid out as `width` `u32` pixels with packing\n",
      "`(MSB) 2X | 10R | 10G | 10B (LSB)`. The byte order of each\n",
      "32-bit word is selected by the parent\n",
      "[`X2Rgb10Frame<'_, BE>`] / sinker `<const BE>` parameter.\n",
      "\n",
      "Bit layout per 32-bit word:\n",
      "\n",
      "| Bits   | Field |\n",
      "|--------|-------|\n",
      "| 31:30  | padding (ignored on read; RGBA outputs force α=`0xFF`) |\n",
      "| 29:20  | R (10 bits) |\n",
      "| 19:10  | G (10 bits) |\n",
      "| 9:0    | B (10 bits) |\n",
      "\n",
      "Sink authors: each pixel is one `u32` reconstructed from 4\n",
      "consecutive bytes of the slice in the BE-or-LE order set by the\n",
      "Frame. Each 10-bit channel ranges `[0, 1023]`.",
    ),
    walker_doc: "Walks an [`X2Rgb10Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`X2Rgb10Sink<BE>`].",
  }
}
