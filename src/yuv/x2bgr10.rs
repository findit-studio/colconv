//! Packed **X2BGR10** source (`AV_PIX_FMT_X2BGR10{LE,BE}`) — 10 bits per
//! channel, 32-bit word with `(MSB) 2X | 10B | 10G | 10R (LSB)`. Channel
//! positions reversed relative to [`super::X2Rgb10`].
//!
//! The marker carries `<const BE: bool = false>`: `X2Bgr10` (= `X2Bgr10<false>`)
//! is the LE source; `X2Bgr10<true>` is the BE source. The walker
//! [`x2bgr10_to::<BE>`] propagates `BE` from [`X2Bgr10Frame<'_, BE>`] into the
//! sinker dispatch.
//!
//! Outputs (Ship 9e):
//! - `with_rgb` — `x2bgr10_to_rgb_row` (extract the 10-bit channels
//!   from the swapped positions, down-shift to 8 bits, output
//!   `R, G, B`).
//! - `with_rgba` — `x2bgr10_to_rgba_row` (same extraction + force
//!   alpha to `0xFF`).
//! - `with_rgb_u16` — `x2bgr10_to_rgb_u16_row` (native 10-bit
//!   precision, low-bit aligned).
//! - `with_luma` / `with_hsv` — same scratch path as `X2Rgb10`,
//!   reusing the existing `rgb_to_luma_row` / `rgb_to_hsv_row`
//!   kernels.

use crate::frame::X2Bgr10Frame;

walker! {
  packed_be {
    /// Zero‑sized marker for the packed **X2BGR10** source format
    /// (`AV_PIX_FMT_X2BGR10{LE,BE}`). `<const BE: bool>` defaults to `false`
    /// (LE); the alias `X2Bgr10` resolves to `X2Bgr10<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: X2Bgr10,
    frame: X2Bgr10Frame,
    row: X2Bgr10Row,
    sink: X2Bgr10Sink,
    walker: x2bgr10_to,
    buf_field: x2bgr10,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: concat!(
      "One output row of an [`X2Bgr10`] source — `width * 4` bytes\n",
      "laid out as `width` `u32` pixels with packing\n",
      "`(MSB) 2X | 10B | 10G | 10R (LSB)`. The byte order of each\n",
      "32-bit word is selected by the parent\n",
      "[`X2Bgr10Frame<'_, BE>`] / sinker `<const BE>` parameter.\n",
      "\n",
      "Bit layout per 32-bit word:\n",
      "\n",
      "| Bits   | Field |\n",
      "|--------|-------|\n",
      "| 31:30  | padding (ignored on read; RGBA outputs force α=`0xFF`) |\n",
      "| 29:20  | B (10 bits) |\n",
      "| 19:10  | G (10 bits) |\n",
      "| 9:0    | R (10 bits) |\n",
      "\n",
      "Channel positions reversed relative to [`crate::yuv::X2Rgb10`].\n",
      "Sink authors: each pixel is one `u32` reconstructed from 4\n",
      "consecutive bytes of the slice in the BE-or-LE order set by the\n",
      "Frame. Each 10-bit channel ranges `[0, 1023]`.",
    ),
    walker_doc: "Walks an [`X2Bgr10Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`X2Bgr10Sink<BE>`].",
  }
}
