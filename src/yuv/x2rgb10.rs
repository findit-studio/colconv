//! Packed **X2RGB10** source (`AV_PIX_FMT_X2RGB10LE`) — 10 bits per
//! channel, 32-bit little-endian word with `(MSB) 2X | 10R | 10G |
//! 10B (LSB)`. The 2 leading bits are **ignored padding**.
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
  packed {
    /// Zero‑sized marker for the packed **X2RGB10** (LE) source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: X2Rgb10,
    frame: X2Rgb10Frame<'_>,
    row: X2Rgb10Row,
    sink: X2Rgb10Sink,
    walker: x2rgb10_to,
    buf_field: x2rgb10,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of an [`X2Rgb10`] source — `width * 4` bytes\n\
              laid out as `width` little-endian `u32` pixels with packing\n\
              `(MSB) 2X | 10R | 10G | 10B (LSB)`.",
    walker_doc: "Walks an [`X2Rgb10Frame`] row by row into the sink.",
  }
}
