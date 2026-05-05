//! Packed **X2BGR10** source (`AV_PIX_FMT_X2BGR10LE`) — 10 bits per
//! channel, 32-bit little-endian word with `(MSB) 2X | 10B | 10G |
//! 10R (LSB)`. Channel positions reversed relative to
//! [`super::X2Rgb10`].
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
  packed {
    /// Zero‑sized marker for the packed **X2BGR10** (LE) source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: X2Bgr10,
    frame: X2Bgr10Frame<'_>,
    row: X2Bgr10Row,
    sink: X2Bgr10Sink,
    walker: x2bgr10_to,
    buf_field: x2bgr10,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of an [`X2Bgr10`] source — `width * 4` bytes\n\
              laid out as `width` little-endian `u32` pixels with packing\n\
              `(MSB) 2X | 10B | 10G | 10R (LSB)`.",
    walker_doc: "Walks an [`X2Bgr10Frame`] row by row into the sink.",
  }
}
