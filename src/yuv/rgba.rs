//! Packed RGBA source (`AV_PIX_FMT_RGBA`) — 8 bits per channel,
//! byte order `R, G, B, A`. The 4th byte is real alpha (not
//! padding).
//!
//! Outputs (Ship 9b):
//! - `with_rgb` — `rgba_to_rgb_row` (drop alpha; identity copy of
//!   the first 3 bytes per pixel).
//! - `with_rgba` — identity row copy (input == output layout).
//! - `with_luma` — drop alpha into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — drop alpha into `rgb_scratch`, then
//!   `rgb_to_hsv_row`.

use crate::frame::RgbaFrame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **RGBA** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgba,
    frame: RgbaFrame<'_>,
    row: RgbaRow,
    sink: RgbaSink,
    walker: rgba_to,
    buf_field: rgba,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of an [`Rgba`] source — `width * 4` packed\n\
              `R, G, B, A` bytes.",
    walker_doc: "Walks an [`RgbaFrame`] row by row into the sink.",
  }
}
