//! Packed ARGB source (`AV_PIX_FMT_ARGB`) — 8 bits per channel,
//! byte order `A, R, G, B`. The 1st byte is real alpha (not
//! padding); leading-alpha layout is the only difference from
//! [`super::Rgba`].
//!
//! Outputs (Ship 9c):
//! - `with_rgb` — `argb_to_rgb_row` (drop leading alpha).
//! - `with_rgba` — `argb_to_rgba_row` (rotate alpha to trailing).
//! - `with_luma` — drop leading alpha into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::frame::ArgbFrame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **ARGB** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Argb,
    frame: ArgbFrame<'_>,
    row: ArgbRow,
    sink: ArgbSink,
    walker: argb_to,
    buf_field: argb,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of an [`Argb`] source — `width * 4` packed\n\
              `A, R, G, B` bytes.",
    walker_doc: "Walks an [`ArgbFrame`] row by row into the sink.",
  }
}
