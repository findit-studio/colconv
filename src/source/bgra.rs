//! Packed BGRA source (`AV_PIX_FMT_BGRA`) — 8 bits per channel,
//! byte order `B, G, R, A`. The 4th byte is real alpha (not
//! padding); only the channel order on the first three bytes
//! distinguishes this from [`super::Rgba`].
//!
//! Outputs (Ship 9b):
//! - `with_rgb` — `bgra_to_rgb_row` (R↔B swap + drop alpha).
//! - `with_rgba` — `bgra_to_rgba_row` (R↔B swap, alpha preserved).
//! - `with_luma` — `bgra_to_rgb_row` into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::frame::BgraFrame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **BGRA** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Bgra,
    frame: BgraFrame<'_>,
    row: BgraRow,
    sink: BgraSink,
    walker: bgra_to,
    buf_field: bgra,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of a [`Bgra`] source — `width * 4` packed\n\
              `B, G, R, A` bytes.",
    walker_doc: "Walks a [`BgraFrame`] row by row into the sink.",
  }
}
