//! Packed ABGR source (`AV_PIX_FMT_ABGR`) — 8 bits per channel,
//! byte order `A, B, G, R`. Leading alpha + reversed RGB order
//! relative to [`super::Argb`].
//!
//! Outputs (Ship 9c):
//! - `with_rgb` — `abgr_to_rgb_row` (drop alpha + R↔B swap).
//! - `with_rgba` — `abgr_to_rgba_row` (full byte reverse: alpha
//!   rotates to trailing AND inner three bytes flip).
//! - `with_luma` — same swap path into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::frame::AbgrFrame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **ABGR** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Abgr,
    frame: AbgrFrame<'_>,
    row: AbgrRow,
    sink: AbgrSink,
    walker: abgr_to,
    buf_field: abgr,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of an [`Abgr`] source — `width * 4` packed\n\
              `A, B, G, R` bytes.",
    walker_doc: "Walks an [`AbgrFrame`] row by row into the sink.",
  }
}
