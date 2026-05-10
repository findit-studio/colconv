//! Packed BGR24 source (`AV_PIX_FMT_BGR24`) — 8 bits per channel,
//! byte order `B, G, R`. Storage and validation mirror
//! [`super::Rgb24`]; only the channel order at the row level differs.
//!
//! Outputs:
//! - `with_rgb` — `bgr_to_rgb_row` (B↔R swap during the copy).
//! - `with_rgba` — swap then append `0xFF` alpha (sinker calls
//!   `bgr_to_rgb_row` into a scratch buffer first).
//! - `with_luma` — swap then `rgb_to_luma_row` (RGB-input kernel).
//! - `with_hsv` — swap then `rgb_to_hsv_row`.

use crate::frame::Bgr24Frame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **BGR24** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Bgr24,
    frame: Bgr24Frame<'_>,
    row: Bgr24Row,
    sink: Bgr24Sink,
    walker: bgr24_to,
    buf_field: bgr,
    elem_type: u8,
    row_elems: |w| w * 3,
    row_doc: "One output row of a [`Bgr24`] source — `width * 3` packed\n\
              `B, G, R` bytes.",
    walker_doc: "Walks a [`Bgr24Frame`] row by row into the sink.",
  }
}
