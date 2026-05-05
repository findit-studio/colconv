//! Packed XBGR source (`AV_PIX_FMT_0BGR`) — 8 bits per channel,
//! byte order `X, B, G, R`. Leading padding + reversed RGB order
//! relative to [`super::Xrgb`].
//!
//! Outputs (Ship 9d):
//! - `with_rgb` — `abgr_to_rgb_row` (drop leading byte + R↔B swap;
//!   identical to the [`Abgr`](super::Abgr) RGB path because both
//!   ignore byte 0).
//! - `with_rgba` — `xbgr_to_rgba_row` (drop padding + R↔B swap +
//!   force alpha to `0xFF`).
//! - `with_luma` — same swap+drop path into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::frame::XbgrFrame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **XBGR** (a.k.a. `0bgr`) source
    /// format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Xbgr,
    frame: XbgrFrame<'_>,
    row: XbgrRow,
    sink: XbgrSink,
    walker: xbgr_to,
    buf_field: xbgr,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of an [`Xbgr`] source — `width * 4` packed\n\
              `X, B, G, R` bytes.",
    walker_doc: "Walks an [`XbgrFrame`] row by row into the sink.",
  }
}
