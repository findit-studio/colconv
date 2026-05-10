//! Packed XRGB source (`AV_PIX_FMT_0RGB`) — 8 bits per channel,
//! byte order `X, R, G, B`. The 1st byte is **ignored padding**
//! (not real alpha — see [`super::Argb`] for the alpha-bearing
//! analogue).
//!
//! Outputs (Ship 9d):
//! - `with_rgb` — `argb_to_rgb_row` (drop leading byte; identical to
//!   the [`Argb`](super::Argb) RGB path because both ignore byte 0).
//! - `with_rgba` — `xrgb_to_rgba_row` (drop padding + force alpha to
//!   `0xFF`).
//! - `with_luma` — drop padding into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::frame::XrgbFrame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **XRGB** (a.k.a. `0rgb`) source
    /// format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Xrgb,
    frame: XrgbFrame<'_>,
    row: XrgbRow,
    sink: XrgbSink,
    walker: xrgb_to,
    buf_field: xrgb,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of an [`Xrgb`] source — `width * 4` packed\n\
              `X, R, G, B` bytes.",
    walker_doc: "Walks an [`XrgbFrame`] row by row into the sink.",
  }
}
