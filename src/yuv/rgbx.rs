//! Packed RGBX source (`AV_PIX_FMT_RGB0`) — 8 bits per channel,
//! byte order `R, G, B, X`. The 4th byte is **ignored padding**
//! (not real alpha — see [`super::Rgba`] for the alpha-bearing
//! analogue).
//!
//! Outputs (Ship 9d):
//! - `with_rgb` — `rgba_to_rgb_row` (drop trailing byte; identical to
//!   the [`Rgba`](super::Rgba) RGB path because both ignore byte 3).
//! - `with_rgba` — `rgbx_to_rgba_row` (memcpy first 3 bytes + force
//!   alpha to `0xFF`).
//! - `with_luma` — drop padding into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::frame::RgbxFrame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **RGBX** (a.k.a. `rgb0`) source
    /// format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgbx,
    frame: RgbxFrame<'_>,
    row: RgbxRow,
    sink: RgbxSink,
    walker: rgbx_to,
    buf_field: rgbx,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of an [`Rgbx`] source — `width * 4` packed\n\
              `R, G, B, X` bytes.",
    walker_doc: "Walks an [`RgbxFrame`] row by row into the sink.",
  }
}
