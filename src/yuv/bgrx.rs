//! Packed BGRX source (`AV_PIX_FMT_BGR0`) — 8 bits per channel,
//! byte order `B, G, R, X`. Trailing padding + reversed RGB order
//! relative to [`super::Rgbx`].
//!
//! Outputs (Ship 9d):
//! - `with_rgb` — `bgra_to_rgb_row` (drop trailing byte + R↔B swap;
//!   identical to the [`Bgra`](super::Bgra) RGB path because both
//!   ignore byte 3).
//! - `with_rgba` — `bgrx_to_rgba_row` (R↔B swap + force alpha to
//!   `0xFF`).
//! - `with_luma` — same swap+drop path into `rgb_scratch`, then
//!   `rgb_to_luma_row`.
//! - `with_hsv` — same scratch path, then `rgb_to_hsv_row`.

use crate::frame::BgrxFrame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **BGRX** (a.k.a. `bgr0`) source
    /// format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Bgrx,
    frame: BgrxFrame<'_>,
    row: BgrxRow,
    sink: BgrxSink,
    walker: bgrx_to,
    buf_field: bgrx,
    elem_type: u8,
    row_elems: |w| w * 4,
    row_doc: "One output row of a [`Bgrx`] source — `width * 4` packed\n\
              `B, G, R, X` bytes.",
    walker_doc: "Walks a [`BgrxFrame`] row by row into the sink.",
  }
}
