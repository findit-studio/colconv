//! Packed RGB24 source (`AV_PIX_FMT_RGB24`) — 8 bits per channel,
//! byte order `R, G, B`.
//!
//! Unlike every other source format in this crate, the input is
//! already **RGB**, not YUV — there's no chroma matrix work. Outputs
//! are produced by:
//! - `with_rgb` — identity copy (RGB in → RGB out).
//! - `with_rgba` — `expand_rgb_to_rgba_row` with constant `0xFF` alpha.
//! - `with_luma` — `rgb_to_luma_row` (BT.709 / 601 / etc. coefficients).
//! - `with_hsv` — existing `rgb_to_hsv_row` kernel.
//!
//! The companion [`super::Bgr24`] format swaps R↔B at the row level
//! before reusing the same RGB-input kernels.

use crate::frame::Rgb24Frame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **RGB24** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgb24,
    frame: Rgb24Frame<'_>,
    row: Rgb24Row,
    sink: Rgb24Sink,
    walker: rgb24_to,
    buf_field: rgb,
    elem_type: u8,
    row_elems: |w| w * 3,
    row_doc: "One output row of an [`Rgb24`] source — `width * 3` packed\n\
              `R, G, B` bytes.",
    walker_doc: "Walks an [`Rgb24Frame`] row by row into the sink.",
  }
}
