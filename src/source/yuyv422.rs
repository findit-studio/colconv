//! Packed YUV 4:2:2 source (`AV_PIX_FMT_YUYV422`, also known as
//! YUY2). One plane, byte order `Y0, U0, Y1, V0` per 2-pixel
//! block — Y in even byte positions, U/V in odd positions with
//! U preceding V.
//!
//! Common output of older codecs (M-JPEG, DV), Windows DirectShow /
//! V4L2 webcams, and 8-bit SDI capture in YUY2 mode.
//!
//! Outputs are produced via:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline.
//! - `with_luma` — copies the Y bytes from even positions of the row.
//! - `with_hsv` — stages an internal RGB scratch and runs the
//!   existing `rgb_to_hsv_row` kernel.
//!
//! The companion [`super::Uyvy422`] format swaps Y and UV positions;
//! [`super::Yvyu422`] swaps U and V relative to YUYV. All three reuse
//! the same const-generic `yuv422_packed_to_rgb_or_rgba_row` template
//! across scalar + every SIMD backend.

use crate::frame::Yuyv422Frame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **YUYV422** source format. Used
    /// as the `F` type parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuyv422,
    frame: Yuyv422Frame<'_>,
    row: Yuyv422Row,
    sink: Yuyv422Sink,
    walker: yuyv422_to,
    buf_field: yuyv,
    elem_type: u8,
    row_elems: |w| w * 2,
    row_doc: "One output row of a [`Yuyv422`] source — `2 * width` packed\n\
              `Y0, U0, Y1, V0, …` bytes.",
    walker_doc: "Walks a [`Yuyv422Frame`] row by row into the sink.",
  }
}
