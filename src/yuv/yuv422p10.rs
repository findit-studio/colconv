//! YUV 4:2:2 planar 10‑bit (`AV_PIX_FMT_YUV422P10LE`).
//!
//! Same `u16`‑backed layout as [`super::Yuv420p10`] with 4:2:2 chroma
//! (half‑width, **full‑height**). Per‑row kernel reuses the 4:2:0
//! family — [`crate::row::yuv420p10_to_rgb_row`] — verbatim. See
//! [`super::Yuv422p`] for the axis‑difference rationale.

use crate::frame::Yuv422p10Frame;

walker! {
  planar3 {
    /// Zero‑sized marker for the YUV 4:2:2 **10‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv422p10,
    frame: Yuv422p10Frame<'_>,
    row: Yuv422p10Row,
    sink: Yuv422p10Sink,
    walker: yuv422p10_to,
    elem_type: u16,
    chroma_h: half,
    chroma_v: full,
    row_doc: "One output row of a [`Yuv422p10`] source handed to a [`Yuv422p10Sink`].",
    walker_doc: "Walks a [`Yuv422p10Frame`] row by row into the sink. Chroma\n\
                 advances every row (vs 4:2:0's `row / 2`).",
  }
}
