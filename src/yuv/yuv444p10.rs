//! YUV 4:4:4 planar 10‑bit (`AV_PIX_FMT_YUV444P10LE`).
//!
//! Full-resolution chroma, 1:1 with Y. Uses the new const-generic
//! `yuv_444p_n_to_rgb_*<BITS>` kernel family (like [`super::Yuv420p10`]
//! uses `yuv_420p_n_to_rgb_*<BITS>`), parameterized on
//! `BITS ∈ {10, 12, 14}`.

use crate::frame::Yuv444p10Frame;

walker! {
  planar3 {
    /// Zero‑sized marker for the YUV 4:4:4 **10‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv444p10,
    frame: Yuv444p10Frame<'_>,
    row: Yuv444p10Row,
    sink: Yuv444p10Sink,
    walker: yuv444p10_to,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuv444p10`] source.",
    walker_doc: "Walks a [`Yuv444p10Frame`] row by row into the sink.",
  }
}
