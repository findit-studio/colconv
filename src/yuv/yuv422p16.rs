//! YUV 4:2:2 planar 16‑bit (`AV_PIX_FMT_YUV422P16LE`). Reuses the
//! 4:2:0 16‑bit kernels — per‑row shape is identical; only the
//! vertical walker differs.

use crate::frame::Yuv422p16Frame;

walker! {
  planar3 {
    /// Zero‑sized marker for the YUV 4:2:2 **16‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv422p16,
    frame: Yuv422p16Frame<'_>,
    row: Yuv422p16Row,
    sink: Yuv422p16Sink,
    walker: yuv422p16_to,
    elem_type: u16,
    chroma_h: half,
    chroma_v: full,
    row_doc: "One output row of a [`Yuv422p16`] source.",
    walker_doc: "Walks a [`Yuv422p16Frame`] row by row into the sink.",
  }
}
