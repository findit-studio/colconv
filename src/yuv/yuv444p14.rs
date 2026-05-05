//! YUV 4:4:4 planar 14‑bit (`AV_PIX_FMT_YUV444P14LE`). See
//! [`super::Yuv444p10`] for the 4:4:4 family structure.

use crate::frame::Yuv444p14Frame;

walker! {
  planar3 {
    /// Zero‑sized marker for the YUV 4:4:4 **14‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv444p14,
    frame: Yuv444p14Frame<'_>,
    row: Yuv444p14Row,
    sink: Yuv444p14Sink,
    walker: yuv444p14_to,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuv444p14`] source.",
    walker_doc: "Walks a [`Yuv444p14Frame`] row by row into the sink.",
  }
}
