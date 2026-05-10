//! YUV 4:2:2 planar 14‑bit (`AV_PIX_FMT_YUV422P14LE`). See
//! [`super::Yuv422p10`] for the 4:2:2 family structure.

use crate::frame::Yuv422pFrame16;

walker! {
  planar3_be {
    /// Zero‑sized marker for the YUV 4:2:2 **14‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv422p14,
    frame: Yuv422pFrame16<'_, 14, BE>,
    row: Yuv422p14Row,
    sink: Yuv422p14Sink,
    walker: yuv422p14_to,
    elem_type: u16,
    chroma_h: half,
    chroma_v: full,
    row_doc: "One output row of a [`Yuv422p14`] source.",
    walker_doc: "Walks a [`Yuv422p14Frame`] row by row into the sink.",
  }
}
