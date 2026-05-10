//! YUV 4:4:4 planar 12‑bit (`AV_PIX_FMT_YUV444P12LE`). See
//! [`super::Yuv444p10`] for the 4:4:4 family structure.

use crate::frame::Yuv444pFrame16;

walker! {
  planar3_be {
    /// Zero‑sized marker for the YUV 4:4:4 **12‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv444p12,
    frame: Yuv444pFrame16<'_, 12, BE>,
    frame_le: Yuv444pFrame16<'_, 12, false>,
    row: Yuv444p12Row,
    sink: Yuv444p12Sink,
    walker: yuv444p12_to,
    walker_endian: yuv444p12_to_endian,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuv444p12`] source.",
    walker_doc: "Walks a [`Yuv444p12Frame`] row by row into the sink.",
  }
}
