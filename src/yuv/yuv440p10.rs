//! YUV 4:4:0 planar 10‑bit (`AV_PIX_FMT_YUV440P10LE`).
//!
//! Full-width × half-height chroma at 10 bits per sample. Reuses
//! the const-generic `yuv_444p_n_to_rgb_*<10>` kernel family — same
//! shape (full-width chroma, no horizontal duplication); only the
//! walker reads chroma row `r / 2`.

use crate::frame::Yuv440pFrame16;

walker! {
  planar3_be {
    /// Zero‑sized marker for the YUV 4:4:0 **10‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv440p10,
    frame: Yuv440pFrame16<'_, 10, BE>,
    frame_le: Yuv440pFrame16<'_, 10, false>,
    row: Yuv440p10Row,
    sink: Yuv440p10Sink,
    walker: yuv440p10_to,
    walker_endian: yuv440p10_to_endian,
    elem_type: u16,
    chroma_h: full,
    chroma_v: half,
    row_doc: "One output row of a [`Yuv440p10`] source.",
    walker_doc: "Walks a [`Yuv440p10Frame`] row by row into the sink. Y row `r`\n\
                 reads chroma row `r / 2` (half-height vertical subsampling).",
  }
}
