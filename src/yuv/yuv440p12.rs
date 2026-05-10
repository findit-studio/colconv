//! YUV 4:4:0 planar 12‑bit (`AV_PIX_FMT_YUV440P12LE`).
//!
//! Full-width × half-height chroma at 12 bits per sample. Reuses
//! the const-generic `yuv_444p_n_to_rgb_*<12>` kernel family.

use crate::frame::Yuv440pFrame16;

walker! {
  planar3_be {
    /// Zero‑sized marker for the YUV 4:4:0 **12‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv440p12,
    frame: Yuv440pFrame16<'_, 12, BE>,
    row: Yuv440p12Row,
    sink: Yuv440p12Sink,
    walker: yuv440p12_to,
    elem_type: u16,
    chroma_h: full,
    chroma_v: half,
    row_doc: "One output row of a [`Yuv440p12`] source.",
    walker_doc: "Walks a [`Yuv440p12Frame`] row by row into the sink. Y row `r`\n\
                 reads chroma row `r / 2` (half-height vertical subsampling).",
  }
}
