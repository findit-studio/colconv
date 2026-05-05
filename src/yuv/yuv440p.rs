//! YUV 4:4:0 planar 8-bit (`AV_PIX_FMT_YUV440P` / `AV_PIX_FMT_YUVJ440P`).
//!
//! Full-width chroma, **half-height** — the axis-flipped counterpart
//! to [`super::Yuv422p`]. Mostly seen from JPEG decoders that
//! subsample vertically only.
//!
//! Per-row kernel reuses [`super::Yuv444p`]'s `yuv_444_to_rgb_row`:
//! per-row math is identical (full-width chroma, no horizontal
//! duplication); only the walker reads chroma row `r / 2` instead
//! of `r`.

use crate::frame::Yuv440pFrame;

walker! {
  planar3 {
    /// Zero‑sized marker for the YUV 4:4:0 source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv440p,
    frame: Yuv440pFrame<'_>,
    row: Yuv440pRow,
    sink: Yuv440pSink,
    walker: yuv440p_to,
    elem_type: u8,
    chroma_h: full,
    chroma_v: half,
    row_doc: "One output row of a [`Yuv440p`] source. Y row `r` carries chroma row\n\
              `r / 2` (half-height vertical subsampling, full-width horizontal).",
    walker_doc: "Walks a [`Yuv440pFrame`] row by row into the sink. Y row `r` reads\n\
                 chroma row `r / 2` (half-height vertical subsampling).",
  }
}
