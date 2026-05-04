//! YUV 4:2:2 planar (`AV_PIX_FMT_YUV422P`, `yuvj422p`).
//!
//! Three planes: full-size Y + half-width, **full-height** U/V.
//! The per-row kernel is identical to [`super::Yuv420p`]'s — the
//! 4:2:0 → 4:2:2 difference is purely vertical: YUV420p reads chroma
//! row `r / 2`, YUV422p reads chroma row `r`. The sinker calls
//! [`crate::row::yuv_420_to_rgb_row`] directly.

use crate::frame::Yuv422pFrame;

walker! {
  planar3 {
    /// Zero-sized marker for the YUV 4:2:2 source format. Used as the
    /// `F` type parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv422p,
    frame: Yuv422pFrame<'_>,
    row: Yuv422pRow,
    sink: Yuv422pSink,
    walker: yuv422p_to,
    elem_type: u8,
    chroma_h: half,
    chroma_v: full,
    row_doc: "One output row of a YUV 4:2:2 source handed to a [`Yuv422pSink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width U/V) plus\n\
              the row index and matrix/range carry-throughs. Unlike 4:2:0, no two Y\n\
              rows share a chroma row — the walker advances U/V every row.",
    walker_doc: "Converts a YUV 4:2:2 frame by walking its rows and feeding each one\n\
                 to the [`Yuv422pSink`]. Chroma advances every row (vs 4:2:0's `row / 2`).",
  }
}
