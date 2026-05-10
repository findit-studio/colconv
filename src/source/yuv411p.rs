//! YUV 4:1:1 planar (`AV_PIX_FMT_YUV411P`).
//!
//! Three planes: full-size Y + **quarter-width**, **full-height** U/V.
//! Per-row contract: every chroma sample covers four Y columns
//! (4:1:1 horizontal subsampling); chroma rows are fully sampled
//! vertically (one chroma row per Y row, like 4:2:2). The kernel
//! upsamples chroma 1→4 in registers; no intermediate memory traffic.
//!
//! Common in DV-NTSC video (legacy). The structural shape mirrors
//! [`super::Yuv422p`] — only the horizontal chroma subsampling factor
//! changes from 2× to 4×.

use crate::frame::Yuv411pFrame;

walker! {
  planar3 {
    /// Zero-sized marker for the YUV 4:1:1 source format. Used as the
    /// `F` type parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv411p,
    frame: Yuv411pFrame<'_>,
    row: Yuv411pRow,
    sink: Yuv411pSink,
    walker: yuv411p_to,
    elem_type: u8,
    chroma_h: quarter,
    chroma_v: full,
    row_doc: "One output row of a YUV 4:1:1 source handed to a [`Yuv411pSink`].\n\n\
              Carries borrows to the source slices (full-width Y, quarter-width U/V)\n\
              plus the row index and matrix/range carry-throughs. Chroma is fully\n\
              sampled vertically — one chroma row per Y row — and **quarter-sampled**\n\
              horizontally (one chroma sample per four Y columns).",
    walker_doc: "Converts a YUV 4:1:1 frame by walking its rows and feeding each\n\
                 one to the [`Yuv411pSink`]. Chroma advances every row (full-height\n\
                 chroma like 4:2:2); the chroma row covers `width.div_ceil(4)`\n\
                 samples (4× horizontal subsampling, FFmpeg ceiling — for widths\n\
                 not divisible by 4 the final chroma sample covers a partial\n\
                 1..3-pixel group of trailing Y columns).",
  }
}
