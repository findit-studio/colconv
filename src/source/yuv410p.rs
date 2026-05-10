//! YUV 4:1:0 planar (`AV_PIX_FMT_YUV410P`) — Cinepak / Sorenson
//! legacy 4:1:0 planar YUV.
//!
//! See the module docs in [`super`] for the Sink-based conversion
//! model. At 4:1:0, chroma is subsampled 4:1 in **both** axes — one
//! chroma sample covers a 4×4 block of luma (16 pixels share one
//! chroma pair). The walker passes the same chroma row to four
//! consecutive Y rows; within each row, four adjacent Y pixels share
//! one (U, V) pair via horizontal duplication in the kernel.
//!
//! Tier 1 row 7 — P3 legacy. Mostly historical interest (Cinepak,
//! Sorenson, FFmpeg's `yuv410p` test fixtures); modern pipelines
//! almost never see it.

use crate::frame::Yuv410pFrame;

walker! {
  planar3 {
    /// Zero-sized marker for the YUV 4:1:0 source format. Used as the
    /// `F` type parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv410p,
    frame: Yuv410pFrame<'_>,
    row: Yuv410pRow,
    sink: Yuv410pSink,
    walker: yuv410p_to,
    elem_type: u8,
    chroma_h: quarter,
    chroma_v: quarter,
    row_doc: "One output row of a YUV 4:1:0 source handed to a [`Yuv410pSink`].\n\n\
              Carries borrows to the source slices (full-width Y, quarter-width U/V) plus\n\
              the row index and matrix/range carry-throughs. Sinks fan one chroma sample\n\
              across four adjacent Y columns inline via the crate's fused row primitives.",
    walker_doc: "Converts a YUV 4:1:0 frame by walking its rows and feeding each one\n\
                 to the [`Yuv410pSink`].\n\n\
                 The kernel is a pure row walker — no color arithmetic happens here.\n\
                 Slice math picks the Y row and the correct chroma row for each output\n\
                 row (`chroma_row = row / 4` for 4:1:0 — same chroma row covers four\n\
                 consecutive Y rows) and hands borrows to the Sink. The Sink decides\n\
                 what to derive and where to write.",
  }
}
