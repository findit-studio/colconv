//! YUV 4:2:0 planar (`AV_PIX_FMT_YUV420P`, `yuvj420p`, `yuv420p9/10/…`
//! once we parameterize depth).
//!
//! See the module docs in [`super`] for the Sink-based conversion
//! model. At 4:2:0 the kernel reads one chroma row per *two* Y rows;
//! both Y rows of a pair receive the same chroma row when the kernel
//! hands them to the Sink.

use crate::frame::Yuv420pFrame;

walker! {
  planar3 {
    /// Zero-sized marker for the YUV 4:2:0 source format. Used as the
    /// `F` type parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv420p,
    frame: Yuv420pFrame<'_>,
    row: Yuv420pRow,
    sink: Yuv420pSink,
    walker: yuv420p_to,
    elem_type: u8,
    chroma_h: half,
    chroma_v: half,
    row_doc: "One output row of a YUV 4:2:0 source handed to a [`Yuv420pSink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width U/V) plus\n\
              the row index and matrix/range carry-throughs. Sinks that need full-width\n\
              chroma upsample inline via the crate's fused row primitives.",
    walker_doc: "Converts a YUV 4:2:0 frame by walking its rows and feeding each one\n\
                 to the [`Yuv420pSink`].\n\n\
                 The kernel is a pure row walker — no color arithmetic happens here.\n\
                 Slice math picks the Y row and the correct chroma row for each output\n\
                 row (`chroma_row = row / 2` for 4:2:0) and hands borrows to the Sink.\n\
                 The Sink decides what to derive and where to write.",
  }
}
