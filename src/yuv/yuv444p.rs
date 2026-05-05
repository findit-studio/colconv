//! YUV 4:4:4 planar (`AV_PIX_FMT_YUV444P`, `yuvj444p`).
//!
//! Three planes, all full-size. One UV pair per Y pixel, no chroma
//! subsampling. Per-row kernel math is the same 4:4:4 arithmetic
//! used by [`super::Nv24`] / [`super::Nv42`] — one `u` sample and
//! one `v` sample per pixel — but U and V come from separate planes
//! instead of an interleaved UV / VU plane.

use crate::frame::Yuv444pFrame;

walker! {
  planar3 {
    /// Zero-sized marker for the YUV 4:4:4 source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv444p,
    frame: Yuv444pFrame<'_>,
    row: Yuv444pRow,
    sink: Yuv444pSink,
    walker: yuv444p_to,
    elem_type: u8,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a YUV 4:4:4 source handed to a [`Yuv444pSink`].\n\n\
              U and V are full-width (1:1 with Y) — no chroma subsampling.",
    walker_doc: "Converts a YUV 4:4:4 frame by walking its rows and feeding each\n\
                 one to the [`Yuv444pSink`].",
  }
}
