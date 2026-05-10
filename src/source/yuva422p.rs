//! YUVA 4:2:2 planar (`AV_PIX_FMT_YUVA422P`).
//!
//! Storage mirrors [`super::Yuv422p`] (Y full-size, U / V half-width
//! × full-height — 4:2:2 only subsamples chroma horizontally) plus a
//! fourth full-resolution alpha plane (1:1 with Y).
//!
//! Per-row dispatcher reuses the 4:2:0 alpha-source kernel
//! (`yuv_420_to_rgba_with_alpha_src_row`) at the row level: for any
//! given Y row the chroma layout is identical to 4:2:0 (half-width
//! U/V) — the only difference is in the vertical walker (chroma row
//! `r` vs `r / 2`).

use crate::frame::Yuva422pFrame;

walker! {
  planar4 {
    /// Zero‑sized marker for the YUVA 4:2:2 **8‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva422p,
    frame: Yuva422pFrame<'_>,
    row: Yuva422pRow,
    sink: Yuva422pSink,
    walker: yuva422p_to,
    elem_type: u8,
    chroma_h: half,
    chroma_v: full,
    row_doc: "One output row of a [`Yuva422p`] source.",
    walker_doc: "Walks a [`Yuva422pFrame`] row by row into the sink.",
  }
}
