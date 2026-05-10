//! YUVA 4:2:2 planar 10‑bit (`AV_PIX_FMT_YUVA422P10LE`).
//!
//! Storage mirrors [`super::Yuv422p10`] (Y full-width × full-height,
//! U / V half-width × full-height) plus a fourth full-resolution
//! alpha plane (1:1 with Y; only chroma is subsampled in 4:2:2).
//! Sample width is **`u16`** (10 active bits in the low bits of each
//! element).
//!
//! Per-row dispatcher reuses
//! `yuv_420p_n_to_rgba*_with_alpha_src_row::<10>` (in `crate::row`) at
//! the row level — chroma layout for any single Y row is identical to
//! 4:2:0 (half-width U/V); the 4:2:0 vs 4:2:2 difference is purely in
//! the vertical walker.

use crate::frame::Yuva422pFrame16;

walker! {
  planar4_bits_be {
    /// Zero‑sized marker for the YUVA 4:2:2 **10‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva422p10,
    frame: Yuva422pFrame16<'_, 10, BE>,
    generic_frame: Yuva422pFrame16<'_, BITS, BE>,
    bits: 10,
    row: Yuva422p10Row,
    sink: Yuva422p10Sink,
    walker: yuva422p10_to,
    walker_inner: yuva422p10_walker,
    elem_type: u16,
    chroma_h: half,
    chroma_v: full,
    row_doc: "One output row of a [`Yuva422p10`] source.",
    walker_doc: "Walks a [`Yuva422p10Frame`] row by row into the sink.",
  }
}
