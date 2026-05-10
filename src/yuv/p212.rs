//! P212 — semi‑planar 4:2:2, 12‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P212LE`, FFmpeg 5.0+).
//!
//! Same layout as [`super::P210`] but with 12 active bits in the high
//! 12 positions of each `u16` (low 4 bits zero). Per-row kernel reuses
//! the 4:2:0 `p_n_to_rgb_*<12>` family verbatim; only the walker
//! reads chroma row `r` instead of `r / 2` (4:2:2 vs 4:2:0).

use crate::frame::PnFrame422;

walker! {
  semi_planar_be {
    /// Zero‑sized marker for the P212 source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: P212,
    frame: PnFrame422<'_, 12, BE>,
    frame_le: PnFrame422<'_, 12, false>,
    row: P212Row,
    sink: P212Sink,
    walker: p212_to,
    walker_endian: p212_to_endian,
    elem_type: u16,
    chroma_field: uv_half,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| w,
    chroma_v: full,
    row_doc: "One output row of a P212 source handed to a [`P212Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width interleaved\n\
              UV — full-height) plus the row index and matrix/range carry-throughs. Each\n\
              `u16` element is high-bit-packed at 12 bits.",
    walker_doc: "Walks a [`P212Frame`] row by row into the sink. Each Y row has its\n\
                 own corresponding UV row (4:2:2 — full-height chroma).",
  }
}
