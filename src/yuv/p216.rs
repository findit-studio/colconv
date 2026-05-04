//! P216 — semi‑planar 4:2:2, 16‑bit (`AV_PIX_FMT_P216LE`).
//!
//! 4:2:2 twin of [`super::P016`]. At 16 bits the high-vs-low packing
//! distinction degenerates — every bit is active. Per-row kernel
//! reuses the 4:2:0 `p16_to_rgb_*` parallel i64-chroma family
//! verbatim; only the walker reads chroma row `r` instead of `r / 2`.

use crate::frame::P216Frame;

walker! {
  semi_planar {
    /// Zero‑sized marker for the P216 source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: P216,
    frame: P216Frame<'_>,
    row: P216Row,
    sink: P216Sink,
    walker: p216_to,
    elem_type: u16,
    chroma_field: uv_half,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| w,
    chroma_v: full,
    row_doc: "One output row of a P216 source handed to a [`P216Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width interleaved\n\
              UV — full-height) plus the row index and matrix/range carry-throughs. All\n\
              16 bits of each `u16` element are active.",
    walker_doc: "Walks a [`P216Frame`] row by row into the sink. Each Y row has its\n\
                 own corresponding UV row (4:2:2 — full-height chroma).",
  }
}
