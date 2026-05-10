//! P210 — semi‑planar 4:2:2, 10‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P210LE`).
//!
//! 4:2:2 twin of [`super::P010`]: same Y + interleaved-UV plane shape
//! and same high-bit-packed `u16` convention (10 active bits in the
//! high 10 positions, low 6 zero), but chroma is **full-height** —
//! one chroma row per Y row instead of one per two. NVDEC / CUDA HDR
//! 4:2:2 download target and some QSV configurations.
//!
//! Per-row kernel reuses the 4:2:0 `p_n_to_rgb_*<10>` family verbatim
//! (the per-row UV layout is identical to P010 — half-width
//! interleaved); only the walker reads chroma row `r` instead of
//! `r / 2`.

use crate::frame::PnFrame422;

walker! {
  semi_planar_be {
    /// Zero‑sized marker for the P210 source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: P210,
    frame: PnFrame422<'_, 10, BE>,
    row: P210Row,
    sink: P210Sink,
    walker: p210_to,
    elem_type: u16,
    chroma_field: uv_half,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| w,
    chroma_v: full,
    row_doc: "One output row of a P210 source handed to a [`P210Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width interleaved\n\
              UV — full-height, one UV row per Y row) plus the row index and\n\
              matrix/range carry-throughs. Each `u16` element is high-bit-packed.",
    walker_doc: "Walks a [`P210Frame`] row by row into the sink. Each Y row has its\n\
                 own corresponding UV row (4:2:2 — full-height chroma).",
  }
}
