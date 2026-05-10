//! P412 — semi‑planar 4:4:4, 12‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P412LE`, FFmpeg 5.0+).
//!
//! Same layout as [`super::P410`] but with 12 active bits in the high
//! 12 positions of each `u16` (low 4 bits zero). Per-row kernel reuses
//! the 4:4:4 `p_n_444_to_rgb_*<12>` family; chroma is full-width × full-height.

use crate::frame::PnFrame444;

walker! {
  semi_planar_be {
    /// Zero‑sized marker for the P412 source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: P412,
    frame: PnFrame444<'_, 12, BE>,
    frame_le: PnFrame444<'_, 12, false>,
    row: P412Row,
    sink: P412Sink,
    walker: p412_to,
    walker_endian: p412_to_endian,
    elem_type: u16,
    chroma_field: uv_full,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| 2 * w,
    chroma_v: full,
    row_doc: "One output row of a P412 source handed to a [`P412Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, full-width interleaved\n\
              UV — `2 * width` u16 elements) plus the row index and matrix/range\n\
              carry-throughs. Each `u16` element is high-bit-packed at 12 bits.",
    walker_doc: "Walks a [`P412Frame`] row by row into the sink.",
  }
}
