//! P416 — semi‑planar 4:4:4, 16‑bit (`AV_PIX_FMT_P416LE`).
//!
//! 4:4:4 twin of [`super::P016`]. At 16 bits every bit is active.
//! Per-row kernel uses the parallel i64-chroma `p_n_444_16_to_rgb_*`
//! family (chroma matrix multiply-add overflows i32 at 16 bits, same
//! rationale as `p16_to_rgb_*` and `yuv444p16_to_rgb_*`).

use crate::frame::PnFrame444;

walker! {
  semi_planar_be {
    /// Zero‑sized marker for the P416 source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: P416,
    frame: PnFrame444<'_, 16, BE>,
    frame_le: PnFrame444<'_, 16, false>,
    row: P416Row,
    sink: P416Sink,
    walker: p416_to,
    walker_endian: p416_to_endian,
    elem_type: u16,
    chroma_field: uv_full,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| 2 * w,
    chroma_v: full,
    row_doc: "One output row of a P416 source handed to a [`P416Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, full-width interleaved\n\
              UV — `2 * width` u16 elements) plus the row index and matrix/range\n\
              carry-throughs. All 16 bits of each `u16` element are active.",
    walker_doc: "Walks a [`P416Frame`] row by row into the sink.",
  }
}
