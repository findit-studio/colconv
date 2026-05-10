//! P016 — semi‑planar 4:2:0, 16‑bit (`AV_PIX_FMT_P016LE`).
//!
//! Storage is identical to [`super::P010`] / [`super::P012`]: one
//! full‑size Y plane plus one interleaved UV plane at half width and
//! half height. At 16 bits there is no high‑vs‑low distinction — the
//! full `u16` range is active, so `P016` and a hypothetical
//! `yuv420p16le`‑shaped `PnFrame<16>` are numerically identical (the
//! layout difference is only in the plane count / interleave, not
//! sample packing).
//!
//! Runs on the **parallel i64 kernel family** —
//! [`crate::row::p016_to_rgb_row`] dispatches to
//! `scalar::p16_to_rgb_*` plus the matching per-backend SIMD kernels,
//! which widen the chroma matrix multiply to i64.

use crate::frame::PnFrame;

walker! {
  semi_planar_be {
    /// Zero‑sized marker for the P016 source format. Used as the `F` type
    /// parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: P016,
    frame: PnFrame<'_, 16, BE>,
    row: P016Row,
    sink: P016Sink,
    walker: p016_to,
    elem_type: u16,
    chroma_field: uv_half,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| w,
    chroma_v: half,
    row_doc: "One output row of a P016 source handed to a [`P016Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width interleaved\n\
              UV) plus the row index and matrix/range carry-throughs. All 16 bits of\n\
              each `u16` element are active.",
    walker_doc: "Converts a P016 frame by walking its rows and feeding each one to\n\
                 the [`P016Sink`]. `chroma_row = row / 2` (4:2:0).",
  }
}
