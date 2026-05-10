//! NV16 — semi‑planar 4:2:2 (`AV_PIX_FMT_NV16`).
//!
//! Layout: one full‑size Y plane + one interleaved UV plane at half
//! width and **full height** (one UV row per Y row, vs NV12's one UV
//! row per two Y rows). Each UV row is `U0, V0, U1, V1, …`.
//!
//! Per‑row kernel reuses [`super::Nv12`]'s `nv12_to_rgb_row` verbatim
//! — the half‑width interleaved UV layout is identical. The 4:2:0 →
//! 4:2:2 difference is purely vertical: Nv12 reads UV row `r / 2`,
//! Nv16 reads UV row `r`. The sinker calls
//! [`crate::row::nv12_to_rgb_row`] directly.

use crate::frame::Nv16Frame;

walker! {
  semi_planar {
    /// Zero‑sized marker for the NV16 source format. Used as the `F` type
    /// parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Nv16,
    frame: Nv16Frame<'_>,
    row: Nv16Row,
    sink: Nv16Sink,
    walker: nv16_to,
    elem_type: u8,
    chroma_field: uv,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| w,
    chroma_v: full,
    row_doc: "One output row of an NV16 source handed to an [`Nv16Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width interleaved\n\
              UV) plus the row index and matrix/range carry-throughs. Unlike NV12, no two\n\
              Y rows share a UV row.",
    walker_doc: "Converts an NV16 frame by walking its rows and feeding each one to\n\
                 the [`Nv16Sink`]. Chroma advances every row (vs NV12's `row / 2`).",
  }
}
