//! NV24 — semi‑planar 4:4:4 (`AV_PIX_FMT_NV24`).
//!
//! Layout: one full‑size Y plane + one interleaved UV plane at **full
//! width and full height**. Each UV row is `U0, V0, U1, V1, …` —
//! 2·width bytes of payload per Y row. One UV pair per Y pixel, no
//! chroma upsampling.
//!
//! Compared to [`super::Nv12`] / [`super::Nv16`]: same interleaved‑UV
//! structure, zero subsampling. Width has no parity constraint.

use crate::frame::Nv24Frame;

walker! {
  semi_planar {
    /// Zero‑sized marker for the NV24 source format. Used as the `F` type
    /// parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Nv24,
    frame: Nv24Frame<'_>,
    row: Nv24Row,
    sink: Nv24Sink,
    walker: nv24_to,
    elem_type: u8,
    chroma_field: uv,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| 2 * w,
    chroma_v: full,
    row_doc: "One output row of an NV24 source handed to an [`Nv24Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, full-width interleaved\n\
              UV) plus the row index and matrix/range carry-throughs. 1:1 with Y.",
    walker_doc: "Converts an NV24 frame by walking its rows and feeding each one to\n\
                 the [`Nv24Sink`].",
  }
}
