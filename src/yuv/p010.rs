//! P010 — semi‑planar 4:2:0, 10‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P010LE`).
//!
//! Storage is a 2‑plane layout: one full‑size Y plane plus one
//! interleaved UV plane at half width and half height. Sample width
//! is `u16` with the 10 active bits in the **high** 10 positions of
//! each element (`sample = value << 6`), low 6 bits zero. This is
//! Microsoft's P010 convention and what every HDR hardware decoder
//! emits — Apple VideoToolbox, VA‑API, NVDEC, D3D11VA, Intel QSV.
//!
//! Conversion semantics mirror [`super::Nv12`] on the layout side and
//! [`super::Yuv420p10`] on the Q‑math side: two consecutive Y rows
//! share one UV row (4:2:0), chroma is nearest‑neighbor upsampled in
//! registers inside the row primitive, and every SIMD backend shifts
//! each `u16` load right by 6 to extract the 10‑bit value before
//! running the same Q15 pipeline used by [`super::Yuv420p10`].

use crate::frame::PnFrame;

walker! {
  semi_planar_be {
    /// Zero‑sized marker for the P010 source format. Used as the `F` type
    /// parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: P010,
    frame: PnFrame<'_, 10, BE>,
    row: P010Row,
    sink: P010Sink,
    walker: p010_to,
    elem_type: u16,
    chroma_field: uv_half,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| w,
    chroma_v: half,
    row_doc: "One output row of a P010 source handed to a [`P010Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width interleaved\n\
              UV) plus the row index and matrix/range carry-throughs. Each `u16` element\n\
              is high-bit-packed (10 active bits in the high 10 of each element).",
    walker_doc: "Converts a P010 frame by walking its rows and feeding each one to\n\
                 the [`P010Sink`]. `chroma_row = row / 2` (4:2:0).",
  }
}
