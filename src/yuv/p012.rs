//! P012 — semi‑planar 4:2:0, 12‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P012LE`).
//!
//! Storage is a 2‑plane layout identical to [`super::P010`]: one full‑
//! size Y plane plus one interleaved UV plane at half width and half
//! height. Sample width is `u16` with the 12 active bits in the
//! **high** 12 positions of each element (`sample = value << 4`), low
//! 4 bits zero. This is the 12‑bit sibling of Microsoft's P010
//! convention and what HEVC Main 12 / VP9 Profile 3 hardware decoders
//! emit.
//!
//! Conversion semantics mirror [`super::P010`] on the layout side and
//! [`super::Yuv420p12`] on the Q‑math side.

use crate::frame::P012Frame;

walker! {
  semi_planar {
    /// Zero‑sized marker for the P012 source format. Used as the `F` type
    /// parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: P012,
    frame: P012Frame<'_>,
    row: P012Row,
    sink: P012Sink,
    walker: p012_to,
    elem_type: u16,
    chroma_field: uv_half,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| w,
    chroma_v: half,
    row_doc: "One output row of a P012 source handed to a [`P012Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width interleaved\n\
              UV) plus the row index and matrix/range carry-throughs. Each `u16` element\n\
              is high-bit-packed (12 active bits in the high 12 of each element).",
    walker_doc: "Converts a P012 frame by walking its rows and feeding each one to\n\
                 the [`P012Sink`]. `chroma_row = row / 2` (4:2:0).",
  }
}
