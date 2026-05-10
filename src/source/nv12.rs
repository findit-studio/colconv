//! NV12 — semi‑planar 4:2:0 (`AV_PIX_FMT_NV12`).
//!
//! Layout: one full‑size Y plane + one interleaved UV plane at half
//! width and half height. Each UV row is `U0, V0, U1, V1, …` (U at even
//! byte offsets, V at odd). This is the canonical 8‑bit output of
//! Apple VideoToolbox, VA‑API, NVDEC, D3D11VA, and Android MediaCodec.
//!
//! Conversion semantics mirror [`super::Yuv420p`]: two consecutive Y
//! rows share one UV row (4:2:0), chroma is nearest‑neighbor upsampled
//! **in registers** inside the row primitive — no intermediate U / V
//! scratch plane.

use crate::frame::Nv12Frame;

walker! {
  semi_planar {
    /// Zero‑sized marker for the NV12 source format. Used as the `F` type
    /// parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Nv12,
    frame: Nv12Frame<'_>,
    row: Nv12Row,
    sink: Nv12Sink,
    walker: nv12_to,
    elem_type: u8,
    chroma_field: uv_half,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| w,
    chroma_v: half,
    row_doc: "One output row of an NV12 source handed to an [`Nv12Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width interleaved\n\
              UV) plus the row index and matrix/range carry-throughs. Row primitives\n\
              deinterleave + nearest-neighbor upsample inline.",
    walker_doc: "Converts an NV12 frame by walking its rows and feeding each one to\n\
                 the [`Nv12Sink`]. `chroma_row = row / 2` (4:2:0).",
  }
}
