//! NV21 — semi‑planar 4:2:0 with VU-ordered chroma (`AV_PIX_FMT_NV21`).
//!
//! Storage is identical to [`super::Nv12`] — one full-size Y plane
//! plus one interleaved chroma plane at half width and half height —
//! but the chroma bytes are **VU-ordered**: `V0, U0, V1, U1, …`
//! instead of NV12's `U0, V0, U1, V1, …`. Android MediaCodec's
//! default output for 8-bit decoded frames and some iOS camera
//! configurations emit NV21.
//!
//! Conversion semantics mirror [`super::Nv12`]: two consecutive Y
//! rows share one VU row (4:2:0), chroma is nearest-neighbor
//! upsampled in registers inside the row primitive — no intermediate
//! U / V scratch plane.

use crate::frame::Nv21Frame;

walker! {
  semi_planar {
    /// Zero-sized marker for the NV21 source format. Used as the `F`
    /// type parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Nv21,
    frame: Nv21Frame<'_>,
    row: Nv21Row,
    sink: Nv21Sink,
    walker: nv21_to,
    elem_type: u8,
    chroma_field: vu_half,
    chroma_plane: vu,
    chroma_stride: vu_stride,
    chroma_elems_per_row: |w| w,
    chroma_v: half,
    row_doc: "One output row of an NV21 source handed to an [`Nv21Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, half-width interleaved\n\
              VU — V-first) plus the row index and matrix/range carry-throughs. Row\n\
              primitives deinterleave + nearest-neighbor upsample inline.",
    walker_doc: "Converts an NV21 frame by walking its rows and feeding each one to\n\
                 the [`Nv21Sink`]. `chroma_row = row / 2` (4:2:0).",
  }
}
