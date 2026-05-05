//! YUV 4:2:0 planar 10‑bit (`AV_PIX_FMT_YUV420P10LE`).
//!
//! Storage mirrors [`super::Yuv420p`] — three planes, Y at full size
//! plus U / V at half width and half height — but sample width is
//! **`u16`** (10 active bits in the low bits of each element). The
//! [`Yuv420p10Frame`] type alias pins the bit depth; the underlying
//! [`Yuv420pFrame16`] struct is const‑generic over `BITS` and the
//! 12‑bit / 14‑bit siblings ([`super::Yuv420p12`] / [`super::Yuv420p14`])
//! reuse the same scalar + SIMD kernel family with a different
//! monomorphization.
//!
//! Kernel semantics match [`super::Yuv420p`]: two consecutive Y rows
//! share one chroma row (4:2:0), chroma is nearest‑neighbor upsampled
//! in registers inside the row primitive.

use crate::frame::{Yuv420p10Frame, Yuv420pFrame16};

walker! {
  planar3_bits {
    /// Zero‑sized marker for the YUV 4:2:0 **10‑bit** source format. Used
    /// as the `F` type parameter on [`crate::sinker::MixedSinker`].
    ///
    /// 12‑bit and 14‑bit siblings ship as separate markers
    /// ([`super::Yuv420p12`] / [`super::Yuv420p14`]) on the same
    /// [`Yuv420pFrame16`] struct with different `BITS` values. 16‑bit
    /// needs a different kernel family (Q15 chroma_sum overflows i32) and
    /// is not yet shipped.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv420p10,
    frame: Yuv420p10Frame<'_>,
    generic_frame: Yuv420pFrame16<'_, BITS>,
    bits: 10,
    row: Yuv420p10Row,
    sink: Yuv420p10Sink,
    walker: yuv420p10_to,
    walker_inner: yuv420p10_walker,
    elem_type: u16,
    chroma_h: half,
    chroma_v: half,
    row_doc: "One output row of a 10‑bit YUV 4:2:0 source handed to a\n\
              [`Yuv420p10Sink`]. Structurally identical to [`super::Yuv420pRow`],\n\
              just `u16` samples.",
    walker_doc: "Converts a 10‑bit YUV 4:2:0 frame by walking its rows and feeding\n\
                 each one to the [`Yuv420p10Sink`]. See [`super::yuv420p_to`] for\n\
                 the shared design rationale.",
  }
}
