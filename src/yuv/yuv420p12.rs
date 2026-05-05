//! YUV 4:2:0 planar 12‑bit (`AV_PIX_FMT_YUV420P12LE`).
//!
//! Storage mirrors [`super::Yuv420p10`] — three planes, Y at full size
//! plus U / V at half width and half height — with **`u16`** samples
//! (12 active bits in the **low** 12 of each element, upper 4 zero).
//! The [`Yuv420p12Frame`] type alias pins the bit depth; the underlying
//! [`Yuv420pFrame16`] struct is const‑generic over `BITS`, so the same
//! Q15 scalar + SIMD kernel family that powers `Yuv420p10` runs
//! unchanged against the 12‑bit instantiation.
//!
//! Ships in colconv v0.2a alongside [`super::Yuv420p14`] and
//! [`super::P012`]. Kernel semantics match [`super::Yuv420p10`].

use crate::frame::{Yuv420p12Frame, Yuv420pFrame16};

walker! {
  planar3_bits {
    /// Zero‑sized marker for the YUV 4:2:0 **12‑bit** source format. Used
    /// as the `F` type parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv420p12,
    frame: Yuv420p12Frame<'_>,
    generic_frame: Yuv420pFrame16<'_, BITS>,
    bits: 12,
    row: Yuv420p12Row,
    sink: Yuv420p12Sink,
    walker: yuv420p12_to,
    walker_inner: yuv420p12_walker,
    elem_type: u16,
    chroma_h: half,
    chroma_v: half,
    row_doc: "One output row of a 12‑bit YUV 4:2:0 source handed to a\n\
              [`Yuv420p12Sink`]. Structurally identical to [`super::Yuv420p10Row`],\n\
              just with values in `[0, 4095]` instead of `[0, 1023]`.",
    walker_doc: "Converts a 12‑bit YUV 4:2:0 frame by walking its rows and feeding\n\
                 each one to the [`Yuv420p12Sink`]. Mirrors [`super::yuv420p10_to`].",
  }
}
