//! YUV 4:2:0 planar 14‑bit (`AV_PIX_FMT_YUV420P14LE`).
//!
//! Storage mirrors [`super::Yuv420p10`] — three planes, Y at full size
//! plus U / V at half width and half height — with **`u16`** samples
//! (14 active bits in the **low** 14 of each element, upper 2 zero).
//! The [`Yuv420p14Frame`] type alias pins the bit depth; the underlying
//! [`Yuv420pFrame16`] struct is const‑generic over `BITS`, so the same
//! Q15 scalar + SIMD kernel family that powers `Yuv420p10` /
//! `Yuv420p12` runs unchanged against the 14‑bit instantiation.
//!
//! Kernel math constraint: at 14 bits, chroma_sum still fits in i32
//! (~10⁹ ≤ 2³¹), so the Q15 pipeline stays unchanged. 16‑bit would
//! overflow and needs a separate kernel family.

use crate::frame::Yuv420pFrame16;

walker! {
  planar3_bits_be {
    /// Zero‑sized marker for the YUV 4:2:0 **14‑bit** source format. Used
    /// as the `F` type parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv420p14,
    frame: Yuv420pFrame16<'_, 14, BE>,
    frame_le: Yuv420pFrame16<'_, 14, false>,
    generic_frame: Yuv420pFrame16<'_, BITS, BE>,
    bits: 14,
    row: Yuv420p14Row,
    sink: Yuv420p14Sink,
    walker: yuv420p14_to,
    walker_endian: yuv420p14_to_endian,
    walker_inner: yuv420p14_walker,
    elem_type: u16,
    chroma_h: half,
    chroma_v: half,
    row_doc: "One output row of a 14‑bit YUV 4:2:0 source handed to a\n\
              [`Yuv420p14Sink`]. Structurally identical to [`super::Yuv420p12Row`],\n\
              just with values in `[0, 16383]` instead of `[0, 4095]`.",
    walker_doc: "Converts a 14‑bit YUV 4:2:0 frame by walking its rows and feeding\n\
                 each one to the [`Yuv420p14Sink`]. Mirrors [`super::yuv420p12_to`].",
  }
}
