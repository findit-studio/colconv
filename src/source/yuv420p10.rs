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

use crate::frame::Yuv420pFrame16;

walker! {
  planar3_bits_be {
    /// Zero‑sized marker for the YUV 4:2:0 **10‑bit** source format. Used
    /// as the `F` type parameter on [`crate::sinker::MixedSinker`].
    ///
    /// Phase 4 — `<const BE: bool = false>` selects LE vs BE plane bytes
    /// (`AV_PIX_FMT_YUV420P10LE` vs `AV_PIX_FMT_YUV420P10BE`). The
    /// default keeps existing LE-only callers compiling unchanged.
    ///
    /// 12‑bit and 14‑bit siblings ship as separate markers
    /// ([`super::Yuv420p12`] / [`super::Yuv420p14`]) on the same
    /// [`Yuv420pFrame16`] struct with different `BITS` values. 16‑bit
    /// uses a different kernel family (Q15 chroma_sum overflows i32 and
    /// gets a parallel i64 path).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv420p10,
    frame: Yuv420pFrame16<'_, 10, BE>,
    frame_le: Yuv420pFrame16<'_, 10, false>,
    generic_frame: Yuv420pFrame16<'_, BITS, BE>,
    bits: 10,
    row: Yuv420p10Row,
    sink: Yuv420p10Sink,
    walker: yuv420p10_to,
    walker_endian: yuv420p10_to_endian,
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

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // Compile-pass regression for the codex round-1 finding on PR #110
  // (`planar3_bits_be` arm). The macro emits an LE-only `yuv420p10_to`
  // wrapper alongside the const-generic `yuv420p10_to_endian` so
  // explicit-turbofish callers like `yuv420p10_to::<MySink>(...)` keep
  // compiling.
  #[test]
  fn yuv420p10_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Yuv420p10Sink>() {
      let _: fn(
        &crate::frame::Yuv420p10LeFrame<'_>,
        bool,
        ColorMatrix,
        &mut S,
      ) -> Result<(), S::Error> = yuv420p10_to::<S>;
    }
  }
}
