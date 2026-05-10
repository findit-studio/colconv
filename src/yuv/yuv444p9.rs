//! YUV 4:4:4 planar 9‑bit (`AV_PIX_FMT_YUV444P9LE`).
//!
//! Full-resolution chroma, 1:1 with Y. 9 active bits in the low 9 of
//! each `u16`. Niche format (AVC High 9 profile only). Reuses the
//! const-generic `yuv_444p_n_to_rgb_*<BITS>` kernel family.

use crate::frame::Yuv444pFrame16;

walker! {
  planar3_be {
    /// Zero‑sized marker for the YUV 4:4:4 **9‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv444p9,
    frame: Yuv444pFrame16<'_, 9, BE>,
    frame_le: Yuv444pFrame16<'_, 9, false>,
    row: Yuv444p9Row,
    sink: Yuv444p9Sink,
    walker: yuv444p9_to,
    walker_endian: yuv444p9_to_endian,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuv444p9`] source.",
    walker_doc: "Walks a [`Yuv444p9Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // Compile-pass regression for the codex round-1 finding on PR #110
  // (`planar3_be` arm). The macro emits an LE-only `yuv444p9_to` wrapper
  // alongside the const-generic `yuv444p9_to_endian` so explicit-turbofish
  // callers like `yuv444p9_to::<MySink>(...)` keep compiling.
  #[test]
  fn yuv444p9_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Yuv444p9Sink>() {
      let _: fn(
        &crate::frame::Yuv444p9LeFrame<'_>,
        bool,
        ColorMatrix,
        &mut S,
      ) -> Result<(), S::Error> = yuv444p9_to::<S>;
    }
  }
}
