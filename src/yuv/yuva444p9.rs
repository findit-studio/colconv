//! YUVA 4:4:4 planar 9‑bit (`AV_PIX_FMT_YUVA444P9LE`).
//!
//! Full‑resolution chroma + an alpha plane, 1:1 with Y. Mirrors
//! [`super::Yuv444p9`] but additionally carries a per‑row alpha slice
//! (also `width` `u16` samples, low‑bit‑packed at 9 bits).
//!
//! Ship 8b‑3 wires this format end to end. The per‑row dispatcher
//! hands the alpha source straight through to the
//! `yuv_444p_n_to_rgba*_with_alpha_src_row::<9>` SIMD/scalar path —
//! per‑arch SIMD comes free because the BITS-generic template
//! already covers `BITS ∈ {9, 10, 12, 14}` for the existing 4:4:4
//! kernels, so the dispatcher selects SIMD when `use_simd` is true
//! and falls back to scalar otherwise.

use crate::frame::Yuva444pFrame16;

walker! {
  planar4_be {
    /// Zero‑sized marker for the YUVA 4:4:4 **9‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva444p9,
    frame: Yuva444pFrame16<'_, 9, BE>,
    frame_le: Yuva444pFrame16<'_, 9, false>,
    row: Yuva444p9Row,
    sink: Yuva444p9Sink,
    walker: yuva444p9_to,
    walker_endian: yuva444p9_to_endian,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuva444p9`] source.",
    walker_doc: "Walks a [`Yuva444p9Frame`] row by row into the sink.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // Compile-pass regression for the codex round-1 finding on PR #110
  // (`planar4_be` arm). The macro emits an LE-only `yuva444p9_to` wrapper
  // alongside the const-generic `yuva444p9_to_endian` so explicit-turbofish
  // callers like `yuva444p9_to::<MySink>(...)` keep compiling.
  #[test]
  fn yuva444p9_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Yuva444p9Sink>() {
      let _: fn(
        &crate::frame::Yuva444p9LeFrame<'_>,
        bool,
        ColorMatrix,
        &mut S,
      ) -> Result<(), S::Error> = yuva444p9_to::<S>;
    }
  }
}
