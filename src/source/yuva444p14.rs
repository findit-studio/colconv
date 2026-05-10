//! YUVA 4:4:4 planar 14‑bit. FFmpeg does not ship a `yuva444p14`
//! pixel format; this module exists for symmetry with
//! [`super::Yuv444p14`] (which the colconv 4:4:4 BITS-generic kernel
//! templates already cover) so callers can opt into 14‑bit YUVA
//! through the same per‑arch SIMD path used for the FFmpeg-shipped
//! 9 / 10 / 12 / 16 depths.
//!
//! Full‑resolution chroma + an alpha plane, 1:1 with Y. Mirrors
//! [`super::Yuv444p14`] but additionally carries a per‑row alpha slice
//! (also `width` `u16` samples, low‑bit‑packed at 14 bits).
//!
//! Ship 8b‑4 wires this format end to end. The per‑row dispatcher
//! hands the alpha source straight through to the
//! `yuv_444p_n_to_rgba*_with_alpha_src_row::<14>` SIMD/scalar path —
//! per‑arch SIMD comes free because the BITS-generic 4:4:4 template
//! already covers `BITS ∈ {9, 10, 12, 14}`, so the dispatcher selects
//! SIMD when `use_simd` is true and falls back to scalar otherwise.

use crate::frame::Yuva444pFrame16;

walker! {
  planar4_be {
    /// Zero‑sized marker for the YUVA 4:4:4 **14‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva444p14,
    frame: Yuva444pFrame16<'_, 14, BE>,
    frame_le: Yuva444pFrame16<'_, 14, false>,
    row: Yuva444p14Row,
    sink: Yuva444p14Sink,
    walker: yuva444p14_to,
    walker_endian: yuva444p14_to_endian,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuva444p14`] source.",
    walker_doc: "Walks a [`Yuva444p14Frame`] row by row into the sink.",
  }
}
