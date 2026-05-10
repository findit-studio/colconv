//! YUVA 4:4:4 planar 12‑bit (`AV_PIX_FMT_YUVA444P12LE`).
//!
//! Full‑resolution chroma + an alpha plane, 1:1 with Y. Mirrors
//! [`super::Yuv444p12`] but additionally carries a per‑row alpha slice
//! (also `width` `u16` samples, low‑bit‑packed at 12 bits).
//!
//! Ship 8b‑4 wires this format end to end. The per‑row dispatcher
//! hands the alpha source straight through to the
//! `yuv_444p_n_to_rgba*_with_alpha_src_row::<12>` SIMD/scalar path —
//! per‑arch SIMD comes free because the BITS-generic 4:4:4 template
//! already covers `BITS ∈ {9, 10, 12, 14}`, so the dispatcher selects
//! SIMD when `use_simd` is true and falls back to scalar otherwise.

use crate::frame::Yuva444pFrame16;

walker! {
  planar4_be {
    /// Zero‑sized marker for the YUVA 4:4:4 **12‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva444p12,
    frame: Yuva444pFrame16<'_, 12, BE>,
    row: Yuva444p12Row,
    sink: Yuva444p12Sink,
    walker: yuva444p12_to,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuva444p12`] source.",
    walker_doc: "Walks a [`Yuva444p12Frame`] row by row into the sink.",
  }
}
