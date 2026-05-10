//! YUVA 4:4:4 planar 10‑bit (`AV_PIX_FMT_YUVA444P10LE`).
//!
//! Full‑resolution chroma + an alpha plane, 1:1 with Y. Mirrors
//! [`super::Yuv444p10`] but additionally carries a per‑row alpha slice
//! (also `width` `u16` samples, low‑bit‑packed at 10 bits).
//!
//! Tranche 8b‑1a ships the scalar prep — the per‑row dispatcher hands
//! the alpha source straight through to the
//! `yuv_444p_n_to_rgba*_with_alpha_src_row::<10>` scalar path. Per‑arch
//! SIMD wiring lands in 8b‑1b (`u8` RGBA) and 8b‑1c (`u16` RGBA).

use crate::frame::Yuva444pFrame16;

walker! {
  planar4_be {
    /// Zero‑sized marker for the YUVA 4:4:4 **10‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva444p10,
    frame: Yuva444pFrame16<'_, 10, BE>,
    row: Yuva444p10Row,
    sink: Yuva444p10Sink,
    walker: yuva444p10_to,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuva444p10`] source.",
    walker_doc: "Walks a [`Yuva444p10Frame`] row by row into the sink.",
  }
}
