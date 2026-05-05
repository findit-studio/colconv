//! YUVA 4:4:4 planar (`AV_PIX_FMT_YUVA444P`) — 8 bit per sample.
//!
//! Storage mirrors [`super::Yuv444p`] (Y / U / V each full-resolution
//! `u8`) plus a fourth full-resolution alpha plane (1:1 with Y).
//!
//! Per-row dispatcher hands the alpha source straight through to the
//! `yuv_444_to_rgba_with_alpha_src_row` SIMD/scalar paths — same shape
//! as the 4:2:0 sibling [`super::Yuva420p`]. Per-arch SIMD coverage is
//! shipped together with the format wiring (Ship 8b‑6).

use crate::frame::Yuva444pFrame;

walker! {
  planar4 {
    /// Zero‑sized marker for the YUVA 4:4:4 **8‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva444p,
    frame: Yuva444pFrame<'_>,
    row: Yuva444pRow,
    sink: Yuva444pSink,
    walker: yuva444p_to,
    elem_type: u8,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuva444p`] source.",
    walker_doc: "Walks a [`Yuva444pFrame`] row by row into the sink.",
  }
}
