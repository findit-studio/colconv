//! YUVA 4:2:0 planar (`AV_PIX_FMT_YUVA420P`).
//!
//! Storage mirrors [`super::Yuv420p`] — three planes for Y / U / V at
//! the standard 4:2:0 layout (Y full-size, U / V half-width × half-
//! height) — plus a fourth full-resolution alpha plane (1:1 with Y;
//! only chroma is subsampled in 4:2:0).
//!
//! Tranche 8b‑2a ships the scalar prep — the per‑row dispatcher hands
//! the alpha source straight through to the
//! `yuv_420_to_rgba_with_alpha_src_row` scalar path. Per‑arch SIMD
//! wiring lands in 8b‑2b (`u8` RGBA).

use crate::frame::Yuva420pFrame;

walker! {
  planar4 {
    /// Zero‑sized marker for the YUVA 4:2:0 **8‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva420p,
    frame: Yuva420pFrame<'_>,
    row: Yuva420pRow,
    sink: Yuva420pSink,
    walker: yuva420p_to,
    elem_type: u8,
    chroma_h: half,
    chroma_v: half,
    row_doc: "One output row of a [`Yuva420p`] source.\n\n\
              Y / U / V follow the 4:2:0 chroma-pair convention (two consecutive\n\
              Y rows share one U/V row); A is full-resolution (one alpha row per\n\
              Y row).",
    walker_doc: "Walks a [`Yuva420pFrame`] row by row into the sink.",
  }
}
