//! YUVA 4:2:0 planar 10‑bit (`AV_PIX_FMT_YUVA420P10LE`).
//!
//! Storage mirrors [`super::Yuv420p10`] — three planes for Y / U / V at
//! the standard 4:2:0 layout (Y full-size, U / V half-width × half-
//! height) — plus a fourth full-resolution alpha plane (1:1 with Y;
//! only chroma is subsampled in 4:2:0). Sample width is **`u16`**
//! (10 active bits in the low bits of each element).
//!
//! Tranche 8b‑2a ships the scalar prep — the per‑row dispatcher hands
//! the alpha source straight through to the
//! `yuv_420p_n_to_rgba*_with_alpha_src_row::<10>` scalar path. Per‑arch
//! SIMD wiring lands in 8b‑2b (`u8` RGBA) and 8b‑2c (`u16` RGBA).

use crate::frame::Yuva420pFrame16;

walker! {
  planar4_bits_be {
    /// Zero‑sized marker for the YUVA 4:2:0 **10‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva420p10,
    frame: Yuva420pFrame16<'_, 10, BE>,
    generic_frame: Yuva420pFrame16<'_, BITS, BE>,
    bits: 10,
    row: Yuva420p10Row,
    sink: Yuva420p10Sink,
    walker: yuva420p10_to,
    walker_inner: yuva420p10_walker,
    elem_type: u16,
    chroma_h: half,
    chroma_v: half,
    row_doc: "One output row of a [`Yuva420p10`] source.",
    walker_doc: "Walks a [`Yuva420p10Frame`] row by row into the sink.",
  }
}
