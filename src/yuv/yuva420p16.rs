//! YUVA 4:2:0 planar 16‑bit (`AV_PIX_FMT_YUVA420P16LE`).
//!
//! Storage mirrors [`super::Yuv420p16`] — three planes for Y / U / V
//! at the standard 4:2:0 layout (Y full-size, U / V half-width × half-
//! height) — plus a fourth full-resolution alpha plane (1:1 with Y;
//! only chroma is subsampled in 4:2:0). Sample width is **`u16`**.
//! At 16 bits there is no upper-bit-zero slack; the full `u16` range
//! is active.
//!
//! Runs on the **parallel i64 kernel family** for the u16 RGBA path
//! (Q15 chroma sum overflows i32 at 16 bits); the u8 RGBA path stays
//! on the i32 pipeline (output-range scaling keeps `coeff × u_d`
//! inside i32). 9/10‑bit YUVA siblings use the Q15 i32 family for
//! both u8 and u16 outputs.
//!
//! Tranche 8b‑2a ships the scalar prep — the per‑row dispatcher hands
//! the alpha source straight through to the
//! `yuv_420p16_to_rgba*_with_alpha_src_row` scalar paths. Per‑arch
//! SIMD wiring lands in 8b‑2b (`u8` RGBA) and 8b‑2c (`u16` RGBA).

use crate::frame::{Yuva420p16Frame, Yuva420pFrame16};

walker! {
  planar4_bits {
    /// Zero‑sized marker for the YUVA 4:2:0 **16‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva420p16,
    frame: Yuva420p16Frame<'_>,
    generic_frame: Yuva420pFrame16<'_, BITS>,
    bits: 16,
    row: Yuva420p16Row,
    sink: Yuva420p16Sink,
    walker: yuva420p16_to,
    walker_inner: yuva420p16_walker,
    elem_type: u16,
    chroma_h: half,
    chroma_v: half,
    row_doc: "One output row of a [`Yuva420p16`] source.",
    walker_doc: "Walks a [`Yuva420p16Frame`] row by row into the sink.",
  }
}
