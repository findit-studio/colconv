//! YUVA 4:4:4 planar 16‑bit (`AV_PIX_FMT_YUVA444P16LE`).
//!
//! Storage mirrors [`super::Yuv444p16`] (Y / U / V each full-resolution,
//! `u16` samples — at 16 bits there is no upper-bit-zero slack; the
//! full `u16` range is active) plus a fourth full-resolution alpha
//! plane (1:1 with Y).
//!
//! For the native-depth `u16` output path, this uses the **dedicated
//! i64 4:4:4 kernel family** because the Q15 chroma sum overflows
//! i32 at 16 bits. The `u8` output path stays on the scaled Q15 i32
//! route (output-target scaling keeps `coeff × u_d` inside i32).
//! Either way it sits separate from the BITS-generic Q15 i32 template
//! that covers `BITS ∈ {9, 10, 12, 14}`. Mirrors the 4:2:0 sibling
//! [`super::Yuva420p16`].
//!
//! Tranche 8b‑5a ships the scalar prep — the per‑row dispatcher hands
//! the alpha source straight through to the
//! `yuv_444p16_to_rgba*_with_alpha_src_row` scalar paths. Per‑arch
//! SIMD wiring lands in 8b‑5b (`u8` RGBA) and 8b‑5c (`u16` RGBA).

use crate::frame::Yuva444pFrame16;

walker! {
  planar4_be {
    /// Zero‑sized marker for the YUVA 4:4:4 **16‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuva444p16,
    frame: Yuva444pFrame16<'_, 16, BE>,
    row: Yuva444p16Row,
    sink: Yuva444p16Sink,
    walker: yuva444p16_to,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuva444p16`] source.",
    walker_doc: "Walks a [`Yuva444p16Frame`] row by row into the sink.",
  }
}
