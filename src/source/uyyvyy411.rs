//! Packed YUV 4:1:1 source (`AV_PIX_FMT_UYYVYY411`). One plane,
//! 6 bytes per 4-pixel block, byte order
//! `U0, Y0, Y1, V0, Y2, Y3` — one (U, V) chroma pair shared by four
//! luma samples (4:1:1 horizontal subsampling, 12 bpp).
//!
//! Common in DV 4:1:1 NTSC capture (legacy). Tier 5.25 P3 format.
//!
//! Structurally analogous to [`super::Uyvy422`] (one chroma pair
//! shared by 2 luma samples) but with chroma horizontal subsampling
//! at 4× instead of 2×. Reuses the `walker!(packed { ... })` macro
//! for marker / row / sink / walker boilerplate.

use crate::frame::Uyyvyy411Frame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **UYYVYY411** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Uyyvyy411,
    frame: Uyyvyy411Frame<'_>,
    row: Uyyvyy411Row,
    sink: Uyyvyy411Sink,
    walker: uyyvyy411_to,
    buf_field: uyyvyy,
    elem_type: u8,
    row_elems: |w| w * 3 / 2,
    row_doc: "One output row of a [`Uyyvyy411`] source — `width * 3 / 2` packed\n\
              `U0, Y0, Y1, V0, Y2, Y3, …` bytes (12 bpp, 4 pixels per 6-byte\n\
              block). Width is a multiple of 4.",
    walker_doc: "Walks a [`Uyyvyy411Frame`] row by row into the sink.",
  }
}
