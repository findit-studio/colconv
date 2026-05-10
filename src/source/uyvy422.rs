//! Packed YUV 4:2:2 source (`AV_PIX_FMT_UYVY422`). One plane, byte
//! order `U0, Y0, V0, Y1` per 2-pixel block — Y in odd byte
//! positions, U/V in even positions.
//!
//! De-facto SDI capture format on Apple QuickTime / VideoToolbox
//! 8-bit paths, also widely emitted by professional capture cards
//! in 8-bit mode.
//!
//! Reuses the same const-generic packed-YUV-422 → RGB kernel
//! template as [`super::Yuyv422`] / [`super::Yvyu422`]; the only
//! difference is Y / UV byte positions, selected at compile time.

use crate::frame::Uyvy422Frame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **UYVY422** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Uyvy422,
    frame: Uyvy422Frame<'_>,
    row: Uyvy422Row,
    sink: Uyvy422Sink,
    walker: uyvy422_to,
    buf_field: uyvy,
    elem_type: u8,
    row_elems: |w| w * 2,
    row_doc: "One output row of a [`Uyvy422`] source — `2 * width` packed\n\
              `U0, Y0, V0, Y1, …` bytes.",
    walker_doc: "Walks a [`Uyvy422Frame`] row by row into the sink.",
  }
}
