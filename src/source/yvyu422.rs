//! Packed YUV 4:2:2 source (`AV_PIX_FMT_YVYU422`). One plane, byte
//! order `Y0, V0, Y1, U0` per 2-pixel block — same Y positions
//! as YUYV422 but with V/U swapped (V precedes U).
//!
//! Common on Android camera HAL outputs and a small handful of
//! older capture devices.
//!
//! Reuses the same const-generic packed-YUV-422 → RGB kernel
//! template as [`super::Yuyv422`] / [`super::Uyvy422`]; the only
//! difference from YUYV is the UV byte order, selected via the
//! `SWAP_UV` const generic at compile time.

use crate::frame::Yvyu422Frame;

walker! {
  packed {
    /// Zero‑sized marker for the packed **YVYU422** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yvyu422,
    frame: Yvyu422Frame<'_>,
    row: Yvyu422Row,
    sink: Yvyu422Sink,
    walker: yvyu422_to,
    buf_field: yvyu,
    elem_type: u8,
    row_elems: |w| w * 2,
    row_doc: "One output row of a [`Yvyu422`] source — `2 * width` packed\n\
              `Y0, V0, Y1, U0, …` bytes.",
    walker_doc: "Walks a [`Yvyu422Frame`] row by row into the sink.",
  }
}
