//! Walker spec for `Gray12` (FFmpeg `gray12{le,be}`).
//!
//! The marker carries `<const BE: bool = false>`; see [`Gray9`](crate::yuv::Gray9)
//! for the full BE-flag contract.

use crate::frame::{Gray12Frame, GrayNFrame};

walker! {
  planar1_bits_be {
    /// Marker type for the `Gray12` source format (12-bit low-packed u16).
    /// `<const BE: bool>` defaults to `false` (LE).
    marker: Gray12,
    frame: Gray12Frame,
    generic_frame: GrayNFrame,
    bits: 12,
    row: Gray12Row,
    sink: Gray12Sink,
    walker: gray12_to,
    walker_inner: gray12_to_inner,
    elem_type: u16,
    row_doc: "A single row from a [`Gray12Frame`].",
    walker_doc: "Walks a [`Gray12Frame<'_, BE>`] row by row, dispatching each \
                 row to the sink. Propagates `<const BE: bool>` from the \
                 frame into [`Gray12Sink<BE>`].",
  }
}
