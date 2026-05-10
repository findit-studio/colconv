//! Walker spec for `Gray14` (FFmpeg `gray14{le,be}`).
//!
//! The marker carries `<const BE: bool = false>`; see [`Gray9`](crate::yuv::Gray9)
//! for the full BE-flag contract.

use crate::frame::{Gray14Frame, GrayNFrame};

walker! {
  planar1_bits_be {
    /// Marker type for the `Gray14` source format (14-bit low-packed u16).
    /// `<const BE: bool>` defaults to `false` (LE).
    marker: Gray14,
    frame: Gray14Frame,
    generic_frame: GrayNFrame,
    bits: 14,
    row: Gray14Row,
    sink: Gray14Sink,
    walker: gray14_to,
    walker_inner: gray14_to_inner,
    elem_type: u16,
    row_doc: "A single row from a [`Gray14Frame`].",
    walker_doc: "Walks a [`Gray14Frame<'_, BE>`] row by row, dispatching each \
                 row to the sink. Propagates `<const BE: bool>` from the \
                 frame into [`Gray14Sink<BE>`].",
  }
}
