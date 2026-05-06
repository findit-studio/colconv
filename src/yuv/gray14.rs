//! Walker spec for `Gray14` (FFmpeg `gray14le`).

use crate::frame::{Gray14Frame, GrayNFrame};

walker! {
  planar1_bits {
    /// Marker type for the `Gray14` source format (14-bit low-packed u16).
    marker: Gray14,
    frame: Gray14Frame<'_>,
    generic_frame: GrayNFrame<'_, BITS>,
    bits: 14,
    row: Gray14Row,
    sink: Gray14Sink,
    walker: gray14_to,
    walker_inner: gray14_to_inner,
    elem_type: u16,
    row_doc: "A single row from a [`Gray14Frame`].",
    walker_doc: "Walks a [`Gray14Frame`] row by row, dispatching each row to the sink.",
  }
}
