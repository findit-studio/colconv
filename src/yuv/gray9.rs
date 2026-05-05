//! Walker spec for `Gray9` (FFmpeg `gray9le`).

use crate::frame::{Gray9Frame, GrayNFrame};

walker! {
  planar1_bits {
    /// Marker type for the `Gray9` source format (9-bit low-packed u16).
    marker: Gray9,
    frame: Gray9Frame<'_>,
    generic_frame: GrayNFrame<'_, BITS>,
    bits: 9,
    row: Gray9Row,
    sink: Gray9Sink,
    walker: gray9_to,
    walker_inner: gray9_to_inner,
    elem_type: u16,
    row_doc: "A single row from a [`Gray9Frame`].",
    walker_doc: "Walks a [`Gray9Frame`] row by row, dispatching each row to the sink.",
  }
}
