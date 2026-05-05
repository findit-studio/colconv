//! Walker spec for `Gray10` (FFmpeg `gray10le`).

use crate::frame::{Gray10Frame, GrayNFrame};

walker! {
  planar1_bits {
    /// Marker type for the `Gray10` source format (10-bit low-packed u16).
    marker: Gray10,
    frame: Gray10Frame<'_>,
    generic_frame: GrayNFrame<'_, BITS>,
    bits: 10,
    row: Gray10Row,
    sink: Gray10Sink,
    walker: gray10_to,
    walker_inner: gray10_to_inner,
    elem_type: u16,
    row_doc: "A single row from a [`Gray10Frame`].",
    walker_doc: "Walks a [`Gray10Frame`] row by row, dispatching each row to the sink.",
  }
}
