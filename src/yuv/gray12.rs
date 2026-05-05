//! Walker spec for `Gray12` (FFmpeg `gray12le`).

use crate::frame::{Gray12Frame, GrayNFrame};

walker! {
  planar1_bits {
    /// Marker type for the `Gray12` source format (12-bit low-packed u16).
    marker: Gray12,
    frame: Gray12Frame<'_>,
    generic_frame: GrayNFrame<'_, BITS>,
    bits: 12,
    row: Gray12Row,
    sink: Gray12Sink,
    walker: gray12_to,
    walker_inner: gray12_to_inner,
    elem_type: u16,
    row_doc: "A single row from a [`Gray12Frame`].",
    walker_doc: "Walks a [`Gray12Frame`] row by row, dispatching each row to the sink.",
  }
}
