//! Walker spec for the `Gray16` source format (FFmpeg `gray16le`).
//!
//! Single `u16` luma plane, all 16 bits active. No chroma.

use crate::frame::Gray16Frame;

walker! {
  planar1 {
    /// Marker type for the `Gray16` source format (16-bit native u16).
    marker: Gray16,
    frame: Gray16Frame<'_>,
    row: Gray16Row,
    sink: Gray16Sink,
    walker: gray16_to,
    elem_type: u16,
    row_doc: "A single row from a [`Gray16Frame`].",
    walker_doc: "Walks a [`Gray16Frame`] row by row, dispatching each row to the sink.",
  }
}
