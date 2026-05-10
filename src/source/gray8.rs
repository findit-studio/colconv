//! Walker spec for the `Gray8` source format (FFmpeg `gray` / `AV_PIX_FMT_GRAY8`).
//!
//! Single `u8` luma plane. No chroma. The walker hands each row to the
//! sink as a [`Gray8Row`] containing the Y slice plus matrix / range metadata.

use crate::frame::Gray8Frame;

walker! {
  planar1 {
    /// Marker type for the `Gray8` source format.
    marker: Gray8,
    frame: Gray8Frame<'_>,
    row: Gray8Row,
    sink: Gray8Sink,
    walker: gray8_to,
    elem_type: u8,
    row_doc: "A single row from a [`Gray8Frame`].",
    walker_doc: "Walks a [`Gray8Frame`] row by row, dispatching each row to the sink.",
  }
}
