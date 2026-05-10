//! Walker spec for the `Gray16` source format (FFmpeg `gray16{le,be}`).
//!
//! Single `u16` luma plane, all 16 bits active. No chroma.
//!
//! The marker carries `<const BE: bool = false>`: `Gray16` (= `Gray16<false>`)
//! is the LE source; `Gray16<true>` is the BE source. The walker
//! [`gray16_to::<BE>`] propagates `BE` from [`Gray16Frame<'_, BE>`] into the
//! sinker dispatch.

use crate::frame::Gray16Frame;

walker! {
  planar1_be {
    /// Marker type for the `Gray16` source format (16-bit native u16).
    /// `<const BE: bool>` defaults to `false` (LE).
    marker: Gray16,
    frame: Gray16Frame,
    row: Gray16Row,
    sink: Gray16Sink,
    walker: gray16_to,
    elem_type: u16,
    row_doc: "A single row from a [`Gray16Frame`].",
    walker_doc: "Walks a [`Gray16Frame<'_, BE>`] row by row, dispatching each \
                 row to the sink. Propagates `<const BE: bool>` from the \
                 frame into [`Gray16Sink<BE>`].",
  }
}
