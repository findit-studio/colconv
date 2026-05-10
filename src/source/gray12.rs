//! Walker spec for `Gray12` (FFmpeg `gray12{le,be}`).
//!
//! The marker carries `<const BE: bool = false>`; see [`Gray9`](crate::source::Gray9)
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
    walker_endian: gray12_to_endian,
    walker_inner: gray12_to_inner,
    elem_type: u16,
    row_doc: "A single row from a [`Gray12Frame`].",
    walker_doc: "Walks a [`Gray12Frame<'_, BE>`] row by row, dispatching each \
                 row to the sink. Propagates `<const BE: bool>` from the \
                 frame into [`Gray12Sink<BE>`].",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // Compile-pass regression for the codex round-1 finding on PR #106
  // (`planar1_bits_be` arm). See `gray9::tests` for the full rationale.
  #[test]
  fn gray12_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Gray12Sink>() {
      let _: fn(
        &crate::frame::Gray12LeFrame<'_>,
        bool,
        ColorMatrix,
        &mut S,
      ) -> Result<(), S::Error> = gray12_to::<S>;
    }
  }
}
