//! Walker spec for `Gray10` (FFmpeg `gray10{le,be}`).
//!
//! The marker carries `<const BE: bool = false>`; see [`Gray9`](crate::source::Gray9)
//! for the full BE-flag contract.

use crate::frame::{Gray10Frame, GrayNFrame};

walker! {
  planar1_bits_be {
    /// Marker type for the `Gray10` source format (10-bit low-packed u16).
    /// `<const BE: bool>` defaults to `false` (LE).
    marker: Gray10,
    frame: Gray10Frame,
    generic_frame: GrayNFrame,
    bits: 10,
    row: Gray10Row,
    sink: Gray10Sink,
    walker: gray10_to,
    walker_endian: gray10_to_endian,
    walker_inner: gray10_to_inner,
    elem_type: u16,
    row_doc: "A single row from a [`Gray10Frame`].",
    walker_doc: "Walks a [`Gray10Frame<'_, BE>`] row by row, dispatching each \
                 row to the sink. Propagates `<const BE: bool>` from the \
                 frame into [`Gray10Sink<BE>`].",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // Compile-pass regression for the codex round-1 finding on PR #106
  // (`planar1_bits_be` arm). See `gray9::tests` for the full rationale.
  #[test]
  fn gray10_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Gray10Sink>() {
      let _: fn(
        &crate::frame::Gray10LeFrame<'_>,
        bool,
        ColorMatrix,
        &mut S,
      ) -> Result<(), S::Error> = gray10_to::<S>;
    }
  }
}
