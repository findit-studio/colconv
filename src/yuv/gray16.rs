//! Walker spec for the `Gray16` source format (FFmpeg `gray16{le,be}`).
//!
//! Single `u16` luma plane, all 16 bits active. No chroma.
//!
//! The marker carries `<const BE: bool = false>`: `Gray16` (= `Gray16<false>`)
//! is the LE source; `Gray16<true>` is the BE source. Two walker entry points
//! are emitted: [`gray16_to`] is an LE-only compatibility wrapper preserving
//! the single-generic signature `gray16_to::<S>`; [`gray16_to_endian::<S, BE>`]
//! is the const-generic entry point for BE-aware callers, propagating `BE`
//! from [`Gray16Frame<'_, BE>`] into the sinker dispatch.

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
    walker_endian: gray16_to_endian,
    elem_type: u16,
    row_doc: "A single row from a [`Gray16Frame`].",
    walker_doc: "Walks a [`Gray16Frame<'_, BE>`] row by row, dispatching each \
                 row to the sink. Propagates `<const BE: bool>` from the \
                 frame into [`Gray16Sink<BE>`].",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // Compile-pass regression for the codex round-1 finding on PR #106
  // (`planar1_be` arm). Switching the Gray16 walker macro from `planar1`
  // to `planar1_be` without an LE wrapper would change the public
  // `gray16_to` signature from one generic param (`S`) to two
  // (`S, const BE: bool`), breaking downstream callers using the explicit
  // sink spelling `gray16_to::<MySink>(...)`. Function-position
  // const-generic defaults aren't allowed, so the macro emits an LE-only
  // wrapper preserving the original signature; this test pins it.
  #[test]
  fn gray16_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Gray16Sink>() {
      let _: fn(
        &crate::frame::Gray16LeFrame<'_>,
        bool,
        ColorMatrix,
        &mut S,
      ) -> Result<(), S::Error> = gray16_to::<S>;
    }
  }
}
