//! Planar GBR 10-bit (`AV_PIX_FMT_GBRP10{LE,BE}`) — three full-resolution
//! `u16` planes in **G, B, R** order (FFmpeg convention).
//!
//! Samples are stored in the low 10 bits of each `u16` element.
//!
//! The marker carries `<const BE: bool = false>`: `Gbrp10` (= `Gbrp10<false>`)
//! is the LE source; `Gbrp10<true>` is the BE source. The walker
//! [`gbrp10_to::<BE>`] propagates `BE` from [`Gbrp10Frame<'_, BE>`] into the
//! sinker dispatch.

use crate::frame::{Gbrp10Frame, GbrpHighBitFrame};

walker! {
  planar3_bits_be {
    /// Zero-sized marker for the planar GBR 10-bit source format
    /// (`AV_PIX_FMT_GBRP10{LE,BE}`). `<const BE: bool>` defaults to `false`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrp10,
    frame: Gbrp10Frame,
    generic_frame: GbrpHighBitFrame,
    bits: 10,
    row: Gbrp10Row,
    sink: Gbrp10Sink,
    walker: gbrp10_to,
    walker_endian: gbrp10_to_endian,
    walker_inner: gbrp10_walker,
    elem_type: u16,
    row_doc: "One output row of a [`Gbrp10`] source — three full-width\n\
              `u16` planes in G / B / R order (samples in low 10 bits).",
    walker_doc: "Walks a [`Gbrp10Frame<'_, BE>`] row by row into the sink.",
  }
}

impl<'a> Gbrp10Row<'a> {
  /// Green plane row — full width, samples in [0, 1023].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn g(&self) -> &'a [u16] {
    self.y()
  }
  /// Blue plane row — full width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn b(&self) -> &'a [u16] {
    self.u()
  }
  /// Red plane row — full width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn r(&self) -> &'a [u16] {
    self.v()
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // Compile-pass regression for the codex round-1 finding on PR #109
  // (`planar3_bits_be` arm). Switching from `walker!(planar3_bits)` to
  // `walker!(planar3_bits_be)` would otherwise change the public
  // `gbrp10_to` signature from one generic param (`S`) to two
  // (`S, const BE: bool`), which breaks downstream callers using the
  // previous explicit sink spelling `gbrp10_to::<MySink>(...)`.
  // Function-position const-generic defaults aren't allowed, so the
  // macro emits an LE-only wrapper preserving the original signature.
  // BE-aware callers should use `gbrp10_to_endian::<S, BE>` directly.
  #[test]
  fn gbrp10_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Gbrp10Sink>() {
      let _: fn(
        &crate::frame::Gbrp10LeFrame<'_>,
        bool,
        ColorMatrix,
        &mut S,
      ) -> Result<(), S::Error> = gbrp10_to::<S>;
    }
  }
}
