//! Planar GBR+A 10-bit (`AV_PIX_FMT_GBRAP10{LE,BE}`) — four full-resolution
//! `u16` planes in **G, B, R, A** order (FFmpeg convention).
//!
//! Samples are stored in the low 10 bits of each `u16` element.
//! Alpha is real per-pixel α (1:1 with G); not padding.

use crate::frame::{Gbrap10Frame, GbrapHighBitFrame};

walker! {
  planar4_bits_be {
    /// Zero-sized marker for the planar GBR+A 10-bit source format
    /// (`AV_PIX_FMT_GBRAP10{LE,BE}`). `<const BE: bool>` defaults to `false`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrap10,
    frame: Gbrap10Frame,
    generic_frame: GbrapHighBitFrame,
    bits: 10,
    row: Gbrap10Row,
    sink: Gbrap10Sink,
    walker: gbrap10_to,
    walker_endian: gbrap10_to_endian,
    walker_inner: gbrap10_walker,
    elem_type: u16,
    row_doc: "One output row of a [`Gbrap10`] source — four full-width\n\
              `u16` planes in G / B / R / A order (samples in low 10 bits).",
    walker_doc: "Walks a [`Gbrap10Frame<'_, BE>`] row by row into the sink.",
  }
}

impl<'a> Gbrap10Row<'a> {
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
  // Alpha row is already exposed as `self.a()` by the macro — no rename needed.
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // Compile-pass regression for the codex round-1 finding on PR #109
  // (`planar4_bits_be` arm). See `gbrp10::tests` for the full rationale.
  // BE-aware callers should use `gbrap10_to_endian::<S, BE>` directly.
  #[test]
  fn gbrap10_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Gbrap10Sink>() {
      let _: fn(
        &crate::frame::Gbrap10LeFrame<'_>,
        bool,
        ColorMatrix,
        &mut S,
      ) -> Result<(), S::Error> = gbrap10_to::<S>;
    }
  }
}
