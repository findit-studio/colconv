//! Planar GBR 10-bit (`AV_PIX_FMT_GBRP10LE`) — three full-resolution
//! `u16` planes in **G, B, R** order (FFmpeg convention).
//!
//! Samples are stored in the low 10 bits of each `u16` element.

use crate::frame::{Gbrp10Frame, GbrpHighBitFrame};

walker! {
  planar3_bits {
    /// Zero-sized marker for the planar GBR 10-bit source format
    /// (`AV_PIX_FMT_GBRP10LE`).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrp10,
    frame: Gbrp10Frame<'_>,
    generic_frame: GbrpHighBitFrame<'_, BITS>,
    bits: 10,
    row: Gbrp10Row,
    sink: Gbrp10Sink,
    walker: gbrp10_to,
    walker_inner: gbrp10_walker,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Gbrp10`] source — three full-width\n\
              `u16` planes in G / B / R order (samples in low 10 bits).",
    walker_doc: "Walks a [`Gbrp10Frame`] row by row into the sink.",
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
