//! Planar GBR 9-bit (`AV_PIX_FMT_GBRP9LE`) — three full-resolution
//! `u16` planes in **G, B, R** order (FFmpeg convention).
//!
//! Samples are stored in the low 9 bits of each `u16` element.

use crate::frame::{Gbrp9Frame, GbrpHighBitFrame};

walker! {
  planar3_bits {
    /// Zero-sized marker for the planar GBR 9-bit source format
    /// (`AV_PIX_FMT_GBRP9LE`).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrp9,
    frame: Gbrp9Frame<'_>,
    generic_frame: GbrpHighBitFrame<'_, BITS>,
    bits: 9,
    row: Gbrp9Row,
    sink: Gbrp9Sink,
    walker: gbrp9_to,
    walker_inner: gbrp9_walker,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Gbrp9`] source — three full-width\n\
              `u16` planes in G / B / R order (samples in low 9 bits).",
    walker_doc: "Walks a [`Gbrp9Frame`] row by row into the sink.",
  }
}

impl<'a> Gbrp9Row<'a> {
  /// Green plane row — full width, samples in [0, 511].
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
