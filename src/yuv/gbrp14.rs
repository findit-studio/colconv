//! Planar GBR 14-bit (`AV_PIX_FMT_GBRP14{LE,BE}`) — three full-resolution
//! `u16` planes in **G, B, R** order (FFmpeg convention).
//!
//! Samples are stored in the low 14 bits of each `u16` element.

use crate::frame::{Gbrp14Frame, GbrpHighBitFrame};

walker! {
  planar3_bits_be {
    /// Zero-sized marker for the planar GBR 14-bit source format
    /// (`AV_PIX_FMT_GBRP14{LE,BE}`). `<const BE: bool>` defaults to `false`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrp14,
    frame: Gbrp14Frame,
    generic_frame: GbrpHighBitFrame,
    bits: 14,
    row: Gbrp14Row,
    sink: Gbrp14Sink,
    walker: gbrp14_to,
    walker_inner: gbrp14_walker,
    elem_type: u16,
    row_doc: "One output row of a [`Gbrp14`] source — three full-width\n\
              `u16` planes in G / B / R order (samples in low 14 bits).",
    walker_doc: "Walks a [`Gbrp14Frame<'_, BE>`] row by row into the sink.",
  }
}

impl<'a> Gbrp14Row<'a> {
  /// Green plane row — full width, samples in [0, 16383].
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
