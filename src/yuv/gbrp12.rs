//! Planar GBR 12-bit (`AV_PIX_FMT_GBRP12{LE,BE}`) — three full-resolution
//! `u16` planes in **G, B, R** order (FFmpeg convention).
//!
//! Samples are stored in the low 12 bits of each `u16` element.

use crate::frame::{Gbrp12Frame, GbrpHighBitFrame};

walker! {
  planar3_bits_be {
    /// Zero-sized marker for the planar GBR 12-bit source format
    /// (`AV_PIX_FMT_GBRP12{LE,BE}`). `<const BE: bool>` defaults to `false`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrp12,
    frame: Gbrp12Frame,
    generic_frame: GbrpHighBitFrame,
    bits: 12,
    row: Gbrp12Row,
    sink: Gbrp12Sink,
    walker: gbrp12_to,
    walker_inner: gbrp12_walker,
    elem_type: u16,
    row_doc: "One output row of a [`Gbrp12`] source — three full-width\n\
              `u16` planes in G / B / R order (samples in low 12 bits).",
    walker_doc: "Walks a [`Gbrp12Frame<'_, BE>`] row by row into the sink.",
  }
}

impl<'a> Gbrp12Row<'a> {
  /// Green plane row — full width, samples in [0, 4095].
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
