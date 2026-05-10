//! Planar GBR 16-bit (`AV_PIX_FMT_GBRP16{LE,BE}`) — three full-resolution
//! `u16` planes in **G, B, R** order (FFmpeg convention).
//!
//! All 16 bits of each `u16` element are active (full `u16` range).

use crate::frame::{Gbrp16Frame, GbrpHighBitFrame};

walker! {
  planar3_bits_be {
    /// Zero-sized marker for the planar GBR 16-bit source format
    /// (`AV_PIX_FMT_GBRP16{LE,BE}`). `<const BE: bool>` defaults to `false`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrp16,
    frame: Gbrp16Frame,
    generic_frame: GbrpHighBitFrame,
    bits: 16,
    row: Gbrp16Row,
    sink: Gbrp16Sink,
    walker: gbrp16_to,
    walker_endian: gbrp16_to_endian,
    walker_inner: gbrp16_walker,
    elem_type: u16,
    row_doc: "One output row of a [`Gbrp16`] source — three full-width\n\
              `u16` planes in G / B / R order (full 16-bit range).",
    walker_doc: "Walks a [`Gbrp16Frame<'_, BE>`] row by row into the sink.",
  }
}

impl<'a> Gbrp16Row<'a> {
  /// Green plane row — full width, samples in [0, 65535].
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
