//! Planar GBR+A 16-bit (`AV_PIX_FMT_GBRAP16{LE,BE}`) — four full-resolution
//! `u16` planes in **G, B, R, A** order (FFmpeg convention).
//!
//! All 16 bits of each `u16` element are active (full `u16` range).
//! Alpha is real per-pixel α (1:1 with G); not padding.

use crate::frame::{Gbrap16Frame, GbrapHighBitFrame};

walker! {
  planar4_bits_be {
    /// Zero-sized marker for the planar GBR+A 16-bit source format
    /// (`AV_PIX_FMT_GBRAP16{LE,BE}`). `<const BE: bool>` defaults to `false`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrap16,
    frame: Gbrap16Frame,
    generic_frame: GbrapHighBitFrame,
    bits: 16,
    row: Gbrap16Row,
    sink: Gbrap16Sink,
    walker: gbrap16_to,
    walker_inner: gbrap16_walker,
    elem_type: u16,
    row_doc: "One output row of a [`Gbrap16`] source — four full-width\n\
              `u16` planes in G / B / R / A order (full 16-bit range).",
    walker_doc: "Walks a [`Gbrap16Frame<'_, BE>`] row by row into the sink.",
  }
}

impl<'a> Gbrap16Row<'a> {
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
  // Alpha row is already exposed as `self.a()` by the macro — no rename needed.
}
