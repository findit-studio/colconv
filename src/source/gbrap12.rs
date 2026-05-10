//! Planar GBR+A 12-bit (`AV_PIX_FMT_GBRAP12{LE,BE}`) — four full-resolution
//! `u16` planes in **G, B, R, A** order (FFmpeg convention).
//!
//! Samples are stored in the low 12 bits of each `u16` element.
//! Alpha is real per-pixel α (1:1 with G); not padding.

use crate::frame::{Gbrap12Frame, GbrapHighBitFrame};

walker! {
  planar4_bits_be {
    /// Zero-sized marker for the planar GBR+A 12-bit source format
    /// (`AV_PIX_FMT_GBRAP12{LE,BE}`). `<const BE: bool>` defaults to `false`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrap12,
    frame: Gbrap12Frame,
    generic_frame: GbrapHighBitFrame,
    bits: 12,
    row: Gbrap12Row,
    sink: Gbrap12Sink,
    walker: gbrap12_to,
    walker_endian: gbrap12_to_endian,
    walker_inner: gbrap12_walker,
    elem_type: u16,
    row_doc: "One output row of a [`Gbrap12`] source — four full-width\n\
              `u16` planes in G / B / R / A order (samples in low 12 bits).",
    walker_doc: "Walks a [`Gbrap12Frame<'_, BE>`] row by row into the sink.",
  }
}

impl<'a> Gbrap12Row<'a> {
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
  // Alpha row is already exposed as `self.a()` by the macro — no rename needed.
}
