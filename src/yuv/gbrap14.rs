//! Planar GBR+A 14-bit (`AV_PIX_FMT_GBRAP14{LE,BE}`) — four full-resolution
//! `u16` planes in **G, B, R, A** order (FFmpeg convention).
//!
//! Samples are stored in the low 14 bits of each `u16` element.
//! Alpha is real per-pixel α (1:1 with G); not padding.

use crate::frame::{Gbrap14Frame, GbrapHighBitFrame};

walker! {
  planar4_bits_be {
    /// Zero-sized marker for the planar GBR+A 14-bit source format
    /// (`AV_PIX_FMT_GBRAP14{LE,BE}`). `<const BE: bool>` defaults to `false`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrap14,
    frame: Gbrap14Frame,
    generic_frame: GbrapHighBitFrame,
    bits: 14,
    row: Gbrap14Row,
    sink: Gbrap14Sink,
    walker: gbrap14_to,
    walker_endian: gbrap14_to_endian,
    walker_inner: gbrap14_walker,
    elem_type: u16,
    row_doc: "One output row of a [`Gbrap14`] source — four full-width\n\
              `u16` planes in G / B / R / A order (samples in low 14 bits).",
    walker_doc: "Walks a [`Gbrap14Frame<'_, BE>`] row by row into the sink.",
  }
}

impl<'a> Gbrap14Row<'a> {
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
  // Alpha row is already exposed as `self.a()` by the macro — no rename needed.
}
