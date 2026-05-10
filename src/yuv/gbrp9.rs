//! Planar GBR 9-bit (`AV_PIX_FMT_GBRP9{LE,BE}`) — three full-resolution
//! `u16` planes in **G, B, R** order (FFmpeg convention).
//!
//! Samples are stored in the low 9 bits of each `u16` element.
//!
//! The marker carries `<const BE: bool = false>`: `Gbrp9` (= `Gbrp9<false>`)
//! is the LE source; `Gbrp9<true>` is the BE source. The walker
//! [`gbrp9_to::<BE>`] propagates `BE` from [`Gbrp9Frame<'_, BE>`] into the
//! sinker dispatch.

use crate::frame::{Gbrp9Frame, GbrpHighBitFrame};

walker! {
  planar3_bits_be {
    /// Zero-sized marker for the planar GBR 9-bit source format
    /// (`AV_PIX_FMT_GBRP9{LE,BE}`). `<const BE: bool>` defaults to `false`
    /// (LE); the alias `Gbrp9` resolves to `Gbrp9<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrp9,
    frame: Gbrp9Frame,
    generic_frame: GbrpHighBitFrame,
    bits: 9,
    row: Gbrp9Row,
    sink: Gbrp9Sink,
    walker: gbrp9_to,
    walker_inner: gbrp9_walker,
    elem_type: u16,
    row_doc: "One output row of a [`Gbrp9`] source — three full-width\n\
              `u16` planes in G / B / R order (samples in low 9 bits).\n\
              Endianness is recorded on the parent\n\
              [`Gbrp9Frame<'_, BE>`] / sinker, not on the Row itself.",
    walker_doc: "Walks a [`Gbrp9Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`Gbrp9Sink<BE>`].",
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
