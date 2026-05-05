//! Planar GBR + A 8-bit (`AV_PIX_FMT_GBRAP`) — four full-resolution
//! `u8` planes in **G, B, R, A** order.
//!
//! Same structure as [`super::Gbrp`] with an additional alpha plane
//! (1:1 with the colour planes — real per-pixel α, not padding). The
//! walker is the `planar4` shape used by [`super::Yuva444p`]; the
//! rename trick from `Gbrp` (Y / U / V → G / B / R) carries through
//! and adds an `a()` accessor for the alpha plane (the macro-generated
//! `a()` already has the right name).

use crate::frame::GbrapFrame;

walker! {
  planar4 {
    /// Zero-sized marker for the planar GBRAP 8-bit source format
    /// (`AV_PIX_FMT_GBRAP`).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrap,
    frame: GbrapFrame<'_>,
    row: GbrapRow,
    sink: GbrapSink,
    walker: gbrap_to,
    elem_type: u8,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Gbrap`] source — four full-width\n\
              planes in G / B / R / A order. Alpha is real (not\n\
              padding) and is passed through to RGBA output. Prefer\n\
              the externally-correct [`Self::g`] / [`Self::b`] /\n\
              [`Self::r`] / [`Self::a`] accessors over the\n\
              macro-generated `y()` / `u()` / `v()` for clarity.",
    walker_doc: "Walks a [`GbrapFrame`] row by row into the sink.",
  }
}

impl<'a> GbrapRow<'a> {
  /// Green plane row — full width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn g(&self) -> &'a [u8] {
    self.y()
  }
  /// Blue plane row — full width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn b(&self) -> &'a [u8] {
    self.u()
  }
  /// Red plane row — full width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn r(&self) -> &'a [u8] {
    self.v()
  }
  // Alpha row is already exposed by the macro as [`Self::a`] — same
  // name, no rename needed.
}
