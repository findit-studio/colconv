//! Planar GBR 8-bit (`AV_PIX_FMT_GBRP`) — three full-resolution `u8`
//! planes in **G, B, R** order (FFmpeg convention).
//!
//! Unlike every YUV source in this crate, the input is already
//! component RGB — there's no chroma matrix work. The walker is the
//! same `planar3` shape used by [`super::Yuv444p`] (full-width planes,
//! no chroma subsampling, no width parity constraint), but the per-row
//! kernels reorder G/B/R into packed RGB rather than running the YUV →
//! RGB matrix.
//!
//! # Walker macro reuse
//!
//! The `walker!` macro's `planar3` arm uses fixed `y/u/v` field names
//! internally. We feed the macro the same shape and then add a thin
//! `impl GbrpRow` block below the macro invocation that exposes
//! externally-correct `g()` / `b()` / `r()` accessors mapping to the
//! underlying `y()` / `u()` / `v()`. The macro-generated `y()` /
//! `u()` / `v()` accessors stay `pub` (bumping the API surface
//! slightly), but the externally-named accessors are what callers
//! should use — only those are documented as part of the GBR contract.

use crate::frame::GbrpFrame;

walker! {
  planar3 {
    /// Zero-sized marker for the planar GBR 8-bit source format
    /// (`AV_PIX_FMT_GBRP`).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Gbrp,
    frame: GbrpFrame<'_>,
    row: GbrpRow,
    sink: GbrpSink,
    walker: gbrp_to,
    elem_type: u8,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Gbrp`] source — three full-width\n\
              planes in G / B / R order. The `y` / `u` / `v`\n\
              accessors below are macro-generated; prefer the\n\
              externally-correct [`Self::g`] / [`Self::b`] /\n\
              [`Self::r`] accessors for clarity.",
    walker_doc: "Walks a [`GbrpFrame`] row by row into the sink.",
  }
}

impl<'a> GbrpRow<'a> {
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
}
