//! Walker for the `Gbrpf16` source format (`AV_PIX_FMT_GBRPF16{LE,BE}`) — three
//! full-resolution `half::f16` planes in **G, B, R** order.
//!
//! Nominal range `[0.0, 1.0]`; HDR values > 1.0 are permitted. Integer
//! outputs widen to `f32` then clamp; float outputs are lossless interleave
//! (f16 pass-through) or widening interleave (→ f32).
//!
//! The marker carries `<const BE: bool = false>`: `Gbrpf16`
//! (= `Gbrpf16<false>`) is the LE source; `Gbrpf16<true>` is the BE source.
//! The walker [`gbrpf16_to::<BE>`] propagates `BE` from
//! [`Gbrpf16Frame<'_, BE>`] into the sinker dispatch.

use crate::{
  PixelSink, SourceFormat,
  frame::{Gbrpf16Frame, Gbrpf16LeFrame},
  sealed::Sealed,
};

/// Zero-sized marker for the planar GBR float-16 source format
/// (`AV_PIX_FMT_GBRPF16{LE,BE}`). `<const BE: bool>` defaults to `false`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gbrpf16<const BE: bool = false>;

impl<const BE: bool> Sealed for Gbrpf16<BE> {}
impl<const BE: bool> SourceFormat for Gbrpf16<BE> {}

/// One output row from a [`Gbrpf16Frame`] — three full-width `half::f16`
/// slices in G / B / R order. Use [`Self::g`] / [`Self::b`] / [`Self::r`].
#[derive(Debug, Clone, Copy)]
pub struct Gbrpf16Row<'a> {
  g: &'a [half::f16],
  b: &'a [half::f16],
  r: &'a [half::f16],
  row: usize,
}

impl<'a> Gbrpf16Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    g: &'a [half::f16],
    b: &'a [half::f16],
    r: &'a [half::f16],
    row: usize,
  ) -> Self {
    Self { g, b, r, row }
  }

  /// Green plane row — `width` `half::f16` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn g(&self) -> &'a [half::f16] {
    self.g
  }
  /// Blue plane row — `width` `half::f16` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn b(&self) -> &'a [half::f16] {
    self.b
  }
  /// Red plane row — `width` `half::f16` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn r(&self) -> &'a [half::f16] {
    self.r
  }
  /// Output row index within the frame (0-based).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Sinks that consume rows of a [`Gbrpf16`] source. Defaults to LE
/// (`BE = false`) for back-compat.
pub trait Gbrpf16Sink<const BE: bool = false>:
  for<'a> PixelSink<Input<'a> = Gbrpf16Row<'a>>
{
}

/// Walks a [`Gbrpf16Frame<'_, BE>`] row by row, dispatching each row to the
/// sink. Propagates `<const BE: bool>` from the frame into
/// [`Gbrpf16Sink<BE>`]. Use the LE-only [`gbrpf16_to`] wrapper for
/// pre-Phase-4 explicit-turbofish callers.
pub fn gbrpf16_to_endian<S, const BE: bool>(
  src: &Gbrpf16Frame<'_, BE>,
  sink: &mut S,
) -> Result<(), S::Error>
where
  S: Gbrpf16Sink<BE>,
{
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let g_plane = src.g();
  let b_plane = src.b();
  let r_plane = src.r();
  let g_stride = src.g_stride() as usize;
  let b_stride = src.b_stride() as usize;
  let r_stride = src.r_stride() as usize;

  for row in 0..h {
    let g = &g_plane[row * g_stride..row * g_stride + w];
    let b = &b_plane[row * b_stride..row * b_stride + w];
    let r = &r_plane[row * r_stride..row * r_stride + w];
    sink.process(Gbrpf16Row::new(g, b, r, row))?;
  }
  Ok(())
}

/// LE-only back-compat wrapper preserving the pre-Phase-4 walker
/// signature. Forwards to [`gbrpf16_to_endian`] with `BE = false`.
///
/// Rust forbids defaults on function-position const-generic parameters,
/// so an explicit-turbofish caller written before the Phase-4 BE
/// migration (`gbrpf16_to::<MySink>(...)`) would otherwise fail to
/// compile. Keeping this single-generic wrapper preserves source
/// compatibility for those call sites. BE-aware callers should use
/// [`gbrpf16_to_endian`] directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbrpf16_to<S>(src: &Gbrpf16LeFrame<'_>, sink: &mut S) -> Result<(), S::Error>
where
  S: Gbrpf16Sink<false>,
{
  gbrpf16_to_endian::<S, false>(src, sink)
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::PixelSink;
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_g_len: usize,
    last_row_idx: usize,
  }

  impl PixelSink for CountingSink {
    type Input<'r> = Gbrpf16Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Gbrpf16Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_g_len = row.g().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }

  impl Gbrpf16Sink for CountingSink {}

  // Compile-pass regression for the codex round-1 finding on PR #109
  // (hand-written `gbrpf16_to`). See `gbrpf32::tests` for full rationale.
  // BE-aware callers should use `gbrpf16_to_endian::<S, BE>` directly.
  #[test]
  fn gbrpf16_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Gbrpf16Sink>() {
      let _: fn(&Gbrpf16LeFrame<'_>, &mut S) -> Result<(), S::Error> = gbrpf16_to::<S>;
    }
  }

  #[test]
  fn gbrpf16_walker_visits_every_row_once() {
    let buf = std::vec![half::f16::ZERO; 4 * 4];
    let frame = Gbrpf16LeFrame::try_new(&buf, &buf, &buf, 4, 4, 4, 4, 4).unwrap();
    let mut sink = CountingSink {
      rows_seen: 0,
      last_g_len: 0,
      last_row_idx: 0,
    };
    gbrpf16_to(&frame, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_g_len, 4);
    assert_eq!(sink.last_row_idx, 3);
  }
}
