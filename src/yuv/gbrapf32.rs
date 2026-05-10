//! Walker for the `Gbrapf32` source format (`AV_PIX_FMT_GBRAPF32{LE,BE}`) — four
//! full-resolution `f32` planes in **G, B, R, A** order.
//!
//! Alpha is real per-pixel; nominal range `[0.0, 1.0]` (opaque = 1.0).
//! Integer outputs clamp colour channels to `[0.0, 1.0]` before scaling;
//! float outputs are lossless pass-through.
//!
//! The marker carries `<const BE: bool = false>`: `Gbrapf32`
//! (= `Gbrapf32<false>`) is the LE source; `Gbrapf32<true>` is the BE
//! source. The walker [`gbrapf32_to::<BE>`] propagates `BE` from
//! [`Gbrapf32Frame<'_, BE>`] into the sinker dispatch.

use crate::{
  PixelSink, SourceFormat,
  frame::{Gbrapf32Frame, Gbrapf32LeFrame},
  sealed::Sealed,
};

/// Zero-sized marker for the planar GBRAP float-32 source format
/// (`AV_PIX_FMT_GBRAPF32{LE,BE}`). `<const BE: bool>` defaults to `false`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Gbrapf32<const BE: bool = false>;

impl<const BE: bool> Sealed for Gbrapf32<BE> {}
impl<const BE: bool> SourceFormat for Gbrapf32<BE> {}

/// One output row from a [`Gbrapf32Frame`] — four full-width `f32` slices
/// in G / B / R / A order. Use [`Self::g`] / [`Self::b`] / [`Self::r`] /
/// [`Self::a`].
#[derive(Debug, Clone, Copy)]
pub struct Gbrapf32Row<'a> {
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  a: &'a [f32],
  row: usize,
}

impl<'a> Gbrapf32Row<'a> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(g: &'a [f32], b: &'a [f32], r: &'a [f32], a: &'a [f32], row: usize) -> Self {
    Self { g, b, r, a, row }
  }

  /// Green plane row — `width` `f32` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn g(&self) -> &'a [f32] {
    self.g
  }
  /// Blue plane row — `width` `f32` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn b(&self) -> &'a [f32] {
    self.b
  }
  /// Red plane row — `width` `f32` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn r(&self) -> &'a [f32] {
    self.r
  }
  /// Alpha plane row — `width` `f32` elements (opaque = 1.0).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn a(&self) -> &'a [f32] {
    self.a
  }
  /// Output row index within the frame (0-based).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }
}

/// Sinks that consume rows of a [`Gbrapf32`] source. Defaults to LE
/// (`BE = false`) for back-compat.
pub trait Gbrapf32Sink<const BE: bool = false>:
  for<'a> PixelSink<Input<'a> = Gbrapf32Row<'a>>
{
}

/// Walks a [`Gbrapf32Frame<'_, BE>`] row by row, dispatching each row to
/// the sink. Propagates `<const BE: bool>` from the frame into
/// [`Gbrapf32Sink<BE>`]. Use the LE-only [`gbrapf32_to`] wrapper for
/// pre-Phase-4 explicit-turbofish callers.
pub fn gbrapf32_to_endian<S, const BE: bool>(
  src: &Gbrapf32Frame<'_, BE>,
  sink: &mut S,
) -> Result<(), S::Error>
where
  S: Gbrapf32Sink<BE>,
{
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let g_plane = src.g();
  let b_plane = src.b();
  let r_plane = src.r();
  let a_plane = src.a();
  let g_stride = src.g_stride() as usize;
  let b_stride = src.b_stride() as usize;
  let r_stride = src.r_stride() as usize;
  let a_stride = src.a_stride() as usize;

  for row in 0..h {
    let g = &g_plane[row * g_stride..row * g_stride + w];
    let b = &b_plane[row * b_stride..row * b_stride + w];
    let r = &r_plane[row * r_stride..row * r_stride + w];
    let a = &a_plane[row * a_stride..row * a_stride + w];
    sink.process(Gbrapf32Row::new(g, b, r, a, row))?;
  }
  Ok(())
}

/// LE-only back-compat wrapper preserving the pre-Phase-4 walker
/// signature. Forwards to [`gbrapf32_to_endian`] with `BE = false`.
///
/// Rust forbids defaults on function-position const-generic parameters,
/// so an explicit-turbofish caller written before the Phase-4 BE
/// migration (`gbrapf32_to::<MySink>(...)`) would otherwise fail to
/// compile. Keeping this single-generic wrapper preserves source
/// compatibility for those call sites. BE-aware callers should use
/// [`gbrapf32_to_endian`] directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn gbrapf32_to<S>(src: &Gbrapf32LeFrame<'_>, sink: &mut S) -> Result<(), S::Error>
where
  S: Gbrapf32Sink<false>,
{
  gbrapf32_to_endian::<S, false>(src, sink)
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::PixelSink;
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_a_len: usize,
    last_row_idx: usize,
  }

  impl PixelSink for CountingSink {
    type Input<'r> = Gbrapf32Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Gbrapf32Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_a_len = row.a().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }

  impl Gbrapf32Sink for CountingSink {}

  // Compile-pass regression for the codex round-1 finding on PR #109
  // (hand-written `gbrapf32_to`). See `gbrpf32::tests` for full rationale.
  // BE-aware callers should use `gbrapf32_to_endian::<S, BE>` directly.
  #[test]
  fn gbrapf32_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Gbrapf32Sink>() {
      let _: fn(&Gbrapf32LeFrame<'_>, &mut S) -> Result<(), S::Error> = gbrapf32_to::<S>;
    }
  }

  #[test]
  fn gbrapf32_walker_visits_every_row_once() {
    let buf = std::vec![1.0f32; 4 * 4];
    let frame = Gbrapf32LeFrame::try_new(&buf, &buf, &buf, &buf, 4, 4, 4, 4, 4, 4).unwrap();
    let mut sink = CountingSink {
      rows_seen: 0,
      last_a_len: 0,
      last_row_idx: 0,
    };
    gbrapf32_to(&frame, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_a_len, 4);
    assert_eq!(sink.last_row_idx, 3);
  }
}
