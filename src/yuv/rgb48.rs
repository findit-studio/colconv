//! Packed RGB48 source (`AV_PIX_FMT_RGB48{LE,BE}`) — 16 bits per channel,
//! `u16` element order `R, G, B`. Stride in u16 elements (≥ `3 * width`).
//!
//! The marker carries `<const BE: bool = false>`: `Rgb48` (= `Rgb48<false>`)
//! is the LE source; `Rgb48<true>` is the BE source. The walker
//! [`rgb48_to::<BE>`] propagates `BE` from [`Rgb48Frame<'_, BE>`] into the
//! sinker dispatch.
//!
//! Outputs (Tier 8 finish):
//! - `with_rgb`      — narrow each channel `>> 8`, pack as R, G, B.
//! - `with_rgba`     — same narrow + alpha = `0xFF`.
//! - `with_rgb_u16`  — native u16 passthrough (R, G, B order preserved).
//! - `with_rgba_u16` — native u16 passthrough + alpha = `0xFFFF`.
//! - `with_luma`     — Y′ from R/G/B after narrowing to u8.
//! - `with_luma_u16` — Y′ computed at u8 precision (matching `with_luma`'s
//!   output) and zero-extended to u16. Same convention as the 8-bit-source
//!   family; not native 16-bit luma precision.
//! - `with_hsv`      — HSV via u8 RGB staging.

use crate::frame::Rgb48Frame;

walker! {
  packed_be {
    /// Zero-sized marker for the packed **RGB48** source format
    /// (`AV_PIX_FMT_RGB48{LE,BE}`). `<const BE: bool>` defaults to `false`
    /// (LE); the alias `Rgb48` resolves to `Rgb48<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgb48,
    frame: Rgb48Frame,
    row: Rgb48Row,
    sink: Rgb48Sink,
    walker: rgb48_to,
    buf_field: rgb48,
    elem_type: u16,
    row_elems: |w| w * 3,
    row_doc: "One row of an [`Rgb48`] source — `width * 3` u16 elements \
              (`R, G, B` per pixel, each channel 16 bits). Endianness is \
              recorded on the parent [`Rgb48Frame<'_, BE>`] / sinker, not on \
              the Row itself — the kernel monomorphizes on `BE` at the \
              sinker dispatch.",
    walker_doc: "Walks an [`Rgb48Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`Rgb48Sink<BE>`].",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Rgb48Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Rgb48Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Rgb48Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.rgb48().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Rgb48Sink for CountingSink {}

  #[test]
  fn rgb48_walker_visits_every_row_once() {
    // width=4, stride=12 (3*4), height=4 → plane needs 48 u16 elements
    let buf = std::vec![0u16; 12 * 4];
    let frame = Rgb48Frame::new(&buf, 4, 4, 12);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    rgb48_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 12); // width * 3 u16 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }

  // Compile-pass regression for the LE-only custom sink spelling. The
  // generated `$sink<const BE: bool = false>` carries an LE default so
  // downstream callers can keep writing `impl Rgb48Sink for MySink`
  // (no `<false>`) and `S: Rgb48Sink` bounds. This mirrors the fix for
  // codex high-severity finding on `walker_macro.rs:242`.
  #[test]
  fn rgb48_sink_le_default_compiles_without_const_arg() {
    // `impl Rgb48Sink for CountingSink` (above) would already fail to
    // compile if the LE default regressed; this test additionally pins
    // the bare-bound form `S: Rgb48Sink` and confirms it monomorphizes
    // to the LE walker.
    fn walks_le<S: Rgb48Sink>(frame: &Rgb48Frame<'_>, sink: &mut S) -> Result<(), S::Error> {
      rgb48_to(frame, true, ColorMatrix::Bt709, sink)
    }

    let buf = std::vec![0u16; 12 * 4];
    let frame = Rgb48Frame::new(&buf, 4, 4, 12);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    walks_le(&frame, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
  }
}
