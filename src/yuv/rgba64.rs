//! Packed RGBA64 source (`AV_PIX_FMT_RGBA64{LE,BE}`) — 16 bits per channel,
//! `u16` element order `R, G, B, A`. Stride in u16 elements (≥ `4 * width`).
//!
//! The marker carries `<const BE: bool = false>`: `Rgba64` (= `Rgba64<false>`)
//! is the LE source; `Rgba64<true>` is the BE source. The walker
//! [`rgba64_to::<BE>`] propagates `BE` from [`Rgba64Frame<'_, BE>`] into the
//! sinker dispatch.
//!
//! Outputs (Tier 8 finish):
//! - `with_rgb`      — drop alpha, narrow each R/G/B channel `>> 8`, pack as R, G, B.
//! - `with_rgba`     — all four channels narrowed `>> 8`; source alpha passed through.
//! - `with_rgb_u16`  — drop alpha, native u16 passthrough (R, G, B order).
//! - `with_rgba_u16` — all four channels passed through as-is; source alpha preserved.
//! - `with_luma`     — Y′ from R/G/B after narrowing to u8 (alpha ignored).
//! - `with_luma_u16` — Y′ computed at u8 precision (matching `with_luma`'s
//!   output) and zero-extended to u16; alpha ignored. Same convention as
//!   the 8-bit-source family; not native 16-bit luma precision.
//! - `with_hsv`      — HSV via u8 RGB staging (alpha ignored).

use crate::frame::Rgba64Frame;

walker! {
  packed_be {
    /// Zero-sized marker for the packed **RGBA64** source format
    /// (`AV_PIX_FMT_RGBA64{LE,BE}`). `<const BE: bool>` defaults to `false`
    /// (LE); the alias `Rgba64` resolves to `Rgba64<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgba64,
    frame: Rgba64Frame,
    row: Rgba64Row,
    sink: Rgba64Sink,
    walker: rgba64_to,
    buf_field: rgba64,
    elem_type: u16,
    row_elems: |w| w * 4,
    row_doc: "One row of an [`Rgba64`] source — `width * 4` u16 elements \
              (`R, G, B, A` per pixel, each channel 16 bits; alpha is real). \
              Endianness is recorded on the parent [`Rgba64Frame<'_, BE>`] / \
              sinker, not on the Row itself.",
    walker_doc: "Walks an [`Rgba64Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`Rgba64Sink<BE>`].",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Rgba64Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Rgba64Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Rgba64Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.rgba64().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Rgba64Sink<false> for CountingSink {}

  #[test]
  fn rgba64_walker_visits_every_row_once() {
    // width=4, stride=16 (4*4), height=4 → plane needs 64 u16 elements
    let buf = std::vec![0u16; 16 * 4];
    let frame = Rgba64Frame::new(&buf, 4, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    rgba64_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 16); // width * 4 u16 elements per row
    assert_eq!(sink.last_row_idx, 3);
  }
}
