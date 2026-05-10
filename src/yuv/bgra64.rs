//! Packed BGRA64 source (`AV_PIX_FMT_BGRA64{LE,BE}`) — 16 bits per channel,
//! `u16` element order `B, G, R, A`. Stride in u16 elements (≥ `4 * width`).
//!
//! The marker carries `<const BE: bool = false>`: `Bgra64` (= `Bgra64<false>`)
//! is the LE source; `Bgra64<true>` is the BE source. The walker
//! [`bgra64_to::<BE>`] propagates `BE` from [`Bgra64Frame<'_, BE>`] into the
//! sinker dispatch.
//!
//! Outputs (Tier 8 finish):
//! - `with_rgb`      — swap B↔R, drop alpha, narrow each channel `>> 8`, pack as R, G, B.
//! - `with_rgba`     — swap B↔R on RGB, all four channels narrowed `>> 8`; source alpha passed through.
//! - `with_rgb_u16`  — swap B↔R, drop alpha, native u16 passthrough (R, G, B output order).
//! - `with_rgba_u16` — swap B↔R on RGB, all four channels as-is; source alpha preserved.
//! - `with_luma`     — Y′ from R/G/B after channel swap and narrowing to u8 (alpha ignored).
//! - `with_luma_u16` — Y′ computed at u8 precision (matching `with_luma`'s
//!   output, with the same B↔R swap applied first) and zero-extended to
//!   u16; alpha ignored. Same convention as the 8-bit-source family; not
//!   native 16-bit luma precision.
//! - `with_hsv`      — HSV via u8 RGB staging (alpha ignored).

use crate::frame::Bgra64Frame;

walker! {
  packed_be {
    /// Zero-sized marker for the packed **BGRA64** source format
    /// (`AV_PIX_FMT_BGRA64{LE,BE}`). `<const BE: bool>` defaults to `false`
    /// (LE); the alias `Bgra64` resolves to `Bgra64<false>`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Bgra64,
    frame: Bgra64Frame,
    row: Bgra64Row,
    sink: Bgra64Sink,
    walker: bgra64_to,
    buf_field: bgra64,
    elem_type: u16,
    row_elems: |w| w * 4,
    row_doc: "One row of a [`Bgra64`] source — `width * 4` u16 elements \
              (`B, G, R, A` per pixel, each channel 16 bits; alpha is real). \
              Endianness is recorded on the parent [`Bgra64Frame<'_, BE>`] / \
              sinker, not on the Row itself.",
    walker_doc: "Walks a [`Bgra64Frame<'_, BE>`] row by row into the sink. \
                 Propagates `<const BE: bool>` from the frame into \
                 [`Bgra64Sink<BE>`].",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Bgra64Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
    last_width: usize,
    last_row_idx: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Bgra64Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, row: Bgra64Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      self.last_width = row.bgra64().len();
      self.last_row_idx = row.row();
      Ok(())
    }
  }
  impl Bgra64Sink for CountingSink {}

  #[test]
  fn bgra64_walker_visits_every_row_once() {
    let buf = std::vec![0u16; 16 * 4];
    let frame = Bgra64Frame::new(&buf, 4, 4, 16);
    let mut sink = CountingSink {
      rows_seen: 0,
      last_width: 0,
      last_row_idx: 0,
    };
    bgra64_to(&frame, true, ColorMatrix::Bt709, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
    assert_eq!(sink.last_width, 16);
    assert_eq!(sink.last_row_idx, 3);
  }
}
