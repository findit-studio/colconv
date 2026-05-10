//! Packed **RGBF32** source (FFmpeg `AV_PIX_FMT_RGBF32`) — 32-bit
//! float per channel, byte order `R, G, B` per pixel (12 bytes /
//! 3 × `f32` per pixel).
//!
//! Like the Tier 6 8-bit packed-RGB family ([`super::Rgb24`] etc.),
//! the input is already RGB — there is no chroma matrix work. Outputs
//! map to the sink's standard channels (with a saturating cast back
//! to integer for u8 / u16 / luma / HSV outputs):
//! - `with_rgb` — clamp `[0, 1]` × 255 → packed `R, G, B` u8.
//! - `with_rgba` — same RGB conversion + constant `0xFF` alpha.
//! - `with_rgb_u16` — clamp `[0, 1]` × 65535 → packed `R, G, B` u16.
//! - `with_rgba_u16` — same RGB conversion + constant `0xFFFF` alpha.
//! - `with_luma` / `with_luma_u16` — derives Y' from R/G/B (after the
//!   clamp + cast to u8) using the existing `rgb_to_luma_row` /
//!   `rgb_to_luma_u16_row` kernels.
//! - `with_hsv` — clamp + cast to u8 staging followed by the existing
//!   `rgb_to_hsv_row` kernel.
//! - `with_rgb_f32` — **lossless** float pass-through: the source row
//!   is copied verbatim into the output buffer (HDR values > 1.0 are
//!   preserved).
//!
//! HDR values > 1.0 in the source saturate to the output range for
//! every integer output. No tone mapping is applied.

use crate::frame::Rgbf32Frame;

walker! {
  packed_be {
    /// Zero-sized marker for the packed **RGBF32** source format.
    /// `<const BE: bool = false>` mirrors the parent
    /// [`Rgbf32Frame`]'s endian flag — `false` (default) selects
    /// `AV_PIX_FMT_RGBF32LE`, `true` selects `AV_PIX_FMT_RGBF32BE`.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgbf32,
    frame: Rgbf32Frame,
    row: Rgbf32Row,
    sink: Rgbf32Sink,
    walker: rgbf32_to,
    walker_endian: rgbf32_to_endian,
    buf_field: rgb,
    elem_type: f32,
    row_elems: |w| w * 3,
    row_doc: "One row of an [`Rgbf32`] source — `width * 3` packed\n\
              `f32` samples (`R, G, B` per pixel). The Row type is\n\
              **not** parameterized on `BE` — it just borrows the\n\
              underlying byte slice; the kernel's BE-aware byte-swap\n\
              is monomorphized via the parent `Rgbf32<BE>` marker.",
    walker_doc: "Walks an [`Rgbf32Frame`] row by row into the sink.\n\
                 The `<const BE>` parameter is propagated from the\n\
                 frame to the sink-trait bound (`S: Rgbf32Sink<BE>`)\n\
                 so the row-kernel call inside `process` monomorphizes\n\
                 against the same byte order.",
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::{ColorMatrix, PixelSink, frame::Rgbf32Frame};
  use core::convert::Infallible;

  struct CountingSink {
    rows_seen: usize,
  }
  impl PixelSink for CountingSink {
    type Input<'r> = Rgbf32Row<'r>;
    type Error = Infallible;
    fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Infallible> {
      Ok(())
    }
    fn process(&mut self, _row: Rgbf32Row<'_>) -> Result<(), Infallible> {
      self.rows_seen += 1;
      Ok(())
    }
  }
  impl Rgbf32Sink for CountingSink {}

  // Compile-pass regression for the LE-only custom sink spelling. The
  // generated `$sink<const BE: bool = false>` carries an LE default so
  // downstream callers can keep writing `impl Rgbf32Sink for MySink`
  // (no `<false>`) and `S: Rgbf32Sink` bounds.
  #[test]
  fn rgbf32_sink_le_default_compiles_without_const_arg() {
    fn walks_le<S: Rgbf32Sink>(frame: &Rgbf32Frame<'_>, sink: &mut S) -> Result<(), S::Error> {
      rgbf32_to(frame, true, ColorMatrix::Bt709, sink)
    }

    let buf = std::vec![0.0_f32; 12 * 4];
    let frame = Rgbf32Frame::new(&buf, 4, 4, 12);
    let mut sink = CountingSink { rows_seen: 0 };
    walks_le(&frame, &mut sink).unwrap();
    assert_eq!(sink.rows_seen, 4);
  }

  // Compile-pass regression for the codex finding (PR #105 review). Switching
  // from `walker!(packed)` to `walker!(packed_be)` would otherwise change the
  // public `rgbf32_to` signature from one generic param (`S`) to two
  // (`S, const BE: bool`), which breaks downstream callers using the previous
  // explicit sink spelling `rgbf32_to::<MySink>(...)`. Function-position
  // const-generic defaults aren't allowed, so the macro emits an LE-only
  // wrapper preserving the original signature.
  #[test]
  fn rgbf32_to_explicit_turbofish_one_generic_compiles() {
    #[allow(clippy::type_complexity)]
    fn _check<S: Rgbf32Sink>() {
      let _: fn(
        &crate::frame::Rgbf32LeFrame<'_>,
        bool,
        ColorMatrix,
        &mut S,
      ) -> Result<(), S::Error> = rgbf32_to::<S>;
    }
  }
}
