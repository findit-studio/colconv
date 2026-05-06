//! Packed **RGBF16** source (FFmpeg `AV_PIX_FMT_RGBF16`) — 16-bit
//! half-precision float per channel, byte order `R, G, B` per pixel
//! (6 bytes / 3 × `half::f16` per pixel).
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
//! - `with_rgb_f16` — **lossless** half-float pass-through: the source
//!   row is copied verbatim into the output buffer (HDR values > 1.0
//!   are preserved).
//! - `with_rgb_f32` — lossless widening: each `f16` element is widened
//!   to `f32` (HDR values > 1.0 are preserved).
//!
//! HDR values > 1.0 in the source saturate to the output range for
//! every integer output. No tone mapping is applied.
//!
//! Downstream conversion widens `f16` → `f32` at row entry, then
//! reuses the existing `rgbf32_to_*_row` kernels (Tier 9 completion).

use crate::frame::Rgbf16Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **RGBF16** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgbf16,
    frame: Rgbf16Frame<'_>,
    row: Rgbf16Row,
    sink: Rgbf16Sink,
    walker: rgbf16_to,
    buf_field: rgb,
    elem_type: half::f16,
    row_elems: |w| w * 3,
    row_doc: "One row of an [`Rgbf16`] source — `width * 3` packed\n\
              `half::f16` samples (`R, G, B` per pixel).",
    walker_doc: "Walks an [`Rgbf16Frame`] row by row into the sink.",
  }
}
