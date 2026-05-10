//! Packed **RGB565** source (`AV_PIX_FMT_RGB565LE`) — 5-bit R, 6-bit G, 5-bit B.
//! One pixel per 16-bit LE word: bits [15:11]=R5, [10:5]=G6, [4:0]=B5. No alpha.
//!
//! Outputs (Tier 7):
//! - `with_rgb`      — expand each channel to u8 via bit-replication, pack as `R, G, B`.
//! - `with_rgba`     — same + constant α=`0xFF`.
//! - `with_rgb_u16`  — native 5/6/5-bit precision, low-bit aligned in `u16`.
//! - `with_rgba_u16` — same + constant α=`0xFFFF`.
//! - `with_luma`     — Y′ luma staged through u8 RGB scratch.
//! - `with_luma_u16` — zero-extended u8 luma widened to u16.
//! - `with_hsv`      — HSV staged through u8 RGB scratch.

use crate::frame::Rgb565Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **RGB565** (`AV_PIX_FMT_RGB565LE`) source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgb565,
    frame: Rgb565Frame<'_>,
    row: Rgb565Row,
    sink: Rgb565Sink,
    walker: rgb565_to,
    buf_field: rgb565,
    elem_type: u8,
    row_elems: |w| w * 2,
    row_doc: "One output row of an [`Rgb565`] source — `width * 2` bytes\n\
              laid out as `width` little-endian `u16` pixels.\n\
              \n\
              Bit layout per 16-bit word (LE):\n\
              \n\
              | Bits   | Field |\n\
              |--------|-------|\n\
              | 15:11  | R (5 bits, range [0, 31]) |\n\
              | 10:5   | G (6 bits, range [0, 63]) |\n\
              | 4:0    | B (5 bits, range [0, 31]) |\n\
              \n\
              No source alpha; RGBA outputs force α=`0xFF` / `0xFFFF`.",
    walker_doc: "Walks an [`Rgb565Frame`] row by row into the sink.",
  }
}
