//! Packed **RGB444** source (`AV_PIX_FMT_RGB444LE`) — 4-bit padding + 4-bit R, G, B.
//! One pixel per 16-bit LE word: bits [15:12] are unused padding; bits [11:8]=R4, [7:4]=G4,
//! [3:0]=B4. No alpha.
//!
//! Outputs (Tier 7):
//! - `with_rgb`      — expand each channel to u8 via bit-replication `(c<<4)|c`, pack as `R,G,B`.
//! - `with_rgba`     — same + constant α=`0xFF`.
//! - `with_rgb_u16`  — native 4/4/4-bit precision, low-bit aligned in `u16`.
//! - `with_rgba_u16` — same + constant α=`0xFFFF`.
//! - `with_luma`     — Y′ luma staged through u8 RGB scratch.
//! - `with_luma_u16` — zero-extended u8 luma widened to u16.
//! - `with_hsv`      — HSV staged through u8 RGB scratch.

use crate::frame::Rgb444Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **RGB444** (`AV_PIX_FMT_RGB444LE`) source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Rgb444,
    frame: Rgb444Frame<'_>,
    row: Rgb444Row,
    sink: Rgb444Sink,
    walker: rgb444_to,
    buf_field: rgb444,
    elem_type: u8,
    row_elems: |w| w * 2,
    row_doc: "One output row of an [`Rgb444`] source — `width * 2` bytes\n\
              laid out as `width` little-endian `u16` pixels.\n\
              \n\
              Bit layout per 16-bit word (LE):\n\
              \n\
              | Bits   | Field |\n\
              |--------|-------|\n\
              | 15:12  | padding (ignored on read) |\n\
              | 11:8   | R (4 bits, range [0, 15]) |\n\
              | 7:4    | G (4 bits, range [0, 15]) |\n\
              | 3:0    | B (4 bits, range [0, 15]) |\n\
              \n\
              No source alpha; RGBA outputs force α=`0xFF` / `0xFFFF`.",
    walker_doc: "Walks an [`Rgb444Frame`] row by row into the sink.",
  }
}
