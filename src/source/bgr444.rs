//! Packed **BGR444** source (`AV_PIX_FMT_BGR444LE`) — 4-bit padding + 4-bit B, G, R.
//! One pixel per 16-bit LE word: bits [15:12] are unused padding; bits [11:8]=B4, [7:4]=G4,
//! [3:0]=R4. No alpha. Output byte order is always R, G, B regardless of source channel order.
//!
//! Outputs (Tier 7):
//! - `with_rgb`      — expand each channel to u8 via bit-replication `(c<<4)|c`, pack as `R,G,B`.
//! - `with_rgba`     — same + constant α=`0xFF`.
//! - `with_rgb_u16`  — native 4/4/4-bit precision, low-bit aligned in `u16`, order R, G, B.
//! - `with_rgba_u16` — same + constant α=`0xFFFF`.
//! - `with_luma`     — Y′ luma staged through u8 RGB scratch.
//! - `with_luma_u16` — zero-extended u8 luma widened to u16.
//! - `with_hsv`      — HSV staged through u8 RGB scratch.

use crate::frame::Bgr444Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **BGR444** (`AV_PIX_FMT_BGR444LE`) source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Bgr444,
    frame: Bgr444Frame<'_>,
    row: Bgr444Row,
    sink: Bgr444Sink,
    walker: bgr444_to,
    buf_field: bgr444,
    elem_type: u8,
    row_elems: |w| w * 2,
    row_doc: "One output row of a [`Bgr444`] source — `width * 2` bytes\n\
              laid out as `width` little-endian `u16` pixels.\n\
              \n\
              Bit layout per 16-bit word (LE):\n\
              \n\
              | Bits   | Field |\n\
              |--------|-------|\n\
              | 15:12  | padding (ignored on read) |\n\
              | 11:8   | B (4 bits, range [0, 15]) |\n\
              | 7:4    | G (4 bits, range [0, 15]) |\n\
              | 3:0    | R (4 bits, range [0, 15]) |\n\
              \n\
              Channel positions reversed vs [`crate::source::Rgb444`].\n\
              No source alpha; RGBA outputs force α=`0xFF` / `0xFFFF`.",
    walker_doc: "Walks a [`Bgr444Frame`] row by row into the sink.",
  }
}
