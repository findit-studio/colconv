//! Packed **BGR555** source (`AV_PIX_FMT_BGR555LE`) — 1-bit padding, 5-bit B, G, R.
//! One pixel per 16-bit LE word: bit 15 is unused padding; bits [14:10]=B5, [9:5]=G5, [4:0]=R5.
//! No alpha. Output byte order is always R, G, B regardless of source channel order.
//!
//! Outputs (Tier 7):
//! - `with_rgb`      — expand each channel to u8 via bit-replication, pack as `R, G, B`.
//! - `with_rgba`     — same + constant α=`0xFF`.
//! - `with_rgb_u16`  — native 5/5/5-bit precision, low-bit aligned in `u16`, order R, G, B.
//! - `with_rgba_u16` — same + constant α=`0xFFFF`.
//! - `with_luma`     — Y′ luma staged through u8 RGB scratch.
//! - `with_luma_u16` — zero-extended u8 luma widened to u16.
//! - `with_hsv`      — HSV staged through u8 RGB scratch.

use crate::frame::Bgr555Frame;

walker! {
  packed {
    /// Zero-sized marker for the packed **BGR555** (`AV_PIX_FMT_BGR555LE`) source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Bgr555,
    frame: Bgr555Frame<'_>,
    row: Bgr555Row,
    sink: Bgr555Sink,
    walker: bgr555_to,
    buf_field: bgr555,
    elem_type: u8,
    row_elems: |w| w * 2,
    row_doc: "One output row of a [`Bgr555`] source — `width * 2` bytes\n\
              laid out as `width` little-endian `u16` pixels.\n\
              \n\
              Bit layout per 16-bit word (LE):\n\
              \n\
              | Bits   | Field |\n\
              |--------|-------|\n\
              | 15     | padding (ignored on read) |\n\
              | 14:10  | B (5 bits, range [0, 31]) |\n\
              | 9:5    | G (5 bits, range [0, 31]) |\n\
              | 4:0    | R (5 bits, range [0, 31]) |\n\
              \n\
              Channel positions reversed vs [`crate::yuv::Rgb555`].\n\
              No source alpha; RGBA outputs force α=`0xFF` / `0xFFFF`.",
    walker_doc: "Walks a [`Bgr555Frame`] row by row into the sink.",
  }
}
