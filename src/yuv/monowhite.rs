//! [`MonowhiteFrame`] walker — 1-bit-per-pixel, MSB-first encoding,
//! bit=0 → white (Y=255), bit=1 → black (Y=0). Inverted polarity from
//! Monoblack.

use crate::frame::MonowhiteFrame;

walker! {
  planar1 {
    /// Marker type for the `Monowhite` source format (FFmpeg
    /// `AV_PIX_FMT_MONOWHITE`).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Monowhite,
    frame: MonowhiteFrame<'_>,
    row: MonowhiteRow,
    sink: MonowhiteSink,
    walker: monowhite_to,
    elem_type: u8,
    row_doc: "A single row from a [`MonowhiteFrame`] — byte buffer\
       (8 pixels per byte, MSB first, inverted polarity).",
    walker_doc: "Walks a [`MonowhiteFrame`] row by row, dispatching\
       each row to the sink.",
  }
}
