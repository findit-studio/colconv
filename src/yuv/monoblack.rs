//! [`MonoblackFrame`] walker — 1-bit-per-pixel, MSB-first encoding,
//! bit=0 → black (Y=0), bit=1 → white (Y=255).

use crate::frame::MonoblackFrame;

walker! {
  planar1 {
    /// Marker type for the `Monoblack` source format (FFmpeg
    /// `AV_PIX_FMT_MONOBLACK`).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Monoblack,
    frame: MonoblackFrame<'_>,
    row: MonoblackRow,
    sink: MonoblackSink,
    walker: monoblack_to,
    elem_type: u8,
    row_doc: "A single row from a [`MonoblackFrame`] — byte buffer\
       (8 pixels per byte, MSB first).",
    walker_doc: "Walks a [`MonoblackFrame`] row by row, dispatching\
       each row to the sink.",
  }
}
