//! YUV 4:2:2 planar 9‑bit (`AV_PIX_FMT_YUV422P9LE`).
//!
//! Same `u16`-backed layout as [`super::Yuv422p10`] with 9 active
//! bits in the low 9 of each element. Niche format — AVC High 9
//! profile only. Per-row kernel reuses the 4:2:0 family at
//! `BITS = 9` (`yuv_420p_n_to_rgb_row::<9>` and friends, internal
//! to `crate::row`) verbatim — same shape (half-width chroma per
//! row), only the vertical walk differs.

use crate::frame::Yuv422pFrame16;

walker! {
  planar3_be {
    /// Zero‑sized marker for the YUV 4:2:2 **9‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv422p9,
    frame: Yuv422pFrame16<'_, 9, BE>,
    frame_le: Yuv422pFrame16<'_, 9, false>,
    row: Yuv422p9Row,
    sink: Yuv422p9Sink,
    walker: yuv422p9_to,
    walker_endian: yuv422p9_to_endian,
    elem_type: u16,
    chroma_h: half,
    chroma_v: full,
    row_doc: "One output row of a [`Yuv422p9`] source.",
    walker_doc: "Walks a [`Yuv422p9Frame`] row by row into the sink.",
  }
}
