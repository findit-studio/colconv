//! YUV 4:2:0 planar 9‑bit (`AV_PIX_FMT_YUV420P9LE`).
//!
//! Niche format — used by AVC High 9 Profile only; HEVC / VP9 / AV1
//! don't produce 9-bit. Reuses the same Q15 i32 kernel family as the
//! 10/12/14-bit siblings (`yuv_420p_n_to_rgb_*<BITS>`); the only
//! per-call difference is the const-generic `BITS = 9`, which fixes
//! the AND-mask to `0x1FF` and the Q15 scale via
//! `range_params_n::<9, _>`.

use crate::frame::Yuv420pFrame16;

walker! {
  planar3_bits_be {
    /// Zero‑sized marker for the YUV 4:2:0 **9‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv420p9,
    frame: Yuv420pFrame16<'_, 9, BE>,
    frame_le: Yuv420pFrame16<'_, 9, false>,
    generic_frame: Yuv420pFrame16<'_, BITS, BE>,
    bits: 9,
    row: Yuv420p9Row,
    sink: Yuv420p9Sink,
    walker: yuv420p9_to,
    walker_endian: yuv420p9_to_endian,
    walker_inner: yuv420p9_walker,
    elem_type: u16,
    chroma_h: half,
    chroma_v: half,
    row_doc: "One output row of a 9‑bit YUV 4:2:0 source.",
    walker_doc: "Walks a [`Yuv420p9Frame`] row by row into the sink.",
  }
}
