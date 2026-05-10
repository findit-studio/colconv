//! YUV 4:2:0 planar 16‑bit (`AV_PIX_FMT_YUV420P16LE`).
//!
//! Storage mirrors [`super::Yuv420p10`] / [`super::Yuv420p12`] /
//! [`super::Yuv420p14`] — three planes, Y at full size plus U / V at
//! half width and half height — with **`u16`** samples. At 16 bits
//! there is no upper-bit-zero slack; the full `u16` range is active.
//!
//! Runs on the **parallel i64 kernel family** —
//! [`crate::row::yuv420p16_to_rgb_row`] and companions dispatch to
//! `scalar::yuv_420p16_to_rgb_*` plus the matching per-backend SIMD
//! kernels, which carry i64 intermediates for the chroma matrix
//! multiply. The 10/12/14 families stay on the Q15 i32 pipeline.

use crate::frame::Yuv420pFrame16;

walker! {
  planar3_bits_be {
    /// Zero‑sized marker for the YUV 4:2:0 **16‑bit** source format. Used
    /// as the `F` type parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv420p16,
    frame: Yuv420pFrame16<'_, 16, BE>,
    generic_frame: Yuv420pFrame16<'_, BITS, BE>,
    bits: 16,
    row: Yuv420p16Row,
    sink: Yuv420p16Sink,
    walker: yuv420p16_to,
    walker_inner: yuv420p16_walker,
    elem_type: u16,
    chroma_h: half,
    chroma_v: half,
    row_doc: "One output row of a 16‑bit YUV 4:2:0 source handed to a\n\
              [`Yuv420p16Sink`]. Structurally identical to [`super::Yuv420p10Row`],\n\
              just with values covering the full `u16` range.",
    walker_doc: "Converts a 16‑bit YUV 4:2:0 frame by walking its rows and feeding\n\
                 each one to the [`Yuv420p16Sink`]. Pure row walker — all color\n\
                 arithmetic happens inside the Sink via the i64 16‑bit kernel\n\
                 family.",
  }
}
