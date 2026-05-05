//! YUV 4:4:4 planar 16‑bit (`AV_PIX_FMT_YUV444P16LE`). The HW→SW
//! download target for CUDA / NVDEC 4:4:4 HDR content.
//!
//! Uses a **parallel i64 kernel family** for the u16‑output path —
//! same rationale as [`super::Yuv420p16`] (`coeff × u_d` at 16 bits
//! overflows i32 for Bt2020 blue).

use crate::frame::Yuv444p16Frame;

walker! {
  planar3 {
    /// Zero‑sized marker for the YUV 4:4:4 **16‑bit** source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Yuv444p16,
    frame: Yuv444p16Frame<'_>,
    row: Yuv444p16Row,
    sink: Yuv444p16Sink,
    walker: yuv444p16_to,
    elem_type: u16,
    chroma_h: full,
    chroma_v: full,
    row_doc: "One output row of a [`Yuv444p16`] source.",
    walker_doc: "Walks a [`Yuv444p16Frame`] row by row into the sink.",
  }
}
