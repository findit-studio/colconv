//! P410 — semi‑planar 4:4:4, 10‑bit, high‑bit‑packed
//! (`AV_PIX_FMT_P410LE`).
//!
//! 4:4:4 twin of [`super::P010`]: same high-bit-packed `u16`
//! convention (10 active bits in the high 10 positions), but chroma
//! is **full-width × full-height** (1:1 with Y, no subsampling).
//! Each chroma row holds `2 * width` `u16` elements (= `width`
//! interleaved `U, V` pairs). NVDEC / CUDA HDR 4:4:4 download target.
//!
//! Per-row kernel: a dedicated 4:4:4 high-bit-depth semi-planar
//! family `p_n_444_to_rgb_*<10>` (full-width interleaved UV, no
//! horizontal duplication step). Differs from the 4:2:0 / 4:2:2
//! `p_n_to_rgb_*<10>` family in the chroma layout only.

use crate::frame::PnFrame444;

walker! {
  semi_planar_be {
    /// Zero‑sized marker for the P410 source format.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: P410,
    frame: PnFrame444<'_, 10, BE>,
    frame_le: PnFrame444<'_, 10, false>,
    row: P410Row,
    sink: P410Sink,
    walker: p410_to,
    walker_endian: p410_to_endian,
    elem_type: u16,
    chroma_field: uv_full,
    chroma_plane: uv,
    chroma_stride: uv_stride,
    chroma_elems_per_row: |w| 2 * w,
    chroma_v: full,
    row_doc: "One output row of a P410 source handed to a [`P410Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, full-width interleaved\n\
              UV — `2 * width` u16 elements) plus the row index and matrix/range\n\
              carry-throughs. Each `u16` element is high-bit-packed (10 bits).",
    walker_doc: "Walks a [`P410Frame`] row by row into the sink.",
  }
}
