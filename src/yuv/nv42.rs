//! NV42 — semi‑planar 4:4:4 (`AV_PIX_FMT_NV42`), VU‑ordered.
//!
//! Layout: one full‑size Y plane + one interleaved VU plane at full
//! width and full height. Each VU row is `V0, U0, V1, U1, …` — the
//! byte‑order twin of NV24's UV ordering. Shares per‑row kernel math
//! with [`super::Nv24`]; only the chroma‑byte parity differs (swapped
//! inside the SIMD/scalar kernel via a `SWAP_UV` const generic).

use crate::frame::Nv42Frame;

walker! {
  semi_planar {
    /// Zero‑sized marker for the NV42 source format. Used as the `F` type
    /// parameter on [`crate::sinker::MixedSinker`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
    marker: Nv42,
    frame: Nv42Frame<'_>,
    row: Nv42Row,
    sink: Nv42Sink,
    walker: nv42_to,
    elem_type: u8,
    chroma_field: vu,
    chroma_plane: vu,
    chroma_stride: vu_stride,
    chroma_elems_per_row: |w| 2 * w,
    chroma_v: full,
    row_doc: "One output row of an NV42 source handed to an [`Nv42Sink`].\n\n\
              Carries borrows to the source slices (full-width Y, full-width interleaved\n\
              VU — V-first) plus the row index and matrix/range carry-throughs.",
    walker_doc: "Converts an NV42 frame by walking its rows and feeding each one to\n\
                 the [`Nv42Sink`].",
  }
}
