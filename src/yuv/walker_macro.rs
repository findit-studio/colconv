//! Internal macro generating the per-format walker boilerplate
//! (marker zero-sized type, `Row` struct, `Sink` subtrait, walker
//! function) shared by every module under [`super`].
//!
//! Each walker module followed the same shape — a marker, a `Row`
//! struct holding borrowed slices + the row index + matrix/range
//! carry-throughs, a `Sink` subtrait pinning the row type, and a
//! walker `fn` doing per-frame preflight + slice math + sink dispatch.
//! The structural sameness was ~85% of every module; this macro
//! consolidates it into ~10 LOC of spec per format.
//!
//! # Forms
//!
//! The macro has one entry rule per *plane topology*. Pick the one
//! that matches the format's storage:
//!
//! - `walker!(packed { … })` — single-buffer formats (packed YUV
//!   422/444, packed RGB, AYUV64, etc.).
//! - `walker!(semi_planar { … })` — 2-plane Y + interleaved
//!   chroma (Nv12/Nv21/Nv24/Nv42, P010/P012/P016, P210/P212/P216,
//!   P410/P412/P416).
//! - `walker!(planar3 { … })` — 3-plane Y + U + V (`Yuv*p`
//!   family). Optional `bits_generic: yes` switches the walker into
//!   a const-generic `BITS` shape (used by 9/10/12/14/16-bit families
//!   that share the underlying `Yuv*pFrame16<BITS>` struct).
//! - `walker!(planar4 { … })` — 4-plane Y + U + V + A (`Yuva*p`
//!   family). Same `bits_generic: yes` switch as `planar3`.
//!
//! Per-row chroma vertical sampling (4:2:0 vs 4:2:2/4:4:4) is selected
//! per-format by the `chroma_v: half | full` field in the
//! `semi_planar`/`planar3`/`planar4` forms.
//!
//! # Why a macro and not a generic walker?
//!
//! Each format has a *distinct* `Sink` subtrait (`Yuv420pSink`,
//! `Nv12Sink`, etc.) so callers can constrain "I take YUV 4:2:0 rows"
//! at the type level. A single generic walker would need a unified
//! row trait — possible but significantly more API surface, and
//! defeats the point of the per-format Sink. The macro keeps the
//! per-format vocabulary identical to the hand-written modules.

/// Generates the marker / `Row` / `Sink` / walker quartet for a YUV
/// (or RGB) source format.
///
/// See the module-level docs for the four invocation forms.
macro_rules! walker {
  // ---------- packed (single-buffer) ----------------------------------------
  //
  // Used by every single-plane source: packed YUV 4:2:2 (Yuyv422,
  // Uyvy422, Yvyu422), packed RGB (Rgb24, Bgr24, Rgba, Bgra, Abgr,
  // Argb, Xrgb, Xbgr, Rgbx, Bgrx), 10-bit packed RGB (X2Rgb10,
  // X2Bgr10), packed YUV 4:4:4 (Vuya, Vuyx, V210, V30X, V410, Xv36,
  // Ayuv64), packed YUV 4:2:2 high-bit (Y210, Y212, Y216).
  //
  // The walker computes `start = row * stride`, slices `row_elems`
  // out of the single plane, and hands the slice to the sink.
  //
  // `row_elems` is an expression in `w` (the width as `usize`).
  (
    packed {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ty,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      // Field name on `Row` and the corresponding accessor on the
      // frame returning a `&[$elem]` plane and a `_stride()` accessor.
      buf_field: $buf:ident,
      elem_type: $elem:ty,
      // Closure-style spec for the per-row slice length, given the
      // width-as-usize binding.
      row_elems: |$w:ident| $row_elems:expr,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      $buf: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub(crate) fn new(
        $buf: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { $buf, row, matrix, full_range }
      }
      /// Packed source row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn $buf(&self) -> &'a [$elem] {
        self.$buf
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV/RGB conversion matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range vs limited-range flag carried through from the
      /// kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let $w = src.width() as usize;
      let h = src.height() as usize;
      let stride = src.stride() as usize;
      let row_elems: usize = $row_elems;
      let plane = src.$buf();

      for row in 0..h {
        let start = row * stride;
        let $buf = &plane[start..start + row_elems];
        sink.process($row::new($buf, row, matrix, full_range))?;
      }
      Ok(())
    }
  };

  // ---------- semi-planar (2 planes: Y + interleaved chroma) ---------------
  //
  // Used by Nv* (8-bit) and P*/P*1*/P*2*/P*4* (high-bit-packed u16)
  // families. Two planes:
  //   - Y plane (full-resolution)
  //   - chroma plane (interleaved UV/VU, `chroma_elems` per row)
  //
  // `chroma_v: half | full` selects vertical sampling:
  //   - `half`  → chroma_row = row / 2 (4:2:0)
  //   - `full`  → chroma_row = row     (4:2:2 / 4:4:4)
  //
  // `chroma_field` is the source-side field name (`uv` for normal
  // ordering, `vu` for swapped). The sub-rules below choose between
  // half-width (`*_half`) and full-width (`*`) variants. We keep the
  // names symmetric with the hand-written Row structs (`uv_half`,
  // `vu_half`, `uv`, `vu`).
  //
  // `chroma_elems_per_row: |w| expr` is the per-row payload length in
  // `$elem` units (e.g. `w` for half-width interleaved, `2 * w` for
  // full-width interleaved).
  (
    semi_planar {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ty,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      elem_type: $elem:ty,
      // Chroma field name and stride accessor name on the frame.
      // Field name is e.g. `uv_half`, `vu_half`, `uv`, `vu`.
      // Frame accessor is e.g. `uv()`, `vu()`; stride is
      // `uv_stride()`, `vu_stride()`.
      chroma_field: $chroma_field:ident,
      chroma_plane: $chroma_plane:ident,
      chroma_stride: $chroma_stride:ident,
      chroma_elems_per_row: |$w:ident| $chroma_row_elems:expr,
      chroma_v: $chroma_v:tt,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      $chroma_field: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub(crate) fn new(
        y: &'a [$elem],
        $chroma_field: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, $chroma_field, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Interleaved chroma row (UV-ordered or VU-ordered per format).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn $chroma_field(&self) -> &'a [$elem] {
        self.$chroma_field
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV → RGB matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let $w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let chroma_stride = src.$chroma_stride() as usize;
      let chroma_row_elems: usize = $chroma_row_elems;

      let y_plane = src.y();
      let chroma_plane = src.$chroma_plane();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + $w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let chroma_start = chroma_row * chroma_stride;
        let $chroma_field = &chroma_plane[chroma_start..chroma_start + chroma_row_elems];

        sink.process($row::new(y, $chroma_field, row, matrix, full_range))?;
      }
      Ok(())
    }
  };

  // ---------- planar3 (3 planes: Y + U + V) -------------------------------
  //
  // Used by every Yuv*p (planar) format. Chroma horizontal sampling
  // is captured by `chroma_field_kind` (`half` vs `full`) via the
  // sub-rules `@p3_*` below — half-width gives `u_half`/`v_half`
  // accessors with `width / 2` slicing, full-width gives `u`/`v`
  // accessors with `width` slicing.
  (
    planar3 {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ty,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      elem_type: $elem:ty,
      chroma_h: $chroma_h:tt,
      chroma_v: $chroma_v:tt,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    walker!(@p3_emit $chroma_h
      $(#[$marker_meta])*
      marker: $marker,
      frame: $frame,
      row: $row,
      sink: $sink,
      walker: $walker,
      elem_type: $elem,
      chroma_v: $chroma_v,
      $(#[$row_meta])*
      row_doc: $row_doc,
      $(#[$walker_meta])*
      walker_doc: $walker_doc,
    );
  };

  // ---------- planar3, BITS-generic ---------------------------------------
  //
  // Used by the 9/10/12/14/16-bit Yuv*p families that share an
  // underlying `*Frame16<BITS>` struct. The walker is dispatched with
  // an explicit `BITS` value and forwards to a const-generic inner.
  (
    planar3_bits {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ty,
      // Const-generic frame type the inner walker takes.
      generic_frame: $gframe:ty,
      bits: $bits:expr,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      walker_inner: $walker_inner:ident,
      elem_type: $elem:ty,
      chroma_h: $chroma_h:tt,
      chroma_v: $chroma_v:tt,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    walker!(@p3_emit_bits $chroma_h
      $(#[$marker_meta])*
      marker: $marker,
      frame: $frame,
      generic_frame: $gframe,
      bits: $bits,
      row: $row,
      sink: $sink,
      walker: $walker,
      walker_inner: $walker_inner,
      elem_type: $elem,
      chroma_v: $chroma_v,
      $(#[$row_meta])*
      row_doc: $row_doc,
      $(#[$walker_meta])*
      walker_doc: $walker_doc,
    );
  };

  // ---------- planar4 (4 planes: Y + U + V + A) ---------------------------
  //
  // Used by every Yuva*p planar format. Chroma horizontal sampling
  // (`half` vs `full`) chooses between half-width (`u_half`/`v_half`)
  // and full-width (`u`/`v`) accessors. Alpha is always full-resolution
  // (1:1 with Y).
  (
    planar4 {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ty,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      elem_type: $elem:ty,
      chroma_h: $chroma_h:tt,
      chroma_v: $chroma_v:tt,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    walker!(@p4_emit $chroma_h
      $(#[$marker_meta])*
      marker: $marker,
      frame: $frame,
      row: $row,
      sink: $sink,
      walker: $walker,
      elem_type: $elem,
      chroma_v: $chroma_v,
      $(#[$row_meta])*
      row_doc: $row_doc,
      $(#[$walker_meta])*
      walker_doc: $walker_doc,
    );
  };

  // ---------- planar4, BITS-generic ---------------------------------------
  (
    planar4_bits {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ty,
      generic_frame: $gframe:ty,
      bits: $bits:expr,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      walker_inner: $walker_inner:ident,
      elem_type: $elem:ty,
      chroma_h: $chroma_h:tt,
      chroma_v: $chroma_v:tt,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    walker!(@p4_emit_bits $chroma_h
      $(#[$marker_meta])*
      marker: $marker,
      frame: $frame,
      generic_frame: $gframe,
      bits: $bits,
      row: $row,
      sink: $sink,
      walker: $walker,
      walker_inner: $walker_inner,
      elem_type: $elem,
      chroma_v: $chroma_v,
      $(#[$row_meta])*
      row_doc: $row_doc,
      $(#[$walker_meta])*
      walker_doc: $walker_doc,
    );
  };

  // ===== Internal sub-rules ================================================

  // chroma_v selector: `half` → `row / 2` (4:2:0/4:4:0); `full` → `row`.
  (@chroma_row half $row:expr) => { $row / 2 };
  (@chroma_row full $row:expr) => { $row };

  // ---------- planar3 emitters: half (u_half/v_half) -----------------------
  (@p3_emit half
    $(#[$marker_meta:meta])*
    marker: $marker:ident,
    frame: $frame:ty,
    row: $row:ident,
    sink: $sink:ident,
    walker: $walker:ident,
    elem_type: $elem:ty,
    chroma_v: $chroma_v:tt,
    $(#[$row_meta:meta])*
    row_doc: $row_doc:expr,
    $(#[$walker_meta:meta])*
    walker_doc: $walker_doc:expr,
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      u_half: &'a [$elem],
      v_half: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      #[allow(clippy::too_many_arguments)]
      pub(crate) fn new(
        y: &'a [$elem],
        u_half: &'a [$elem],
        v_half: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, u_half, v_half, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Half-width U (Cb) row — `width / 2` samples.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn u_half(&self) -> &'a [$elem] {
        self.u_half
      }
      /// Half-width V (Cr) row — `width / 2` samples.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn v_half(&self) -> &'a [$elem] {
        self.v_half
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV → RGB matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let u_stride = src.u_stride() as usize;
      let v_stride = src.v_stride() as usize;
      let chroma_width = w / 2;

      let y_plane = src.y();
      let u_plane = src.u();
      let v_plane = src.v();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let u_start = chroma_row * u_stride;
        let v_start = chroma_row * v_stride;
        let u_half = &u_plane[u_start..u_start + chroma_width];
        let v_half = &v_plane[v_start..v_start + chroma_width];

        sink.process($row::new(y, u_half, v_half, row, matrix, full_range))?;
      }
      Ok(())
    }
  };

  // ---------- planar3 emitters: full (u/v) ---------------------------------
  (@p3_emit full
    $(#[$marker_meta:meta])*
    marker: $marker:ident,
    frame: $frame:ty,
    row: $row:ident,
    sink: $sink:ident,
    walker: $walker:ident,
    elem_type: $elem:ty,
    chroma_v: $chroma_v:tt,
    $(#[$row_meta:meta])*
    row_doc: $row_doc:expr,
    $(#[$walker_meta:meta])*
    walker_doc: $walker_doc:expr,
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      u: &'a [$elem],
      v: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      #[allow(clippy::too_many_arguments)]
      pub(crate) fn new(
        y: &'a [$elem],
        u: &'a [$elem],
        v: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, u, v, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Full-width U (Cb) row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn u(&self) -> &'a [$elem] {
        self.u
      }
      /// Full-width V (Cr) row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn v(&self) -> &'a [$elem] {
        self.v
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV → RGB matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let u_stride = src.u_stride() as usize;
      let v_stride = src.v_stride() as usize;

      let y_plane = src.y();
      let u_plane = src.u();
      let v_plane = src.v();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let u_start = chroma_row * u_stride;
        let v_start = chroma_row * v_stride;
        let u = &u_plane[u_start..u_start + w];
        let v = &v_plane[v_start..v_start + w];

        sink.process($row::new(y, u, v, row, matrix, full_range))?;
      }
      Ok(())
    }
  };

  // ---------- planar3 BITS-generic emitters: half --------------------------
  (@p3_emit_bits half
    $(#[$marker_meta:meta])*
    marker: $marker:ident,
    frame: $frame:ty,
    generic_frame: $gframe:ty,
    bits: $bits:expr,
    row: $row:ident,
    sink: $sink:ident,
    walker: $walker:ident,
    walker_inner: $walker_inner:ident,
    elem_type: $elem:ty,
    chroma_v: $chroma_v:tt,
    $(#[$row_meta:meta])*
    row_doc: $row_doc:expr,
    $(#[$walker_meta:meta])*
    walker_doc: $walker_doc:expr,
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      u_half: &'a [$elem],
      v_half: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      #[allow(clippy::too_many_arguments)]
      pub(crate) fn new(
        y: &'a [$elem],
        u_half: &'a [$elem],
        v_half: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, u_half, v_half, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Half-width U (Cb) row — `width / 2` samples.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn u_half(&self) -> &'a [$elem] {
        self.u_half
      }
      /// Half-width V (Cr) row — `width / 2` samples.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn v_half(&self) -> &'a [$elem] {
        self.v_half
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV → RGB matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      $walker_inner::<{ $bits }, S>(src, full_range, matrix, sink)
    }

    #[cfg_attr(not(tarpaulin), inline(always))]
    fn $walker_inner<const BITS: u32, S: $sink>(
      src: &$gframe,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let u_stride = src.u_stride() as usize;
      let v_stride = src.v_stride() as usize;
      let chroma_width = w / 2;

      let y_plane = src.y();
      let u_plane = src.u();
      let v_plane = src.v();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let u_start = chroma_row * u_stride;
        let v_start = chroma_row * v_stride;
        let u_half = &u_plane[u_start..u_start + chroma_width];
        let v_half = &v_plane[v_start..v_start + chroma_width];

        sink.process($row::new(y, u_half, v_half, row, matrix, full_range))?;
      }
      Ok(())
    }
  };

  // ---------- planar3 BITS-generic emitters: full --------------------------
  (@p3_emit_bits full
    $(#[$marker_meta:meta])*
    marker: $marker:ident,
    frame: $frame:ty,
    generic_frame: $gframe:ty,
    bits: $bits:expr,
    row: $row:ident,
    sink: $sink:ident,
    walker: $walker:ident,
    walker_inner: $walker_inner:ident,
    elem_type: $elem:ty,
    chroma_v: $chroma_v:tt,
    $(#[$row_meta:meta])*
    row_doc: $row_doc:expr,
    $(#[$walker_meta:meta])*
    walker_doc: $walker_doc:expr,
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      u: &'a [$elem],
      v: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      #[allow(clippy::too_many_arguments)]
      pub(crate) fn new(
        y: &'a [$elem],
        u: &'a [$elem],
        v: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, u, v, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Full-width U (Cb) row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn u(&self) -> &'a [$elem] {
        self.u
      }
      /// Full-width V (Cr) row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn v(&self) -> &'a [$elem] {
        self.v
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV → RGB matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      $walker_inner::<{ $bits }, S>(src, full_range, matrix, sink)
    }

    #[cfg_attr(not(tarpaulin), inline(always))]
    fn $walker_inner<const BITS: u32, S: $sink>(
      src: &$gframe,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let u_stride = src.u_stride() as usize;
      let v_stride = src.v_stride() as usize;

      let y_plane = src.y();
      let u_plane = src.u();
      let v_plane = src.v();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let u_start = chroma_row * u_stride;
        let v_start = chroma_row * v_stride;
        let u = &u_plane[u_start..u_start + w];
        let v = &v_plane[v_start..v_start + w];

        sink.process($row::new(y, u, v, row, matrix, full_range))?;
      }
      Ok(())
    }
  };

  // ---------- planar4 emitters: half (u_half/v_half) -----------------------
  (@p4_emit half
    $(#[$marker_meta:meta])*
    marker: $marker:ident,
    frame: $frame:ty,
    row: $row:ident,
    sink: $sink:ident,
    walker: $walker:ident,
    elem_type: $elem:ty,
    chroma_v: $chroma_v:tt,
    $(#[$row_meta:meta])*
    row_doc: $row_doc:expr,
    $(#[$walker_meta:meta])*
    walker_doc: $walker_doc:expr,
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      u_half: &'a [$elem],
      v_half: &'a [$elem],
      a: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      #[allow(clippy::too_many_arguments)]
      pub(crate) fn new(
        y: &'a [$elem],
        u_half: &'a [$elem],
        v_half: &'a [$elem],
        a: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, u_half, v_half, a, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Half-width U (Cb) row — `width / 2` samples.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn u_half(&self) -> &'a [$elem] {
        self.u_half
      }
      /// Half-width V (Cr) row — `width / 2` samples.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn v_half(&self) -> &'a [$elem] {
        self.v_half
      }
      /// Full-width alpha row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn a(&self) -> &'a [$elem] {
        self.a
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV → RGB matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let u_stride = src.u_stride() as usize;
      let v_stride = src.v_stride() as usize;
      let a_stride = src.a_stride() as usize;
      let chroma_width = w / 2;

      let y_plane = src.y();
      let u_plane = src.u();
      let v_plane = src.v();
      let a_plane = src.a();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let u_start = chroma_row * u_stride;
        let v_start = chroma_row * v_stride;
        let u_half = &u_plane[u_start..u_start + chroma_width];
        let v_half = &v_plane[v_start..v_start + chroma_width];

        let a_start = row * a_stride;
        let a = &a_plane[a_start..a_start + w];

        sink.process($row::new(
          y, u_half, v_half, a, row, matrix, full_range,
        ))?;
      }
      Ok(())
    }
  };

  // ---------- planar4 emitters: full (u/v) ---------------------------------
  (@p4_emit full
    $(#[$marker_meta:meta])*
    marker: $marker:ident,
    frame: $frame:ty,
    row: $row:ident,
    sink: $sink:ident,
    walker: $walker:ident,
    elem_type: $elem:ty,
    chroma_v: $chroma_v:tt,
    $(#[$row_meta:meta])*
    row_doc: $row_doc:expr,
    $(#[$walker_meta:meta])*
    walker_doc: $walker_doc:expr,
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      u: &'a [$elem],
      v: &'a [$elem],
      a: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      #[allow(clippy::too_many_arguments)]
      pub(crate) fn new(
        y: &'a [$elem],
        u: &'a [$elem],
        v: &'a [$elem],
        a: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, u, v, a, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Full-width U (Cb) row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn u(&self) -> &'a [$elem] {
        self.u
      }
      /// Full-width V (Cr) row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn v(&self) -> &'a [$elem] {
        self.v
      }
      /// Full-width alpha row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn a(&self) -> &'a [$elem] {
        self.a
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV → RGB matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let u_stride = src.u_stride() as usize;
      let v_stride = src.v_stride() as usize;
      let a_stride = src.a_stride() as usize;

      let y_plane = src.y();
      let u_plane = src.u();
      let v_plane = src.v();
      let a_plane = src.a();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let u_start = chroma_row * u_stride;
        let v_start = chroma_row * v_stride;
        let u = &u_plane[u_start..u_start + w];
        let v = &v_plane[v_start..v_start + w];

        let a_start = row * a_stride;
        let a = &a_plane[a_start..a_start + w];

        sink.process($row::new(
          y, u, v, a, row, matrix, full_range,
        ))?;
      }
      Ok(())
    }
  };

  // ---------- planar4 BITS-generic emitters: half --------------------------
  (@p4_emit_bits half
    $(#[$marker_meta:meta])*
    marker: $marker:ident,
    frame: $frame:ty,
    generic_frame: $gframe:ty,
    bits: $bits:expr,
    row: $row:ident,
    sink: $sink:ident,
    walker: $walker:ident,
    walker_inner: $walker_inner:ident,
    elem_type: $elem:ty,
    chroma_v: $chroma_v:tt,
    $(#[$row_meta:meta])*
    row_doc: $row_doc:expr,
    $(#[$walker_meta:meta])*
    walker_doc: $walker_doc:expr,
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      u_half: &'a [$elem],
      v_half: &'a [$elem],
      a: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      #[allow(clippy::too_many_arguments)]
      pub(crate) fn new(
        y: &'a [$elem],
        u_half: &'a [$elem],
        v_half: &'a [$elem],
        a: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, u_half, v_half, a, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Half-width U (Cb) row — `width / 2` samples.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn u_half(&self) -> &'a [$elem] {
        self.u_half
      }
      /// Half-width V (Cr) row — `width / 2` samples.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn v_half(&self) -> &'a [$elem] {
        self.v_half
      }
      /// Full-width alpha row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn a(&self) -> &'a [$elem] {
        self.a
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV → RGB matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      $walker_inner::<{ $bits }, S>(src, full_range, matrix, sink)
    }

    #[cfg_attr(not(tarpaulin), inline(always))]
    fn $walker_inner<const BITS: u32, S: $sink>(
      src: &$gframe,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let u_stride = src.u_stride() as usize;
      let v_stride = src.v_stride() as usize;
      let a_stride = src.a_stride() as usize;
      let chroma_width = w / 2;

      let y_plane = src.y();
      let u_plane = src.u();
      let v_plane = src.v();
      let a_plane = src.a();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let u_start = chroma_row * u_stride;
        let v_start = chroma_row * v_stride;
        let u_half = &u_plane[u_start..u_start + chroma_width];
        let v_half = &v_plane[v_start..v_start + chroma_width];

        let a_start = row * a_stride;
        let a = &a_plane[a_start..a_start + w];

        sink.process($row::new(
          y, u_half, v_half, a, row, matrix, full_range,
        ))?;
      }
      Ok(())
    }
  };

  // ---------- planar1 (single plane — gray / luma-only) --------------------
  //
  // Used by Gray8 (u8 plane) and Gray16 (u16 plane). No chroma planes.
  // The walker reads one Y row per iteration. `elem_type` is `u8` or `u16`.
  (
    planar1 {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ty,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      elem_type: $elem:ty,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub(crate) fn new(
        y: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// Color matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let y_plane = src.y();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];
        sink.process($row::new(y, row, matrix, full_range))?;
      }
      Ok(())
    }
  };

  // ---------- planar1_bits (single u16 plane — GrayN<BITS>) ----------------
  //
  // Used by Gray9/10/12/14. The outer walker is monomorphic over the
  // specific BITS value; the inner walker is const-generic. Same pattern
  // as `planar3_bits`.
  (
    planar1_bits {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ty,
      generic_frame: $gframe:ty,
      bits: $bits:expr,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      walker_inner: $walker_inner:ident,
      elem_type: $elem:ty,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub(crate) fn new(
        y: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// Color matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      $walker_inner::<{ $bits }, S>(src, full_range, matrix, sink)
    }

    #[cfg_attr(not(tarpaulin), inline(always))]
    fn $walker_inner<const BITS: u32, S: $sink>(
      src: &$gframe,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let y_plane = src.y();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];
        sink.process($row::new(y, row, matrix, full_range))?;
      }
      Ok(())
    }
  };

  // ---------- planar4 BITS-generic emitters: full --------------------------
  (@p4_emit_bits full
    $(#[$marker_meta:meta])*
    marker: $marker:ident,
    frame: $frame:ty,
    generic_frame: $gframe:ty,
    bits: $bits:expr,
    row: $row:ident,
    sink: $sink:ident,
    walker: $walker:ident,
    walker_inner: $walker_inner:ident,
    elem_type: $elem:ty,
    chroma_v: $chroma_v:tt,
    $(#[$row_meta:meta])*
    row_doc: $row_doc:expr,
    $(#[$walker_meta:meta])*
    walker_doc: $walker_doc:expr,
  ) => {
    $(#[$marker_meta])*
    pub struct $marker;

    impl $crate::sealed::Sealed for $marker {}
    impl $crate::SourceFormat for $marker {}

    $(#[$row_meta])*
    #[doc = $row_doc]
    #[derive(Debug, Clone, Copy)]
    pub struct $row<'a> {
      y: &'a [$elem],
      u: &'a [$elem],
      v: &'a [$elem],
      a: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      #[allow(clippy::too_many_arguments)]
      pub(crate) fn new(
        y: &'a [$elem],
        u: &'a [$elem],
        v: &'a [$elem],
        a: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, u, v, a, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Full-width U (Cb) row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn u(&self) -> &'a [$elem] {
        self.u
      }
      /// Full-width V (Cr) row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn v(&self) -> &'a [$elem] {
        self.v
      }
      /// Full-width alpha row — `width` samples (1:1 with Y).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn a(&self) -> &'a [$elem] {
        self.a
      }
      /// Output row index within the frame.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn row(&self) -> usize {
        self.row
      }
      /// YUV → RGB matrix carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn matrix(&self) -> $crate::ColorMatrix {
        self.matrix
      }
      /// Full-range flag carried through from the kernel call.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub const fn full_range(&self) -> bool {
        self.full_range
      }
    }

    /// Sinks that consume rows of this source format.
    pub trait $sink: for<'a> $crate::PixelSink<Input<'a> = $row<'a>> {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S: $sink>(
      src: &$frame,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      $walker_inner::<{ $bits }, S>(src, full_range, matrix, sink)
    }

    #[cfg_attr(not(tarpaulin), inline(always))]
    fn $walker_inner<const BITS: u32, S: $sink>(
      src: &$gframe,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error> {
      sink.begin_frame(src.width(), src.height())?;

      let w = src.width() as usize;
      let h = src.height() as usize;
      let y_stride = src.y_stride() as usize;
      let u_stride = src.u_stride() as usize;
      let v_stride = src.v_stride() as usize;
      let a_stride = src.a_stride() as usize;

      let y_plane = src.y();
      let u_plane = src.u();
      let v_plane = src.v();
      let a_plane = src.a();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let u_start = chroma_row * u_stride;
        let v_start = chroma_row * v_stride;
        let u = &u_plane[u_start..u_start + w];
        let v = &v_plane[v_start..v_start + w];

        let a_start = row * a_stride;
        let a = &a_plane[a_start..a_start + w];

        sink.process($row::new(
          y, u, v, a, row, matrix, full_range,
        ))?;
      }
      Ok(())
    }
  };
}
