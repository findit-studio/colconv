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

  // ---------- packed_be (single-buffer with `<const BE: bool>`) -------------
  //
  // Phase 4 — Frame BE flag. Same shape as `packed { ... }` above, but the
  // marker, Sink subtrait, and walker fn carry a `<const BE: bool>` parameter
  // (defaulted on the marker to `false` for back-compat). The frame type is
  // expected to also be `<'a, const BE: bool>` (defaulted to `false`). Sinker
  // impls then specialize as `MixedSinker<Marker<BE>>` and propagate `BE`
  // into the row-kernel call.
  //
  // The Row type itself is **not** parameterized on BE — Row is just borrowed
  // bytes; the kernel monomorphization picks up `BE` from the sinker type.
  //
  // Two walker fns are generated:
  //   - `$walker_endian<S, const BE: bool>(&$frame<'_, BE>, ...)` — the
  //     full const-generic helper (LE + BE callers).
  //   - `$walker<S>(&$frame<'_, false>, ...)` — LE-only back-compat
  //     wrapper preserving the pre-Phase-4 (`packed`) signature so
  //     downstream explicit-turbofish callers (`$walker::<MySink>(...)`)
  //     keep compiling. Function-position const-generic defaults aren't
  //     allowed by Rust, so the wrapper is required for source compat.
  //
  // Used by Tier 8 trial: Rgb48, Bgr48, Rgba64, Bgra64, X2Rgb10, X2Bgr10.
  //
  // NOTE: The Y2xx family (Y210/Y212/Y216) is intentionally handled by a
  // separate `packed_be_y2xx` arm below rather than reusing this arm. The
  // axis of difference is the *frame type's const-generic shape*: this arm
  // emits `&$frame<'_, BE>` (one const param), while the Y2xx frame is
  // `Y2xxFrame<'a, BITS, BE>` (two const params, requiring the BITS literal
  // to be threaded through the macro spec). A unified arm would either need
  // an awkward optional `bits:` metavariable conditionally injected into the
  // type spelling, or a `frame_ty: $ty` capture that requires every existing
  // caller of this arm to migrate from the `frame: $ident` shorthand.
  // Neither buys much over keeping the two arms parallel, so we accept the
  // duplication. See the symmetric note on `packed_be_y2xx` below.
  (
    packed_be {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ident,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      walker_endian: $walker_endian:ident,
      buf_field: $buf:ident,
      elem_type: $elem:ty,
      row_elems: |$w:ident| $row_elems:expr,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    $(#[$marker_meta])*
    pub struct $marker<const BE: bool = false>;

    impl<const BE: bool> $crate::sealed::Sealed for $marker<BE> {}
    impl<const BE: bool> $crate::SourceFormat for $marker<BE> {}

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

    /// Sinks that consume rows of this source format. The `<const BE>`
    /// parameter encodes the source byte-order — sinkers typically impl
    /// for one specific `BE` matching their stored `MixedSinker<Marker<BE>>`
    /// monomorphization. The Row type does not carry `BE`; the BE-aware
    /// kernel dispatch happens inside `process` via the sinker's own
    /// `<const BE>` parameter.
    ///
    /// `BE` defaults to `false` (LE) so downstream LE-only custom sinks
    /// can keep writing `impl $sink for MySink` / `S: $sink` without
    /// migrating to an explicit const argument.
    pub trait $sink<const BE: bool = false>:
      for<'a> $crate::PixelSink<Input<'a> = $row<'a>>
    {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker_endian<S, const BE: bool>(
      src: &$frame<'_, BE>,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error>
    where
      S: $sink<BE>,
    {
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

    /// LE-only back-compat wrapper preserving the pre-Phase-4 `packed`
    /// walker signature. Forwards to the const-generic helper with
    /// `BE = false`.
    ///
    /// Rust forbids defaults on function-position const-generic
    /// parameters, so an explicit-turbofish caller written before the
    /// `packed` → `packed_be` migration (`$walker::<MySink>(...)`)
    /// would otherwise fail to compile. Keeping this single-generic
    /// wrapper preserves source compatibility for those call sites.
    /// BE-aware callers should use the `_endian` helper directly.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn $walker<S>(
      src: &$frame<'_, false>,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error>
    where
      S: $sink<false>,
    {
      $walker_endian::<S, false>(src, full_range, matrix, sink)
    }
  };

  // ---------- packed_be_y2xx (single-buffer Y2xx with `<const BE>`) --------
  //
  // Phase 4 Tier 4 — variant of `packed_be` for the Y2xx family
  // (`Y210`/`Y212`/`Y216`) whose underlying frame type is the shared
  // `Y2xxFrame<'a, const BITS: u32, const BE: bool = false>`. The macro
  // takes both the underlying `frame_inner: $frame_inner:ident` (always
  // `Y2xxFrame`) and the `bits: $bits:literal` BITS literal so it can
  // emit `$frame_inner<'_, $bits, BE>` in the walker signature.
  //
  // The marker, Sink subtrait, and Row are identical in shape to
  // `packed_be`. The walker fn signature differs: the frame parameter
  // is `&Y2xxFrame<'_, BITS, BE>` instead of `&$frame<'_, BE>`.
  //
  // Two walker fns are generated (mirroring `packed_be`):
  //   - `$walker_endian<S, const BE: bool>(&Y2xxFrame<'_, BITS, BE>, ...)`
  //     — the full const-generic helper (LE + BE callers).
  //   - `$walker<S>(&Y2xxFrame<'_, BITS, false>, ...)` — LE-only
  //     back-compat wrapper preserving the pre-Phase-4 signature so
  //     downstream explicit-turbofish callers (`$walker::<MySink>(...)`)
  //     keep compiling.
  //
  // Used by Tier 4: Y210, Y212, Y216 (each pinning a different BITS
  // literal).
  //
  // NOTE: This arm is kept separate from `packed_be` rather than unified
  // for the reason called out at the head of the `packed_be` arm: the
  // const-generic shape of the frame type (1 vs 2 const params) is the
  // axis of difference, and folding both into one arm would need either a
  // conditional `bits:` metavariable injected into the type spelling or a
  // breaking migration of every existing `packed_be` caller from the
  // `frame: $ident` shorthand to a full `frame_ty: $ty` capture. Two
  // ~80-line arms with a clearly named distinction is the cleaner trade.
  (
    packed_be_y2xx {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame_inner: $frame_inner:ident,
      bits: $bits:literal,
      row: $row:ident,
      sink: $sink:ident,
      walker: $walker:ident,
      walker_endian: $walker_endian:ident,
      buf_field: $buf:ident,
      elem_type: $elem:ty,
      row_elems: |$w:ident| $row_elems:expr,
      $(#[$row_meta:meta])*
      row_doc: $row_doc:expr,
      $(#[$walker_meta:meta])*
      walker_doc: $walker_doc:expr,
    }
  ) => {
    $(#[$marker_meta])*
    pub struct $marker<const BE: bool = false>;

    impl<const BE: bool> $crate::sealed::Sealed for $marker<BE> {}
    impl<const BE: bool> $crate::SourceFormat for $marker<BE> {}

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

    /// Sinks that consume rows of this Y2xx source format. The `<const BE>`
    /// parameter encodes the source byte-order — sinkers typically impl
    /// for one specific `BE`. `BE` defaults to `false` (LE) so downstream
    /// LE-only custom sinks can keep writing `impl $sink for MySink`
    /// without migrating to an explicit const argument.
    pub trait $sink<const BE: bool = false>:
      for<'a> $crate::PixelSink<Input<'a> = $row<'a>>
    {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker_endian<S, const BE: bool>(
      src: &$crate::frame::$frame_inner<'_, $bits, BE>,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error>
    where
      S: $sink<BE>,
    {
      sink.begin_frame(src.width(), src.height())?;

      let $w = src.width() as usize;
      let h = src.height() as usize;
      let stride = src.stride() as usize;
      let row_elems: usize = $row_elems;
      let plane = src.packed();

      for row in 0..h {
        let start = row * stride;
        let $buf = &plane[start..start + row_elems];
        sink.process($row::new($buf, row, matrix, full_range))?;
      }
      Ok(())
    }

    /// LE-only back-compat wrapper preserving the pre-Phase-4 walker
    /// signature. Forwards to the const-generic helper with `BE = false`.
    ///
    /// Rust forbids defaults on function-position const-generic
    /// parameters, so an explicit-turbofish caller written before the
    /// `packed` → `packed_be_y2xx` migration (`$walker::<MySink>(...)`)
    /// would otherwise fail to compile. Keeping this single-generic
    /// wrapper preserves source compatibility for those call sites.
    /// BE-aware callers should use the `_endian` helper directly.
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn $walker<S>(
      src: &$crate::frame::$frame_inner<'_, $bits, false>,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error>
    where
      S: $sink<false>,
    {
      $walker_endian::<S, false>(src, full_range, matrix, sink)
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

  // chroma_v selector: `half` → `row / 2` (4:2:0/4:4:0); `full` → `row`;
  // `quarter` → `row / 4` (4:1:0 — chroma row covers 4 consecutive Y rows).
  (@chroma_row half $row:expr) => { $row / 2 };
  (@chroma_row full $row:expr) => { $row };
  (@chroma_row quarter $row:expr) => { $row / 4 };

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

  // ---------- planar3 emitters: quarter (u_quarter/v_quarter) ------------
  //
  // Used by 4:1:1 planar (`Yuv411p`) — chroma is **quarter-width**,
  // full-height — and by 4:1:0 planar (`Yuv410p`) — chroma is
  // quarter-width AND quarter-height. The walker slices `width / 4`
  // chroma samples per `width` Y samples; the per-call `chroma_v`
  // selector (`full` vs `quarter`) controls vertical row mapping
  // via `@chroma_row`. The Sink subtrait pins the row type so
  // direct callers can constrain "I take quarter-width rows" at the
  // type level.
  (@p3_emit quarter
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
      u_quarter: &'a [$elem],
      v_quarter: &'a [$elem],
      row: usize,
      matrix: $crate::ColorMatrix,
      full_range: bool,
    }

    impl<'a> $row<'a> {
      #[cfg_attr(not(tarpaulin), inline(always))]
      #[allow(clippy::too_many_arguments)]
      pub(crate) fn new(
        y: &'a [$elem],
        u_quarter: &'a [$elem],
        v_quarter: &'a [$elem],
        row: usize,
        matrix: $crate::ColorMatrix,
        full_range: bool,
      ) -> Self {
        Self { y, u_quarter, v_quarter, row, matrix, full_range }
      }
      /// Full-width Y (luma) row.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn y(&self) -> &'a [$elem] {
        self.y
      }
      /// Quarter-width U (Cb) row — `width.div_ceil(4)` samples. Each
      /// sample is duplicated across 4 adjacent Y columns by the kernel
      /// (4:1:1 / 4:1:0). For Yuv410p `width % 4 == 0` is enforced at
      /// the frame layer so `width.div_ceil(4) == width / 4`; Yuv411p
      /// accepts arbitrary widths via FFmpeg ceiling chroma, so the
      /// final chroma sample may cover a partial 1..3-pixel group of
      /// trailing Y columns.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn u_quarter(&self) -> &'a [$elem] {
        self.u_quarter
      }
      /// Quarter-width V (Cr) row — `width.div_ceil(4)` samples
      /// (4:1:1 / 4:1:0). See [`Self::u_quarter`] for the Yuv410p-vs-
      /// Yuv411p width-rounding distinction.
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn v_quarter(&self) -> &'a [$elem] {
        self.v_quarter
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
      // 4:1:1 / 4:1:0 quarter chroma horizontally. Yuv410p still
      // requires `w % 4 == 0` at the frame layer; Yuv411p relaxes to
      // FFmpeg-style ceiling chroma (`(w + 3) >> 2`) so non-4-aligned
      // widths get a partial 1..3-pixel final chroma group. Using
      // `div_ceil(4)` here is exact for both: for 4-multiple widths
      // `div_ceil(4) == w / 4`.
      let chroma_width = w.div_ceil(4);

      let y_plane = src.y();
      let u_plane = src.u();
      let v_plane = src.v();

      for row in 0..h {
        let y_start = row * y_stride;
        let y = &y_plane[y_start..y_start + w];

        let chroma_row = walker!(@chroma_row $chroma_v row);
        let u_start = chroma_row * u_stride;
        let v_start = chroma_row * v_stride;
        let u_quarter = &u_plane[u_start..u_start + chroma_width];
        let v_quarter = &v_plane[v_start..v_start + chroma_width];

        sink.process($row::new(y, u_quarter, v_quarter, row, matrix, full_range))?;
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

  // ---------- planar1_be (single plane with `<const BE: bool>`) ------------
  //
  // Phase 4 — Frame BE flag. Same shape as `planar1 { ... }` above, but the
  // marker, Sink subtrait, and walker fn carry a `<const BE: bool>` parameter
  // (defaulted on the marker to `false` for back-compat). The frame type is
  // expected to also be `<'a, const BE: bool>` (defaulted to `false`).
  //
  // The Row type itself is **not** parameterized on BE — the kernel
  // monomorphization picks up `BE` from the sinker type at the dispatch site.
  //
  // Used by Tier 11: Gray16 (u16 plane), Grayf32 (f32 plane).
  (
    planar1_be {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ident,
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
    pub struct $marker<const BE: bool = false>;

    impl<const BE: bool> $crate::sealed::Sealed for $marker<BE> {}
    impl<const BE: bool> $crate::SourceFormat for $marker<BE> {}

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

    /// Sinks that consume rows of this source format. The `<const BE>`
    /// parameter encodes the source byte-order — sinkers typically impl
    /// for one specific `BE` matching their stored
    /// `MixedSinker<Marker<BE>>` monomorphization.
    pub trait $sink<const BE: bool>:
      for<'a> $crate::PixelSink<Input<'a> = $row<'a>>
    {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S, const BE: bool>(
      src: &$frame<'_, BE>,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error>
    where
      S: $sink<BE>,
    {
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

  // ---------- planar1_bits_be (BITS-generic + BE-generic, single u16 plane) -
  //
  // Phase 4 — Frame BE flag. Same shape as `planar1_bits { ... }` above, but
  // the marker, Sink subtrait, and walker fn carry an additional
  // `<const BE: bool>` parameter (defaulted on the marker to `false` for
  // back-compat). The outer walker is monomorphic over the specific BITS
  // value but generic over `BE`; the inner walker is const-generic over
  // both BITS and BE.
  //
  // Used by Tier 11: Gray9 / Gray10 / Gray12 / Gray14.
  (
    planar1_bits_be {
      $(#[$marker_meta:meta])*
      marker: $marker:ident,
      frame: $frame:ident,
      generic_frame: $gframe:ident,
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
    pub struct $marker<const BE: bool = false>;

    impl<const BE: bool> $crate::sealed::Sealed for $marker<BE> {}
    impl<const BE: bool> $crate::SourceFormat for $marker<BE> {}

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

    /// Sinks that consume rows of this source format. The `<const BE>`
    /// parameter encodes the source byte-order — sinkers typically impl
    /// for one specific `BE` matching their stored
    /// `MixedSinker<Marker<BE>>` monomorphization.
    pub trait $sink<const BE: bool>:
      for<'a> $crate::PixelSink<Input<'a> = $row<'a>>
    {}

    $(#[$walker_meta])*
    #[doc = $walker_doc]
    pub fn $walker<S, const BE: bool>(
      src: &$frame<'_, BE>,
      full_range: bool,
      matrix: $crate::ColorMatrix,
      sink: &mut S,
    ) -> Result<(), S::Error>
    where
      S: $sink<BE>,
    {
      $walker_inner::<{ $bits }, BE, S>(src, full_range, matrix, sink)
    }

    #[cfg_attr(not(tarpaulin), inline(always))]
    fn $walker_inner<const BITS: u32, const BE: bool, S: $sink<BE>>(
      src: &$gframe<'_, BITS, BE>,
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
