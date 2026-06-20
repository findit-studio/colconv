//! Sinker impl for the packed v210 source format — Ship 11a (Tier 4
//! 10-bit pro-broadcast SDI). Full output coverage: u8 + native-depth
//! u16 RGB / RGBA / luma + u8 HSV.
//!
//! v210 packs 12 x 10-bit samples per 16-byte word = 6 pixels (4:2:2
//! with 6 Y + 3 Cb + 3 Cr per word). The sinker's configured width
//! must be **even** (4:2:2 chroma pair) — partial last words (widths
//! not divisible by 6, e.g. 720p = 1280) are supported. Odd widths
//! surface as [`MixedSinkerError::WidthAlignment`] (with
//! [`WidthAlignmentRequirement::Even`]) before any kernel runs,
//! preserving the no-panic contract.
//!
//! [`WidthAlignmentRequirement::Even`]: super::WidthAlignmentRequirement::Even
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline at
//!   `BITS = 10`, downshifted to u8; RGBA alpha is forced to `0xFF`
//!   (v210 has no alpha channel).
//! - `with_rgb_u16` / `with_rgba_u16` — same pipeline at native
//!   10-bit depth, low-bit-packed in `u16`; RGBA alpha is `1023`.
//! - `with_luma` — extracts the 6 Y values per word and downshifts
//!   `>> 2` to u8.
//! - `with_luma_u16` — extracts the 10-bit Y values into u16
//!   (low-bit-packed). Tier 4 is the first consumer of this API.
//! - `with_hsv` — stages an internal RGB scratch (or the user's RGB
//!   buffer if attached) and runs the existing `rgb_to_hsv_row`
//!   kernel on the staged u8 RGB.
//!
//! When both u8 RGB and u8 RGBA outputs are requested, the RGBA plane
//! is derived from the just-computed u8 RGB row via
//! [`expand_rgb_to_rgba_row`] (Strategy A) instead of running a
//! second YUV→RGB kernel. The same Strategy A applies on the u16
//! path via [`expand_rgb_u16_to_rgba_u16_row::<10>`]. When only the
//! RGBA variant is wanted, the dedicated `_to_rgba_row` /
//! `_to_rgba_u16_row` kernel writes the output buffer directly
//! without staging RGB.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  packed_yuv422_triple_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
// `NativeRouteChanged` is raised only by the native fast tier's route-flip
// guard, and `v210_process_native` exists only when the reused high-bit planar
// join is compiled in. Gated to the native tier's feature intersection.
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
use super::{NativeRouteChanged, v210_process_native};
// The RFC #238 insertion-point selector decides the native-vs-row-stage
// splice; it is only consulted inside the native tier's `cfg`, so its
// import shares that intersection.
#[cfg(all(feature = "v210", feature = "yuv-planar"))]
use crate::resample::{AveragingDomain, InsertionContext, InsertionPoint, select_insertion_point};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row,
    v210_to_luma_row_endian, v210_to_luma_u16_row_endian, v210_to_rgb_row_endian,
    v210_to_rgb_u16_row_endian, v210_to_rgba_row_endian, v210_to_rgba_u16_row_endian,
  },
  source::{V210, V210Row, V210Sink},
};

impl<'a, R, const BE: bool> MixedSinker<'a, V210<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (v210 has no alpha channel).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaBuffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer. 10-bit
  /// low-bit-packed (`[0, 1023]`); length is measured in `u16`
  /// **elements** (`width x height x 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGBA output buffer. 10-bit
  /// low-bit-packed (`[0, 1023]`); alpha element is `1023`. Length
  /// is measured in `u16` **elements** (`width x height x 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a native-depth **`u16`** luma output buffer. v210 is
  /// the first consumer of this Tier 4 API: the 10-bit Y samples
  /// are extracted directly out of the v210 word packing into the
  /// caller's `u16` buffer (low-bit-packed, `[0, 1023]`). Length
  /// is measured in `u16` **elements** (`width x height`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_luma_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_pixels()?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaU16Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

impl<R, const BE: bool> V210Sink<BE> for MixedSinker<'_, V210<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, V210<BE>, R> {
  type Input<'r> = V210Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if !self.width.is_multiple_of(2) {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    // New frame: restart the three row-stage streams (lazily created in
    // `process`, so a direct-`process` caller that skips `begin_frame`
    // still gets a correctly initialized first frame) and drop the frozen
    // output set.
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    // New frame: restart the native join and clear the per-frame frozen
    // native/row-stage route so the next frame may pick either tier; a
    // mid-frame flip stays rejected. Gated to the native tier's feature
    // intersection (the planar join the native tier reuses is compiled only
    // under `yuv-planar`).
    #[cfg(all(feature = "v210", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar_u16.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "v210", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: V210Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if !w.is_multiple_of(2) {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }

    let packed_expected =
      w.div_ceil(6)
        .checked_mul(16)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 16,
        )))?;
    if row.v210().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::V210Packed,
        idx,
        packed_expected,
        row.v210().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    let Self {
      rgb,
      rgb_u16,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      resample_outputs,
      plan,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      native_planar_u16,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      v210_y_full,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      v210_u_half,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      v210_v_half,
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      frozen_native_route,
      ..
    } = self;
    let packed = row.v210();

    // Non-identity plan. V210 is area-only at the row stage (no filter route),
    // so a `Filter` plan falls straight through to `packed_yuv422_triple_resample`,
    // which rejects it as `unsupported_filter` before any work. For an `Area`
    // plan: when the native tier is enabled (and the planar join it reuses is
    // compiled in), bin the native Y / U / V planes at output resolution and
    // convert once per output row, de-packing the V210 word packing into
    // wrapper-owned logical scratch first; otherwise (under `with_native(false)`,
    // or a `v210`-solo build where the planar join is absent) feed the shared
    // area triple-resample tail, which bins the three native-precision
    // conversions (u8 colour, native u16 colour, native Y) directly. The reused
    // planar join now emits the native-depth `luma_u16` (the clamped binned Y),
    // so attaching `luma_u16` no longer forces row-stage — the route depends only
    // on `with_native`. The output set is frozen on the first resampled row, so
    // the native/row-stage route stays stable across a frame and the mid-frame
    // flip guard catches a `set_native` toggle.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // Whether this call carries any output a tier ACCEPTS — the EXACT set the
      // tier preflight tests. The route freezes only on an output-bearing row a
      // tier accepts; a no-output call consumes no stream state, so it must not
      // freeze. (Only consulted when the native tier is compiled in; a filter
      // plan never reaches here — it falls through to the row-stage reject.)
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      let need_output = !plan.kind().is_filter()
        && (luma.is_some()
          || luma_u16.is_some()
          || rgb.is_some()
          || rgba.is_some()
          || rgb_u16.is_some()
          || rgba_u16.is_some()
          || hsv.is_some());
      // The reused high-bit non-4:2:0 planar join now emits BOTH the u8 `luma`
      // and the native-depth `luma_u16` (the clamped binned Y), so the native
      // tier serves every output set V210 exposes; route to native purely on
      // `with_native`, for an `Area` plan only (the native tier is area-only).
      // The output-set freeze keeps this invariant across a frame. The
      // splice stage is the RFC #238 selector's; folding the area-only guard
      // into `area_plan` reproduces the former `*native && !is_filter`
      // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      let take_native = matches!(
        select_insertion_point(
          AveragingDomain::Encoded,
          InsertionContext {
            native_eligible: cfg!(all(feature = "v210", feature = "yuv-planar")),
            with_native: *native,
            area_plan: !plan.kind().is_filter(),
          },
        ),
        InsertionPoint::NativeCodes
      );
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch (the two tiers carry independent, in-order, once-only stream
      // state). CHECKED here and frozen below ONLY on an output-bearing row a
      // tier ACCEPTS — both gate on `need_output`. (Mirrors the high-bit packed
      // 4:2:2 `y2xx`.)
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != take_native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      if take_native {
        // Dispatch first; freeze the route to native ONLY after the call
        // returns Ok on an output-bearing row.
        v210_process_native::<BE>(
          plan,
          native_planar_u16,
          v210_y_full,
          v210_u_half,
          v210_v_half,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          packed,
          matrix,
          full_range,
          idx,
          w,
          h,
          use_simd,
        )?;
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(true);
        }
        return Ok(());
      }
      // Row-stage area tail (the route under `with_native(false)`, the only
      // route under a `v210`-solo build where the planar join is absent, and the
      // path a `Filter` plan takes to be rejected as `unsupported_filter`). Same
      // CHECK-before / SET-after split. V210 reuses the y2xx route at `BITS = 10`
      // with its own word-packing decode closures.
      packed_yuv422_triple_resample::<BITS>(
        luma_stream_u16,
        rgb_stream,
        rgb_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        luma_scratch_u16,
        rgb_scratch,
        rgb_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        matrix,
        full_range,
        |scratch| v210_to_luma_u16_row_endian(packed, scratch, w, use_simd, BE),
        |scratch| v210_to_rgb_row_endian(packed, scratch, w, matrix, full_range, use_simd, BE),
        |scratch| v210_to_rgb_u16_row_endian(packed, scratch, w, matrix, full_range, use_simd, BE),
      )?;
      #[cfg(all(feature = "v210", feature = "yuv-planar"))]
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma u8 — extract 8-bit Y bytes from the v210 plane via the
    // dedicated kernel (downshifts 10→8 inline).
    if let Some(buf) = luma.as_deref_mut() {
      v210_to_luma_row_endian(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
      );
    }
    // Luma u16 — extract 10-bit Y values at native depth.
    if let Some(buf) = luma_u16.as_deref_mut() {
      v210_to_luma_u16_row_endian(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
      );
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      // Standalone u16 RGBA fast path — write directly into the
      // caller's buffer; no staging.
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      v210_to_rgba_u16_row_endian(
        packed,
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      v210_to_rgb_u16_row_endian(
        packed,
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
        // Strategy A u16 fan-out — derive RGBA from the just-computed
        // RGB row instead of running a second YUV→RGB kernel.
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A) =====
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb_kernel = want_rgb || want_hsv;

    // Standalone u8 RGBA fast path — no RGB / HSV requested. Run the
    // dedicated RGBA kernel directly into the output buffer; avoids
    // both the scratch allocation and the RGB→RGBA expand pass.
    if want_rgba && !need_u8_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      v210_to_rgba_row_endian(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_u8_rgb_kernel {
      return Ok(());
    }

    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    v210_to_rgb_row_endian(
      packed,
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
      BE,
    );

    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      rgb_to_hsv_row(
        rgb_row,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Strategy A u8 fan-out — derive RGBA from the just-computed RGB
    // row instead of running a second YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
