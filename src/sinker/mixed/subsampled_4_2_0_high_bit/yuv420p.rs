use super::{
  super::{
    GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, NativeRouteChanged,
    RowIndexOutOfRange, RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
    deinterleave_y_high_bit, packed_yuv422_triple_filter_resample, packed_yuv422_triple_resample,
    reset_high_bit_yuv_streams, rgb_row_buf_or_scratch, rgba_plane_row_slice,
    rgba_u16_plane_row_slice,
  },
  yuv420p16_process_native,
};
use crate::{PixelSink, row::*, source::*};

// ---- Yuv420p9 impl -----------------------------------------------------
//
// 9-bit 4:2:0 planar. AV_PIX_FMT_YUV420P9LE — niche AVC High 9 only.
// Reuses the Q15 i32 kernel family at `BITS = 9` via the
// `yuv420p9_to_rgb_*` row primitives (which dispatch to
// `yuv_420p_n_to_rgb_*<9>` internally).

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p9<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 9‑bit low‑packed
  /// (`(1 << 9) - 1 = 511` max).
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

  /// Attaches a packed **8‑bit** RGBA output buffer. The 9‑bit YUV
  /// source is converted to 8‑bit RGBA via the same `BITS = 9` Q15
  /// kernel family used by [`Self::with_rgb`]; the fourth byte per
  /// pixel is alpha = `0xFF` (Yuv420p9 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 9‑bit low‑packed
  /// (`(1 << 9) - 1 = 511` max). Length is measured in `u16`
  /// **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 9) - 1`.
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
}

impl<const BE: bool> Yuv420p9Sink<BE> for MixedSinker<'_, Yuv420p9<BE>> {}

impl<const BE: bool> PixelSink for MixedSinker<'_, Yuv420p9<BE>> {
  type Input<'r> = Yuv420p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Yuv420p9Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 9;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y9,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf9,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf9,
        idx,
        w / 2,
        row.v_half().len(),
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
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // Compute u16 RGB once (to caller's buffer when attached) and fan
    // out to u16 RGBA via the cheap per-pixel pad. RGBA-only avoids the
    // RGB kernel entirely and writes RGBA directly.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p9_to_rgba_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p9_to_rgb_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
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
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p9_to_rgba_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
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

    yuv420p9_to_rgb_row_endian(
      row.y(),
      row.u_half(),
      row.v_half(),
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv420p10 impl -----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p10<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Only available on
  /// sinkers whose source format populates native‑depth `u16` RGB —
  /// calling `with_rgb_u16` on an 8‑bit source sinker (e.g.
  /// [`MixedSinker<Yuv420p>`]) is a compile error rather than a
  /// silent no‑op that would leave the caller's buffer stale.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width x height x 3`. Each element carries a 10‑bit value in
  /// the **low** 10 bits (upper 6 bits zero), matching FFmpeg's
  /// `yuv420p10le` convention. This is **not** the `p010` layout
  /// (which stores samples in the high 10 bits); callers feeding a
  /// p010 consumer must shift the output left by 6.
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if
  /// `buf.len() < width x height x 3`, or `Err(GeometryOverflow)`
  /// on 32‑bit targets when the product overflows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16). The
  /// required length is measured in `u16` **elements**, not bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    // Packed RGB requires `width x height x 3` channel values —
    // that's the same count whether the element type is `u8` or
    // `u16`, so the [`Self::frame_elems`] helper (named for the u8
    // RGB path's byte count) gives the element count here too. No
    // size conversion needed.
    let expected_elements = self.frame_elems(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected_elements, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 10‑bit YUV
  /// source is converted to 8‑bit RGBA via the `BITS = 10` Q15 kernel
  /// family; the fourth byte per pixel is alpha = `0xFF` (Yuv420p10
  /// has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10‑bit
  /// low‑packed (`(1 << 10) - 1 = 1023` max). Length is measured in
  /// `u16` **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 10) - 1`.
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
}

impl<R, const BE: bool> Yuv420p10Sink<BE> for MixedSinker<'_, Yuv420p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv420p10<BE>, R> {
  type Input<'r> = Yuv420p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  #[allow(clippy::too_many_lines)]
  fn process(&mut self, row: Yuv420p10Row<'_>) -> Result<(), Self::Error> {
    // Bit depth is fixed by the format (10) — declared as a const so
    // the downshift for u8 luma stays obvious at the call site.
    const BITS: u32 = 10;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense in depth — see the [`Yuv420p`] impl for the rationale.
    // Row slice checks use the 10‑bit variants of [`RowSlice`] so
    // downstream log output disambiguates from the 8‑bit source impls.
    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y10,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf10,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf10,
        idx,
        w / 2,
        row.v_half().len(),
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
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      native,
      native_420_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: the native tier bins the host-native Y / U / V
    // planes at output resolution and converts ONCE per output row at
    // output width (4:4:4 kernels); the row-stage tier
    // ([`packed_yuv422_triple_resample`]) converts each source row at
    // source width then area-streams it (u8 color, independent native-u16
    // color, native Y). `with_native(false)` forces the latter. The half-
    // width U / V planes are horizontally upsampled in-register by the
    // shared 4:2:0 row kernels — 4:2:0's vertical chroma sharing is
    // already resolved by the walker, which hands this luma row its
    // (vertically-shared) `u_half` / `v_half`, so the per-row chroma
    // contract is identical to 4:2:2's and the same tail binds. Yuv420p
    // exposes no `luma_u16` output, so it is `&mut None` and only `luma`
    // (binned native Y `>> (BITS - 8)`) is emitted.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // A `Filter` plan routes to the filter resampler BEFORE the
      // native/row-stage route machinery: the native fast tier is an
      // area-specific optimization that never sees a filter plan, and the
      // per-sink plan kind is fixed at construction, so a filter sink bypasses
      // the `frozen_native_route` interaction entirely. It converts the
      // separate Y/U/V planes to a source-width u8 + native-u16 RGB row (the
      // SAME closures the row-stage tier uses) and filter-resamples them plus
      // the native Y — the filter twin of the row-stage tier. The shared tail
      // clamps every sub-16-bit colour sample AND the native Y to
      // `(1 << BITS) - 1`. Yuv420p exposes no `luma_u16`, so it is `&mut None`.
      if plan.kind().is_filter() {
        return packed_yuv422_triple_filter_resample::<BITS>(
          luma_filter_stream_u16,
          rgb_filter_stream,
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
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
          |scratch| deinterleave_y_high_bit::<BE>(y, scratch, w),
          |scratch| {
            yuv420p10_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p10_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
      }
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests (`need_luma || need_color` =
      // `luma || rgb || rgba || hsv || rgb_u16 || rgba_u16`). The route
      // freezes only on an output-bearing row a tier ACCEPTS; a no-output
      // call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch. The two tiers carry independent, in-order, once-only
      // stream state, so splitting a frame across them yields a
      // mixed/partial frame rather than a deterministic rejection. The route
      // is both CHECKED here and frozen below (the SET) ONLY on an
      // output-bearing row a tier ACCEPTS — both gate on `need_output`. A
      // no-output call therefore neither checks nor freezes the route: it is
      // a true no-op, route-invisible regardless of row index. A
      // preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route`
      // untouched and a later same-or-other-route retry is not falsely
      // rejected.
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      if *native {
        // Dispatch first; freeze the route to native ONLY after the call
        // returns Ok on an output-bearing row. A no-output call returns
        // Ok(()) with `need_output` false (no freeze); an out-of-sequence /
        // frozen row returns Err via `?` (no freeze) — so only an accepted
        // output-bearing row commits the route.
        yuv420p16_process_native::<BITS, BE>(
          plan,
          native_420_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          y,
          u_half,
          v_half,
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
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
      // freeze the route to row-stage only when the call accepts an
      // output-bearing row (a no-output call returns Ok with `need_output`
      // false; an out-of-sequence / frozen row returns Err via `?`).
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
        &mut None,
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
        |scratch| deinterleave_y_high_bit::<BE>(y, scratch, w),
        |scratch| {
          yuv420p10_to_rgb_row_endian(
            y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
          )
        },
        |scratch| {
          yuv420p10_to_rgb_u16_row_endian(
            y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
          )
        },
      )?;
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma: downshift 10‑bit Y to 8‑bit for the existing u8 luma
    // buffer contract. Bit‑extension by `(BITS - 8)` preserves the
    // most significant bits — functionally equivalent to FFmpeg's
    // `>> (BITS - 8)` conversion used by many downstream analyses.
    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    // u16 outputs are written via the native-depth row primitive, kept
    // independent of the u8 path: the two have different scale params
    // inside `range_params_n` and can't share an intermediate without
    // losing precision. Within the u16 family, however, the RGB row
    // and RGBA row are bit-identical for R/G/B, so we run the RGB
    // kernel once and fan out to RGBA via the cheap pad.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p10_to_rgba_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p10_to_rgb_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
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
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p10_to_rgba_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
      return Ok(());
    }

    // 8‑bit RGB path — either writes to the caller's buffer (when
    // `with_rgb` is set) or to the lazily‑grown scratch (when HSV is
    // requested without RGB). Mirrors the 8‑bit source impls' layout.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    yuv420p10_to_rgb_row_endian(
      row.y(),
      row.u_half(),
      row.v_half(),
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv420p12 impl ----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p12<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Mirrors
  /// [`MixedSinker<Yuv420p10>::with_rgb_u16`] but produces 12‑bit
  /// output (values in `[0, 4095]` in the low 12 of each `u16`, upper
  /// 4 zero). Length is measured in `u16` **elements** (`width x
  /// height x 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_elems(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected_elements, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 12‑bit YUV
  /// source is converted to 8‑bit RGBA via the `BITS = 12` Q15 kernel
  /// family; alpha = `0xFF` (Yuv420p12 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 12‑bit
  /// low‑packed (`(1 << 12) - 1 = 4095` max). Length is measured in
  /// `u16` **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 12) - 1`.
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
}

impl<R, const BE: bool> Yuv420p12Sink<BE> for MixedSinker<'_, Yuv420p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv420p12<BE>, R> {
  type Input<'r> = Yuv420p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  #[allow(clippy::too_many_lines)]
  fn process(&mut self, row: Yuv420p12Row<'_>) -> Result<(), Self::Error> {
    // Bit depth is fixed by the format (12) — declared as a const so
    // the downshift for u8 luma stays obvious at the call site.
    const BITS: u32 = 12;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y12,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf12,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf12,
        idx,
        w / 2,
        row.v_half().len(),
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
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      native,
      native_420_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: native tier (bin native planes, convert once at
    // output width via 4:4:4 kernels) vs row-stage tier (convert each
    // source row then bin); `with_native(false)` forces the latter. See
    // the Yuv420p10 impl for the full chroma-contract rationale.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // A `Filter` plan routes to the filter resampler BEFORE the
      // native/row-stage route machinery: the native fast tier is an
      // area-specific optimization that never sees a filter plan, and the
      // per-sink plan kind is fixed at construction, so a filter sink bypasses
      // the `frozen_native_route` interaction entirely. It converts the
      // separate Y/U/V planes to a source-width u8 + native-u16 RGB row (the
      // SAME closures the row-stage tier uses) and filter-resamples them plus
      // the native Y — the filter twin of the row-stage tier. The shared tail
      // clamps every sub-16-bit colour sample AND the native Y to
      // `(1 << BITS) - 1`. Yuv420p exposes no `luma_u16`, so it is `&mut None`.
      if plan.kind().is_filter() {
        return packed_yuv422_triple_filter_resample::<BITS>(
          luma_filter_stream_u16,
          rgb_filter_stream,
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
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
          |scratch| deinterleave_y_high_bit::<BE>(y, scratch, w),
          |scratch| {
            yuv420p12_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p12_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
      }
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests (`need_luma || need_color` =
      // `luma || rgb || rgba || hsv || rgb_u16 || rgba_u16`). The route
      // freezes only on an output-bearing row a tier ACCEPTS; a no-output
      // call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch. The two tiers carry independent, in-order, once-only
      // stream state, so splitting a frame across them yields a
      // mixed/partial frame rather than a deterministic rejection. The route
      // is both CHECKED here and frozen below (the SET) ONLY on an
      // output-bearing row a tier ACCEPTS — both gate on `need_output`. A
      // no-output call therefore neither checks nor freezes the route: it is
      // a true no-op, route-invisible regardless of row index. A
      // preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route`
      // untouched and a later same-or-other-route retry is not falsely
      // rejected.
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      if *native {
        // Dispatch first; freeze the route to native ONLY after the call
        // returns Ok on an output-bearing row. A no-output call returns
        // Ok(()) with `need_output` false (no freeze); an out-of-sequence /
        // frozen row returns Err via `?` (no freeze) — so only an accepted
        // output-bearing row commits the route.
        yuv420p16_process_native::<BITS, BE>(
          plan,
          native_420_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          y,
          u_half,
          v_half,
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
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
      // freeze the route to row-stage only when the call accepts an
      // output-bearing row (a no-output call returns Ok with `need_output`
      // false; an out-of-sequence / frozen row returns Err via `?`).
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
        &mut None,
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
        |scratch| deinterleave_y_high_bit::<BE>(y, scratch, w),
        |scratch| {
          yuv420p12_to_rgb_row_endian(
            y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
          )
        },
        |scratch| {
          yuv420p12_to_rgb_u16_row_endian(
            y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
          )
        },
      )?;
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p12_to_rgba_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p12_to_rgb_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
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
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p12_to_rgba_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
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

    yuv420p12_to_rgb_row_endian(
      row.y(),
      row.u_half(),
      row.v_half(),
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv420p14 impl ----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p14<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 14‑bit
  /// output (values in `[0, 16383]` in the low 14 of each `u16`, upper
  /// 2 zero). Length is measured in `u16` **elements** (`width x
  /// height x 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_elems(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected_elements, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 14‑bit YUV
  /// source is converted to 8‑bit RGBA via the `BITS = 14` Q15 kernel
  /// family; alpha = `0xFF` (Yuv420p14 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 14‑bit
  /// low‑packed (`(1 << 14) - 1 = 16383` max). Length is measured in
  /// `u16` **elements** (`width x height x 4`). Alpha element is
  /// `(1 << 14) - 1`.
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
}

impl<R, const BE: bool> Yuv420p14Sink<BE> for MixedSinker<'_, Yuv420p14<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv420p14<BE>, R> {
  type Input<'r> = Yuv420p14Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  #[allow(clippy::too_many_lines)]
  fn process(&mut self, row: Yuv420p14Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 14;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y14,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf14,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf14,
        idx,
        w / 2,
        row.v_half().len(),
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
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      native,
      native_420_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: native tier (bin native planes, convert once at
    // output width via 4:4:4 kernels) vs row-stage tier (convert each
    // source row then bin); `with_native(false)` forces the latter. See
    // the Yuv420p10 impl for the full chroma-contract rationale.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // A `Filter` plan routes to the filter resampler BEFORE the
      // native/row-stage route machinery: the native fast tier is an
      // area-specific optimization that never sees a filter plan, and the
      // per-sink plan kind is fixed at construction, so a filter sink bypasses
      // the `frozen_native_route` interaction entirely. It converts the
      // separate Y/U/V planes to a source-width u8 + native-u16 RGB row (the
      // SAME closures the row-stage tier uses) and filter-resamples them plus
      // the native Y — the filter twin of the row-stage tier. The shared tail
      // clamps every sub-16-bit colour sample AND the native Y to
      // `(1 << BITS) - 1`. Yuv420p exposes no `luma_u16`, so it is `&mut None`.
      if plan.kind().is_filter() {
        return packed_yuv422_triple_filter_resample::<BITS>(
          luma_filter_stream_u16,
          rgb_filter_stream,
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
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
          |scratch| deinterleave_y_high_bit::<BE>(y, scratch, w),
          |scratch| {
            yuv420p14_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p14_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
      }
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests (`need_luma || need_color` =
      // `luma || rgb || rgba || hsv || rgb_u16 || rgba_u16`). The route
      // freezes only on an output-bearing row a tier ACCEPTS; a no-output
      // call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch. The two tiers carry independent, in-order, once-only
      // stream state, so splitting a frame across them yields a
      // mixed/partial frame rather than a deterministic rejection. The route
      // is both CHECKED here and frozen below (the SET) ONLY on an
      // output-bearing row a tier ACCEPTS — both gate on `need_output`. A
      // no-output call therefore neither checks nor freezes the route: it is
      // a true no-op, route-invisible regardless of row index. A
      // preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route`
      // untouched and a later same-or-other-route retry is not falsely
      // rejected.
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      if *native {
        // Dispatch first; freeze the route to native ONLY after the call
        // returns Ok on an output-bearing row. A no-output call returns
        // Ok(()) with `need_output` false (no freeze); an out-of-sequence /
        // frozen row returns Err via `?` (no freeze) — so only an accepted
        // output-bearing row commits the route.
        yuv420p16_process_native::<BITS, BE>(
          plan,
          native_420_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          y,
          u_half,
          v_half,
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
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
      // freeze the route to row-stage only when the call accepts an
      // output-bearing row (a no-output call returns Ok with `need_output`
      // false; an out-of-sequence / frozen row returns Err via `?`).
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
        &mut None,
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
        |scratch| deinterleave_y_high_bit::<BE>(y, scratch, w),
        |scratch| {
          yuv420p14_to_rgb_row_endian(
            y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
          )
        },
        |scratch| {
          yuv420p14_to_rgb_u16_row_endian(
            y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
          )
        },
      )?;
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p14_to_rgba_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p14_to_rgb_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
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
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p14_to_rgba_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
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

    yuv420p14_to_rgb_row_endian(
      row.y(),
      row.u_half(),
      row.v_half(),
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Yuv420p16 impl ----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv420p16<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Produces 16‑bit
  /// output (values in `[0, 65535]` — full `u16` range). Length is
  /// measured in `u16` **elements** (`width x height x 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }

  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elements = self.frame_elems(3)?;
    if buf.len() < expected_elements {
      return Err(MixedSinkerError::InsufficientRgbU16Buffer(
        InsufficientBuffer::new(expected_elements, buf.len()),
      ));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **8‑bit** RGBA output buffer. The 16‑bit YUV
  /// source is converted to 8‑bit RGBA via the dedicated `BITS = 16`
  /// kernel family; alpha = `0xFF` (Yuv420p16 has no alpha plane).
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

  /// Attaches a packed **`u16`** RGBA output buffer. 16‑bit output
  /// (full `u16` range). Length is measured in `u16` **elements**
  /// (`width x height x 4`). Alpha element is `u16::MAX`.
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
}

impl<R, const BE: bool> Yuv420p16Sink<BE> for MixedSinker<'_, Yuv420p16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv420p16<BE>, R> {
  type Input<'r> = Yuv420p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  #[allow(clippy::too_many_lines)]
  fn process(&mut self, row: Yuv420p16Row<'_>) -> Result<(), Self::Error> {
    // Luma downshift is `>> 8` — top 8 bits of the 16-bit Y value.
    const BITS: u32 = 16;

    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y16,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf16,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf16,
        idx,
        w / 2,
        row.v_half().len(),
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
      hsv,
      rgb_scratch,
      rgb_scratch_u16,
      luma_scratch_u16,
      rgb_stream,
      rgb_stream_u16,
      luma_stream_u16,
      rgb_filter_stream,
      rgb_filter_stream_u16,
      luma_filter_stream_u16,
      resample_outputs,
      plan,
      native,
      native_420_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: native tier (bin native planes, convert once at
    // output width via 4:4:4 kernels — the dedicated 16-bit i64-chroma
    // family for BITS = 16) vs row-stage tier (convert each source row
    // then bin); `with_native(false)` forces the latter. See the Yuv420p10
    // impl for the full chroma-contract rationale.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u_half, v_half) = (row.y(), row.u_half(), row.v_half());
      // A `Filter` plan routes to the filter resampler BEFORE the
      // native/row-stage route machinery: the native fast tier is an
      // area-specific optimization that never sees a filter plan, and the
      // per-sink plan kind is fixed at construction, so a filter sink bypasses
      // the `frozen_native_route` interaction entirely. It converts the
      // separate Y/U/V planes to a source-width u8 + native-u16 RGB row (the
      // SAME closures the row-stage tier uses) and filter-resamples them plus
      // the native Y — the filter twin of the row-stage tier. The shared tail
      // clamps every sub-16-bit colour sample AND the native Y to
      // `(1 << BITS) - 1`. Yuv420p exposes no `luma_u16`, so it is `&mut None`.
      if plan.kind().is_filter() {
        return packed_yuv422_triple_filter_resample::<BITS>(
          luma_filter_stream_u16,
          rgb_filter_stream,
          rgb_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
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
          |scratch| deinterleave_y_high_bit::<BE>(y, scratch, w),
          |scratch| {
            yuv420p16_to_rgb_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
          |scratch| {
            yuv420p16_to_rgb_u16_row_endian(
              y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
            )
          },
        );
      }
      // Whether this call carries any output — the EXACT set both tiers'
      // preflight tests (`need_luma || need_color` =
      // `luma || rgb || rgba || hsv || rgb_u16 || rgba_u16`). The route
      // freezes only on an output-bearing row a tier ACCEPTS; a no-output
      // call consumes no stream state, so it must not freeze.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch. The two tiers carry independent, in-order, once-only
      // stream state, so splitting a frame across them yields a
      // mixed/partial frame rather than a deterministic rejection. The route
      // is both CHECKED here and frozen below (the SET) ONLY on an
      // output-bearing row a tier ACCEPTS — both gate on `need_output`. A
      // no-output call therefore neither checks nor freezes the route: it is
      // a true no-op, route-invisible regardless of row index. A
      // preflight-rejected (out-of-sequence / frozen) output-bearing call
      // returns Err before the SET, so it leaves `frozen_native_route`
      // untouched and a later same-or-other-route retry is not falsely
      // rejected.
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      if *native {
        // Dispatch first; freeze the route to native ONLY after the call
        // returns Ok on an output-bearing row. A no-output call returns
        // Ok(()) with `need_output` false (no freeze); an out-of-sequence /
        // frozen row returns Err via `?` (no freeze) — so only an accepted
        // output-bearing row commits the route.
        yuv420p16_process_native::<BITS, BE>(
          plan,
          native_420_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          y,
          u_half,
          v_half,
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
      // Row-stage tail. Same CHECK-before / SET-after split: dispatch, then
      // freeze the route to row-stage only when the call accepts an
      // output-bearing row (a no-output call returns Ok with `need_output`
      // false; an out-of-sequence / frozen row returns Err via `?`).
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
        &mut None,
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
        |scratch| deinterleave_y_high_bit::<BE>(y, scratch, w),
        |scratch| {
          yuv420p16_to_rgb_row_endian(
            y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
          )
        },
        |scratch| {
          yuv420p16_to_rgb_u16_row_endian(
            y, u_half, v_half, scratch, w, matrix, full_range, use_simd, BE,
          )
        },
      )?;
      if frozen_native_route.is_none() && need_output {
        *frozen_native_route = Some(false);
      }
      return Ok(());
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — without this, a valid BE mid-gray sample
        // (`1 << (BITS - 1)`, e.g. `0x0100` for 9-bit, `0x0200` for
        // 10-bit, `0x0800` for 12-bit) would be byte-swapped on a LE
        // host and the `>> (BITS - 8)` would write 0 instead of 128.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) — see Yuv420p10 for rationale.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p16_to_rgba_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
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
      yuv420p16_to_rgb_u16_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      if want_rgba_u16 {
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
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv420p16_to_rgba_row_endian(
        row.y(),
        row.u_half(),
        row.v_half(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if !need_rgb_kernel {
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

    yuv420p16_to_rgb_row_endian(
      row.y(),
      row.u_half(),
      row.v_half(),
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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
