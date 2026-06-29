use super::super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, NativeRouteChanged,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, check_dimensions_match,
  deinterleave_y_high_bit_masked, packed_yuv444_triple_filter_resample,
  packed_yuv444_triple_resample, reset_high_bit_yuv_streams, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice, yuv_planar16_process_native,
};
use crate::{
  PixelSink,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, ResamplePlan, select_insertion_point,
  },
  row::*,
  source::*,
};

/// The high-bit 4:4:0 planar formats (`Yuv440p10` / `Yuv440p12`) ship the
/// non-4:2:0 native planar fast tier ([`yuv_planar16_process_native`]), so
/// each is statically eligible to splice an [`AveragingDomain::Encoded`] area
/// downscale at the native codes.
const YUV440P_HIGH_BIT_NATIVE_ELIGIBLE: bool = true;

// ---- Yuv440p10 impl -----------------------------------------------------
//
// 4:4:0 planar 10‑bit. Same row math as 4:4:4 10-bit; reuses
// `yuv444p10_to_rgb_*`. Walker handles the half-height chroma.

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv440p10<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 10-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
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

  /// Attaches a packed **8-bit** RGBA output buffer. Yuv440p10 reuses
  /// the `BITS = 10` 4:4:4 RGBA kernel; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 10-bit low-packed
  /// (`[0, 1023]`); alpha element is `1023`.
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

impl<R, const BE: bool> Yuv440p10Sink<BE> for MixedSinker<'_, Yuv440p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv440p10<BE>, R> {
  type Input<'r> = Yuv440p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv440p10Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 10;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y10,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull10,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull10,
        idx,
        w,
        row.v().len(),
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
      native_planar_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:4:4 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). 4:4:0 is
    // full-width chroma (no horizontal upsampling, same per-row contract
    // as 4:4:4) — its vertical chroma sharing is already resolved by the
    // walker, which hands this luma row the (vertically-shared) full-width
    // `u` / `v`, so the converted RGB is full-res and the 4:4:4 tail binds
    // via the shared `yuv444pN_to_rgb_*` kernels. Yuv440p exposes no
    // `luma_u16` output, so it is `&mut None` and only `luma` (binned
    // native Y `>> (BITS - 8)`) is emitted. The span kind picks the engine:
    // area binning, or the signed-coefficient filter twin (both convert the
    // YUV to RGB with the same closures and resample in RGB space, so filter
    // colour equals the RGB filter of the converted pixels and matches area
    // up to the kernel). The filter tail clamps every sub-16-bit colour
    // sample AND the native Y to `(1 << BITS) - 1` before publishing.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u, v) = (row.y(), row.u(), row.v());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            yuv444p10_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p10_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
        );
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV440P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:0: chroma `w x h/2` — full width, half height; a chroma row per
          // TWO Y rows (`chroma_vsub = 2`, like 4:2:0 vertically; `chroma_w = w`),
          // chroma plan full-width horizontal + luma-domain `area_halved`
          // vertical.
          yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:4:0 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u,
            v,
            matrix,
            full_range,
            idx,
            w,
            h,
            2,
            w,
            || ResamplePlan::area_chroma_440(w, h, plan.out_w(), plan.out_h(), 0.0, 0.0),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          packed_yuv444_triple_resample::<BITS>(
            rgb_stream,
            rgb_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| {
              yuv444p10_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| {
              yuv444p10_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Resolve the output set up front so the atomicity preflight below runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 sibling): reserve the only growable row scratch this
    // identity row can touch — the u8 RGB row buffer — BEFORE any output row
    // is written (the luma plane below, then the u16 RGB / RGBA fan-out), so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. The u16 RGB / RGBA outputs
    // need no preflight: they write straight into their caller buffers (the
    // rgb_u16 plane itself stages the rgba_u16 expand) and never grow a scratch;
    // this format exposes no luma_u16 output. `rgb_row_buf_or_scratch`'s
    // allocating (rgb=None) arm is reached exactly when a colour decode needs an
    // RGB row but no caller RGB buffer is borrowable — for this
    // convert-once-then-derive path that is `want_hsv && want_rgba && !want_rgb`
    // (HSV-only routes through a direct kernel that needs no RGB scratch). The
    // later decode reuses the already-sized buffer, so the default path is
    // byte-identical; only the failure-path ordering changes.
    if want_hsv && want_rgba && !want_rgb {
      rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see Yuv420p9 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p10_to_rgba_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
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
      yuv444p10_to_rgb_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
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
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p10_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv444p10_to_hsv_row_endian(
        row.y(),
        row.u(),
        row.v(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p10_to_rgba_row_endian(
        row.y(),
        row.u(),
        row.v(),
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

    yuv444p10_to_rgb_row_endian(
      row.y(),
      row.u(),
      row.v(),
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

// ---- Yuv440p12 impl -----------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuv440p12<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. 12-bit low-packed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant.
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

  /// Attaches a packed **8-bit** RGBA output buffer. Yuv440p12 reuses
  /// the `BITS = 12` 4:4:4 RGBA kernel; alpha = `0xFF`.
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

  /// Attaches a packed **`u16`** RGBA output buffer. 12-bit low-packed
  /// (`[0, 4095]`); alpha element is `4095`.
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

impl<R, const BE: bool> Yuv440p12Sink<BE> for MixedSinker<'_, Yuv440p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuv440p12<BE>, R> {
  type Input<'r> = Yuv440p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuv_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuv440p12Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 12;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y12,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull12,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull12,
        idx,
        w,
        row.v().len(),
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
      native_planar_u16,
      frozen_native_route,
      ..
    } = self;

    // Non-identity plan: feed the shared high-bit 4:4:4 triple-resample
    // tail (u8 color, independent native-u16 color, native Y). 4:4:0 is
    // full-width chroma (no horizontal upsampling, same per-row contract
    // as 4:4:4) — its vertical chroma sharing is already resolved by the
    // walker, which hands this luma row the (vertically-shared) full-width
    // `u` / `v`, so the converted RGB is full-res and the 4:4:4 tail binds
    // via the shared `yuv444pN_to_rgb_*` kernels. Yuv440p exposes no
    // `luma_u16` output, so it is `&mut None` and only `luma` (binned
    // native Y `>> (BITS - 8)`) is emitted. The span kind picks the engine
    // (area bin or signed-coefficient filter twin) — see the Yuv440p10 impl
    // for the full rationale; the filter tail clamps every sub-16-bit colour
    // sample AND the native Y to `(1 << BITS) - 1`.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let (y, u, v) = (row.y(), row.u(), row.v());
      if plan.kind().is_filter() {
        return packed_yuv444_triple_filter_resample::<BITS>(
          rgb_filter_stream,
          rgb_filter_stream_u16,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          &mut None,
          hsv,
          rgb_scratch,
          rgb_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          matrix,
          full_range,
          |scratch| {
            yuv444p12_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| {
            yuv444p12_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
          },
          |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
        );
      }
      // Native / row-stage route split — see the high-bit 4:2:0 Yuv420p impl
      // for the CHECK-before / SET-after `frozen_native_route` contract.
      let need_output = luma.is_some()
        || rgb.is_some()
        || rgba.is_some()
        || hsv.is_some()
        || rgb_u16.is_some()
        || rgba_u16.is_some();
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV440P_HIGH_BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:0: chroma `w x h/2` — full width, half height; a chroma row per
          // TWO Y rows (`chroma_vsub = 2`, like 4:2:0 vertically; `chroma_w = w`),
          // chroma plan full-width horizontal + luma-domain `area_halved`
          // vertical.
          yuv_planar16_process_native::<BITS, BE>(
            plan,
            native_planar_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            // The high-bit planar 4:4:0 family exposes no `luma_u16` output.
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            y,
            u,
            v,
            matrix,
            full_range,
            idx,
            w,
            h,
            2,
            w,
            || ResamplePlan::area_chroma_440(w, h, plan.out_w(), plan.out_h(), 0.0, 0.0),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          packed_yuv444_triple_resample::<BITS>(
            rgb_stream,
            rgb_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            &mut None,
            hsv,
            rgb_scratch,
            rgb_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            matrix,
            full_range,
            |scratch| {
              yuv444p12_to_rgb_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| {
              yuv444p12_to_rgb_u16_row_endian(y, u, v, scratch, w, matrix, full_range, use_simd, BE)
            },
            |scratch| deinterleave_y_high_bit_masked::<BITS, BE>(y, scratch, w),
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Resolve the output set up front so the atomicity preflight below runs
    // before any output row is written.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Atomicity preflight (#308, cf. the crate's #180 resample fix and the
    // high-bit 4:2:0 sibling): reserve the only growable row scratch this
    // identity row can touch — the u8 RGB row buffer — BEFORE any output row
    // is written (the luma plane below, then the u16 RGB / RGBA fan-out), so an
    // allocator refusal returns a typed `AllocationFailed` leaving the output
    // frame untouched rather than partially mutated. The u16 RGB / RGBA outputs
    // need no preflight: they write straight into their caller buffers (the
    // rgb_u16 plane itself stages the rgba_u16 expand) and never grow a scratch;
    // this format exposes no luma_u16 output. `rgb_row_buf_or_scratch`'s
    // allocating (rgb=None) arm is reached exactly when a colour decode needs an
    // RGB row but no caller RGB buffer is borrowable — for this
    // convert-once-then-derive path that is `want_hsv && want_rgba && !want_rgb`
    // (HSV-only routes through a direct kernel that needs no RGB scratch). The
    // later decode reuses the already-sized buffer, so the default path is
    // byte-identical; only the failure-path ordering changes.
    if want_hsv && want_rgba && !want_rgb {
      rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
    }

    if let Some(luma) = luma.as_deref_mut() {
      let dst = &mut luma[one_plane_start..one_plane_end];
      for (d, &s) in dst.iter_mut().zip(row.y().iter()) {
        // Normalize BE-encoded wire bytes to host-native before the
        // luma downshift — see Yuv420p9 luma path for rationale.
        let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
        *d = (logical >> (BITS - 8)) as u8;
      }
    }

    // ===== u16 RGB / RGBA path (Strategy A) =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p12_to_rgba_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
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
      yuv444p12_to_rgb_u16_row_endian(
        row.y(),
        row.u(),
        row.v(),
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
    // HSV-without-RGB-or-RGBA goes through the direct `yuv444p12_to_hsv_row_endian`
    // kernel (no source-width RGB scratch — the SIMD path stages a fixed
    // 8-bit RGB chunk internally). RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel`.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv444p12_to_hsv_row_endian(
        row.y(),
        row.u(),
        row.v(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv444p12_to_rgba_row_endian(
        row.y(),
        row.u(),
        row.v(),
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

    yuv444p12_to_rgb_row_endian(
      row.y(),
      row.u(),
      row.v(),
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
