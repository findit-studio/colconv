//! Sinker impls for source-side YUVA 4:2:2 formats — Yuva422p (8-bit),
//! Yuva422p9, Yuva422p10, Yuva422p12, Yuva422p16. Wiring-only: per-row chroma
//! layout is identical to YUVA 4:2:0 (half-width U / V), so this file
//! delegates row-level work to the existing
//! `yuva420p*_to_rgba*_with_alpha_src_row` dispatchers from Ship 8b‑2.
//! The 4:2:0 vs 4:2:2 difference is purely in the vertical walker
//! (chroma row index `r / 2` vs `r`) and is handled in the walker
//! / sinker layer.
//!
//! Each format gets:
//! - **RGBA output paths (u8 + native-depth u16, where applicable)** —
//!   alpha is sourced from the source alpha plane via the
//!   `yuva420p*_to_rgba*_with_alpha_src_row` dispatchers. The 8-bit
//!   `Yuva422p` has no native `u16` RGBA output (mirrors `Yuva420p`).
//! - **RGB / RGB-u16 / luma / HSV alpha-drop paths** — these reuse the
//!   existing 4:2:0 row dispatchers
//!   (`yuv420p*_to_rgb*` / `yuv420p_to_rgb_row`) directly, since the
//!   per-row chroma layout is identical between 4:2:0 and 4:2:2 (the
//!   YUV 4:2:2 sinkers do the same). The alpha plane is ignored;
//!   output bytes / elements match what the corresponding
//!   `Yuv422p<BITS>` source would produce given the same Y/U/V data.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match, check_frozen_alpha_mode,
  deinterleave_y_high_bit, packed_yuva444_filter_resample, packed_yuva444_resample,
  reset_high_bit_yuva_streams, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
use crate::{PixelSink, row::*, source::*};

// ---- Yuva422p impl (8-bit) ---------------------------------------------

impl<'a, R> MixedSinker<'a, Yuva422p, R> {
  /// Attaches a packed **8‑bit** RGBA output buffer. The 8‑bit YUVA
  /// 4:2:2 source is converted to 8‑bit RGBA via the same Q15 i32
  /// 8‑bit kernel that backs [`MixedSinker<Yuv422p>::with_rgba`] (the
  /// 4:2:0 row kernel — `yuva420p_to_rgba_row` — applies verbatim
  /// since per-row chroma layout is identical between 4:2:0 and
  /// 4:2:2); the per-pixel alpha byte is **sourced from the alpha
  /// plane** — not constant `0xFF`.
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

  /// Attaches a **`u16`** luma output buffer. The 8-bit Y plane is
  /// zero-extended to u16 (`out[x] = Y_byte as u16`). Length in u16
  /// **elements** (`width x height`).
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if `buf.len() < width x height`,
  /// or `Err(GeometryOverflow)` on 32-bit targets.
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

impl<R> Yuva422pSink for MixedSinker<'_, Yuva422p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuva422p, R> {
  type Input<'r> = Yuva422pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 4-channel u8 RGBA colour stream and the
    // native-Y luma stream (both lazily created in `process`) and re-arm the
    // alpha-mode snapshot, mirroring the alpha-aware packed-YUVA (`Vuya`)
    // sink. The luma stream kind depends on the plan: the area path bins the
    // native Y at u16 (`luma_stream_u16`), the filter path resamples the
    // contiguous native Y at u8 (`luma_filter_stream`, parity with the
    // no-alpha `Yuv422p`). The 8-bit `Yuva422p` exposes no u16 colour
    // outputs, so its u16 RGBA streams are never created.
    if let Some(stream) = self.rgba_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Yuva422pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf,
        idx,
        w / 2,
        row.v_half().len(),
      )));
    }
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::AFull,
        idx,
        w,
        row.a().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan: `Yuva422p` is 8-bit planar 4:2:2 YUV **with a
    // real full-resolution source alpha plane**. Route through the
    // packed-YUVA tail at `SRC_BITS = 8` exactly like `Yuva420p`: the u8
    // colour stream resamples the converted u8 RGBA row, the native-Y luma
    // stream resamples the (zero-extended) Y plane directly. The per-row
    // chroma layout (half-width U / V) is identical to 4:2:0, so the colour
    // closure uses the same `yuva420p_to_rgba_row` kernel the direct
    // `Yuva422p` path already reuses (the 4:2:0-vs-4:2:2 difference is the
    // vertical chroma-row index, owned by the walker). The 8-bit
    // `Yuva422p` exposes no u16 colour outputs, so the tail's u16 colour
    // resampling is never active and its `convert_rgba_u16` closure is never
    // invoked.
    //
    // The span kind picks the engine (mirrors `Yuva420p`): `Area` bins (the
    // alpha-aware tail — premultiplied colour binned premultiplied then
    // un-premultiplied); `Filter` runs the signed-coefficient filter on the
    // same converted RGBA (straight alpha only). A premultiplied `Filter`
    // plan is routed to the area tail so it surfaces the typed
    // `UnsupportedFilter` rather than straight-filtering premultiplied colour.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let matrix = row.matrix();
      let full_range = row.full_range();
      let y = row.y();
      let u_half = row.u_half();
      let v_half = row.v_half();
      let a = row.a();
      let Self {
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        plan,
        rgba_stream,
        rgba_stream_u16,
        luma_stream_u16,
        rgba_filter_stream,
        rgba_filter_stream_u16,
        luma_filter_stream,
        luma_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      return match plan.kind() {
        crate::resample::SpanKind::Area => packed_yuva444_resample::<8>(
          rgba_stream,
          rgba_stream_u16,
          luma_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          rgba_scratch_u16,
          rgba_color_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          alpha_mode,
          |dst| yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd),
          // `Yuva422p` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          |dst| {
            for (d, &s) in dst.iter_mut().zip(y) {
              *d = s as u16;
            }
          },
        ),
        crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
          // Premultiplied + filter has no analogue: route to the area tail
          // with the filter plan so it returns the typed `UnsupportedFilter`.
          packed_yuva444_resample::<8>(
            rgba_stream,
            rgba_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            rgba_scratch,
            rgb_scratch,
            rgba_scratch_u16,
            rgba_color_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            alpha_mode,
            |dst| yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd),
            |_dst: &mut [u16]| {},
            |dst| {
              for (d, &s) in dst.iter_mut().zip(y) {
                *d = s as u16;
              }
            },
          )
        }
        crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<8, true>(
          rgba_filter_stream,
          rgba_filter_stream_u16,
          luma_filter_stream,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          rgba_scratch_u16,
          rgba_color_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          // 8-bit native-Y luma rides the u8 stream (parity with `Yuv422p`):
          // the contiguous Y plane is fed directly.
          y,
          |dst| yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd),
          // `Yuva422p` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          // u8-luma path: the u16 luma stream is detached, so this is never
          // called.
          |_dst: &mut [u16]| {},
        ),
      };
    }

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // Acquire the u8 RGB row buffer up front — before any caller-output
    // write below — so an allocator refusal in the HSV-only scratch path
    // returns a recoverable error rather than leaving a partially-written
    // caller buffer (luma / luma_u16). `rgb_row_buf_or_scratch` is the
    // only fallible allocation on this direct path. The RGB kernel
    // reuses the 4:2:0 row dispatcher (yuv_420_to_rgb_row); per-row
    // chroma layout is identical to 4:2:2.
    let rgb_row = if need_rgb_kernel {
      Some(rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?)
    } else {
      None
    };

    // Luma — copy the Y plane verbatim (8-bit YUVA's Y is already u8).
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the Y plane (`out[x] = Y_byte as u16`).
    if let Some(luma_u16) = luma_u16.as_deref_mut() {
      for (d, &s) in luma_u16[one_plane_start..one_plane_end]
        .iter_mut()
        .zip(&row.y()[..w])
      {
        *d = s as u16;
      }
    }

    // `need_rgb_kernel` / `rgb_row` were computed and (when needed)
    // allocated at the top, before any caller-output write.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuva420p_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    // RGB kernel — alpha-drop reuses the 4:2:0 row dispatcher
    // (yuv_420_to_rgb_row); per-row chroma layout is identical to
    // 4:2:2.
    let Some(rgb_row) = rgb_row else {
      return Ok(());
    };
    yuv_420_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
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

    if want_rgba {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      // Strategy A+: RGB output already computed in rgb_row above.
      // Expand RGB → RGBA (fills α with 0xFF), then overwrite α slot
      // from the source alpha plane — avoids a second chroma kernel.
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      crate::row::alpha_extract::copy_alpha_plane_u8(row.a(), rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Yuva422p9 impl ---------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva422p9<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 1`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (9‑bit
  /// low‑packed, `[0, 511]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
  /// Output is identical to [`MixedSinker<Yuv422p9>::with_rgb_u16`].
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
  /// (`width x height`).
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

impl<R, const BE: bool> Yuva422p9Sink<BE> for MixedSinker<'_, Yuva422p9<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva422p9<BE>, R> {
  type Input<'r> = Yuva422p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva422p9Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva422p_high_bit_resample::<9, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y9,
        RowSlice::UHalf9,
        RowSlice::VHalf9,
        RowSlice::AFull9,
        yuva420p9_to_rgba_row_endian,
        yuva420p9_to_rgba_u16_row_endian,
      );
    }
    yuva422p_high_bit_process::<9, BE, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y9,
      RowSlice::UHalf9,
      RowSlice::VHalf9,
      RowSlice::AFull9,
      yuv420p9_to_rgb_row_endian,
      yuv420p9_to_rgb_u16_row_endian,
      yuva420p9_to_rgba_row_endian,
      yuva420p9_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva422p10 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva422p10<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 2`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (10‑bit
  /// low‑packed, `[0, 1023]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
  /// (`width x height`).
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

impl<R, const BE: bool> Yuva422p10Sink<BE> for MixedSinker<'_, Yuva422p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva422p10<BE>, R> {
  type Input<'r> = Yuva422p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva422p10Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva422p_high_bit_resample::<10, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y10,
        RowSlice::UHalf10,
        RowSlice::VHalf10,
        RowSlice::AFull10,
        yuva420p10_to_rgba_row_endian,
        yuva420p10_to_rgba_u16_row_endian,
      );
    }
    yuva422p_high_bit_process::<10, BE, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y10,
      RowSlice::UHalf10,
      RowSlice::VHalf10,
      RowSlice::AFull10,
      yuv420p10_to_rgb_row_endian,
      yuv420p10_to_rgb_u16_row_endian,
      yuva420p10_to_rgba_row_endian,
      yuva420p10_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva422p12 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva422p12<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 4`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (12‑bit
  /// low‑packed, `[0, 4095]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
  /// (`width x height`).
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

impl<R, const BE: bool> Yuva422p12Sink<BE> for MixedSinker<'_, Yuva422p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva422p12<BE>, R> {
  type Input<'r> = Yuva422p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva422p12Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva422p_high_bit_resample::<12, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y12,
        RowSlice::UHalf12,
        RowSlice::VHalf12,
        RowSlice::AFull12,
        yuva420p12_to_rgba_row_endian,
        yuva420p12_to_rgba_u16_row_endian,
      );
    }
    yuva422p_high_bit_process::<12, BE, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y12,
      RowSlice::UHalf12,
      RowSlice::VHalf12,
      RowSlice::AFull12,
      yuv420p12_to_rgb_row_endian,
      yuv420p12_to_rgb_u16_row_endian,
      yuva420p12_to_rgba_row_endian,
      yuva420p12_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva422p16 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva422p16<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 8`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (16‑bit, full
  /// `u16` range `[0, 65535]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
  /// (`width x height`).
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

impl<R, const BE: bool> Yuva422p16Sink<BE> for MixedSinker<'_, Yuva422p16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva422p16<BE>, R> {
  type Input<'r> = Yuva422p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva422p16Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva422p_high_bit_resample::<16, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y16,
        RowSlice::UHalf16,
        RowSlice::VHalf16,
        RowSlice::AFull16,
        yuva420p16_to_rgba_row_endian,
        yuva420p16_to_rgba_u16_row_endian,
      );
    }
    yuva422p_high_bit_process::<16, BE, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y16,
      RowSlice::UHalf16,
      RowSlice::VHalf16,
      RowSlice::AFull16,
      yuv420p16_to_rgb_row_endian,
      yuv420p16_to_rgb_u16_row_endian,
      yuva420p16_to_rgba_row_endian,
      yuva420p16_to_rgba_u16_row_endian,
    )
  }
}

// ---- Shared high-bit YUVA 4:2:2 process body --------------------------
//
// Same shape as the 4:2:0 sibling helper (`yuva420p_high_bit_process`)
// — half-width chroma per row — except this one is called from the
// 4:2:2 walker (vertical chroma index `r` rather than `r / 2`). The
// 4:2:0 helper would work too if we threaded the row dispatcher
// through, but Rust's monomorphization gives the same code either way
// and this keeps the symbol table organized by source family.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva422p_high_bit_process<
  const BITS: u32,
  const BE: bool,
  F: crate::SourceFormat,
  R,
  RgbRowFn: Fn(&[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool, bool),
  RgbU16RowFn: Fn(&[u16], &[u16], &[u16], &mut [u16], usize, crate::ColorMatrix, bool, bool, bool),
  RgbaRowFn: Fn(&[u16], &[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool, bool),
>(
  sinker: &mut MixedSinker<'_, F, R>,
  idx: usize,
  y_row: &[u16],
  u_half_row: &[u16],
  v_half_row: &[u16],
  a_row: &[u16],
  matrix: crate::ColorMatrix,
  full_range: bool,
  y_slice: RowSlice,
  u_slice: RowSlice,
  v_slice: RowSlice,
  a_slice: RowSlice,
  rgb_dispatch: RgbRowFn,
  rgb_u16_dispatch: RgbU16RowFn,
  rgba_dispatch: RgbaRowFn,
  rgba_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let h = sinker.height;
  let use_simd = sinker.simd;

  if w & 1 != 0 {
    return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
  }
  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      y_slice,
      idx,
      w,
      y_row.len(),
    )));
  }
  if u_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      u_slice,
      idx,
      w / 2,
      u_half_row.len(),
    )));
  }
  if v_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      v_slice,
      idx,
      w / 2,
      v_half_row.len(),
    )));
  }
  if a_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      a_slice,
      idx,
      w,
      a_row.len(),
    )));
  }
  if idx >= sinker.height {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, sinker.height),
    ));
  }

  let MixedSinker {
    rgb,
    rgb_u16,
    rgba,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    ..
  } = sinker;
  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();
  let need_rgb_kernel = want_rgb || want_hsv;

  // Acquire the u8 RGB row buffer up front — before any caller-output
  // write below — so an allocator refusal in the HSV-only scratch path
  // returns a recoverable error rather than a partially-written caller
  // buffer. See `yuva420p_high_bit_process` for the full rationale.
  let rgb_row = if need_rgb_kernel {
    Some(rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?)
  } else {
    None
  };

  // ---- luma (native Y; luma narrows `>> (BITS - 8)`, luma_u16 is the
  // host-native logical Y) — see `yuva420p_high_bit_process`. -----------
  if luma.is_some() || luma_u16.is_some() {
    let mut luma_row = luma
      .as_deref_mut()
      .map(|b| &mut b[one_plane_start..one_plane_end]);
    let mut luma_u16_row = luma_u16
      .as_deref_mut()
      .map(|b| &mut b[one_plane_start..one_plane_end]);
    for (i, &s) in y_row.iter().enumerate().take(w) {
      let logical = if BE { u16::from_be(s) } else { u16::from_le(s) };
      if let Some(row) = luma_row.as_deref_mut() {
        row[i] = (logical >> (BITS - 8)) as u8;
      }
      if let Some(row) = luma_u16_row.as_deref_mut() {
        row[i] = logical;
      }
    }
  }

  // ---- u16 RGB / RGBA path --------------------------------------
  // rgb_u16 (alpha-drop) reuses the non-alpha dispatcher.
  // rgba_u16 — when rgb_u16 is also present (combo case), Strategy A+:
  // expand the already-computed rgb_u16_row → rgba_u16_row then
  // overwrite α slot from the source plane (avoids a second chroma
  // kernel). When rgba_u16 is alone, delegate to the alpha-source-aware
  // dispatcher directly (standalone path — already optimal).
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba_u16 = rgba_u16.is_some();
  if want_rgb_u16 {
    let buf = rgb_u16.as_deref_mut().unwrap();
    let rgb_plane_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        w, h, 3,
      )))?;
    let rgb_plane_start = one_plane_start * 3;
    let rgb_u16_row = &mut buf[rgb_plane_start..rgb_plane_end];
    rgb_u16_dispatch(
      y_row,
      u_half_row,
      v_half_row,
      rgb_u16_row,
      w,
      matrix,
      full_range,
      use_simd,
      BE,
    );
    if want_rgba_u16 {
      // Combo: expand rgb_u16_row → rgba_u16_row, then overwrite α slot.
      let rgba_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      // `BE` is plumbed from the row type so the alpha-plane copy honors
      // the same wire endianness as the caller's high-bit Yuva*p input.
      crate::row::alpha_extract::copy_alpha_plane_u16::<BITS, BE>(a_row, rgba_u16_row, w, use_simd);
    }
  } else if want_rgba_u16 {
    // Standalone rgba_u16: delegate to the alpha-source-aware dispatcher.
    let rgba_buf = rgba_u16.as_deref_mut().unwrap();
    let rgba_u16_row = rgba_u16_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_u16_dispatch(
      y_row,
      u_half_row,
      v_half_row,
      a_row,
      rgba_u16_row,
      w,
      matrix,
      full_range,
      use_simd,
      BE,
    );
  }

  // ---- u8 RGB / RGBA / HSV path ----------------------------------
  // `need_rgb_kernel` / `rgb_row` were computed and (when needed)
  // allocated at the top, before any caller-output write.
  if want_rgba && !need_rgb_kernel {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_dispatch(
      y_row, u_half_row, v_half_row, a_row, rgba_row, w, matrix, full_range, use_simd, BE,
    );
    return Ok(());
  }

  let Some(rgb_row) = rgb_row else {
    return Ok(());
  };
  rgb_dispatch(
    y_row, u_half_row, v_half_row, rgb_row, w, matrix, full_range, use_simd, BE,
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

  // Both rgb and rgba attached (combo case): Strategy A+ — expand the
  // already-computed rgb_row → rgba_row (fills α = 0xFF), then
  // overwrite α slot from the source alpha plane with depth-conv
  // `>> (BITS - 8)`. Avoids a second chroma kernel.
  if want_rgba {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    // BE = false: see the rgba_u16 branch above for rationale.
    crate::row::alpha_extract::copy_alpha_plane_u16_to_u8::<BITS, BE>(a_row, rgba_row, w, use_simd);
  }

  Ok(())
}

// ---- Shared high-bit YUVA 4:2:2 resample-routing body -----------------
//
// The non-identity plan branch for the 9 / 10 / 12 / 16-bit YUVA 4:2:2
// sinks — the 4:2:2 sibling of the 4:2:0 routing helper. Per-row chroma is
// half-width (identical to 4:2:0), so the same half-chroma `_to_rgba` /
// `_to_rgba_u16` kernels apply; the 4:2:0-vs-4:2:2 difference (vertical
// chroma index `r / 2` vs `r`) is owned by the walker. The `Area` arm routes
// through the shared packed-YUVA area tail with THREE independent binnings:
// u8 colour, the **independent** native u16 colour (never a narrowing of the
// u8 bin), and the **low-packed** native-Y luma (`deinterleave_y_high_bit`,
// raw host-native copy — planar YUVA Y is low-packed, so luma is
// `binned_Y >> (BITS - 8)`, NOT the semi-planar `>> (16 - BITS)` de-pack).
// The `Filter` arm routes the SAME converted RGBA / native-Y through the
// signed-coefficient filter tail (`NATIVE_LUMA_U8 = false`, the u16-luma
// branch — native Y is u16), straight alpha only.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva422p_high_bit_resample<const BITS: u32, const BE: bool>(
  sinker: &mut MixedSinker<'_, impl crate::SourceFormat, impl Sized>,
  idx: usize,
  y_row: &[u16],
  u_half_row: &[u16],
  v_half_row: &[u16],
  a_row: &[u16],
  matrix: crate::ColorMatrix,
  full_range: bool,
  y_slice: RowSlice,
  u_slice: RowSlice,
  v_slice: RowSlice,
  a_slice: RowSlice,
  rgba_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  rgba_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let use_simd = sinker.simd;

  if w & 1 != 0 {
    return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
  }
  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      y_slice,
      idx,
      w,
      y_row.len(),
    )));
  }
  if u_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      u_slice,
      idx,
      w / 2,
      u_half_row.len(),
    )));
  }
  if v_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      v_slice,
      idx,
      w / 2,
      v_half_row.len(),
    )));
  }
  if a_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      a_slice,
      idx,
      w,
      a_row.len(),
    )));
  }
  if idx >= sinker.height {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, sinker.height),
    ));
  }

  let alpha_mode = sinker.alpha_mode;
  let MixedSinker {
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgba_scratch,
    rgb_scratch,
    rgba_scratch_u16,
    rgba_color_scratch_u16,
    luma_scratch_u16,
    plan,
    rgba_stream,
    rgba_stream_u16,
    luma_stream_u16,
    rgba_filter_stream,
    rgba_filter_stream_u16,
    luma_filter_stream_u16,
    resample_outputs,
    frozen_alpha_mode,
    ..
  } = sinker;
  let plan = plan.as_ref().expect("plan.is_some() checked by the caller");
  check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
  // The span kind picks the engine (mirrors the 8-bit `Yuva422p` and packed
  // `Vuya`): `Area` bins (the alpha-aware tail — premultiplied colour binned
  // premultiplied then un-premultiplied) at native precision; `Filter` runs
  // the signed-coefficient filter on the same converted RGBA (straight alpha
  // only). The high-bit native Y is genuinely `u16`, so the filter route uses
  // `NATIVE_LUMA_U8 = false` — the same u16-luma branch `Vuya` / `Vuyx` use:
  // luma rides the `u16` filter stream over the de-interleaved native Y. The
  // filter tail clamps every sub-16-bit colour AND native-Y overshoot to the
  // source's native max `(1 << BITS) - 1` (a value no-op at `BITS = 16`),
  // matching the in-range area path. A premultiplied `Filter` plan has no
  // analogue (the engine cannot un-premultiply), so it is routed to the area
  // tail, which surfaces the typed `UnsupportedFilter`.
  match plan.kind() {
    crate::resample::SpanKind::Area => packed_yuva444_resample::<BITS>(
      rgba_stream,
      rgba_stream_u16,
      luma_stream_u16,
      resample_outputs,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgba_scratch,
      rgb_scratch,
      rgba_scratch_u16,
      rgba_color_scratch_u16,
      luma_scratch_u16,
      w,
      plan,
      idx,
      use_simd,
      alpha_mode,
      |dst| {
        rgba_dispatch(
          y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| {
        rgba_u16_dispatch(
          y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| deinterleave_y_high_bit::<BE>(y_row, dst, w),
    ),
    crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
      // Premultiplied + filter has no analogue: route to the area tail with
      // the filter plan so it returns the typed `UnsupportedFilter`.
      packed_yuva444_resample::<BITS>(
        rgba_stream,
        rgba_stream_u16,
        luma_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        alpha_mode,
        |dst| {
          rgba_dispatch(
            y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
          )
        },
        |dst| {
          rgba_u16_dispatch(
            y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
          )
        },
        |dst| deinterleave_y_high_bit::<BE>(y_row, dst, w),
      )
    }
    crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<BITS, false>(
      rgba_filter_stream,
      rgba_filter_stream_u16,
      // High-bit planar YUVA never uses the u8 native-Y luma stream
      // (`NATIVE_LUMA_U8 = false`); pass an inert slot.
      &mut None,
      luma_filter_stream_u16,
      resample_outputs,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgba_scratch,
      rgb_scratch,
      rgba_scratch_u16,
      rgba_color_scratch_u16,
      luma_scratch_u16,
      w,
      plan,
      idx,
      use_simd,
      // Luma rides `deinterleave_y` + the u16 stream (native Y is u16), so the
      // u8-luma input is unused.
      &[],
      |dst| {
        rgba_dispatch(
          y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| {
        rgba_u16_dispatch(
          y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| deinterleave_y_high_bit::<BE>(y_row, dst, w),
    ),
  }
}
