//! Sinker impls for 32-bit packed-RGB **source** formats (Rgb96 / Rgba128).
//!
//! Sources:
//! - [`Rgb96`]   — `R, G, B` u32 per pixel (`AV_PIX_FMT_RGB96LE`).
//! - [`Rgba128`] — `R, G, B, A` u32 per pixel (`AV_PIX_FMT_RGBA128LE`).
//!
//! All 7 output paths per format:
//! - `with_rgb`      — packed 32-bit → packed u8 RGB (narrow `>> 24` per channel).
//! - `with_rgba`     — same narrow; for Rgb96 alpha = `0xFF` (no source α);
//!   for Rgba128 source α is passed through (`>> 24`).
//! - `with_rgb_u16`  — narrow `>> 16` (3 elements per pixel, R/G/B order).
//! - `with_rgba_u16` — narrow `>> 16`; for Rgb96 alpha = `0xFFFF`; for Rgba128
//!   source α is narrowed `>> 16`.
//! - `with_luma`     — Y' derived from narrowed u8 RGB via `rgb_to_luma_row`.
//! - `with_luma_u16` — Y' derived from narrowed u8 RGB, zero-extended to u16.
//! - `with_hsv`      — HSV derived from narrowed u8 RGB via `rgb_to_hsv_row`.
//!
//! ## Resampling
//!
//! Non-identity plans stage a source-width **native u16** RGB row (the
//! `>> 16` narrow) and reuse the shared `packed_rgb_u16_*` engine at 16-bit
//! depth — identical to the 16-bit packed-RGB family. The `>> 16` staging plus
//! the engine's `>> 8` u8 derivation reproduce the direct path's `>> 24` u8
//! narrow.
//!
//! ### Precision (`full_range = false`) — issue #289
//!
//! Because the wire u32 is narrowed `>> 16` **before** binning, a downscaled
//! output is the area/filter mean of the *narrowed* samples — within 1 LSB of
//! an exact u32-domain mean (most visible for `full_range = false`). Every
//! **direct** (identity-plan) conversion is exact and byte-identical, and
//! full-range downscale is unaffected by the limited-range scaling. This u32
//! family deliberately ACCEPTS the ≤1-LSB limited-range resample rather than
//! building new u32 resample infrastructure; the exact 0-ULP fix (a `u128`
//! area tier + a `u32` filter tier) is tracked in issue #289. The current
//! narrow-first behaviour is pinned by the `full_range = false` resample tests
//! (area + filter, LE + BE) in `tests/resample_packed_rgb_32bit.rs`.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  packed_rgb_u16_filter_stream, packed_rgb_u16_resample_emit, packed_rgb_u16_resample_preflight,
  packed_rgb_u16_resample_stream, packed_rgba_u16_filter_resample, packed_rgba_u16_resample,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice, source_rgb_u16_scratch,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row, rgb_to_luma_row,
    rgb_to_luma_u16_row, rgb96_to_rgb_row_endian, rgb96_to_rgb_u16_row_endian,
    rgb96_to_rgba_row_endian, rgb96_to_rgba_u16_row_endian, rgba128_to_rgb_row_endian,
    rgba128_to_rgb_u16_row_endian, rgba128_to_rgba_row_endian, rgba128_to_rgba_u16_row_endian,
  },
  source::{Rgb96, Rgb96Row, Rgb96Sink, Rgba128, Rgba128Row, Rgba128Sink},
};

// ---- Rgb96 -----------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Rgb96<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Each 32-bit channel is
  /// narrowed `>> 24` and alpha is forced to `0xFF` (no source alpha in Rgb96).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height x 4`,
  /// or `Err(GeometryOverflow)` on 32-bit targets when the product overflows.
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

  /// Attaches a native **`u16`** RGB output buffer. Length in `u16` **elements**
  /// (`width x height x 3`). Each 32-bit channel is narrowed `>> 16`.
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. Length in `u16` **elements**
  /// (`width x height x 4`). R/G/B narrowed `>> 16`; alpha forced to `0xFFFF`.
  ///
  /// Returns `Err(InsufficientRgbaU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer. Length in `u16` **elements**
  /// (`width x height`). Y' is computed at 8-bit precision and zero-extended.
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if the buffer is too short.
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

impl<R, const BE: bool> Rgb96Sink<BE> for MixedSinker<'_, Rgb96<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Rgb96<BE>, R> {
  type Input<'r> = Rgb96Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Rgb96Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected =
      w.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 3,
        )))?;
    if row.rgb96().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Rgb96Packed,
        idx,
        packed_expected,
        row.rgb96().len(),
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
      rgb_stream_u16,
      rgb_filter_stream_u16,
      resample_outputs,
      plan,
      ..
    } = self;

    // Non-identity plan: convert the wire row to source-width host u16 RGB
    // (the `>> 16` narrow), bin it at native 16-bit depth, then derive every
    // attached output from each finalized output row (native-depth u16 outputs
    // copy the binned row; u8 / luma_u16 outputs narrow it `>> 8`, reproducing
    // the direct path's `>> 24`). The span kind picks the engine — integer
    // area or signed-coefficient filter (both bin the same staged native-u16
    // row). Narrowing `>> 16` before binning makes a `full_range = false`
    // downscale within 1 LSB of an exact u32-domain mean (direct conversion is
    // exact); the u32 family accepts this — 0-ULP fix tracked in issue #289.
    if let Some(plan) = plan.as_ref() {
      let stream_next_y = match plan.kind() {
        crate::resample::SpanKind::Area => rgb_stream_u16.as_ref().map_or(0, |s| s.next_y()),
        crate::resample::SpanKind::Filter => {
          rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y())
        }
      };
      if !packed_rgb_u16_resample_preflight(
        resample_outputs,
        rgb,
        rgba,
        luma,
        rgb_u16,
        rgba_u16,
        luma_u16,
        hsv,
        stream_next_y,
        idx,
      )? {
        return Ok(());
      }
      return match plan.kind() {
        crate::resample::SpanKind::Area => {
          let stream = packed_rgb_u16_resample_stream(rgb_stream_u16, plan, idx)?;
          let src_u16 = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
          rgb96_to_rgb_u16_row_endian::<BE>(row.rgb96(), src_u16, w, use_simd);
          packed_rgb_u16_resample_emit::<16, false>(
            stream,
            plan,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            src_u16,
            rgb_scratch,
            row.matrix(),
            row.full_range(),
            idx,
            use_simd,
          )
        }
        crate::resample::SpanKind::Filter => {
          let stream = packed_rgb_u16_filter_stream(rgb_filter_stream_u16, plan, idx)?;
          let src_u16 = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
          rgb96_to_rgb_u16_row_endian::<BE>(row.rgb96(), src_u16, w, use_simd);
          packed_rgb_u16_resample_emit::<16, false>(
            stream,
            plan,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            src_u16,
            rgb_scratch,
            row.matrix(),
            row.full_range(),
            idx,
            use_simd,
          )
        }
      };
    }

    let ps = idx * w;
    let pe = ps + w;
    let in96 = row.rgb96();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // u8 RGB staging — required when any of: with_rgb, with_luma,
    // with_luma_u16, or with_hsv is attached.
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      rgb96_to_rgb_row_endian::<BE>(in96, rgb_row, w, use_simd);

      if let Some(luma_buf) = luma.as_deref_mut() {
        rgb_to_luma_row(
          rgb_row,
          &mut luma_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
        rgb_to_luma_u16_row(
          rgb_row,
          &mut luma_u16_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(hsv_bufs) = hsv.as_mut() {
        let (h, s, v) = hsv_bufs.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut h[ps..pe],
          &mut s[ps..pe],
          &mut v[ps..pe],
          w,
          use_simd,
        );
      }
    }

    // u8 RGBA — single-pass kernel, alpha forced to 0xFF.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, ps, pe, w, h)?;
      rgb96_to_rgba_row_endian::<BE>(in96, rgba_row, w, use_simd);
    }

    // u16 RGB — narrow `>> 16`.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let end =
        pe.checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      rgb96_to_rgb_u16_row_endian::<BE>(in96, &mut buf[ps * 3..end], w, use_simd);
    }

    // u16 RGBA — narrow `>> 16`, alpha forced to 0xFFFF.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_u16_row = rgba_u16_plane_row_slice(buf, ps, pe, w, h)?;
      rgb96_to_rgba_u16_row_endian::<BE>(in96, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Rgba128 ---------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Rgba128<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Each 32-bit channel is
  /// narrowed `>> 24`; the **source alpha** at slot 3 is depth-converted and
  /// passed through (not forced to `0xFF`).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height x 4`.
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

  /// Attaches a native **`u16`** RGB output buffer. Length in `u16` **elements**
  /// (`width x height x 3`). Alpha slot dropped; R/G/B narrowed `>> 16`.
  ///
  /// Returns `Err(InsufficientRgbU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. Length in `u16` **elements**
  /// (`width x height x 4`). Source α at slot 3 is narrowed `>> 16`.
  ///
  /// Returns `Err(InsufficientRgbaU16Buffer)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer (`width x height` elements).
  /// Y' is derived from narrowed u8 RGB and zero-extended to u16.
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if the buffer is too short.
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

impl<R, const BE: bool> Rgba128Sink<BE> for MixedSinker<'_, Rgba128<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Rgba128<BE>, R> {
  type Input<'r> = Rgba128Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Rgba128Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.rgba128().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Rgba128Packed,
        idx,
        packed_expected,
        row.rgba128().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan. Route the alpha-aware 4-channel u16 tail when
    // resampled alpha would be dropped (rgba / rgba_u16 attached) or the color
    // must be alpha-weighted (premultiplied); otherwise the rgb-only straight
    // outputs keep the 3-channel u16 RGB path. Rgba128 stages canonical
    // host-native RGBA via `rgba128_to_rgba_u16_row_endian` (all four channels
    // narrowed `>> 16`, α pass-through) and drop-alpha RGB via
    // `rgba128_to_rgb_u16_row_endian`. Narrowing `>> 16` before binning makes a
    // `full_range = false` downscale within 1 LSB of an exact u32-domain mean
    // (direct conversion is exact); accepted — 0-ULP fix tracked in issue #289.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let in128 = row.rgba128();
      let matrix = row.matrix();
      let full_range = row.full_range();
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
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        rgb_stream_u16,
        rgba_stream_u16,
        rgb_filter_stream_u16,
        rgba_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        plan,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      // The alpha mode is snapshotted at begin_frame; reject a mid-frame change
      // here, before route selection, so a flip can neither reroute nor mix
      // modes. The span kind then picks the engine — see the `Rgba64` impl for
      // the routing rationale.
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      match plan.kind() {
        crate::resample::SpanKind::Area => {
          if rgba.is_some() || rgba_u16.is_some() || alpha_mode.is_premultiplied() {
            return packed_rgba_u16_resample::<16, false, false>(
              rgba_stream_u16,
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              matrix,
              full_range,
              |dst| rgba128_to_rgba_u16_row_endian::<BE>(in128, dst, w, use_simd),
              |_| {},
            );
          }
          if !packed_rgb_u16_resample_preflight(
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            rgb_stream_u16.as_ref().map_or(0, |s| s.next_y()),
            idx,
          )? {
            return Ok(());
          }
          let stream = packed_rgb_u16_resample_stream(rgb_stream_u16, plan, idx)?;
          let src_u16 = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
          rgba128_to_rgb_u16_row_endian::<BE>(in128, src_u16, w, use_simd);
          return packed_rgb_u16_resample_emit::<16, false>(
            stream,
            plan,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            src_u16,
            rgb_scratch,
            matrix,
            full_range,
            idx,
            use_simd,
          );
        }
        crate::resample::SpanKind::Filter => {
          // Premultiplied alpha has no filter analogue; surface the typed
          // `UnsupportedFilter` via the area tail's reject. Straight alpha:
          // filter all four native u16 channels independently when alpha
          // survives, else the 3-channel u16 filter for rgb-only outputs.
          if alpha_mode.is_premultiplied() {
            return packed_rgba_u16_resample::<16, false, false>(
              rgba_stream_u16,
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              matrix,
              full_range,
              |dst| rgba128_to_rgba_u16_row_endian::<BE>(in128, dst, w, use_simd),
              |_| {},
            );
          }
          if rgba.is_some() || rgba_u16.is_some() {
            return packed_rgba_u16_filter_resample::<16, false>(
              rgba_filter_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              |dst| rgba128_to_rgba_u16_row_endian::<BE>(in128, dst, w, use_simd),
            );
          }
          if !packed_rgb_u16_resample_preflight(
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y()),
            idx,
          )? {
            return Ok(());
          }
          let stream = packed_rgb_u16_filter_stream(rgb_filter_stream_u16, plan, idx)?;
          let src_u16 = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
          rgba128_to_rgb_u16_row_endian::<BE>(in128, src_u16, w, use_simd);
          return packed_rgb_u16_resample_emit::<16, false>(
            stream,
            plan,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            src_u16,
            rgb_scratch,
            matrix,
            full_range,
            idx,
            use_simd,
          );
        }
      }
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
      ..
    } = self;
    let ps = idx * w;
    let pe = ps + w;
    let in128 = row.rgba128();

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // ===== u8 path =====

    // Standalone RGBA u8 fast path — only rgba attached. Source α passes
    // through via the kernel.
    if want_rgba && !need_u8_rgb && !want_rgb_u16 && !want_rgba_u16 {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      rgba128_to_rgba_row_endian::<BE>(in128, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone RGBA u16 fast path — only rgba_u16 attached, no u8 work.
    if want_rgba_u16 && !want_rgb_u16 && !need_u8_rgb && !want_rgba {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      rgba128_to_rgba_u16_row_endian::<BE>(in128, rgba_u16_row, w, use_simd);
      return Ok(());
    }

    // u8 RGB staging — drives with_rgb / with_luma / with_luma_u16 / with_hsv,
    // and Strategy A+ RGBA fan-out.
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      rgba128_to_rgb_row_endian::<BE>(in128, rgb_row, w, use_simd);

      if let Some(luma_buf) = luma.as_deref_mut() {
        rgb_to_luma_row(
          rgb_row,
          &mut luma_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
        rgb_to_luma_u16_row(
          rgb_row,
          &mut luma_u16_buf[ps..pe],
          w,
          row.matrix(),
          row.full_range(),
          use_simd,
        );
      }

      if let Some(hsv_bufs) = hsv.as_mut() {
        let (h, s, v) = hsv_bufs.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut h[ps..pe],
          &mut s[ps..pe],
          &mut v[ps..pe],
          w,
          use_simd,
        );
      }

      // Strategy A+ u8: RGBA also attached — derive from the just-computed RGB
      // row (writes α=0xFF), then overwrite α slot from packed source (slot 3,
      // depth-conv >> 24). Output is byte-identical to rgba128_to_rgba_row.
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        crate::row::scalar::alpha_extract::copy_alpha_packed_u32x4_to_u8_at_3::<BE>(
          in128, rgba_row, w,
        );
      }
    }

    // Standalone RGBA u8 path — want_rgba without need_u8_rgb (combined with
    // u16 work only). Run rgba128_to_rgba_row directly; source α depth-conv.
    if want_rgba && !need_u8_rgb {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      rgba128_to_rgba_row_endian::<BE>(in128, rgba_row, w, use_simd);
    }

    // ===== u16 path =====

    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let end =
        pe.checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[ps * 3..end];
      rgba128_to_rgb_u16_row_endian::<BE>(in128, rgb_u16_row, w, use_simd);

      // Strategy A+ u16: RGBA u16 also attached — derive from the just-computed
      // u16 RGB row (writes α=0xFFFF), then overwrite α slot from packed source
      // (slot 3, narrowed >> 16).
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        crate::row::scalar::alpha_extract::copy_alpha_packed_u32x4_to_u16_at_3::<BE>(
          in128,
          rgba_u16_row,
          w,
        );
      }
    }

    // Standalone RGBA u16 path — want_rgba_u16 without want_rgb_u16 (combined
    // with u8 work). Run rgba128_to_rgba_u16_row directly.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      rgba128_to_rgba_u16_row_endian::<BE>(in128, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}
