//! Sinker impl for the Tier 5.25 packed YUV 4:1:1 source format —
//! UYYVYY411 (`AV_PIX_FMT_UYYVYY411`, DV legacy).
//!
//! Single packed plane carrying `width * 3 / 2` bytes per row (12 bpp)
//! with byte order `U, Y, Y, V, Y, Y` per 6-byte / 4-pixel block —
//! one (U, V) chroma pair shared by four luma samples. Width must be
//! a multiple of 4.
//!
//! Output channels mirror the Tier 3 packed YUV 4:2:2 sinker
//! ([`super::packed_yuv_8bit`]):
//!
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline (full
//!   `ColorMatrix` + range support inherited from the row); RGBA
//!   alpha is forced to `0xFF` (the source has no alpha channel).
//! - `with_luma` — extracts the Y bytes from the packed plane via
//!   the dedicated luma kernel.
//! - `with_luma_u16` — zero-extends Y bytes to u16.
//! - `with_hsv` — stages an internal RGB scratch (or the user's RGB
//!   buffer if attached) and runs the existing `rgb_to_hsv_row`
//!   kernel.
//!
//! When both RGB and RGBA outputs are requested, the RGBA plane is
//! derived from the just-computed RGB row via
//! [`expand_rgb_to_rgba_row`] (Strategy A — memory-bound copy + 0xFF
//! alpha pad) instead of running a second YUV→RGB kernel.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  packed_yuv_8bit::packed_yuv422_dual_filter_resample, planar_resample::packed_yuv_dual_resample,
  rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  resample::SpanKind,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, uyyvyy411_to_luma_row, uyyvyy411_to_luma_u16_row,
    uyyvyy411_to_rgb_row, uyyvyy411_to_rgba_row,
  },
  source::{Uyyvyy411, Uyyvyy411Row, Uyyvyy411Sink},
};

impl<'a, R> MixedSinker<'a, Uyyvyy411, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
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

  /// Attaches a **`u16`** luma output buffer. Y bytes are zero-extended
  /// to u16 (`out[x] = Y_byte as u16`). Length in u16 **elements**
  /// (`width x height`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_luma_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected_elems = self.frame_pixels()?;
    if buf.len() < expected_elems {
      return Err(MixedSinkerError::InsufficientLumaU16Buffer(
        InsufficientBuffer::new(expected_elems, buf.len()),
      ));
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

impl<R> Uyyvyy411Sink for MixedSinker<'_, Uyyvyy411, R> {}

impl<R> PixelSink for MixedSinker<'_, Uyyvyy411, R> {
  type Input<'r> = Uyyvyy411Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 3 != 0 {
      return Err(MixedSinkerError::WidthAlignment(
        WidthAlignment::multiple_of_four(self.width),
      ));
    }
    // New frame: restart the row-stage resample streams and re-freeze
    // the output set so a reused sink starts each frame clean.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Uyyvyy411Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 3 != 0 {
      return Err(MixedSinkerError::WidthAlignment(
        WidthAlignment::multiple_of_four(w),
      ));
    }

    // Row length: `width * 3 / 2` (12 bpp). `w` is a multiple of 4 by
    // the gate above, so `w * 3` is also a multiple of 4 and the
    // `/ 2` is exact. Check the `* 3` for 32-bit overflow.
    let packed_expected =
      w.checked_mul(3)
        .map(|n| n / 2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 3,
        )))?;
    if row.uyyvyy().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Uyyvyy411Packed,
        idx,
        packed_expected,
        row.uyyvyy().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      luma_scratch,
      plan,
      rgb_stream,
      luma_stream,
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: row-stage fused resample. De-interleave the Y
    // bytes out of the packed plane for luma (the YUV luma contract —
    // luma resamples Y, never RGB-derived luma); for colour, convert the
    // packed row to a source-width RGB row with the same fused
    // `uyyvyy411_to_rgb_row` kernel the identity path uses (chroma
    // de-interleave + 4:1:1 horizontal upsample in registers), then resample
    // it. The span kind picks the engine — area binning (RGB equals an
    // `Rgb24` area-resample of the identity-converted frame) or
    // signed-coefficient filter (RGB equals the `Rgb24` filter of those
    // converted pixels; luma stays native Y). The filter arm shares the 4:2:2
    // tail: 4:1:1 differs only in the convert closures, which have the same
    // shape.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let packed = row.uyyvyy();
      return match plan.kind() {
        SpanKind::Area => packed_yuv_dual_resample(
          luma_stream,
          rgb_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| {
            uyyvyy411_to_luma_row(packed, scratch, w, use_simd);
          },
          |scratch| {
            uyyvyy411_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd);
          },
        ),
        SpanKind::Filter => packed_yuv422_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          luma_scratch,
          rgb_scratch,
          w,
          plan,
          idx,
          use_simd,
          |scratch| {
            uyyvyy411_to_luma_row(packed, scratch, w, use_simd);
          },
          |scratch| {
            uyyvyy411_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd);
          },
        ),
      };
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let packed = row.uyyvyy();

    // Luma u8 — extract Y bytes from packed plane via dedicated kernel.
    if let Some(luma) = luma.as_deref_mut() {
      uyyvyy411_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    // Luma u16 — zero-extend Y bytes to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      uyyvyy411_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // Standalone RGBA fast path — no RGB / HSV requested. Run the
    // dedicated RGBA kernel directly into the output buffer; avoids
    // both the scratch allocation and the RGB→RGBA expand pass.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      uyyvyy411_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
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
    uyyvyy411_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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

    // Strategy A: when both RGB and RGBA are requested, derive RGBA
    // from the just-computed RGB row instead of running a second
    // YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
