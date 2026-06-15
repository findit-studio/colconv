//! Sinker impl for the packed VUYX source format — Ship 12c (Tier 5
//! 8-bit packed YUV 4:4:4 with padding α byte).
//!
//! VUYX (FFmpeg `AV_PIX_FMT_VUYX`) packs **four u8 bytes per pixel**
//! (`[V, U, Y, X]`). The X byte is **padding** — not real source alpha.
//! RGBA outputs always force α to `0xFF`; the padding byte is ignored.
//! The packed slice type is `&[u8]`, with `4 x width` byte elements per
//! row. There is no chroma subsampling — every pixel carries its own
//! independent V / U / Y triplet (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; padding discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; **α is forced to
//!   `0xFF`** (the X byte is padding, never real alpha).
//! - `with_luma` — extracts the Y byte at offset 2 of each pixel
//!   directly (no YUV→RGB pipeline).
//! - `with_luma_u16` — zero-extends the Y byte to u16
//!   (`out[x] = Y_byte as u16`).
//! - `with_hsv` — stages u8 RGB into the user's RGB buffer (if
//!   attached) or a scratch buffer, then runs `rgb_to_hsv_row`.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, packed_yuv444_triple_resample,
  rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, vuyx_to_luma_row, vuyx_to_luma_u16_row,
    vuyx_to_rgb_row, vuyx_to_rgba_row,
  },
  source::{Vuyx, VuyxRow, VuyxSink},
};

impl<'a, R> MixedSinker<'a, Vuyx, R> {
  /// Attaches a **`u16`** luma output buffer. Y bytes from the packed VUYX
  /// `[V, U, Y, X]` layout are zero-extended to u16
  /// (`out[x] = Y_byte as u16`). Length in u16 **elements**
  /// (`width x height`).
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

  /// Attaches a packed **8-bit** RGBA output buffer. When VUYX is the
  /// source, the per-pixel alpha byte is always forced to `0xFF` —
  /// the X (padding) byte in the source is never read as alpha.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ## Strategy note
  ///
  /// α=`0xFF` is guaranteed in **all** paths (standalone or combined
  /// with `with_rgb` / `with_hsv`). When combined with `with_rgb`,
  /// RGBA is derived via Strategy A fan-out (`expand_rgb_to_rgba_row`)
  /// instead of a second YUV→RGB kernel call — both produce α=`0xFF`,
  /// so the outputs are semantically identical (spec § 8.4).
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
}

impl<R> VuyxSink for MixedSinker<'_, Vuyx, R> {}

impl<R> PixelSink for MixedSinker<'_, Vuyx, R> {
  type Input<'r> = VuyxRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the row-stage streams (lazily created in
    // `process`) and drop the frozen output set. `Vuyx` exposes no u16
    // colour outputs, so its u16 colour stream is never created; resetting
    // it unconditionally is a harmless no-op.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: VuyxRow<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 8;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // VUYX row = `width x 4` bytes (one quadruple per pixel).
    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VuyxPacked,
        idx,
        packed_expected,
        row.packed().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan: `Vuyx` has NO alpha (the X byte is padding,
    // forced opaque), so it routes exactly like the no-alpha packed 4:4:4
    // YUV siblings (`V30X` / `V410` / `Xv36`) through the three-stream tail
    // — u8 colour bins the converted u8 RGB row, native Y bins the
    // de-interleaved Y plane (the colour outputs force α opaque, the
    // padding byte is never read). `Vuyx` exposes no u16 colour outputs, so
    // the u16 colour binning is never active (`rgb_u16` / `rgba_u16` stay
    // `None`); its `convert_rgb_u16` closure is therefore never invoked.
    if let Some(plan) = self.plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let packed = row.packed();
      let Self {
        rgb,
        rgba,
        rgb_u16,
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
        ..
      } = self;
      return packed_yuv444_triple_resample::<BITS>(
        rgb_stream,
        rgb_stream_u16,
        luma_stream_u16,
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
        luma_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        matrix,
        full_range,
        |scratch| vuyx_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
        // `Vuyx` has no u16 colour outputs, so `need_u16_color` is always
        // false in the tail and this closure is never called.
        |_scratch: &mut [u16]| {},
        |scratch| vuyx_to_luma_u16_row(packed, scratch, w, use_simd),
      );
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
    let packed = row.packed();

    // Luma u8 — extract Y byte (offset 2 in each VUYX quadruple) directly.
    // `vuyx_to_luma_row` is a re-export of `vuya_to_luma_row` — the
    // byte stream is identical (Y at offset 2 regardless of α semantics).
    if let Some(buf) = luma.as_deref_mut() {
      vuyx_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Luma u16 — extract Y bytes and zero-extend to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      vuyx_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // ===== u8 RGB / RGBA / HSV path (Strategy A for VUYX) =====
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // Standalone RGBA fast path — no RGB / HSV requested. Run the
    // dedicated RGBA kernel directly into the output buffer; avoids
    // both the scratch allocation and the RGB→RGBA expand pass.
    // α is forced to `0xFF` by `vuyx_to_rgba_row` (ALPHA_SRC = false).
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      vuyx_to_rgba_row(
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

    // RGB kernel — write into the user's RGB buffer (if attached) or the
    // internal scratch buffer. Required when with_rgb or with_hsv is set.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    // `vuyx_to_rgb_row` is a re-export of `vuya_to_rgb_row` — the padding
    // byte is irrelevant when there is no α channel in the output.
    vuyx_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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
    // row instead of running a second YUV→RGB kernel. For VUYX,
    // α=`0xFF` is semantically correct in both paths (padding byte
    // is never real alpha), so Strategy A applies (spec § 8.4).
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}
