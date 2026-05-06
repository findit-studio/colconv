//! Sinker impl for the Tier 9 packed-half-float-RGB **source** format
//! ([`Rgbf16`] — FFmpeg `AV_PIX_FMT_RGBF16`).
//!
//! Each pixel is `3 × half::f16` (linear `R, G, B`). Output paths:
//! - `with_rgb` — clamp `[0, 1]` × 255 → packed `R, G, B` u8
//!   (`rgbf16_to_rgb_row`).
//! - `with_rgba` — same conversion + constant `0xFF` alpha.
//! - `with_rgb_u16` — clamp `[0, 1]` × 65535 → packed `R, G, B` u16.
//!   **Full-range scaling** (see [`with_rgb_u16`](MixedSinker::with_rgb_u16)
//!   for the divergence note vs the integer-source family).
//! - `with_rgba_u16` — same + constant `0xFFFF` alpha.
//! - `with_rgb_f16` — **NEW** lossless half-float pass-through (HDR > 1.0
//!   and negatives preserved bit-exact).
//! - `with_rgb_f32` — lossless widening to `f32` (HDR preserved).
//! - `with_luma` / `with_luma_u16` — staged through a u8 RGB scratch
//!   row (or the user's `with_rgb` buffer if attached) and the
//!   existing `rgb_to_luma_row` / `rgb_to_luma_u16_row` kernels.
//! - `with_hsv` — same staging, then `rgb_to_hsv_row`.
//!
//! HDR values > 1.0 saturate to the integer output range; the float
//! output preserves them losslessly.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    rgb_to_hsv_row, rgb_to_luma_row, rgb_to_luma_u16_row, rgbf16_to_rgb_f16_row,
    rgbf16_to_rgb_f32_row, rgbf16_to_rgb_row, rgbf16_to_rgb_u16_row, rgbf16_to_rgba_row,
    rgbf16_to_rgba_u16_row,
  },
  yuv::{Rgbf16, Rgbf16Row, Rgbf16Sink},
};

// ---- Rgbf16 impl -------------------------------------------------------

impl<'a> MixedSinker<'a, Rgbf16> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source values are
  /// clamped to `[0, 1]` and scaled by 255; alpha is forced to `0xFF`
  /// (the float source has no alpha channel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }

  /// Attaches a `u16` RGB output buffer (`width × height × 3`
  /// elements). Each `half::f16` channel is clamped to `[0, 1]` and
  /// **scaled to the full u16 range** (×65535).
  ///
  /// # Naming consistency note
  ///
  /// Other source families' `with_rgb_u16` accessor preserves the
  /// source's *native integer precision* in a u16 carrier (e.g.
  /// 10-bit YUV stays in `[0, 1023]`). The `Rgbf16` variant has no
  /// native integer range to preserve, so it instead applies full-
  /// range scaling — a deliberate divergence to give callers a useful
  /// u16 output rather than refusing the operation.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a `u16` RGBA output buffer. Same `[0, 1]` × 65535
  /// **full-range scaling** as
  /// [`with_rgb_u16`](Self::with_rgb_u16); alpha is forced to `0xFFFF`
  /// (the float source has no alpha channel). See
  /// [`with_rgb_u16`](Self::with_rgb_u16) for the divergence note vs
  /// integer-source families.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a **`u16`** luma output buffer. Y' is computed at u8
  /// precision (matching `with_luma`'s output) and zero-extended to
  /// `u16` — same convention as the packed-YUV `with_luma_u16` family.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_luma_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::LumaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches an `f32` RGB output buffer (`width × height × 3`
  /// elements). Each `half::f16` channel is widened to `f32` **losslessly**
  /// — HDR values > 1.0 and negative values are preserved bit-exact.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f32`](Self::with_rgb_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbF32BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a `half::f16` RGB output buffer (`width × height × 3`
  /// elements). The source is copied **losslessly** — HDR values > 1.0
  /// and negative values are preserved bit-exact. This is a direct
  /// `copy_from_slice` — no conversion is performed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f16`](Self::with_rgb_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbF16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_f16 = Some(buf);
    Ok(self)
  }
}

impl Rgbf16Sink for MixedSinker<'_, Rgbf16> {}

impl PixelSink for MixedSinker<'_, Rgbf16> {
  type Input<'r> = Rgbf16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Rgbf16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.rgb().len() != w * 3 {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::RgbF16Packed,
        row: idx,
        expected: w * 3,
        actual: row.rgb().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    let Self {
      rgb,
      rgb_u16,
      rgb_f32,
      rgb_f16,
      rgba,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let rgb_in = row.rgb();

    // Lossless f16 pass-through — emit first (independent of all other paths).
    if let Some(buf) = rgb_f16.as_deref_mut() {
      let f16_start = one_plane_start * 3;
      let f16_end = one_plane_end * 3;
      rgbf16_to_rgb_f16_row(rgb_in, &mut buf[f16_start..f16_end], w, use_simd);
    }

    // Lossless f32 widen — also independent of integer conversion paths.
    if let Some(buf) = rgb_f32.as_deref_mut() {
      let f32_start = one_plane_start * 3;
      let f32_end = one_plane_end * 3;
      rgbf16_to_rgb_f32_row(rgb_in, &mut buf[f32_start..f32_end], w, use_simd);
    }

    // u16 RGB output — direct half-float → u16 conversion (no staging).
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let u16_start = one_plane_start * 3;
      let u16_end = one_plane_end * 3;
      rgbf16_to_rgb_u16_row(rgb_in, &mut buf[u16_start..u16_end], w, use_simd);
    }

    // u16 RGBA output — direct half-float → u16 conversion (no staging).
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      rgbf16_to_rgba_u16_row(rgb_in, rgba_row, w, use_simd);
    }

    // u8 RGBA standalone fast path — direct float → u8 when no RGB / luma /
    // HSV consumer needs the staged u8 RGB row.
    let want_rgba_u8 = rgba.is_some();
    let want_rgb_u8 = rgb.is_some();
    let want_luma_u8 = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb = want_rgb_u8 || want_luma_u8 || want_luma_u16 || want_hsv;

    if want_rgba_u8 && !need_u8_rgb {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      rgbf16_to_rgba_row(rgb_in, rgba_row, w, use_simd);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba_u8 {
      return Ok(());
    }

    // Stage the u8 RGB scratch row once. When the user requested an RGB output
    // buffer it doubles as the shared u8 RGB row; otherwise use the
    // lazily-grown scratch.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    rgbf16_to_rgb_row(rgb_in, rgb_row, w, use_simd);

    if let Some(luma) = luma.as_deref_mut() {
      rgb_to_luma_row(
        rgb_row,
        &mut luma[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if let Some(luma_buf) = luma_u16.as_deref_mut() {
      rgb_to_luma_u16_row(
        rgb_row,
        &mut luma_buf[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u8 RGBA output (combined with RGB/luma/HSV path) — direct from
    // half-float source to keep alpha-fill cheap; avoids an extra pass
    // over `rgb_row` via `expand_rgb_to_rgba_row`.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      rgbf16_to_rgba_row(rgb_in, rgba_row, w, use_simd);
    }

    Ok(())
  }
}
