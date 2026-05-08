//! Sinker impls for 16-bit packed-RGB **source** formats (Tier 8 finish).
//!
//! Sources:
//! - [`Rgb48`]  — `R, G, B` u16 per pixel (`AV_PIX_FMT_RGB48LE`).
//! - [`Bgr48`]  — `B, G, R` u16 per pixel (`AV_PIX_FMT_BGR48LE`).
//! - [`Rgba64`] — `R, G, B, A` u16 per pixel (`AV_PIX_FMT_RGBA64LE`).
//! - [`Bgra64`] — `B, G, R, A` u16 per pixel (`AV_PIX_FMT_BGRA64LE`).
//!
//! All 7 output paths per format:
//! - `with_rgb`      — packed 16-bit → packed u8 RGB (narrow `>> 8` per channel).
//! - `with_rgba`     — same narrow; for Rgb48/Bgr48 alpha = `0xFF` (no source α);
//!   for Rgba64/Bgra64 source α is passed through (`>> 8`).
//! - `with_rgb_u16`  — native u16 passthrough (3 elements per pixel, R/G/B order).
//! - `with_rgba_u16` — native u16; for Rgb48/Bgr48 alpha = `0xFFFF`; for Rgba64/
//!   Bgra64 source α is copied verbatim (no shift).
//! - `with_luma`     — Y' derived from narrowed u8 RGB via `rgb_to_luma_row`.
//! - `with_luma_u16` — Y' derived from narrowed u8 RGB, zero-extended to u16.
//! - `with_hsv`      — HSV derived from narrowed u8 RGB via `rgb_to_hsv_row`.
//!
//! ## Alpha semantics — Rgba64 / Bgra64 (Strategy A+)
//!
//! When both `with_rgb` and `with_rgba` are attached, the u8 RGB kernel runs
//! **once** and RGBA is derived via `expand_rgb_to_rgba_row` (writes α=`0xFF`)
//! followed by `copy_alpha_packed_u16x4_to_u8_at_3` to overwrite slot 3 from
//! the source α. Output is byte-identical to calling `rgba64_to_rgba_row`
//! (standalone path) directly — spec § 3.2 / § 7.2.
//!
//! The same Strategy A+ applies on the u16 path: `expand_rgb_u16_to_rgba_u16_row`
//! fans out, then `copy_alpha_packed_u16x4_at_3` overwrites α from slot 3.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    bgr48_to_rgb_row_endian, bgr48_to_rgb_u16_row_endian, bgr48_to_rgba_row_endian,
    bgr48_to_rgba_u16_row_endian, bgra64_to_rgb_row_endian, bgra64_to_rgb_u16_row_endian,
    bgra64_to_rgba_row_endian, bgra64_to_rgba_u16_row_endian, expand_rgb_to_rgba_row,
    expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row, rgb_to_luma_row, rgb_to_luma_u16_row,
    rgb48_to_rgb_row_endian, rgb48_to_rgb_u16_row_endian, rgb48_to_rgba_row_endian,
    rgb48_to_rgba_u16_row_endian, rgba64_to_rgb_row_endian, rgba64_to_rgb_u16_row_endian,
    rgba64_to_rgba_row_endian, rgba64_to_rgba_u16_row_endian,
  },
  yuv::{
    Bgr48, Bgr48Row, Bgr48Sink, Bgra64, Bgra64Row, Bgra64Sink, Rgb48, Rgb48Row, Rgb48Sink, Rgba64,
    Rgba64Row, Rgba64Sink,
  },
};

// ---- Rgb48 -----------------------------------------------------------------

impl<'a> MixedSinker<'a, Rgb48> {
  /// Attaches a packed **8-bit** RGBA output buffer. Each 16-bit channel is
  /// narrowed `>> 8` and alpha is forced to `0xFF` (no source alpha in Rgb48).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if `buf.len() < width × height × 4`,
  /// or `Err(GeometryOverflow)` on 32-bit targets when the product overflows.
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

  /// Attaches a native **`u16`** RGB output buffer. Length in `u16` **elements**
  /// (`width × height × 3`). Channels are passed through verbatim (no shift).
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. Length in `u16` **elements**
  /// (`width × height × 4`). Alpha is forced to `0xFFFF` (no source alpha).
  ///
  /// Returns `Err(RgbaU16BufferTooShort)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer. Length in `u16` **elements**
  /// (`width × height`). Y' is computed at 8-bit precision and zero-extended.
  ///
  /// Returns `Err(LumaU16BufferTooShort)` if the buffer is too short.
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
}

impl Rgb48Sink for MixedSinker<'_, Rgb48> {}

impl PixelSink for MixedSinker<'_, Rgb48> {
  type Input<'r> = Rgb48Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Rgb48Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 3,
    })?;
    if row.rgb48().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Rgb48Packed,
        row: idx,
        expected: packed_expected,
        actual: row.rgb48().len(),
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
    let in48 = row.rgb48();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // u8 RGB staging — required when any of: with_rgb, with_luma,
    // with_luma_u16, or with_hsv is attached.
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      rgb48_to_rgb_row_endian::<false>(in48, rgb_row, w, use_simd);

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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv_bufs.h[ps..pe],
          &mut hsv_bufs.s[ps..pe],
          &mut hsv_bufs.v[ps..pe],
          w,
          use_simd,
        );
      }
    }

    // u8 RGBA — single-pass kernel, alpha forced to 0xFF.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, ps, pe, w, h)?;
      rgb48_to_rgba_row_endian::<false>(in48, rgba_row, w, use_simd);
    }

    // u16 RGB — native passthrough.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      let end = pe
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
      rgb48_to_rgb_u16_row_endian::<false>(in48, &mut buf[ps * 3..end], w, use_simd);
    }

    // u16 RGBA — native passthrough, alpha forced to 0xFFFF.
    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_u16_row = rgba_u16_plane_row_slice(buf, ps, pe, w, h)?;
      rgb48_to_rgba_u16_row_endian::<false>(in48, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Bgr48 -----------------------------------------------------------------

impl<'a> MixedSinker<'a, Bgr48> {
  /// Attaches a packed **8-bit** RGBA output buffer. B/R channels are swapped
  /// on output; each 16-bit channel is narrowed `>> 8`; alpha is forced to
  /// `0xFF` (no source alpha in Bgr48).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if `buf.len() < width × height × 4`.
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

  /// Attaches a native **`u16`** RGB output buffer. Length in `u16` **elements**
  /// (`width × height × 3`). B/R channels are swapped on output; no shift.
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. Length in `u16` **elements**
  /// (`width × height × 4`). B/R channels swapped; alpha forced to `0xFFFF`.
  ///
  /// Returns `Err(RgbaU16BufferTooShort)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer (`width × height` elements).
  /// Y' is computed at 8-bit precision and zero-extended to u16.
  ///
  /// Returns `Err(LumaU16BufferTooShort)` if the buffer is too short.
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
}

impl Bgr48Sink for MixedSinker<'_, Bgr48> {}

impl PixelSink for MixedSinker<'_, Bgr48> {
  type Input<'r> = Bgr48Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Bgr48Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 3,
    })?;
    if row.bgr48().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bgr48Packed,
        row: idx,
        expected: packed_expected,
        actual: row.bgr48().len(),
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
    let in48 = row.bgr48();

    let want_rgb = rgb.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      bgr48_to_rgb_row_endian::<false>(in48, rgb_row, w, use_simd);

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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv_bufs.h[ps..pe],
          &mut hsv_bufs.s[ps..pe],
          &mut hsv_bufs.v[ps..pe],
          w,
          use_simd,
        );
      }
    }

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, ps, pe, w, h)?;
      bgr48_to_rgba_row_endian::<false>(in48, rgba_row, w, use_simd);
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      let end = pe
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
      bgr48_to_rgb_u16_row_endian::<false>(in48, &mut buf[ps * 3..end], w, use_simd);
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      let rgba_u16_row = rgba_u16_plane_row_slice(buf, ps, pe, w, h)?;
      bgr48_to_rgba_u16_row_endian::<false>(in48, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Rgba64 ----------------------------------------------------------------

impl<'a> MixedSinker<'a, Rgba64> {
  /// Attaches a packed **8-bit** RGBA output buffer. Each 16-bit channel is
  /// narrowed `>> 8`; the **source alpha** at slot 3 of each pixel is
  /// depth-converted and passed through (not forced to `0xFF`).
  ///
  /// ## Strategy note
  ///
  /// Source-α pass-through is guaranteed in **all** paths. When standalone
  /// (no `with_rgb` / `with_hsv`), `rgba64_to_rgba_row` runs directly.
  /// When combined with `with_rgb`, Strategy A+ applies:
  /// `expand_rgb_to_rgba_row` fans out the RGB row (α=`0xFF`) and
  /// `copy_alpha_packed_u16x4_to_u8_at_3` overwrites the α slot from the
  /// source — output is byte-identical to the standalone path.
  ///
  /// Returns `Err(RgbaBufferTooShort)` if `buf.len() < width × height × 4`.
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

  /// Attaches a native **`u16`** RGB output buffer. Length in `u16` **elements**
  /// (`width × height × 3`). Alpha slot dropped (RGB only, 3 channels).
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. Length in `u16` **elements**
  /// (`width × height × 4`). Source α at slot 3 is copied verbatim (no shift).
  ///
  /// ## Strategy note
  ///
  /// When `with_rgb_u16` is also attached, Strategy A+ applies on the u16
  /// path: `expand_rgb_u16_to_rgba_u16_row` fans out and
  /// `copy_alpha_packed_u16x4_at_3` overwrites α — output byte-identical
  /// to the standalone `rgba64_to_rgba_u16_row` path.
  ///
  /// Returns `Err(RgbaU16BufferTooShort)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer (`width × height` elements).
  /// Y' is derived from narrowed u8 RGB and zero-extended to u16.
  ///
  /// Returns `Err(LumaU16BufferTooShort)` if the buffer is too short.
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
}

impl Rgba64Sink for MixedSinker<'_, Rgba64> {}

impl PixelSink for MixedSinker<'_, Rgba64> {
  type Input<'r> = Rgba64Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Rgba64Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected = w.checked_mul(4).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 4,
    })?;
    if row.rgba64().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Rgba64Packed,
        row: idx,
        expected: packed_expected,
        actual: row.rgba64().len(),
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
    let in64 = row.rgba64();

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // ===== u8 path =====

    // Standalone RGBA u8 fast path — only rgba attached (no u8 RGB or u16
    // work). Source α passes through via the kernel.
    if want_rgba && !need_u8_rgb && !want_rgb_u16 && !want_rgba_u16 {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      rgba64_to_rgba_row_endian::<false>(in64, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone RGBA u16 fast path — only rgba_u16 attached, no u8 work.
    if want_rgba_u16 && !want_rgb_u16 && !need_u8_rgb && !want_rgba {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      rgba64_to_rgba_u16_row_endian::<false>(in64, rgba_u16_row, w, use_simd);
      return Ok(());
    }

    // u8 RGB staging — drives with_rgb / with_luma / with_luma_u16 / with_hsv,
    // and Strategy A+ RGBA fan-out.
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      rgba64_to_rgb_row_endian::<false>(in64, rgb_row, w, use_simd);

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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv_bufs.h[ps..pe],
          &mut hsv_bufs.s[ps..pe],
          &mut hsv_bufs.v[ps..pe],
          w,
          use_simd,
        );
      }

      // Strategy A+ u8: RGBA also attached — derive from the just-computed
      // RGB row (writes α=0xFF), then overwrite α slot from packed source
      // (slot 3, depth-conv >> 8). Output is byte-identical to calling
      // rgba64_to_rgba_row directly (spec § 3.2 / § 7.2).
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        // `Rgba64Frame` / `Bgra64Frame` are LE-encoded per the unified Frame
        // contract → `BE = false`.
        crate::row::scalar::alpha_extract::copy_alpha_packed_u16x4_to_u8_at_3::<false>(
          in64, rgba_row, w,
        );
      }
    }

    // Standalone RGBA u8 path — want_rgba without need_u8_rgb (combined with
    // u16 work only). Run rgba64_to_rgba_row directly; source α depth-conv.
    if want_rgba && !need_u8_rgb {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      rgba64_to_rgba_row_endian::<false>(in64, rgba_row, w, use_simd);
    }

    // ===== u16 path =====

    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let end = pe
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
      let rgb_u16_row = &mut rgb_u16_buf[ps * 3..end];
      rgba64_to_rgb_u16_row_endian::<false>(in64, rgb_u16_row, w, use_simd);

      // Strategy A+ u16: RGBA u16 also attached — derive from the
      // just-computed u16 RGB row (writes α=0xFFFF), then overwrite α
      // slot from packed source (slot 3, u16 verbatim).
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        // `Rgba64Frame` / `Bgra64Frame` are LE-encoded per the unified Frame
        // contract → `BE = false`.
        crate::row::scalar::alpha_extract::copy_alpha_packed_u16x4_at_3::<false>(
          in64,
          rgba_u16_row,
          w,
        );
      }
    }

    // Standalone RGBA u16 path — want_rgba_u16 without want_rgb_u16 (combined
    // with u8 work). Run rgba64_to_rgba_u16_row directly.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      rgba64_to_rgba_u16_row_endian::<false>(in64, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Bgra64 ----------------------------------------------------------------

impl<'a> MixedSinker<'a, Bgra64> {
  /// Attaches a packed **8-bit** RGBA output buffer. B/R channels swapped on
  /// output; each 16-bit channel narrowed `>> 8`; the **source alpha** at slot
  /// 3 of each pixel is depth-converted and passed through (not forced).
  ///
  /// Same Strategy A+ semantics as [`MixedSinker::<Rgba64>::with_rgba`] —
  /// see that method's doc for the standalone vs combo behaviour.
  ///
  /// Returns `Err(RgbaBufferTooShort)` if `buf.len() < width × height × 4`.
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

  /// Attaches a native **`u16`** RGB output buffer. B/R channels swapped on
  /// output; length in `u16` **elements** (`width × height × 3`).
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if the buffer is too short.
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

  /// Attaches a native **`u16`** RGBA output buffer. B/R channels swapped;
  /// source α at slot 3 copied verbatim. Length in `u16` **elements**
  /// (`width × height × 4`).
  ///
  /// Returns `Err(RgbaU16BufferTooShort)` if the buffer is too short.
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

  /// Attaches a native **`u16`** luma output buffer (`width × height` elements).
  /// Y' is derived from narrowed u8 RGB and zero-extended to u16.
  ///
  /// Returns `Err(LumaU16BufferTooShort)` if the buffer is too short.
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
}

impl Bgra64Sink for MixedSinker<'_, Bgra64> {}

impl PixelSink for MixedSinker<'_, Bgra64> {
  type Input<'r> = Bgra64Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Bgra64Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let packed_expected = w.checked_mul(4).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 4,
    })?;
    if row.bgra64().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bgra64Packed,
        row: idx,
        expected: packed_expected,
        actual: row.bgra64().len(),
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
    let in64 = row.bgra64();

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();
    let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

    // Standalone RGBA u8 fast path.
    if want_rgba && !need_u8_rgb && !want_rgb_u16 && !want_rgba_u16 {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      bgra64_to_rgba_row_endian::<false>(in64, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone RGBA u16 fast path.
    if want_rgba_u16 && !want_rgb_u16 && !need_u8_rgb && !want_rgba {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      bgra64_to_rgba_u16_row_endian::<false>(in64, rgba_u16_row, w, use_simd);
      return Ok(());
    }

    // u8 RGB staging path.
    if need_u8_rgb {
      let rgb_row = rgb_row_buf_or_scratch(rgb.as_deref_mut(), rgb_scratch, ps, pe, w, h)?;
      bgra64_to_rgb_row_endian::<false>(in64, rgb_row, w, use_simd);

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
        rgb_to_hsv_row(
          rgb_row,
          &mut hsv_bufs.h[ps..pe],
          &mut hsv_bufs.s[ps..pe],
          &mut hsv_bufs.v[ps..pe],
          w,
          use_simd,
        );
      }

      // Strategy A+ u8: RGBA also attached.
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        // `Rgba64Frame` / `Bgra64Frame` are LE-encoded per the unified Frame
        // contract → `BE = false`.
        crate::row::scalar::alpha_extract::copy_alpha_packed_u16x4_to_u8_at_3::<false>(
          in64, rgba_row, w,
        );
      }
    }

    // Standalone RGBA u8 path — combined with u16 work only.
    if want_rgba && !need_u8_rgb {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, ps, pe, w, h)?;
      bgra64_to_rgba_row_endian::<false>(in64, rgba_row, w, use_simd);
    }

    // u16 RGB path.
    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let end = pe
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
      let rgb_u16_row = &mut rgb_u16_buf[ps * 3..end];
      bgra64_to_rgb_u16_row_endian::<false>(in64, rgb_u16_row, w, use_simd);

      // Strategy A+ u16: RGBA u16 also attached.
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        // `Rgba64Frame` / `Bgra64Frame` are LE-encoded per the unified Frame
        // contract → `BE = false`.
        crate::row::scalar::alpha_extract::copy_alpha_packed_u16x4_at_3::<false>(
          in64,
          rgba_u16_row,
          w,
        );
      }
    }

    // Standalone RGBA u16 path — combined with u8 work.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_u16_buf, ps, pe, w, h)?;
      bgra64_to_rgba_u16_row_endian::<false>(in64, rgba_u16_row, w, use_simd);
    }

    Ok(())
  }
}
