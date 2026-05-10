//! `MixedSinker<'_, Pal8>` — 8-bit indexed-color (`AV_PIX_FMT_PAL8`) sinker.
//!
//! Each pixel is a `u8` index into a 256-entry BGRA palette threaded via
//! [`Pal8Row`](crate::raw::Pal8Row). No new `MixedSinker` fields are added;
//! the palette is accessed per-row from the walker's `Pal8Row`.
//!
//! ## Output channels
//!
//! | Accessor | Output |
//! |---|---|
//! | [`with_rgb`](MixedSinker::with_rgb) | 8-bit packed `[R, G, B]` |
//! | [`with_rgba`](MixedSinker::with_rgba) | 8-bit packed `[R, G, B, A]` — α from palette |
//! | [`with_rgb_u16`](MixedSinker::<Pal8>::with_rgb_u16) | `u16` packed `[R, G, B]`, `(x << 8) \| x` scaled |
//! | [`with_rgba_u16`](MixedSinker::<Pal8>::with_rgba_u16) | `u16` packed `[R, G, B, A]` — α from palette |
//! | [`with_luma`](MixedSinker::with_luma) | BT.709 luma from RGB |
//! | [`with_luma_u16`](MixedSinker::<Pal8>::with_luma_u16) | `(y << 8) \| y` widened luma |
//! | [`with_hsv`](MixedSinker::with_hsv) | HSV via RGB scratch |
//!
//! ## Strategy A+ for `with_rgb + with_rgba` combo
//!
//! When both are attached, `pal8_to_rgba_row` runs once (filling the RGBA
//! buffer with real per-pixel palette alpha); RGB bytes are then stripped
//! from the RGBA buffer. This avoids a second palette lookup.
//!
//! The same Strategy A+ applies to `with_rgb_u16 + with_rgba_u16`.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgb_row_to_luma_row, rgb_row_to_luma_u16_row, rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  raw::*,
  row::{
    pal8_to_rgb_row, pal8_to_rgb_u16_row, pal8_to_rgba_row, pal8_to_rgba_u16_row, rgb_to_hsv_row,
  },
};

// ---- Format-specific accessor block ----------------------------------------

impl<'a> MixedSinker<'a, Pal8> {
  /// Attaches a packed 8-bit RGBA output buffer.
  ///
  /// Alpha byte per pixel is sourced from the palette entry's `A` field
  /// (FFmpeg PAL8 `[B, G, R, A]` order). Returns `Err(RgbaBufferTooShort)`
  /// if `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32-bit overflow.
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

  /// Attaches a packed `u16` RGB output buffer. Each 8-bit palette channel is
  /// widened via `(x << 8) | x` (`0 → 0x0000`, `255 → 0xFFFF`). Alpha is
  /// dropped. Length measured in `u16` elements (`width × height × 3`).
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

  /// Attaches a packed `u16` RGBA output buffer. Each 8-bit palette channel
  /// (including alpha) is widened via `(x << 8) | x`. Length measured in
  /// `u16` elements (`width × height × 4`).
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

  /// Attaches a `u16` luma output buffer. The luma byte derived from RGB is
  /// widened via `(y << 8) | y` (`0 → 0x0000`, `255 → 0xFFFF`). Length
  /// measured in `u16` elements (`width × height`).
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
      return Err(MixedSinkerError::LumaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

// ---- Sink trait impl -------------------------------------------------------

impl Pal8Sink for MixedSinker<'_, Pal8> {}

// ---- PixelSink impl --------------------------------------------------------

impl PixelSink for MixedSinker<'_, Pal8> {
  type Input<'r> = Pal8Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Pal8Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.idx();
    let use_simd = self.simd;

    // Defense-in-depth: the walker always sends matching slices, but a caller
    // that drives the sink manually (e.g. in tests) needs a clean error rather
    // than an out-of-bounds index inside the kernel.
    if row.row().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Pal8IndexRow,
        row: idx,
        expected: w,
        actual: row.row().len(),
      });
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: h,
      });
    }

    // Capture Copy fields before the `Self { .. }` destructure to avoid
    // re-borrowing `self` inside the luma path.
    let luma_coeffs_q8 = self.luma_coefficients_q8;

    let Self {
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;

    let palette = row.palette();
    let indices = row.row();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // ---- u8 RGB / RGBA (Strategy A+ for combo) ----------------------------
    //
    // PAL8 carries real per-pixel alpha in the palette — so we use the RGBA
    // kernel as the primary, then strip RGB bytes from the RGBA output.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();

    if want_rgb && want_rgba {
      // Strategy A+: one palette lookup → fill RGBA → strip RGB.
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      pal8_to_rgba_row(indices, palette, rgba_row, use_simd);
      // Copy RGB bytes from RGBA output into the RGB buffer.
      let rgb_buf = rgb.as_deref_mut().unwrap();
      for i in 0..w {
        let rgba_base = i * 4;
        let rgb_base = (one_plane_start + i) * 3;
        rgb_buf[rgb_base] = rgba_row[rgba_base];
        rgb_buf[rgb_base + 1] = rgba_row[rgba_base + 1];
        rgb_buf[rgb_base + 2] = rgba_row[rgba_base + 2];
      }
    } else if want_rgba {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      pal8_to_rgba_row(indices, palette, rgba_row, use_simd);
    } else if want_rgb {
      let rgb_buf = rgb.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      pal8_to_rgb_row(
        indices,
        palette,
        &mut rgb_buf[rgb_plane_start..rgb_plane_end],
        use_simd,
      );
    }

    // ---- u16 RGB / RGBA (Strategy A+ for combo) ---------------------------
    //
    // Same pattern: pal8_to_rgba_u16_row as primary; strip RGB-u16 from RGBA-u16.
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgb_u16 && want_rgba_u16 {
      // Strategy A+: fill rgba_u16 first, then copy RGB elements.
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      pal8_to_rgba_u16_row(indices, palette, rgba_u16_row, use_simd);
      // Copy RGB-u16 elements (3 of 4 per pixel) into the RGB-u16 buffer.
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      for i in 0..w {
        let rgba_base = i * 4;
        let rgb_base = rgb_plane_start + i * 3;
        rgb_u16_buf[rgb_base] = rgba_u16_row[rgba_base];
        rgb_u16_buf[rgb_base + 1] = rgba_u16_row[rgba_base + 1];
        rgb_u16_buf[rgb_base + 2] = rgba_u16_row[rgba_base + 2];
      }
    } else if want_rgba_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      pal8_to_rgba_u16_row(indices, palette, rgba_u16_row, use_simd);
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      pal8_to_rgb_u16_row(
        indices,
        palette,
        &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end],
        use_simd,
      );
    }

    // ---- Luma / HSV (staged via u8 RGB scratch) ---------------------------
    let want_luma = luma.is_some();
    let want_luma_u16 = luma_u16.is_some();
    let want_hsv = hsv.is_some();

    if want_luma || want_luma_u16 || want_hsv {
      // Obtain an RGB row: reuse the just-computed `rgb` slice when attached;
      // otherwise fall through to the scratch buffer.
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;

      // Only run the RGB kernel if `with_rgb` was NOT set (it already
      // populated the rgb slice above). If `want_rgb` was true, `rgb_row`
      // now points into the rgb buffer which is already populated.
      if !want_rgb {
        pal8_to_rgb_row(indices, palette, rgb_row, use_simd);
      }

      if let Some(luma_buf) = luma.as_deref_mut() {
        rgb_row_to_luma_row(
          rgb_row,
          &mut luma_buf[one_plane_start..one_plane_end],
          luma_coeffs_q8,
        );
      }

      if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
        rgb_row_to_luma_u16_row(
          rgb_row,
          &mut luma_u16_buf[one_plane_start..one_plane_end],
          luma_coeffs_q8,
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
    }

    Ok(())
  }
}
