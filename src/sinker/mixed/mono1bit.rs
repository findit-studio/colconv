//! [`PixelSink`] implementations for Monoblack and Monowhite sources.

use super::{
  BufferTooShort, MixedSinker, MixedSinkerError, RowIndexOutOfRange, RowShapeMismatch, RowSlice,
  check_dimensions_match,
};
use crate::{
  PixelSink, row,
  source::{Monoblack, MonoblackRow, MonoblackSink, Monowhite, MonowhiteRow, MonowhiteSink},
};

// ---- Monoblack impl ---------------------------------------------------------

impl<'a> MixedSinker<'a, Monoblack> {
  /// Attaches a packed **`u8`** RGBA output buffer.
  ///
  /// Length is measured in `u8` **bytes**: minimum `width * height * 4`.
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
      return Err(MixedSinkerError::RgbaBufferTooShort(BufferTooShort {
        expected,
        actual: buf.len(),
      }));
    }
    self.rgba = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 3`.
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
      return Err(MixedSinkerError::RgbU16BufferTooShort(BufferTooShort {
        expected,
        actual: buf.len(),
      }));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGBA output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 4`.
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
      return Err(MixedSinkerError::RgbaU16BufferTooShort(BufferTooShort {
        expected,
        actual: buf.len(),
      }));
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a planar **`u16`** luma output buffer.
  ///
  /// Luma is derived from RGB via BT.709 weights (by default).
  /// Length: minimum `width * height`.
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
      return Err(MixedSinkerError::LumaU16BufferTooShort(BufferTooShort {
        expected,
        actual: buf.len(),
      }));
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

impl MonoblackSink for MixedSinker<'_, Monoblack> {}

impl PixelSink for MixedSinker<'_, Monoblack> {
  type Input<'i> = MonoblackRow<'i>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Self::Input<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let min_bytes = w.div_ceil(8);
    if row.data().len() < min_bytes {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: min_bytes,
        actual: row.data().len(),
      }));
    }

    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(RowIndexOutOfRange {
        row: idx,
        configured_height: h,
      }));
    }

    let Self {
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(buf) = rgb.as_deref_mut() {
      row::monoblack_to_rgb_or_rgba_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      row::monoblack_to_rgb_or_rgba_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      row::monoblack_to_rgb_u16_or_rgba_u16_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      row::monoblack_to_rgb_u16_or_rgba_u16_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma.as_deref_mut() {
      row::monoblack_to_luma_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma_u16.as_deref_mut() {
      row::monoblack_to_luma_u16_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      row::monoblack_to_hsv_row(
        row.data(),
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    Ok(())
  }
}

// ---- Monowhite impl ---------------------------------------------------------

impl<'a> MixedSinker<'a, Monowhite> {
  /// Attaches a packed **`u8`** RGBA output buffer.
  ///
  /// Length is measured in `u8` **bytes**: minimum `width * height * 4`.
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
      return Err(MixedSinkerError::RgbaBufferTooShort(BufferTooShort {
        expected,
        actual: buf.len(),
      }));
    }
    self.rgba = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 3`.
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
      return Err(MixedSinkerError::RgbU16BufferTooShort(BufferTooShort {
        expected,
        actual: buf.len(),
      }));
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGBA output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 4`.
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
      return Err(MixedSinkerError::RgbaU16BufferTooShort(BufferTooShort {
        expected,
        actual: buf.len(),
      }));
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a planar **`u16`** luma output buffer.
  ///
  /// Luma is derived from RGB via BT.709 weights (by default).
  /// Length: minimum `width * height`.
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
      return Err(MixedSinkerError::LumaU16BufferTooShort(BufferTooShort {
        expected,
        actual: buf.len(),
      }));
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

impl MonowhiteSink for MixedSinker<'_, Monowhite> {}

impl PixelSink for MixedSinker<'_, Monowhite> {
  type Input<'i> = MonowhiteRow<'i>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Self::Input<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let min_bytes = w.div_ceil(8);
    if row.data().len() < min_bytes {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: min_bytes,
        actual: row.data().len(),
      }));
    }

    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(RowIndexOutOfRange {
        row: idx,
        configured_height: h,
      }));
    }

    let Self {
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(buf) = rgb.as_deref_mut() {
      row::monowhite_to_rgb_or_rgba_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      row::monowhite_to_rgb_or_rgba_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      row::monowhite_to_rgb_u16_or_rgba_u16_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      row::monowhite_to_rgb_u16_or_rgba_u16_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma.as_deref_mut() {
      row::monowhite_to_luma_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma_u16.as_deref_mut() {
      row::monowhite_to_luma_u16_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      row::monowhite_to_hsv_row(
        row.data(),
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    Ok(())
  }
}
