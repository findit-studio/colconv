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
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, uyyvyy411_to_luma_row, uyyvyy411_to_luma_u16_row,
    uyyvyy411_to_rgb_row, uyyvyy411_to_rgba_row,
  },
  yuv::{Uyyvyy411, Uyyvyy411Row, Uyyvyy411Sink},
};

impl<'a> MixedSinker<'a, Uyyvyy411> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
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

  /// Attaches a **`u16`** luma output buffer. Y bytes are zero-extended
  /// to u16 (`out[x] = Y_byte as u16`). Length in u16 **elements**
  /// (`width × height`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_luma_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    // `buf` is `&mut [u16]`, so `buf.len()` is in u16 elements. Compute
    // the expected length explicitly in elements (`width * height`)
    // rather than reusing `frame_bytes` — its name implies bytes, even
    // though for `channels = 1` it numerically matches the element
    // count for a single-channel buffer.
    let expected_elems =
      (self.width)
        .checked_mul(self.height)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width: self.width,
          height: self.height,
          channels: 1,
        })?;
    if buf.len() < expected_elems {
      return Err(MixedSinkerError::LumaU16BufferTooShort {
        expected: expected_elems,
        actual: buf.len(),
      });
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

impl Uyyvyy411Sink for MixedSinker<'_, Uyyvyy411> {}

impl PixelSink for MixedSinker<'_, Uyyvyy411> {
  type Input<'r> = Uyyvyy411Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 3 != 0 {
      return Err(MixedSinkerError::WidthNotMultipleOf4 { width: self.width });
    }
    Ok(())
  }

  fn process(&mut self, row: Uyyvyy411Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 3 != 0 {
      return Err(MixedSinkerError::WidthNotMultipleOf4 { width: w });
    }

    // Row length: `width * 3 / 2` (12 bpp). `w` is a multiple of 4 by
    // the gate above, so `w * 3` is also a multiple of 4 and the
    // `/ 2` is exact. Check the `* 3` for 32-bit overflow.
    let packed_expected =
      w.checked_mul(3)
        .map(|n| n / 2)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 3,
        })?;
    if row.uyyvyy().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Uyyvyy411Packed,
        row: idx,
        expected: packed_expected,
        actual: row.uyyvyy().len(),
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
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      ..
    } = self;
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
      rgb_to_hsv_row(
        rgb_row,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
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
