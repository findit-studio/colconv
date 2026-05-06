//! `MixedSinker` impls for gray source formats: `Gray8`, `GrayN<BITS>`, `Gray16`.
//!
//! Gray sources are achromatic — every pixel has luma only, no chroma.
//! All gray→RGB conversions broadcast Y to R=G=B. All gray→HSV outputs
//! have H=0 and S=0 (achromatic convention, matching OpenCV).
//!
//! Gray8 (u8 plane):
//! - `with_rgb`  → broadcast Y to [Y, Y, Y] u8.
//! - `with_rgba` → broadcast Y to [Y, Y, Y, 0xFF] u8.
//! - `with_luma` → copy Y plane (memcpy); no dedicated kernel needed.
//! - `with_luma_u16` → zero-extend Y bytes to u16.
//! - `with_hsv`  → H=0, S=0, V=Y.
//!
//! GrayN (u16 low-bit-packed, BITS ∈ {9,10,12,14}):
//! - `with_rgb`       → mask + shift (BITS→8) → broadcast to u8 RGB.
//! - `with_rgba`      → same + alpha=0xFF.
//! - `with_rgb_u16`   → mask → broadcast to u16 RGB.
//! - `with_rgba_u16`  → mask → broadcast + alpha = bits_mask<BITS>().
//! - `with_luma`      → mask + shift → u8.
//! - `with_luma_u16`  → mask → u16.
//! - `with_hsv`       → H=0, S=0, V = mask+shift→u8.
//!
//! Gray16 (u16 native):
//! - `with_rgb`       → `>> 8` → broadcast to u8 RGB.
//! - `with_rgba`      → `>> 8` → broadcast + alpha=0xFF.
//! - `with_rgb_u16`   → identity → broadcast to u16 RGB.
//! - `with_rgba_u16`  → identity → broadcast + alpha=0xFFFF.
//! - `with_luma`      → `>> 8` → u8.
//! - `with_luma_u16`  → copy (memcpy).
//! - `with_hsv`       → H=0, S=0, V = `>> 8`.
//!
//! Strategy A: when both u8 RGB and u8 RGBA are requested, compute RGB once
//! then fan out to RGBA via `expand_rgb_to_rgba_row`. Same on the u16 path.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, gray_n_to_hsv_row, gray_n_to_luma_row,
    gray_n_to_luma_u16_row, gray_n_to_rgb_row, gray_n_to_rgb_u16_row, gray_n_to_rgba_row,
    gray_n_to_rgba_u16_row, gray8_to_hsv_row, gray8_to_rgb_row, gray8_to_rgba_row,
    gray16_to_hsv_row, gray16_to_luma_row, gray16_to_luma_u16_row, gray16_to_rgb_row,
    gray16_to_rgb_u16_row, gray16_to_rgba_row, gray16_to_rgba_u16_row, rgb_to_hsv_row,
    y_plane_to_luma_u16_row,
  },
  yuv::{Gray8, Gray8Row, Gray8Sink, Gray16, Gray16Row, Gray16Sink},
};

// ---- Gray8 impl -------------------------------------------------------------

impl<'a> MixedSinker<'a, Gray8> {
  /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`
  /// (Gray8 has no alpha channel).
  ///
  /// Returns `Err(RgbaBufferTooShort)` if `buf.len() < width × height × 4`,
  /// or `Err(GeometryOverflow)` on 32-bit overflow.
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

  /// Attaches a u16 luma output buffer. Gray8 Y bytes are zero-extended
  /// to u16 (each output element equals `y_byte as u16`). Length measured
  /// in `u16` elements (`width × height`).
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

impl Gray8Sink for MixedSinker<'_, Gray8> {}

impl PixelSink for MixedSinker<'_, Gray8> {
  type Input<'r> = Gray8Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Gray8Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Row shape check — defense-in-depth before any unsafe kernel.
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: h,
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
    let y_plane = row.y();
    let full_range = row.full_range();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma u8 — Gray8: Y IS luma; copy directly (no kernel overhead).
    // Luma outputs always pass raw Y through — no full_range rescaling.
    if let Some(buf) = luma.as_deref_mut() {
      buf[one_plane_start..one_plane_end].copy_from_slice(y_plane);
    }

    // Luma u16 — zero-extend u8 Y to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      y_plane_to_luma_u16_row(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u8 RGB / RGBA / HSV path.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();

    // Standalone RGBA fast path — no RGB or HSV requested.
    if want_rgba && !want_rgb && !want_hsv {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gray8_to_rgba_row(y_plane, rgba_row, w, use_simd, full_range);
      return Ok(());
    }

    // Standalone HSV fast path — for gray sources, H=0/S=0/V=Y (rescaled if
    // limited-range) without any RGB computation. Use the dedicated kernel
    // when neither RGB nor RGBA is also requested.
    if want_hsv && !want_rgb && !want_rgba {
      let hsv = hsv.as_mut().unwrap();
      gray8_to_hsv_row(
        y_plane,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
        full_range,
      );
      return Ok(());
    }

    if !want_rgb && !want_rgba && !want_hsv {
      return Ok(());
    }

    // At least RGB or RGBA (or HSV+RGB/RGBA) requested — run the RGB kernel.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    gray8_to_rgb_row(y_plane, rgb_row, w, use_simd, full_range);

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

    // Strategy A fan-out — derive RGBA from the just-computed RGB row.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- GrayN impl (const BITS) ------------------------------------------------
//
// We ship one const-generic helper that serves all 4 bit depths (9/10/12/14).
// Each alias (Gray9/10/12/14) gets its own builder impl, all forwarding to
// the same MixedSinker fields and the same const-generic kernels.

/// Internal process implementation for GrayN formats. Called by all four
/// `PixelSink::process` impls via their per-format `const BITS: u32`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn process_gray_n<'a, const BITS: u32>(
  w: usize,
  h: usize,
  idx: usize,
  use_simd: bool,
  full_range: bool,
  y_plane: &[u16],
  rgb: &mut Option<&'a mut [u8]>,
  rgb_u16: &mut Option<&'a mut [u16]>,
  rgba: &mut Option<&'a mut [u8]>,
  rgba_u16: &mut Option<&'a mut [u16]>,
  luma: &mut Option<&'a mut [u8]>,
  luma_u16: &mut Option<&'a mut [u16]>,
  hsv: &mut Option<crate::HsvBuffers<'a>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
) -> Result<(), MixedSinkerError> {
  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  // Luma u8 — always passes raw Y through, no full_range rescaling.
  if let Some(buf) = luma.as_deref_mut() {
    gray_n_to_luma_row::<BITS>(
      y_plane,
      &mut buf[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
  }

  // Luma u16 — always passes raw Y through, no full_range rescaling.
  if let Some(buf) = luma_u16.as_deref_mut() {
    gray_n_to_luma_u16_row::<BITS>(
      y_plane,
      &mut buf[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
  }

  // u16 RGB / RGBA path (Strategy A).
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba_u16 = rgba_u16.is_some();

  if want_rgba_u16 && !want_rgb_u16 {
    let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
    let rgba_u16_row =
      rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
    gray_n_to_rgba_u16_row::<BITS>(y_plane, rgba_u16_row, w, use_simd, full_range);
  } else if want_rgb_u16 {
    let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
    let rgb_plane_start = one_plane_start * 3;
    let rgb_plane_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow {
        width: w,
        height: h,
        channels: 3,
      })?;
    let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
    gray_n_to_rgb_u16_row::<BITS>(y_plane, rgb_u16_row, w, use_simd, full_range);
    if want_rgba_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
    }
  }

  // u8 RGB / RGBA / HSV path.
  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();

  // Standalone RGBA fast path.
  if want_rgba && !want_rgb && !want_hsv {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    gray_n_to_rgba_row::<BITS>(y_plane, rgba_row, w, use_simd, full_range);
    return Ok(());
  }

  // Standalone HSV fast path — gray sources always have H=0, S=0, V=Y8
  // (rescaled if limited-range).
  if want_hsv && !want_rgb && !want_rgba {
    let hsv = hsv.as_mut().unwrap();
    gray_n_to_hsv_row::<BITS>(
      y_plane,
      &mut hsv.h[one_plane_start..one_plane_end],
      &mut hsv.s[one_plane_start..one_plane_end],
      &mut hsv.v[one_plane_start..one_plane_end],
      w,
      use_simd,
      full_range,
    );
    return Ok(());
  }

  if !want_rgb && !want_rgba && !want_hsv {
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
  gray_n_to_rgb_row::<BITS>(y_plane, rgb_row, w, use_simd, full_range);

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

  if let Some(buf) = rgba.as_deref_mut() {
    let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
    expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
  }

  Ok(())
}

/// Common row-shape validator for GrayN sinkers.
#[inline(always)]
fn check_gray_n_row_shape(
  y_len: usize,
  w: usize,
  idx: usize,
  h: usize,
) -> Result<(), MixedSinkerError> {
  if y_len != w {
    return Err(MixedSinkerError::RowShapeMismatch {
      which: RowSlice::Y,
      row: idx,
      expected: w,
      actual: y_len,
    });
  }
  if idx >= h {
    return Err(MixedSinkerError::RowIndexOutOfRange {
      row: idx,
      configured_height: h,
    });
  }
  Ok(())
}

// ---- Per-bit-depth builder impls for GrayN ----------------------------------

macro_rules! impl_gray_n_sinker {
  ($marker:ty, $row:ident, $sink:ty, $bits:expr) => {
    impl<'a> MixedSinker<'a, $marker> {
      /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`.
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

      /// Attaches a u16 RGB output buffer. Samples are masked to the low
      /// `BITS` bits; length is in `u16` elements (`width × height × 3`).
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

      /// Attaches a u16 RGBA output buffer. Samples masked to low `BITS` bits;
      /// alpha = `(1 << BITS) - 1` (full-range opaque). Length in `u16` elements
      /// (`width × height × 4`).
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

      /// Attaches a u16 luma output buffer. Samples masked to low `BITS`
      /// bits; length in `u16` elements (`width × height`).
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

    impl $sink for MixedSinker<'_, $marker> {}

    impl PixelSink for MixedSinker<'_, $marker> {
      type Input<'r> = $row<'r>;
      type Error = MixedSinkerError;

      fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
        check_dimensions_match(self.width, self.height, width, height)
      }

      fn process(&mut self, row: $row<'_>) -> Result<(), Self::Error> {
        let w = self.width;
        let h = self.height;
        let use_simd = self.simd;
        let idx = row.row();
        let full_range = row.full_range();
        check_gray_n_row_shape(row.y().len(), w, idx, h)?;
        let y_plane = row.y();
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
        process_gray_n::<$bits>(
          w,
          h,
          idx,
          use_simd,
          full_range,
          y_plane,
          rgb,
          rgb_u16,
          rgba,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
        )
      }
    }
  };
}

// Import the gray walker types for the macro instantiation.
use crate::yuv::{
  Gray9, Gray9Row, Gray9Sink, Gray10, Gray10Row, Gray10Sink, Gray12, Gray12Row, Gray12Sink, Gray14,
  Gray14Row, Gray14Sink,
};

impl_gray_n_sinker!(Gray9, Gray9Row, Gray9Sink, 9);
impl_gray_n_sinker!(Gray10, Gray10Row, Gray10Sink, 10);
impl_gray_n_sinker!(Gray12, Gray12Row, Gray12Sink, 12);
impl_gray_n_sinker!(Gray14, Gray14Row, Gray14Sink, 14);

// ---- Gray16 impl ------------------------------------------------------------

impl<'a> MixedSinker<'a, Gray16> {
  /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`.
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

  /// Attaches a u16 RGB output buffer (`>> 8` is NOT applied — native
  /// 16-bit broadcast). Length in `u16` elements (`width × height × 3`).
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

  /// Attaches a u16 RGBA output buffer (native 16-bit broadcast; alpha
  /// = `0xFFFF`). Length in `u16` elements (`width × height × 4`).
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

  /// Attaches a u16 luma output buffer (identity copy of the Gray16 Y
  /// plane). Length in `u16` elements (`width × height`).
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

impl Gray16Sink for MixedSinker<'_, Gray16> {}

impl PixelSink for MixedSinker<'_, Gray16> {
  type Input<'r> = Gray16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Gray16Row<'_>) -> Result<(), Self::Error> {
    const BITS: u32 = 16;
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    let full_range = row.full_range();

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: w,
        actual: row.y().len(),
      });
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: h,
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
    let y_plane = row.y();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma u8 — shift >> 8.
    if let Some(buf) = luma.as_deref_mut() {
      gray16_to_luma_row(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Luma u16 — identity copy.
    if let Some(buf) = luma_u16.as_deref_mut() {
      gray16_to_luma_u16_row(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path (Strategy A).
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      gray16_to_rgba_u16_row(y_plane, rgba_u16_row, w, use_simd, full_range);
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
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      gray16_to_rgb_u16_row(y_plane, rgb_u16_row, w, use_simd, full_range);
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // u8 RGB / RGBA / HSV (Strategy A).
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    // Only need the RGB kernel when an RGB output is requested, or when both
    // HSV and at least one u8 RGB/RGBA output are requested simultaneously.
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    // Standalone RGBA fast path (no RGB or HSV output needed).
    if want_rgba && !need_rgb_kernel && !want_hsv {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gray16_to_rgba_row(y_plane, rgba_row, w, use_simd, full_range);
      return Ok(());
    }

    // Standalone HSV fast path — gray sources always have H=0, S=0, V=Y>>8.
    // Skip RGB scratch entirely when only HSV (and optionally RGBA) is needed.
    if want_hsv && !want_rgb {
      let hsv = hsv.as_mut().unwrap();
      gray16_to_hsv_row(
        y_plane,
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
        full_range,
      );
      if let Some(buf) = rgba.as_deref_mut() {
        let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
        gray16_to_rgba_row(y_plane, rgba_row, w, use_simd, full_range);
      }
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
    gray16_to_rgb_row(y_plane, rgb_row, w, use_simd, full_range);

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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Integration tests -------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use crate::{
    ColorMatrix,
    frame::{Gray8Frame, Gray16Frame, GrayNFrame},
    sinker::MixedSinker,
    yuv::{gray8_to, gray9_to, gray10_to, gray12_to, gray14_to, gray16_to},
  };

  // Gray formats are luma-only; full_range and matrix are unused by the kernels
  // but are required by the walker signature. Use full_range=true, Bt709.
  const FR: bool = true;
  const M: ColorMatrix = ColorMatrix::Bt709;

  fn make_gray8_frame(data: &[u8], w: u32, h: u32) -> Gray8Frame<'_> {
    Gray8Frame::new(data, w, h, w)
  }
  fn make_gray10_frame(data: &[u16], w: u32, h: u32) -> GrayNFrame<'_, 10> {
    GrayNFrame::new(data, w, h, w)
  }
  fn make_gray16_frame(data: &[u16], w: u32, h: u32) -> Gray16Frame<'_> {
    Gray16Frame::new(data, w, h, w)
  }

  #[test]
  fn gray8_with_rgb_broadcasts_to_packed() {
    // 4×1 frame: [0, 64, 128, 255]
    let plane = [0u8, 64, 128, 255];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut rgb = std::vec![0u8; 4 * 3];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray8_to(&frame, FR, M, &mut sink).unwrap();
    // Each pixel should be [Y, Y, Y]
    assert_eq!(rgb[0..3], [0, 0, 0]);
    assert_eq!(rgb[3..6], [64, 64, 64]);
    assert_eq!(rgb[6..9], [128, 128, 128]);
    assert_eq!(rgb[9..12], [255, 255, 255]);
  }

  #[test]
  fn gray8_with_rgba_alpha_is_0xff() {
    let plane = [100u8; 4];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut rgba = std::vec![0u8; 4 * 4];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_rgba(&mut rgba)
      .unwrap();
    gray8_to(&frame, FR, M, &mut sink).unwrap();
    // Alpha byte (index 3, 7, 11, 15) should be 0xFF.
    for i in 0..4 {
      assert_eq!(rgba[i * 4 + 3], 0xFF, "pixel {i} alpha");
      assert_eq!(rgba[i * 4], 100, "pixel {i} R");
    }
  }

  #[test]
  fn gray8_with_luma_copies_plane() {
    let plane: Vec<u8> = (0..16u8).collect();
    let frame = make_gray8_frame(&plane, 4, 4);
    let mut luma = std::vec![0u8; 16];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 4)
      .with_luma(&mut luma)
      .unwrap();
    gray8_to(&frame, FR, M, &mut sink).unwrap();
    assert_eq!(luma, plane);
  }

  #[test]
  fn gray8_with_luma_u16_zero_extends() {
    let plane = [0u8, 64, 128, 255];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut lu16 = std::vec![0u16; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_luma_u16(&mut lu16)
      .unwrap();
    gray8_to(&frame, FR, M, &mut sink).unwrap();
    assert_eq!(lu16, [0, 64, 128, 255]);
  }

  #[test]
  fn gray8_with_hsv_h_s_zero_v_equals_y() {
    let plane = [50u8, 100, 200, 0];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut h = std::vec![0xFFu8; 4];
    let mut s = std::vec![0xFFu8; 4];
    let mut v = std::vec![0u8; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    gray8_to(&frame, FR, M, &mut sink).unwrap();
    assert_eq!(h, [0, 0, 0, 0], "H must be 0");
    assert_eq!(s, [0, 0, 0, 0], "S must be 0");
    assert_eq!(v, plane.as_slice(), "V must equal Y");
  }

  #[test]
  fn gray10_with_rgb_masks_and_shifts() {
    // 10-bit sample: value 512 = 0b10_0000_0000, masked = 512, >> 2 = 128
    let plane = [512u16; 4];
    let frame = make_gray10_frame(&plane, 4, 1);
    let mut rgb = std::vec![0u8; 12];
    let mut sink = MixedSinker::<crate::yuv::Gray10>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray10_to(&frame, FR, M, &mut sink).unwrap();
    // 512 & 0x3FF = 512, >> 2 = 128. All channels should be 128.
    assert_eq!(rgb[0..3], [128, 128, 128]);
    assert_eq!(rgb[3..6], [128, 128, 128]);
  }

  #[test]
  fn gray10_with_luma_u16_masks_only() {
    // 10-bit, over-range sample: 0x0800 (bit 11 set) masked → 0.
    let plane = [0x0800u16, 0x03FFu16, 0x0200u16, 0x0001u16];
    let frame = make_gray10_frame(&plane, 4, 1);
    let mut lu16 = std::vec![0u16; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray10>::new(4, 1)
      .with_luma_u16(&mut lu16)
      .unwrap();
    gray10_to(&frame, FR, M, &mut sink).unwrap();
    assert_eq!(lu16, [0x0000, 0x03FF, 0x0200, 0x0001]);
  }

  #[test]
  fn gray16_with_rgb_shifts_to_u8() {
    // Gray16 sample 0x8000 → >> 8 = 0x80 = 128.
    let plane = [0x8000u16, 0xFFFFu16, 0x0000u16, 0x0100u16];
    let frame = make_gray16_frame(&plane, 4, 1);
    let mut rgb = std::vec![0u8; 12];
    let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray16_to(&frame, FR, M, &mut sink).unwrap();
    // Each pixel [Y>>8, Y>>8, Y>>8]
    assert_eq!(rgb[0..3], [0x80, 0x80, 0x80]);
    assert_eq!(rgb[3..6], [0xFF, 0xFF, 0xFF]);
    assert_eq!(rgb[6..9], [0x00, 0x00, 0x00]);
    assert_eq!(rgb[9..12], [0x01, 0x01, 0x01]);
  }

  #[test]
  fn gray16_with_luma_u16_copies_plane() {
    let plane: Vec<u16> = (0u16..16).map(|x| x * 4096).collect();
    let frame = make_gray16_frame(&plane, 4, 4);
    let mut lu16 = std::vec![0u16; 16];
    let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 4)
      .with_luma_u16(&mut lu16)
      .unwrap();
    gray16_to(&frame, FR, M, &mut sink).unwrap();
    assert_eq!(lu16, plane);
  }

  #[test]
  fn gray16_with_rgba_u16_alpha_is_0xffff() {
    let plane = [0x1234u16; 4];
    let frame = make_gray16_frame(&plane, 4, 1);
    let mut rgba_u16 = std::vec![0u16; 16];
    let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
      .with_rgba_u16(&mut rgba_u16)
      .unwrap();
    gray16_to(&frame, FR, M, &mut sink).unwrap();
    for i in 0..4 {
      assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF, "pixel {i} alpha");
      assert_eq!(rgba_u16[i * 4], 0x1234, "pixel {i} R");
    }
  }

  #[test]
  fn gray9_walker_smoke_test() {
    use crate::frame::GrayNFrame;
    let plane = [100u16; 4];
    let frame: GrayNFrame<'_, 9> = GrayNFrame::new(&plane, 4, 1, 4);
    let mut luma = std::vec![0u8; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray9>::new(4, 1)
      .with_luma(&mut luma)
      .unwrap();
    gray9_to(&frame, FR, M, &mut sink).unwrap();
    // 100 & 0x1FF = 100, >> 1 = 50.
    assert_eq!(luma, [50, 50, 50, 50]);
  }

  #[test]
  fn gray12_walker_smoke_test() {
    use crate::frame::GrayNFrame;
    let plane = [0x0FFFu16; 4];
    let frame: GrayNFrame<'_, 12> = GrayNFrame::new(&plane, 4, 1, 4);
    let mut luma = std::vec![0u8; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray12>::new(4, 1)
      .with_luma(&mut luma)
      .unwrap();
    gray12_to(&frame, FR, M, &mut sink).unwrap();
    // 0x0FFF & 0x0FFF = 0x0FFF = 4095. >> 4 = 255.
    assert_eq!(luma, [255, 255, 255, 255]);
  }

  #[test]
  fn gray14_walker_smoke_test() {
    use crate::frame::GrayNFrame;
    let plane = [0x3FFFu16; 4];
    let frame: GrayNFrame<'_, 14> = GrayNFrame::new(&plane, 4, 1, 4);
    let mut luma = std::vec![0u8; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray14>::new(4, 1)
      .with_luma(&mut luma)
      .unwrap();
    gray14_to(&frame, FR, M, &mut sink).unwrap();
    // 0x3FFF & 0x3FFF = 0x3FFF = 16383. >> 6 = 255.
    assert_eq!(luma, [255, 255, 255, 255]);
  }

  // ---- Limited-range integration tests ----------------------------------------
  //
  // For 8-bit limited-range: black=16, white=235, range=219.
  //   rescale(y) = clamp_u8(((y - 16) * 255 + 109) / 219)
  // For N-bit limited-range: black = 16 << (N-8), range = 219 << (N-8).
  //   rescale(y) = clamp_u8(((y - black) * 255 + range/2) / range)
  // Luma outputs always pass raw Y through (no rescaling regardless of
  // full_range).

  #[test]
  fn gray8_limited_range_black_maps_to_zero() {
    // Y=16 (limited-range black) → RGB(0, 0, 0).
    let plane = [16u8; 4];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut rgb = std::vec![0xFFu8; 12];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray8_to(&frame, false, M, &mut sink).unwrap();
    for i in 0..4 {
      assert_eq!(rgb[i * 3..i * 3 + 3], [0, 0, 0], "pixel {i}");
    }
  }

  #[test]
  fn gray8_limited_range_white_maps_to_255() {
    // Y=235 (limited-range white) → RGB(255, 255, 255).
    let plane = [235u8; 4];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut rgb = std::vec![0u8; 12];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray8_to(&frame, false, M, &mut sink).unwrap();
    for i in 0..4 {
      assert_eq!(rgb[i * 3..i * 3 + 3], [255, 255, 255], "pixel {i}");
    }
  }

  #[test]
  fn gray8_limited_range_midpoint() {
    // Y=125 → ((125-16)*255+109)/219 = 27904/219 = 127.
    let plane = [125u8; 4];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut rgb = std::vec![0u8; 12];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray8_to(&frame, false, M, &mut sink).unwrap();
    for i in 0..4 {
      assert_eq!(rgb[i * 3], 127, "pixel {i} R");
    }
  }

  #[test]
  fn gray8_limited_range_luma_passthrough_unchanged() {
    // Luma output must pass raw Y through even for limited-range; no rescaling.
    let plane = [16u8, 235u8, 125u8, 0u8];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut luma = std::vec![0xAAu8; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_luma(&mut luma)
      .unwrap();
    gray8_to(&frame, false, M, &mut sink).unwrap();
    assert_eq!(luma, [16, 235, 125, 0]);
  }

  #[test]
  fn gray8_limited_range_rgba_alpha_is_0xff() {
    // Verify limited-range RGBA: alpha=0xFF, channels rescaled.
    let plane = [235u8; 4];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut rgba = std::vec![0u8; 16];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_rgba(&mut rgba)
      .unwrap();
    gray8_to(&frame, false, M, &mut sink).unwrap();
    for i in 0..4 {
      assert_eq!(rgba[i * 4], 255, "pixel {i} R");
      assert_eq!(rgba[i * 4 + 3], 0xFF, "pixel {i} alpha");
    }
  }

  #[test]
  fn gray8_limited_range_hsv_v_is_rescaled() {
    // HSV V channel must use rescaled Y in limited-range mode.
    let plane = [235u8; 4];
    let frame = make_gray8_frame(&plane, 4, 1);
    let mut h = std::vec![0xFFu8; 4];
    let mut s = std::vec![0xFFu8; 4];
    let mut v = std::vec![0u8; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    gray8_to(&frame, false, M, &mut sink).unwrap();
    assert_eq!(h, [0, 0, 0, 0], "H must be 0");
    assert_eq!(s, [0, 0, 0, 0], "S must be 0");
    assert_eq!(v, [255, 255, 255, 255], "V must be 255 for white");
  }

  #[test]
  fn gray10_limited_range_black_and_white() {
    use crate::frame::GrayNFrame;
    // 10-bit: black=64, white=940, range=876.
    let plane = [64u16, 940, 64, 940];
    let frame: GrayNFrame<'_, 10> = GrayNFrame::new(&plane, 4, 1, 4);
    let mut rgb = std::vec![0x80u8; 12];
    let mut sink = MixedSinker::<crate::yuv::Gray10>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray10_to(&frame, false, M, &mut sink).unwrap();
    assert_eq!(rgb[0..3], [0, 0, 0], "Y=64 → black");
    assert_eq!(rgb[3..6], [255, 255, 255], "Y=940 → white");
    assert_eq!(rgb[6..9], [0, 0, 0], "Y=64 → black");
    assert_eq!(rgb[9..12], [255, 255, 255], "Y=940 → white");
  }

  #[test]
  fn gray12_limited_range_black_and_white() {
    use crate::frame::GrayNFrame;
    // 12-bit: black=256, white=3760, range=3504.
    let plane = [256u16, 3760, 256, 3760];
    let frame: GrayNFrame<'_, 12> = GrayNFrame::new(&plane, 4, 1, 4);
    let mut rgb = std::vec![0x80u8; 12];
    let mut sink = MixedSinker::<crate::yuv::Gray12>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray12_to(&frame, false, M, &mut sink).unwrap();
    assert_eq!(rgb[0..3], [0, 0, 0], "Y=256 → black");
    assert_eq!(rgb[3..6], [255, 255, 255], "Y=3760 → white");
    assert_eq!(rgb[6..9], [0, 0, 0], "Y=256 → black");
    assert_eq!(rgb[9..12], [255, 255, 255], "Y=3760 → white");
  }

  #[test]
  fn gray14_limited_range_black_and_white() {
    use crate::frame::GrayNFrame;
    // 14-bit: black=1024, white=15040, range=14016.
    let plane = [1024u16, 15040, 1024, 15040];
    let frame: GrayNFrame<'_, 14> = GrayNFrame::new(&plane, 4, 1, 4);
    let mut rgb = std::vec![0x80u8; 12];
    let mut sink = MixedSinker::<crate::yuv::Gray14>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray14_to(&frame, false, M, &mut sink).unwrap();
    assert_eq!(rgb[0..3], [0, 0, 0], "Y=1024 → black");
    assert_eq!(rgb[3..6], [255, 255, 255], "Y=15040 → white");
    assert_eq!(rgb[6..9], [0, 0, 0], "Y=1024 → black");
    assert_eq!(rgb[9..12], [255, 255, 255], "Y=15040 → white");
  }

  #[test]
  fn gray16_limited_range_black_and_white() {
    // 16-bit: black=4096, white=60160, range=56064.
    let plane = [4096u16, 60160, 4096, 60160];
    let frame = make_gray16_frame(&plane, 4, 1);
    let mut rgb = std::vec![0x80u8; 12];
    let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
      .with_rgb(&mut rgb)
      .unwrap();
    gray16_to(&frame, false, M, &mut sink).unwrap();
    assert_eq!(rgb[0..3], [0, 0, 0], "Y=4096 → black");
    assert_eq!(rgb[3..6], [255, 255, 255], "Y=60160 → white");
    assert_eq!(rgb[6..9], [0, 0, 0], "Y=4096 → black");
    assert_eq!(rgb[9..12], [255, 255, 255], "Y=60160 → white");
  }

  #[test]
  fn gray16_limited_range_luma_passthrough_unchanged() {
    // Luma u16 must copy raw Y regardless of full_range.
    let plane = [4096u16, 60160, 32768, 0];
    let frame = make_gray16_frame(&plane, 4, 1);
    let mut lu16 = std::vec![0u16; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
      .with_luma_u16(&mut lu16)
      .unwrap();
    gray16_to(&frame, false, M, &mut sink).unwrap();
    assert_eq!(lu16, [4096, 60160, 32768, 0]);
  }

  #[test]
  fn gray16_limited_range_rgba_u16_alpha_is_0xffff() {
    // RGBA u16 — alpha=0xFFFF; channels hold the native Y broadcast.
    // In limited-range the u16 RGB path passes native Y through (no >>8).
    let plane = [4096u16; 4];
    let frame = make_gray16_frame(&plane, 4, 1);
    let mut rgba_u16 = std::vec![0u16; 16];
    let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
      .with_rgba_u16(&mut rgba_u16)
      .unwrap();
    gray16_to(&frame, false, M, &mut sink).unwrap();
    for i in 0..4 {
      assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF, "pixel {i} alpha");
    }
  }

  #[test]
  fn gray16_limited_range_rgba_u16_channels_rescale_at_boundaries() {
    // Regression for the i32-overflow bug at BITS=16: limited-range white
    // 60160 × max_native 65535 ≈ 3.67e9 overflows i32. Math runs in i64;
    // assert that RGB channels reach black=0 and white=65535 at the
    // limited-range boundaries (codex finding requested
    // u16-channel-value asserts, not only alpha).
    let plane = [4096u16, 60160u16, 65535u16, 0u16];
    let frame = make_gray16_frame(&plane, 4, 1);
    let mut rgba_u16 = std::vec![0u16; 16];
    let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
      .with_rgba_u16(&mut rgba_u16)
      .unwrap();
    gray16_to(&frame, false, M, &mut sink).unwrap();
    // pixel 0: limited black 4096 → 0
    assert_eq!(&rgba_u16[0..3], &[0, 0, 0]);
    // pixel 1: limited white 60160 → 65535 (over-i32 path)
    assert_eq!(&rgba_u16[4..7], &[65535, 65535, 65535]);
    // pixel 2: over-white 65535 → clamped to 65535
    assert_eq!(&rgba_u16[8..11], &[65535, 65535, 65535]);
    // pixel 3: below-black 0 → clamped to 0
    assert_eq!(&rgba_u16[12..15], &[0, 0, 0]);
    // alpha unchanged
    for i in 0..4 {
      assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF);
    }
  }

  #[test]
  fn gray16_limited_range_rgb_u16_channels_rescale_at_boundaries() {
    // Same i32-overflow regression on the with_rgb_u16 path.
    let plane = [4096u16, 60160u16];
    let frame = make_gray16_frame(&plane, 2, 1);
    let mut rgb_u16 = std::vec![0u16; 6];
    let mut sink = MixedSinker::<crate::yuv::Gray16>::new(2, 1)
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
    gray16_to(&frame, false, M, &mut sink).unwrap();
    assert_eq!(&rgb_u16[0..3], &[0, 0, 0]);
    assert_eq!(&rgb_u16[3..6], &[65535, 65535, 65535]);
  }

  #[test]
  fn gray16_limited_range_hsv_v_is_rescaled() {
    // HSV V must reflect limited-range rescaling.
    let plane = [60160u16; 4]; // white
    let frame = make_gray16_frame(&plane, 4, 1);
    let mut h = std::vec![0xFFu8; 4];
    let mut s = std::vec![0xFFu8; 4];
    let mut v = std::vec![0u8; 4];
    let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    gray16_to(&frame, false, M, &mut sink).unwrap();
    assert_eq!(h, [0, 0, 0, 0], "H must be 0");
    assert_eq!(s, [0, 0, 0, 0], "S must be 0");
    assert_eq!(v, [255, 255, 255, 255], "V must be 255 for white");
  }
}
