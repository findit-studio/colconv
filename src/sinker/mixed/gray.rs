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
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, gray_n_to_hsv_row, gray_n_to_luma_row,
    gray_n_to_luma_u16_row, gray_n_to_rgb_row, gray_n_to_rgb_u16_row, gray_n_to_rgba_row,
    gray_n_to_rgba_u16_row, gray8_to_hsv_row, gray8_to_rgb_row, gray8_to_rgba_row,
    gray16_to_hsv_row, gray16_to_luma_row, gray16_to_luma_u16_row, gray16_to_rgb_row,
    gray16_to_rgb_u16_row, gray16_to_rgba_row, gray16_to_rgba_u16_row, grayf32_to_hsv_row,
    grayf32_to_luma_f32_row, grayf32_to_luma_row, grayf32_to_luma_u16_row, grayf32_to_rgb_f32_row,
    grayf32_to_rgb_row, grayf32_to_rgb_u16_row, grayf32_to_rgba_row, grayf32_to_rgba_u16_row,
    rgb_to_hsv_row,
    scalar::alpha_extract::{copy_alpha_ya_u8, copy_alpha_ya_u16, copy_alpha_ya_u16_to_u8},
    y_plane_to_luma_u16_row, ya8_to_hsv_row, ya8_to_luma_row, ya8_to_luma_u16_row, ya8_to_rgb_row,
    ya8_to_rgb_u16_row, ya8_to_rgba_row, ya8_to_rgba_u16_row, ya16_to_hsv_row, ya16_to_luma_row,
    ya16_to_luma_u16_row, ya16_to_rgb_row, ya16_to_rgb_u16_row, ya16_to_rgba_row,
    ya16_to_rgba_u16_row,
  },
  source::{
    Gray8, Gray8Row, Gray8Sink, Gray16, Gray16Row, Gray16Sink, Grayf32, Grayf32Row, Grayf32Sink,
    Ya8, Ya8Row, Ya8Sink, Ya16, Ya16Row, Ya16Sink,
  },
};

// ---- Gray8 impl -------------------------------------------------------------

impl<'a, R> MixedSinker<'a, Gray8, R> {
  /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`
  /// (Gray8 has no alpha channel).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if `buf.len() < width x height x 4`,
  /// or `Err(GeometryOverflow)` on 32-bit overflow.
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

  /// Attaches a u16 luma output buffer. Gray8 Y bytes are zero-extended
  /// to u16 (each output element equals `y_byte as u16`). Length measured
  /// in `u16` elements (`width x height`).
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
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
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
      let (h, s, v) = hsv.hsv();
      gray8_to_hsv_row(
        y_plane,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
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
fn process_gray_n<'a, const BITS: u32, const BE: bool>(
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
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'a>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
) -> Result<(), MixedSinkerError> {
  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  // Luma u8 — always passes raw Y through, no full_range rescaling.
  if let Some(buf) = luma.as_deref_mut() {
    gray_n_to_luma_row::<BITS, BE>(
      y_plane,
      &mut buf[one_plane_start..one_plane_end],
      w,
      use_simd,
    );
  }

  // Luma u16 — always passes raw Y through, no full_range rescaling.
  if let Some(buf) = luma_u16.as_deref_mut() {
    gray_n_to_luma_u16_row::<BITS, BE>(
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
    gray_n_to_rgba_u16_row::<BITS, BE>(y_plane, rgba_u16_row, w, use_simd, full_range);
  } else if want_rgb_u16 {
    let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
    let rgb_plane_start = one_plane_start * 3;
    let rgb_plane_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        w, h, 3,
      )))?;
    let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
    gray_n_to_rgb_u16_row::<BITS, BE>(y_plane, rgb_u16_row, w, use_simd, full_range);
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
    gray_n_to_rgba_row::<BITS, BE>(y_plane, rgba_row, w, use_simd, full_range);
    return Ok(());
  }

  // Standalone HSV fast path — gray sources always have H=0, S=0, V=Y8
  // (rescaled if limited-range).
  if want_hsv && !want_rgb && !want_rgba {
    let hsv = hsv.as_mut().unwrap();
    let (h, s, v) = hsv.hsv();
    gray_n_to_hsv_row::<BITS, BE>(
      y_plane,
      &mut h[one_plane_start..one_plane_end],
      &mut s[one_plane_start..one_plane_end],
      &mut v[one_plane_start..one_plane_end],
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
  gray_n_to_rgb_row::<BITS, BE>(y_plane, rgb_row, w, use_simd, full_range);

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
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      RowSlice::Y,
      idx,
      w,
      y_len,
    )));
  }
  if idx >= h {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, h),
    ));
  }
  Ok(())
}

// ---- Per-bit-depth builder impls for GrayN ----------------------------------

macro_rules! impl_gray_n_sinker {
  ($marker:ident, $row:ident, $sink:ident, $bits:expr) => {
    impl<'a, R, const BE: bool> MixedSinker<'a, $marker<BE>, R> {
      /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`.
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

      /// Attaches a u16 RGB output buffer. Samples are masked to the low
      /// `BITS` bits; length is in `u16` elements (`width x height x 3`).
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

      /// Attaches a u16 RGBA output buffer. Samples masked to low `BITS` bits;
      /// alpha = `(1 << BITS) - 1` (full-range opaque). Length in `u16` elements
      /// (`width x height x 4`).
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

      /// Attaches a u16 luma output buffer. Samples masked to low `BITS`
      /// bits; length in `u16` elements (`width x height`).
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

    impl<const BE: bool> $sink<BE> for MixedSinker<'_, $marker<BE>> {}

    impl<const BE: bool> PixelSink for MixedSinker<'_, $marker<BE>> {
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
        process_gray_n::<$bits, BE>(
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
use crate::source::{
  Gray9, Gray9Row, Gray9Sink, Gray10, Gray10Row, Gray10Sink, Gray12, Gray12Row, Gray12Sink, Gray14,
  Gray14Row, Gray14Sink,
};

impl_gray_n_sinker!(Gray9, Gray9Row, Gray9Sink, 9);
impl_gray_n_sinker!(Gray10, Gray10Row, Gray10Sink, 10);
impl_gray_n_sinker!(Gray12, Gray12Row, Gray12Sink, 12);
impl_gray_n_sinker!(Gray14, Gray14Row, Gray14Sink, 14);

// ---- Gray16 impl ------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Gray16<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. Alpha is forced to `0xFF`.
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

  /// Attaches a u16 RGB output buffer (`>> 8` is NOT applied — native
  /// 16-bit broadcast). Length in `u16` elements (`width x height x 3`).
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

  /// Attaches a u16 RGBA output buffer (native 16-bit broadcast; alpha
  /// = `0xFFFF`). Length in `u16` elements (`width x height x 4`).
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

  /// Attaches a u16 luma output buffer (identity copy of the Gray16 Y
  /// plane). Length in `u16` elements (`width x height`).
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

impl<const BE: bool> Gray16Sink<BE> for MixedSinker<'_, Gray16<BE>> {}

impl<const BE: bool> PixelSink for MixedSinker<'_, Gray16<BE>> {
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
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
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
      ..
    } = self;
    let y_plane = row.y();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma u8 — shift >> 8.
    if let Some(buf) = luma.as_deref_mut() {
      gray16_to_luma_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Luma u16 — identity copy.
    if let Some(buf) = luma_u16.as_deref_mut() {
      gray16_to_luma_u16_row::<BE>(
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
      gray16_to_rgba_u16_row::<BE>(y_plane, rgba_u16_row, w, use_simd, full_range);
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      gray16_to_rgb_u16_row::<BE>(y_plane, rgb_u16_row, w, use_simd, full_range);
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
      gray16_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd, full_range);
      return Ok(());
    }

    // Standalone HSV fast path — gray sources always have H=0, S=0, V=Y>>8.
    // Skip RGB scratch entirely when only HSV (and optionally RGBA) is needed.
    if want_hsv && !want_rgb {
      let hsv = hsv.as_mut().unwrap();
      let (hp, sp, vp) = hsv.hsv();
      gray16_to_hsv_row::<BE>(
        y_plane,
        &mut hp[one_plane_start..one_plane_end],
        &mut sp[one_plane_start..one_plane_end],
        &mut vp[one_plane_start..one_plane_end],
        w,
        use_simd,
        full_range,
      );
      if let Some(buf) = rgba.as_deref_mut() {
        let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
        gray16_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd, full_range);
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
    gray16_to_rgb_row::<BE>(y_plane, rgb_row, w, use_simd, full_range);

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

    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Grayf32 impl -----------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Grayf32<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. α is forced to `0xFF`
  /// (Grayf32 has no alpha channel).
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

  /// Attaches a u16 RGB output buffer.
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

  /// Attaches a u16 RGBA output buffer. α = `0xFFFF`.
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

  /// Attaches a u16 luma output buffer (`clamp(Y,0,1) x 65535`).
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

  /// Attaches a packed f32 RGB output buffer. Lossless replicate of Y → R=G=B.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f32`](Self::with_rgb_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientRgbF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.rgb_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches an f32 luma output buffer. Lossless pass-through of Y.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_luma_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_f32`](Self::with_luma_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_elems(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::InsufficientLumaF32Buffer(
        InsufficientBuffer::new(expected, buf.len()),
      ));
    }
    self.luma_f32 = Some(buf);
    Ok(self)
  }
}

impl<const BE: bool> Grayf32Sink<BE> for MixedSinker<'_, Grayf32<BE>> {}

impl<const BE: bool> PixelSink for MixedSinker<'_, Grayf32<BE>> {
  type Input<'r> = Grayf32Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Grayf32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    let y_plane = row.y();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // luma f32 pass-through — highest priority (no clamp, no round).
    if let Some(buf) = self.luma_f32.as_deref_mut() {
      grayf32_to_luma_f32_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // rgb_f32 — lossless replicate Y → R=G=B.
    if let Some(buf) = self.rgb_f32.as_deref_mut() {
      let rgb_f32_start = one_plane_start * 3;
      let rgb_f32_end = one_plane_end
        .checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 3,
        )))?;
      grayf32_to_rgb_f32_row::<BE>(y_plane, &mut buf[rgb_f32_start..rgb_f32_end], w, use_simd);
    }

    // luma u8.
    if let Some(buf) = self.luma.as_deref_mut() {
      grayf32_to_luma_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // luma u16.
    if let Some(buf) = self.luma_u16.as_deref_mut() {
      grayf32_to_luma_u16_row::<BE>(
        y_plane,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path (Strategy A).
    let want_rgb_u16 = self.rgb_u16.is_some();
    let want_rgba_u16 = self.rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = self.rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      grayf32_to_rgba_u16_row::<BE>(y_plane, rgba_u16_row, w, use_simd);
    } else if want_rgb_u16 {
      let rgb_u16_buf = self.rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      grayf32_to_rgb_u16_row::<BE>(y_plane, rgb_u16_row, w, use_simd);
      if want_rgba_u16 {
        let rgba_u16_buf = self.rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
      }
    }

    // u8 RGB / RGBA / HSV path.
    let want_rgb = self.rgb.is_some();
    let want_rgba = self.rgba.is_some();
    let want_hsv = self.hsv.is_some();

    // Standalone RGBA fast path.
    if want_rgba && !want_rgb && !want_hsv {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      grayf32_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone HSV fast path — Grayf32 always has H=0, S=0, V=clamp(Y)x255.
    if want_hsv && !want_rgb {
      let hsv = self.hsv.as_mut().unwrap();
      let (hp, sp, vp) = hsv.hsv();
      grayf32_to_hsv_row::<BE>(
        y_plane,
        &mut hp[one_plane_start..one_plane_end],
        &mut sp[one_plane_start..one_plane_end],
        &mut vp[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
      if let Some(buf) = self.rgba.as_deref_mut() {
        let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
        grayf32_to_rgba_row::<BE>(y_plane, rgba_row, w, use_simd);
      }
      return Ok(());
    }

    if !want_rgb && !want_rgba && !want_hsv {
      return Ok(());
    }

    let rgb_row = rgb_row_buf_or_scratch(
      self.rgb.as_deref_mut(),
      &mut self.rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    grayf32_to_rgb_row::<BE>(y_plane, rgb_row, w, use_simd);

    if let Some(hsv) = self.hsv.as_mut() {
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

    if let Some(buf) = self.rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Ya8 impl ---------------------------------------------------------------

impl<'a, R> MixedSinker<'a, Ya8, R> {
  /// Attaches an 8-bit RGBA output buffer. α is passed from the source.
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

  /// Attaches a u16 RGB output buffer.
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

  /// Attaches a u16 RGBA output buffer. α zero-extended from source.
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

  /// Attaches a u16 luma output buffer (zero-extend Y → u16).
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

impl Ya8Sink for MixedSinker<'_, Ya8> {}

impl PixelSink for MixedSinker<'_, Ya8> {
  type Input<'r> = Ya8Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Ya8Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    let packed = row.packed(); // &[u8], length = width * 2

    if packed.len() != w * 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w * 2,
        packed.len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // luma u8.
    if let Some(buf) = self.luma.as_deref_mut() {
      ya8_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // luma u16.
    if let Some(buf) = self.luma_u16.as_deref_mut() {
      ya8_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path. Each path is independent (α is embedded in ya8_to_rgba_u16_row).
    if let Some(buf) = self.rgb_u16.as_deref_mut() {
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      ya8_to_rgb_u16_row(
        packed,
        &mut buf[rgb_plane_start..rgb_plane_end],
        w,
        use_simd,
      );
    }
    if let Some(buf) = self.rgba_u16.as_deref_mut() {
      let rgba_u16_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      ya8_to_rgba_u16_row(packed, rgba_u16_row, w, use_simd);
    }

    // u8 RGB / RGBA / HSV path. Strategy A+: rgb first, then copy α into rgba.
    let want_rgb = self.rgb.is_some();
    let want_rgba = self.rgba.is_some();
    let want_hsv = self.hsv.is_some();

    // Standalone RGBA fast path (no RGB or HSV).
    if want_rgba && !want_rgb && !want_hsv {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      ya8_to_rgba_row(packed, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone HSV fast path.
    if want_hsv && !want_rgb && !want_rgba {
      let hsv = self.hsv.as_mut().unwrap();
      let (h, s, v) = hsv.hsv();
      ya8_to_hsv_row(
        packed,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
      return Ok(());
    }

    if !want_rgb && !want_rgba && !want_hsv {
      return Ok(());
    }

    // RGB kernel (used for HSV + Strategy A+ fan-out).
    let rgb_row = rgb_row_buf_or_scratch(
      self.rgb.as_deref_mut(),
      &mut self.rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    ya8_to_rgb_row(packed, rgb_row, w, use_simd);

    if let Some(hsv) = self.hsv.as_mut() {
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

    // Strategy A+: expand RGB→RGBA then patch α from source.
    if let Some(buf) = self.rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      // Overwrite the α channel with real source α.
      copy_alpha_ya_u8(packed, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Ya16 impl --------------------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Ya16<BE>, R> {
  /// Attaches an 8-bit RGBA output buffer. α is `source_A >> 8`.
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

  /// Attaches a u16 RGB output buffer.
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

  /// Attaches a u16 RGBA output buffer. α from source (native u16).
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

  /// Attaches a u16 luma output buffer (native pass-through).
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

impl<const BE: bool> Ya16Sink<BE> for MixedSinker<'_, Ya16<BE>> {}

impl<const BE: bool> PixelSink for MixedSinker<'_, Ya16<BE>> {
  type Input<'r> = Ya16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Ya16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;
    let packed = row.packed(); // &[u16], length = width * 2

    if packed.len() != w * 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w * 2,
        packed.len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // luma u8 — `Y >> 8`.
    if let Some(buf) = self.luma.as_deref_mut() {
      ya16_to_luma_row::<BE>(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // luma u16 — native pass-through.
    if let Some(buf) = self.luma_u16.as_deref_mut() {
      ya16_to_luma_u16_row::<BE>(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // u16 RGB / RGBA path. Strategy A + α-patch for RGBA.
    let want_rgb_u16 = self.rgb_u16.is_some();
    let want_rgba_u16 = self.rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = self.rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      ya16_to_rgba_u16_row::<BE>(packed, rgba_u16_row, w, use_simd);
    } else if want_rgb_u16 {
      let rgb_u16_buf = self.rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_start = one_plane_start * 3;
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      ya16_to_rgb_u16_row::<BE>(packed, rgb_u16_row, w, use_simd);
      if want_rgba_u16 {
        let rgba_u16_buf = self.rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        // Patch α from source (native u16 depth). `BE` is propagated from
        // the parent `Ya16Frame<'_, BE>` so the loader byte-swaps correctly
        // for both LE and BE inputs.
        copy_alpha_ya_u16::<BE>(packed, rgba_u16_row, w);
      }
    }

    // u8 RGB / RGBA / HSV path. Strategy A+: rgb first, then copy α into rgba.
    let want_rgb = self.rgb.is_some();
    let want_rgba = self.rgba.is_some();
    let want_hsv = self.hsv.is_some();

    // Standalone RGBA fast path.
    if want_rgba && !want_rgb && !want_hsv {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      ya16_to_rgba_row::<BE>(packed, rgba_row, w, use_simd);
      return Ok(());
    }

    // Standalone HSV fast path.
    if want_hsv && !want_rgb && !want_rgba {
      let hsv = self.hsv.as_mut().unwrap();
      let (h, s, v) = hsv.hsv();
      ya16_to_hsv_row::<BE>(
        packed,
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
      return Ok(());
    }

    if !want_rgb && !want_rgba && !want_hsv {
      return Ok(());
    }

    // RGB kernel.
    let rgb_row = rgb_row_buf_or_scratch(
      self.rgb.as_deref_mut(),
      &mut self.rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    ya16_to_rgb_row::<BE>(packed, rgb_row, w, use_simd);

    if let Some(hsv) = self.hsv.as_mut() {
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

    // Strategy A+: expand RGB→RGBA then patch α from source.
    if let Some(buf) = self.rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      // Overwrite the α channel with real source α (>> 8 for u8 output).
      // `BE` is propagated from the parent `Ya16Frame<'_, BE>`.
      copy_alpha_ya_u16_to_u8::<BE>(packed, rgba_row, w);
    }

    Ok(())
  }
}
