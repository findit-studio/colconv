//! Sinker impls for legacy 16-bit packed-RGB **source** formats (Tier 7).
//!
//! Sources covered:
//! - [`Rgb565`] — bits [15:11]=R5, [10:5]=G6, [4:0]=B5 (FFmpeg `RGB565LE`).
//! - [`Bgr565`] — bits [15:11]=B5, [10:5]=G6, [4:0]=R5 (FFmpeg `BGR565LE`).
//! - [`Rgb555`] — bits [14:10]=R5, [9:5]=G5, [4:0]=B5; bit 15 unused.
//! - [`Bgr555`] — bits [14:10]=B5, [9:5]=G5, [4:0]=R5; bit 15 unused.
//! - [`Rgb444`] — bits [11:8]=R4, [7:4]=G4, [3:0]=B4; bits [15:12] unused.
//! - [`Bgr444`] — bits [11:8]=B4, [7:4]=G4, [3:0]=R4; bits [15:12] unused.
//!
//! All six sources have **no** source alpha. Outputs map to the sink's
//! standard channels:
//!
//! - `with_rgb` / `with_rgba` — expand channels to u8 via bit-replication
//!   (`(c5 << 3) | (c5 >> 2)` for 5-bit, `(c6 << 2) | (c6 >> 4)` for 6-bit,
//!   `(c4 << 4) | c4` for 4-bit); `with_rgba` forces α=`0xFF`.
//! - `with_rgb_u16` — native bit-width, low-bit aligned in `u16`; no expansion.
//!   Max values: R5=G6=31/63 (RGB565), R5=G5=B5=31 (RGB555), R4=G4=B4=15 (RGB444).
//! - `with_rgba_u16` — same native-precision channels + α=`0xFFFF`.
//! - `with_luma` — stages u8 RGB via `rgb_to_luma_row`.
//! - `with_luma_u16` — zero-extended u8 luma (same `[0, 255]` range) via
//!   `rgb_to_luma_u16_row`; no native luma precision exists for these formats.
//! - `with_hsv` — stages u8 RGB via `rgb_to_hsv_row`.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    bgr444_to_rgb_row, bgr444_to_rgb_u16_row, bgr444_to_rgba_row, bgr444_to_rgba_u16_row,
    bgr555_to_rgb_row, bgr555_to_rgb_u16_row, bgr555_to_rgba_row, bgr555_to_rgba_u16_row,
    bgr565_to_rgb_row, bgr565_to_rgb_u16_row, bgr565_to_rgba_row, bgr565_to_rgba_u16_row,
    rgb_to_hsv_row, rgb_to_luma_row, rgb_to_luma_u16_row, rgb444_to_rgb_row, rgb444_to_rgb_u16_row,
    rgb444_to_rgba_row, rgb444_to_rgba_u16_row, rgb555_to_rgb_row, rgb555_to_rgb_u16_row,
    rgb555_to_rgba_row, rgb555_to_rgba_u16_row, rgb565_to_rgb_row, rgb565_to_rgb_u16_row,
    rgb565_to_rgba_row, rgb565_to_rgba_u16_row,
  },
  source::{
    Bgr444, Bgr444Row, Bgr444Sink, Bgr555, Bgr555Row, Bgr555Sink, Bgr565, Bgr565Row, Bgr565Sink,
    Rgb444, Rgb444Row, Rgb444Sink, Rgb555, Rgb555Row, Rgb555Sink, Rgb565, Rgb565Row, Rgb565Sink,
  },
};

// ============================================================================
// Shared helpers — checked accessor for the u16 RGB plane row slice
// ============================================================================

/// Slice out a `3 * width` `u16` sub-range from a flat u16 RGB plane.
/// Returns `Err(GeometryOverflow)` on 32-bit targets if `one_plane_end × 3`
/// wraps `usize`.
#[inline(always)]
fn rgb_u16_plane_row_slice(
  buf: &mut [u16],
  one_plane_start: usize,
  one_plane_end: usize,
  width: usize,
  height: usize,
) -> Result<&mut [u16], MixedSinkerError> {
  let end = one_plane_end
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(width, height, 3)))?;
  let start = one_plane_start * 3;
  Ok(&mut buf[start..end])
}

// ============================================================================
// Macro: emit one complete sinker impl block for a legacy RGB format.
// ============================================================================
//
// Parameters:
//   $marker      — marker type (e.g. `Rgb565`)
//   $sink_trait  — Sink subtrait (e.g. `Rgb565Sink`)
//   $row_ty      — Row type (e.g. `Rgb565Row`)
//   $buf_field   — row accessor method (e.g. `rgb565`)
//   $row_slice   — `RowSlice` variant (e.g. `RowSlice::Rgb565Packed`)
//   $to_rgb      — rgb_row dispatcher fn
//   $to_rgba     — rgba_row dispatcher fn
//   $to_rgb_u16  — rgb_u16_row dispatcher fn
//   $to_rgba_u16 — rgba_u16_row dispatcher fn
macro_rules! impl_legacy_rgb_sinker {
  (
    marker:      $marker:ident,
    sink_trait:  $sink_trait:ident,
    row_ty:      $row_ty:ident,
    buf_field:   $buf_field:ident,
    row_slice:   $row_slice:expr,
    to_rgb:      $to_rgb:ident,
    to_rgba:     $to_rgba:ident,
    to_rgb_u16:  $to_rgb_u16:ident,
    to_rgba_u16: $to_rgba_u16:ident,
  ) => {
    // ---- per-format accessors ------------------------------------------------

    impl<'a> MixedSinker<'a, $marker> {
      /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled with
      /// constant `0xFF` (this source format has no alpha channel).
      ///
      /// Returns `Err(InsufficientRgbaBuffer)` if
      /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
      /// 32-bit targets when the product overflows.
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

      /// Attaches a **native-depth `u16`** RGB output buffer. Each channel is
      /// stored low-bit aligned at its native bit width — no expansion applied.
      /// Length is measured in `u16` elements (`width × height × 3`).
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

      /// Attaches a **native-depth `u16`** RGBA output buffer. Same native
      /// bit-width channels as `with_rgb_u16` plus α=`0xFFFF` (the source
      /// has no alpha). Length is measured in `u16` elements
      /// (`width × height × 4`).
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

      /// Attaches a **`u16`** luma output buffer. Luma is derived from
      /// expanded u8 RGB via `rgb_to_luma_u16_row` (zero-extended `u8`
      /// result, range `[0, 255]`). No native luma precision exists for
      /// these formats. Length in `u16` elements (`width × height`).
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

    // ---- Sink subtrait -------------------------------------------------------

    impl $sink_trait for MixedSinker<'_, $marker> {}

    // ---- PixelSink ----------------------------------------------------------

    impl PixelSink for MixedSinker<'_, $marker> {
      type Input<'r> = $row_ty<'r>;
      type Error = MixedSinkerError;

      fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
        check_dimensions_match(self.width, self.height, width, height)
      }

      fn process(&mut self, row: $row_ty<'_>) -> Result<(), Self::Error> {
        let w = self.width;
        let h = self.height;
        let idx = row.row();
        let use_simd = self.simd;

        // Each pixel is 2 bytes (one LE u16 word).
        if row.$buf_field().len() != w * 2 {
          return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
            $row_slice,
            idx,
            w * 2,
            row.$buf_field().len(),
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
          ..
        } = self;
        let one_plane_start = idx * w;
        let one_plane_end = one_plane_start + w;
        let src = row.$buf_field();

        // ---- native u16 RGB output ----------------------------------------
        if let Some(buf) = rgb_u16.as_deref_mut() {
          let rgb_u16_row = rgb_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          $to_rgb_u16(src, rgb_u16_row, w, use_simd);
        }

        // ---- native u16 RGBA output (forces α=0xFFFF) ---------------------
        if let Some(buf) = rgba_u16.as_deref_mut() {
          let rgba_u16_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          $to_rgba_u16(src, rgba_u16_row, w, use_simd);
        }

        // ---- u8 RGBA output (forces α=0xFF) --------------------------------
        // Dispatched via dedicated kernel — no RGB staging required.
        let want_rgb = rgb.is_some();
        let want_luma = luma.is_some();
        let want_luma_u16 = luma_u16.is_some();
        let want_hsv = hsv.is_some();
        let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

        if !need_u8_rgb {
          // Standalone RGBA fast path — write directly; avoid scratch alloc.
          if let Some(buf) = rgba.as_deref_mut() {
            let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
            $to_rgba(src, rgba_row, w, use_simd);
          }
          return Ok(());
        }

        // ---- u8 RGB staging (drives rgb / luma / luma_u16 / hsv) ----------
        let rgb_row = rgb_row_buf_or_scratch(
          rgb.as_deref_mut(),
          rgb_scratch,
          one_plane_start,
          one_plane_end,
          w,
          h,
        )?;
        $to_rgb(src, rgb_row, w, use_simd);

        if let Some(luma_buf) = luma.as_deref_mut() {
          rgb_to_luma_row(
            rgb_row,
            &mut luma_buf[one_plane_start..one_plane_end],
            w,
            row.matrix(),
            row.full_range(),
            use_simd,
          );
        }

        if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
          rgb_to_luma_u16_row(
            rgb_row,
            &mut luma_u16_buf[one_plane_start..one_plane_end],
            w,
            row.matrix(),
            row.full_range(),
            use_simd,
          );
        }

        if let Some(hsv_bufs) = hsv.as_mut() {
          rgb_to_hsv_row(
            rgb_row,
            &mut hsv_bufs.h[one_plane_start..one_plane_end],
            &mut hsv_bufs.s[one_plane_start..one_plane_end],
            &mut hsv_bufs.v[one_plane_start..one_plane_end],
            w,
            use_simd,
          );
        }

        // RGBA u8 fan-out via dedicated kernel (not Strategy A — avoids
        // double-pass without a shared RGB→RGBA expand for these formats).
        if let Some(buf) = rgba.as_deref_mut() {
          let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          $to_rgba(src, rgba_row, w, use_simd);
        }

        Ok(())
      }
    }
  };
}

// ============================================================================
// Six format instantiations
// ============================================================================

impl_legacy_rgb_sinker! {
  marker:      Rgb565,
  sink_trait:  Rgb565Sink,
  row_ty:      Rgb565Row,
  buf_field:   rgb565,
  row_slice:   RowSlice::Rgb565Packed,
  to_rgb:      rgb565_to_rgb_row,
  to_rgba:     rgb565_to_rgba_row,
  to_rgb_u16:  rgb565_to_rgb_u16_row,
  to_rgba_u16: rgb565_to_rgba_u16_row,
}

impl_legacy_rgb_sinker! {
  marker:      Bgr565,
  sink_trait:  Bgr565Sink,
  row_ty:      Bgr565Row,
  buf_field:   bgr565,
  row_slice:   RowSlice::Bgr565Packed,
  to_rgb:      bgr565_to_rgb_row,
  to_rgba:     bgr565_to_rgba_row,
  to_rgb_u16:  bgr565_to_rgb_u16_row,
  to_rgba_u16: bgr565_to_rgba_u16_row,
}

impl_legacy_rgb_sinker! {
  marker:      Rgb555,
  sink_trait:  Rgb555Sink,
  row_ty:      Rgb555Row,
  buf_field:   rgb555,
  row_slice:   RowSlice::Rgb555Packed,
  to_rgb:      rgb555_to_rgb_row,
  to_rgba:     rgb555_to_rgba_row,
  to_rgb_u16:  rgb555_to_rgb_u16_row,
  to_rgba_u16: rgb555_to_rgba_u16_row,
}

impl_legacy_rgb_sinker! {
  marker:      Bgr555,
  sink_trait:  Bgr555Sink,
  row_ty:      Bgr555Row,
  buf_field:   bgr555,
  row_slice:   RowSlice::Bgr555Packed,
  to_rgb:      bgr555_to_rgb_row,
  to_rgba:     bgr555_to_rgba_row,
  to_rgb_u16:  bgr555_to_rgb_u16_row,
  to_rgba_u16: bgr555_to_rgba_u16_row,
}

impl_legacy_rgb_sinker! {
  marker:      Rgb444,
  sink_trait:  Rgb444Sink,
  row_ty:      Rgb444Row,
  buf_field:   rgb444,
  row_slice:   RowSlice::Rgb444Packed,
  to_rgb:      rgb444_to_rgb_row,
  to_rgba:     rgb444_to_rgba_row,
  to_rgb_u16:  rgb444_to_rgb_u16_row,
  to_rgba_u16: rgb444_to_rgba_u16_row,
}

impl_legacy_rgb_sinker! {
  marker:      Bgr444,
  sink_trait:  Bgr444Sink,
  row_ty:      Bgr444Row,
  buf_field:   bgr444,
  row_slice:   RowSlice::Bgr444Packed,
  to_rgb:      bgr444_to_rgb_row,
  to_rgba:     bgr444_to_rgba_row,
  to_rgb_u16:  bgr444_to_rgb_u16_row,
  to_rgba_u16: bgr444_to_rgba_u16_row,
}
