//! Sinker impls for high-bit-depth planar GBR source formats (Tier 10b).
//!
//! Covers all ten high-bit formats:
//! - `Gbrp9` / `Gbrp10` / `Gbrp12` / `Gbrp14` / `Gbrp16` — three planes
//!   (G, B, R), no alpha. `AV_PIX_FMT_GBRP{9,10,12,14,16}LE`.
//! - `Gbrap9` / `Gbrap10` / `Gbrap12` / `Gbrap14` / `Gbrap16` — four planes
//!   (G, B, R, A), real per-pixel α. `AV_PIX_FMT_GBRAP{9,10,12,14,16}LE`.
//!
//! # Output paths
//!
//! - `with_rgb` — interleave G/B/R → packed `R, G, B` bytes (downshift by
//!   `BITS - 8`).
//! - `with_rgb_u16` — interleave G/B/R → packed `R, G, B` u16 elements at
//!   native depth (no shift; values in `[0, (1 << BITS) - 1]`).
//! - `with_rgba` — for `GbrpN`: standalone `gbr_to_rgba_opaque_high_bit_row`
//!   (α = `0xFF`); combo with `with_rgb` uses Strategy A (expand). For
//!   `GbrapN`: standalone `gbra_to_rgba_high_bit_row` (real α downshifted by
//!   `BITS - 8`); combo with `with_rgb` uses Strategy A+ (expand + α-overwrite
//!   from source plane).
//! - `with_rgba_u16` — same as above but u16 output; opaque α =
//!   `(1 << BITS) - 1` for `GbrpN`; real α at native depth for `GbrapN`.
//! - `with_luma` — derived from staged RGB (u8) via `rgb_to_luma_row`.
//! - `with_luma_u16` — derived from staged RGB via a u8 intermediary:
//!   downshift to u8, compute luma, then upshift result back to native depth
//!   (`luma_u16[i] = (y_u8 as u16) << (BITS - 8)`). This preserves the same
//!   relative luma precision as the u8 path without a new BITS-generic kernel.
//! - `with_hsv` — derived from staged RGB via `rgb_to_hsv_row`.
//!
//! # Strategy A+ (Gbrap combo path)
//!
//! When both `with_rgb` and `with_rgba` are attached to a `GbrapN` sinker:
//! 1. Stage G/B/R → RGB row.
//! 2. Expand RGB → RGBA (α = `0xFF` stub).
//! 3. Overwrite α bytes from the source A plane via
//!    `alpha_extract::copy_alpha_plane_u16_to_u8::<BITS>`.
//!
//! This avoids two calls to the full 4-channel kernel and matches the shape
//! of the 8-bit `Gbrap` sinker post-codex-fix.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    alpha_extract, expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, gbr_to_rgb_high_bit_row,
    gbr_to_rgb_u16_high_bit_row, gbr_to_rgba_opaque_high_bit_row,
    gbr_to_rgba_opaque_u16_high_bit_row, gbra_to_rgba_high_bit_row, gbra_to_rgba_u16_high_bit_row,
    rgb_to_hsv_row, rgb_to_luma_row,
  },
};

// ---- Gbrp accessors (via format-specific impl blocks) -------------------
//
// Each format gets its own impl block. The pattern is identical for all 5
// depths — only the const BITS value and the concrete Row/Sink types differ.
// We use a local macro to avoid 400 lines of repetition.
//
// The macro generates:
//   impl<'a> MixedSinker<'a, $marker> { with_rgb_u16 / set_rgb_u16 /
//                                        with_rgba / set_rgba /
//                                        with_rgba_u16 / set_rgba_u16 /
//                                        with_luma_u16 / set_luma_u16 }
//   impl $sink for MixedSinker<'_, $marker> {}
//   impl PixelSink for MixedSinker<'_, $marker> { ... }

macro_rules! impl_gbrp_high_bit {
  ($marker:ident, $sink:ident, $row:ident, $bits:literal) => {
    impl<'a> MixedSinker<'a, crate::yuv::$marker> {
      /// Attaches a packed **`u16`** RGB output buffer. Samples are in
      /// `[0, (1 << BITS) - 1]` (native depth, no depth conversion).
      /// Length is measured in `u16` **elements** (`width × height × 3`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_rgb_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgb_u16(
        &mut self,
        buf: &'a mut [u16],
      ) -> Result<&mut Self, MixedSinkerError> {
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

      /// Attaches a packed **8-bit** RGBA output buffer. Alpha is opaque
      /// (`0xFF`) — the GBR format has no alpha plane. Length in bytes
      /// (`width × height × 4`).
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

      /// Attaches a packed **`u16`** RGBA output buffer. Alpha is opaque
      /// (`(1 << BITS) - 1`). Length in `u16` elements (`width × height × 4`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_rgba_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgba_u16(
        &mut self,
        buf: &'a mut [u16],
      ) -> Result<&mut Self, MixedSinkerError> {
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

      /// Attaches a `u16` luma output buffer. Luma is derived from G/B/R
      /// via the staged-RGB path: downshift to u8, compute luma with
      /// `rgb_to_luma_row`, then upshift result back to native depth
      /// (`y_u16[i] = (y_u8 as u16) << (BITS - 8)`). Length in `u16`
      /// elements (`width × height`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_luma_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_luma_u16(
        &mut self,
        buf: &'a mut [u16],
      ) -> Result<&mut Self, MixedSinkerError> {
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

    impl crate::yuv::$sink for MixedSinker<'_, crate::yuv::$marker> {}

    impl PixelSink for MixedSinker<'_, crate::yuv::$marker> {
      type Input<'r> = crate::yuv::$row<'r>;
      type Error = MixedSinkerError;

      fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
        check_dimensions_match(self.width, self.height, width, height)
      }

      fn process(&mut self, row: crate::yuv::$row<'_>) -> Result<(), Self::Error> {
        const BITS: u32 = $bits;
        let w = self.width;
        let h = self.height;
        let idx = row.row();
        let use_simd = self.simd;

        if row.g().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch {
            which: RowSlice::GPlane,
            row: idx,
            expected: w,
            actual: row.g().len(),
          });
        }
        if row.b().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch {
            which: RowSlice::BPlane,
            row: idx,
            expected: w,
            actual: row.b().len(),
          });
        }
        if row.r().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch {
            which: RowSlice::RPlane,
            row: idx,
            expected: w,
            actual: row.r().len(),
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
        let one_plane_start = idx * w;
        let one_plane_end = one_plane_start + w;
        let g_in = row.g();
        let b_in = row.b();
        let r_in = row.r();

        // ---- u16 RGB / RGBA output (Strategy A) -------------------------
        //
        // Native-depth path: no u8 staging needed. When both rgb_u16 and
        // rgba_u16 are requested, stage into rgb_u16 then expand; when only
        // rgba_u16 is requested, use the opaque direct kernel.
        let want_rgb_u16 = rgb_u16.is_some();
        let want_rgba_u16 = rgba_u16.is_some();

        if want_rgba_u16 && !want_rgb_u16 {
          let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
          let rgba_u16_row =
            rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
          gbr_to_rgba_opaque_u16_high_bit_row::<BITS>(g_in, b_in, r_in, rgba_u16_row, w, use_simd);
        } else if want_rgb_u16 {
          let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
          let rgb_plane_end = one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow { width: w, height: h, channels: 3 })?;
          let rgb_plane_start = one_plane_start * 3;
          let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
          gbr_to_rgb_u16_high_bit_row::<BITS>(g_in, b_in, r_in, rgb_u16_row, w, use_simd);
          if want_rgba_u16 {
            let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
            let rgba_u16_row =
              rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
            expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
          }
        }

        // ---- u8 RGB / RGBA / luma / HSV output (Strategy A) -----------
        let want_rgb = rgb.is_some();
        let want_rgba = rgba.is_some();
        let want_luma = luma.is_some();
        let want_luma_u16 = luma_u16.is_some();
        let want_hsv = hsv.is_some();
        let need_rgb_staging = want_rgb || want_luma || want_luma_u16 || want_hsv;

        // RGBA-only fast path: use the 4-channel opaque kernel directly.
        if want_rgba && !need_rgb_staging {
          let rgba_buf = rgba.as_deref_mut().unwrap();
          let rgba_row =
            rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
          gbr_to_rgba_opaque_high_bit_row::<BITS>(g_in, b_in, r_in, rgba_row, w, use_simd);
          return Ok(());
        }

        if !need_rgb_staging && !want_rgba {
          return Ok(());
        }

        // Stage RGB once (into user buffer or scratch).
        let rgb_row = rgb_row_buf_or_scratch(
          rgb.as_deref_mut(),
          rgb_scratch,
          one_plane_start,
          one_plane_end,
          w,
          h,
        )?;
        gbr_to_rgb_high_bit_row::<BITS>(g_in, b_in, r_in, rgb_row, w, use_simd);

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

        if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
          // Derive luma via u8 intermediate, then upshift to native depth.
          // This preserves luma-precision parity with the u8 path and avoids
          // a new BITS-generic luma kernel.
          const STACK_CAP: usize = 8192;
          let luma_out = &mut luma_u16_buf[one_plane_start..one_plane_end];
          if w <= STACK_CAP {
            let mut scratch = [0u8; STACK_CAP];
            let scratch = &mut scratch[..w];
            rgb_to_luma_row(rgb_row, scratch, w, row.matrix(), row.full_range(), use_simd);
            for (d, &s) in luma_out.iter_mut().zip(scratch.iter()) {
              *d = (s as u16) << (BITS - 8);
            }
          } else {
            let mut scratch = std::vec![0u8; w];
            rgb_to_luma_row(
              rgb_row,
              &mut scratch,
              w,
              row.matrix(),
              row.full_range(),
              use_simd,
            );
            for (d, &s) in luma_out.iter_mut().zip(scratch.iter()) {
              *d = (s as u16) << (BITS - 8);
            }
          }
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

        if let Some(buf) = rgba.as_deref_mut() {
          // Strategy A: expand already-computed rgb_row → rgba (opaque).
          let rgba_row =
            rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        }

        Ok(())
      }
    }
  };
}

macro_rules! impl_gbrap_high_bit {
  ($marker:ident, $sink:ident, $row:ident, $bits:literal) => {
    impl<'a> MixedSinker<'a, crate::yuv::$marker> {
      /// Attaches a packed **`u16`** RGB output buffer. Samples are in
      /// `[0, (1 << BITS) - 1]` (native depth, no depth conversion).
      /// Length is measured in `u16` **elements** (`width × height × 3`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_rgb_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgb_u16(
        &mut self,
        buf: &'a mut [u16],
      ) -> Result<&mut Self, MixedSinkerError> {
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

      /// Attaches a packed **8-bit** RGBA output buffer. Alpha is sourced
      /// from the source A plane, downshifted by `BITS - 8`.
      /// Length in bytes (`width × height × 4`).
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

      /// Attaches a packed **`u16`** RGBA output buffer. Alpha is sourced
      /// from the source A plane at native depth (no depth conversion).
      /// Length in `u16` elements (`width × height × 4`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_rgba_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_rgba_u16(
        &mut self,
        buf: &'a mut [u16],
      ) -> Result<&mut Self, MixedSinkerError> {
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

      /// Attaches a `u16` luma output buffer. Same derivation as `GbrpN` —
      /// downshift to u8, compute luma, upshift: `y_u16[i] = (y_u8 as u16) <<
      /// (BITS - 8)`. Length in `u16` elements (`width × height`).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
        self.set_luma_u16(buf)?;
        Ok(self)
      }
      /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
      #[cfg_attr(not(tarpaulin), inline(always))]
      pub fn set_luma_u16(
        &mut self,
        buf: &'a mut [u16],
      ) -> Result<&mut Self, MixedSinkerError> {
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

    impl crate::yuv::$sink for MixedSinker<'_, crate::yuv::$marker> {}

    impl PixelSink for MixedSinker<'_, crate::yuv::$marker> {
      type Input<'r> = crate::yuv::$row<'r>;
      type Error = MixedSinkerError;

      fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
        check_dimensions_match(self.width, self.height, width, height)
      }

      fn process(&mut self, row: crate::yuv::$row<'_>) -> Result<(), Self::Error> {
        const BITS: u32 = $bits;
        let w = self.width;
        let h = self.height;
        let idx = row.row();
        let use_simd = self.simd;

        if row.g().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch {
            which: RowSlice::GPlane,
            row: idx,
            expected: w,
            actual: row.g().len(),
          });
        }
        if row.b().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch {
            which: RowSlice::BPlane,
            row: idx,
            expected: w,
            actual: row.b().len(),
          });
        }
        if row.r().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch {
            which: RowSlice::RPlane,
            row: idx,
            expected: w,
            actual: row.r().len(),
          });
        }
        if row.a().len() != w {
          return Err(MixedSinkerError::RowShapeMismatch {
            which: RowSlice::AFull,
            row: idx,
            expected: w,
            actual: row.a().len(),
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
        let one_plane_start = idx * w;
        let one_plane_end = one_plane_start + w;
        let g_in = row.g();
        let b_in = row.b();
        let r_in = row.r();
        let a_in = row.a();

        // ---- u16 RGB / RGBA output (Strategy A+) -------------------------
        //
        // For GbrapN: rgba_u16 = stage rgb_u16 → expand → α-overwrite from
        // source plane (no depth conv needed — both at native BITS depth).
        let want_rgb_u16 = rgb_u16.is_some();
        let want_rgba_u16 = rgba_u16.is_some();

        if want_rgba_u16 && !want_rgb_u16 {
          // Standalone u16 RGBA — direct 4-channel kernel with real α.
          let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
          let rgba_u16_row =
            rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
          gbra_to_rgba_u16_high_bit_row::<BITS>(
            g_in, b_in, r_in, a_in, rgba_u16_row, w, use_simd,
          );
        } else if want_rgb_u16 {
          let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
          let rgb_plane_end = one_plane_end
            .checked_mul(3)
            .ok_or(MixedSinkerError::GeometryOverflow { width: w, height: h, channels: 3 })?;
          let rgb_plane_start = one_plane_start * 3;
          let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
          gbr_to_rgb_u16_high_bit_row::<BITS>(g_in, b_in, r_in, rgb_u16_row, w, use_simd);
          if want_rgba_u16 {
            // Strategy A+: expand RGB → RGBA, then overwrite α from source plane.
            let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
            let rgba_u16_row =
              rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
            expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
            // Overwrite α slot from source plane (native depth, no shift).
            alpha_extract::copy_alpha_plane_u16::<BITS>(a_in, rgba_u16_row, w, use_simd);
          }
        }

        // ---- u8 RGB / RGBA / luma / HSV output --------------------------
        let want_rgb = rgb.is_some();
        let want_rgba = rgba.is_some();
        let want_luma = luma.is_some();
        let want_luma_u16 = luma_u16.is_some();
        let want_hsv = hsv.is_some();
        let need_rgb_staging = want_rgb || want_luma || want_luma_u16 || want_hsv;

        // RGBA-only fast path — direct 4-channel kernel with real α.
        if want_rgba && !need_rgb_staging {
          let rgba_buf = rgba.as_deref_mut().unwrap();
          let rgba_row =
            rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
          gbra_to_rgba_high_bit_row::<BITS>(g_in, b_in, r_in, a_in, rgba_row, w, use_simd);
          return Ok(());
        }

        if !need_rgb_staging && !want_rgba {
          return Ok(());
        }

        // Stage RGB once.
        let rgb_row = rgb_row_buf_or_scratch(
          rgb.as_deref_mut(),
          rgb_scratch,
          one_plane_start,
          one_plane_end,
          w,
          h,
        )?;
        gbr_to_rgb_high_bit_row::<BITS>(g_in, b_in, r_in, rgb_row, w, use_simd);

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

        if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
          const STACK_CAP: usize = 8192;
          let luma_out = &mut luma_u16_buf[one_plane_start..one_plane_end];
          if w <= STACK_CAP {
            let mut scratch = [0u8; STACK_CAP];
            let scratch = &mut scratch[..w];
            rgb_to_luma_row(rgb_row, scratch, w, row.matrix(), row.full_range(), use_simd);
            for (d, &s) in luma_out.iter_mut().zip(scratch.iter()) {
              *d = (s as u16) << (BITS - 8);
            }
          } else {
            let mut scratch = std::vec![0u8; w];
            rgb_to_luma_row(
              rgb_row,
              &mut scratch,
              w,
              row.matrix(),
              row.full_range(),
              use_simd,
            );
            for (d, &s) in luma_out.iter_mut().zip(scratch.iter()) {
              *d = (s as u16) << (BITS - 8);
            }
          }
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

        if let Some(buf) = rgba.as_deref_mut() {
          // Strategy A+: expand rgb_row → RGBA (opaque stub), then
          // overwrite α bytes from the source A plane.
          let rgba_row =
            rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
          alpha_extract::copy_alpha_plane_u16_to_u8::<BITS>(a_in, rgba_row, w, use_simd);
        }

        Ok(())
      }
    }
  };
}

// ---- Gbrp formats (no alpha) -------------------------------------------

impl_gbrp_high_bit!(Gbrp9, Gbrp9Sink, Gbrp9Row, 9);
impl_gbrp_high_bit!(Gbrp10, Gbrp10Sink, Gbrp10Row, 10);
impl_gbrp_high_bit!(Gbrp12, Gbrp12Sink, Gbrp12Row, 12);
impl_gbrp_high_bit!(Gbrp14, Gbrp14Sink, Gbrp14Row, 14);
impl_gbrp_high_bit!(Gbrp16, Gbrp16Sink, Gbrp16Row, 16);

// ---- Gbrap formats (with real α plane) ---------------------------------

impl_gbrap_high_bit!(Gbrap9, Gbrap9Sink, Gbrap9Row, 9);
impl_gbrap_high_bit!(Gbrap10, Gbrap10Sink, Gbrap10Row, 10);
impl_gbrap_high_bit!(Gbrap12, Gbrap12Sink, Gbrap12Row, 12);
impl_gbrap_high_bit!(Gbrap14, Gbrap14Sink, Gbrap14Row, 14);
impl_gbrap_high_bit!(Gbrap16, Gbrap16Sink, Gbrap16Row, 16);
