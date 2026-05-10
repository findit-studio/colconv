//! Sinker impl for the packed AYUV64 source format — Ship 12d (Tier 5
//! 16-bit packed YUV 4:4:4 with real source alpha).
//!
//! AYUV64 (FFmpeg `AV_PIX_FMT_AYUV64LE`) packs **four u16 slots per
//! pixel** (`[A, Y, U, V]`). All channels are 16-bit native — no
//! padding bits, no shift required. The A slot is **real source alpha**
//! — not padding. The packed slice type is `&[u16]`, with `4 × width`
//! u16 elements per row. There is no chroma subsampling — every pixel
//! carries its own independent A / Y / U / V quadruple (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; alpha discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; **source α is
//!   depth-converted to u8 via `>> 8`** from slot 0 of each pixel.
//! - `with_rgb_u16` — packed YUV → RGB u16 pipeline; alpha discarded;
//!   i64 chroma path.
//! - `with_rgba_u16` — packed YUV → RGBA u16 pipeline; **source α u16
//!   is written direct** (no conversion); i64 chroma path.
//! - `with_luma` — extracts the Y u16 from slot 1 and downshifts `>> 8`
//!   to u8; no YUV→RGB pipeline.
//! - `with_luma_u16` — extracts the Y u16 at full 16-bit native depth;
//!   no YUV→RGB pipeline.
//! - `with_hsv` — stages u8 RGB into the user's RGB buffer (if
//!   attached) or a scratch buffer, then runs `rgb_to_hsv_row`.
//!
//! ## Alpha semantics (`§ 7.2` / Tier 5 spec rules)
//!
//! - **Standalone RGBA u8** (`with_rgba` attached, no `with_rgb`, no
//!   `with_hsv`): `ayuv64_to_rgba_row` runs directly — source α is
//!   depth-converted via `>> 8` in the kernel.
//! - **Standalone RGBA u16** (`with_rgba_u16` attached, no
//!   `with_rgb_u16`): `ayuv64_to_rgba_u16_row` runs directly — source
//!   α is written direct as u16.
//! - **RGB + RGBA** (both attached, with or without HSV): Strategy A+
//!   combo — `with_rgb` calls `ayuv64_to_rgb_row` (chroma kernel runs
//!   ONCE); `with_rgba` is derived by `expand_rgb_to_rgba_row` (writes
//!   α=`0xFF`) followed by
//!   `alpha_extract::copy_alpha_packed_u16x4_to_u8_at_0` to
//!   overwrite the α slot from the packed source (slot 0, depth-conv
//!   `>> 8`). Output is byte-identical to calling `ayuv64_to_rgba_row`
//!   directly (spec § 3.2 / § 7.2).
//! - **RGB u16 + RGBA u16** (both attached): same A+ pattern on the u16
//!   path — `expand_rgb_u16_to_rgba_u16_row::<16>` fans out, then
//!   `copy_alpha_packed_u16x4_at_0` overwrites α from packed slot 0.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    ayuv64_to_luma_row, ayuv64_to_luma_u16_row, ayuv64_to_rgb_row, ayuv64_to_rgb_u16_row,
    ayuv64_to_rgba_row, ayuv64_to_rgba_u16_row, expand_rgb_to_rgba_row,
    expand_rgb_u16_to_rgba_u16_row, rgb_to_hsv_row,
  },
  source::{Ayuv64, Ayuv64Row, Ayuv64Sink},
};

impl<'a, const BE: bool> MixedSinker<'a, Ayuv64<BE>> {
  /// Attaches a packed **8-bit** RGBA output buffer. When AYUV64 is the
  /// source, the per-pixel alpha value is **sourced from the A u16 at
  /// slot 0 of each pixel quadruple**, depth-converted to u8 via `>> 8`
  /// — not forced to `0xFF`.
  ///
  /// Returns `Err(RgbaBufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ## Strategy note
  ///
  /// Source-α pass-through is guaranteed in **all** paths (standalone or
  /// combined with `with_rgb` / `with_hsv`). When standalone (no
  /// `with_rgb` / `with_hsv`), `ayuv64_to_rgba_row` runs directly.
  /// When combined with `with_rgb`, Strategy A+ applies:
  /// `expand_rgb_to_rgba_row` fans out the RGB row (α=`0xFF`) and
  /// `alpha_extract::copy_alpha_packed_u16x4_to_u8_at_0`
  /// overwrites the α slot — output is byte-identical to the standalone
  /// path (spec § 3.2 / § 7.2).
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

  /// Attaches a packed **`u16`** RGB output buffer. Native 16-bit depth;
  /// length is measured in `u16` **elements** (`width × height × 3`).
  /// Alpha is discarded.
  ///
  /// Returns `Err(RgbU16BufferTooShort)` if
  /// `buf.len() < width × height × 3`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
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

  /// Attaches a packed **`u16`** RGBA output buffer. Native 16-bit
  /// depth; source α u16 at slot 0 of each pixel quadruple is written
  /// **direct** (no conversion). Length is measured in `u16`
  /// **elements** (`width × height × 4`).
  ///
  /// Returns `Err(RgbaU16BufferTooShort)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ## Strategy note
  ///
  /// Source-α pass-through (u16 direct) is guaranteed in **all** paths.
  /// When standalone (no `with_rgb_u16`), `ayuv64_to_rgba_u16_row` runs
  /// directly. When combined with `with_rgb_u16`, Strategy A+ applies:
  /// `expand_rgb_u16_to_rgba_u16_row::<16>` fans out the u16 RGB row and
  /// `alpha_extract::copy_alpha_packed_u16x4_at_0` overwrites
  /// the α slot — output is byte-identical to the standalone path (spec
  /// § 3.2 / § 7.2).
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

  /// Attaches a native-depth **`u16`** luma output buffer. The 16-bit Y
  /// value at slot 1 of each AYUV64 quadruple is written direct (no
  /// shift — 16-bit native). Length is measured in `u16` **elements**
  /// (`width × height`).
  ///
  /// Returns `Err(LumaU16BufferTooShort)` if
  /// `buf.len() < width × height`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets.
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

impl<const BE: bool> Ayuv64Sink<BE> for MixedSinker<'_, Ayuv64<BE>> {}

impl<const BE: bool> PixelSink for MixedSinker<'_, Ayuv64<BE>> {
  type Input<'r> = Ayuv64Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    Ok(())
  }

  fn process(&mut self, row: Ayuv64Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // AYUV64 row = `width × 4` u16 elements (one quadruple per pixel).
    let packed_expected = w.checked_mul(4).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 4,
    })?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Ayuv64Packed,
        row: idx,
        expected: packed_expected,
        actual: row.packed().len(),
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
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let packed = row.packed();

    // Luma u8 — extract Y value from slot 1 of each AYUV64 quadruple
    // and downshift `>> 8` to u8.
    if let Some(buf) = luma.as_deref_mut() {
      ayuv64_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
      );
    }

    // Luma u16 — extract Y value at native 16-bit depth (written direct,
    // no shift).
    if let Some(buf) = luma_u16.as_deref_mut() {
      ayuv64_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
        BE,
      );
    }

    // ===== u8 RGB / RGBA / HSV path =====
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // ===== u16 RGB / RGBA path =====
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    // Standalone RGBA u8 fast path — spec § 7.2: when only RGBA u8 (no
    // RGB, no HSV) is requested AND no u16 work is needed, run the
    // dedicated RGBA kernel directly and return early. Source α is
    // depth-converted via `>> 8` in the kernel.
    if want_rgba && !need_rgb_kernel && !want_rgb_u16 && !want_rgba_u16 {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      ayuv64_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    // Standalone RGBA u16 fast path — when only RGBA u16 (no RGB u16) is
    // requested AND no u8 work is needed, run the dedicated kernel
    // directly and return early; source α is written direct as u16.
    if want_rgba_u16 && !want_rgb_u16 && !need_rgb_kernel && !want_rgba {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      ayuv64_to_rgba_u16_row(
        packed,
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
      return Ok(());
    }

    // ===== Combo / mixed paths =====
    //
    // Reached when at least two of {rgb, rgba, hsv, rgb_u16, rgba_u16}
    // are attached, or when the single standalone fast paths didn't fire.

    // u8 RGB path — write into the user's RGB buffer (if attached) or the
    // internal scratch buffer. Required when with_rgb or with_hsv is set.
    if need_rgb_kernel {
      let rgb_row = rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?;
      ayuv64_to_rgb_row(
        packed,
        rgb_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );

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

      // Strategy A+ combo (u8): RGBA also attached — derive from the
      // just-computed RGB row (writes α=0xFF), then overwrite α slot from
      // packed source (slot 0, depth-conv >> 8). Output is byte-identical
      // to ayuv64_to_rgba_row directly (spec § 3.2 / § 7.2).
      // See spec docs/superpowers/specs/2026-05-04-pr4-strategy-a-plus-design.md.
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        // BE propagated from the parent `Ayuv64Frame<'_, BE>` via the
        // sinker's `MixedSinker<Ayuv64<BE>>` monomorphization (Phase 4).
        crate::row::alpha_extract::copy_alpha_packed_u16x4_to_u8_at_0::<BE>(
          packed, rgba_row, w, use_simd,
        );
      }
    }

    // Standalone RGBA u8 path — want_rgba without need_rgb_kernel (so
    // want_rgba must be combined with want_rgb_u16 or want_rgba_u16 only).
    // Run ayuv64_to_rgba_row directly; source α depth-converted >> 8.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      ayuv64_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    }

    // u16 RGB path — run when rgb_u16 is attached.
    if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow {
            width: w,
            height: h,
            channels: 3,
          })?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      ayuv64_to_rgb_u16_row(
        packed,
        rgb_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );

      // Strategy A+ combo (u16): RGBA u16 also attached — derive from the
      // just-computed u16 RGB row (writes α=max at 16 bits), then overwrite
      // α slot from packed source (slot 0, u16 direct). Output is
      // byte-identical to ayuv64_to_rgba_u16_row directly (spec § 3.2).
      // See spec docs/superpowers/specs/2026-05-04-pr4-strategy-a-plus-design.md.
      if want_rgba_u16 {
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        // BE propagated from the parent `Ayuv64Frame<'_, BE>` via the
        // sinker's `MixedSinker<Ayuv64<BE>>` monomorphization (Phase 4).
        crate::row::alpha_extract::copy_alpha_packed_u16x4_at_0::<BE>(
          packed,
          rgba_u16_row,
          w,
          use_simd,
        );
      }
    }

    // Standalone RGBA u16 path — want_rgba_u16 without want_rgb_u16 (so
    // want_rgba_u16 must be combined with need_rgb_kernel or want_rgba).
    // Run ayuv64_to_rgba_u16_row directly; source α u16 written direct.
    if want_rgba_u16 && !want_rgb_u16 {
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      ayuv64_to_rgba_u16_row(
        packed,
        rgba_u16_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
        BE,
      );
    }

    Ok(())
  }
}
