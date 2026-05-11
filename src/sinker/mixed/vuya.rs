//! Sinker impl for the packed VUYA source format — Ship 12c (Tier 5
//! 8-bit packed YUV 4:4:4 with real source alpha).
//!
//! VUYA (FFmpeg `AV_PIX_FMT_VUYA`) packs **four u8 bytes per pixel**
//! (`[V, U, Y, A]`). The A byte is **real source alpha** — not padding.
//! The packed slice type is `&[u8]`, with `4 × width` byte elements per
//! row. There is no chroma subsampling — every pixel carries its own
//! independent V / U / Y triplet (4:4:4).
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` — packed YUV → RGB 8-bit pipeline; alpha discarded.
//! - `with_rgba` — packed YUV → RGBA 8-bit pipeline; **source α byte
//!   is passed through** verbatim from byte 3 of each pixel (not
//!   substituted with `0xFF`).
//! - `with_luma` — extracts the Y byte at offset 2 of each pixel
//!   directly (no YUV→RGB pipeline).
//! - `with_luma_u16` — zero-extends the Y byte to u16
//!   (`out[x] = Y_byte as u16`).
//! - `with_hsv` — stages u8 RGB into the user's RGB buffer (if
//!   attached) or a scratch buffer, then runs `rgb_to_hsv_row`.
//!
//! ## Alpha semantics (`§ 7.2` / `§ 7.3` rules)
//!
//! - **Standalone RGBA** (`with_rgba` attached, no `with_rgb`, no
//!   `with_hsv`): `vuya_to_rgba_row` runs directly — source α passes
//!   through via the kernel.
//! - **RGB + RGBA** (both attached, with or without HSV): Strategy A+
//!   combo — `with_rgb` calls `vuya_to_rgb_row` (chroma kernel runs
//!   ONCE); `with_rgba` is derived by `expand_rgb_to_rgba_row` (writes
//!   α=`0xFF`) followed by
//!   `alpha_extract::copy_alpha_packed_u8x4_at_3` to overwrite
//!   the α slot from the packed source (slot 3). Output is byte-identical
//!   to calling `vuya_to_rgba_row` directly (spec § 3.2 / § 7.2).

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, vuya_to_luma_row, vuya_to_luma_u16_row,
    vuya_to_rgb_row, vuya_to_rgba_row,
  },
  source::{Vuya, VuyaRow, VuyaSink},
};

impl<'a> MixedSinker<'a, Vuya> {
  /// Attaches a **`u16`** luma output buffer. Y bytes from the packed VUYA
  /// `[V, U, Y, A]` layout are zero-extended to u16
  /// (`out[x] = Y_byte as u16`). Length in u16 **elements**
  /// (`width × height`).
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if `buf.len() < width × height`,
  /// or `Err(GeometryOverflow)` on 32-bit targets.
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

  /// Attaches a packed **8-bit** RGBA output buffer. When VUYA is the
  /// source, the per-pixel alpha byte is **sourced from the A byte of
  /// each pixel quadruple** — not forced to `0xFF`.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width × height × 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ## Strategy note
  ///
  /// Source-α pass-through is guaranteed in **all** paths (standalone or
  /// combined with `with_rgb` / `with_hsv`). When standalone (no
  /// `with_rgb` / `with_hsv`), `vuya_to_rgba_row` runs directly from the
  /// packed source. When combined with `with_rgb`, Strategy A+ applies:
  /// `expand_rgb_to_rgba_row` fans out the RGB row (α=`0xFF`) and
  /// `alpha_extract::copy_alpha_packed_u8x4_at_3` overwrites
  /// the α slot — output is byte-identical to the standalone path (spec
  /// § 3.2 / § 7.2).
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
}

impl VuyaSink for MixedSinker<'_, Vuya> {}

impl PixelSink for MixedSinker<'_, Vuya> {
  type Input<'r> = VuyaRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    Ok(())
  }

  fn process(&mut self, row: VuyaRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // VUYA row = `width × 4` bytes (one quadruple per pixel).
    let packed_expected =
      w.checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 4,
        )))?;
    if row.packed().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VuyaPacked,
        idx,
        packed_expected,
        row.packed().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
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
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;
    let packed = row.packed();

    // Luma u8 — extract Y byte (offset 2 in each VUYA quadruple) directly.
    if let Some(buf) = luma.as_deref_mut() {
      vuya_to_luma_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Luma u16 — extract Y bytes and zero-extend to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      vuya_to_luma_u16_row(
        packed,
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // ===== u8 RGB / RGBA / HSV path =====
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // RGB kernel — write into the user's RGB buffer (if attached) or the
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
      vuya_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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

      // Strategy A+ combo: RGBA both attached — derive from the just-computed
      // RGB row (writes α=0xFF), then overwrite α slot from packed source.
      // Output is byte-identical to vuya_to_rgba_row directly (spec § 3.2).
      // See spec docs/superpowers/specs/2026-05-04-pr4-strategy-a-plus-design.md.
      if want_rgba {
        let rgba_buf = rgba.as_deref_mut().unwrap();
        let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
        crate::row::alpha_extract::copy_alpha_packed_u8x4_at_3(packed, rgba_row, w, use_simd);
      }
    }

    // Standalone RGBA path — no RGB/HSV requested. Run vuya_to_rgba_row
    // directly from the packed source; source α passes through in the
    // kernel. This path is already optimal — unchanged (spec § 7.2).
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      vuya_to_rgba_row(
        packed,
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    Ok(())
  }
}
