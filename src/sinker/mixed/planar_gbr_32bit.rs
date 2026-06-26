//! Sinker impl for the 32-bit planar GBR + alpha source format
//! (`AV_PIX_FMT_GBRAP32{LE,BE}`).
//!
//! Four full-resolution `u32` planes in **G, B, R, A** order; all 32 bits
//! active and the alpha plane is real per-pixel α. This is the full-bit `u32`
//! twin of the 16-bit [`Gbrap16`](super::super) family: once each plane is
//! narrowed (`>> 24` for u8, `>> 16` for native u16) the packed layout and the
//! binning / resample tails are identical, so this impl reuses the alpha-aware
//! high-bit packed-RGBA tail at `BITS = 16` unchanged.
//!
//! # Output paths
//!
//! - `with_rgb` — interleave G/B/R → packed `R, G, B` bytes (`>> 24`).
//! - `with_rgb_u16` — interleave G/B/R → packed `R, G, B` u16 (`>> 16`).
//! - `with_rgba` — `gbra32_to_rgba_row` (real α `>> 24`); combo with `with_rgb`
//!   uses Strategy A+ (expand + α-overwrite from the source plane).
//! - `with_rgba_u16` — `gbra32_to_rgba_u16_row` (real α `>> 16`); combo uses
//!   Strategy A+.
//! - `with_luma` — derived from staged u8 RGB via `rgb_to_luma_row`.
//! - `with_luma_u16` — native-precision Q15 luma from the `>> 16`-narrowed
//!   G/B/R via `gbr32_to_luma_u16_row` (i64 intermediates).
//! - `with_hsv` — derived from staged u8 RGB via `rgb_to_hsv_row`.
//!
//! # Fused area / filter resample (`with_resampler`)
//!
//! On a non-identity plan the native-depth G/B/R/A planes are de-interleaved
//! into a canonical host-native `R, G, B, A` u16 row (`gbra32_to_rgba_u16_row`,
//! the `>> 16` narrow) and fed to the **alpha-aware** 4-channel high-bit packed
//! RGBA tail at `BITS = 16` — the same tail `Rgba64` / `Gbrap16` take.
//! Resampled alpha is a real native area mean, and under
//! [`AlphaMode::Premultiplied`](super::AlphaMode::Premultiplied) the color is
//! binned premultiplied and un-premultiplied per output row. A straight
//! rgb-only sink (alpha dropped) keeps the 3-channel u16 RGB path.
//! `luma_u16` is computed at native precision from the binned RGB
//! (`NATIVE_LUMA16 = true`).
//!
//! ## Precision (resample) — issue #289
//!
//! Because each wire `u32` is narrowed `>> 16` **before** binning, a
//! downscaled / filtered output — for **both** `full_range = true` and
//! `full_range = false` — is the area/filter mean of the *narrowed* high 16
//! bits, i.e. within 1 LSB of an exact u32-domain mean (averaging the full
//! `u32` samples and narrowing only at the end). Only the **direct**
//! (identity-plan, 1:1) conversion is exact and byte-identical; **every**
//! resampled output (RGB / RGBA / luma / alpha, either range) is within 1 LSB.
//! The limited-range case is merely the *most visible* — its luma rescale
//! amplifies the dropped low 16 bits — but the full-range resample is
//! within-1-LSB too, not 0-ULP. This u32 family deliberately ACCEPTS the
//! ≤1-LSB resample gap rather than building new u32 resample infrastructure
//! (mirrors the merged `Gray32` / `Rgb96` / `Rgba128` decision); the exact
//! 0-ULP fix (a `u128` area tier + a `u32` filter tier) is tracked in issue
//! #289. The current narrow-first behaviour is pinned by the full-range AND
//! limited-range resample tests (area + filter, LE + BE) in
//! `tests/resample_gbrap_32bit.rs`.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, check_frozen_alpha_mode,
  packed_rgb_u16_filter_stream, packed_rgb_u16_resample_emit, packed_rgb_u16_resample_preflight,
  packed_rgb_u16_resample_stream, packed_rgba_u16_filter_resample, packed_rgba_u16_resample,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice, source_rgb_u16_scratch,
};
use crate::{
  PixelSink,
  row::{
    expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row, gbr32_to_luma_u16_row,
    gbr32_to_rgb_row, gbr32_to_rgb_u16_row, gbra32_to_rgba_row, gbra32_to_rgba_u16_row,
    rgb_to_hsv_row, rgb_to_luma_row, scalar::alpha_extract,
  },
  source::{Gbrap32, Gbrap32Row, Gbrap32Sink},
};

impl<'a, R, const BE: bool> MixedSinker<'a, Gbrap32<BE>, R> {
  /// Attaches a packed **`u16`** RGB output buffer. Samples are the `>> 16`
  /// narrow of each `u32` channel. Length in `u16` elements
  /// (`width x height x 3`).
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

  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is sourced from the
  /// source A plane, narrowed `>> 24`. Length in bytes (`width x height x 4`).
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

  /// Attaches a packed **`u16`** RGBA output buffer. Alpha is sourced from the
  /// source A plane, narrowed `>> 16`. Length in `u16` elements
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

  /// Attaches a `u16` luma output buffer. Luma is computed directly from the
  /// `>> 16`-narrowed G/B/R planes via Q15 coefficients (i64 intermediates).
  /// Values are in `[0, 65535]` (full-range) or `[4096, 60160]`
  /// (limited-range). Length in `u16` elements (`width x height`).
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

impl<R, const BE: bool> Gbrap32Sink<BE> for MixedSinker<'_, Gbrap32<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Gbrap32<BE>, R> {
  type Input<'r> = Gbrap32Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgb_filter_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Gbrap32Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if row.g().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::GPlane,
        idx,
        w,
        row.g().len(),
      )));
    }
    if row.b().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::BPlane,
        idx,
        w,
        row.b().len(),
      )));
    }
    if row.r().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::RPlane,
        idx,
        w,
        row.r().len(),
      )));
    }
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::AFull,
        idx,
        w,
        row.a().len(),
      )));
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, h),
      ));
    }

    // Non-identity plan. Route the alpha-aware 4-channel u16 tail when
    // resampled alpha would be dropped (rgba / rgba_u16 attached) or the color
    // must be alpha-weighted (premultiplied); otherwise the rgb-only straight
    // outputs keep the 3-channel u16 RGB path. The G/B/R/A planes are
    // de-interleaved into the canonical host-native RGBA row
    // (`gbra32_to_rgba_u16_row`, all four channels narrowed `>> 16`) and the
    // high-bit packed RGBA tail bins at BITS = 16.
    //
    // #289: narrowing each wire u32 `>> 16` BEFORE binning makes any resampled
    // output — BOTH full_range = true and false — within 1 LSB of the exact
    // u32-domain mean (only the direct identity-plan path is exact). Accepted —
    // 0-ULP fix (u128 area + u32 filter tier) tracked in issue #289.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let matrix = row.matrix();
      let full_range = row.full_range();
      let g_in = row.g();
      let b_in = row.b();
      let r_in = row.r();
      let a_in = row.a();
      let Self {
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        rgb_scratch_u16,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        rgb_stream_u16,
        rgba_stream_u16,
        rgb_filter_stream_u16,
        rgba_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        plan,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      // The alpha mode is snapshotted at begin_frame; reject a mid-frame change
      // here, before route selection, so a flip can neither reroute nor mix
      // modes.
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      match plan.kind() {
        crate::resample::SpanKind::Area => {
          if rgba.is_some() || rgba_u16.is_some() || alpha_mode.is_premultiplied() {
            return packed_rgba_u16_resample::<16, true, false>(
              rgba_stream_u16,
              // No native-Y luma stream: Gbrap32 luma_u16 is native-precision
              // color-derived (`NATIVE_LUMA16 = true`, `NATIVE_Y_LUMA = false`).
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              matrix,
              full_range,
              // #289: staged native u16 = each plane narrowed `>> 16` before
              // binning (≤1-LSB vs the exact u32-domain mean, either range).
              |dst| gbra32_to_rgba_u16_row::<BE>(g_in, b_in, r_in, a_in, dst, w, use_simd),
              |_| {},
            );
          }
          // Straight rgb-only (alpha dropped): scatter the `>> 16`-narrowed
          // G/B/R planes into the source-width packed u16 RGB row and feed the
          // 3-channel high-bit tail (luma_u16 native — `NATIVE_LUMA16 = true`).
          if !packed_rgb_u16_resample_preflight(
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            rgb_stream_u16.as_ref().map_or(0, |s| s.next_y()),
            idx,
          )? {
            return Ok(());
          }
          let stream = packed_rgb_u16_resample_stream(rgb_stream_u16, plan, idx)?;
          let src_u16 = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
          // #289: `>> 16` narrow before binning (≤1-LSB, either range).
          gbr32_to_rgb_u16_row::<BE>(g_in, b_in, r_in, src_u16, w, use_simd);
          return packed_rgb_u16_resample_emit::<16, true>(
            stream,
            plan,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            src_u16,
            rgb_scratch,
            matrix,
            full_range,
            idx,
            use_simd,
          );
        }
        crate::resample::SpanKind::Filter => {
          // Premultiplied alpha has no filter analogue; surface the typed
          // `UnsupportedFilter` via the area tail's reject. Straight alpha:
          // filter all four native u16 channels independently when alpha
          // survives, else the 3-channel u16 filter for rgb-only outputs.
          if alpha_mode.is_premultiplied() {
            return packed_rgba_u16_resample::<16, true, false>(
              rgba_stream_u16,
              &mut None,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              rgb_scratch_u16,
              &mut std::vec::Vec::new(),
              w,
              plan,
              idx,
              use_simd,
              alpha_mode,
              matrix,
              full_range,
              |dst| gbra32_to_rgba_u16_row::<BE>(g_in, b_in, r_in, a_in, dst, w, use_simd),
              |_| {},
            );
          }
          if rgba.is_some() || rgba_u16.is_some() {
            return packed_rgba_u16_filter_resample::<16, true>(
              rgba_filter_stream_u16,
              resample_outputs,
              rgb,
              rgba,
              rgb_u16,
              rgba_u16,
              luma,
              luma_u16,
              hsv,
              rgba_scratch_u16,
              rgba_color_scratch_u16,
              rgb_scratch,
              w,
              plan,
              idx,
              use_simd,
              matrix,
              full_range,
              // #289: `>> 16` narrow before filtering (≤1-LSB, either range).
              |dst| gbra32_to_rgba_u16_row::<BE>(g_in, b_in, r_in, a_in, dst, w, use_simd),
            );
          }
          if !packed_rgb_u16_resample_preflight(
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            rgb_filter_stream_u16.as_ref().map_or(0, |s| s.next_y()),
            idx,
          )? {
            return Ok(());
          }
          let stream = packed_rgb_u16_filter_stream(rgb_filter_stream_u16, plan, idx)?;
          let src_u16 = source_rgb_u16_scratch(rgb_scratch_u16, w, plan)?;
          // #289: `>> 16` narrow before filtering (≤1-LSB, either range).
          gbr32_to_rgb_u16_row::<BE>(g_in, b_in, r_in, src_u16, w, use_simd);
          return packed_rgb_u16_resample_emit::<16, true>(
            stream,
            plan,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            src_u16,
            rgb_scratch,
            matrix,
            full_range,
            idx,
            use_simd,
          );
        }
      }
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

    // ---- u16 RGB / RGBA output (Strategy A+) -------------------------------
    let want_rgb_u16 = rgb_u16.is_some();
    let want_rgba_u16 = rgba_u16.is_some();

    if want_rgba_u16 && !want_rgb_u16 {
      // Standalone u16 RGBA — direct 4-channel kernel with real α.
      let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row =
        rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
      gbra32_to_rgba_u16_row::<BE>(g_in, b_in, r_in, a_in, rgba_u16_row, w, use_simd);
    } else if want_rgb_u16 {
      let rgb_u16_buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_plane_end =
        one_plane_end
          .checked_mul(3)
          .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
            w, h, 3,
          )))?;
      let rgb_plane_start = one_plane_start * 3;
      let rgb_u16_row = &mut rgb_u16_buf[rgb_plane_start..rgb_plane_end];
      gbr32_to_rgb_u16_row::<BE>(g_in, b_in, r_in, rgb_u16_row, w, use_simd);
      if want_rgba_u16 {
        // Strategy A+: expand RGB → RGBA (opaque), then overwrite α from the
        // source plane (native depth, `>> 16`). Scalar-only α scatter.
        let rgba_u16_buf = rgba_u16.as_deref_mut().unwrap();
        let rgba_u16_row =
          rgba_u16_plane_row_slice(rgba_u16_buf, one_plane_start, one_plane_end, w, h)?;
        expand_rgb_u16_to_rgba_u16_row::<16>(rgb_u16_row, rgba_u16_row, w);
        alpha_extract::copy_alpha_plane_u32::<BE>(a_in, rgba_u16_row, w);
      }
    }

    // ---- native-depth luma output (Q15 from G/B/R, no RGB staging) ----------
    if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
      gbr32_to_luma_u16_row::<BE>(
        g_in,
        b_in,
        r_in,
        &mut luma_u16_buf[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
    }

    // ---- u8 RGB / RGBA / luma / HSV output ---------------------------------
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_luma = luma.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_staging = want_rgb || want_luma || want_hsv;

    // RGBA-only fast path — direct 4-channel kernel with real α.
    if want_rgba && !need_rgb_staging {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gbra32_to_rgba_row::<BE>(g_in, b_in, r_in, a_in, rgba_row, w, use_simd);
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
    gbr32_to_rgb_row::<BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

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
      // Strategy A+: expand rgb_row → RGBA (opaque stub), then overwrite α
      // bytes from the source A plane (`>> 24`). Scalar-only α scatter.
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      alpha_extract::copy_alpha_plane_u32_to_u8::<BE>(a_in, rgba_row, w);
    }

    Ok(())
  }
}
