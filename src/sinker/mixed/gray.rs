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
  RowShapeMismatch, RowSlice, check_dimensions_match, frozen_outputs_check, rgb_row_buf_or_scratch,
  rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  resample::{AreaStream, OutOfSequenceRow, ResampleError, ResamplePlan},
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

impl<R> Gray8Sink for MixedSinker<'_, Gray8, R> {}

impl<R> PixelSink for MixedSinker<'_, Gray8, R> {
  type Input<'r> = Gray8Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the luma stream (lazily created in `process`,
    // so a direct-`process` caller that skips a fresh stream still gets
    // a correctly initialized first frame) and clear the frozen output
    // snapshot.
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
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
      plan,
      luma_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: Gray *is* a luma plane, so a single 1-channel
    // `AreaStream<u8>` bins the source Y row, then every attached output
    // derives from the binned Y exactly as the direct path does below —
    // luma copy, luma_u16 zero-extend, RGB broadcast, RGBA broadcast +
    // 0xFF, HSV (H=0/S=0/V=Y). Row-stage only.
    if let Some(plan) = plan.as_ref() {
      let full_range = row.full_range();
      return gray8_process_resampled(
        luma_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        row.y(),
        plan,
        idx,
        use_simd,
        full_range,
      );
    }

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

/// Row-stage fused downscale for [`Gray8`]: a single 1-channel
/// `AreaStream<u8>` bins the source Y plane (Gray *is* a luma plane —
/// luma is not re-derived from RGB), then every attached output derives
/// from the binned Y row using the very kernels the direct path uses,
/// so a resampled output equals the direct Gray8 path run over a frame
/// that already holds the binned Y. Atomic preflight: freeze, sequence
/// check, stream creation, and (for the colour group) scratch growth
/// all precede the first feed, so a failure mutates no caller output.
#[allow(clippy::too_many_arguments)]
fn gray8_process_resampled(
  luma_stream: &mut Option<AreaStream<u8>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  full_range: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();
  // The RGB kernel runs when RGB output is requested, or HSV is wanted
  // alongside RGBA (HSV-only and RGBA-only take dedicated fast paths).
  let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    idx,
  )?;
  // Sequence-check before allocating (mirrors the planar helpers): a
  // fresh stream expects row 0, so an out-of-sequence first row is
  // rejected before the stream or any scratch is created, and
  // AllocationFailed never masks OutOfSequenceRow. A no-output call has
  // no stream to sequence and stays a no-op regardless of the row index.
  let expected = luma_stream.as_ref().map_or(0, |stream| stream.next_y());
  let any_output = luma.is_some() || luma_u16.is_some() || want_rgb || want_rgba || want_hsv;
  if any_output && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(expected, idx),
    )));
  }
  if !any_output {
    return Ok(());
  }
  if luma_stream.is_none() {
    *luma_stream = Some(AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      1,
    )?);
  }
  // The RGB kernel writes into the user buffer when RGB is attached,
  // else into an output-width scratch shared with the HSV-from-RGB step.
  // Size it in the preflight so the feed closure stays infallible.
  if need_rgb_kernel && !want_rgb {
    let row_bytes =
      ow.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          ow,
          plan.out_h(),
          3,
        )))?;
    if rgb_scratch.len() < row_bytes {
      rgb_scratch
        .try_reserve_exact(row_bytes - rgb_scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(
            crate::resample::PlanGeometry::new(
              plan.src_w(),
              plan.src_h(),
              plan.out_w(),
              plan.out_h(),
            ),
          ))
        })?;
      rgb_scratch.resize(row_bytes, 0);
    }
  }

  let stream = luma_stream.as_mut().expect("created in the preflight");
  stream.feed_row(idx, y_row, use_simd, |oy, binned_y| {
    // Luma u8 — Gray8: Y IS luma; copy the binned row directly.
    if let Some(buf) = luma.as_deref_mut() {
      buf[oy * ow..(oy + 1) * ow].copy_from_slice(binned_y);
    }
    // Luma u16 — zero-extend the binned Y bytes to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      y_plane_to_luma_u16_row(binned_y, &mut buf[oy * ow..(oy + 1) * ow], ow, use_simd);
    }

    // Standalone RGBA fast path — no RGB or HSV requested.
    if want_rgba && !want_rgb && !want_hsv {
      let buf = rgba.as_deref_mut().unwrap();
      gray8_to_rgba_row(
        binned_y,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
        use_simd,
        full_range,
      );
      return;
    }

    // Standalone HSV fast path — H=0/S=0/V=Y with no RGB computation.
    if want_hsv && !want_rgb && !want_rgba {
      let hsv = hsv.as_mut().unwrap();
      let (hp, sp, vp) = hsv.hsv();
      gray8_to_hsv_row(
        binned_y,
        &mut hp[oy * ow..(oy + 1) * ow],
        &mut sp[oy * ow..(oy + 1) * ow],
        &mut vp[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
        full_range,
      );
      return;
    }

    if !need_rgb_kernel {
      return;
    }

    // RGB kernel once — into the user buffer if attached, else scratch.
    if let Some(buf) = rgb.as_deref_mut() {
      let rgb_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      gray8_to_rgb_row(binned_y, rgb_row, ow, use_simd, full_range);
      if let Some(hsv) = hsv.as_mut() {
        let (hp, sp, vp) = hsv.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
      }
    } else {
      let rgb_row = &mut rgb_scratch[..ow * 3];
      gray8_to_rgb_row(binned_y, rgb_row, ow, use_simd, full_range);
      if let Some(hsv) = hsv.as_mut() {
        let (hp, sp, vp) = hsv.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
      }
    }
  })?;

  Ok(())
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

impl<R, const BE: bool> Gray16Sink<BE> for MixedSinker<'_, Gray16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Gray16<BE>, R> {
  type Input<'r> = Gray16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the u16 luma stream (lazily created in
    // `process`) and clear the frozen output snapshot, mirroring the
    // Gray8 path so a reused resampling sink re-sequences from row 0.
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
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
      plan,
      luma_stream_u16,
      luma_scratch_u16,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: Gray16 *is* a u16 luma plane, so the wire row
    // converts to a source-width host-native u16 luma plane (the same
    // kernel the direct `luma_u16` path uses), a single 1-channel
    // `AreaStream<u16>` bins it at u16 precision, then every attached
    // output derives from each finalized binned u16 luma row exactly as
    // the direct path does below. Row-stage only.
    if let Some(plan) = plan.as_ref() {
      return gray16_process_resampled::<BE>(
        luma_stream_u16,
        luma_scratch_u16,
        resample_outputs,
        rgb,
        rgb_u16,
        rgba,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        row.y(),
        plan,
        w,
        idx,
        use_simd,
        full_range,
      );
    }

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

/// Row-stage fused downscale for [`Gray16`]: the wire row converts to a
/// source-width **host-native** `u16` luma plane via the very kernel the
/// direct `luma_u16` path uses (`gray16_to_luma_u16_row::<BE>` with the
/// source wire `BE`), a single 1-channel `AreaStream<u16>` bins it at
/// u16 precision, then every attached output derives from each finalized
/// binned u16 luma row using the direct kernels. Because the binned row
/// is already host-native, those derive kernels run with
/// `HOST_NATIVE_BE = cfg!(target_endian = "big")` — `::<HOST_NATIVE_BE>`
/// is the identity recovery for an already-host-native sample, so on a BE
/// host the source→luma swap and the luma→output no-op are not double-
/// swapped. The result equals the direct Gray16 path run over a frame
/// that already holds the binned u16 luma. Atomic preflight: freeze,
/// sequence check, stream creation, and (for the colour group) source +
/// scratch growth all precede the first feed, so a failure mutates no
/// caller output.
#[allow(clippy::too_many_arguments)]
fn gray16_process_resampled<const BE: bool>(
  luma_stream_u16: &mut Option<AreaStream<u16>>,
  luma_scratch_u16: &mut std::vec::Vec<u16>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba: &mut Option<&mut [u8]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<mediaframe::source::HsvFrameMut<'_>>,
  y_row: &[u16],
  plan: &ResamplePlan,
  w: usize,
  idx: usize,
  use_simd: bool,
  full_range: bool,
) -> Result<(), MixedSinkerError> {
  // The binned u16 luma row is host-native; the direct kernels recover
  // an already-host-native sample with `::<HOST_NATIVE_BE>` (a no-op
  // swap), matching the direct path's `::<BE>` applied to a wire sample.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  let ow = plan.out_w();
  let want_rgb = rgb.is_some();
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba = rgba.is_some();
  let want_rgba_u16 = rgba_u16.is_some();
  let want_hsv = hsv.is_some();
  // The u8 RGB kernel runs only when RGB output is requested; HSV (with
  // or without RGBA) and standalone RGBA derive directly from luma, so
  // the resample path needs no RGB scratch.
  let need_rgb_kernel = want_rgb;

  frozen_outputs_check(
    resample_outputs,
    luma,
    luma_u16,
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    idx,
  )?;
  // Sequence-check before allocating (mirrors the Gray8 / planar
  // helpers): a fresh stream expects row 0, so an out-of-sequence first
  // row is rejected before the stream or the source-luma staging is
  // created, and AllocationFailed never masks OutOfSequenceRow. A
  // no-output call has no stream to sequence and stays a no-op regardless
  // of the row index.
  let expected = luma_stream_u16.as_ref().map_or(0, |stream| stream.next_y());
  let any_output = luma.is_some()
    || luma_u16.is_some()
    || want_rgb
    || want_rgb_u16
    || want_rgba
    || want_rgba_u16
    || want_hsv;
  if any_output && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(expected, idx),
    )));
  }
  if !any_output {
    return Ok(());
  }
  // Recoverable source-width host-native u16 luma staging, allocated
  // before any caller-buffer write.
  if luma_scratch_u16.len() < w {
    luma_scratch_u16
      .try_reserve_exact(w - luma_scratch_u16.len())
      .map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    luma_scratch_u16.resize(w, 0);
  }
  if luma_stream_u16.is_none() {
    *luma_stream_u16 = Some(AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      1,
    )?);
  }
  // Convert the wire Gray16 row to host-native u16 luma — the source
  // wire `::<BE>`, the same kernel the direct `luma_u16` path uses.
  let src_luma = &mut luma_scratch_u16[..w];
  gray16_to_luma_u16_row::<BE>(y_row, src_luma, w, use_simd);

  let stream = luma_stream_u16.as_mut().expect("created in the preflight");
  stream.feed_row(idx, src_luma, use_simd, |oy, binned_y| {
    // Luma u16 — host-native pass-through of the binned u16 luma.
    if let Some(buf) = luma_u16.as_deref_mut() {
      gray16_to_luma_u16_row::<HOST_NATIVE_BE>(
        binned_y,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
    // Luma u8 — `>> 8` narrowing of the binned u16 luma.
    if let Some(buf) = luma.as_deref_mut() {
      gray16_to_luma_row::<HOST_NATIVE_BE>(
        binned_y,
        &mut buf[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }

    // u16 RGB / RGBA (Strategy A) — native 16-bit broadcast.
    if want_rgba_u16 && !want_rgb_u16 {
      let buf = rgba_u16.as_deref_mut().unwrap();
      gray16_to_rgba_u16_row::<HOST_NATIVE_BE>(
        binned_y,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
        use_simd,
        full_range,
      );
    } else if want_rgb_u16 {
      let buf = rgb_u16.as_deref_mut().unwrap();
      let rgb_u16_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
      gray16_to_rgb_u16_row::<HOST_NATIVE_BE>(binned_y, rgb_u16_row, ow, use_simd, full_range);
      if let Some(buf) = rgba_u16.as_deref_mut() {
        expand_rgb_u16_to_rgba_u16_row::<16>(
          rgb_u16_row,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
        );
      }
    }

    // Standalone u8 RGBA fast path — no RGB or HSV requested.
    if want_rgba && !need_rgb_kernel && !want_hsv {
      let buf = rgba.as_deref_mut().unwrap();
      gray16_to_rgba_row::<HOST_NATIVE_BE>(
        binned_y,
        &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
        ow,
        use_simd,
        full_range,
      );
      return;
    }

    // Standalone HSV fast path — H=0/S=0/V=Y>>8 with no RGB computation
    // (plus an optional standalone RGBA, matching the direct path).
    if want_hsv && !want_rgb {
      let hsv = hsv.as_mut().unwrap();
      let (hp, sp, vp) = hsv.hsv();
      gray16_to_hsv_row::<HOST_NATIVE_BE>(
        binned_y,
        &mut hp[oy * ow..(oy + 1) * ow],
        &mut sp[oy * ow..(oy + 1) * ow],
        &mut vp[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
        full_range,
      );
      if let Some(buf) = rgba.as_deref_mut() {
        gray16_to_rgba_row::<HOST_NATIVE_BE>(
          binned_y,
          &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow],
          ow,
          use_simd,
          full_range,
        );
      }
      return;
    }

    if !need_rgb_kernel {
      return;
    }

    // Reached only when RGB is attached (need_rgb_kernel == want_rgb), so
    // the kernel writes the user buffer; HSV-from-RGB and the RGBA
    // fan-out follow, exactly as the direct path's RGB-kernel branch does.
    let buf = rgb
      .as_deref_mut()
      .expect("need_rgb_kernel implies RGB is attached");
    let rgb_row = &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow];
    gray16_to_rgb_row::<HOST_NATIVE_BE>(binned_y, rgb_row, ow, use_simd, full_range);
    if let Some(hsv) = hsv.as_mut() {
      let (hp, sp, vp) = hsv.hsv();
      rgb_to_hsv_row(
        rgb_row,
        &mut hp[oy * ow..(oy + 1) * ow],
        &mut sp[oy * ow..(oy + 1) * ow],
        &mut vp[oy * ow..(oy + 1) * ow],
        ow,
        use_simd,
      );
    }
    if let Some(buf) = rgba.as_deref_mut() {
      expand_rgb_to_rgba_row(rgb_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
    }
  })?;

  Ok(())
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
