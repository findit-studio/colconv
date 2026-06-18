//! Sinker impls for packed YUV 4:2:2 (8-bit) source formats — Tier 3,
//! Ship 10.
//!
//! Source family covered here:
//! - [`Yuyv422`] — `Y0, U0, Y1, V0, …` (FFmpeg `yuyv422` / YUY2).
//! - [`Uyvy422`] — `U0, Y0, V0, Y1, …` (FFmpeg `uyvy422` / UYVY).
//! - [`Yvyu422`] — `Y0, V0, Y1, U0, …` (FFmpeg `yvyu422` / YVYU).
//!
//! All three formats carry one packed plane of `2 * width` bytes per
//! row. The differences are pure byte permutation within each
//! 4-byte / 2-pixel block; the three dispatchers
//! ([`yuyv422_to_rgb_row`], [`uyvy422_to_rgb_row`],
//! [`yvyu422_to_rgb_row`] and the matching `_to_rgba_row` /
//! `_to_luma_row` siblings) hide that permutation behind a single
//! const-generic kernel template.
//!
//! Outputs map to the sink's standard channels:
//! - `with_rgb` / `with_rgba` — packed YUV → RGB Q15 pipeline (full
//!   `ColorMatrix` + range support inherited from the row); RGBA
//!   alpha is forced to `0xFF` (the source has no alpha channel).
//! - `with_luma` — extracts the Y bytes from the packed plane via
//!   the dedicated luma kernel (much cheaper than a full YUV→RGB
//!   pass).
//! - `with_hsv` — stages an internal RGB scratch (or the user's RGB
//!   buffer if attached) and runs the existing `rgb_to_hsv_row`
//!   kernel.
//!
//! When both RGB and RGBA outputs are requested, the RGBA plane is
//! derived from the just-computed RGB row via
//! [`expand_rgb_to_rgba_row`] (Strategy A — memory-bound copy + 0xFF
//! alpha pad) instead of running a second YUV→RGB kernel. When only
//! RGBA is wanted, the dedicated `_to_rgba_row` kernel writes the
//! RGBA buffer directly without staging RGB.
//!
//! ## Fused downscale
//!
//! Under a non-identity [`ResamplePlan`] the source rows feed the
//! shared area-resample engine through [`packed_yuv422_dual_resample`],
//! mirroring the planar YUV row-stage tier:
//! - **luma / luma_u16** de-interleave the **Y bytes** into a
//!   source-width row (the format's own `*_to_luma_row` kernel — the
//!   exact Y→luma derivation the direct path uses) and area-bin them
//!   through a 1-channel stream. Luma is taken from Y, *not* re-derived
//!   from converted RGB: under saturated / clamped chroma the two
//!   diverge, and the direct path takes luma from Y.
//! - **rgb / rgba / hsv** convert each packed row to canonical RGB at
//!   source width (the format's own fused `*_to_rgb_row` kernel does the
//!   chroma de-interleave + 4:2:2 horizontal upsample in-register,
//!   exactly as the identity path) and area-bin that RGB row through the
//!   3-channel stream, deriving every colour output from each finalized
//!   output row.
//!
//! RGB is byte-identical to an `Rgb24` area-resample of the
//! identity-converted frame; luma equals the area-downscaled Y plane.

use super::{
  FrozenOutputs, GeometryOverflow, HsvFrameMut, InsufficientBuffer, MixedSinker, MixedSinkerError,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  frozen_outputs_check, rgb_row_buf_or_scratch, rgba_plane_row_slice, source_luma_scratch,
  source_rgb_scratch,
};
use crate::{
  PixelSink,
  resample::{AreaStream, OutOfSequenceRow, ResampleError, ResamplePlan},
  row::{
    expand_rgb_to_rgba_row, rgb_to_hsv_row, uyvy422_to_luma_row, uyvy422_to_luma_u16_row,
    uyvy422_to_rgb_row, uyvy422_to_rgba_row, yuyv422_to_luma_row, yuyv422_to_luma_u16_row,
    yuyv422_to_rgb_row, yuyv422_to_rgba_row, yvyu422_to_luma_row, yvyu422_to_luma_u16_row,
    yvyu422_to_rgb_row, yvyu422_to_rgba_row,
  },
  source::{
    Uyvy422, Uyvy422Row, Uyvy422Sink, Yuyv422, Yuyv422Row, Yuyv422Sink, Yvyu422, Yvyu422Row,
    Yvyu422Sink,
  },
};

/// Row-stage fused downscale shared by the three packed YUV 4:2:2
/// formats. Mirrors the planar YUV dual-stream path: **luma / luma_u16
/// area-resample the de-interleaved Y bytes directly** (the YUV luma
/// contract — luma is *not* re-derived from converted RGB), while RGB /
/// RGBA / HSV bin a converted source-width RGB row.
///
/// `deinterleave_y` fills a source-width scratch with the Y samples
/// pulled from the packed row (the format's own `*_to_luma_row`
/// kernel), and runs only when a luma output is attached. `convert_rgb`
/// fills a source-width RGB scratch from the packed row (the format's
/// own `*_to_rgb_row` kernel), and runs only when a colour output is
/// attached.
///
/// Atomic preflight: every fallible step (freeze, sequence check,
/// stream creation, scratch growth + conversion) precedes the first
/// feed, so a failure mutates no caller output. Sequencing is checked
/// before any allocation, so an out-of-sequence row is rejected without
/// allocating and `AllocationFailed` never masks `OutOfSequenceRow`; a
/// no-output call is a true no-op regardless of the row index.
#[allow(clippy::too_many_arguments)]
fn packed_yuv422_dual_resample(
  luma_stream: &mut Option<AreaStream<u8>>,
  rgb_stream: &mut Option<AreaStream<u8>>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  luma_scratch: &mut std::vec::Vec<u8>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  deinterleave_y: impl FnOnce(&mut [u8]),
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (packed YUV 4:2:2 8-bit is not routed to the filter
  // path): reject a filter plan before any work, so the plan's empty area
  // spans never reach an area stream.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

  // Single sequence check, on whichever stream is fed every row (all
  // attached streams advance in lockstep). A no-output call (neither luma
  // nor color) has no stream to sequence and stays a no-op regardless of
  // the row index — returned before the freeze so it stores no snapshot a
  // later attach-then-retry would trip on.
  let expected = if need_luma {
    luma_stream.as_ref().map_or(0, |stream| stream.next_y())
  } else if need_color {
    rgb_stream.as_ref().map_or(0, |stream| stream.next_y())
  } else {
    return Ok(());
  };
  // First row: reject an out-of-sequence row BEFORE the freeze, so a
  // rejected first row stores no snapshot that would poison a retry. On a
  // later row the freeze runs first (below), so a mid-frame output-set
  // change is reported as ResampleOutputsChanged rather than masked by a
  // freshly-attached stream's row-0 sequence mismatch (attaching a luma or
  // colour output mid-frame spins that stream fresh at row 0).
  if resample_outputs.is_none() && expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(expected, idx),
    )));
  }
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
    &None,
    idx,
  )?;
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(expected, idx),
    )));
  }
  if need_luma && luma_stream.is_none() {
    *luma_stream = Some(AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      1,
    )?);
  }
  if need_color && rgb_stream.is_none() {
    *rgb_stream = Some(AreaStream::new(
      plan.h(),
      plan.v(),
      plan.src_w(),
      plan.src_h(),
      3,
    )?);
  }
  // Stage the source-width rows (both fallible growths run before the
  // feeds, keeping the call atomic). The Y row uses its own scratch so
  // it does not collide with the colour stream's RGB scratch.
  let luma_row = if need_luma {
    let scratch = source_luma_scratch(luma_scratch, w, plan)?;
    deinterleave_y(scratch);
    Some(scratch)
  } else {
    None
  };
  let color_row = if need_color {
    let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb(scratch);
    Some(scratch)
  } else {
    None
  };

  if let Some(y_row) = luma_row {
    let stream = luma_stream.as_mut().expect("created in the preflight");
    stream.feed_row(idx, y_row, use_simd, |oy, out_row| {
      if let Some(buf) = luma.as_deref_mut() {
        buf[oy * ow..(oy + 1) * ow].copy_from_slice(out_row);
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(out_row) {
          *dst = src as u16;
        }
      }
    })?;
  }

  if let Some(scratch) = color_row {
    let stream = rgb_stream.as_mut().expect("created in the preflight");
    stream.feed_row(idx, scratch, use_simd, |oy, out_row| {
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_row);
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        rgb_to_hsv_row(
          out_row,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        expand_rgb_to_rgba_row(out_row, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
      }
    })?;
  }

  Ok(())
}

// ---- Yuyv422 impl ------------------------------------------------------

impl<'a, R> MixedSinker<'a, Yuyv422, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
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

  /// Attaches a **`u16`** luma output buffer. Y bytes are zero-extended
  /// to u16 (`out[x] = Y_byte as u16`). Length in u16 **elements**
  /// (`width x height`).
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

impl<R> Yuyv422Sink for MixedSinker<'_, Yuyv422, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuyv422, R> {
  type Input<'r> = Yuyv422Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    // New frame: restart the row-stage streams (lazily created in
    // `process`, so a direct-`process` caller that skips `begin_frame`
    // still gets a correctly initialized first frame) and drop the
    // frozen output set.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuyv422Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }

    let packed_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.yuyv().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Yuyv422Packed,
        idx,
        packed_expected,
        row.yuyv().len(),
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
      luma_scratch,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      resample_outputs,
      ..
    } = self;
    let packed = row.yuyv();

    // Non-identity plan: feed the shared packed-YUV dual-stream tail —
    // luma de-interleaves the Y bytes and bins them; colour converts the
    // packed row to RGB and bins that. Freeze + sequence-check before
    // staging, so a no-output sink stays a no-op and an out-of-sequence
    // row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      return packed_yuv422_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        luma_scratch,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| yuyv422_to_luma_row(packed, scratch, w, use_simd),
        |scratch| yuyv422_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma u8 — extract Y bytes from packed plane via dedicated kernel.
    if let Some(luma) = luma.as_deref_mut() {
      yuyv422_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    // Luma u16 — zero-extend Y bytes to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      yuyv422_to_luma_u16_row(
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
      yuyv422_to_rgba_row(
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
    yuyv422_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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

// ---- Uyvy422 impl ------------------------------------------------------

impl<'a, R> MixedSinker<'a, Uyvy422, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
  ///
  /// See [`MixedSinker::<Yuyv422>::with_rgba`] for the same rationale
  /// and constraints; UYVY differs only in byte position (Y in odd
  /// vs even slots).
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

  /// Attaches a **`u16`** luma output buffer. Y bytes (at offset 1 of
  /// each UYVY pair) are zero-extended to u16. Length in u16 **elements**
  /// (`width x height`).
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

impl<R> Uyvy422Sink for MixedSinker<'_, Uyvy422, R> {}

impl<R> PixelSink for MixedSinker<'_, Uyvy422, R> {
  type Input<'r> = Uyvy422Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Uyvy422Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }

    let packed_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.uyvy().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Uyvy422Packed,
        idx,
        packed_expected,
        row.uyvy().len(),
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
      luma_scratch,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      resample_outputs,
      ..
    } = self;
    let packed = row.uyvy();

    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      return packed_yuv422_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        luma_scratch,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| uyvy422_to_luma_row(packed, scratch, w, use_simd),
        |scratch| uyvy422_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      uyvy422_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    // Luma u16 — zero-extend Y bytes to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      uyvy422_to_luma_u16_row(
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

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      uyvy422_to_rgba_row(
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
    uyvy422_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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

// ---- Yvyu422 impl ------------------------------------------------------

impl<'a, R> MixedSinker<'a, Yvyu422, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled
  /// with constant `0xFF` (the source has no alpha channel).
  ///
  /// See [`MixedSinker::<Yuyv422>::with_rgba`] for the same rationale
  /// and constraints; YVYU differs only in chroma byte order (V
  /// before U).
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

  /// Attaches a **`u16`** luma output buffer. Y bytes are zero-extended
  /// to u16 (`out[x] = Y_byte as u16`). Length in u16 **elements**
  /// (`width x height`).
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

impl<R> Yvyu422Sink for MixedSinker<'_, Yvyu422, R> {}

impl<R> PixelSink for MixedSinker<'_, Yvyu422, R> {
  type Input<'r> = Yvyu422Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yvyu422Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }

    let packed_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.yvyu().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Yvyu422Packed,
        idx,
        packed_expected,
        row.yvyu().len(),
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
      luma_scratch,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      resample_outputs,
      ..
    } = self;
    let packed = row.yvyu();

    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      return packed_yuv422_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        luma_scratch,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| yvyu422_to_luma_row(packed, scratch, w, use_simd),
        |scratch| yvyu422_to_rgb_row(packed, scratch, w, matrix, full_range, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      yvyu422_to_luma_row(
        packed,
        &mut luma[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }
    // Luma u16 — zero-extend Y bytes to u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      yvyu422_to_luma_u16_row(
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

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yvyu422_to_rgba_row(
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
    yvyu422_to_rgb_row(packed, rgb_row, w, row.matrix(), row.full_range(), use_simd);

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
