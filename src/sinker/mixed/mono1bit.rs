//! [`PixelSink`] implementations for Monoblack and Monowhite sources.
//!
//! Both formats are 1-bit-per-pixel achromatic bilevel: each source bit
//! expands to a u8 luma of `0` or `255` (Monoblack: bit=1 → 255;
//! Monowhite: bit=0 → 255). Every output is a broadcast of that luma —
//! `with_rgb` / `with_rgba` / `with_rgb_u16` / `with_rgba_u16` replicate
//! Y to R=G=B (alpha forced opaque), `with_luma` copies it,
//! `with_luma_u16` zero-extends it, and `with_hsv` emits H=0, S=0, V=Y
//! (achromatic convention). There is no chroma matrix and no palette, so
//! the area-resampled path bins the **expanded** luma plane directly via
//! a single-channel [`AreaStream<u8>`](crate::resample::AreaStream) and
//! derives every attached output from each finalized binned luma row —
//! identical to the direct path with the binned mean standing in for the
//! per-pixel expanded value.

use super::{
  InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange, RowShapeMismatch,
  RowSlice, check_dimensions_match, frozen_outputs_check, source_luma_scratch,
};
use crate::{
  PixelSink,
  resample::{AreaStream, OutOfSequenceRow, ResampleError, ResamplePlan},
  row,
  source::{Monoblack, MonoblackRow, MonoblackSink, Monowhite, MonowhiteRow, MonowhiteSink},
};
use mediaframe::source::HsvFrameMut;

/// Fused area-downscale tail shared by Monoblack and Monowhite. The
/// closure `expand_luma` fills a source-width `u8` scratch with the
/// expanded 0/255 luma of the current 1-bit source row (the same
/// `mono*_to_luma_row` kernel the direct path uses); that luma plane is
/// binned through a single-channel area stream, and every attached
/// output is derived from each finalized binned luma row — a copy for
/// luma, a zero-extend for luma_u16, a broadcast to R=G=B for the RGB
/// outputs (alpha forced opaque), and H=0/S=0/V=Y for HSV — mirroring
/// the direct mono kernels exactly.
///
/// Atomic preflight: the output set is frozen and stream sequencing
/// checked **before** any scratch allocation or stream feed, so a
/// failed call mutates no caller output and a no-output sink stays a
/// legal no-op.
#[allow(clippy::too_many_arguments)]
fn mono_luma_resample(
  luma_stream: &mut Option<AreaStream<u8>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  scratch: &mut std::vec::Vec<u8>,
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  expand_luma: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let any_output = luma.is_some()
    || luma_u16.is_some()
    || rgb.is_some()
    || rgba.is_some()
    || rgb_u16.is_some()
    || rgba_u16.is_some()
    || hsv.is_some();

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
    hsv,
    idx,
  )?;

  // Sequence-check before allocating (mirrors the planar / packed-RGB
  // helpers): a fresh stream expects row 0, so an out-of-sequence first
  // row is rejected before the output-width buffers or the source-width
  // scratch are created, and AllocationFailed never masks
  // OutOfSequenceRow. A no-output call has no stream to sequence and
  // stays a no-op regardless of the row index.
  if any_output {
    let expected = luma_stream.as_ref().map_or(0, |stream| stream.next_y());
    if expected != idx {
      return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
        OutOfSequenceRow::new(expected, idx),
      )));
    }
  } else {
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

  // Expand the 1-bit source row to source-width 0/255 luma in the shared
  // scratch (fallible growth runs before the feed, keeping the call
  // atomic), then bin it through the single-channel stream.
  let luma_row = source_luma_scratch(scratch, w, plan)?;
  expand_luma(luma_row);

  let stream = luma_stream.as_mut().expect("created above");
  stream.feed_row(idx, luma_row, use_simd, |oy, out_row| {
    if let Some(buf) = luma.as_deref_mut() {
      buf[oy * ow..(oy + 1) * ow].copy_from_slice(out_row);
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      for (dst, &y) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(out_row) {
        *dst = y as u16;
      }
    }
    if let Some(buf) = rgb.as_deref_mut() {
      for (px, &y) in buf[oy * 3 * ow..(oy + 1) * 3 * ow]
        .chunks_exact_mut(3)
        .zip(out_row)
      {
        px[0] = y;
        px[1] = y;
        px[2] = y;
      }
    }
    if let Some(buf) = rgba.as_deref_mut() {
      for (px, &y) in buf[oy * 4 * ow..(oy + 1) * 4 * ow]
        .chunks_exact_mut(4)
        .zip(out_row)
      {
        px[0] = y;
        px[1] = y;
        px[2] = y;
        px[3] = 0xFF;
      }
    }
    if let Some(buf) = rgb_u16.as_deref_mut() {
      for (px, &y) in buf[oy * 3 * ow..(oy + 1) * 3 * ow]
        .chunks_exact_mut(3)
        .zip(out_row)
      {
        let y16 = y as u16;
        px[0] = y16;
        px[1] = y16;
        px[2] = y16;
      }
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      for (px, &y) in buf[oy * 4 * ow..(oy + 1) * 4 * ow]
        .chunks_exact_mut(4)
        .zip(out_row)
      {
        let y16 = y as u16;
        px[0] = y16;
        px[1] = y16;
        px[2] = y16;
        px[3] = 0x00FF;
      }
    }
    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      let span = oy * ow..(oy + 1) * ow;
      v[span.clone()].copy_from_slice(out_row);
      for px in &mut h[span.clone()] {
        *px = 0;
      }
      for px in &mut s[span] {
        *px = 0;
      }
    }
  })?;

  Ok(())
}

// ---- Monoblack impl ---------------------------------------------------------

impl<'a, R> MixedSinker<'a, Monoblack, R> {
  /// Attaches a packed **`u8`** RGBA output buffer.
  ///
  /// Length is measured in `u8` **bytes**: minimum `width * height * 4`.
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

  /// Attaches a packed **`u16`** RGB output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 3`.
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

  /// Attaches a packed **`u16`** RGBA output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 4`.
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

  /// Attaches a planar **`u16`** luma output buffer.
  ///
  /// Luma is derived from RGB via BT.709 weights (by default).
  /// Length: minimum `width * height`.
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

impl<R> MonoblackSink for MixedSinker<'_, Monoblack, R> {}

impl<R> PixelSink for MixedSinker<'_, Monoblack, R> {
  type Input<'i> = MonoblackRow<'i>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Self::Input<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let min_bytes = w.div_ceil(8);
    if row.data().len() < min_bytes {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        min_bytes,
        row.data().len(),
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
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      plan,
      luma_stream,
      resample_outputs,
      rgb_scratch,
      ..
    } = self;

    // Non-identity plan: expand the 1-bit source row to source-width
    // 0/255 luma (the same Monoblack→luma kernel the direct path uses),
    // bin that luma plane, and derive every output from each finalized
    // binned luma row. The mean is of the EXPANDED luma — expand then
    // bin, the resample oracle for an achromatic source.
    if let Some(plan) = plan.as_ref() {
      let data = row.data();
      return mono_luma_resample(
        luma_stream,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| row::monoblack_to_luma_row(data, scratch, w, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(buf) = rgb.as_deref_mut() {
      row::monoblack_to_rgb_or_rgba_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      row::monoblack_to_rgb_or_rgba_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      row::monoblack_to_rgb_u16_or_rgba_u16_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      row::monoblack_to_rgb_u16_or_rgba_u16_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma.as_deref_mut() {
      row::monoblack_to_luma_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma_u16.as_deref_mut() {
      row::monoblack_to_luma_u16_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      row::monoblack_to_hsv_row(
        row.data(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    Ok(())
  }
}

// ---- Monowhite impl ---------------------------------------------------------

impl<'a, R> MixedSinker<'a, Monowhite, R> {
  /// Attaches a packed **`u8`** RGBA output buffer.
  ///
  /// Length is measured in `u8` **bytes**: minimum `width * height * 4`.
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

  /// Attaches a packed **`u16`** RGB output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 3`.
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

  /// Attaches a packed **`u16`** RGBA output buffer.
  ///
  /// Length is measured in `u16` **elements** (not bytes): minimum
  /// `width * height * 4`.
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

  /// Attaches a planar **`u16`** luma output buffer.
  ///
  /// Luma is derived from RGB via BT.709 weights (by default).
  /// Length: minimum `width * height`.
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

impl<R> MonowhiteSink for MixedSinker<'_, Monowhite, R> {}

impl<R> PixelSink for MixedSinker<'_, Monowhite, R> {
  type Input<'i> = MonowhiteRow<'i>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Self::Input<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let min_bytes = w.div_ceil(8);
    if row.data().len() < min_bytes {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        min_bytes,
        row.data().len(),
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
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      plan,
      luma_stream,
      resample_outputs,
      rgb_scratch,
      ..
    } = self;

    // Non-identity plan: same expand-then-bin oracle as Monoblack, with
    // Monowhite's inverted polarity baked into the luma kernel.
    if let Some(plan) = plan.as_ref() {
      let data = row.data();
      return mono_luma_resample(
        luma_stream,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        w,
        plan,
        idx,
        use_simd,
        |scratch| row::monowhite_to_luma_row(data, scratch, w, use_simd),
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(buf) = rgb.as_deref_mut() {
      row::monowhite_to_rgb_or_rgba_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      row::monowhite_to_rgb_or_rgba_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      row::monowhite_to_rgb_u16_or_rgba_u16_row::<false>(
        row.data(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      row::monowhite_to_rgb_u16_or_rgba_u16_row::<true>(
        row.data(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma.as_deref_mut() {
      row::monowhite_to_luma_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma_u16.as_deref_mut() {
      row::monowhite_to_luma_u16_row(
        row.data(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      let (h, s, v) = hsv.hsv();
      row::monowhite_to_hsv_row(
        row.data(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    Ok(())
  }
}
