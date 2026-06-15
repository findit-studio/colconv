//! 8-bit planar YUV `MixedSinker` impls: Yuv410p / Yuv420p / Yuv422p / Yuv444p / Yuv440p.

use super::{
  GeometryOverflow, HsvFrameMut, InsufficientBuffer, MixedSinker, MixedSinkerError,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  frozen_outputs_check, planar_resample::planar_dual_resample, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  ColorMatrix, PixelSink,
  resample::{AreaStream, OutOfSequenceRow, PlanGeometry, ResampleError, ResamplePlan, try_zeroed},
  row::*,
  source::*,
};

// ---- Yuv420p impl --------------------------------------------------------

impl<'a, R> MixedSinker<'a, Yuv420p, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — calling `with_rgba` on a sink that doesn't (e.g.
  /// [`MixedSinker<Nv12>`] today) is a compile error rather than a
  /// silent no‑op that would leave the caller's buffer stale while
  /// [`Self::produces_rgba`] returned `true`. The compile-time
  /// scoping is load-bearing: if a future format adds RGBA, it must
  /// add its own impl block here, which both wires the new path and
  /// prevents accidental cross-format leakage.
  ///
  /// The fourth byte per pixel is alpha. [`Yuv420p`] has no alpha
  /// plane, so every alpha byte is filled with `0xFF` (opaque).
  /// Future YUVA source impls will copy alpha through from the
  /// source plane.
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
  /// 32‑bit targets when the product overflows.
  ///
  /// ```compile_fail
  /// // Attaching RGBA to a sink that doesn't write it is rejected
  /// // at compile time. `Bayer` (RAW Bayer-mosaic) has no RGBA path —
  /// // there's no inherent alpha channel and the format demosaics to
  /// // RGB only. Once / if a future PR adds RGBA, the negative example
  /// // here moves to the next not‑yet‑wired format.
  /// use colconv::{sinker::MixedSinker, raw::Bayer};
  /// let mut buf = vec![0u8; 16 * 8 * 4];
  /// let _ = MixedSinker::<Bayer>::new(16, 8).with_rgba(&mut buf);
  /// ```
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16` (i.e. each output element equals
  /// `y_byte as u16`). Length is measured in `u16` elements
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

impl<R> PixelSink for MixedSinker<'_, Yuv420p, R> {
  type Input<'r> = Yuv420pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Reject odd-width sinkers up front — the underlying row
    // primitives assume `width & 1 == 0` and would panic on the
    // first `process` call otherwise (`MixedSinker::new` is
    // infallible and accepts any width).
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the row-stage streams (the streams are
    // lazily created in `process`, so a direct-`process` caller that
    // skips `begin_frame` still gets a correctly initialized first
    // frame).
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    if let Some(native) = self.native_420.as_mut() {
      native.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuv420pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense in depth: `begin_frame` already validated frame‑level
    // dimensions, so these checks are unreachable from the walker.
    // They guard direct `process` callers (hand-crafted rows, row
    // replay) from handing a wrong-shaped row or out-of-range index
    // to unsafe SIMD kernels. Report the offending slice length and
    // row index directly — don't reuse `DimensionMismatch`, whose
    // `frame_w` / `frame_h` fields would be meaningless here.
    //
    // Odd-width check first: the row primitives assume
    // `width & 1 == 0` and would panic past this point. Keeping the
    // check here (and in `begin_frame`) preserves the no-panic
    // contract for direct `process` callers that skip `begin_frame`.
    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf,
        idx,
        w / 2,
        row.v_half().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Split-borrow so the `rgb_scratch` path and the `hsv` write don't
    // collide with the `rgb` read-after-write chain below.
    let Self {
      rgb,
      rgba,
      luma,
      luma_u16,
      hsv,
      rgb_scratch,
      plan,
      rgb_stream,
      luma_stream,
      resample_outputs,
      native,
      native_420,
      ..
    } = self;

    // Non-identity plan: the native tier bins the Y/U/V planes at
    // output resolution and converts once per output row; the
    // row-stage tier converts this source row at source width, then
    // area-streams it. `with_native(false)` forces the latter.
    if let Some(plan) = plan.as_ref() {
      if *native {
        return yuv420p_process_native(
          plan,
          native_420,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.u_half(),
          row.v_half(),
          row.matrix(),
          row.full_range(),
          idx,
          w,
          h,
          use_simd,
        );
      }
      return yuv420p_process_resampled(
        plan,
        rgb_stream,
        luma_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        row.y(),
        row.u_half(),
        row.v_half(),
        row.matrix(),
        row.full_range(),
        idx,
        w,
        use_simd,
      );
    }

    // Single-plane row ranges are guaranteed not to overflow: `idx <
    // h` and `with_luma` / `with_hsv` validated `w x h x 1` fits
    // usize, so `(idx + 1) * w ≤ h * w` fits too. The `x 3` RGB
    // ranges are only needed when RGB output is requested — computed
    // lazily below with overflow checking.
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma — YUV420p luma *is* the Y plane. Just copy.
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Output mode resolution (Strategy A):
    // - RGBA-only: run dedicated `yuv_420_to_rgba_row` (4 bpp store).
    // - RGB / HSV (with or without RGBA): run RGB kernel once, then if
    //   RGBA is also requested, fan it out via `expand_rgb_to_rgba_row`
    //   (memory-bound copy + 0xFF alpha pad). Saves the second YUV→RGB
    //   per-pixel math when both buffers are attached.
    // - None of the above: nothing to do beyond luma above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv_420_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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

    // Pick where the RGB row lands. If the caller wants RGB in their
    // own buffer, write directly there; otherwise use the scratch.
    // Either way, the slice we hold is `&mut [u8]` that we then
    // reborrow as `&[u8]` for the HSV step.
    //
    // RGB byte ranges use `checked_mul` because `w x 3` (and
    // `(idx + 1) x w x 3`) can wrap 32-bit `usize` for large widths
    // even when the single-plane ranges fit — a caller can attach
    // only `with_hsv` (which validates `w x h x 1`) and never go
    // through the `x 3` check at buffer attachment. Overflow here
    // returns `GeometryOverflow` instead of panicking inside the row
    // dispatcher's own checked multiplication.
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;

    // Fused YUV→RGB: upsample chroma in registers inside the row
    // primitive, no intermediate memory.
    yuv_420_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

    // HSV from the RGB row we just wrote.
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

impl<R> Yuv420pSink for MixedSinker<'_, Yuv420p, R> {}

/// Native decimation join for 4:2:0 planar sources: Y streams on the
/// frame grid, U and V on the chroma grid (half width, ceil-half
/// height — possibly the upsample direction), every plane binned to
/// FULL output resolution. Each plane's in-order emissions stage into
/// a two-slot ring; the moment all three planes hold an output row it
/// finalizes through the 4:4:4 kernels at output width — so no
/// alignment constraint ever applies to the output geometry. A plane
/// may lead another by at most one source row (the grids are within a
/// factor of two), which two slots absorb.
pub(super) struct NativeYuv420 {
  y: AreaStream<u8>,
  /// Two-slot staging ring, `2 * out_w` (slot = `out_y & 1`).
  y_stage: std::vec::Vec<u8>,
  /// Chroma half of the join — absent for luma-only sinks, which
  /// therefore never read the chroma planes (the documented fast
  /// path). Safe to decide at creation: the frozen-output contract
  /// makes the attached set frame-constant.
  chroma: Option<NativeChroma>,
  /// `staged[plane][slot]` — plane 0 = Y, 1 = U, 2 = V.
  staged: [[bool; 2]; 3],
  /// Next output row to finalize.
  next_emit: usize,
}

/// Chroma-grid streams and staging of [`NativeYuv420`].
pub(super) struct NativeChroma {
  u: AreaStream<u8>,
  v: AreaStream<u8>,
  u_stage: std::vec::Vec<u8>,
  v_stage: std::vec::Vec<u8>,
}

impl NativeYuv420 {
  fn new(plan: &ResamplePlan, w: usize, h: usize, need_color: bool) -> Result<Self, ResampleError> {
    let y = AreaStream::new(plan.h(), plan.v(), w, h, 1)?;
    let alloc =
      |_| ResampleError::AllocationFailed(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()));
    let stage_len = plan.out_w().checked_mul(2).ok_or_else(|| {
      ResampleError::Overflow(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()))
    })?;
    let chroma = if need_color {
      let cw = w / 2;
      // Vertical chroma weighting runs in the LUMA domain so an odd
      // trailing luma row weights its chroma row by half; the plan's
      // stored dims (cw, h) are the per-plane denominators.
      let cplan = ResamplePlan::area_chroma_420(cw, h, plan.out_w(), plan.out_h())?;
      Some(NativeChroma {
        u: AreaStream::new(cplan.h(), cplan.v(), cplan.src_w(), cplan.src_h(), 1)?,
        v: AreaStream::new(cplan.h(), cplan.v(), cplan.src_w(), cplan.src_h(), 1)?,
        u_stage: try_zeroed(stage_len).map_err(alloc)?,
        v_stage: try_zeroed(stage_len).map_err(alloc)?,
      })
    } else {
      None
    };
    Ok(Self {
      y,
      y_stage: try_zeroed(stage_len).map_err(alloc)?,
      chroma,
      staged: [[false; 2]; 3],
      next_emit: 0,
    })
  }

  pub(super) fn reset(&mut self) {
    self.y.reset();
    if let Some(chroma) = self.chroma.as_mut() {
      chroma.u.reset();
      chroma.v.reset();
    }
    self.staged = [[false; 2]; 3];
    self.next_emit = 0;
  }

  /// Sequencing preflight across all three plane streams — checked
  /// before any plane is fed so a violating call mutates nothing.
  /// Chroma rows advance once per source-row pair, so their expected
  /// counter is the ceiling half of the source row.
  fn check_sequence(&self, idx: usize) -> Result<(), MixedSinkerError> {
    if self.y.next_y() != idx {
      return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
        OutOfSequenceRow::new(self.y.next_y(), idx),
      )));
    }
    if let Some(chroma) = self.chroma.as_ref() {
      let chroma_expected = idx.div_ceil(2);
      for stream in [&chroma.u, &chroma.v] {
        if stream.next_y() != chroma_expected {
          return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
            OutOfSequenceRow::new(stream.next_y().saturating_mul(2), idx),
          )));
        }
      }
    }
    Ok(())
  }
}

/// Native-tier path for [`MixedSinker<Yuv420p, R>`]: see
/// [`NativeYuv420`]. Phasing mirrors the row-stage tier — frozen
/// configuration, join creation, sequencing, color scratch sizing,
/// then the feeds, with nothing fallible after the first feed.
///
/// Takes the Y plane and the **separate** half-width U / V planes
/// (rather than a `Yuv420pRow`) so the 4:2:0 semi-planar family
/// ([`Nv12`](crate::source::Nv12) / [`Nv21`](crate::source::Nv21)) can
/// reuse it verbatim after de-interleaving its interleaved chroma plane
/// into U / V scratch — every 4:2:0 source then bins Y + U + V through
/// the same per-plane join, byte-identical to a `Yuv420p` conversion of
/// the pre-binned planes.
#[allow(clippy::too_many_arguments)]
pub(super) fn yuv420p_process_native(
  plan: &ResamplePlan,
  native_420: &mut Option<NativeYuv420>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

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
  // The join's chroma half is fixed at creation; if the frame's color
  // capability differs (outputs attached since the previous frame —
  // the frozen check pins them WITHIN a frame, not across frames),
  // rebuild it rather than silently skip or needlessly read chroma.
  if native_420
    .as_ref()
    .is_some_and(|join| join.chroma.is_some() != need_color)
  {
    *native_420 = None;
  }
  let join = match native_420 {
    Some(join) => join,
    None => native_420.insert(NativeYuv420::new(plan, w, h, need_color)?),
  };
  join.check_sequence(idx)?;
  if need_color {
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
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      rgb_scratch.resize(row_bytes, 0);
    }
  }

  // Feed the planes; everything past this point is infallible.
  let NativeYuv420 {
    y,
    y_stage,
    chroma,
    staged,
    next_emit,
  } = join;
  y.feed_row(idx, y_row, use_simd, |oy, out_row| {
    let slot = oy & 1;
    y_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
    staged[0][slot] = true;
  })?;
  if let Some(c) = chroma.as_mut()
    && idx.is_multiple_of(2)
  {
    let cidx = idx / 2;
    let NativeChroma {
      u,
      v,
      u_stage,
      v_stage,
    } = c;
    u.feed_row(cidx, u_half, use_simd, |oy, out_row| {
      let slot = oy & 1;
      u_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[1][slot] = true;
    })?;
    v.feed_row(cidx, v_half, use_simd, |oy, out_row| {
      let slot = oy & 1;
      v_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[2][slot] = true;
    })?;
  }

  // Drain every output row whose participating planes are staged.
  while *next_emit < plan.out_h() {
    let slot = *next_emit & 1;
    let chroma_ready = match chroma.as_ref() {
      Some(_) => staged[1][slot] && staged[2][slot],
      None => true,
    };
    if !(staged[0][slot] && chroma_ready) {
      break;
    }
    let oy = *next_emit;
    let y_out = &y_stage[slot * ow..slot * ow + ow];

    if let Some(buf) = luma.as_deref_mut() {
      buf[oy * ow..(oy + 1) * ow].copy_from_slice(y_out);
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(y_out) {
        *dst = src as u16;
      }
    }
    if let Some(c) = chroma.as_ref() {
      let u_row = &c.u_stage[slot * ow..slot * ow + ow];
      let v_row = &c.v_stage[slot * ow..slot * ow + ow];
      let out_rgb = &mut rgb_scratch[..ow * 3];
      yuv_444_to_rgb_row(
        y_out, u_row, v_row, out_rgb, ow, matrix, full_range, use_simd,
      );
      if let Some(buf) = rgb.as_deref_mut() {
        buf[oy * 3 * ow..(oy + 1) * 3 * ow].copy_from_slice(out_rgb);
      }
      if let Some(hsv) = hsv.as_mut() {
        let (hp, sp, vp) = hsv.hsv();
        rgb_to_hsv_row(
          out_rgb,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
      if let Some(buf) = rgba.as_deref_mut() {
        expand_rgb_to_rgba_row(out_rgb, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow);
      }
    }
    staged[0][slot] = false;
    staged[1][slot] = false;
    staged[2][slot] = false;
    *next_emit += 1;
  }
  Ok(())
}

/// Row-stage tier for the 4:2:0 planar family (the `with_native(false)`
/// path). Takes the Y plane and the **separate** half-width U / V planes
/// so the 4:2:0 semi-planar family reuses it after de-interleave.
#[allow(clippy::too_many_arguments)]
pub(super) fn yuv420p_process_resampled(
  plan: &ResamplePlan,
  rgb_stream: &mut Option<AreaStream<u8>>,
  luma_stream: &mut Option<AreaStream<u8>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

  // Atomic preflight: every fallible step runs before any stream is
  // fed, so a failed call mutates no caller output and the frame can
  // restart via begin_frame.
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
  // Create every requested stream, then check all of them against
  // this row index; a stream attached mid-frame starts at row 0 and
  // fails here.
  // Sequence-check before allocating (mirrors the packed-RGB helpers):
  // an out-of-sequence first row is rejected before any output-width
  // buffer is created, so AllocationFailed never masks OutOfSequenceRow.
  // A no-output call has no stream to sequence and stays a no-op.
  let expected = if need_luma {
    luma_stream.as_ref().map_or(0, |stream| stream.next_y())
  } else if need_color {
    rgb_stream.as_ref().map_or(0, |stream| stream.next_y())
  } else {
    idx
  };
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
  // (3) Color-group preparation is also fallible (scratch sizing) and
  // scratch-mutating, so it runs before the luma feed too. The user
  // RGB buffer is output-sized; the source-width row always lands in
  // the scratch. (The overflow arm is defense in depth: any geometry
  // large enough to wrap w * 3 cannot plan — its span arena alloc is
  // out of reach first.)
  let color_row = if need_color {
    let row_bytes =
      w.checked_mul(3)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w,
          plan.src_h(),
          3,
        )))?;
    if rgb_scratch.len() < row_bytes {
      // Same recoverable-allocation contract as the planner and the
      // stream buffers: the scratch is source-width-proportional, so
      // refusal surfaces as an error in the preflight phase instead
      // of aborting inside infallible growth. The exact reserve makes
      // the resize below incapable of reallocating.
      rgb_scratch
        .try_reserve_exact(row_bytes - rgb_scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      rgb_scratch.resize(row_bytes, 0);
    }
    let scratch = &mut rgb_scratch[..row_bytes];
    yuv_420_to_rgb_row(
      y_row, u_half, v_half, scratch, w, matrix, full_range, use_simd,
    );
    Some(scratch)
  } else {
    None
  };

  if need_luma {
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

// ---- Yuv410p impl -------------------------------------------------------
//
// 4:1:0 planar 8-bit (Cinepak / Sorenson legacy, Tier 1 P3). Chroma
// is subsampled 4:1 in **both** axes — `width / 4` chroma bytes per
// row, `height / 4` chroma rows. Per-row math reuses the dedicated
// `yuv_410_to_rgb_row` / `yuv_410_to_rgba_row` kernels (4:2:0's
// half-width chroma layout doesn't apply — each chroma sample
// covers four Y columns instead of two).
//
// Output set matches the Tier 1 contract:
// `with_rgb` / `with_rgba` / `with_luma` / `with_luma_u16` / `with_hsv`.
// No source alpha (no YUVA 4:1:0 format). Strategy A applies for the
// RGB+RGBA combo: run the 3-channel kernel once, fan out via
// `expand_rgb_to_rgba_row`.

impl<'a, R> MixedSinker<'a, Yuv410p, R> {
  /// Attaches a packed 32-bit RGBA output buffer.
  ///
  /// See [`MixedSinker::<Yuv420p>::with_rgba`] for the rationale and
  /// constraints. Yuv410p has no alpha plane, so every alpha byte is
  /// filled with `0xFF` (opaque).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`.
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

impl<R> Yuv410pSink for MixedSinker<'_, Yuv410p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv410p, R> {
  type Input<'r> = Yuv410pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // Yuv410p requires width to be a multiple of 4 (the row kernels
    // operate on 4-pixel chroma groups). Height is unconstrained — the
    // walker reuses the trailing chroma row for the final 1..=3 Y rows.
    // The frame `try_new` already enforces width alignment, but a
    // hand-crafted row walker (driving `process` directly) might bypass
    // it — guard at the sinker boundary so unsafe SIMD dispatchers
    // never see a non-multiple-of-4 width.
    if self.width & 3 != 0 {
      return Err(MixedSinkerError::WidthAlignment(
        WidthAlignment::multiple_of_four(self.width),
      ));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuv410pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense in depth — see Yuv420p impl.
    if w & 3 != 0 {
      return Err(MixedSinkerError::WidthAlignment(
        WidthAlignment::multiple_of_four(w),
      ));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_quarter().len() != w / 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UQuarter,
        idx,
        w / 4,
        row.u_quarter().len(),
      )));
    }
    if row.v_quarter().len() != w / 4 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VQuarter,
        idx,
        w / 4,
        row.v_quarter().len(),
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
      plan,
      rgb_stream,
      luma_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: convert the source row to canonical RGB at
    // source width in the shared scratch (the same `yuv_410_to_rgb_row`
    // kernel the identity path uses — 4:1:0's 1→4 chroma upsample), then
    // feed the shared planar resample tail. Row-stage only — converting
    // each source row to RGB and binning is the whole job. Freeze the
    // output set and check stream sequencing before staging, so a
    // no-output sink stays a no-op and an out-of-sequence row is
    // rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      return planar_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        row.y(),
        w,
        plan,
        idx,
        use_simd,
        |scratch| {
          yuv_410_to_rgb_row(
            row.y(),
            row.u_quarter(),
            row.v_quarter(),
            scratch,
            w,
            matrix,
            full_range,
            use_simd,
          );
        },
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma — Yuv410p luma *is* the Y plane. Just copy.
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv_410_to_rgba_row(
        row.y(),
        row.u_quarter(),
        row.v_quarter(),
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

    yuv_410_to_rgb_row(
      row.y(),
      row.u_quarter(),
      row.v_quarter(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

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

// ---- Yuv422p impl -------------------------------------------------------
//
// 4:2:2 is 4:2:0's vertical-axis twin: same per-row chroma shape
// (half-width U / V, one pair per Y pair), just one chroma row per Y
// row instead of one per two. This impl reuses `yuv_420_to_rgb_row`
// (and `yuv_420_to_rgba_row` for the RGBA path) — no new kernels
// needed.

impl<'a, R> MixedSinker<'a, Yuv422p, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. Yuv422p has no alpha plane, so every
  /// alpha byte is filled with `0xFF` (opaque).
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`. Length is measured in `u16`
  /// elements (`width x height`).
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

impl<R> Yuv422pSink for MixedSinker<'_, Yuv422p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv422p, R> {
  type Input<'r> = Yuv422pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuv422pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    if w & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
    }
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UHalf,
        idx,
        w / 2,
        row.u_half().len(),
      )));
    }
    if row.v_half().len() != w / 2 {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VHalf,
        idx,
        w / 2,
        row.v_half().len(),
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
      plan,
      rgb_stream,
      luma_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: convert the source row to canonical RGB at
    // source width in the shared scratch (same 4:2:0 per-row dispatcher
    // 4:2:2 reuses on its identity path), then feed the shared planar
    // resample tail. Row-stage only — converting each source row to RGB
    // and binning is the whole job. Freeze the output set and check
    // stream sequencing before staging, so a no-output sink stays a
    // no-op and an out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      return planar_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        row.y(),
        w,
        plan,
        idx,
        use_simd,
        |scratch| {
          yuv_420_to_rgb_row(
            row.y(),
            row.u_half(),
            row.v_half(),
            scratch,
            w,
            matrix,
            full_range,
            use_simd,
          );
        },
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    // Reuses Yuv420p dispatchers (RGB and RGBA) since 4:2:2's per-row
    // contract is identical (half-width chroma, one pair per Y pair).
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv_420_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
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

    // Reuses the Yuv420p dispatcher — 4:2:2's per-row contract is
    // identical (half-width chroma, one pair per Y pair).
    yuv_420_to_rgb_row(
      row.y(),
      row.u_half(),
      row.v_half(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

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

// ---- Yuv444p impl -------------------------------------------------------
//
// 4:4:4 planar: U and V are full-width, full-height. No width parity
// constraint. Uses the `yuv_444_to_rgb_row` / `yuv_444_to_rgba_row`
// kernel family.

impl<'a, R> MixedSinker<'a, Yuv444p, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. Yuv444p has no alpha plane, so every
  /// alpha byte is filled with `0xFF` (opaque).
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`. Length is measured in `u16`
  /// elements (`width x height`).
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

impl<R> Yuv444pSink for MixedSinker<'_, Yuv444p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv444p, R> {
  type Input<'r> = Yuv444pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuv444pRow<'_>) -> Result<(), Self::Error> {
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
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull,
        idx,
        w,
        row.v().len(),
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
      plan,
      rgb_stream,
      luma_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: convert the source row to canonical RGB at
    // source width in the shared scratch (the same 4:4:4 kernel the
    // identity path uses), then feed the shared planar resample tail.
    // Row-stage only — converting each source row to RGB and binning is
    // the whole job. Freeze the output set and check stream sequencing
    // before staging, so a no-output sink stays a no-op and an
    // out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      return planar_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        row.y(),
        w,
        plan,
        idx,
        use_simd,
        |scratch| {
          yuv_444_to_rgb_row(
            row.y(),
            row.u(),
            row.v(),
            scratch,
            w,
            matrix,
            full_range,
            use_simd,
          );
        },
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv_444_to_rgba_row(
        row.y(),
        row.u(),
        row.v(),
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

    yuv_444_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

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

// ---- Yuv440p impl -------------------------------------------------------
//
// 4:4:0 planar 8‑bit — full-width chroma, half-height. Per-row math
// matches 4:4:4 (full-width U / V); only the walker reads chroma row
// `r / 2`. Reuses `yuv_444_to_rgb_row` and `yuv_444_to_rgba_row`
// verbatim.

impl<'a, R> MixedSinker<'a, Yuv440p, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// See [`MixedSinker::<Yuv420p>::with_rgba`] for the rationale and
  /// constraints. Yuv440p has no alpha plane, so every alpha byte is
  /// filled with `0xFF` (opaque).
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`. Length is measured in `u16`
  /// elements (`width x height`).
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

impl<R> Yuv440pSink for MixedSinker<'_, Yuv440p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv440p, R> {
  type Input<'r> = Yuv440pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)?;
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuv440pRow<'_>) -> Result<(), Self::Error> {
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
    if row.u().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UFull,
        idx,
        w,
        row.u().len(),
      )));
    }
    if row.v().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VFull,
        idx,
        w,
        row.v().len(),
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
      plan,
      rgb_stream,
      luma_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: convert the source row to canonical RGB at
    // source width in the shared scratch (the same `yuv_444_to_rgb_row`
    // kernel the identity path uses — 4:4:0's per-row math is identical
    // to 4:4:4, full-width chroma), then feed the shared planar resample
    // tail. Row-stage only — converting each source row to RGB and
    // binning is the whole job. Freeze the output set and check stream
    // sequencing before staging, so a no-output sink stays a no-op and
    // an out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      return planar_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        row.y(),
        w,
        plan,
        idx,
        use_simd,
        |scratch| {
          yuv_444_to_rgb_row(
            row.y(),
            row.u(),
            row.v(),
            scratch,
            w,
            matrix,
            full_range,
            use_simd,
          );
        },
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    // Reuses the Yuv444p RGBA dispatcher since 4:4:0's per-row math
    // is identical (full-width chroma).
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv_444_to_rgba_row(
        row.y(),
        row.u(),
        row.v(),
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

    yuv_444_to_rgb_row(
      row.y(),
      row.u(),
      row.v(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

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

// ---- Yuv411p impl -------------------------------------------------------
//
// 4:1:1 planar 8-bit — quarter-width chroma, full-height. DV-NTSC
// legacy. Per-row math reuses the dedicated `yuv_411_to_rgb_row` /
// `yuv_411_to_rgba_row` family (1→4 chroma upsample). Following
// FFmpeg's `AV_PIX_FMT_YUV411P` semantics, chroma row width is
// `width.div_ceil(4)`: non-4-aligned widths get a partial 1..3-pixel
// final chroma group, handled by the scalar tail.

impl<'a, R> MixedSinker<'a, Yuv411p, R> {
  /// Attaches a packed 32-bit RGBA output buffer.
  ///
  /// See [`MixedSinker::<Yuv420p>::with_rgba`] for the rationale and
  /// constraints. Yuv411p has no alpha plane, so every alpha byte is
  /// filled with `0xFF` (opaque).
  ///
  /// Returns `Err(InsufficientRgbaBuffer)` if
  /// `buf.len() < width x height x 4`, or `Err(GeometryOverflow)` on
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

  /// Attaches a `u16` luma output buffer. The 8-bit Y plane samples
  /// are zero-extended into `u16`. Length is measured in `u16`
  /// elements (`width x height`).
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

impl<R> Yuv411pSink for MixedSinker<'_, Yuv411p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuv411p, R> {
  type Input<'r> = Yuv411pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    // FFmpeg-compatible: arbitrary widths accepted (chroma row is
    // `width.div_ceil(4)` samples; the scalar kernel handles a
    // partial 1..3-pixel final chroma group). No width-parity
    // restriction here.
    check_dimensions_match(self.width, self.height, width, height)?;
    // Resampling carries frame progress in the stream; reset it and
    // re-freeze the output set so each frame starts clean.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Yuv411pRow<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Chroma row shape: `width.div_ceil(4)` samples (FFmpeg
    // `AV_PIX_FMT_YUV411P`). For widths divisible by 4 this matches
    // `w / 4`; for non-aligned widths the trailing 1..3 Y pixels
    // share the last (partial) chroma sample.
    let chroma_w = w.div_ceil(4);
    if row.y().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::Y,
        idx,
        w,
        row.y().len(),
      )));
    }
    if row.u_quarter().len() != chroma_w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UQuarter,
        idx,
        chroma_w,
        row.u_quarter().len(),
      )));
    }
    if row.v_quarter().len() != chroma_w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VQuarter,
        idx,
        chroma_w,
        row.v_quarter().len(),
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
      plan,
      rgb_stream,
      luma_stream,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: freeze the output set, then check stream
    // sequencing — both before touching the scratch — so a no-output
    // sink stays a no-op and an out-of-sequence row is rejected
    // without the source-width allocation/conversion. Only then
    // upsample chroma into a full-width RGB row (the same
    // `yuv_411_to_rgb_row` kernel the identity path uses) and feed
    // the one packed-RGB resample tail. Yuv411p is row-stage only —
    // every output derives from the binned RGB rows.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      return planar_dual_resample(
        luma_stream,
        rgb_stream,
        resample_outputs,
        rgb,
        rgba,
        luma,
        luma_u16,
        hsv,
        rgb_scratch,
        row.y(),
        w,
        plan,
        idx,
        use_simd,
        |scratch| {
          yuv_411_to_rgb_row(
            row.y(),
            row.u_quarter(),
            row.v_quarter(),
            scratch,
            w,
            matrix,
            full_range,
            use_simd,
          );
        },
      );
    }

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma — Yuv411p luma *is* the Y plane. Just copy.
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the 8-bit Y plane into u16.
    if let Some(buf) = luma_u16.as_deref_mut() {
      crate::row::y_plane_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    // Strategy A output mode resolution — see Yuv420p impl above.
    // 4:1:1 has its own dedicated `yuv_411_to_rgb_row` /
    // `yuv_411_to_rgba_row` kernels (1→4 chroma upsample); these
    // can't be reused from 4:2:0 / 4:2:2.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuv_411_to_rgba_row(
        row.y(),
        row.u_quarter(),
        row.v_quarter(),
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

    yuv_411_to_rgb_row(
      row.y(),
      row.u_quarter(),
      row.v_quarter(),
      rgb_row,
      w,
      row.matrix(),
      row.full_range(),
      use_simd,
    );

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
