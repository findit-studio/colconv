//! Format-agnostic row-stage planar-YUV resample helper shared by the
//! 8-bit planar family ([`super::planar_8bit`]) and the semi-planar
//! family ([`super::semi_planar_8bit`]). [`planar_dual_resample`] bins the
//! Y plane for luma and bins a caller-converted source-width RGB row for
//! colour, so it references no source-format kernel — the caller supplies
//! the conversion closure (the planar formats convert their separate
//! planes, the semi-planar formats convert their interleaved chroma row
//! with the matching `nv*` kernel). Both routes are byte-identical to an
//! `Rgb24` area-resample of the identity-converted frame.

use super::{HsvFrameMut, MixedSinkerError, frozen_outputs_check};
use crate::{
  resample::{AreaStream, OutOfSequenceRow, ResampleError, ResamplePlan, RowResampler},
  row::*,
};

/// Row-stage fused downscale shared by the planar formats with no
/// native tier (Yuv411p / Yuv422p / Yuv444p). Mirrors the Yuv420p
/// row-stage path: **luma / luma_u16 area-resample the Y plane
/// directly** (a 1-channel stream over `y_row`, the YUV luma
/// contract — luma is *not* re-derived from converted RGB), while RGB
/// / RGBA / HSV bin a converted source-width RGB row (the 3-channel
/// stream). `convert_rgb` fills the source-width scratch with RGB
/// using the format's own conversion kernel, and runs only when a
/// colour output is attached. Atomic preflight: every fallible step
/// (freeze, stream creation, sequence check, scratch growth +
/// conversion) precedes the first feed, so a failure mutates no
/// caller output.
#[cfg_attr(
  not(any(feature = "yuv-planar", feature = "yuv-semi-planar")),
  allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
pub(super) fn planar_dual_resample(
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  rgb_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (these planar YUV families are not routed to the filter
  // path): reject a filter plan before any work, so the plan's empty area
  // spans never reach an area stream.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

  // Single sequence check, on whichever stream is fed every row (all
  // attached streams advance in lockstep). A no-output call has no stream
  // to sequence and stays a no-op regardless of the row index — returned
  // before the freeze so it stores no snapshot a later attach-then-retry
  // would trip on.
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
    *luma_stream = Some({
      let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_color && rgb_stream.is_none() {
    *rgb_stream = Some({
      let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }

  // Stage + feed + emit. Shared with the filter path
  // ([`planar_dual_filter_resample`]) so the area and filter arms run the
  // identical convert-then-resample tail — the only difference is the
  // stream kind built above.
  planar_dual_feed_emit(
    luma_stream.as_mut(),
    rgb_stream.as_mut(),
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    y_row,
    w,
    plan,
    idx,
    use_simd,
    convert_rgb,
  )
}

/// Shared stage-then-feed tail for the 8-bit planar YUV family, used by
/// both [`planar_dual_resample`] (area) and [`planar_dual_filter_resample`]
/// (filter). The two paths differ only in the resampler kind built by the
/// caller — the convert-then-resample staging and per-output emit are
/// identical, so they live here behind the
/// [`RowResampler`](crate::resample::RowResampler) trait (which both
/// [`AreaStream`](crate::resample::AreaStream) and
/// [`FilterStream`](crate::resample::FilterStream) implement). Keeping the
/// emit byte-identical between the arms is what makes the filter output
/// match the area output up to the kernel weights.
///
/// Luma is the native Y resampled directly (the YUV luma contract — `luma`
/// copies each finalized `u8` Y row, `luma_u16` zero-extends it); colour
/// bins a caller-converted source-width RGB row (`convert_rgb` fills the
/// scratch only when a colour output is attached). The scratch grows via
/// the recoverable-allocation helper before the first feed, so a failure
/// mutates no caller output. These sources are 8-bit, so no native-depth
/// clamp applies — the `u8` stream finalizes to the full `u8` range, which
/// is the native range.
#[cfg_attr(
  not(any(feature = "yuv-planar", feature = "yuv-semi-planar")),
  allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
fn planar_dual_feed_emit<LS, CS>(
  luma_stream: Option<&mut LS>,
  rgb_stream: Option<&mut CS>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError>
where
  LS: RowResampler<u8>,
  CS: RowResampler<u8>,
{
  let ow = plan.out_w();
  // Stage the source-width colour scratch (the fallible growth runs before
  // the first feed, keeping the call atomic) only when a colour output is
  // attached.
  let color_row = if rgb_stream.is_some() {
    let scratch = super::source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb(scratch);
    Some(scratch)
  } else {
    None
  };

  if let Some(stream) = luma_stream {
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
    let stream = rgb_stream.expect("staged only when present");
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

/// Separable-filter fused resize for the 8-bit planar YUV family — the
/// [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`planar_dual_resample`]. It mirrors the area path exactly: the separate
/// Y/U/V planes are converted to a source-width RGB row by the **same**
/// `convert_rgb` closure (which upsamples 4:2:0 / 4:2:2 / 4:1:0 / 4:4:0
/// chroma), then the RGB is resampled by the signed-coefficient
/// [`FilterStream`] (the filter twin of the area bin) and the **same** emit
/// ([`planar_dual_feed_emit`]) is run — so the resampled colour output
/// equals the equivalent packed-RGB filter resample of the converted
/// pixels, and (because the area path converts-then-bins the same RGB)
/// matches the area output up to the kernel weights.
///
/// Luma stays the native Y filter-resampled (the filter twin of the area
/// path's native-Y bin): a 1-channel [`FilterStream<u8>`] resamples the Y
/// plane directly, so luma is taken from Y, never colour-derived. These
/// sources are 8-bit, so the `u8` stream finalizes to the full `u8` range,
/// which *is* the native range — no sub-16-bit native-depth clamp applies
/// (unlike the high-bit planar / packed YUV filter routes). `luma_u16`
/// zero-extends each resampled Y byte.
///
/// Atomic preflight (mirrors [`planar_dual_resample`]): a single
/// [`frozen_outputs_check`] over the output set, then a single sequence
/// check on whichever stream is fed every row **before any allocation** (an
/// out-of-sequence first row is rejected before the freeze, storing no
/// snapshot to poison a retry; on a later row the freeze runs first so a
/// mid-frame output change trips `ResampleOutputsChanged`), then every
/// stream and the source-width scratch is created before the first feed —
/// so a failure mutates no caller output. A no-output call has no stream to
/// sequence and stays a no-op.
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn planar_dual_filter_resample(
  luma_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  rgb_filter_stream: &mut Option<std::boxed::Box<crate::resample::FilterStream<u8>>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  w: usize,
  plan: &ResamplePlan,
  idx: usize,
  use_simd: bool,
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // This single-kernel tail filters ONE converted RGB row; a BICUBLIN plan
  // ([`Bicublin`](crate::resample::Bicublin)) carries a second (chroma) window
  // set that only the `Yuv420p` per-plane route reads, so reject it here rather
  // than silently filtering every plane with the luma kernel. Every non-4:2:0
  // planar / semi-planar format routes its filter dispatch through this tail,
  // so the one guard fences the whole family.
  plan.ensure_single_kernel_filter()?;
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

  let (fh, fv) = (
    plan
      .filter_h()
      .expect("filter plan carries horizontal windows"),
    plan
      .filter_v()
      .expect("filter plan carries vertical windows"),
  );

  // Single sequence check, on whichever stream is fed every row (all
  // attached streams advance in lockstep). A no-output call has no stream
  // to sequence and stays a no-op regardless of the row index — returned
  // before the freeze so it stores no snapshot a later attach-then-retry
  // would trip on.
  let expected = if need_luma {
    luma_filter_stream
      .as_ref()
      .map_or(0, |stream| stream.next_y())
  } else if need_color {
    rgb_filter_stream
      .as_ref()
      .map_or(0, |stream| stream.next_y())
  } else {
    return Ok(());
  };
  // First row: reject an out-of-sequence row BEFORE the freeze, so a
  // rejected first row stores no snapshot that would poison a retry. On a
  // later row the freeze runs first (below), so a mid-frame output-set
  // change is reported as ResampleOutputsChanged rather than masked by a
  // freshly-attached stream's row-0 sequence mismatch.
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
  if need_luma && luma_filter_stream.is_none() {
    *luma_filter_stream = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_color && rgb_filter_stream.is_none() {
    *rgb_filter_stream = Some({
      let stream = crate::resample::FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }

  // Stage + feed + emit. Shared with the area path
  // ([`planar_dual_resample`]) so the area and filter arms run the
  // identical convert-then-resample tail — the only difference is the
  // stream kind built above.
  planar_dual_feed_emit(
    luma_filter_stream.as_mut(),
    rgb_filter_stream.as_mut(),
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    y_row,
    w,
    plan,
    idx,
    use_simd,
    convert_rgb,
  )
}

/// Row-stage fused downscale for the **packed** YUV formats whose Y
/// samples are interleaved in the source plane (packed 4:1:1 —
/// [`Uyyvyy411`](crate::source::Uyyvyy411)). The planar twin
/// [`planar_dual_resample`] takes a ready source-width Y plane; here Y
/// must first be de-interleaved out of the packed plane, so this helper
/// owns a **second** scratch (`luma_scratch`, distinct from the colour
/// `rgb_scratch`) and the caller supplies a `convert_luma` closure that
/// fills it (the format's own `*_to_luma_row` kernel — the YUV luma
/// contract, *not* RGB-derived luma). Colour binning is identical to
/// the planar twin: `convert_rgb` fills the colour scratch with a
/// source-width RGB row via the format's fused `*_to_rgb_row` kernel
/// (chroma de-interleave + horizontal upsample in registers) and the
/// 3-channel stream bins it, so RGB equals an `Rgb24` area-resample of
/// the identity-converted frame.
///
/// Atomic preflight (matching [`planar_dual_resample`]): the output set
/// is frozen, then stream sequencing is checked, **both before any
/// allocation** — so a no-output sink stays a no-op, an out-of-sequence
/// row is rejected without staging a buffer, and `AllocationFailed`
/// can never mask `OutOfSequenceRow`. Only then are the (separate) luma
/// and colour scratches grown via the recoverable-allocation helpers
/// and their conversions run, before the first feed; a failure at any
/// step mutates no caller output.
#[cfg(feature = "yuv-packed")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv_dual_resample(
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  rgb_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
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
  convert_luma: impl FnOnce(&mut [u8]),
  convert_rgb: impl FnOnce(&mut [u8]),
) -> Result<(), MixedSinkerError> {
  // Area-only sink (packed YUV 4:2:2 is not routed to the filter path):
  // reject a filter plan before any work, so the plan's empty area spans
  // never reach an area stream.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();

  // Single sequence check, on whichever stream is fed every row (all
  // attached streams advance in lockstep). A no-output call has no stream
  // to sequence and stays a no-op regardless of the row index — returned
  // before the freeze so it stores no snapshot a later attach-then-retry
  // would trip on.
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
    *luma_stream = Some({
      let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  if need_color && rgb_stream.is_none() {
    *rgb_stream = Some({
      let stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 3)?;
      crate::resample::try_box(stream).map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?
    });
  }
  // Stage both source-width scratches (each via its own recoverable
  // grow) and run the conversions before any feed, keeping the call
  // atomic. The luma scratch is the de-interleaved Y plane; the colour
  // scratch is the source-width RGB row.
  let luma_row = if need_luma {
    let scratch = super::source_luma_scratch(luma_scratch, w, plan)?;
    convert_luma(scratch);
    Some(scratch)
  } else {
    None
  };
  let color_row = if need_color {
    let scratch = super::source_rgb_scratch(rgb_scratch, w, plan)?;
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
