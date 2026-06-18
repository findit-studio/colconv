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
  resample::{AreaStream, OutOfSequenceRow, ResampleError, ResamplePlan},
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
  luma_stream: &mut Option<AreaStream<u8>>,
  rgb_stream: &mut Option<AreaStream<u8>>,
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
  let color_row = if need_color {
    let scratch = super::source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb(scratch);
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
  luma_stream: &mut Option<AreaStream<u8>>,
  rgb_stream: &mut Option<AreaStream<u8>>,
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
