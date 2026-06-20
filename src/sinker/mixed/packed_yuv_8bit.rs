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
  resample::{
    AreaStream, FilterStream, OutOfSequenceRow, ResampleError, ResamplePlan, RowResampler, SpanKind,
  },
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

#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
use super::{
  NativeRouteChanged,
  planar_8bit::{NativePlanarYuv, native_planar_preflight, yuv_planar_process_native},
};
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
use crate::{
  ColorMatrix,
  resample::{
    AveragingDomain, InsertionContext, InsertionPoint, PlanGeometry, select_insertion_point,
  },
};

/// Shared stage-then-feed tail for the 8-bit packed-YUV dual-stream
/// resample, used by both [`packed_yuv422_dual_resample`] (area) and
/// [`packed_yuv422_dual_filter_resample`] (filter). The two paths differ
/// only in the resampler kind built by the caller — the convert-then-bin
/// staging and per-output emit are identical, so they live here behind the
/// [`RowResampler<u8>`] trait (which both
/// [`AreaStream<u8>`](crate::resample::AreaStream) and
/// [`FilterStream<u8>`](crate::resample::FilterStream) implement). Keeping
/// the emit byte-identical between the arms is what makes the filter output
/// equal the equivalent RGB filter of the converted pixels, and match the
/// area output up to the kernel weights.
///
/// `deinterleave_y` fills a source-width scratch with the Y samples pulled
/// from the packed row (the format's own `*_to_luma_row` kernel) and runs
/// only when a luma output is attached; `convert_rgb` fills a source-width
/// RGB scratch from the packed row (the format's own `*_to_rgb_row` kernel,
/// which de-interleaves + horizontally upsamples the chroma) and runs only
/// when a colour output is attached. The two scratches are distinct fields
/// and never alias.
///
/// No native-depth clamp: these are 8-bit sources, so the source's native
/// range *is* the full `u8` range and the stream's own finalize keeps every
/// binned sample in range (`AreaStream` averages within `[0, 255]`;
/// `FilterStream<u8>`'s `clip8` clamps a signed-kernel overshoot to
/// `[0, 255]`). Both streams are created by the caller post-sequence-check;
/// staging runs every fallible growth + conversion before the first feed so
/// a failure mutates no caller output.
#[cfg(feature = "yuv-packed")]
#[allow(clippy::too_many_arguments)]
fn packed_yuv422_dual_feed_emit<LS, CS>(
  luma_stream: Option<&mut LS>,
  rgb_stream: Option<&mut CS>,
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
) -> Result<(), MixedSinkerError>
where
  LS: RowResampler<u8>,
  CS: RowResampler<u8>,
{
  let ow = plan.out_w();
  // Stage the source-width rows (both fallible growths run before the
  // feeds, keeping the call atomic). The Y row uses its own scratch so
  // it does not collide with the colour stream's RGB scratch.
  let luma_row = if luma_stream.is_some() {
    let scratch = source_luma_scratch(luma_scratch, w, plan)?;
    deinterleave_y(scratch);
    Some(scratch)
  } else {
    None
  };
  let color_row = if rgb_stream.is_some() {
    let scratch = source_rgb_scratch(rgb_scratch, w, plan)?;
    convert_rgb(scratch);
    Some(scratch)
  } else {
    None
  };

  if let Some(y_row) = luma_row {
    let stream = luma_stream.expect("staged only when present");
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

/// Row-stage **area** fused downscale shared by the three packed YUV 4:2:2
/// formats and (for its filter twin) the packed YUV 4:1:1 `Uyyvyy411`.
/// Mirrors the planar YUV dual-stream path: **luma / luma_u16 area-resample
/// the de-interleaved Y bytes directly** (the YUV luma contract — luma is
/// *not* re-derived from converted RGB), while RGB / RGBA / HSV bin a
/// converted source-width RGB row.
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
#[cfg(feature = "yuv-packed")]
#[allow(clippy::too_many_arguments)]
fn packed_yuv422_dual_resample(
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  rgb_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
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

  packed_yuv422_dual_feed_emit(
    luma_stream.as_mut(),
    rgb_stream.as_mut(),
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
    deinterleave_y,
    convert_rgb,
  )
}

/// Row-stage **separable-filter** fused resample — the
/// [`SpanKind::Filter`](crate::resample::SpanKind) twin of
/// [`packed_yuv422_dual_resample`], shared by the three packed YUV 4:2:2
/// formats and the packed YUV 4:1:1 `Uyyvyy411` (their convert closures
/// have the same shape — de-interleave Y for luma, convert the packed row
/// to a source-width RGB row for colour). Luma stays **native Y**: the
/// de-interleaved Y bytes feed a 1-channel
/// [`FilterStream<u8>`](crate::resample::FilterStream)
/// (`luma_filter_stream`), so luma is byte-exact to the direct `*_to_luma*`
/// kernels' filter resample, never colour-derived. Colour feeds the
/// 3-channel [`FilterStream<u8>`](crate::resample::FilterStream)
/// (`rgb_filter_stream`), so `rgb` / `rgba` / `hsv` equal the equivalent
/// `Rgb24` filter of the converted pixels.
///
/// No native-depth clamp: 8-bit source, so the stream's `clip8` (a
/// signed-kernel overshoot clamped to `[0, 255]`) already keeps every
/// sample in the native range — the colour and luma emit are identical to
/// the area path's.
///
/// Atomic preflight mirrors [`packed_yuv422_dual_resample`]: a no-output
/// call returns before the freeze; a single [`frozen_outputs_check`] runs,
/// then a single sequence check on whichever stream is fed every row, both
/// **before any allocation** — an out-of-sequence first row is rejected
/// before the freeze (storing no snapshot to poison a retry), and on a
/// later row the freeze runs first (a mid-frame output change trips
/// `ResampleOutputsChanged`). Both streams are created after the sequence
/// check, then the shared staging + feed runs.
#[cfg(feature = "yuv-packed")]
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_yuv422_dual_filter_resample(
  luma_filter_stream: &mut Option<std::boxed::Box<FilterStream<u8>>>,
  rgb_filter_stream: &mut Option<std::boxed::Box<FilterStream<u8>>>,
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
  // Single-kernel filter tail — reject a BICUBLIN plan (its chroma windows are
  // read only by the `Yuv420p` per-plane route) before any state change.
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

  // Single sequence check before any allocation, on whichever attached
  // stream is fed every row (all attached streams advance in lockstep). A
  // no-output call has no stream to sequence and stays a no-op.
  let expected = if need_luma {
    luma_filter_stream.as_ref().map_or(0, |s| s.next_y())
  } else if need_color {
    rgb_filter_stream.as_ref().map_or(0, |s| s.next_y())
  } else {
    return Ok(());
  };
  // First row: reject an out-of-sequence row BEFORE the freeze, so a
  // rejected first row stores no snapshot that would poison a retry.
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
      let stream = FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 1)?;
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
      let stream = FilterStream::new(fh, fv, plan.src_w(), plan.src_h(), 3)?;
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

  packed_yuv422_dual_feed_emit(
    luma_filter_stream.as_mut(),
    rgb_filter_stream.as_mut(),
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
    deinterleave_y,
    convert_rgb,
  )
}

/// Native fast-tier decimator for the 8-bit PACKED 4:2:2 YUV family
/// ([`Yuyv422`](crate::source::Yuyv422) `Y U Y V …` /
/// [`Uyvy422`](crate::source::Uyvy422) `U Y V Y …` /
/// [`Yvyu422`](crate::source::Yvyu422) `Y V Y U …`): the PACKED analog of
/// the semi-planar non-4:2:0 native wrapper
/// ([`super::semi_planar_8bit`]'s `semi_planar_process_native_non420`).
/// De-PACKS the fully-interleaved source row into the sink's separate
/// Y (`w`) / U (`w / 2`) / V (`w / 2`) scratch planes at the per-format byte
/// offsets, then reuses the planar twin's non-4:2:0 join verbatim
/// ([`yuv_planar_process_native`]) at [`Yuv422p`](crate::source::Yuv422p)
/// geometry (chroma `w / 2 x h`, `chroma_vsub = 1`) — so every output is
/// byte-identical to a [`Yuv422p`](crate::source::Yuv422p) native conversion
/// of those de-packed planes, and within ±1 LSB of the packed row-stage tier
/// (the conversion-order rounding caveat the planar tiers already carry).
/// Luma is bit-identical (both bin the same native Y).
///
/// `y0_off` / `y1_off` are the two Y byte positions within each 4-byte /
/// 2-pixel group; `u_off` / `v_off` the chroma positions. Like the
/// semi-planar non-4:2:0 wrapper the chroma cadence is one row per Y row, so
/// the U / V de-pack runs on EVERY colour row. On luma-only / no-colour rows
/// only Y is de-packed — the join never reads chroma there, and the chroma
/// scratch (left as-is) is handed to the join as empty slices, so a
/// luma-only sink never plans or allocates chroma state (the lazy-chroma
/// contract `NativePlanarYuv::new` upholds under `need_color`).
///
/// 8-bit source, so no native-depth clamp is needed (the source's native
/// range is the full `u8` range and the join's averaging keeps every sample
/// in range).
///
/// Atomicity mirrors the semi-planar non-4:2:0 wrapper: the join's COMPLETE
/// pre-feed rejection preflight runs FIRST (via [`native_planar_preflight`]),
/// before the fallible Y / U / V scratch grow, so a rejected row returns its
/// deterministic typed error (`OutOfSequenceRow` / `ResampleOutputsChanged`),
/// never `AllocationFailed`, and grows no sink state; the de-pack writes only
/// the private scratch, so no caller output is touched until the join's own
/// preflight (re-run inside the delegate) clears.
#[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn packed_yuv422_process_native(
  plan: &ResamplePlan,
  native_planar: &mut Option<std::boxed::Box<NativePlanarYuv>>,
  y_scratch: &mut std::vec::Vec<u8>,
  u_scratch: &mut std::vec::Vec<u8>,
  v_scratch: &mut std::vec::Vec<u8>,
  resample_outputs: &mut Option<FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  packed: &[u8],
  y0_off: usize,
  y1_off: usize,
  u_off: usize,
  v_off: usize,
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || rgba.is_some() || hsv.is_some();
  let cw = w / 2;

  // Run the join's COMPLETE pre-feed rejection preflight FIRST — before the
  // fallible Y / U / V de-pack scratch grow — so EVERY rejection case
  // (out-of-sequence first row OR mid-frame output change) returns its
  // deterministic typed error, never AllocationFailed under allocation
  // pressure, and leaves the scratch untouched (the crate's
  // preflight-atomicity contract). `Ok(false)` is the no-output no-op:
  // return without reserving. `yuv_planar_process_native` re-runs this
  // identical preflight harmlessly, keeping a single source of truth.
  if !native_planar_preflight(
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }

  // De-pack the interleaved row into the private Y / U / V scratch. Y is
  // always de-packed (the join bins Y for both luma and colour); U / V are
  // de-packed only on a colour row (chroma_vsub == 1: a chroma row per Y
  // row). The de-pack writes only this private scratch, so no caller output
  // is touched until the join's own preflight (re-run inside the delegate
  // below) clears. On luma-only / no-colour rows the join never reads chroma,
  // so the scratch is left as-is and the join gets empty U / V slices —
  // keeping a luma-only sink from planning or allocating chroma state.
  let grow = |scratch: &mut std::vec::Vec<u8>, len: usize| -> Result<(), MixedSinkerError> {
    if scratch.len() < len {
      scratch
        .try_reserve_exact(len - scratch.len())
        .map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
      scratch.resize(len, 0);
    }
    Ok(())
  };
  grow(y_scratch, w)?;
  for (i, group) in packed.chunks_exact(4).enumerate() {
    y_scratch[i * 2] = group[y0_off];
    y_scratch[i * 2 + 1] = group[y1_off];
  }
  if need_color {
    grow(u_scratch, cw)?;
    grow(v_scratch, cw)?;
    for (i, group) in packed.chunks_exact(4).enumerate() {
      u_scratch[i] = group[u_off];
      v_scratch[i] = group[v_off];
    }
  }

  let (u_plane, v_plane): (&[u8], &[u8]) = if need_color {
    (&u_scratch[..cw], &v_scratch[..cw])
  } else {
    (&[], &[])
  };
  yuv_planar_process_native(
    plan,
    native_planar,
    resample_outputs,
    rgb,
    rgba,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    &y_scratch[..w],
    u_plane,
    v_plane,
    matrix,
    full_range,
    idx,
    w,
    h,
    1,
    || ResamplePlan::area(cw, h, plan.out_w(), plan.out_h()),
    use_simd,
  )
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
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    // New frame: restart the native join and clear the per-frame frozen
    // native/row-stage route so the next frame may pick either tier; a
    // mid-frame flip stays rejected. Gated to the native tier's feature
    // intersection (the planar join the native tier reuses is compiled only
    // under `yuv-planar`).
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
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
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native_planar,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_y_full,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_u_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_v_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_native_route,
      ..
    } = self;
    let packed = row.yuyv();

    // Non-identity plan: feed the shared packed-YUV dual-stream tail —
    // luma de-interleaves the Y bytes and bins them; colour converts the
    // packed row to RGB and bins that. The span kind picks the engine —
    // area binning or signed-coefficient filter (both stage the same
    // de-interleaved Y / converted RGB and share the emit, so filter colour
    // equals the RGB filter of the converted pixels and luma stays native
    // Y). Freeze + sequence-check before staging, so a no-output sink stays
    // a no-op and an out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // A `Filter` plan routes to the filter resampler (the native fast tier
      // is area-only and never sees a filter plan; the per-sink plan kind is
      // fixed at construction, so a filter sink bypasses the native/row-stage
      // route machinery entirely). Branched FIRST, before the native-route
      // guard below.
      if let SpanKind::Filter = plan.kind() {
        return packed_yuv422_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
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
      // Area plan. When the native tier is enabled (and the planar join it
      // reuses is compiled in), de-pack the interleaved row into Y / U / V
      // scratch and bin those planes at output resolution, converting once
      // per output row (YUYV: Y at 0,2 / U at 1 / V at 3). Otherwise (or under
      // `with_native(false)`) take the row-stage tier: bin the de-interleaved
      // Y for luma, convert the packed row to a source-width RGB row and bin
      // that.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        // Whether this call bears any output — the EXACT set both tiers'
        // preflight tests. The route freezes only on an output-bearing row a
        // tier ACCEPTS; a no-output call must not freeze (route-invisible).
        let need_output =
          luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
        // The RFC #238 splice stage. A filter plan already returned above, so
        // `area_plan` is true and the selector reproduces the former `*native`
        // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
        let take_native = matches!(
          select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: cfg!(all(feature = "yuv-packed", feature = "yuv-planar")),
              with_native: *native,
              area_plan: true,
            },
          ),
          InsertionPoint::NativeCodes
        );
        // Reject a mid-frame native/row-stage route flip BEFORE either tier's
        // dispatch (the #186 CHECK-before / SET-after template).
        if need_output
          && let Some(frozen) = *frozen_native_route
          && frozen != take_native
        {
          return Err(MixedSinkerError::NativeRouteChanged(
            NativeRouteChanged::new(idx),
          ));
        }
        if take_native {
          // Dispatch first; freeze the route to native ONLY after the call
          // returns Ok on an output-bearing row.
          packed_yuv422_process_native(
            plan,
            native_planar,
            packed_yuv_y_full,
            packed_yuv_u_half,
            packed_yuv_v_half,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            packed,
            0,
            2,
            1,
            3,
            matrix,
            full_range,
            idx,
            w,
            h,
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
      }
      // Row-stage tail (the only route under a `yuv-packed`-solo build).
      // Dispatch, then under the native tier freeze the route to row-stage
      // only when the call accepts an output-bearing row.
      packed_yuv422_dual_resample(
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
      )?;
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        let need_output =
          luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
      }
      return Ok(());
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
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
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
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native_planar,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_y_full,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_u_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_v_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_native_route,
      ..
    } = self;
    let packed = row.uyvy();

    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // Filter plan: native is area-only, so route straight to the filter
      // resampler (see the Yuyv422 impl above for the full rationale).
      if let SpanKind::Filter = plan.kind() {
        return packed_yuv422_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
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
      // Area plan — UYVY: Y at 1,3 / U at 0 / V at 2.
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        let need_output =
          luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
        // The RFC #238 splice stage. A filter plan already returned above, so
        // `area_plan` is true and the selector reproduces the former `*native`
        // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
        let take_native = matches!(
          select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: cfg!(all(feature = "yuv-packed", feature = "yuv-planar")),
              with_native: *native,
              area_plan: true,
            },
          ),
          InsertionPoint::NativeCodes
        );
        if need_output
          && let Some(frozen) = *frozen_native_route
          && frozen != take_native
        {
          return Err(MixedSinkerError::NativeRouteChanged(
            NativeRouteChanged::new(idx),
          ));
        }
        if take_native {
          packed_yuv422_process_native(
            plan,
            native_planar,
            packed_yuv_y_full,
            packed_yuv_u_half,
            packed_yuv_v_half,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            packed,
            1,
            3,
            0,
            2,
            matrix,
            full_range,
            idx,
            w,
            h,
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
      }
      packed_yuv422_dual_resample(
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
      )?;
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        let need_output =
          luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
      }
      return Ok(());
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
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
    {
      self.frozen_native_route = None;
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
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      native_planar,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_y_full,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_u_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      packed_yuv_v_half,
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      frozen_native_route,
      ..
    } = self;
    let packed = row.yvyu();

    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      // Filter plan: native is area-only, so route straight to the filter
      // resampler (see the Yuyv422 impl above for the full rationale).
      if let SpanKind::Filter = plan.kind() {
        return packed_yuv422_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
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
      // Area plan — YVYU: Y at 0,2 / V at 1 / U at 3 (V/U swapped vs YUYV).
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        let need_output =
          luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
        // The RFC #238 splice stage. A filter plan already returned above, so
        // `area_plan` is true and the selector reproduces the former `*native`
        // boolean bit-for-bit (`cfg!` is true wherever this block compiles).
        let take_native = matches!(
          select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: cfg!(all(feature = "yuv-packed", feature = "yuv-planar")),
              with_native: *native,
              area_plan: true,
            },
          ),
          InsertionPoint::NativeCodes
        );
        if need_output
          && let Some(frozen) = *frozen_native_route
          && frozen != take_native
        {
          return Err(MixedSinkerError::NativeRouteChanged(
            NativeRouteChanged::new(idx),
          ));
        }
        if take_native {
          packed_yuv422_process_native(
            plan,
            native_planar,
            packed_yuv_y_full,
            packed_yuv_u_half,
            packed_yuv_v_half,
            resample_outputs,
            rgb,
            rgba,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            packed,
            0,
            2,
            3,
            1,
            matrix,
            full_range,
            idx,
            w,
            h,
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          return Ok(());
        }
      }
      packed_yuv422_dual_resample(
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
      )?;
      #[cfg(all(feature = "yuv-packed", feature = "yuv-planar"))]
      {
        let need_output =
          luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
        if frozen_native_route.is_none() && need_output {
          *frozen_native_route = Some(false);
        }
      }
      return Ok(());
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
