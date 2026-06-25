//! 8-bit planar YUV `MixedSinker` impls: Yuv410p / Yuv420p / Yuv422p / Yuv444p / Yuv440p.

use super::{
  AveragingDomainChanged, GeometryOverflow, HsvFrameMut, InsufficientBuffer, MixedSinker,
  MixedSinkerError, NativeRouteChanged, RowIndexOutOfRange, RowShapeMismatch, RowSlice,
  WidthAlignment, check_dimensions_match, frozen_outputs_check,
  planar_resample::{planar_dual_filter_resample, planar_dual_resample},
  rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{
  ColorMatrix, PixelSink,
  resample::{
    AreaStream, AveragingDomain, FilterStream, InsertionContext, InsertionPoint, OutOfSequenceRow,
    PlanGeometry, ResampleError, ResamplePlan, select_insertion_point, try_zeroed,
  },
  row::*,
  source::*,
};
// The RFC #238 linear-light tail + its caller-configurable transfer curve
// drive the `rgb`-gated `AveragingDomain::Linear` dispatch only.
#[cfg(feature = "rgb")]
use super::linear_light;
#[cfg(feature = "rgb")]
use crate::resample::TransferFunction;

/// `Yuv420p` ships the native 4:2:0 fast tier ([`yuv420p_process_native`]),
/// so it is statically eligible to splice an [`AveragingDomain::Encoded`]
/// area downscale at the native codes.
const YUV420P_NATIVE_ELIGIBLE: bool = true;

/// `Yuv422p` / `Yuv444p` / `Yuv440p` ship the non-4:2:0 native planar fast
/// tier ([`yuv_planar_process_native`]), so each is statically eligible to
/// splice an [`AveragingDomain::Encoded`] area downscale at the native codes.
const YUV_PLANAR_8BIT_NATIVE_ELIGIBLE: bool = true;

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
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(native) = self.native_420.as_mut() {
      native.reset();
    }
    if let Some(bicublin) = self.bicublin_420.as_mut() {
      bicublin.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    // New frame: clear the per-frame frozen native/row-stage route and
    // averaging domain so the next frame may pick either tier / any domain; a
    // mid-frame flip stays rejected.
    self.frozen_native_route = None;
    self.frozen_domain = None;
    self.resample_outputs = None;
    // New frame: drop the RFC #238 linear-light accumulator (if any) so the
    // next frame re-seeds it from row 0.
    #[cfg(feature = "rgb")]
    {
      self.linear_light_frame = None;
    }
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
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      native,
      native_420,
      hsv_planar,
      bicublin_420,
      frozen_native_route,
      frozen_domain,
      averaging_domain,
      #[cfg(feature = "rgb")]
      linear_mode,
      #[cfg(feature = "rgb")]
      linear_light_frame,
      #[cfg(feature = "rgb")]
      linear_scene_scratch,
      #[cfg(feature = "rgb")]
      transfer_function,
      ..
    } = self;

    // Non-identity plan: the native tier bins the Y/U/V planes at
    // output resolution and converts once per output row; the
    // row-stage tier converts this source row at source width, then
    // area-streams it. `with_native(false)` forces the latter.
    if let Some(plan) = plan.as_ref() {
      // RFC #238 Phase 2 — single always-compiled choke point for the averaging
      // domain, BEFORE any filter / native / row-stage branching, so a
      // non-encoded sink can NEVER fall through to the Encoded path under ANY
      // feature combination. The match is EXHAUSTIVE with no wildcard arm: a
      // future `AveragingDomain` variant fails to compile here until it is
      // explicitly handled, so the silent-fallback class is structurally
      // impossible rather than merely audited. The `Encoded` arm is empty —
      // control continues into the encoded dispatch below; `Linear` and
      // `Premultiplied` both return.
      //
      // `need_output` — whether this call carries any output — is the EXACT set
      // both tiers' preflight tests (`need_luma || need_color` =
      // `luma || luma_u16 || rgb || rgba || hsv`); it gates BOTH the domain
      // freeze here and the native/row-stage route freeze below, so a no-output
      // call (which consumes no stream state) freezes neither.
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      // CHECK the averaging-domain freeze BEFORE the choke-point match (so the
      // freeze guards the domain choice itself), parallel to the frozen native
      // route. This is CHECK-ONLY — the matching SET happens AFTER the selected
      // path ACCEPTS an output-bearing row (mirroring `frozen_native_route`'s
      // timing below), never before dispatch. Committing the freeze before the
      // row is accepted would poison a retry: a row the selected path rejects
      // (an unsupported domain / filter plan, an out-of-sequence or
      // output-changed row, an alloc failure) must leave `frozen_domain`
      // UNCHANGED so the caller can correct the config and retry the SAME row.
      // A no-output row neither checks nor sets (a true route-invisible no-op).
      if need_output
        && let Some(frozen) = *frozen_domain
        && frozen != *averaging_domain
      {
        return Err(MixedSinkerError::AveragingDomainChanged(
          AveragingDomainChanged::new(idx),
        ));
      }
      match *averaging_domain {
        AveragingDomain::Encoded => {}
        // `with_averaging_domain` is gated on `yuv-planar` alone, but the
        // linear-light tail decodes to RGB and so compiles only under `rgb`.
        // Under `rgb` it runs the linear tail (which itself rejects a filter
        // plan with the typed `UnsupportedFilter` — the Linear domain is
        // area-only); without `rgb` it returns the typed
        // `LinearDomainUnsupported` rather than silently downgrading to the
        // encoded average.
        AveragingDomain::Linear => {
          #[cfg(feature = "rgb")]
          {
            let matrix = row.matrix();
            let full_range = row.full_range();
            let tf = transfer_function.unwrap_or_else(|| TransferFunction::for_matrix(matrix));
            // Dispatch first; commit the domain freeze to Linear ONLY when the
            // tail ACCEPTS an output-bearing row. `linear_light_resample`
            // returns Ok(()) without consuming for a no-output call and Err for
            // a filter / out-of-sequence / output-changed / alloc reject, so
            // `r.is_ok() && need_output` is exactly an accepted output-bearing
            // row — a rejected row leaves `frozen_domain` unset so a
            // corrected-domain retry of the SAME row is not falsely rejected.
            let r = linear_light::linear_light_resample(
              linear_light_frame,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              linear_scene_scratch,
              tf,
              *linear_mode,
              plan,
              row.y(),
              idx,
              w,
              h,
              use_simd,
              |_idx, dst| {
                yuv_420_to_rgb_row(
                  row.y(),
                  row.u_half(),
                  row.v_half(),
                  dst,
                  w,
                  matrix,
                  full_range,
                  use_simd,
                );
              },
              |_idx, dst| {
                // Scene-referred: the SAME affine matrix, real-valued and
                // unclamped (the 4:2:0 horizontal chroma upsample, as the
                // Q15 `yuv_420_to_rgb_row` above does).
                crate::row::scalar::yuv_420_to_rgb_f32_unclamped_row(
                  row.y(),
                  row.u_half(),
                  row.v_half(),
                  dst,
                  w,
                  matrix,
                  full_range,
                );
              },
            );
            if r.is_ok() && need_output && frozen_domain.is_none() {
              *frozen_domain = Some(AveragingDomain::Linear);
            }
            return r;
          }
          #[cfg(not(feature = "rgb"))]
          {
            return Err(plan.linear_domain_unsupported().into());
          }
        }
        // `Yuv420p` has no alpha plane, so premultiplied weighting is a category
        // error here — reject with a typed error rather than silently running
        // the encoded average. Phase 5 wires Premultiplied for actual alpha
        // formats; on these non-alpha formats rejecting is correct.
        AveragingDomain::Premultiplied => {
          return Err(plan.premultiplied_domain_unsupported().into());
        }
      }
      // A `Filter` plan routes to the filter resampler, which converts the
      // separate Y/U/V planes to a source-width RGB row (the same
      // `yuv_420_to_rgb_row` kernel the row-stage tier uses) and
      // filter-resamples it plus the native Y. The native fast tier is an
      // area-specific optimization, so it never sees a filter plan; the
      // per-sink plan kind is fixed at construction, so a filter sink
      // bypasses the native/row-stage route machinery entirely (no
      // `frozen_native_route` interaction).
      if plan.kind().is_filter() {
        let matrix = row.matrix();
        let full_range = row.full_range();
        // A BICUBLIN plan ([`Bicublin`](crate::resample::Bicublin)) carries a
        // SECOND (chroma) window set, so it cannot route through the
        // single-kernel `planar_dual_filter_resample` (which filters one
        // converted RGB row). It instead filters the Y / U / V planes
        // SEPARATELY in plane space (cubic luma, linear chroma) and converts
        // the filtered planes — the filter twin of the native area tier. The
        // existing single-kernel filter path below is reached only when this
        // is NOT a bicublin plan, so it stays byte-unchanged.
        if plan.is_bicublin() {
          return bicublin_yuv420p_resample(
            plan,
            bicublin_420,
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
            matrix,
            full_range,
            idx,
            w,
            h,
            use_simd,
          );
        }
        return planar_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
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
      // Reject a mid-frame native/row-stage route flip BEFORE either tier's
      // dispatch. The two tiers carry independent, in-order, once-only
      // stream state, so splitting a frame across them yields a
      // mixed/partial frame rather than a deterministic rejection. The route
      // is both CHECKED here and frozen below (the SET) ONLY on an
      // output-bearing row a tier ACCEPTS — both gate on `need_output`. A
      // no-output call therefore neither checks nor freezes the route: it is
      // a true no-op, route-invisible regardless of row index, so it can
      // never spuriously trip `NativeRouteChanged` after the route is
      // frozen. A preflight-rejected (out-of-sequence / frozen)
      // output-bearing call returns Err before the SET, so it leaves
      // `frozen_native_route` untouched and a later same-or-other-route
      // retry is not falsely rejected.
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection. The native-vs-row-stage choice is
      // re-expressed through the framework selector: for the encoded domain
      // it returns `NativeCodes` exactly when this format is native-eligible,
      // the sink enabled the native tier (`*native`), and the plan is an area
      // downscale (a filter plan already returned above, so `area_plan` is
      // always true here). That reproduces the former `if *native` boolean
      // bit-for-bit — the route-freeze and rejection above are unchanged, so
      // the dispatch stays byte-identical. The native tier is
      // [`InsertionPoint::NativeCodes`] (bin codes, then convert); the
      // row-stage tier is [`InsertionPoint::EncodedOutput`] (convert, then
      // area-stream the output).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV420P_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // Dispatch first; freeze the route to native ONLY after the call
          // returns Ok on an output-bearing row. A no-output call returns
          // Ok(()) with `need_output` false (no freeze); an out-of-sequence /
          // frozen row returns Err via `?` (no freeze) — so only an accepted
          // output-bearing row commits the route.
          yuv420p_process_native(
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
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // Row-stage tail. Same CHECK-before / SET-after split: dispatch,
          // then freeze the route to row-stage only when the call accepts an
          // output-bearing row (a no-output call returns Ok with
          // `need_output` false; an out-of-sequence / frozen row returns Err
          // via `?`).
          yuv420p_process_resampled(
            plan,
            rgb_stream,
            luma_stream,
            hsv_planar,
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
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
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
    // HSV-without-RGB-or-RGBA goes through the direct `yuv_420_to_hsv_row`
    // kernel (no source-width RGB scratch). When RGB or RGBA is *also*
    // attached the RGB kernel runs anyway, so HSV derives off that buffer
    // for free — the cheap path — and `need_rgb_kernel` keeps it alive.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv_420_to_hsv_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

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
    // The native 4:2:0 join is integer area-only; a filter plan reaches it
    // when a FilteredResampler is attached with the native tier enabled, so
    // reject it before building any plane's area stream.
    if plan.kind().is_filter() {
      return Err(plan.unsupported_filter());
    }
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

/// The COMPLETE pre-feed rejection preflight shared by the native 4:2:0
/// path and its semi-planar reuse: the no-output short-circuit, the
/// first-row out-of-sequence check, the frozen-output check, AND the
/// post-freeze sequence check — every rejection [`yuv420p_process_native`]
/// performs before its first fallible allocation (the join build / chroma
/// reserve / feed). All four run BEFORE any fallible allocation so a
/// rejected row leaves no state change (the crate's preflight-atomicity /
/// recoverable-allocation contract).
/// Returns `Ok(false)` for a no-output call (the caller should no-op),
/// `Ok(true)` to proceed into the join, `Err(OutOfSequenceRow)` for a
/// rejected out-of-sequence first row, or `Err(ResampleOutputsChanged)`
/// for a mid-frame output-set change.
///
/// The semi-planar wrapper runs this FIRST, before reserving / filling its
/// U / V de-interleave scratch, so NO rejection case — no-output, OOS
/// first row, OR mid-frame output change — can reach a wrapper allocation
/// (which under allocation pressure would surface AllocationFailed instead
/// of the deterministic typed error). `yuv420p_process_native` re-runs it
/// in place of its inline block; the double-run is idempotent (the freeze
/// stores on the first output-bearing row, the second run is a matching
/// check, and the OOS-first-row branch is `is_none()`-guarded so it is
/// skipped once frozen).
///
/// A no-output call has nothing to sequence and stays a no-op regardless
/// of the row index — returned before the freeze, the Y-stream sequence
/// check, and the join allocation so it stores no frozen-output snapshot
/// that a later attach-then-retry would trip on. The Y stream is bound to
/// has-output here, not always fed: a no-output row must not advance it
/// (otherwise the snapshot taken on the first output-bearing row of a
/// retry mismatches the rejected no-output row's frozen-as-absent set).
///
/// The native 4:2:0 path bins the Y plane on every output-bearing row
/// (luma is implicit), so the Y stream is the canonical per-row sequence
/// counter. The conditional ordering is load-bearing: on the first row of
/// a frame nothing is frozen yet, so the out-of-sequence row is rejected
/// here — BEFORE the freeze — so a rejected first row stores no snapshot
/// that would poison a retry. A later row runs the freeze (the frozen
/// check below) first, so a mid-frame output-set change is reported as
/// ResampleOutputsChanged rather than masked by the join being rebuilt at
/// row 0; the post-freeze sequence check then rejects an out-of-sequence
/// later row (including the failure-retry case where the join was never
/// built, so `expected == 0`) before the caller's fallible allocation.
#[allow(clippy::too_many_arguments)]
pub(super) fn yuv420p_native_preflight(
  native_420: &Option<std::boxed::Box<NativeYuv420>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
  need_luma: bool,
  need_color: bool,
) -> Result<bool, MixedSinkerError> {
  // The 8-bit planar / semi-planar join has no native-depth u16 colour
  // outputs, so `rgb_u16` / `rgba_u16` are frozen as absent.
  native_preflight_core(
    native_420.as_ref().map_or(0, |join| join.y.next_y()),
    resample_outputs,
    rgb,
    rgba,
    &None,
    &None,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )
}

/// The COMPLETE 4-point pre-feed rejection logic shared by the 8-bit
/// ([`yuv420p_native_preflight`]) and the high-bit
/// ([`crate::sinker::mixed::subsampled_4_2_0_high_bit::yuv420p16_native_preflight`])
/// native 4:2:0 fast tiers: the no-output short-circuit, the first-row
/// pre-freeze out-of-sequence check, [`frozen_outputs_check`], AND the
/// post-freeze sequence check — every rejection point a native path must
/// run before its first fallible allocation. The join-typed expected-row
/// computation (`join.y.next_y()`) lives in the thin per-element wrappers;
/// each passes its already-computed `expected` here so this body stays
/// element-agnostic (the u8 join carries [`NativeYuv420`], the u16 join
/// `NativeYuv420U16`).
///
/// Returns `Ok(false)` for a no-output call (caller no-ops), `Ok(true)`
/// to proceed into the join, `Err(OutOfSequenceRow)` for a rejected
/// out-of-sequence first OR post-freeze row, or
/// `Err(ResampleOutputsChanged)` for a mid-frame output-set change. The
/// conditional ordering is load-bearing — see [`yuv420p_native_preflight`]
/// and the crate's preflight-atomicity contract.
#[allow(clippy::too_many_arguments)]
pub(super) fn native_preflight_core(
  expected: usize,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  rgb_u16: &Option<&mut [u16]>,
  rgba_u16: &Option<&mut [u16]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
  need_luma: bool,
  need_color: bool,
) -> Result<bool, MixedSinkerError> {
  if !need_luma && !need_color {
    return Ok(false);
  }
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
    rgb_u16,
    rgba_u16,
    &None,
    &None,
    &None,
    &None,
    &None,
    hsv,
    &None,
    idx,
  )?;
  // Post-freeze sequence check: once `resample_outputs` is frozen the
  // pre-freeze first-row branch above is skipped, so an out-of-sequence
  // row whose outputs match the frozen set (the failure-retry case — the
  // join may be `None`, leaving `expected == 0`) must be rejected here,
  // BEFORE the caller's fallible chroma / de-interleave allocation, rather
  // than only at the join's own `check_sequence`. The freeze does not
  // advance the Y stream, so `expected` is unchanged; running this after
  // the frozen check preserves error precedence (a row that is both
  // output-changed and out-of-sequence reports ResampleOutputsChanged).
  if expected != idx {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(expected, idx),
    )));
  }
  Ok(true)
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
  native_420: &mut Option<std::boxed::Box<NativeYuv420>>,
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
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();
  // HSV-without-RGB-or-RGBA emits directly from the binned output-width
  // Y/U/V via `yuv_444_to_hsv_row` — chroma must still be decimated
  // (`need_color`), but no output-width RGB scratch is staged. RGB or
  // RGBA attached keeps the convert-once-then-derive path.
  let want_hsv_direct = hsv.is_some() && rgb.is_none() && rgba.is_none();

  // Complete pre-feed rejection preflight — no-output short-circuit,
  // first-row out-of-sequence rejection, the frozen-output check, AND the
  // post-freeze sequence check, all ahead of any fallible allocation (see
  // [`yuv420p_native_preflight`]). Extracted in full so the semi-planar
  // caller can run this identical gate BEFORE it reserves and fills its
  // U / V de-interleave scratch — otherwise a rejected row (out-of-sequence
  // first OR later row, or a mid-frame output change) would grow sink state
  // under allocation pressure and surface AllocationFailed instead of the
  // deterministic typed error. `Ok(false)` is the no-output no-op; the
  // `check_sequence` below is now redundant but kept (it stays
  // behavior-identical for in-sequence rows).
  if !yuv420p_native_preflight(
    native_420,
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
    None => {
      // Build the join (its U / V de-interleave scratch allocates recoverably)
      // and box it recoverably (`try_box`, not `Box::new` — the latter aborts
      // on OOM). Both fallible steps run BEFORE `native_420.insert`, so a
      // refusal at either returns `Err` with the field still `None` — no caller
      // output is touched and the next row retries (the first-row-transactional
      // contract).
      let join = NativeYuv420::new(plan, w, h, need_color)?;
      let boxed = crate::resample::try_box(join).map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )))
      })?;
      native_420.insert(boxed)
    }
  };
  join.check_sequence(idx)?;
  // No output-width RGB scratch for the HSV-direct case — the emit loop
  // converts the binned Y/U/V straight to HSV.
  if need_color && !want_hsv_direct {
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
  } = &mut **join;
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
      if want_hsv_direct {
        // RGB-free: convert the binned output-width Y/U/V straight to HSV.
        let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
        let (hp, sp, vp) = hsv.hsv();
        yuv_444_to_hsv_row(
          y_out,
          u_row,
          v_row,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      } else {
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
    }
    staged[0][slot] = false;
    staged[1][slot] = false;
    staged[2][slot] = false;
    *next_emit += 1;
  }
  Ok(())
}

/// BICUBLIN per-plane filter join for 4:2:0 — the signed-coefficient twin of
/// [`NativeYuv420`]. Y streams on the frame grid through the **cubic** luma
/// filter, U and V on the chroma grid (half width, ceil-half height) through
/// the **linear** chroma filter, every plane resampled to FULL output
/// resolution. Each plane's in-order emissions land in a full-output-plane
/// buffer; the moment all participating planes have emitted an output row it
/// is converted through the 4:4:4 kernel at output width.
///
/// Unlike the area join, a filter window is wide, so a plane can lead another
/// by **more than one** output row (the cubic luma vertical filter and the
/// linear chroma vertical filter complete output rows at different cadences
/// relative to the interleaved source feed). A fixed two-slot ring cannot
/// absorb that, so each plane lands in its own full-output-resolution buffer
/// and a per-plane "rows emitted" cursor tracks how far each has progressed;
/// output rows are converted up to the minimum cursor across the participating
/// planes. The buffers are output-proportional (caller geometry), reserved
/// fallibly at join creation.
pub(super) struct BicublinYuv420 {
  /// Luma plane filter stream (cubic kernel), `src_w x src_h -> out`.
  y: FilterStream<u8>,
  /// Full-output-resolution landing buffer for the filtered Y, `out_w x
  /// out_h`. Always present — colour reads it for the convert and luma copies
  /// from it.
  y_plane: std::vec::Vec<u8>,
  /// Output rows the Y stream has emitted so far (the stream emits in order,
  /// so this is a dense `0..` count).
  y_emitted: usize,
  /// Chroma half of the join — absent for luma-only sinks (which never touch
  /// the chroma planes). Frame-constant by the frozen-output contract.
  chroma: Option<BicublinChroma>,
  /// Next output row to convert / emit.
  next_emit: usize,
}

/// Chroma-grid filter streams and landing buffers of [`BicublinYuv420`].
struct BicublinChroma {
  /// U plane filter stream (linear kernel), `chroma_w x chroma_h -> out`.
  u: FilterStream<u8>,
  /// V plane filter stream (linear kernel), `chroma_w x chroma_h -> out`.
  v: FilterStream<u8>,
  u_plane: std::vec::Vec<u8>,
  v_plane: std::vec::Vec<u8>,
  /// Output rows the U / V streams have emitted so far (both advance in
  /// lockstep — same geometry, same chroma feed — but tracked independently so
  /// a future asymmetry cannot silently desync the drain).
  u_emitted: usize,
  v_emitted: usize,
}

impl BicublinYuv420 {
  /// Builds the per-plane BICUBLIN join from a [`ResamplePlan::is_bicublin`]
  /// plan: the luma stream from the cubic windows
  /// ([`ResamplePlan::filter_h`]/[`ResamplePlan::filter_v`]) over the frame
  /// grid, and (when colour is attached) the U / V streams from the linear
  /// chroma windows
  /// ([`ResamplePlan::filter_h_chroma`]/[`ResamplePlan::filter_v_chroma`])
  /// over the chroma grid. Each plane lands in a full-output-resolution
  /// buffer.
  fn new(plan: &ResamplePlan, w: usize, h: usize, need_color: bool) -> Result<Self, ResampleError> {
    let (fh, fv) = (
      plan
        .filter_h()
        .expect("bicublin plan carries luma horizontal windows"),
      plan
        .filter_v()
        .expect("bicublin plan carries luma vertical windows"),
    );
    let y = FilterStream::<u8>::new(fh, fv, w, h, 1)?;
    let alloc =
      |_| ResampleError::AllocationFailed(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()));
    let plane_len = plan.out_w().checked_mul(plan.out_h()).ok_or_else(|| {
      ResampleError::Overflow(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()))
    })?;
    let chroma = if need_color {
      let (fhc, fvc) = (
        plan
          .filter_h_chroma()
          .expect("bicublin plan carries chroma horizontal windows"),
        plan
          .filter_v_chroma()
          .expect("bicublin plan carries chroma vertical windows"),
      );
      // 4:2:0 chroma grid: half width, ceil-half height (matches the plan's
      // chroma window source dims).
      let (cw, ch) = (w / 2, h.div_ceil(2));
      Some(BicublinChroma {
        u: FilterStream::<u8>::new(fhc, fvc, cw, ch, 1)?,
        v: FilterStream::<u8>::new(fhc, fvc, cw, ch, 1)?,
        u_plane: try_zeroed(plane_len).map_err(alloc)?,
        v_plane: try_zeroed(plane_len).map_err(alloc)?,
        u_emitted: 0,
        v_emitted: 0,
      })
    } else {
      None
    };
    Ok(Self {
      y,
      y_plane: try_zeroed(plane_len).map_err(alloc)?,
      y_emitted: 0,
      chroma,
      next_emit: 0,
    })
  }

  pub(super) fn reset(&mut self) {
    self.y.reset();
    self.y_emitted = 0;
    if let Some(chroma) = self.chroma.as_mut() {
      chroma.u.reset();
      chroma.v.reset();
      chroma.u_emitted = 0;
      chroma.v_emitted = 0;
    }
    self.next_emit = 0;
  }

  /// Sequencing preflight across all three plane streams — checked before any
  /// plane is fed so a violating call mutates nothing. Chroma rows advance
  /// once per source-row pair, so their expected counter is the ceiling half
  /// of the source row. Mirrors [`NativeYuv420::check_sequence`].
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

/// BICUBLIN per-plane filter path for [`MixedSinker<Yuv420p, R>`]: filters
/// the Y plane with the cubic luma kernel and the U / V planes with the linear
/// chroma kernel — each in YUV-plane space at its own resolution — then
/// converts the filtered planes to RGB at output resolution through the 4:4:4
/// kernel (the **same** convert the native area tier
/// [`yuv420p_process_native`] runs on its binned planes). It is the filter
/// analog of that area tier; the only differences are the signed-coefficient
/// [`FilterStream`]s in place of the integer [`AreaStream`]s and the
/// full-output-plane landing in place of the two-slot ring (a filter window is
/// wide, so a plane can lead by more than one output row).
///
/// Atomic preflight, mirroring [`yuv420p_process_native`] /
/// [`planar_dual_filter_resample`]: the complete pre-feed rejection
/// ([`yuv420p_native_preflight`] — no-output short-circuit, first-row
/// out-of-sequence, frozen-output, post-freeze sequence) runs before any
/// fallible allocation, then the join build, the [`Self::check_sequence`]
/// across all three streams, and the colour scratch growth all precede the
/// first `feed_row`. So a rejected or failed row mutates no caller output and
/// returns the deterministic typed error (never `AllocationFailed`-on-abort),
/// and a corrected retry of the SAME row is accepted. Luma is the native Y
/// filter-resampled (the YUV luma contract — never colour-derived); these are
/// 8-bit sources, so the `u8` stream finalises to the full `u8` range, which
/// IS the native range (no sub-16-bit clamp).
#[allow(clippy::too_many_arguments)]
pub(super) fn bicublin_yuv420p_resample(
  plan: &ResamplePlan,
  bicublin_420: &mut Option<std::boxed::Box<BicublinYuv420>>,
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
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();
  // HSV-without-RGB-or-RGBA emits HSV straight from the filtered
  // output-width Y/U/V (no output-width RGB scratch); chroma is still
  // filtered (`need_color`). See [`yuv420p_process_native`].
  let want_hsv_direct = hsv.is_some() && rgb.is_none() && rgba.is_none();

  // Complete pre-feed rejection preflight ahead of any fallible allocation
  // (no-output short-circuit, first-row out-of-sequence, frozen-output, AND
  // post-freeze sequence) — the SAME gate the native tier runs, keyed on the
  // Y stream's `next_y` (luma is implicit on every output-bearing row). A
  // rejected row therefore allocates nothing and a no-output call is a true
  // route-invisible no-op.
  if !yuv420p_native_preflight_bicublin(
    bicublin_420,
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
  // The join's chroma half is fixed at creation; if the frame's colour
  // capability differs (outputs attached since the previous frame — the frozen
  // check pins them WITHIN a frame, not across frames), rebuild it.
  if bicublin_420
    .as_ref()
    .is_some_and(|join| join.chroma.is_some() != need_color)
  {
    *bicublin_420 = None;
  }
  let join = match bicublin_420 {
    Some(join) => join.as_mut(),
    None => {
      // Build the join (its Y / U / V filter streams allocate recoverably) and
      // box it recoverably (`try_box`, not `Box::new` — the latter aborts on
      // OOM). Both fallible steps run BEFORE `bicublin_420.insert`, so a refusal
      // at either returns `Err` with the field still `None` — no caller output
      // is touched and the next row retries from scratch (the
      // first-row-transactional contract).
      let join = BicublinYuv420::new(plan, w, h, need_color)?;
      let boxed = crate::resample::try_box(join).map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )))
      })?;
      bicublin_420.insert(boxed).as_mut()
    }
  };
  join.check_sequence(idx)?;
  // No output-width RGB scratch for the HSV-direct case — the emit loop
  // converts the binned Y/U/V straight to HSV.
  if need_color && !want_hsv_direct {
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

  // Feed the planes; everything past this point is infallible. Each stream
  // lands every emitted output row in its full-output-resolution plane buffer
  // and advances that plane's emitted cursor.
  let BicublinYuv420 {
    y,
    y_plane,
    y_emitted,
    chroma,
    next_emit,
  } = join;
  y.feed_row(idx, y_row, use_simd, |oy, out_row| {
    y_plane[oy * ow..oy * ow + ow].copy_from_slice(out_row);
    *y_emitted = oy + 1;
  })?;
  if let Some(c) = chroma.as_mut()
    && idx.is_multiple_of(2)
  {
    let cidx = idx / 2;
    let BicublinChroma {
      u,
      v,
      u_plane,
      v_plane,
      u_emitted,
      v_emitted,
    } = c;
    u.feed_row(cidx, u_half, use_simd, |oy, out_row| {
      u_plane[oy * ow..oy * ow + ow].copy_from_slice(out_row);
      *u_emitted = oy + 1;
    })?;
    v.feed_row(cidx, v_half, use_simd, |oy, out_row| {
      v_plane[oy * ow..oy * ow + ow].copy_from_slice(out_row);
      *v_emitted = oy + 1;
    })?;
  }

  // Convert every output row all participating planes have now emitted — up to
  // the minimum emitted cursor across Y (and U / V when colour is attached).
  let ready = match chroma.as_ref() {
    Some(c) => (*y_emitted).min(c.u_emitted).min(c.v_emitted),
    None => *y_emitted,
  };
  while *next_emit < ready {
    let oy = *next_emit;
    let y_out = &y_plane[oy * ow..oy * ow + ow];

    if let Some(buf) = luma.as_deref_mut() {
      buf[oy * ow..(oy + 1) * ow].copy_from_slice(y_out);
    }
    if let Some(buf) = luma_u16.as_deref_mut() {
      for (dst, &src) in buf[oy * ow..(oy + 1) * ow].iter_mut().zip(y_out) {
        *dst = src as u16;
      }
    }
    if let Some(c) = chroma.as_ref() {
      let u_out = &c.u_plane[oy * ow..oy * ow + ow];
      let v_out = &c.v_plane[oy * ow..oy * ow + ow];
      if want_hsv_direct {
        // RGB-free: convert the filtered output-width Y/U/V straight to HSV.
        let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
        let (hp, sp, vp) = hsv.hsv();
        yuv_444_to_hsv_row(
          y_out,
          u_out,
          v_out,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      } else {
        let out_rgb = &mut rgb_scratch[..ow * 3];
        yuv_444_to_rgb_row(
          y_out, u_out, v_out, out_rgb, ow, matrix, full_range, use_simd,
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
    }
    *next_emit += 1;
  }
  Ok(())
}

/// BICUBLIN preflight — the per-plane filter twin of
/// [`yuv420p_native_preflight`], keyed on the BICUBLIN join's Y stream
/// `next_y` (the canonical per-row counter; luma is implicit on every
/// output-bearing row). Runs [`native_preflight_core`] so the no-output
/// short-circuit, first-row out-of-sequence, frozen-output, and post-freeze
/// sequence checks all precede any fallible allocation, exactly as the native
/// tier does.
#[allow(clippy::too_many_arguments)]
fn yuv420p_native_preflight_bicublin(
  bicublin_420: &Option<std::boxed::Box<BicublinYuv420>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
  need_luma: bool,
  need_color: bool,
) -> Result<bool, MixedSinkerError> {
  native_preflight_core(
    bicublin_420.as_ref().map_or(0, |join| join.y.next_y()),
    resample_outputs,
    rgb,
    rgba,
    &None,
    &None,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )
}

/// Row-stage tier for the 4:2:0 planar family (the `with_native(false)`
/// path). Takes the Y plane and the **separate** half-width U / V planes
/// so the 4:2:0 semi-planar family reuses it after de-interleave.
///
/// #263 follow-up — **HSV-only** (no RGB / RGBA): instead of staging a
/// source-width RGB row, Y / U / V are binned on their own grids via the
/// shared [`HsvDirectPlanarYuv`](super::planar_resample::HsvDirectPlanarYuv)
/// join (4:2:0 chroma: half-width, ceil-half-height, `chroma_vsub = 2`) and
/// each output row is converted through `yuv_444_to_hsv_row` at output width
/// — RGB-free, and bit-identical to the native fast tier
/// ([`yuv420p_process_native`]). Luma (if also attached) derives from the
/// SAME binned Y.
#[allow(clippy::too_many_arguments)]
pub(super) fn yuv420p_process_resampled(
  plan: &ResamplePlan,
  rgb_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  luma_stream: &mut Option<std::boxed::Box<AreaStream<u8>>>,
  hsv_planar: &mut Option<std::boxed::Box<super::planar_resample::HsvDirectPlanarYuv>>,
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
  // Row-stage 4:2:0 tail is integer area-only: reject a filter plan before
  // any work, so the plan's empty area spans never reach an area stream.
  if plan.kind().is_filter() {
    return Err(plan.unsupported_filter().into());
  }
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();
  // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output row,
  // RGB-free — bit-identical to the native fast tier. RGB or RGBA attached
  // keeps the cheap RGB-staged path below. 4:2:0 chroma is half-width,
  // ceil-half-height (`chroma_vsub = 2`); the chroma plan weights an odd
  // trailing luma row by half (luma-domain vertical), exactly as the native
  // tier's `area_chroma_420`.
  if hsv.is_some() && rgb.is_none() && rgba.is_none() {
    return super::planar_resample::hsv_direct_resample(
      hsv_planar,
      resample_outputs,
      luma,
      luma_u16,
      hsv,
      y_row,
      u_half,
      v_half,
      matrix,
      full_range,
      2,
      || ResamplePlan::area_chroma_420(w / 2, h, plan.out_w(), plan.out_h()),
      w,
      plan,
      idx,
      use_simd,
    );
  }

  // Atomic preflight: every fallible step runs before any stream is
  // fed, so a failed call mutates no caller output and the frame can
  // restart via begin_frame.
  //
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
  // freshly-attached stream's row-0 sequence mismatch (a stream attached
  // mid-frame starts at row 0).
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
  // (3) Color-group preparation is also fallible (scratch sizing) and
  // scratch-mutating, so it runs before the luma feed too. The user
  // RGB buffer is output-sized; the source-width row always lands in
  // the scratch. (The overflow arm is defense in depth: any geometry
  // large enough to wrap w * 3 cannot plan — its span arena alloc is
  // out of reach first.)
  //
  // NOTE (#263 follow-up): the HSV-without-RGB case still stages a
  // SOURCE-WIDTH RGB row here. The direct + native fast tiers go
  // RGB-free, but the row-stage resample tail bins an RGB stream
  // (the `AreaStream` is keyed on the 3-channel RGB row) and derives
  // HSV per OUTPUT row off that stream. Eliminating the RGB scratch
  // for resample+HSV-only needs a dedicated HSV-plane resample (resample
  // the HSV planes, or resample Y/U/V then convert per output row) and
  // is deferred to a follow-up PR to keep this one scoped. `need_color`
  // therefore still triggers the source-width grow on the resample path.
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

// ---- Native fast tier: STRAIGHT-alpha 4:2:0 planar (Yuva420p) -----------
//
// Gated on `yuva` (which implies `yuv-planar`): this join, its α helper, and
// its entry point reference the alpha helpers (`row::alpha_extract`) that exist
// only under `yuva`, and their sole consumer is the `yuva`-gated
// `Yuva420p` sink. Without the gate a `yuv-planar`-without-`yuva` build pulls
// these items in but not their `yuva` dependencies, so it fails to compile.

/// Native decimation join for the **straight-alpha** 4:2:0 planar source
/// `Yuva420p` (the RFC #238 Phase 5 / #235 alpha resolution) — the
/// alpha-bearing sibling of [`NativeYuv420`].
///
/// The native campaign excluded every alpha format because PREMULTIPLIED
/// alpha is bin-then-convert-incompatible (colour has been scaled by α, so
/// a correct average must convert-at-source, premultiply, bin, then
/// un-premultiply late). STRAIGHT alpha is the exception: colour and α are
/// independent, so binning the native Y / U / V / A codes independently and
/// converting Y / U / V → RGB once per output pixel — with the binned α
/// attached straight — reproduces a `Yuva444p` convert of the pre-binned
/// planes. That is exactly what this join does:
/// - Y / U / V stream and stage exactly as [`NativeYuv420`] (so the rgb /
///   luma / hsv outputs are byte-identical to the no-alpha native tier),
///   embedded here verbatim so the proven 4:2:0 join is reused, not forked.
/// - α streams on the **luma grid** (α is full-resolution in `Yuva420p`,
///   like Y) through its own [`AreaStream<u8>`], staged into a parallel
///   two-slot ring. Per finalized output row the colour RGBA is emitted
///   opaque (the [`NativeYuv420`] path) and the binned α is then scattered
///   into its α slot via [`copy_alpha_plane_u8`](crate::row::alpha_extract::copy_alpha_plane_u8).
///
/// PREMULTIPLIED `Yuva420p` never reaches here — it is routed to the
/// existing packed-YUVA area tail
/// ([`packed_yuva444_resample`](super::packed_yuva444_resample)) BYTE-IDENTICALLY,
/// gated by `AlphaMode::Straight` at the sink.
#[cfg(feature = "yuva")]
pub(super) struct NativeYuva420 {
  /// The embedded no-alpha 4:2:0 join — Y / U / V binning, staging, and the
  /// rgb / rgba(opaque) / luma / hsv emit, reused verbatim.
  inner: NativeYuv420,
  /// Straight α plane, binned on the LUMA grid (full-resolution in
  /// `Yuva420p`). Present only when a colour output that carries α (rgba)
  /// is attached — a luma / rgb / hsv-only sink drops α, so it never bins
  /// the α plane (the documented fast path).
  alpha: Option<NativeAlpha>,
}

/// Luma-grid α stream and staging of [`NativeYuva420`].
#[cfg(feature = "yuva")]
struct NativeAlpha {
  a: AreaStream<u8>,
  /// Two-slot staging ring, `out_w` per slot (slot = `out_y & 1`), parallel
  /// to the embedded [`NativeYuv420`]'s Y / U / V slots.
  a_stage: std::vec::Vec<u8>,
  /// `staged[slot]` — whether this slot holds a finalized α output row.
  staged: [bool; 2],
}

#[cfg(feature = "yuva")]
impl NativeYuva420 {
  /// `need_alpha` is true exactly when an α-carrying colour output (rgba) is
  /// attached AND the alpha mode is straight; the caller resolves it. The
  /// embedded [`NativeYuv420`] is built with `need_color` for the Y / U / V
  /// chroma half (rgb / rgba / hsv all need chroma); α rides its own stream.
  fn new(
    plan: &ResamplePlan,
    w: usize,
    h: usize,
    need_color: bool,
    need_alpha: bool,
  ) -> Result<Self, ResampleError> {
    let inner = NativeYuv420::new(plan, w, h, need_color)?;
    let alpha = if need_alpha {
      // Two-slot staging ring (`slot = out_y & 1`), `2 * out_w` like the
      // embedded Y / U / V stages, so an output row may lead its sibling by
      // one slot without clobbering it.
      let stage_len = plan.out_w().checked_mul(2).ok_or_else(|| {
        ResampleError::Overflow(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()))
      })?;
      let alloc =
        |_| ResampleError::AllocationFailed(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()));
      Some(NativeAlpha {
        // α bins on the luma grid (`w x h`), same plan as Y.
        a: AreaStream::new(plan.h(), plan.v(), w, h, 1)?,
        a_stage: try_zeroed(stage_len).map_err(alloc)?,
        staged: [false; 2],
      })
    } else {
      None
    };
    Ok(Self { inner, alpha })
  }

  pub(super) fn reset(&mut self) {
    self.inner.reset();
    if let Some(alpha) = self.alpha.as_mut() {
      alpha.a.reset();
      alpha.staged = [false; 2];
    }
  }
}

/// Native-tier path for the **straight-alpha** [`MixedSinker<Yuva420p, R>`]:
/// see [`NativeYuva420`]. Mirrors [`yuv420p_process_native`] for the
/// Y / U / V → rgb / luma / hsv outputs and the opaque RGBA fan-out, and
/// additionally bins the straight α plane on the luma grid and substitutes
/// it into the RGBA α slot.
///
/// Atomicity (the Phase 2 lesson, applied from the start): the α stream's
/// sequence is checked and its [`AreaStream`] / staging buffers are
/// allocated in the SAME fallible-before-feed phase as the embedded
/// Y / U / V join, so a rejected row (out-of-sequence, mid-frame output
/// change, or an allocation refusal) mutates no caller output and the frame
/// stays retryable. The α feed only runs after every fallible step, in
/// lockstep with the Y feed (same `idx`, same plan).
///
/// `AlphaMode` is frozen at `begin_frame` and checked by the caller (via
/// [`check_frozen_alpha_mode`](super::check_frozen_alpha_mode)) BEFORE this
/// runs, and the sink only routes here under `AlphaMode::Straight`, so a
/// mid-frame flip to Premultiplied is rejected before any native feed.
#[cfg(feature = "yuva")]
#[allow(clippy::too_many_arguments)]
pub(super) fn yuva420p_process_native(
  plan: &ResamplePlan,
  native: &mut Option<std::boxed::Box<NativeYuva420>>,
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
  a_row: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();
  // α is materialised only into the RGBA output (rgb / luma / hsv drop it).
  let need_alpha = rgba.is_some();

  // Complete pre-feed rejection preflight — no-output short-circuit,
  // first-row out-of-sequence rejection, the frozen-output check, AND the
  // post-freeze sequence check, all ahead of any fallible allocation (the
  // same gate the no-alpha 4:2:0 native tier runs). `Ok(false)` is the
  // no-output no-op. The Y-stream expected row is read from the embedded
  // join; the α stream advances in lockstep with Y (same grid), so the Y
  // sequence governs both.
  if !native_preflight_core(
    native.as_ref().map_or(0, |n| n.inner.y.next_y()),
    resample_outputs,
    rgb,
    rgba,
    &None,
    &None,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )? {
    return Ok(());
  }
  // The join's chroma / α halves are fixed at creation; if the frame's
  // colour-or-alpha capability differs (outputs attached since the previous
  // frame — the frozen check pins them WITHIN a frame, not across frames),
  // rebuild it rather than silently skip or needlessly read α / chroma.
  if native
    .as_ref()
    .is_some_and(|n| n.inner.chroma.is_some() != need_color || n.alpha.is_some() != need_alpha)
  {
    *native = None;
  }
  let join = match native {
    Some(join) => join.as_mut(),
    None => {
      // Build the join (its inner Y / U / V / α streams allocate recoverably)
      // and box it recoverably (`try_box`, not `Box::new` — the latter aborts
      // on OOM). Both fallible steps run BEFORE `native.insert`, so a refusal
      // at either returns `Err` with `native` still `None` — the field is left
      // empty, no caller output is touched, and the next row retries the
      // allocation from scratch (the first-row-transactional contract).
      let join = NativeYuva420::new(plan, w, h, need_color, need_alpha)?;
      let boxed = crate::resample::try_box(join).map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )))
      })?;
      native.insert(boxed).as_mut()
    }
  };
  join.inner.check_sequence(idx)?;
  if let Some(alpha) = join.alpha.as_ref()
    && alpha.a.next_y() != idx
  {
    return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
      OutOfSequenceRow::new(alpha.a.next_y(), idx),
    )));
  }
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

  // Feed the planes; everything past this point is infallible. Feed α FIRST
  // (into its staging ring) so its finalized rows are present when the
  // Y / U / V drain emits the matching output rows below.
  if let Some(alpha) = join.alpha.as_mut() {
    let NativeAlpha { a, a_stage, staged } = alpha;
    a.feed_row(idx, a_row, use_simd, |oy, out_row| {
      let slot = oy & 1;
      a_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[slot] = true;
    })?;
  }

  // Feed Y / U / V through the embedded join and emit. The embedded
  // `NativeYuv420` drains rgb / rgba(opaque) / luma / hsv; we re-bind its
  // fields here (rather than calling `yuv420p_process_native`, whose RGBA
  // emit is opaque-only) so the RGBA α slot can be overwritten with the
  // binned α inside the SAME drain — keeping the colour and α outputs
  // finalized together, byte-identical to the no-alpha native tier except
  // for the α channel. `alpha` is destructured alongside `inner` (disjoint
  // fields) so the drain can read the staged α without re-borrowing `join`.
  let NativeYuva420 { inner, alpha } = join;
  let NativeYuv420 {
    y,
    y_stage,
    chroma,
    staged,
    next_emit,
  } = inner;
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

  // Drain every output row whose participating planes are staged. Mirrors
  // the no-alpha `yuv420p_process_native` drain, plus the binned-α
  // substitution into the RGBA α slot. The α slot (when attached) advances
  // in lockstep with Y, so an output row whose Y is staged also has its α
  // staged.
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
        let rgba_row = &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow];
        // Fan the binned RGB to RGBA (opaque α), then overwrite the α slot
        // with the independently-binned straight α — the #235 straight-alpha
        // resolution: RGB from the YUV convert, α from the native α bin.
        expand_rgb_to_rgba_row(out_rgb, rgba_row, ow);
        if let Some(alpha) = alpha.as_ref() {
          let a_out = &alpha.a_stage[slot * ow..slot * ow + ow];
          crate::row::alpha_extract::copy_alpha_plane_u8(a_out, rgba_row, ow, use_simd);
        }
      }
    }
    staged[0][slot] = false;
    staged[1][slot] = false;
    staged[2][slot] = false;
    if let Some(alpha) = alpha.as_mut() {
      alpha.staged[slot] = false;
    }
    *next_emit += 1;
  }
  Ok(())
}

// ---- Native fast tier: 4:2:2 / 4:4:4 / 4:4:0 planar 8-bit ---------------

/// Native decimation join for the non-4:2:0 8-bit planar families
/// (`Yuv422p` / `Yuv444p` / `Yuv440p`) — the sibling of [`NativeYuv420`]
/// for chroma layouts that are NOT half-resolution in both axes. Y streams
/// on the frame grid; U / V stream on the format's chroma grid, every plane
/// binned to FULL output resolution and converted ONCE per output row at
/// output width through the 4:4:4 kernel (the binned chroma is full-width
/// at output resolution, so the convert is always 4:4:4 — identical to the
/// [`NativeYuv420`] finalize).
///
/// The three formats differ only in the chroma grid and its vertical
/// cadence, both captured here:
/// - `Yuv422p` (4:2:2): chroma `w/2 x h` — half width, FULL height. The
///   chroma plane advances one row per Y row (`chroma_vsub == 1`), so Y and
///   chroma stage in lockstep; the chroma plan is a plain
///   [`ResamplePlan::area`] over `(w/2, h)`.
/// - `Yuv444p` (4:4:4): chroma `w x h` — identical to Y. Same lockstep
///   cadence (`chroma_vsub == 1`); the chroma plan equals the luma plan
///   ([`ResamplePlan::area`] over `(w, h)`).
/// - `Yuv440p` (4:4:0): chroma `w x h/2` — FULL width, half height. The
///   chroma plane advances one row per TWO Y rows (`chroma_vsub == 2`,
///   exactly like 4:2:0 vertically); the chroma plan is
///   [`ResamplePlan::area_chroma_440`] (full-width horizontal,
///   luma-domain `area_halved` vertical so an odd trailing luma row weights
///   its chroma row by half).
///
/// Each plane's in-order emissions stage into a two-slot ring
/// (`out_y & 1`); the moment all participating planes hold an output row it
/// finalizes — so no output-geometry alignment constraint ever applies. A
/// plane may lead another by at most one source row (the chroma grid is
/// within a factor of two of the luma grid vertically), which the two slots
/// absorb. For the lockstep formats (`chroma_vsub == 1`) the planes never
/// lead at all, but the two-slot machinery is harmless and shared.
pub(super) struct NativePlanarYuv {
  y: AreaStream<u8>,
  /// Two-slot staging ring, `2 * out_w` (slot = `out_y & 1`).
  y_stage: std::vec::Vec<u8>,
  /// Chroma half of the join — absent for luma-only sinks, which therefore
  /// never read the chroma planes (the documented fast path). Decided at
  /// creation: the frozen-output contract makes the attached set
  /// frame-constant.
  chroma: Option<NativePlanarChroma>,
  /// Vertical chroma subsample factor: 1 for 4:2:2 / 4:4:4 (a chroma row
  /// per luma row), 2 for 4:4:0 (a chroma row per two luma rows). The
  /// chroma stream is fed when `idx % chroma_vsub == 0`, at
  /// `cidx = idx / chroma_vsub`, and the sequence check expects
  /// `chroma.next_y() == idx.div_ceil(chroma_vsub)`.
  chroma_vsub: usize,
  /// `staged[plane][slot]` — plane 0 = Y, 1 = U, 2 = V.
  staged: [[bool; 2]; 3],
  /// Next output row to finalize.
  next_emit: usize,
}

/// Chroma-grid streams and staging of [`NativePlanarYuv`].
struct NativePlanarChroma {
  u: AreaStream<u8>,
  v: AreaStream<u8>,
  u_stage: std::vec::Vec<u8>,
  v_stage: std::vec::Vec<u8>,
}

#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
std::thread_local! {
  static FORCE_PLANAR_NATIVE_CHROMA_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms a failpoint that fires when (and only when) the non-4:2:0 planar
/// native join PLANS its chroma grid — which happens exactly when colour
/// output is requested. A luma-only sink must never reach it, so an armed
/// flag survives a luma-only row unconsumed (the regression assertion) and is
/// taken by the first colour row. Test-only. Mirrors the high-bit
/// `arm_planar_hb_native_chroma_failure`.
///
/// Only the non-4:2:0 native suites that own a colour oracle arm it, and each
/// reuses the planar join so each also pulls in `yuv-planar`: the packed-8-bit /
/// packed-4:4:4 / 4:1:1 natives, plus the semi-planar 8-bit suite's
/// `yuv-planar`-gated `native_tier` (its colour oracle adds `rgb`). So the
/// setter is dead in a `yuv-planar`-without-any-consumer build even though the
/// thread-local it sets is read by the (always-`yuv-planar`) native join.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-planar",
  any(
    feature = "yuv-packed",
    feature = "yuv-444-packed",
    all(feature = "yuv-semi-planar", feature = "rgb")
  )
))]
pub(crate) fn arm_planar_native_chroma_failure() {
  FORCE_PLANAR_NATIVE_CHROMA_FAILURE.with(|f| f.set(true));
}

impl NativePlanarYuv {
  /// `build_chroma_plan` lazily builds the format's chroma grid against the
  /// SAME output geometry as `plan` (the luma plan) — invoked ONLY when colour
  /// is needed, so a luma-only sink never plans or allocates chroma state.
  /// `chroma_vsub` is its vertical cadence. Both are supplied by the
  /// per-format caller so this body stays layout-agnostic.
  fn new(
    plan: &ResamplePlan,
    build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
    chroma_vsub: usize,
    w: usize,
    h: usize,
    need_color: bool,
  ) -> Result<Self, ResampleError> {
    // The native planar join is integer area-only; reject a filter plan
    // before building any plane's area stream.
    if plan.kind().is_filter() {
      return Err(plan.unsupported_filter());
    }
    let y = AreaStream::new(plan.h(), plan.v(), w, h, 1)?;
    let alloc =
      |_| ResampleError::AllocationFailed(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()));
    let stage_len = plan.out_w().checked_mul(2).ok_or_else(|| {
      ResampleError::Overflow(PlanGeometry::new(w, h, plan.out_w(), plan.out_h()))
    })?;
    let chroma = if need_color {
      #[cfg(all(test, feature = "std", feature = "yuv-planar"))]
      if FORCE_PLANAR_NATIVE_CHROMA_FAILURE.with(|f| f.take()) {
        return Err(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )));
      }
      let chroma_plan = build_chroma_plan()?;
      Some(NativePlanarChroma {
        u: AreaStream::new(
          chroma_plan.h(),
          chroma_plan.v(),
          chroma_plan.src_w(),
          chroma_plan.src_h(),
          1,
        )?,
        v: AreaStream::new(
          chroma_plan.h(),
          chroma_plan.v(),
          chroma_plan.src_w(),
          chroma_plan.src_h(),
          1,
        )?,
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
      chroma_vsub,
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

  /// Sequencing preflight across all three plane streams — checked before
  /// any plane is fed so a violating call mutates nothing. Chroma rows
  /// advance once per `chroma_vsub` source rows, so their expected counter
  /// is `idx.div_ceil(chroma_vsub)`.
  fn check_sequence(&self, idx: usize) -> Result<(), MixedSinkerError> {
    if self.y.next_y() != idx {
      return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
        OutOfSequenceRow::new(self.y.next_y(), idx),
      )));
    }
    if let Some(chroma) = self.chroma.as_ref() {
      let chroma_expected = idx.div_ceil(self.chroma_vsub);
      for stream in [&chroma.u, &chroma.v] {
        if stream.next_y() != chroma_expected {
          return Err(MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(
            OutOfSequenceRow::new(stream.next_y().saturating_mul(self.chroma_vsub), idx),
          )));
        }
      }
    }
    Ok(())
  }
}

/// Thin preflight wrapper over [`native_preflight_core`] for the
/// [`NativePlanarYuv`] join — supplies the join-typed expected row and
/// freezes the 8-bit-absent native-depth u16 colour outputs. Mirrors
/// [`yuv420p_native_preflight`]; see `native_preflight_core` for the
/// 4-point rejection logic and its ordering contract.
#[allow(clippy::too_many_arguments)]
pub(super) fn native_planar_preflight(
  join: &Option<std::boxed::Box<NativePlanarYuv>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &Option<&mut [u8]>,
  rgba: &Option<&mut [u8]>,
  luma: &Option<&mut [u8]>,
  luma_u16: &Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  idx: usize,
  need_luma: bool,
  need_color: bool,
) -> Result<bool, MixedSinkerError> {
  native_preflight_core(
    join.as_ref().map_or(0, |join| join.y.next_y()),
    resample_outputs,
    rgb,
    rgba,
    &None,
    &None,
    luma,
    luma_u16,
    hsv,
    idx,
    need_luma,
    need_color,
  )
}

/// Native-tier path for the non-4:2:0 8-bit planar families
/// (`Yuv422p` / `Yuv444p` / `Yuv440p`): see [`NativePlanarYuv`]. Bins the
/// Y / U / V planes to output resolution and converts ONCE per output row
/// at output width through the 4:4:4 kernel — vs the row-stage tier
/// ([`planar_dual_resample`](super::planar_resample::planar_dual_resample)),
/// which converts each source row at source width then bins. Phasing
/// mirrors [`yuv420p_process_native`]: the complete pre-feed preflight, the
/// join build, sequencing, colour scratch sizing, then the feeds — with
/// nothing fallible after the first feed.
///
/// `chroma_vsub` is the format's vertical chroma cadence (1 for 4:2:2 /
/// 4:4:4, 2 for 4:4:0) and `build_chroma_plan` builds its chroma grid
/// against the same output geometry; both are supplied by the per-format
/// caller so this body is layout-agnostic.
#[allow(clippy::too_many_arguments)]
pub(super) fn yuv_planar_process_native(
  plan: &ResamplePlan,
  native: &mut Option<std::boxed::Box<NativePlanarYuv>>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  u_row: &[u8],
  v_row: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  chroma_vsub: usize,
  build_chroma_plan: impl FnOnce() -> Result<ResamplePlan, ResampleError>,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let need_luma = luma.is_some() || luma_u16.is_some();
  let need_color = rgb.is_some() || hsv.is_some() || rgba.is_some();
  // HSV-without-RGB-or-RGBA emits HSV straight from the binned
  // output-width Y/U/V (no output-width RGB scratch); chroma is still
  // decimated (`need_color`). See [`yuv420p_process_native`].
  let want_hsv_direct = hsv.is_some() && rgb.is_none() && rgba.is_none();

  // Complete pre-feed rejection preflight ahead of any fallible allocation
  // (no-output short-circuit, first-row out-of-sequence, frozen-output,
  // post-freeze sequence) — see [`yuv420p_native_preflight`]. `Ok(false)`
  // is the no-output no-op.
  if !native_planar_preflight(
    native,
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
  // The join's chroma half is fixed at creation; if the frame's colour
  // capability differs (outputs attached since the previous frame — the
  // frozen check pins them WITHIN a frame, not across frames), rebuild it.
  if native
    .as_ref()
    .is_some_and(|join| join.chroma.is_some() != need_color)
  {
    *native = None;
  }
  let join = match native {
    Some(join) => join,
    None => {
      // Build the join (its de-interleave scratch allocates recoverably) and box
      // it recoverably (`try_box`, not `Box::new` — the latter aborts on OOM).
      // Both fallible steps run BEFORE `native.insert`, so a refusal at either
      // returns `Err` with the field still `None` — no caller output is touched
      // and the next row retries (the first-row-transactional contract).
      let join = NativePlanarYuv::new(plan, build_chroma_plan, chroma_vsub, w, h, need_color)?;
      let boxed = crate::resample::try_box(join).map_err(|_| {
        MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
          w,
          h,
          plan.out_w(),
          plan.out_h(),
        )))
      })?;
      native.insert(boxed)
    }
  };
  join.check_sequence(idx)?;
  // No output-width RGB scratch for the HSV-direct case — the emit loop
  // converts the binned Y/U/V straight to HSV.
  if need_color && !want_hsv_direct {
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
  let NativePlanarYuv {
    y,
    y_stage,
    chroma,
    chroma_vsub,
    staged,
    next_emit,
  } = &mut **join;
  y.feed_row(idx, y_row, use_simd, |oy, out_row| {
    let slot = oy & 1;
    y_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
    staged[0][slot] = true;
  })?;
  if let Some(c) = chroma.as_mut()
    && idx.is_multiple_of(*chroma_vsub)
  {
    let cidx = idx / *chroma_vsub;
    let NativePlanarChroma {
      u,
      v,
      u_stage,
      v_stage,
    } = c;
    u.feed_row(cidx, u_row, use_simd, |oy, out_row| {
      let slot = oy & 1;
      u_stage[slot * ow..slot * ow + ow].copy_from_slice(out_row);
      staged[1][slot] = true;
    })?;
    v.feed_row(cidx, v_row, use_simd, |oy, out_row| {
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
      let u_out = &c.u_stage[slot * ow..slot * ow + ow];
      let v_out = &c.v_stage[slot * ow..slot * ow + ow];
      if want_hsv_direct {
        // RGB-free: convert the binned output-width Y/U/V straight to HSV.
        let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
        let (hp, sp, vp) = hsv.hsv();
        yuv_444_to_hsv_row(
          y_out,
          u_out,
          v_out,
          &mut hp[oy * ow..(oy + 1) * ow],
          &mut sp[oy * ow..(oy + 1) * ow],
          &mut vp[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      } else {
        let out_rgb = &mut rgb_scratch[..ow * 3];
        yuv_444_to_rgb_row(
          y_out, u_out, v_out, out_rgb, ow, matrix, full_range, use_simd,
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
    }
    staged[0][slot] = false;
    staged[1][slot] = false;
    staged[2][slot] = false;
    *next_emit += 1;
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
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
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
      rgb_filter_stream,
      luma_filter_stream,
      hsv_planar,
      resample_outputs,
      ..
    } = self;

    // Non-identity plan: convert the source row to canonical RGB at
    // source width in the shared scratch (the same `yuv_410_to_rgb_row`
    // kernel the identity path uses — 4:1:0's 1→4 chroma upsample), then
    // feed the shared planar resample tail. Row-stage only — converting
    // each source row to RGB and binning is the whole job. The `Area` arm
    // bins the converted RGB / native Y; the `Filter` arm filter-resamples
    // them through the merged engine (the same convert-then-resample tail).
    // Both freeze the output set and check stream sequencing before
    // staging, so a no-output sink stays a no-op and an out-of-sequence row
    // is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
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
      };
      // HSV-only (no RGB / RGBA) area plan: bin Y / U / V and convert per
      // output row, RGB-free. 4:1:0 chroma is quarter-width AND
      // quarter-height (`chroma_vsub = 4`). Bit-identical to a YUV-domain
      // bin-then-convert reference. A filter plan keeps the RGB-staged HSV
      // (filter twin), and RGB / RGBA attached keeps the RGB-staged path.
      if plan.kind().is_area() && hsv.is_some() && rgb.is_none() && rgba.is_none() {
        return super::planar_resample::hsv_direct_resample(
          hsv_planar,
          resample_outputs,
          luma,
          luma_u16,
          hsv,
          row.y(),
          row.u_quarter(),
          row.v_quarter(),
          matrix,
          full_range,
          4,
          || ResamplePlan::area_chroma_410(w / 4, h, plan.out_w(), plan.out_h()),
          w,
          plan,
          idx,
          use_simd,
        );
      }
      return match plan.kind() {
        crate::resample::SpanKind::Area => planar_dual_resample(
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
          convert_rgb,
        ),
        crate::resample::SpanKind::Filter => planar_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
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
          convert_rgb,
        ),
      };
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
    // HSV-only (no RGB / RGBA) goes direct through `yuv_410_to_hsv_row`
    // — see the Yuv420p impl for the routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv_410_to_hsv_row(
        row.y(),
        row.u_quarter(),
        row.v_quarter(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

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
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    self.frozen_native_route = None;
    self.frozen_domain = None;
    self.resample_outputs = None;
    // New frame: drop the RFC #238 linear-light accumulator (if any).
    #[cfg(feature = "rgb")]
    {
      self.linear_light_frame = None;
    }
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
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      native,
      native_planar,
      hsv_planar,
      frozen_native_route,
      frozen_domain,
      averaging_domain,
      #[cfg(feature = "rgb")]
      linear_mode,
      #[cfg(feature = "rgb")]
      linear_light_frame,
      #[cfg(feature = "rgb")]
      linear_scene_scratch,
      #[cfg(feature = "rgb")]
      transfer_function,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan converts each source row to a
    // source-width RGB row (the 4:2:0 per-row dispatcher 4:2:2 reuses) and
    // filter-resamples it plus the native Y. An `Area` plan picks the
    // native fast tier (bin Y / U / V to output res, convert once at output
    // width) or the row-stage tier (convert each source row, bin RGB) per
    // `with_native`, frozen per frame. Both tiers freeze the output set and
    // check stream sequencing before staging, so a no-output sink stays a
    // no-op and an out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
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
      };
      // The scene-referred ([`LinearMode::SceneReferred`]) twin of
      // `convert_rgb`: the SAME affine matrix, real-valued and unclamped (4:2:2
      // reuses the 4:2:0 horizontal chroma upsample, as `convert_rgb` reuses
      // `yuv_420_to_rgb_row`). Consumed only by the `rgb`-gated Linear arm.
      #[cfg(feature = "rgb")]
      let convert_rgb_unclamped = |dst: &mut [f32]| {
        crate::row::scalar::yuv_420_to_rgb_f32_unclamped_row(
          row.y(),
          row.u_half(),
          row.v_half(),
          dst,
          w,
          matrix,
          full_range,
        );
      };
      // RFC #238 Phase 2 — single always-compiled choke point for the averaging
      // domain, BEFORE any filter / native / row-stage branching, so a
      // non-encoded sink never falls through to the Encoded path under any
      // feature combination. The match is EXHAUSTIVE with no wildcard arm: a
      // future `AveragingDomain` variant fails to compile until handled here.
      // The `Encoded` arm is empty (control continues into the encoded dispatch
      // below); `Linear` and `Premultiplied` return. See the Yuv420p impl.
      //
      // `need_output` gates BOTH the averaging-domain freeze here and the
      // native/row-stage route freeze below. The domain freeze is CHECK-ONLY
      // here — the matching SET happens AFTER the selected path accepts an
      // output-bearing row (mirroring `frozen_native_route` below), never
      // before dispatch, so a rejected row leaves the freeze unchanged and a
      // corrected-domain retry of the SAME row is not falsely rejected. See the
      // Yuv420p impl for the full CHECK-before / SET-after rationale.
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      if need_output
        && let Some(frozen) = *frozen_domain
        && frozen != *averaging_domain
      {
        return Err(MixedSinkerError::AveragingDomainChanged(
          AveragingDomainChanged::new(idx),
        ));
      }
      match *averaging_domain {
        AveragingDomain::Encoded => {}
        // Under `rgb` it runs the linear tail (which itself rejects a filter
        // plan); without `rgb` it returns the typed `LinearDomainUnsupported`.
        AveragingDomain::Linear => {
          #[cfg(feature = "rgb")]
          {
            let tf = transfer_function.unwrap_or_else(|| TransferFunction::for_matrix(matrix));
            // Dispatch first; commit the domain freeze to Linear ONLY when the
            // tail accepts an output-bearing row (`r.is_ok() && need_output`).
            // A no-output call returns Ok without consuming; a filter /
            // out-of-sequence / output-changed / alloc reject returns Err — so
            // a rejected row leaves `frozen_domain` unset for a corrected retry.
            // See the Yuv420p impl.
            let r = linear_light::linear_light_resample(
              linear_light_frame,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              linear_scene_scratch,
              tf,
              *linear_mode,
              plan,
              row.y(),
              idx,
              w,
              h,
              use_simd,
              |_idx, dst| convert_rgb(dst),
              |_idx, dst| convert_rgb_unclamped(dst),
            );
            if r.is_ok() && need_output && frozen_domain.is_none() {
              *frozen_domain = Some(AveragingDomain::Linear);
            }
            return r;
          }
          #[cfg(not(feature = "rgb"))]
          {
            return Err(plan.linear_domain_unsupported().into());
          }
        }
        // This format has no alpha plane, so premultiplied weighting is a
        // category error — reject with a typed error rather than silently
        // running the encoded average. Phase 5 wires Premultiplied for alpha
        // formats; on these non-alpha formats rejecting is correct.
        AveragingDomain::Premultiplied => {
          return Err(plan.premultiplied_domain_unsupported().into());
        }
      }
      if plan.kind().is_filter() {
        return planar_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
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
          convert_rgb,
        );
      }
      // Native / row-stage route split — see the Yuv420p impl for the
      // CHECK-before / SET-after `frozen_native_route` contract. Reuses the
      // `need_output` computed for the domain freeze above.
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV_PLANAR_8BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:2:2: chroma `w/2 x h` — half width, full height; a chroma row
          // per Y row (`chroma_vsub = 1`), chroma plan a plain `area`.
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
            row.y(),
            row.u_half(),
            row.v_half(),
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            || ResamplePlan::area(w / 2, h, plan.out_w(), plan.out_h()),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output
          // row, RGB-free — bit-identical to the native fast tier. 4:2:2
          // chroma is half-width, full height (`chroma_vsub = 1`); the
          // chroma plan matches the native tier's. Still the ROW-STAGE route
          // (`frozen_native_route = false`).
          if hsv.is_some() && rgb.is_none() && rgba.is_none() {
            super::planar_resample::hsv_direct_resample(
              hsv_planar,
              resample_outputs,
              luma,
              luma_u16,
              hsv,
              row.y(),
              row.u_half(),
              row.v_half(),
              matrix,
              full_range,
              1,
              || ResamplePlan::area(w / 2, h, plan.out_w(), plan.out_h()),
              w,
              plan,
              idx,
              use_simd,
            )?;
          } else {
            planar_dual_resample(
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
              convert_rgb,
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
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
    // HSV-only (no RGB / RGBA) goes direct through `yuv_420_to_hsv_row`
    // — see the Yuv420p impl for the routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv_420_to_hsv_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

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
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    self.frozen_native_route = None;
    self.frozen_domain = None;
    self.resample_outputs = None;
    // New frame: drop the RFC #238 linear-light accumulator (if any).
    #[cfg(feature = "rgb")]
    {
      self.linear_light_frame = None;
    }
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
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      native,
      native_planar,
      hsv_planar,
      frozen_native_route,
      frozen_domain,
      averaging_domain,
      #[cfg(feature = "rgb")]
      linear_mode,
      #[cfg(feature = "rgb")]
      linear_light_frame,
      #[cfg(feature = "rgb")]
      linear_scene_scratch,
      #[cfg(feature = "rgb")]
      transfer_function,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan converts each source row to a
    // source-width RGB row (the same 4:4:4 kernel the identity path uses)
    // and filter-resamples it plus the native Y. An `Area` plan picks the
    // native fast tier (bin Y / U / V to output res, convert once at output
    // width) or the row-stage tier (convert each source row, bin RGB) per
    // `with_native`, frozen per frame. Both tiers freeze the output set and
    // check stream sequencing before staging, so a no-output sink stays a
    // no-op and an out-of-sequence row is rejected without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
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
      };
      // The scene-referred ([`LinearMode::SceneReferred`]) twin of
      // `convert_rgb`: the SAME affine matrix, real-valued and unclamped.
      // Consumed only by the `rgb`-gated Linear arm.
      #[cfg(feature = "rgb")]
      let convert_rgb_unclamped = |dst: &mut [f32]| {
        crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
          row.y(),
          row.u(),
          row.v(),
          dst,
          w,
          matrix,
          full_range,
        );
      };
      // RFC #238 Phase 2 — single always-compiled choke point for the averaging
      // domain, BEFORE any filter / native / row-stage branching, so a
      // non-encoded sink never falls through to the Encoded path under any
      // feature combination. The match is EXHAUSTIVE with no wildcard arm: a
      // future `AveragingDomain` variant fails to compile until handled here.
      // The `Encoded` arm is empty (control continues into the encoded dispatch
      // below); `Linear` and `Premultiplied` return. See the Yuv420p impl.
      //
      // `need_output` gates BOTH the averaging-domain freeze here and the
      // native/row-stage route freeze below. The domain freeze is CHECK-ONLY
      // here — the matching SET happens AFTER the selected path accepts an
      // output-bearing row (mirroring `frozen_native_route` below), never
      // before dispatch, so a rejected row leaves the freeze unchanged and a
      // corrected-domain retry of the SAME row is not falsely rejected. See the
      // Yuv420p impl for the full CHECK-before / SET-after rationale.
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      if need_output
        && let Some(frozen) = *frozen_domain
        && frozen != *averaging_domain
      {
        return Err(MixedSinkerError::AveragingDomainChanged(
          AveragingDomainChanged::new(idx),
        ));
      }
      match *averaging_domain {
        AveragingDomain::Encoded => {}
        // Under `rgb` it runs the linear tail (which itself rejects a filter
        // plan); without `rgb` it returns the typed `LinearDomainUnsupported`.
        AveragingDomain::Linear => {
          #[cfg(feature = "rgb")]
          {
            let tf = transfer_function.unwrap_or_else(|| TransferFunction::for_matrix(matrix));
            // Dispatch first; commit the domain freeze to Linear ONLY when the
            // tail accepts an output-bearing row (`r.is_ok() && need_output`).
            // A no-output call returns Ok without consuming; a filter /
            // out-of-sequence / output-changed / alloc reject returns Err — so
            // a rejected row leaves `frozen_domain` unset for a corrected retry.
            // See the Yuv420p impl.
            let r = linear_light::linear_light_resample(
              linear_light_frame,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              linear_scene_scratch,
              tf,
              *linear_mode,
              plan,
              row.y(),
              idx,
              w,
              h,
              use_simd,
              |_idx, dst| convert_rgb(dst),
              |_idx, dst| convert_rgb_unclamped(dst),
            );
            if r.is_ok() && need_output && frozen_domain.is_none() {
              *frozen_domain = Some(AveragingDomain::Linear);
            }
            return r;
          }
          #[cfg(not(feature = "rgb"))]
          {
            return Err(plan.linear_domain_unsupported().into());
          }
        }
        // This format has no alpha plane, so premultiplied weighting is a
        // category error — reject with a typed error rather than silently
        // running the encoded average. Phase 5 wires Premultiplied for alpha
        // formats; on these non-alpha formats rejecting is correct.
        AveragingDomain::Premultiplied => {
          return Err(plan.premultiplied_domain_unsupported().into());
        }
      }
      if plan.kind().is_filter() {
        return planar_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
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
          convert_rgb,
        );
      }
      // Native / row-stage route split — see the Yuv420p impl for the
      // CHECK-before / SET-after `frozen_native_route` contract. Reuses the
      // `need_output` computed for the domain freeze above.
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV_PLANAR_8BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:4: chroma `w x h` — identical to Y; a chroma row per Y row
          // (`chroma_vsub = 1`), chroma plan equals the luma plan.
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
            row.y(),
            row.u(),
            row.v(),
            matrix,
            full_range,
            idx,
            w,
            h,
            1,
            || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output
          // row, RGB-free — bit-identical to the native fast tier. 4:4:4
          // chroma is full-width, full height (`chroma_vsub = 1`); the chroma
          // plan matches the native tier's. Still the ROW-STAGE route.
          if hsv.is_some() && rgb.is_none() && rgba.is_none() {
            super::planar_resample::hsv_direct_resample(
              hsv_planar,
              resample_outputs,
              luma,
              luma_u16,
              hsv,
              row.y(),
              row.u(),
              row.v(),
              matrix,
              full_range,
              1,
              || ResamplePlan::area(w, h, plan.out_w(), plan.out_h()),
              w,
              plan,
              idx,
              use_simd,
            )?;
          } else {
            planar_dual_resample(
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
              convert_rgb,
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
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
    // HSV-only (no RGB / RGBA) goes direct through `yuv_444_to_hsv_row`
    // — see the Yuv420p impl for the routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv_444_to_hsv_row(
        row.y(),
        row.u(),
        row.v(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

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
    if let Some(stream) = self.rgb_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(native) = self.native_planar.as_mut() {
      native.reset();
    }
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
    }
    self.frozen_native_route = None;
    self.frozen_domain = None;
    self.resample_outputs = None;
    // New frame: drop the RFC #238 linear-light accumulator (if any).
    #[cfg(feature = "rgb")]
    {
      self.linear_light_frame = None;
    }
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
      rgb_filter_stream,
      luma_filter_stream,
      resample_outputs,
      native,
      native_planar,
      hsv_planar,
      frozen_native_route,
      frozen_domain,
      averaging_domain,
      #[cfg(feature = "rgb")]
      linear_mode,
      #[cfg(feature = "rgb")]
      linear_light_frame,
      #[cfg(feature = "rgb")]
      linear_scene_scratch,
      #[cfg(feature = "rgb")]
      transfer_function,
      ..
    } = self;

    // Non-identity plan. A `Filter` plan converts each source row to a
    // source-width RGB row (the same `yuv_444_to_rgb_row` kernel the
    // identity path uses — 4:4:0's per-row math is identical to 4:4:4,
    // full-width chroma) and filter-resamples it plus the native Y. An
    // `Area` plan picks the native fast tier (bin Y / U / V to output res,
    // convert once at output width) or the row-stage tier (convert each
    // source row, bin RGB) per `with_native`, frozen per frame. Both tiers
    // freeze the output set and check stream sequencing before staging, so
    // a no-output sink stays a no-op and an out-of-sequence row is rejected
    // without allocating.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      let convert_rgb = |scratch: &mut [u8]| {
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
      };
      // The scene-referred ([`LinearMode::SceneReferred`]) twin of
      // `convert_rgb`: the SAME affine matrix, real-valued and unclamped.
      // Consumed only by the `rgb`-gated Linear arm.
      #[cfg(feature = "rgb")]
      let convert_rgb_unclamped = |dst: &mut [f32]| {
        crate::row::scalar::yuv_444_to_rgb_f32_unclamped_row(
          row.y(),
          row.u(),
          row.v(),
          dst,
          w,
          matrix,
          full_range,
        );
      };
      // RFC #238 Phase 2 — single always-compiled choke point for the averaging
      // domain, BEFORE any filter / native / row-stage branching, so a
      // non-encoded sink never falls through to the Encoded path under any
      // feature combination. The match is EXHAUSTIVE with no wildcard arm: a
      // future `AveragingDomain` variant fails to compile until handled here.
      // The `Encoded` arm is empty (control continues into the encoded dispatch
      // below); `Linear` and `Premultiplied` return. See the Yuv420p impl.
      //
      // `need_output` gates BOTH the averaging-domain freeze here and the
      // native/row-stage route freeze below. The domain freeze is CHECK-ONLY
      // here — the matching SET happens AFTER the selected path accepts an
      // output-bearing row (mirroring `frozen_native_route` below), never
      // before dispatch, so a rejected row leaves the freeze unchanged and a
      // corrected-domain retry of the SAME row is not falsely rejected. See the
      // Yuv420p impl for the full CHECK-before / SET-after rationale.
      let need_output =
        luma.is_some() || luma_u16.is_some() || rgb.is_some() || rgba.is_some() || hsv.is_some();
      if need_output
        && let Some(frozen) = *frozen_domain
        && frozen != *averaging_domain
      {
        return Err(MixedSinkerError::AveragingDomainChanged(
          AveragingDomainChanged::new(idx),
        ));
      }
      match *averaging_domain {
        AveragingDomain::Encoded => {}
        // Under `rgb` it runs the linear tail (which itself rejects a filter
        // plan); without `rgb` it returns the typed `LinearDomainUnsupported`.
        AveragingDomain::Linear => {
          #[cfg(feature = "rgb")]
          {
            let tf = transfer_function.unwrap_or_else(|| TransferFunction::for_matrix(matrix));
            // Dispatch first; commit the domain freeze to Linear ONLY when the
            // tail accepts an output-bearing row (`r.is_ok() && need_output`).
            // A no-output call returns Ok without consuming; a filter /
            // out-of-sequence / output-changed / alloc reject returns Err — so
            // a rejected row leaves `frozen_domain` unset for a corrected retry.
            // See the Yuv420p impl.
            let r = linear_light::linear_light_resample(
              linear_light_frame,
              resample_outputs,
              rgb,
              rgba,
              luma,
              luma_u16,
              hsv,
              rgb_scratch,
              linear_scene_scratch,
              tf,
              *linear_mode,
              plan,
              row.y(),
              idx,
              w,
              h,
              use_simd,
              |_idx, dst| convert_rgb(dst),
              |_idx, dst| convert_rgb_unclamped(dst),
            );
            if r.is_ok() && need_output && frozen_domain.is_none() {
              *frozen_domain = Some(AveragingDomain::Linear);
            }
            return r;
          }
          #[cfg(not(feature = "rgb"))]
          {
            return Err(plan.linear_domain_unsupported().into());
          }
        }
        // This format has no alpha plane, so premultiplied weighting is a
        // category error — reject with a typed error rather than silently
        // running the encoded average. Phase 5 wires Premultiplied for alpha
        // formats; on these non-alpha formats rejecting is correct.
        AveragingDomain::Premultiplied => {
          return Err(plan.premultiplied_domain_unsupported().into());
        }
      }
      if plan.kind().is_filter() {
        return planar_dual_filter_resample(
          luma_filter_stream,
          rgb_filter_stream,
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
          convert_rgb,
        );
      }
      // Native / row-stage route split — see the Yuv420p impl for the
      // CHECK-before / SET-after `frozen_native_route` contract. Reuses the
      // `need_output` computed for the domain freeze above.
      if need_output
        && let Some(frozen) = *frozen_native_route
        && frozen != *native
      {
        return Err(MixedSinkerError::NativeRouteChanged(
          NativeRouteChanged::new(idx),
        ));
      }
      // RFC #238 splice-stage selection — see the Yuv420p impl for the
      // selector contract; reproduces the former `if *native` boolean
      // bit-for-bit (a filter plan already returned above, so `area_plan` is
      // always true here).
      let insertion = select_insertion_point(
        AveragingDomain::Encoded,
        InsertionContext {
          native_eligible: YUV_PLANAR_8BIT_NATIVE_ELIGIBLE,
          with_native: *native,
          area_plan: true,
        },
      );
      match insertion {
        InsertionPoint::NativeCodes => {
          // 4:4:0: chroma `w x h/2` — full width, half height; a chroma row
          // per TWO Y rows (`chroma_vsub = 2`, like 4:2:0 vertically), chroma
          // plan full-width horizontal + luma-domain `area_halved` vertical.
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
            row.y(),
            row.u(),
            row.v(),
            matrix,
            full_range,
            idx,
            w,
            h,
            2,
            || ResamplePlan::area_chroma_440(w, h, plan.out_w(), plan.out_h()),
            use_simd,
          )?;
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(true);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          return Ok(());
        }
        InsertionPoint::EncodedOutput => {
          // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output
          // row, RGB-free — bit-identical to the native fast tier. 4:4:0
          // chroma is full-width, half height (`chroma_vsub = 2`); the chroma
          // plan matches the native tier's `area_chroma_440` (luma-domain
          // vertical weighting). Still the ROW-STAGE route.
          if hsv.is_some() && rgb.is_none() && rgba.is_none() {
            super::planar_resample::hsv_direct_resample(
              hsv_planar,
              resample_outputs,
              luma,
              luma_u16,
              hsv,
              row.y(),
              row.u(),
              row.v(),
              matrix,
              full_range,
              2,
              || ResamplePlan::area_chroma_440(w, h, plan.out_w(), plan.out_h()),
              w,
              plan,
              idx,
              use_simd,
            )?;
          } else {
            planar_dual_resample(
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
              convert_rgb,
            )?;
          }
          if frozen_native_route.is_none() && need_output {
            *frozen_native_route = Some(false);
          }
          // Encoded domain committed alongside the route, on the same accepted
          // output-bearing row (the `?` above already returned any reject).
          if frozen_domain.is_none() && need_output {
            *frozen_domain = Some(AveragingDomain::Encoded);
          }
          return Ok(());
        }
        // The encoded domain only resolves to the native-codes or
        // encoded-output splice; the linear-light splice is reached via the
        // sink's Linear averaging domain, dispatched before this match.
        InsertionPoint::LinearLight => {
          unreachable!("encoded domain never selects the linear-light splice")
        }
      }
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
    // HSV-only (no RGB / RGBA) goes direct through `yuv_444_to_hsv_row`
    // (4:4:0 reuses the 4:4:4 kernel) — see the Yuv420p impl for the
    // routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv_444_to_hsv_row(
        row.y(),
        row.u(),
        row.v(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

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
    // New frame: restart the RGB-free HSV-only row-stage join (#263 follow-up).
    if let Some(hsv) = self.hsv_planar.as_mut() {
      hsv.reset();
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
      hsv_planar,
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
      // HSV-only (no RGB / RGBA): bin Y / U / V and convert per output row,
      // RGB-free. 4:1:1 chroma is quarter-width (`chroma_w = ceil(w/4)`),
      // full height (`chroma_vsub = 1`). Bit-identical to a YUV-domain
      // bin-then-convert reference.
      if plan.kind().is_area() && hsv.is_some() && rgb.is_none() && rgba.is_none() {
        return super::planar_resample::hsv_direct_resample(
          hsv_planar,
          resample_outputs,
          luma,
          luma_u16,
          hsv,
          row.y(),
          row.u_quarter(),
          row.v_quarter(),
          matrix,
          full_range,
          1,
          || ResamplePlan::area_chroma_411(w, h, plan.out_w(), plan.out_h()),
          w,
          plan,
          idx,
          use_simd,
        );
      }
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
    // HSV-only (no RGB / RGBA) goes direct through `yuv_411_to_hsv_row`
    // — see the Yuv420p impl for the routing rationale.
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    if want_hsv_direct {
      let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
      let (h, s, v) = hsv.hsv();
      yuv_411_to_hsv_row(
        row.y(),
        row.u_quarter(),
        row.v_quarter(),
        &mut h[one_plane_start..one_plane_end],
        &mut s[one_plane_start..one_plane_end],
        &mut v[one_plane_start..one_plane_end],
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

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
