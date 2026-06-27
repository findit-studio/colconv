//! Sinker impls for source-side YUVA 4:2:0 formats — Yuva420p (8-bit),
//! Yuva420p9, Yuva420p10, Yuva420p16.
//!
//! Tranche 8b‑2a wires:
//! - **RGBA output paths (u8 + native-depth u16)** — alpha is sourced
//!   from the source alpha plane via the new
//!   `yuv_420*_to_rgba*_with_alpha_src_row` scalar kernels. SIMD lands
//!   in 8b‑2b (u8) / 8b‑2c (u16). The 8-bit `Yuva420p` has no native
//!   `u16` RGBA output.
//! - **RGB / RGB-u16 / luma / HSV alpha-drop paths** — these reuse the
//!   existing non-alpha 4:2:0 row dispatchers verbatim. The alpha plane
//!   is simply ignored; output bytes/elements match what the
//!   corresponding `Yuv420p*` source would produce given the same
//!   Y/U/V data. Without these paths `MixedSinker::with_rgb` /
//!   `with_luma` / `with_hsv` (declared on the generic impl) would
//!   silently accept a buffer and never write it.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, NativeRouteChanged,
  RowIndexOutOfRange, RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  check_frozen_alpha_mode, deinterleave_y_high_bit_masked, packed_yuva444_filter_resample,
  packed_yuva444_resample, planar_8bit::yuva420p_process_native, reset_high_bit_yuva_streams,
  rgb_row_buf_or_scratch, rgba_plane_row_slice, rgba_u16_plane_row_slice,
};
use crate::{
  PixelSink,
  resample::{AveragingDomain, InsertionContext, InsertionPoint, select_insertion_point},
  row::*,
  source::*,
};

/// `Yuva420p` ships the RFC #238 Phase 5 STRAIGHT-alpha native fast tier
/// ([`yuva420p_process_native`]), so it is statically eligible to splice an
/// area downscale at the native Y / U / V / A codes — but ONLY under
/// [`AlphaMode::Straight`](super::AlphaMode::Straight). Premultiplied alpha
/// is bin-then-convert-incompatible and stays on the packed-YUVA area tail.
const YUVA420P_STRAIGHT_NATIVE_ELIGIBLE: bool = true;

// ---- Yuva420p impl (8-bit) ---------------------------------------------

impl<'a, R> MixedSinker<'a, Yuva420p, R> {
  /// Attaches a packed **8‑bit** RGBA output buffer. The 8‑bit YUVA
  /// source is converted to 8‑bit RGBA via the same Q15 i32 8‑bit
  /// kernel that backs [`MixedSinker<Yuv420p>::with_rgba`]; the
  /// per-pixel alpha byte is **sourced from the alpha plane** — not
  /// constant `0xFF`.
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

  /// Attaches a **`u16`** luma output buffer. The 8-bit Y plane is
  /// zero-extended to u16 (`out[x] = Y_byte as u16`). Length in u16
  /// **elements** (`width x height`).
  ///
  /// Returns `Err(InsufficientLumaU16Buffer)` if `buf.len() < width x height`,
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
}

impl<R> Yuva420pSink for MixedSinker<'_, Yuva420p, R> {}

impl<R> PixelSink for MixedSinker<'_, Yuva420p, R> {
  type Input<'r> = Yuva420pRow<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    // New frame: restart the 4-channel u8 RGBA colour stream and the
    // native-Y luma stream (both lazily created in `process`) and re-arm the
    // alpha-mode snapshot, mirroring the alpha-aware packed-YUVA (`Vuya`)
    // sink. The luma stream kind depends on the plan: the area path bins the
    // native Y at u16 (`luma_stream_u16`), the filter path resamples the
    // contiguous native Y at u8 (`luma_filter_stream`, parity with the
    // no-alpha `Yuv420p`). The 8-bit `Yuva420p` exposes no u16 colour
    // outputs, so its u16 RGBA streams are never created.
    if let Some(stream) = self.rgba_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream_u16.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.rgba_filter_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_filter_stream.as_mut() {
      stream.reset();
    }
    // Restart the RFC #238 Phase 5 straight-alpha native join (Y / U / V + α
    // streams), lazily created in `process`, so the next frame reuses it.
    if let Some(native) = self.native_yuva_420.as_mut() {
      native.reset();
    }
    // Clear the per-frame native/row-stage route freeze so the next frame may
    // pick either tier (the dispatch re-freezes on its first output-bearing
    // resampled row); a flip WITHIN a frame stays rejected. Mirrors the
    // high-bit families' `reset_high_bit_yuv_streams`.
    self.frozen_native_route = None;
    self.resample_outputs = None;
    self.frozen_alpha_mode = Some(self.alpha_mode);
    Ok(())
  }

  fn process(&mut self, row: Yuva420pRow<'_>) -> Result<(), Self::Error> {
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
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::AFull,
        idx,
        w,
        row.a().len(),
      )));
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange(
        RowIndexOutOfRange::new(idx, self.height),
      ));
    }

    // Non-identity plan: `Yuva420p` is 8-bit planar 4:2:0 YUV **with a
    // real full-resolution source alpha plane**. Route through the
    // packed-YUVA tail at `SRC_BITS = 8`: the u8 colour stream resamples the
    // converted u8 RGBA row (`yuva420p_to_rgba_row` — chroma upsampled per
    // 4:2:0, real source α from the alpha plane, NOT forced opaque) so
    // resampled alpha is a real mean and, under `AlphaMode::Premultiplied`,
    // colour is binned premultiplied; the native-Y luma stream resamples the
    // (zero-extended) Y plane directly so luma / luma_u16 are the
    // downscaled native Y, alpha- and range-independent (never derived from
    // the colour). The 8-bit `Yuva420p` exposes no u16 colour outputs, so
    // the tail's u16 colour resampling is never active (`rgb_u16` /
    // `rgba_u16` stay `None`) and its `convert_rgba_u16` closure is never
    // invoked.
    //
    // The span kind picks the engine: `Area` bins (the alpha-aware tail —
    // premultiplied colour is binned premultiplied then un-premultiplied);
    // `Filter` runs the signed-coefficient filter on the same converted RGBA
    // (PIL RGBA semantics — all four channels filtered independently,
    // straight alpha only). Premultiplied alpha has no filter analogue (the
    // engine cannot un-premultiply), so a premultiplied `Filter` plan is
    // routed to the area tail, which surfaces the typed `UnsupportedFilter`
    // rather than emitting straight-filtered premultiplied colour.
    if self.plan.is_some() {
      let alpha_mode = self.alpha_mode;
      let matrix = row.matrix();
      let full_range = row.full_range();
      let y = row.y();
      let u_half = row.u_half();
      let v_half = row.v_half();
      let a = row.a();
      let Self {
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        plan,
        rgba_stream,
        rgba_stream_u16,
        luma_stream_u16,
        rgba_filter_stream,
        rgba_filter_stream_u16,
        luma_filter_stream,
        luma_filter_stream_u16,
        resample_outputs,
        frozen_alpha_mode,
        native,
        native_yuva_420,
        frozen_native_route,
        ..
      } = self;
      let plan = plan.as_ref().expect("plan.is_some() checked above");
      check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
      return match plan.kind() {
        crate::resample::SpanKind::Area => {
          // RFC #238 Phase 5 splice-stage selection for the area downscale.
          // STRAIGHT alpha is the #235 native resolution: bin Y / U / V / A
          // codes, convert once, attach the binned α → straight RGBA. The
          // framework selector reproduces the native-vs-row-stage choice — it
          // returns `NativeCodes` exactly when this format is straight-native
          // eligible, the sink enabled the native tier, AND the alpha mode is
          // straight (a premultiplied frame must convert-at-source then
          // premultiply then bin then un-premultiply late, so it is NOT
          // native-eligible and keeps the packed-YUVA area tail BYTE-IDENTICAL).
          // `area_plan` is always true on this arm (the filter plan is handled
          // by the arms below).
          let insertion = select_insertion_point(
            AveragingDomain::Encoded,
            InsertionContext {
              native_eligible: YUVA420P_STRAIGHT_NATIVE_ELIGIBLE && !alpha_mode.is_premultiplied(),
              with_native: *native,
              area_plan: true,
            },
          );
          // Whether this call carries any output — the EXACT set both tiers'
          // no-output short-circuit tests (`need_color || need_luma` =
          // `rgb || rgba || hsv || luma || luma_u16`; the 8-bit `Yuva420p`
          // exposes no u16 colour, so those fields are always `None`). The
          // route freezes only on an output-bearing row a tier ACCEPTS; a
          // no-output call consumes no stream state, so it must not freeze.
          let need_output = rgb.is_some()
            || rgba.is_some()
            || hsv.is_some()
            || luma.is_some()
            || luma_u16.is_some();
          let take_native = matches!(insertion, InsertionPoint::NativeCodes);
          // Reject a mid-frame native/row-stage route flip (a caller toggling
          // `set_native` between rows — native row 0, row-stage row 1)
          // BEFORE either tier's dispatch: the two tiers carry independent,
          // in-order, once-only stream state, so splitting a frame across them
          // would yield a mixed/partial frame rather than a deterministic
          // rejection. CHECKED here and frozen below (the SET), both gated on
          // `need_output`, so a no-output call is route-invisible. A
          // preflight-rejected output-bearing call returns Err before the SET,
          // leaving `frozen_native_route` untouched so a later retry on the
          // same (or other) route is not falsely rejected.
          if need_output
            && let Some(frozen) = *frozen_native_route
            && frozen != take_native
          {
            return Err(MixedSinkerError::NativeRouteChanged(
              NativeRouteChanged::new(idx),
            ));
          }
          match insertion {
            // Straight-alpha native tier. The AlphaMode freeze
            // (`check_frozen_alpha_mode` above) plus the inherited native
            // preflight (out-of-sequence / frozen-output / alloc, all before
            // any feed) keep the call atomic. Dispatch first; freeze the route
            // to native ONLY after the call returns Ok on an output-bearing
            // row — a no-output call returns Ok(()) with `need_output` false
            // (no freeze); an out-of-sequence / frozen / alloc-refused row
            // returns Err via `?` (no freeze), so only an accepted
            // output-bearing row commits the route (the Phase 2 set-after-accept
            // lesson — never freeze before the row is accepted).
            InsertionPoint::NativeCodes => {
              yuva420p_process_native(
                plan,
                native_yuva_420,
                resample_outputs,
                rgb,
                rgba,
                luma,
                luma_u16,
                hsv,
                rgb_scratch,
                y,
                u_half,
                v_half,
                a,
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
              Ok(())
            }
            // Row-stage tier (the `with_native(false)` path, and EVERY
            // premultiplied frame): convert each source row to RGBA then bin —
            // the packed-YUVA area tail. Premultiplied colour is binned
            // premultiplied then un-premultiplied, byte-identical to today.
            // Same CHECK-before / SET-after split: freeze to row-stage only
            // when the call accepts an output-bearing row.
            InsertionPoint::EncodedOutput => {
              packed_yuva444_resample::<8>(
                rgba_stream,
                rgba_stream_u16,
                luma_stream_u16,
                resample_outputs,
                rgb,
                rgba,
                rgb_u16,
                rgba_u16,
                luma,
                luma_u16,
                hsv,
                rgba_scratch,
                rgb_scratch,
                rgba_scratch_u16,
                rgba_color_scratch_u16,
                luma_scratch_u16,
                w,
                plan,
                idx,
                use_simd,
                alpha_mode,
                |dst| {
                  yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd)
                },
                // `Yuva420p` has no u16 colour outputs, so this is never called.
                |_dst: &mut [u16]| {},
                |dst| {
                  for (d, &s) in dst.iter_mut().zip(y) {
                    *d = s as u16;
                  }
                },
              )?;
              if frozen_native_route.is_none() && need_output {
                *frozen_native_route = Some(false);
              }
              Ok(())
            }
            // The encoded domain only resolves to native-codes / encoded-output;
            // the linear-light splice is reached via the Linear averaging domain,
            // which this 4:2:0 YUVA sink does not expose.
            InsertionPoint::LinearLight => {
              unreachable!("Yuva420p area downscale never selects the linear-light splice")
            }
          }
        }
        crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
          // Premultiplied + filter has no analogue: route to the area tail
          // with the filter plan so it returns the typed `UnsupportedFilter`.
          packed_yuva444_resample::<8>(
            rgba_stream,
            rgba_stream_u16,
            luma_stream_u16,
            resample_outputs,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            rgba_scratch,
            rgb_scratch,
            rgba_scratch_u16,
            rgba_color_scratch_u16,
            luma_scratch_u16,
            w,
            plan,
            idx,
            use_simd,
            alpha_mode,
            |dst| yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd),
            |_dst: &mut [u16]| {},
            |dst| {
              for (d, &s) in dst.iter_mut().zip(y) {
                *d = s as u16;
              }
            },
          )
        }
        crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<8, true, false>(
          rgba_filter_stream,
          rgba_filter_stream_u16,
          luma_filter_stream,
          luma_filter_stream_u16,
          resample_outputs,
          rgb,
          rgba,
          rgb_u16,
          rgba_u16,
          luma,
          luma_u16,
          hsv,
          rgba_scratch,
          rgb_scratch,
          rgba_scratch_u16,
          rgba_color_scratch_u16,
          luma_scratch_u16,
          w,
          plan,
          idx,
          use_simd,
          // 8-bit native-Y luma rides the u8 stream (parity with `Yuv420p`):
          // the contiguous Y plane is fed directly, so no de-interleave scratch.
          y,
          None,
          |dst| yuva420p_to_rgba_row(y, u_half, v_half, a, dst, w, matrix, full_range, use_simd),
          // `Yuva420p` has no u16 colour outputs, so this closure is never called.
          |_dst: &mut [u16]| {},
          // u8-luma path: the u16 luma stream is detached, so this is never
          // called.
          |_dst: &mut [u16]| {},
          // Contiguous Y plane fed directly, so this u8 de-interleave is unused.
          |_dst: &mut [u8]| {},
        ),
      };
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

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    // HSV-without-RGB-or-RGBA goes through the direct `yuv_420_to_hsv_row`
    // kernel (no source-width RGB scratch). HSV is colour-only — the
    // source alpha plane is dropped — so a YUVA HSV is byte-identical to
    // the no-alpha `Yuv420p` HSV on the same Y/U/V; the alpha-drop RGB
    // path already reuses `yuv_420_to_rgb_row`, so HSV reuses the matching
    // planar `yuv_420_to_hsv_row`. RGB or RGBA also attached keeps the
    // convert-once-then-derive path alive via `need_rgb_kernel` (the RGB
    // kernel runs anyway, so HSV derives off that buffer; an RGBA sink
    // still needs the alpha plane, so it stays on the RGB path).
    let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
    let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

    // Acquire the u8 RGB row buffer up front — before any caller-output
    // write below — so an allocator refusal in the HSV-only scratch path
    // returns a recoverable error rather than leaving a partially-written
    // caller buffer (luma / luma_u16). `rgb_row_buf_or_scratch` is the
    // only fallible allocation on this direct path.
    let rgb_row = if need_rgb_kernel {
      Some(rgb_row_buf_or_scratch(
        rgb.as_deref_mut(),
        rgb_scratch,
        one_plane_start,
        one_plane_end,
        w,
        h,
      )?)
    } else {
      None
    };

    // Luma — copy the Y plane verbatim (8-bit YUVA's Y is already u8).
    if let Some(luma) = luma.as_deref_mut() {
      luma[one_plane_start..one_plane_end].copy_from_slice(&row.y()[..w]);
    }

    // Luma u16 — zero-extend the Y plane (`out[x] = Y_byte as u16`).
    if let Some(luma_u16) = luma_u16.as_deref_mut() {
      for (d, &s) in luma_u16[one_plane_start..one_plane_end]
        .iter_mut()
        .zip(&row.y()[..w])
      {
        *d = s as u16;
      }
    }

    // HSV-only (no RGB / RGBA): convert the source Y/U/V straight to HSV
    // via the alpha-drop planar kernel — no source-width RGB scratch.
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

    // ---- u8 RGB / RGBA / HSV path ----------------------------------
    // `need_rgb_kernel` / `rgb_row` were computed and (when needed)
    // allocated at the top, before any caller-output write.
    //
    // Direct-RGBA-only fast path routes through the alpha-source-aware
    // dispatcher.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      yuva420p_to_rgba_row(
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        rgba_row,
        w,
        row.matrix(),
        row.full_range(),
        use_simd,
      );
      return Ok(());
    }

    // RGB kernel — alpha-drop reuses the Yuv420p dispatcher verbatim.
    let Some(rgb_row) = rgb_row else {
      return Ok(());
    };
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

    // Both rgb and rgba attached (combo case): Strategy A+ — expand the
    // already-computed rgb_row → rgba_row (fills α = 0xFF), then
    // overwrite α slot from the source alpha plane. Avoids a second
    // chroma kernel.
    if want_rgba {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      crate::row::alpha_extract::copy_alpha_plane_u8(row.a(), rgba_row, w, use_simd);
    }

    Ok(())
  }
}

// ---- Yuva420p9 impl ---------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva420p9<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 1`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (9‑bit
  /// low‑packed, `[0, 511]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
  /// Output is identical to [`MixedSinker<Yuv420p9>::with_rgb_u16`].
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
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

impl<R, const BE: bool> Yuva420p9Sink<BE> for MixedSinker<'_, Yuva420p9<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva420p9<BE>, R> {
  type Input<'r> = Yuva420p9Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva420p9Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva420p_high_bit_resample::<9, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y9,
        RowSlice::UHalf9,
        RowSlice::VHalf9,
        RowSlice::AFull9,
        yuva420p9_to_rgba_row_endian,
        yuva420p9_to_rgba_u16_row_endian,
      );
    }
    yuva420p_high_bit_process::<9, BE, _, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y9,
      RowSlice::UHalf9,
      RowSlice::VHalf9,
      RowSlice::AFull9,
      yuv420p9_to_rgb_row_endian,
      yuv420p9_to_rgb_u16_row_endian,
      yuva420p9_to_rgba_row_endian,
      yuv420p9_to_hsv_row_endian,
      yuva420p9_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva420p10 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva420p10<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 2`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (10‑bit
  /// low‑packed, `[0, 1023]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
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

impl<R, const BE: bool> Yuva420p10Sink<BE> for MixedSinker<'_, Yuva420p10<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva420p10<BE>, R> {
  type Input<'r> = Yuva420p10Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva420p10Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva420p_high_bit_resample::<10, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y10,
        RowSlice::UHalf10,
        RowSlice::VHalf10,
        RowSlice::AFull10,
        yuva420p10_to_rgba_row_endian,
        yuva420p10_to_rgba_u16_row_endian,
      );
    }
    yuva420p_high_bit_process::<10, BE, _, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y10,
      RowSlice::UHalf10,
      RowSlice::VHalf10,
      RowSlice::AFull10,
      yuv420p10_to_rgb_row_endian,
      yuv420p10_to_rgb_u16_row_endian,
      yuva420p10_to_rgba_row_endian,
      yuv420p10_to_hsv_row_endian,
      yuva420p10_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva420p12 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva420p12<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 4`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (12‑bit
  /// low‑packed, `[0, 4095]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
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

impl<R, const BE: bool> Yuva420p12Sink<BE> for MixedSinker<'_, Yuva420p12<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva420p12<BE>, R> {
  type Input<'r> = Yuva420p12Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva420p12Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva420p_high_bit_resample::<12, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y12,
        RowSlice::UHalf12,
        RowSlice::VHalf12,
        RowSlice::AFull12,
        yuva420p12_to_rgba_row_endian,
        yuva420p12_to_rgba_u16_row_endian,
      );
    }
    yuva420p_high_bit_process::<12, BE, _, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y12,
      RowSlice::UHalf12,
      RowSlice::VHalf12,
      RowSlice::AFull12,
      yuv420p12_to_rgb_row_endian,
      yuv420p12_to_rgb_u16_row_endian,
      yuva420p12_to_rgba_row_endian,
      yuv420p12_to_hsv_row_endian,
      yuva420p12_to_rgba_u16_row_endian,
    )
  }
}

// ---- Yuva420p16 impl --------------------------------------------------

impl<'a, R, const BE: bool> MixedSinker<'a, Yuva420p16<BE>, R> {
  /// Attaches a packed **8-bit** RGBA output buffer. Source-derived
  /// alpha (depth-converted via `>> 8`).
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

  /// Attaches a packed **`u16`** RGBA output buffer (16‑bit, full
  /// `u16` range `[0, 65535]`). Alpha is sourced at native depth.
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

  /// Attaches a packed **`u16`** RGB output buffer (alpha-drop).
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

  /// Attaches a **`u16`** luma output buffer. Luma is the **native Y**
  /// (the binned Y plane at native depth under a non-identity plan; the
  /// Y plane verbatim otherwise). Length in u16 **elements**
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

impl<R, const BE: bool> Yuva420p16Sink<BE> for MixedSinker<'_, Yuva420p16<BE>, R> {}

impl<R, const BE: bool> PixelSink for MixedSinker<'_, Yuva420p16<BE>, R> {
  type Input<'r> = Yuva420p16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    if self.width & 1 != 0 {
      return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(
        self.width,
      )));
    }
    check_dimensions_match(self.width, self.height, width, height)?;
    reset_high_bit_yuva_streams(self);
    Ok(())
  }

  fn process(&mut self, row: Yuva420p16Row<'_>) -> Result<(), Self::Error> {
    if self.plan.is_some() {
      return yuva420p_high_bit_resample::<16, BE>(
        self,
        row.row(),
        row.y(),
        row.u_half(),
        row.v_half(),
        row.a(),
        row.matrix(),
        row.full_range(),
        RowSlice::Y16,
        RowSlice::UHalf16,
        RowSlice::VHalf16,
        RowSlice::AFull16,
        yuva420p16_to_rgba_row_endian,
        yuva420p16_to_rgba_u16_row_endian,
      );
    }
    yuva420p_high_bit_process::<16, BE, _, _, _, _, _, _>(
      self,
      row.row(),
      row.y(),
      row.u_half(),
      row.v_half(),
      row.a(),
      row.matrix(),
      row.full_range(),
      RowSlice::Y16,
      RowSlice::UHalf16,
      RowSlice::VHalf16,
      RowSlice::AFull16,
      yuv420p16_to_rgb_row_endian,
      yuv420p16_to_rgb_u16_row_endian,
      yuva420p16_to_rgba_row_endian,
      yuv420p16_to_hsv_row_endian,
      yuva420p16_to_rgba_u16_row_endian,
    )
  }
}

// ---- Shared high-bit YUVA 4:2:0 process body --------------------------
//
// The 9 / 10 / 16-bit YUVA 4:2:0 sinker `process` bodies are
// structurally identical — only the depth-named row primitives, the
// `RowSlice` variants used in error reports, and the depth-conversion
// shift (`BITS - 8`) for luma differ. Factor the shared body into a
// generic helper to avoid 3x of the same ~70 lines.
//
// Strategy A combine for the alpha-drop paths (rgb_u16 alpha-drop +
// rgba_u16 source alpha): runs the rgb_u16 kernel into the caller
// buffer + the rgba_u16 kernel separately. Cannot reuse rgb_u16's
// output because alpha comes from the source plane; the expand
// helper would splat opaque alpha.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva420p_high_bit_process<
  const BITS: u32,
  const BE: bool,
  F: crate::SourceFormat,
  R,
  // Each dispatch fn is the `_endian` variant — last `bool` is the
  // runtime `big_endian` flag the helper passes from `BE`.
  RgbRowFn: Fn(&[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool, bool),
  RgbU16RowFn: Fn(&[u16], &[u16], &[u16], &mut [u16], usize, crate::ColorMatrix, bool, bool, bool),
  RgbaRowFn: Fn(&[u16], &[u16], &[u16], &[u16], &mut [u8], usize, crate::ColorMatrix, bool, bool, bool),
  // The matching alpha-drop YUV→HSV `_endian` kernel — same Y/U/V inputs
  // as `rgb_dispatch`, three `&mut [u8]` H/S/V outputs.
  HsvRowFn: Fn(
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    &mut [u8],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
>(
  sinker: &mut MixedSinker<'_, F, R>,
  idx: usize,
  y_row: &[u16],
  u_half_row: &[u16],
  v_half_row: &[u16],
  a_row: &[u16],
  matrix: crate::ColorMatrix,
  full_range: bool,
  y_slice: RowSlice,
  u_slice: RowSlice,
  v_slice: RowSlice,
  a_slice: RowSlice,
  rgb_dispatch: RgbRowFn,
  rgb_u16_dispatch: RgbU16RowFn,
  rgba_dispatch: RgbaRowFn,
  hsv_dispatch: HsvRowFn,
  rgba_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let h = sinker.height;
  let use_simd = sinker.simd;

  if w & 1 != 0 {
    return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
  }
  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      y_slice,
      idx,
      w,
      y_row.len(),
    )));
  }
  if u_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      u_slice,
      idx,
      w / 2,
      u_half_row.len(),
    )));
  }
  if v_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      v_slice,
      idx,
      w / 2,
      v_half_row.len(),
    )));
  }
  if a_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      a_slice,
      idx,
      w,
      a_row.len(),
    )));
  }
  if idx >= sinker.height {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, sinker.height),
    ));
  }

  let MixedSinker {
    rgb,
    rgb_u16,
    rgba,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgb_scratch,
    ..
  } = sinker;
  let one_plane_start = idx * w;
  let one_plane_end = one_plane_start + w;

  let want_rgb = rgb.is_some();
  let want_rgba = rgba.is_some();
  let want_hsv = hsv.is_some();
  // HSV-without-RGB-or-RGBA goes through the direct alpha-drop
  // `hsv_dispatch` kernel (no source-width RGB scratch). HSV is
  // colour-only — the source alpha plane is dropped — so a high-bit YUVA
  // HSV is byte-identical to the no-alpha planar HSV on the same Y/U/V
  // (the alpha-drop `rgb_dispatch` and `hsv_dispatch` share the same
  // depth-named kernel family). RGB or RGBA also attached keeps the
  // convert-once-then-derive path alive via `need_rgb_kernel` (an RGBA
  // sink still needs the alpha plane, so it stays on the RGB path).
  let want_hsv_direct = want_hsv && !want_rgb && !want_rgba;
  let need_rgb_kernel = want_rgb || (want_hsv && want_rgba);

  // Acquire the u8 RGB row buffer (caller plane or growing scratch) up
  // front — before any caller-output write below — so an allocator
  // refusal in the HSV-only scratch path returns a recoverable error
  // rather than leaving a partially-written caller buffer (luma_u16 /
  // rgb_u16 / rgba_u16). `rgb_row_buf_or_scratch` is the only fallible
  // allocation on this direct path; the per-row writes that follow are
  // infallible. Mirrors the `packed_rgb_10bit` direct path.
  let rgb_row = if need_rgb_kernel {
    Some(rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?)
  } else {
    None
  };

  // ---- luma (native Y; luma narrows `>> (BITS - 8)`, luma_u16 is the
  // host-native logical Y) ---------------------------------------------
  if luma.is_some() || luma_u16.is_some() {
    let luma_dst = luma.as_deref_mut();
    let luma_u16_dst = luma_u16.as_deref_mut();
    // Borrow each output's row slice up front so the per-pixel loop just
    // writes (the two outputs are disjoint fields).
    let mut luma_row = luma_dst.map(|b| &mut b[one_plane_start..one_plane_end]);
    let mut luma_u16_row = luma_u16_dst.map(|b| &mut b[one_plane_start..one_plane_end]);
    // Mask each decoded Y to the source's native depth `(1 << BITS) - 1`
    // (a no-op at `BITS = 16`). `Yuva420p*Frame::try_new` is geometry-only,
    // so a malformed-but-accepted frame can carry out-of-range Y (e.g.
    // `0x1000` at 12-bit); without this mask `luma_u16` would publish that
    // raw value, inconsistent with the `(1 << BITS) - 1`-masked Y the
    // RGB/RGBA row kernels decode from the same row.
    let sample_mask = ((1u32 << BITS) - 1) as u16;
    for (i, &s) in y_row.iter().enumerate().take(w) {
      // Normalize BE-encoded wire bytes to host-native before use —
      // without this, a valid BE sample like mid-gray `0x0200` (10-bit)
      // would be read as `0x0002` on a LE host and the `>> (BITS - 8)`
      // would write 0 instead of 128.
      let logical = (if BE { u16::from_be(s) } else { u16::from_le(s) }) & sample_mask;
      if let Some(row) = luma_row.as_deref_mut() {
        row[i] = (logical >> (BITS - 8)) as u8;
      }
      if let Some(row) = luma_u16_row.as_deref_mut() {
        row[i] = logical;
      }
    }
  }

  // ---- u16 RGB / RGBA path --------------------------------------
  // rgb_u16 (alpha-drop) reuses the non-alpha dispatcher.
  // rgba_u16 — when rgb_u16 is also present (combo case), Strategy A+:
  // expand the already-computed rgb_u16_row → rgba_u16_row then
  // overwrite α slot from the source plane (avoids a second chroma
  // kernel). When rgba_u16 is alone, delegate to the alpha-source-aware
  // dispatcher directly (standalone path — already optimal).
  let want_rgb_u16 = rgb_u16.is_some();
  let want_rgba_u16 = rgba_u16.is_some();
  if want_rgb_u16 {
    let buf = rgb_u16.as_deref_mut().unwrap();
    let rgb_plane_end = one_plane_end
      .checked_mul(3)
      .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
        w, h, 3,
      )))?;
    let rgb_plane_start = one_plane_start * 3;
    let rgb_u16_row = &mut buf[rgb_plane_start..rgb_plane_end];
    rgb_u16_dispatch(
      y_row,
      u_half_row,
      v_half_row,
      rgb_u16_row,
      w,
      matrix,
      full_range,
      use_simd,
      BE,
    );
    if want_rgba_u16 {
      // Combo: expand rgb_u16_row → rgba_u16_row, then overwrite α slot.
      let rgba_buf = rgba_u16.as_deref_mut().unwrap();
      let rgba_u16_row = rgba_u16_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_u16_to_rgba_u16_row::<BITS>(rgb_u16_row, rgba_u16_row, w);
      // `BE` is plumbed from the row type so the alpha-plane copy honors
      // the same wire endianness as the caller's high-bit Yuva*p input.
      crate::row::alpha_extract::copy_alpha_plane_u16::<BITS, BE>(a_row, rgba_u16_row, w, use_simd);
    }
  } else if want_rgba_u16 {
    // Standalone rgba_u16: delegate to the alpha-source-aware dispatcher.
    let rgba_buf = rgba_u16.as_deref_mut().unwrap();
    let rgba_u16_row = rgba_u16_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_u16_dispatch(
      y_row,
      u_half_row,
      v_half_row,
      a_row,
      rgba_u16_row,
      w,
      matrix,
      full_range,
      use_simd,
      BE,
    );
  }

  // ---- u8 RGB / RGBA / HSV path ----------------------------------
  // `need_rgb_kernel` / `rgb_row` were computed and (when needed)
  // allocated at the top, before any caller-output write.
  //
  // HSV-only (no u8 RGB / RGBA): convert the source Y/U/V straight to HSV
  // via the alpha-drop kernel — no source-width RGB scratch. Any u16
  // RGB/RGBA outputs already ran above.
  if want_hsv_direct {
    let hsv = hsv.as_mut().expect("want_hsv_direct implies hsv attached");
    let (h, s, v) = hsv.hsv();
    hsv_dispatch(
      y_row,
      u_half_row,
      v_half_row,
      &mut h[one_plane_start..one_plane_end],
      &mut s[one_plane_start..one_plane_end],
      &mut v[one_plane_start..one_plane_end],
      w,
      matrix,
      full_range,
      use_simd,
      BE,
    );
    return Ok(());
  }

  if want_rgba && !need_rgb_kernel {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    rgba_dispatch(
      y_row, u_half_row, v_half_row, a_row, rgba_row, w, matrix, full_range, use_simd, BE,
    );
    return Ok(());
  }

  // RGB kernel (alpha-drop reuses the non-alpha dispatcher verbatim).
  let Some(rgb_row) = rgb_row else {
    return Ok(());
  };
  rgb_dispatch(
    y_row, u_half_row, v_half_row, rgb_row, w, matrix, full_range, use_simd, BE,
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

  // Both rgb and rgba attached (combo case): Strategy A+ — expand the
  // already-computed rgb_row → rgba_row (fills α = 0xFF), then
  // overwrite α slot from the source alpha plane with depth-conv
  // `>> (BITS - 8)`. Avoids a second chroma kernel.
  if want_rgba {
    let rgba_buf = rgba.as_deref_mut().unwrap();
    let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
    expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    // BE = false: see the rgba_u16 branch above for rationale.
    crate::row::alpha_extract::copy_alpha_plane_u16_to_u8::<BITS, BE>(a_row, rgba_row, w, use_simd);
  }

  Ok(())
}

// ---- Shared high-bit YUVA 4:2:0 resample-routing body -----------------
//
// The non-identity plan branch for the 9 / 10 / 16-bit YUVA 4:2:0 sinks.
// Mirrors the 8-bit `Yuva420p` plan branch but at native depth `BITS`. The
// `Area` (downscale) arm routes through the shared packed-YUVA area tail with
// THREE independent binnings — the u8 colour stream (the `_to_rgba` u8 kernel
// with real source α, chroma upsampled 4:2:0), the **independent** native u16
// colour stream (the `_to_rgba_u16` kernel — never a narrowing of the u8
// bin), and the **low-packed** native-Y luma stream
// (`deinterleave_y_high_bit_masked::<BITS, BE>`, a host-native copy masked to
// the native depth — planar YUVA Y stores logical values directly, unlike the
// high-bit-packed semi-planar P-formats, so luma is `binned_Y >> (BITS - 8)`).
// The `Filter` arm routes
// the SAME converted RGBA / native-Y through the signed-coefficient filter
// tail (`NATIVE_LUMA_U8 = false`, the u16-luma branch — native Y is u16),
// straight alpha only. The 4:2:0-vs-4:2:2 difference (chroma row index `r / 2`
// vs `r`) lives in the walker, so the 4:2:2 sinks reuse this body verbatim
// with the same half-chroma kernels.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[cfg_attr(not(tarpaulin), inline(always))]
fn yuva420p_high_bit_resample<const BITS: u32, const BE: bool>(
  sinker: &mut MixedSinker<'_, impl crate::SourceFormat, impl Sized>,
  idx: usize,
  y_row: &[u16],
  u_half_row: &[u16],
  v_half_row: &[u16],
  a_row: &[u16],
  matrix: crate::ColorMatrix,
  full_range: bool,
  y_slice: RowSlice,
  u_slice: RowSlice,
  v_slice: RowSlice,
  a_slice: RowSlice,
  rgba_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u8],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
  rgba_u16_dispatch: fn(
    &[u16],
    &[u16],
    &[u16],
    &[u16],
    &mut [u16],
    usize,
    crate::ColorMatrix,
    bool,
    bool,
    bool,
  ),
) -> Result<(), MixedSinkerError> {
  let w = sinker.width;
  let use_simd = sinker.simd;

  if w & 1 != 0 {
    return Err(MixedSinkerError::WidthAlignment(WidthAlignment::odd(w)));
  }
  if y_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      y_slice,
      idx,
      w,
      y_row.len(),
    )));
  }
  if u_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      u_slice,
      idx,
      w / 2,
      u_half_row.len(),
    )));
  }
  if v_half_row.len() != w / 2 {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      v_slice,
      idx,
      w / 2,
      v_half_row.len(),
    )));
  }
  if a_row.len() != w {
    return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
      a_slice,
      idx,
      w,
      a_row.len(),
    )));
  }
  if idx >= sinker.height {
    return Err(MixedSinkerError::RowIndexOutOfRange(
      RowIndexOutOfRange::new(idx, sinker.height),
    ));
  }

  let alpha_mode = sinker.alpha_mode;
  let MixedSinker {
    rgb,
    rgba,
    rgb_u16,
    rgba_u16,
    luma,
    luma_u16,
    hsv,
    rgba_scratch,
    rgb_scratch,
    rgba_scratch_u16,
    rgba_color_scratch_u16,
    luma_scratch_u16,
    plan,
    rgba_stream,
    rgba_stream_u16,
    luma_stream_u16,
    rgba_filter_stream,
    rgba_filter_stream_u16,
    luma_filter_stream_u16,
    resample_outputs,
    frozen_alpha_mode,
    ..
  } = sinker;
  let plan = plan.as_ref().expect("plan.is_some() checked by the caller");
  check_frozen_alpha_mode(*frozen_alpha_mode, alpha_mode, idx)?;
  // The span kind picks the engine (mirrors the 8-bit `Yuva420p` and packed
  // `Vuya`): `Area` bins (the alpha-aware tail — premultiplied colour binned
  // premultiplied then un-premultiplied) at native precision; `Filter` runs
  // the signed-coefficient filter on the same converted RGBA (straight alpha
  // only). Unlike the 8-bit planar sinks (whose native Y is `u8`), the
  // high-bit native Y is genuinely `u16`, so the filter route uses
  // `NATIVE_LUMA_U8 = false` — the same u16-luma branch `Vuya` / `Vuyx` use:
  // luma rides the `u16` filter stream over the de-interleaved native Y. The
  // filter tail clamps every sub-16-bit colour AND native-Y overshoot to the
  // source's native max `(1 << BITS) - 1` (a value no-op at `BITS = 16`),
  // matching the in-range area path. A premultiplied `Filter` plan has no
  // analogue (the engine cannot un-premultiply), so it is routed to the area
  // tail, which surfaces the typed `UnsupportedFilter`.
  match plan.kind() {
    crate::resample::SpanKind::Area => packed_yuva444_resample::<BITS>(
      rgba_stream,
      rgba_stream_u16,
      luma_stream_u16,
      resample_outputs,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgba_scratch,
      rgb_scratch,
      rgba_scratch_u16,
      rgba_color_scratch_u16,
      luma_scratch_u16,
      w,
      plan,
      idx,
      use_simd,
      alpha_mode,
      |dst| {
        rgba_dispatch(
          y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| {
        rgba_u16_dispatch(
          y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
    ),
    crate::resample::SpanKind::Filter if alpha_mode.is_premultiplied() => {
      // Premultiplied + filter has no analogue: route to the area tail with
      // the filter plan so it returns the typed `UnsupportedFilter`.
      packed_yuva444_resample::<BITS>(
        rgba_stream,
        rgba_stream_u16,
        luma_stream_u16,
        resample_outputs,
        rgb,
        rgba,
        rgb_u16,
        rgba_u16,
        luma,
        luma_u16,
        hsv,
        rgba_scratch,
        rgb_scratch,
        rgba_scratch_u16,
        rgba_color_scratch_u16,
        luma_scratch_u16,
        w,
        plan,
        idx,
        use_simd,
        alpha_mode,
        |dst| {
          rgba_dispatch(
            y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
          )
        },
        |dst| {
          rgba_u16_dispatch(
            y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
          )
        },
        |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
      )
    }
    crate::resample::SpanKind::Filter => packed_yuva444_filter_resample::<BITS, false, false>(
      rgba_filter_stream,
      rgba_filter_stream_u16,
      // High-bit planar YUVA never uses the u8 native-Y luma stream
      // (`NATIVE_LUMA_U8 = false`); pass an inert slot.
      &mut None,
      luma_filter_stream_u16,
      resample_outputs,
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      rgba_scratch,
      rgb_scratch,
      rgba_scratch_u16,
      rgba_color_scratch_u16,
      luma_scratch_u16,
      w,
      plan,
      idx,
      use_simd,
      // Luma rides `deinterleave_y` + the u16 stream (native Y is u16), so the
      // u8-luma input is unused.
      &[],
      None,
      |dst| {
        rgba_dispatch(
          y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| {
        rgba_u16_dispatch(
          y_row, u_half_row, v_half_row, a_row, dst, w, matrix, full_range, use_simd, BE,
        )
      },
      |dst| deinterleave_y_high_bit_masked::<BITS, BE>(y_row, dst, w),
      |_dst: &mut [u8]| {},
    ),
  }
}
