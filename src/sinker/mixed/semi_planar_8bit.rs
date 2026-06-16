//! 8-bit semi-planar YUV `MixedSinker` impls: Nv12 / Nv16 / Nv21 / Nv24 / Nv42.
//!
//! On a non-identity plan every member routes through the shared
//! row-stage planar resample ([`super::planar_resample::planar_dual_resample`]):
//! the Y plane area-resamples directly for luma (the YUV luma contract),
//! while RGB / RGBA / HSV bin a source-width RGB row converted with the
//! format's own fused `nv*_to_rgb_row` kernel (chroma de-interleave +
//! upsample happen in registers inside that kernel, exactly as on the
//! identity path). RGB therefore equals an `Rgb24` area-resample of the
//! identity-converted frame — byte-identical to the matching
//! [`Yuv420p`] (row-stage) / [`Yuv422p`] / [`Yuv444p`] resample of the
//! de-interleaved planes. The 4:2:0 native decimation tier is a
//! planar-only optimization and does not apply here.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, WidthAlignment, check_dimensions_match,
  planar_resample::planar_dual_resample, rgb_row_buf_or_scratch, rgba_plane_row_slice,
};
use crate::{PixelSink, row::*, source::*};

#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use super::{HsvFrameMut, planar_8bit::yuv420p_process_native};
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
use crate::{
  ColorMatrix,
  resample::{PlanGeometry, ResampleError, ResamplePlan},
};

// Test-only allocation failpoint for the U / V de-interleave scratch grow
// in `semi_planar_process_native`. When armed, the next chroma-bearing
// reserve returns the crate's recoverable `AllocationFailed` WITHOUT
// growing — letting the atomicity regression test prove the first-row
// out-of-sequence preflight runs BEFORE this fallible grow (so a rejected
// even-colour first row returns OutOfSequenceRow, never AllocationFailed).
// `Cell<bool>` is plenty (single-threaded, take-on-read). Strictly
// test-only — the non-test build is byte-identical (this hook compiles
// away entirely).
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
std::thread_local! {
  static FORCE_DEINTERLEAVE_ALLOC_FAILURE: core::cell::Cell<bool> =
    const { core::cell::Cell::new(false) };
}

/// Arms the de-interleave scratch allocation failpoint for the **next**
/// chroma-bearing native semi-planar row on the current thread. The flag
/// is consumed (take-on-read) by that reserve, so it fires exactly once
/// and cannot leak into a later test. Test-only.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
pub(super) fn arm_deinterleave_alloc_failure() {
  FORCE_DEINTERLEAVE_ALLOC_FAILURE.with(|f| f.set(true));
}

/// Native fast-tier 4:2:0 decimator for the semi-planar family
/// ([`Nv12`](crate::source::Nv12) / [`Nv21`](crate::source::Nv21)): bins
/// the native Y / U / V planes straight to the output grid and converts
/// once per output row at output resolution. Reuses the planar twin's
/// join verbatim ([`yuv420p_process_native`]) after de-interleaving the
/// interleaved chroma row into the sink's U / V scratch — so every output
/// is byte-identical to a [`Yuv420p`](crate::source::Yuv420p) native
/// conversion of the de-interleaved planes, and within ±1 LSB of the
/// semi-planar row-stage tier (the conversion-order rounding caveat the
/// planar tiers already carry).
///
/// `chroma_uv` is the interleaved chroma half-row; `swap_uv = false`
/// reads `U0 V0 U1 V1 …` (NV12), `swap_uv = true` reads `V0 U0 …`
/// (NV21). The chroma row is consumed only on even source rows; the
/// caller passes the full interleaved row regardless and this splits it.
///
/// The U / V scratch is reserved (fallibly) before the call into the
/// planar join, and the de-interleave writes only into that private
/// scratch — so the recoverable-allocation / atomicity contract the join
/// enforces (no caller-output write before the preflight completes) holds
/// across the de-interleave too.
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
#[allow(clippy::too_many_arguments)]
fn semi_planar_process_native(
  plan: &ResamplePlan,
  native_420: &mut Option<super::planar_8bit::NativeYuv420>,
  u_scratch: &mut std::vec::Vec<u8>,
  v_scratch: &mut std::vec::Vec<u8>,
  resample_outputs: &mut Option<super::FrozenOutputs>,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  rgb_scratch: &mut std::vec::Vec<u8>,
  y_row: &[u8],
  chroma_uv: &[u8],
  swap_uv: bool,
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

  // Run the join's COMPLETE pre-feed rejection preflight FIRST — no-output
  // short-circuit, first-row out-of-sequence check, AND the frozen-output
  // (mid-frame output-set change) check — before touching the U / V
  // de-interleave scratch. The de-interleave reserve below is fallible and
  // grows sink state; deferring it until the full preflight clears keeps
  // EVERY rejection case (out-of-sequence first colour row OR mid-frame
  // output change) returning its deterministic typed error
  // (OutOfSequenceRow / ResampleOutputsChanged), never AllocationFailed
  // under allocation pressure, and leaves the scratch untouched — the
  // crate's preflight-atomicity contract. `Ok(false)` is the no-output
  // no-op: return without reserving. `yuv420p_process_native` re-runs this
  // identical preflight harmlessly (the freeze stores on the first
  // output-bearing row, the second run is a matching check), keeping a
  // single source of truth.
  if !super::planar_8bit::yuv420p_native_preflight(
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

  // De-interleave the chroma half-row into the U / V scratch — only on
  // chroma-bearing rows (even source index) when colour is wanted, which
  // is exactly where the planar join reads chroma and nowhere else. The
  // split writes only this private scratch, so no caller output is touched
  // until the join's own preflight (re-run inside the call below) clears.
  // On odd / luma-only / no-colour rows the join never reads chroma, so the
  // scratch is left as-is and the join gets empty U / V slices — which also
  // keeps a direct caller's out-of-sequence odd first row (no
  // `begin_frame`, empty scratch) from indexing past the scratch before the
  // join rejects it.
  let chroma_row = need_color && idx.is_multiple_of(2);
  if chroma_row {
    for scratch in [&mut *u_scratch, &mut *v_scratch] {
      if scratch.len() < cw {
        // Test-only failpoint: simulate a recoverable allocator refusal of
        // the de-interleave scratch grow WITHOUT exhausting memory, so the
        // regression test can prove the first-row preflight already
        // rejected an out-of-sequence colour row (returning
        // OutOfSequenceRow) before this fallible grow is ever reached. With
        // the preflight ordered AFTER this grow (the bug) an armed failure
        // would surface as AllocationFailed instead.
        #[cfg(all(
          test,
          feature = "std",
          feature = "yuv-semi-planar",
          feature = "yuv-planar"
        ))]
        if FORCE_DEINTERLEAVE_ALLOC_FAILURE.with(|f| f.take()) {
          return Err(MixedSinkerError::Resample(ResampleError::AllocationFailed(
            PlanGeometry::new(w, h, plan.out_w(), plan.out_h()),
          )));
        }
        scratch.try_reserve_exact(cw - scratch.len()).map_err(|_| {
          MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
            w,
            h,
            plan.out_w(),
            plan.out_h(),
          )))
        })?;
        scratch.resize(cw, 0);
      }
    }
    // NV12 chroma is `U V U V …` (U at even byte), NV21 is `V U V U …`.
    let (u_off, v_off) = if swap_uv { (1, 0) } else { (0, 1) };
    for (i, pair) in chroma_uv.chunks_exact(2).enumerate() {
      u_scratch[i] = pair[u_off];
      v_scratch[i] = pair[v_off];
    }
  }

  let (u_half, v_half): (&[u8], &[u8]) = if chroma_row {
    (&u_scratch[..cw], &v_scratch[..cw])
  } else {
    (&[], &[])
  };
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
    y_row,
    u_half,
    v_half,
    matrix,
    full_range,
    idx,
    w,
    h,
    use_simd,
  )
}

// ---- Nv12 impl ----------------------------------------------------------

impl<'a, R> MixedSinker<'a, Nv12, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — calling `with_rgba` on a sink that doesn't (e.g. a
  /// not‑yet‑wired `MixedSinker<Nv16>` today) is a compile error
  /// rather than a silent no‑op. Each format that adds RGBA support
  /// adds its own impl block here.
  ///
  /// The fourth byte per pixel is alpha. NV12 has no alpha plane,
  /// so every alpha byte is filled with `0xFF` (opaque). Future
  /// YUVA source impls will copy alpha through from the source
  /// plane.
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

impl<R> Nv12Sink for MixedSinker<'_, Nv12, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv12, R> {
  type Input<'r> = Nv12Row<'r>;
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
    // New frame: restart the row-stage resample streams so a reused sink
    // starts each frame clean.
    if let Some(stream) = self.rgb_stream.as_mut() {
      stream.reset();
    }
    if let Some(stream) = self.luma_stream.as_mut() {
      stream.reset();
    }
    #[cfg(feature = "yuv-planar")]
    if let Some(native) = self.native_420.as_mut() {
      native.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Nv12Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth shape check (see Yuv420p impl above). An NV12
    // UV row is `width` bytes of interleaved U / V payload — same
    // length as Y — so both slices must equal `self.width`. Odd-width
    // check comes first since the row primitive would panic on it.
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
    if row.uv_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvHalf,
        idx,
        w,
        row.uv_half().len(),
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
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_420,
      #[cfg(feature = "yuv-planar")]
      semi_planar_u_half,
      #[cfg(feature = "yuv-planar")]
      semi_planar_v_half,
      ..
    } = self;

    // Non-identity plan. When the native tier is enabled (and the planar
    // join it reuses is compiled in), bin the native Y / U / V planes at
    // output resolution and convert once per output row, de-interleaving
    // the NV12 chroma row into U / V scratch first. Otherwise (or under
    // `with_native(false)`) take the row-stage tier: bin the Y plane for
    // luma directly (the YUV luma contract); for colour, convert the
    // interleaved source row to RGB with the same fused `nv12_to_rgb_row`
    // kernel the identity path uses, then bin the RGB row. RGB therefore
    // equals an `Rgb24` area-resample of the identity-converted frame —
    // byte-identical to the `Yuv420p` row-stage twin.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      #[cfg(feature = "yuv-planar")]
      if *native {
        return semi_planar_process_native(
          plan,
          native_420,
          semi_planar_u_half,
          semi_planar_v_half,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.uv_half(),
          false,
          matrix,
          full_range,
          idx,
          w,
          h,
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
          nv12_to_rgb_row(
            row.y(),
            row.uv_half(),
            scratch,
            w,
            matrix,
            full_range,
            use_simd,
          );
        },
      );
    }

    // Single-plane row ranges are guaranteed to fit; RGB / RGBA
    // ranges use checked arithmetic (see the Yuv420p impl above for
    // the full rationale — hsv-only attachment never validated x 3).
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // Luma — NV12 luma is the Y plane. Copy verbatim.
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
      nv12_to_rgba_row(
        row.y(),
        row.uv_half(),
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

    // Fused NV12 → RGB: UV deinterleave + chroma upsample both happen
    // in registers inside the row primitive, no intermediate memory.
    nv12_to_rgb_row(
      row.y(),
      row.uv_half(),
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

// ---- Nv16 impl ----------------------------------------------------------
//
// 4:2:2 is 4:2:0's vertical‑axis twin: one UV row per Y row instead of
// one per two. Per‑row math is identical, so this impl calls the same
// `nv12_to_rgb_row` / `nv12_to_rgba_row` dispatchers — no new kernels
// needed.

impl<'a, R> MixedSinker<'a, Nv16, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. NV16 has no alpha plane, so every
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

impl<R> Nv16Sink for MixedSinker<'_, Nv16, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv16, R> {
  type Input<'r> = Nv16Row<'r>;
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

  fn process(&mut self, row: Nv16Row<'_>) -> Result<(), Self::Error> {
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
    // NV16 UV row is `width` bytes of interleaved U/V — identical shape
    // to NV12's `uv_half`.
    if row.uv().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvHalf,
        idx,
        w,
        row.uv().len(),
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

    // Non-identity plan: row-stage fused downscale (matches the Yuv422p
    // twin). Bin Y for luma; for colour, convert the interleaved source
    // row to RGB with the fused `nv12_to_rgb_row` kernel the identity
    // path reuses for 4:2:2 (one chroma row per Y row), then bin it.
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
          nv12_to_rgb_row(row.y(), row.uv(), scratch, w, matrix, full_range, use_simd);
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
    // Reuses NV12 dispatchers (RGB and RGBA) since 4:2:2's row
    // contract is identical.
    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      nv12_to_rgba_row(
        row.y(),
        row.uv(),
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

    // Reuses the NV12 dispatcher — 4:2:2's row contract is identical.
    nv12_to_rgb_row(
      row.y(),
      row.uv(),
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

// ---- Nv21 impl ----------------------------------------------------------
//
// Structurally identical to the Nv12 impl — the row primitives hide
// the U/V byte-order difference. Only the trait `Input<'r>` and the
// primitive name change.

impl<'a, R> MixedSinker<'a, Nv21, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Nv12>::with_rgba`] for the same
  /// rationale and constraints. NV21 has no alpha plane, so every
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

impl<R> Nv21Sink for MixedSinker<'_, Nv21, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv21, R> {
  type Input<'r> = Nv21Row<'r>;
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
    #[cfg(feature = "yuv-planar")]
    if let Some(native) = self.native_420.as_mut() {
      native.reset();
    }
    self.resample_outputs = None;
    Ok(())
  }

  fn process(&mut self, row: Nv21Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense in depth: same shape check as the Nv12 impl. A VU row
    // has `width` bytes of interleaved V / U payload — same length
    // as Y — so both slices must equal `self.width`.
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
    if row.vu_half().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VuHalf,
        idx,
        w,
        row.vu_half().len(),
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
      #[cfg(feature = "yuv-planar")]
      native,
      #[cfg(feature = "yuv-planar")]
      native_420,
      #[cfg(feature = "yuv-planar")]
      semi_planar_u_half,
      #[cfg(feature = "yuv-planar")]
      semi_planar_v_half,
      ..
    } = self;

    // Non-identity plan. When the native tier is enabled (and the planar
    // join it reuses is compiled in), bin the native Y / U / V planes at
    // output resolution and convert once per output row, de-interleaving
    // the NV21 VU chroma row into U / V scratch first. Otherwise (or under
    // `with_native(false)`) take the row-stage tier (matches the Yuv420p
    // row-stage twin): bin Y for luma; for colour, convert the interleaved
    // VU source row to RGB with the fused `nv21_to_rgb_row` kernel the
    // identity path uses, then bin the RGB row.
    if let Some(plan) = plan.as_ref() {
      let matrix = row.matrix();
      let full_range = row.full_range();
      #[cfg(feature = "yuv-planar")]
      if *native {
        return semi_planar_process_native(
          plan,
          native_420,
          semi_planar_u_half,
          semi_planar_v_half,
          resample_outputs,
          rgb,
          rgba,
          luma,
          luma_u16,
          hsv,
          rgb_scratch,
          row.y(),
          row.vu_half(),
          true,
          matrix,
          full_range,
          idx,
          w,
          h,
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
          nv21_to_rgb_row(
            row.y(),
            row.vu_half(),
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
      nv21_to_rgba_row(
        row.y(),
        row.vu_half(),
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

    // Fused NV21 → RGB: VU deinterleave + chroma upsample both happen
    // in registers inside the row primitive, no intermediate memory.
    nv21_to_rgb_row(
      row.y(),
      row.vu_half(),
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

// ---- Nv24 impl ----------------------------------------------------------
//
// 4:4:4 semi-planar: UV plane is full-width (`2 * width` bytes per
// row), one UV pair per Y pixel. No width parity constraint. Kernel
// is its own family (`nv24_to_rgb_row`) since chroma is no longer
// duplicated across columns.

impl<'a, R> MixedSinker<'a, Nv24, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// Only available on sinker types whose `PixelSink` impl writes
  /// RGBA — see [`MixedSinker::<Yuv420p>::with_rgba`] for the same
  /// rationale and constraints. Nv24 has no alpha plane, so every
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

impl<R> Nv24Sink for MixedSinker<'_, Nv24, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv24, R> {
  type Input<'r> = Nv24Row<'r>;
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

  fn process(&mut self, row: Nv24Row<'_>) -> Result<(), Self::Error> {
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
    // NV24 UV row is `2 * width` bytes. `checked_mul` covers the
    // boundary where `2 * width` could overflow `usize` on 32-bit
    // targets with very large widths.
    let uv_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.uv().len() != uv_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::UvFull,
        idx,
        uv_expected,
        row.uv().len(),
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

    // Non-identity plan: row-stage fused downscale (matches the Yuv444p
    // twin). Bin Y for luma; for colour, convert the interleaved
    // full-width UV source row to RGB with the fused `nv24_to_rgb_row`
    // kernel the identity path uses, then bin the RGB row.
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
          nv24_to_rgb_row(row.y(), row.uv(), scratch, w, matrix, full_range, use_simd);
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

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    // Standalone RGBA path: the caller wants only RGBA (no RGB / HSV),
    // so run the dedicated RGBA kernel directly into the output buffer.
    // Avoids both the scratch allocation and the expand-pad pass.
    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      nv24_to_rgba_row(
        row.y(),
        row.uv(),
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

    nv24_to_rgb_row(
      row.y(),
      row.uv(),
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

    // Strategy A: when both RGB-side and RGBA outputs are requested,
    // derive RGBA from the just-computed RGB row (memory-bound copy +
    // 0xFF alpha pad) instead of running a second YUV→RGB kernel.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Nv42 impl ----------------------------------------------------------
//
// Structurally identical to the Nv24 impl — the row primitive hides
// the V/U byte-order difference.

impl<'a, R> MixedSinker<'a, Nv42, R> {
  /// Attaches a packed 32‑bit RGBA output buffer.
  ///
  /// See [`MixedSinker::<Nv24>::with_rgba`] for the same rationale and
  /// constraints; Nv42 differs only in chroma byte order (V before U).
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

impl<R> Nv42Sink for MixedSinker<'_, Nv42, R> {}

impl<R> PixelSink for MixedSinker<'_, Nv42, R> {
  type Input<'r> = Nv42Row<'r>;
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

  fn process(&mut self, row: Nv42Row<'_>) -> Result<(), Self::Error> {
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
    let vu_expected =
      w.checked_mul(2)
        .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
          w, h, 2,
        )))?;
    if row.vu().len() != vu_expected {
      return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
        RowSlice::VuFull,
        idx,
        vu_expected,
        row.vu().len(),
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

    // Non-identity plan: row-stage fused downscale (matches the Yuv444p
    // twin). Convert the interleaved VU source row to RGB with the fused
    // `nv42_to_rgb_row` kernel the identity path uses, then bin it.
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
          nv42_to_rgb_row(row.y(), row.vu(), scratch, w, matrix, full_range, use_simd);
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

    let want_rgb = rgb.is_some();
    let want_rgba = rgba.is_some();
    let want_hsv = hsv.is_some();
    let need_rgb_kernel = want_rgb || want_hsv;

    if want_rgba && !need_rgb_kernel {
      let rgba_buf = rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      nv42_to_rgba_row(
        row.y(),
        row.vu(),
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

    nv42_to_rgb_row(
      row.y(),
      row.vu(),
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
