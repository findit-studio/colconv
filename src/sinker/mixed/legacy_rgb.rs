//! Sinker impls for legacy 16-bit packed-RGB **source** formats (Tier 7).
//!
//! Sources covered:
//! - [`Rgb565`] — bits [15:11]=R5, [10:5]=G6, [4:0]=B5 (FFmpeg `RGB565LE`).
//! - [`Bgr565`] — bits [15:11]=B5, [10:5]=G6, [4:0]=R5 (FFmpeg `BGR565LE`).
//! - [`Rgb555`] — bits [14:10]=R5, [9:5]=G5, [4:0]=B5; bit 15 unused.
//! - [`Bgr555`] — bits [14:10]=B5, [9:5]=G5, [4:0]=R5; bit 15 unused.
//! - [`Rgb444`] — bits [11:8]=R4, [7:4]=G4, [3:0]=B4; bits [15:12] unused.
//! - [`Bgr444`] — bits [11:8]=B4, [7:4]=G4, [3:0]=R4; bits [15:12] unused.
//!
//! All six sources have **no** source alpha. Outputs map to the sink's
//! standard channels:
//!
//! - `with_rgb` / `with_rgba` — expand channels to u8 via bit-replication
//!   (`(c5 << 3) | (c5 >> 2)` for 5-bit, `(c6 << 2) | (c6 >> 4)` for 6-bit,
//!   `(c4 << 4) | c4` for 4-bit); `with_rgba` forces α=`0xFF`.
//! - `with_rgb_u16` — native bit-width, low-bit aligned in `u16`; no expansion.
//!   Max values: R5=G6=31/63 (RGB565), R5=G5=B5=31 (RGB555), R4=G4=B4=15 (RGB444).
//! - `with_rgba_u16` — same native-precision channels + α=`0xFFFF`.
//! - `with_luma` — stages u8 RGB via `rgb_to_luma_row`.
//! - `with_luma_u16` — zero-extended u8 luma (same `[0, 255]` range) via
//!   `rgb_to_luma_u16_row`; no native luma precision exists for these formats.
//! - `with_hsv` — stages u8 RGB via `rgb_to_hsv_row`.

use super::{
  GeometryOverflow, InsufficientBuffer, MixedSinker, MixedSinkerError, RowIndexOutOfRange,
  RowShapeMismatch, RowSlice, check_dimensions_match, packed_rgb_resample_stream,
  packed_rgb_u16_resample_preflight, rgb_row_buf_or_scratch, rgba_plane_row_slice,
  rgba_u16_plane_row_slice, source_rgb_scratch,
};
use crate::{
  PixelSink,
  resample::ResamplePlan,
  row::{
    bgr444_to_rgb_row, bgr444_to_rgb_u16_row, bgr444_to_rgba_row, bgr444_to_rgba_u16_row,
    bgr555_to_rgb_row, bgr555_to_rgb_u16_row, bgr555_to_rgba_row, bgr555_to_rgba_u16_row,
    bgr565_to_rgb_row, bgr565_to_rgb_u16_row, bgr565_to_rgba_row, bgr565_to_rgba_u16_row,
    rgb_to_hsv_row, rgb_to_luma_row, rgb_to_luma_u16_row, rgb444_to_rgb_row, rgb444_to_rgb_u16_row,
    rgb444_to_rgba_row, rgb444_to_rgba_u16_row, rgb555_to_rgb_row, rgb555_to_rgb_u16_row,
    rgb555_to_rgba_row, rgb555_to_rgba_u16_row, rgb565_to_rgb_row, rgb565_to_rgb_u16_row,
    rgb565_to_rgba_row, rgb565_to_rgba_u16_row,
  },
  source::{
    Bgr444, Bgr444Row, Bgr444Sink, Bgr555, Bgr555Row, Bgr555Sink, Bgr565, Bgr565Row, Bgr565Sink,
    HsvFrameMut, Rgb444, Rgb444Row, Rgb444Sink, Rgb555, Rgb555Row, Rgb555Sink, Rgb565, Rgb565Row,
    Rgb565Sink,
  },
};

// Shared helper: checked accessor for the u16 RGB plane row slice.

/// Slice out a `3 * width` `u16` sub-range from a flat u16 RGB plane.
/// Returns `Err(GeometryOverflow)` on 32-bit targets if `one_plane_end x 3`
/// wraps `usize`.
#[inline(always)]
fn rgb_u16_plane_row_slice(
  buf: &mut [u16],
  one_plane_start: usize,
  one_plane_end: usize,
  width: usize,
  height: usize,
) -> Result<&mut [u16], MixedSinkerError> {
  let end = one_plane_end
    .checked_mul(3)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width, height, 3,
    )))?;
  let start = one_plane_start * 3;
  Ok(&mut buf[start..end])
}

/// Grows `scratch` to an **out-width** re-packed source row (`2 * width`
/// bytes — one little-endian `u16` word per output pixel) and returns
/// the slice, following the planner's recoverable-allocation contract
/// (the exact reserve makes the resize incapable of reallocating;
/// refusal surfaces as `AllocationFailed`). The per-output-row re-pack
/// target for the legacy 16-bit packed-RGB resample tail.
#[cfg_attr(not(tarpaulin), inline(always))]
fn legacy_rgb_packed_scratch<'s>(
  scratch: &'s mut std::vec::Vec<u8>,
  width: usize,
  plan: &ResamplePlan,
) -> Result<&'s mut [u8], MixedSinkerError> {
  let row_bytes = width
    .checked_mul(2)
    .ok_or(MixedSinkerError::GeometryOverflow(GeometryOverflow::new(
      width,
      plan.out_h(),
      2,
    )))?;
  if scratch.len() < row_bytes {
    scratch
      .try_reserve_exact(row_bytes - scratch.len())
      .map_err(|_| {
        MixedSinkerError::Resample(crate::resample::ResampleError::AllocationFailed(
          crate::resample::PlanGeometry::new(
            plan.src_w(),
            plan.src_h(),
            plan.out_w(),
            plan.out_h(),
          ),
        ))
      })?;
    scratch.resize(row_bytes, 0);
  }
  Ok(&mut scratch[..row_bytes])
}

/// Feeds the prepared source-width **native-channel** R/G/B `u8` row
/// (each value `<= 63`, NOT bit-expanded) into the (already
/// sequence-checked) shared u8 area stream and derives every attached
/// output from each finalized **binned native** output row.
///
/// Because the stream's `u8` accumulation rounds the area mean half-up,
/// the binned channels are the source's native-depth area mean — the
/// signal the direct `rgb_u16` path exposes, not an 8-bit-expanded
/// approximation of it. Per finalized output row each binned native
/// pixel is re-packed (`repack`) back into the source's packed `u16`
/// word, and the **exact** direct `*_to_*` kernels run over that
/// re-packed row, so every output byte equals a direct conversion of the
/// area-downscaled source-format frame — the single-binned-frame
/// contract the `Rgb48` / `X2Rgb10` / `Gbrp16` paths follow.
///
/// `rgb_u16` / `rgba_u16` copy the native channels (their kernels
/// re-extract the native bits); `rgb` / `rgba` bit-expand them; `luma` /
/// `luma_u16` / `hsv` stage through the bit-expanded u8 RGB row (the
/// direct path's source-of-truth ordering). The u8 RGB staging row is
/// sized only when one of the outputs that reads it (`rgb` / `luma` /
/// `luma_u16` / `hsv`) is attached and `rgb` is absent, so a
/// `u16`-/`rgba`-only sink neither grows it nor risks its allocation.
#[allow(clippy::too_many_arguments)]
fn legacy_rgb_resample_emit(
  stream: &mut crate::resample::AreaStream<u8>,
  plan: &ResamplePlan,
  rgb: &mut Option<&mut [u8]>,
  rgba: &mut Option<&mut [u8]>,
  rgb_u16: &mut Option<&mut [u16]>,
  rgba_u16: &mut Option<&mut [u16]>,
  luma: &mut Option<&mut [u8]>,
  luma_u16: &mut Option<&mut [u16]>,
  hsv: &mut Option<HsvFrameMut<'_>>,
  src_native: &[u8],
  rgb_stage_scratch: &mut std::vec::Vec<u8>,
  packed_scratch: &mut std::vec::Vec<u8>,
  matrix: crate::ColorMatrix,
  full_range: bool,
  idx: usize,
  use_simd: bool,
  repack: impl Fn(u8, u8, u8) -> u16,
  to_rgb: fn(&[u8], &mut [u8], usize, bool),
  to_rgba: fn(&[u8], &mut [u8], usize, bool),
  to_rgb_u16: fn(&[u8], &mut [u16], usize, bool),
  to_rgba_u16: fn(&[u8], &mut [u16], usize, bool),
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  // Every attached output reads the re-packed source row, so it is
  // always sized once an output exists (the caller's preflight already
  // returned early for a no-output sink).
  let packed = legacy_rgb_packed_scratch(packed_scratch, ow, plan)?;
  // The bit-expanded u8 RGB row drives rgb / luma / luma_u16 / hsv. When
  // `rgb` is attached the kernel writes straight into it; otherwise these
  // outputs read a scratch. The scratch — and its allocation failure —
  // is risked only when one of them is attached AND `rgb` is absent, so a
  // `u16`-/`rgba`-only sink never grows it. The predicate gates both the
  // sizing here and the staging in the closure, so they cannot drift.
  let need_rgb_stage = rgb.is_none() && (luma.is_some() || luma_u16.is_some() || hsv.is_some());
  let stage: &mut [u8] = if need_rgb_stage {
    source_rgb_scratch(rgb_stage_scratch, ow, plan)?
  } else {
    &mut []
  };
  stream.feed_row(idx, src_native, use_simd, |oy, binned| {
    // Re-pack the binned native R/G/B channels back into the source's
    // packed little-endian `u16` word — the exact wire the direct kernels
    // consume, so each output below is byte-identical to a direct
    // conversion of the area-downscaled source-format frame.
    let prow = &mut packed[..2 * ow];
    for x in 0..ow {
      let word = repack(binned[x * 3], binned[x * 3 + 1], binned[x * 3 + 2]);
      let bytes = word.to_le_bytes();
      prow[x * 2] = bytes[0];
      prow[x * 2 + 1] = bytes[1];
    }
    // Native-depth u16 RGB / RGBA — the kernels re-extract the native
    // bits (no expansion), so these copy the binned native channels.
    if let Some(buf) = rgb_u16.as_deref_mut() {
      to_rgb_u16(prow, &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow], ow, use_simd);
    }
    if let Some(buf) = rgba_u16.as_deref_mut() {
      to_rgba_u16(prow, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow, use_simd);
    }
    // u8 RGBA — bit-expanded channels + forced alpha (exactly the direct
    // kernel), derived straight from the re-packed row.
    if let Some(buf) = rgba.as_deref_mut() {
      to_rgba(prow, &mut buf[oy * 4 * ow..(oy + 1) * 4 * ow], ow, use_simd);
    }
    // u8 RGB stage drives rgb / luma / luma_u16 / hsv. Write into the
    // user's `rgb` buffer when attached (zero extra copy); otherwise the
    // scratch sized above.
    if rgb.is_some() || need_rgb_stage {
      let rgb_row: &mut [u8] = match rgb.as_deref_mut() {
        Some(buf) => &mut buf[oy * 3 * ow..(oy + 1) * 3 * ow],
        None => &mut stage[..3 * ow],
      };
      to_rgb(prow, rgb_row, ow, use_simd);
      if let Some(buf) = luma.as_deref_mut() {
        rgb_to_luma_row(
          rgb_row,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(buf) = luma_u16.as_deref_mut() {
        rgb_to_luma_u16_row(
          rgb_row,
          &mut buf[oy * ow..(oy + 1) * ow],
          ow,
          matrix,
          full_range,
          use_simd,
        );
      }
      if let Some(hsv) = hsv.as_mut() {
        let (h, s, v) = hsv.hsv();
        rgb_to_hsv_row(
          rgb_row,
          &mut h[oy * ow..(oy + 1) * ow],
          &mut s[oy * ow..(oy + 1) * ow],
          &mut v[oy * ow..(oy + 1) * ow],
          ow,
          use_simd,
        );
      }
    }
  })?;
  Ok(())
}

// Macro: emit one complete sinker impl block for a legacy RGB format.
//
// Parameters:
//   $marker      — marker type (e.g. `Rgb565`)
//   $sink_trait  — Sink subtrait (e.g. `Rgb565Sink`)
//   $row_ty      — Row type (e.g. `Rgb565Row`)
//   $buf_field   — row accessor method (e.g. `rgb565`)
//   $row_slice   — `RowSlice` variant (e.g. `RowSlice::Rgb565Packed`)
//   $to_rgb      — rgb_row dispatcher fn
//   $to_rgba     — rgba_row dispatcher fn
//   $to_rgb_u16  — rgb_u16_row dispatcher fn
//   $to_rgba_u16 — rgba_u16_row dispatcher fn
//   $unpack      — `|px: u16| -> (u8, u8, u8)`: extract the **native**
//                  R, G, B channels (canonical R, G, B order; 5/6/5,
//                  5/5/5 or 4/4/4 values, NOT bit-expanded). The fused
//                  resample path bins these at native depth.
//   $repack      — `|r: u8, g: u8, b: u8| -> u16`: re-pack the binned
//                  native channels back into the **source's** packed
//                  little-endian word — the inverse of `$unpack`, so the
//                  source's own `$to_*` kernels re-extract them and every
//                  output equals a direct conversion of the binned frame.
macro_rules! impl_legacy_rgb_sinker {
  (
    marker:      $marker:ident,
    sink_trait:  $sink_trait:ident,
    row_ty:      $row_ty:ident,
    buf_field:   $buf_field:ident,
    row_slice:   $row_slice:expr,
    to_rgb:      $to_rgb:ident,
    to_rgba:     $to_rgba:ident,
    to_rgb_u16:  $to_rgb_u16:ident,
    to_rgba_u16: $to_rgba_u16:ident,
    unpack:      $unpack:expr,
    repack:      $repack:expr,
  ) => {
    // ---- per-format accessors ------------------------------------------------

    impl<'a, R> MixedSinker<'a, $marker, R> {
      /// Attaches a packed **8-bit** RGBA output buffer. Alpha is filled with
      /// constant `0xFF` (this source format has no alpha channel).
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

      /// Attaches a **native-depth `u16`** RGB output buffer. Each channel is
      /// stored low-bit aligned at its native bit width — no expansion applied.
      /// Length is measured in `u16` elements (`width x height x 3`).
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

      /// Attaches a **native-depth `u16`** RGBA output buffer. Same native
      /// bit-width channels as `with_rgb_u16` plus α=`0xFFFF` (the source
      /// has no alpha). Length is measured in `u16` elements
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

      /// Attaches a **`u16`** luma output buffer. Luma is derived from
      /// expanded u8 RGB via `rgb_to_luma_u16_row` (zero-extended `u8`
      /// result, range `[0, 255]`). No native luma precision exists for
      /// these formats. Length in `u16` elements (`width x height`).
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

    // ---- Sink subtrait -------------------------------------------------------
    //
    // R-generic (matching `Rgb24` / `Bgr24`): the sink subtrait and
    // `PixelSink` impl hold for any resampler `R`, so a
    // `MixedSinker<$marker, AreaResampler>` is a legal sink — without this
    // the legacy-RGB sinks would stay pinned to `NoopResampler` and the
    // fused path below would be unreachable.

    impl<R> $sink_trait for MixedSinker<'_, $marker, R> {}

    // ---- PixelSink ----------------------------------------------------------

    impl<R> PixelSink for MixedSinker<'_, $marker, R> {
      type Input<'r> = $row_ty<'r>;
      type Error = MixedSinkerError;

      fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
        check_dimensions_match(self.width, self.height, width, height)?;
        if let Some(stream) = self.rgb_stream.as_mut() {
          stream.reset();
        }
        self.resample_outputs = None;
        Ok(())
      }

      fn process(&mut self, row: $row_ty<'_>) -> Result<(), Self::Error> {
        let w = self.width;
        let h = self.height;
        let idx = row.row();
        let use_simd = self.simd;

        // Each pixel is 2 bytes (one LE u16 word).
        if row.$buf_field().len() != w * 2 {
          return Err(MixedSinkerError::RowShapeMismatch(RowShapeMismatch::new(
            $row_slice,
            idx,
            w * 2,
            row.$buf_field().len(),
          )));
        }
        if idx >= self.height {
          return Err(MixedSinkerError::RowIndexOutOfRange(
            RowIndexOutOfRange::new(idx, self.height),
          ));
        }

        // Non-identity plan: unpack the packed source row to its 3
        // **native** R/G/B channels (5/6/5, 5/5/5 or 4/4/4 values, NOT
        // bit-expanded), bin those at native depth through the shared u8
        // area stream, then re-pack each binned pixel and run the exact
        // direct `$to_*` kernels. Freeze the full output set and
        // sequence-check before staging so a no-output sink stays a no-op
        // and an out-of-sequence row is rejected without the allocation.
        if let Some(plan) = self.plan.as_ref() {
          let Self {
            rgb,
            rgb_u16,
            rgba,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            rgb_scratch,
            legacy_rgb_native_scratch,
            legacy_rgb_packed_scratch,
            rgb_stream,
            resample_outputs,
            ..
          } = self;
          if !packed_rgb_u16_resample_preflight(
            resample_outputs,
            rgb,
            rgba,
            luma,
            rgb_u16,
            rgba_u16,
            luma_u16,
            hsv,
            // Legacy 16-bit RGB bins its native 5/6/5 channels through the
            // u8 `packed_rgb_resample_stream`, so the sequence counter is
            // that u8 stream's (the row index is element-type-agnostic).
            rgb_stream.as_ref().map_or(0, |s| s.next_y()),
            idx,
          )? {
            return Ok(());
          }
          let stream = packed_rgb_resample_stream(rgb_stream, plan, idx)?;
          let native = source_rgb_scratch(legacy_rgb_native_scratch, w, plan)?;
          let src = row.$buf_field();
          let unpack = $unpack;
          for x in 0..w {
            let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
            let (r, g, b) = unpack(px);
            native[x * 3] = r;
            native[x * 3 + 1] = g;
            native[x * 3 + 2] = b;
          }
          return legacy_rgb_resample_emit(
            stream,
            plan,
            rgb,
            rgba,
            rgb_u16,
            rgba_u16,
            luma,
            luma_u16,
            hsv,
            native,
            rgb_scratch,
            legacy_rgb_packed_scratch,
            row.matrix(),
            row.full_range(),
            idx,
            use_simd,
            $repack,
            $to_rgb,
            $to_rgba,
            $to_rgb_u16,
            $to_rgba_u16,
          );
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
          ..
        } = self;
        let one_plane_start = idx * w;
        let one_plane_end = one_plane_start + w;
        let src = row.$buf_field();

        // ---- native u16 RGB output ----------------------------------------
        if let Some(buf) = rgb_u16.as_deref_mut() {
          let rgb_u16_row = rgb_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          $to_rgb_u16(src, rgb_u16_row, w, use_simd);
        }

        // ---- native u16 RGBA output (forces α=0xFFFF) ---------------------
        if let Some(buf) = rgba_u16.as_deref_mut() {
          let rgba_u16_row = rgba_u16_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          $to_rgba_u16(src, rgba_u16_row, w, use_simd);
        }

        // ---- u8 RGBA output (forces α=0xFF) --------------------------------
        // Dispatched via dedicated kernel — no RGB staging required.
        let want_rgb = rgb.is_some();
        let want_luma = luma.is_some();
        let want_luma_u16 = luma_u16.is_some();
        let want_hsv = hsv.is_some();
        let need_u8_rgb = want_rgb || want_luma || want_luma_u16 || want_hsv;

        if !need_u8_rgb {
          // Standalone RGBA fast path — write directly; avoid scratch alloc.
          if let Some(buf) = rgba.as_deref_mut() {
            let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
            $to_rgba(src, rgba_row, w, use_simd);
          }
          return Ok(());
        }

        // ---- u8 RGB staging (drives rgb / luma / luma_u16 / hsv) ----------
        let rgb_row = rgb_row_buf_or_scratch(
          rgb.as_deref_mut(),
          rgb_scratch,
          one_plane_start,
          one_plane_end,
          w,
          h,
        )?;
        $to_rgb(src, rgb_row, w, use_simd);

        if let Some(luma_buf) = luma.as_deref_mut() {
          rgb_to_luma_row(
            rgb_row,
            &mut luma_buf[one_plane_start..one_plane_end],
            w,
            row.matrix(),
            row.full_range(),
            use_simd,
          );
        }

        if let Some(luma_u16_buf) = luma_u16.as_deref_mut() {
          rgb_to_luma_u16_row(
            rgb_row,
            &mut luma_u16_buf[one_plane_start..one_plane_end],
            w,
            row.matrix(),
            row.full_range(),
            use_simd,
          );
        }

        if let Some(hsv_bufs) = hsv.as_mut() {
          let (h, s, v) = hsv_bufs.hsv();
          rgb_to_hsv_row(
            rgb_row,
            &mut h[one_plane_start..one_plane_end],
            &mut s[one_plane_start..one_plane_end],
            &mut v[one_plane_start..one_plane_end],
            w,
            use_simd,
          );
        }

        // RGBA u8 fan-out via dedicated kernel (not Strategy A — avoids
        // double-pass without a shared RGB→RGBA expand for these formats).
        if let Some(buf) = rgba.as_deref_mut() {
          let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
          $to_rgba(src, rgba_row, w, use_simd);
        }

        Ok(())
      }
    }
  };
}

// Six format instantiations.

impl_legacy_rgb_sinker! {
  marker:      Rgb565,
  sink_trait:  Rgb565Sink,
  row_ty:      Rgb565Row,
  buf_field:   rgb565,
  row_slice:   RowSlice::Rgb565Packed,
  to_rgb:      rgb565_to_rgb_row,
  to_rgba:     rgb565_to_rgba_row,
  to_rgb_u16:  rgb565_to_rgb_u16_row,
  to_rgba_u16: rgb565_to_rgba_u16_row,
  // R5 [15:11], G6 [10:5], B5 [4:0] — inverse of `rgb565_to_rgb_u16_row`.
  unpack: |px: u16| -> (u8, u8, u8) {
    (
      ((px >> 11) & 0x1F) as u8,
      ((px >> 5) & 0x3F) as u8,
      (px & 0x1F) as u8,
    )
  },
  repack: |r: u8, g: u8, b: u8| -> u16 {
    ((r as u16) << 11) | ((g as u16) << 5) | (b as u16)
  },
}

impl_legacy_rgb_sinker! {
  marker:      Bgr565,
  sink_trait:  Bgr565Sink,
  row_ty:      Bgr565Row,
  buf_field:   bgr565,
  row_slice:   RowSlice::Bgr565Packed,
  to_rgb:      bgr565_to_rgb_row,
  to_rgba:     bgr565_to_rgba_row,
  to_rgb_u16:  bgr565_to_rgb_u16_row,
  to_rgba_u16: bgr565_to_rgba_u16_row,
  // B5 [15:11], G6 [10:5], R5 [4:0] — inverse of `bgr565_to_rgb_u16_row`
  // (output is canonical R, G, B).
  unpack: |px: u16| -> (u8, u8, u8) {
    (
      (px & 0x1F) as u8,
      ((px >> 5) & 0x3F) as u8,
      ((px >> 11) & 0x1F) as u8,
    )
  },
  repack: |r: u8, g: u8, b: u8| -> u16 {
    ((b as u16) << 11) | ((g as u16) << 5) | (r as u16)
  },
}

impl_legacy_rgb_sinker! {
  marker:      Rgb555,
  sink_trait:  Rgb555Sink,
  row_ty:      Rgb555Row,
  buf_field:   rgb555,
  row_slice:   RowSlice::Rgb555Packed,
  to_rgb:      rgb555_to_rgb_row,
  to_rgba:     rgb555_to_rgba_row,
  to_rgb_u16:  rgb555_to_rgb_u16_row,
  to_rgba_u16: rgb555_to_rgba_u16_row,
  // R5 [14:10], G5 [9:5], B5 [4:0], bit 15 unused — inverse of
  // `rgb555_to_rgb_u16_row`.
  unpack: |px: u16| -> (u8, u8, u8) {
    (
      ((px >> 10) & 0x1F) as u8,
      ((px >> 5) & 0x1F) as u8,
      (px & 0x1F) as u8,
    )
  },
  repack: |r: u8, g: u8, b: u8| -> u16 {
    ((r as u16) << 10) | ((g as u16) << 5) | (b as u16)
  },
}

impl_legacy_rgb_sinker! {
  marker:      Bgr555,
  sink_trait:  Bgr555Sink,
  row_ty:      Bgr555Row,
  buf_field:   bgr555,
  row_slice:   RowSlice::Bgr555Packed,
  to_rgb:      bgr555_to_rgb_row,
  to_rgba:     bgr555_to_rgba_row,
  to_rgb_u16:  bgr555_to_rgb_u16_row,
  to_rgba_u16: bgr555_to_rgba_u16_row,
  // B5 [14:10], G5 [9:5], R5 [4:0], bit 15 unused — inverse of
  // `bgr555_to_rgb_u16_row` (output is canonical R, G, B).
  unpack: |px: u16| -> (u8, u8, u8) {
    (
      (px & 0x1F) as u8,
      ((px >> 5) & 0x1F) as u8,
      ((px >> 10) & 0x1F) as u8,
    )
  },
  repack: |r: u8, g: u8, b: u8| -> u16 {
    ((b as u16) << 10) | ((g as u16) << 5) | (r as u16)
  },
}

impl_legacy_rgb_sinker! {
  marker:      Rgb444,
  sink_trait:  Rgb444Sink,
  row_ty:      Rgb444Row,
  buf_field:   rgb444,
  row_slice:   RowSlice::Rgb444Packed,
  to_rgb:      rgb444_to_rgb_row,
  to_rgba:     rgb444_to_rgba_row,
  to_rgb_u16:  rgb444_to_rgb_u16_row,
  to_rgba_u16: rgb444_to_rgba_u16_row,
  // R4 [11:8], G4 [7:4], B4 [3:0], bits [15:12] unused — inverse of
  // `rgb444_to_rgb_u16_row`.
  unpack: |px: u16| -> (u8, u8, u8) {
    (
      ((px >> 8) & 0x0F) as u8,
      ((px >> 4) & 0x0F) as u8,
      (px & 0x0F) as u8,
    )
  },
  repack: |r: u8, g: u8, b: u8| -> u16 {
    ((r as u16) << 8) | ((g as u16) << 4) | (b as u16)
  },
}

impl_legacy_rgb_sinker! {
  marker:      Bgr444,
  sink_trait:  Bgr444Sink,
  row_ty:      Bgr444Row,
  buf_field:   bgr444,
  row_slice:   RowSlice::Bgr444Packed,
  to_rgb:      bgr444_to_rgb_row,
  to_rgba:     bgr444_to_rgba_row,
  to_rgb_u16:  bgr444_to_rgb_u16_row,
  to_rgba_u16: bgr444_to_rgba_u16_row,
  // B4 [11:8], G4 [7:4], R4 [3:0], bits [15:12] unused — inverse of
  // `bgr444_to_rgb_u16_row` (output is canonical R, G, B).
  unpack: |px: u16| -> (u8, u8, u8) {
    (
      (px & 0x0F) as u8,
      ((px >> 4) & 0x0F) as u8,
      ((px >> 8) & 0x0F) as u8,
    )
  },
  repack: |r: u8, g: u8, b: u8| -> u16 {
    ((b as u16) << 8) | ((g as u16) << 4) | (r as u16)
  },
}
