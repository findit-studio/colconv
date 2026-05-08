//! Sinker impls for the half-float planar GBR source family (Tier 10 f16).
//!
//! Two formats covered in this file:
//! - [`Gbrpf16`] (`AV_PIX_FMT_GBRPF16LE`) — three planes (G, B, R), `half::f16`,
//!   no alpha.
//! - [`Gbrapf16`] (`AV_PIX_FMT_GBRAPF16LE`) — four planes (G, B, R, A),
//!   `half::f16`, real per-pixel α.
//!
//! # Output paths
//!
//! - `with_rgb` / `with_rgba` — delegate to `gbrpf16_to_rgb_row` /
//!   `gbrpf16_to_rgba_row` (dispatcher handles f16 → f32 widening internally
//!   where no fp16/F16C SIMD is available).
//! - `with_rgb_f16` / `with_rgba_f16` — lossless f16 interleave via
//!   `gbrpf16_to_rgb_f16_row` / `gbrpf16_to_rgba_f16_row`; no conversion.
//! - `with_rgb_f32` / `with_rgba_f32` — widen f16 → f32 per-row (using a
//!   `Vec`-backed scratch buffer grown lazily), then call `gbrpf32_to_rgb_f32_row`
//!   / `gbrpf32_to_rgba_f32_row`.
//! - `with_rgb_u16` / `with_rgba_u16` — same widen + `gbrpf32_to_rgb_u16_row`
//!   / `gbrpf32_to_rgba_u16_row`.
//! - `with_luma` / `with_luma_u16` — same widen + `gbrpf32_to_luma_row` /
//!   `gbrpf32_to_luma_u16_row`.
//! - `with_hsv` — same widen + `gbrpf32_to_hsv_row`.
//!
//! For `Gbrapf16`, RGBA outputs use real source α from the A plane:
//! - `with_rgba` — `gbrapf16` → u8 RGBA: combo with `with_rgb` uses Strategy
//!   A+ (expand RGB u8 → RGBA u8, then `copy_alpha_plane_f16_to_u8` overwrites
//!   slot 3 from the source α plane narrowed/scaled).
//! - `with_rgba_f16` — `gbrapf16_to_rgba_f16_row` (lossless source α).
//! - `with_rgba_f32` / `with_rgba_u16` — widen + gbrapf32 kernels with real α.
//!
//! # F16→F32 widening scratch
//!
//! The `MixedSinker` for f16 formats uses the existing `rgb_scratch` heap-
//! allocated buffer re-purposed as a `u8` byte region **plus** three inline
//! f32 scratch slices obtained via `Vec<f32>` grown on demand (analogous to
//! `rgb_scratch` for u8). However, to keep the struct generic and avoid new
//! fields, the sinker widens f16 → f32 into a fresh per-row stack chunk
//! (`const CHUNK: usize = 64`) and calls gbrpf32 dispatchers in strided
//! chunks — identical to the widening pattern used by the dispatch layer
//! for backends without native fp16/F16C support.

use super::{
  MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match, rgb_row_buf_or_scratch,
  rgba_plane_row_slice,
};
use crate::{
  ColorMatrix, PixelSink,
  row::{
    expand_rgb_to_rgba_row, gbrapf16_to_rgba_f16_row, gbrapf32_to_rgba_f32_row,
    gbrapf32_to_rgba_u16_row, gbrpf16_to_rgb_f16_row, gbrpf16_to_rgb_row, gbrpf16_to_rgba_f16_row,
    gbrpf16_to_rgba_row, gbrpf32_to_hsv_row, gbrpf32_to_luma_row, gbrpf32_to_luma_u16_row,
    gbrpf32_to_rgb_f32_row, gbrpf32_to_rgb_u16_row, gbrpf32_to_rgba_f32_row,
    gbrpf32_to_rgba_u16_row, scalar::alpha_extract::copy_alpha_plane_f32_to_u8,
  },
  yuv::{Gbrapf16, Gbrapf16Row, Gbrapf16Sink, Gbrpf16, Gbrpf16Row, Gbrpf16Sink},
};

// Float-planar GBR sources are already component RGB (no chroma matrix).
// BT.709 full-range is the conventional default for luma derivation.
const GBR_F16_LUMA_MATRIX: ColorMatrix = ColorMatrix::Bt709;
const GBR_F16_FULL_RANGE: bool = true;

// Chunk size for the inline f16→f32 widening scratch arrays (stack-allocated).
const WIDEN_CHUNK: usize = 64;

/// `BE` value that makes the `gbrpf16_to_*` / `gbrapf16_to_*` row dispatchers
/// (and the widened `gbrpf32_to_*` chain after `widen_f16_to_f32`) treat
/// their input as **host-native** (a no-op byte-swap).
///
/// [`crate::frame::Gbrpf16Frame`] / [`crate::frame::Gbrapf16Frame`] expose
/// `&[half::f16]` plane rows in **host-native** layout — the API contract
/// is that the caller hands us already-decoded half-floats. The kernel `BE`
/// parameter, however, names the **encoded** byte order (so `BE = false`
/// means "decode LE-encoded bytes" via `u16::from_le`). On a LE host the
/// host-native layout is LE, so `BE = false` is correct; on a BE host the
/// host-native layout is BE, so we must request `BE = true` to make
/// `u16::from_be` no-op the swap. Without this routing the loaders would
/// byte-swap an already-decoded host-native `f16` on BE hosts, corrupting
/// every output path (codex PR #84 Finding 3).
///
/// Crucially, the **widened f32 chain** must also use `HOST_NATIVE_BE`:
/// after [`widen_f16_to_f32`] (which calls `half::f16::to_f32` on host-native
/// f16 bits) the scratch is host-native f32, so the downstream
/// `gbrpf32_to_*` kernel's `from_le`/`from_be` loader must be a no-op —
/// achieved by routing with `HOST_NATIVE_BE`.
///
/// This is the **sinker-layer** complement to the SIMD-backend-internal
/// `HOST_NATIVE_BE` introduced in `c3a6478` and the `Rgbf16` sinker fix in
/// `dcf40a3`. Same truth table:
///
///   • LE host: `HOST_NATIVE_BE = false` → `from_le` (no-op on LE) → correct.
///   • BE host: `HOST_NATIVE_BE = true`  → `from_be` (no-op on BE) → correct.
///
/// The α-plane scatter for [`Gbrapf16`] (Strategy A+ / standalone-RGBA)
/// widens the host-native f16 α plane to host-native f32 via
/// [`widen_f16_to_f32`] then calls `copy_alpha_plane_f32_to_u8` — both
/// operations are endian-agnostic. Mix-mode corruption (LE-decoded RGB +
/// host-native α) is therefore eliminated by routing the RGB chain via
/// `HOST_NATIVE_BE`.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Widen `width` `half::f16` values from `src` into `dst` (f32 elements).
///
/// The source slice is `&[half::f16]` in **host-native** layout (per the
/// `Gbrpf16Frame` / `Gbrapf16Frame` API contract); `to_f32` interprets the
/// bits as host-native and emits host-native `f32`. Downstream `gbrpf32_to_*`
/// callers must therefore route with [`HOST_NATIVE_BE`] (not the encoded
/// `BE`) to avoid double byte-swapping.
#[cfg_attr(not(tarpaulin), inline(always))]
fn widen_f16_to_f32(src: &[half::f16], dst: &mut [f32], count: usize) {
  for i in 0..count {
    dst[i] = src[i].to_f32();
  }
}

// ---- Gbrpf16 accessor impl block ----------------------------------------

impl<'a> MixedSinker<'a, Gbrpf16> {
  /// Attaches a packed **8-bit** RGBA output buffer. α is forced to `0xFF`
  /// (Gbrpf16 has no alpha channel). Length in bytes (`width × height × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer. Each f16 channel is
  /// widened to f32, clamped to `[0, 1]`, and scaled × 65535.
  /// Length in `u16` elements (`width × height × 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGBA output buffer. Same full-range scaling
  /// (× 65535) as `with_rgb_u16`; α is constant `0xFFFF`.
  /// Length in `u16` elements (`width × height × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`f32`** RGB output buffer. f16 channels are widened
  /// to f32 — lossless (f16 ⊂ f32). HDR and special values preserved.
  /// Length in `f32` elements (`width × height × 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f32`](Self::with_rgb_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbF32BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`f32`** RGBA output buffer. f16 widened to f32;
  /// α is constant `1.0f32`. Length in `f32` elements (`width × height × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_f32`](Self::with_rgba_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaF32BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`half::f16`** RGB output buffer. Lossless planar →
  /// packed interleave — no conversion. HDR values, NaN, and Inf preserved
  /// bit-exact. Length in `half::f16` elements (`width × height × 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f16`](Self::with_rgb_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbF16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_f16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`half::f16`** RGBA output buffer. Lossless planar →
  /// packed interleave with constant α = `half::f16::from_f32(1.0)`.
  /// Length in `half::f16` elements (`width × height × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_f16`](Self::with_rgba_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaF16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_f16 = Some(buf);
    Ok(self)
  }

  /// Attaches a `u16` luma output buffer. f16 channels are widened to f32,
  /// then luma is derived (clamp + round-half-up) and zero-extended to u16.
  /// Length in `u16` elements (`width × height`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_luma_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::LumaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

impl Gbrpf16Sink for MixedSinker<'_, Gbrpf16> {}

impl PixelSink for MixedSinker<'_, Gbrpf16> {
  type Input<'r> = Gbrpf16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Gbrpf16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth row-shape checks before any unsafe kernel.
    if row.g().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::GbrF16Plane,
        row: idx,
        expected: w,
        actual: row.g().len(),
      });
    }
    if row.b().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::GbrF16Plane,
        row: idx,
        expected: w,
        actual: row.b().len(),
      });
    }
    if row.r().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::GbrF16Plane,
        row: idx,
        expected: w,
        actual: row.r().len(),
      });
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: h,
      });
    }

    let g_in = row.g();
    let b_in = row.b();
    let r_in = row.r();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // ---- Lossless f16 native pass-through (no conversion) ----------------

    if let Some(buf) = self.rgb_f16.as_deref_mut() {
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf16_to_rgb_f16_row::<HOST_NATIVE_BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    if let Some(buf) = self.rgba_f16.as_deref_mut() {
      let start = one_plane_start * 4;
      let end = one_plane_end
        .checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 4,
        })?;
      gbrpf16_to_rgba_f16_row::<HOST_NATIVE_BE>(
        g_in,
        b_in,
        r_in,
        &mut buf[start..end],
        w,
        use_simd,
      );
    }

    // ---- Paths that require widening f16 → f32 ---------------------------
    //
    // Use a chunk-based inline scratch to avoid heap allocation per row.
    // The chunk size of 64 matches the dispatch layer's widening pattern.
    // When no f32/u16/luma/HSV outputs are attached this block is a no-op.

    let need_wide = self.rgb_f32.is_some()
      || self.rgba_f32.is_some()
      || self.rgb_u16.is_some()
      || self.rgba_u16.is_some()
      || self.luma.is_some()
      || self.luma_u16.is_some()
      || self.hsv.is_some();

    if need_wide {
      let mut gf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut bf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut rf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut offset = 0;
      while offset < w {
        let n = (w - offset).min(WIDEN_CHUNK);
        widen_f16_to_f32(&g_in[offset..], &mut gf_chunk, n);
        widen_f16_to_f32(&b_in[offset..], &mut bf_chunk, n);
        widen_f16_to_f32(&r_in[offset..], &mut rf_chunk, n);
        let gf = &gf_chunk[..n];
        let bf = &bf_chunk[..n];
        let rf = &rf_chunk[..n];

        let chunk_plane_start = one_plane_start + offset;
        let chunk_plane_end = chunk_plane_start + n;

        if let Some(buf) = self.rgb_f32.as_deref_mut() {
          let start = chunk_plane_start * 3;
          let end = chunk_plane_end * 3;
          gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgba_f32.as_deref_mut() {
          let start = chunk_plane_start * 4;
          let end = chunk_plane_end * 4;
          gbrpf32_to_rgba_f32_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgb_u16.as_deref_mut() {
          let start = chunk_plane_start * 3;
          let end = chunk_plane_end * 3;
          gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgba_u16.as_deref_mut() {
          let start = chunk_plane_start * 4;
          let end = chunk_plane_end * 4;
          gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.luma.as_deref_mut() {
          gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut buf[chunk_plane_start..chunk_plane_end],
            n,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            use_simd,
          );
        }

        if let Some(buf) = self.luma_u16.as_deref_mut() {
          gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut buf[chunk_plane_start..chunk_plane_end],
            n,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            use_simd,
          );
        }

        if let Some(hsv) = self.hsv.as_mut() {
          gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut hsv.h[chunk_plane_start..chunk_plane_end],
            &mut hsv.s[chunk_plane_start..chunk_plane_end],
            &mut hsv.v[chunk_plane_start..chunk_plane_end],
            n,
            use_simd,
          );
        }

        offset += n;
      }
    }

    // ---- u8 RGBA standalone fast path ------------------------------------

    let want_rgba = self.rgba.is_some();
    let want_rgb = self.rgb.is_some();
    let need_u8_rgb = want_rgb;

    if want_rgba && !need_u8_rgb {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      gbrpf16_to_rgba_row::<HOST_NATIVE_BE>(g_in, b_in, r_in, rgba_row, w, use_simd);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba {
      return Ok(());
    }

    // ---- Stage u8 RGB once for RGBA fan-out ------------------------------

    let Self {
      rgb,
      rgba,
      rgb_scratch,
      ..
    } = self;
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    gbrpf16_to_rgb_row::<HOST_NATIVE_BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

    // Strategy A: expand RGB → RGBA (constant α = 0xFF).
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
    }

    Ok(())
  }
}

// ---- Gbrapf16 accessor impl block ----------------------------------------

impl<'a> MixedSinker<'a, Gbrapf16> {
  /// Attaches a packed **8-bit** RGBA output buffer. α is sourced from the
  /// A plane (real per-pixel α, clamped to `[0, 1]` and scaled × 255).
  /// Length in bytes (`width × height × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba(mut self, buf: &'a mut [u8]) -> Result<Self, MixedSinkerError> {
    self.set_rgba(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba`](Self::with_rgba).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba(&mut self, buf: &'a mut [u8]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaBufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGB output buffer. Widened f16 → f32,
  /// clamped, scaled × 65535. Length in `u16` elements (`width × height × 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_u16`](Self::with_rgb_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`u16`** RGBA output buffer. Source α widened f16 → f32,
  /// clamped × 65535. Length in `u16` elements (`width × height × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_u16`](Self::with_rgba_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_u16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`f32`** RGB output buffer. f16 widened to f32
  /// (lossless). Length in `f32` elements (`width × height × 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f32`](Self::with_rgb_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbF32BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`f32`** RGBA output buffer. Source α widened f16 → f32
  /// (lossless; HDR, NaN, Inf preserved). Length in `f32` elements
  /// (`width × height × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_f32(mut self, buf: &'a mut [f32]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_f32(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_f32`](Self::with_rgba_f32).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_f32(&mut self, buf: &'a mut [f32]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaF32BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_f32 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`half::f16`** RGB output buffer. Lossless f16
  /// interleave. Length in `half::f16` elements (`width × height × 3`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgb_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgb_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgb_f16`](Self::with_rgb_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgb_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(3)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbF16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgb_f16 = Some(buf);
    Ok(self)
  }

  /// Attaches a packed **`half::f16`** RGBA output buffer. Source α is passed
  /// through losslessly (HDR, NaN, Inf preserved bit-exact).
  /// Length in `half::f16` elements (`width × height × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_rgba_f16(mut self, buf: &'a mut [half::f16]) -> Result<Self, MixedSinkerError> {
    self.set_rgba_f16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_rgba_f16`](Self::with_rgba_f16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_rgba_f16(&mut self, buf: &'a mut [half::f16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(4)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::RgbaF16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.rgba_f16 = Some(buf);
    Ok(self)
  }

  /// Attaches a `u16` luma output buffer. f16 channels widened to f32, then
  /// luma derived (clamp + round-half-up) and zero-extended to u16.
  /// Length in `u16` elements (`width × height`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_luma_u16(mut self, buf: &'a mut [u16]) -> Result<Self, MixedSinkerError> {
    self.set_luma_u16(buf)?;
    Ok(self)
  }
  /// In-place variant of [`with_luma_u16`](Self::with_luma_u16).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_luma_u16(&mut self, buf: &'a mut [u16]) -> Result<&mut Self, MixedSinkerError> {
    let expected = self.frame_bytes(1)?;
    if buf.len() < expected {
      return Err(MixedSinkerError::LumaU16BufferTooShort {
        expected,
        actual: buf.len(),
      });
    }
    self.luma_u16 = Some(buf);
    Ok(self)
  }
}

impl Gbrapf16Sink for MixedSinker<'_, Gbrapf16> {}

impl PixelSink for MixedSinker<'_, Gbrapf16> {
  type Input<'r> = Gbrapf16Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Gbrapf16Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    // Defense-in-depth row-shape checks before any unsafe kernel.
    if row.g().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::GbrF16Plane,
        row: idx,
        expected: w,
        actual: row.g().len(),
      });
    }
    if row.b().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::GbrF16Plane,
        row: idx,
        expected: w,
        actual: row.b().len(),
      });
    }
    if row.r().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::GbrF16Plane,
        row: idx,
        expected: w,
        actual: row.r().len(),
      });
    }
    if row.a().len() != w {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::GbrF16Plane,
        row: idx,
        expected: w,
        actual: row.a().len(),
      });
    }
    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: h,
      });
    }

    let g_in = row.g();
    let b_in = row.b();
    let r_in = row.r();
    let a_in = row.a();
    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    // ---- Lossless f16 native paths (no conversion) -----------------------

    if let Some(buf) = self.rgb_f16.as_deref_mut() {
      // rgb_f16: no source α — use the no-α kernel (lossless scatter).
      let start = one_plane_start * 3;
      let end = one_plane_end * 3;
      gbrpf16_to_rgb_f16_row::<HOST_NATIVE_BE>(g_in, b_in, r_in, &mut buf[start..end], w, use_simd);
    }

    if let Some(buf) = self.rgba_f16.as_deref_mut() {
      // rgba_f16: source α included losslessly via gbrapf16_to_rgba_f16_row.
      let start = one_plane_start * 4;
      let end = one_plane_end
        .checked_mul(4)
        .ok_or(MixedSinkerError::GeometryOverflow {
          width: w,
          height: h,
          channels: 4,
        })?;
      gbrapf16_to_rgba_f16_row::<HOST_NATIVE_BE>(
        g_in,
        b_in,
        r_in,
        a_in,
        &mut buf[start..end],
        w,
        use_simd,
      );
    }

    // ---- Paths that require widening f16 → f32 ---------------------------
    //
    // For Gbrapf16, RGBA outputs include the widened source α. RGB-only
    // outputs (rgb_f32, rgb_u16) discard α.

    let need_wide = self.rgb_f32.is_some()
      || self.rgba_f32.is_some()
      || self.rgb_u16.is_some()
      || self.rgba_u16.is_some()
      || self.luma.is_some()
      || self.luma_u16.is_some()
      || self.hsv.is_some();

    if need_wide {
      let mut gf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut bf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut rf_chunk = [0.0f32; WIDEN_CHUNK];
      let mut af_chunk = [0.0f32; WIDEN_CHUNK];
      let mut offset = 0;
      while offset < w {
        let n = (w - offset).min(WIDEN_CHUNK);
        widen_f16_to_f32(&g_in[offset..], &mut gf_chunk, n);
        widen_f16_to_f32(&b_in[offset..], &mut bf_chunk, n);
        widen_f16_to_f32(&r_in[offset..], &mut rf_chunk, n);
        widen_f16_to_f32(&a_in[offset..], &mut af_chunk, n);
        let gf = &gf_chunk[..n];
        let bf = &bf_chunk[..n];
        let rf = &rf_chunk[..n];
        let af = &af_chunk[..n];

        let chunk_plane_start = one_plane_start + offset;
        let chunk_plane_end = chunk_plane_start + n;

        if let Some(buf) = self.rgb_f32.as_deref_mut() {
          let start = chunk_plane_start * 3;
          let end = chunk_plane_end * 3;
          gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgba_f32.as_deref_mut() {
          // gbrapf32_to_rgba_f32_row with widened source α (lossless).
          let start = chunk_plane_start * 4;
          let end = chunk_plane_end * 4;
          gbrapf32_to_rgba_f32_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            af,
            &mut buf[start..end],
            n,
            use_simd,
          );
        }

        if let Some(buf) = self.rgb_u16.as_deref_mut() {
          let start = chunk_plane_start * 3;
          let end = chunk_plane_end * 3;
          gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(gf, bf, rf, &mut buf[start..end], n, use_simd);
        }

        if let Some(buf) = self.rgba_u16.as_deref_mut() {
          // gbrapf32_to_rgba_u16_row with widened source α.
          let start = chunk_plane_start * 4;
          let end = chunk_plane_end * 4;
          gbrapf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            af,
            &mut buf[start..end],
            n,
            use_simd,
          );
        }

        if let Some(buf) = self.luma.as_deref_mut() {
          gbrpf32_to_luma_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut buf[chunk_plane_start..chunk_plane_end],
            n,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            use_simd,
          );
        }

        if let Some(buf) = self.luma_u16.as_deref_mut() {
          gbrpf32_to_luma_u16_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut buf[chunk_plane_start..chunk_plane_end],
            n,
            GBR_F16_LUMA_MATRIX,
            GBR_F16_FULL_RANGE,
            use_simd,
          );
        }

        if let Some(hsv) = self.hsv.as_mut() {
          gbrpf32_to_hsv_row::<HOST_NATIVE_BE>(
            gf,
            bf,
            rf,
            &mut hsv.h[chunk_plane_start..chunk_plane_end],
            &mut hsv.s[chunk_plane_start..chunk_plane_end],
            &mut hsv.v[chunk_plane_start..chunk_plane_end],
            n,
            use_simd,
          );
        }

        offset += n;
      }
    }

    // ---- u8 RGBA standalone fast path (source α from f16 A plane) --------
    //
    // For standalone RGBA without other u8-dependent outputs, widen the α
    // plane to f32 per-row and use the existing copy_alpha_plane_f32_to_u8.
    // The chroma planes are converted via gbrpf16_to_rgba_row first (which
    // writes opaque α = 0xFF), then α is overwritten from the source.

    let want_rgba = self.rgba.is_some();
    let want_rgb = self.rgb.is_some();
    let need_u8_rgb = want_rgb;

    if want_rgba && !need_u8_rgb {
      let rgba_buf = self.rgba.as_deref_mut().unwrap();
      let rgba_row = rgba_plane_row_slice(rgba_buf, one_plane_start, one_plane_end, w, h)?;
      // Write opaque RGB → RGBA (α = 0xFF), then overwrite α from source.
      gbrpf16_to_rgba_row::<HOST_NATIVE_BE>(g_in, b_in, r_in, rgba_row, w, use_simd);
      // Scatter f16 α → u8 slot 3: widen + clamp + scale.
      widen_and_scatter_f16_alpha_to_u8(a_in, rgba_row, w);
      return Ok(());
    }

    if !need_u8_rgb && !want_rgba {
      return Ok(());
    }

    // ---- Stage u8 RGB once for RGBA fan-out ------------------------------

    let Self {
      rgb,
      rgba,
      rgb_scratch,
      ..
    } = self;
    let rgb_row = rgb_row_buf_or_scratch(
      rgb.as_deref_mut(),
      rgb_scratch,
      one_plane_start,
      one_plane_end,
      w,
      h,
    )?;
    gbrpf16_to_rgb_row::<HOST_NATIVE_BE>(g_in, b_in, r_in, rgb_row, w, use_simd);

    // Strategy A+: expand RGB → RGBA (0xFF stub), then overwrite α from source.
    if let Some(buf) = rgba.as_deref_mut() {
      let rgba_row = rgba_plane_row_slice(buf, one_plane_start, one_plane_end, w, h)?;
      expand_rgb_to_rgba_row(rgb_row, rgba_row, w);
      widen_and_scatter_f16_alpha_to_u8(a_in, rgba_row, w);
    }

    Ok(())
  }
}

/// Widen a `half::f16` α plane to f32, clamp to `[0, 1]`, scale × 255,
/// and scatter into the RGBA slot 3 of a u8 RGBA buffer.
///
/// Used by `Gbrapf16` Strategy A+ and standalone-RGBA paths to overwrite
/// the per-pixel alpha byte from the f16 source α plane.
#[cfg_attr(not(tarpaulin), inline(always))]
fn widen_and_scatter_f16_alpha_to_u8(alpha_f16: &[half::f16], rgba_out: &mut [u8], width: usize) {
  let mut af_chunk = [0.0f32; WIDEN_CHUNK];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(WIDEN_CHUNK);
    widen_f16_to_f32(&alpha_f16[offset..], &mut af_chunk, n);
    copy_alpha_plane_f32_to_u8(&af_chunk[..n], &mut rgba_out[offset * 4..], n);
    offset += n;
  }
}
