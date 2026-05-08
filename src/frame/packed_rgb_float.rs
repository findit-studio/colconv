use derive_more::IsVariant;
use thiserror::Error;

// ============================================================
// Tier 9 — Packed float RGB source-side frame (Rgbf32)
// ============================================================

/// Errors returned by [`Rgbf32Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Rgbf32FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 3 * width` `f32` elements. Each row needs `3 * width`
  /// `f32` samples for packed RGB float.
  #[error("stride ({stride}) is smaller than 3 * width ({min_stride}) f32 elements")]
  StrideTooSmall {
    /// Required minimum stride (`3 * width`) in `f32` elements.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` `f32` elements.
  #[error("RGBF32 plane has {actual} f32 elements but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum `f32` elements required.
    expected: usize,
    /// Actual `f32` elements supplied.
    actual: usize,
  },
  /// `stride * height` overflows `usize`.
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride that overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// `3 * width` overflows `u32`.
  #[error("3 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **RGBF32** frame.
/// One plane, 3 × `f32` per pixel, channel order `R, G, B`.
///
/// Values are **linear** RGB by convention — no gamma / OETF handling
/// is applied by `colconv`. Caller is responsible for applying any
/// gamma / OETF transforms before constructing the frame.
///
/// HDR values (> 1.0) are permitted in the buffer; output paths that
/// target u8 / u16 saturate them to the output range. Output paths
/// targeting `f32` (`with_rgb_f32`) preserve them losslessly.
///
/// `stride` is in **`f32` elements** (≥ `3 * width`), matching the
/// per-format convention that stride aligns with the underlying slice
/// element type. No width parity constraint.
///
/// # Endian contract — **LE-encoded bytes** (`AV_PIX_FMT_RGBF32LE`)
///
/// The `&[f32]` plane is the **LE-encoded byte layout** reinterpreted as
/// `f32`. This frame maps to FFmpeg `AV_PIX_FMT_RGBF32LE`. FFmpeg also
/// defines `AV_PIX_FMT_RGBF32BE` and an unsuffixed `AV_PIX_FMT_RGBF32`
/// alias that is **target-endian** (resolves to `RGBF32LE` on LE hosts and
/// `RGBF32BE` on BE hosts). **Callers on a BE host who hold target-endian
/// `AV_PIX_FMT_RGBF32` bytes must convert them to LE before constructing
/// this frame** — otherwise the LE-decode contract here would re-interpret
/// the BE bytes as LE and produce byte-swapped float data. The 4-channel
/// `AV_PIX_FMT_RGBAF32LE` / `AV_PIX_FMT_RGBAF32BE` pair follows the same
/// `*LE` convention; this frame uses the analogous LE binding.
///
/// On a little-endian host (every CI runner today) LE bytes _are_
/// host-native, so `&[f32]` is also a host-native float slice; on a
/// big-endian host the bytes have to be byte-swapped back to host-native
/// before arithmetic. Downstream row kernels handle this byte-swap (or
/// no-op on LE) under the hood — callers do **not** pre-swap.
///
/// Stride is in **f32 elements** (not bytes). Callers holding a byte buffer
/// from FFmpeg should cast via `bytemuck::cast_slice` and divide
/// `linesize[0]` by 4 before constructing.
#[derive(Debug, Clone, Copy)]
pub struct Rgbf32Frame<'a> {
  rgb: &'a [f32],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Rgbf32Frame<'a> {
  /// Constructs a new [`Rgbf32Frame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    rgb: &'a [f32],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Rgbf32FrameError> {
    if width == 0 || height == 0 {
      return Err(Rgbf32FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(3) {
      Some(v) => v,
      None => return Err(Rgbf32FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Rgbf32FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Rgbf32FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if rgb.len() < plane_min {
      return Err(Rgbf32FrameError::PlaneTooShort {
        expected: plane_min,
        actual: rgb.len(),
      });
    }
    Ok(Self {
      rgb,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Rgbf32Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(rgb: &'a [f32], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(rgb, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Rgbf32Frame dimensions or plane length"),
    }
  }

  /// Packed RGB plane (`R, G, B, R, G, B, …` per row, each value an
  /// `f32`). Length is at least `stride * height` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb(&self) -> &'a [f32] {
    self.rgb
  }
  /// Frame width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Stride in **`f32` elements** per row (`>= 3 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
