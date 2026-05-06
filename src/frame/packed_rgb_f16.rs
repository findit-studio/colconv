use derive_more::IsVariant;
use thiserror::Error;

// ============================================================
// Tier 9 — Packed half-precision float RGB source-side frame (Rgbf16)
// ============================================================

/// Errors returned by [`Rgbf16Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Rgbf16FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 3 * width` `f16` elements. Each row needs `3 * width`
  /// `f16` samples for packed RGB half-precision float.
  #[error("stride ({stride}) is smaller than 3 * width ({min_stride}) f16 elements")]
  StrideTooSmall {
    /// Required minimum stride (`3 * width`) in `f16` elements.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` `f16` elements.
  #[error("RGBF16 plane has {actual} f16 elements but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum `f16` elements required.
    expected: usize,
    /// Actual `f16` elements supplied.
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

/// A validated packed **RGBF16** frame (FFmpeg `AV_PIX_FMT_RGBF16`).
/// One plane, 3 × `f16` per pixel, channel order `R, G, B`.
///
/// Values are **linear** RGB by convention — no gamma / OETF handling
/// is applied by `colconv`. Caller is responsible for applying any
/// gamma / OETF transforms before constructing the frame.
///
/// HDR values (> 1.0) are permitted in the buffer; output paths that
/// target u8 / u16 saturate them to the output range. Output paths
/// targeting `f16` (`with_rgb_f16`) preserve them losslessly, and
/// output paths targeting `f32` (`with_rgb_f32`) widen them losslessly.
///
/// `stride` is in **`f16` elements** (≥ `3 * width`), matching the
/// per-format convention that stride aligns with the underlying slice
/// element type. No width parity constraint.
#[derive(Debug, Clone, Copy)]
pub struct Rgbf16Frame<'a> {
  rgb: &'a [half::f16],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Rgbf16Frame<'a> {
  /// Constructs a new [`Rgbf16Frame`], validating dimensions and
  /// plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    rgb: &'a [half::f16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Rgbf16FrameError> {
    if width == 0 || height == 0 {
      return Err(Rgbf16FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(3) {
      Some(v) => v,
      None => return Err(Rgbf16FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Rgbf16FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Rgbf16FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if rgb.len() < plane_min {
      return Err(Rgbf16FrameError::PlaneTooShort {
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

  /// Constructs a new [`Rgbf16Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(rgb: &'a [half::f16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(rgb, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Rgbf16Frame dimensions or plane length"),
    }
  }

  /// Packed RGB plane (`R, G, B, R, G, B, …` per row, each value a
  /// `half::f16`). Length is at least `stride * height` elements.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb(&self) -> &'a [half::f16] {
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
  /// Stride in **`f16` elements** per row (`>= 3 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
