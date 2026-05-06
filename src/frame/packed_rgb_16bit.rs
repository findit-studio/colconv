//! Packed 16-bit RGB/BGR and RGBA/BGRA source frames:
//! - `AV_PIX_FMT_RGB48LE`  → [`Rgb48Frame`]  (R, G, B; stride in u16 elements ≥ 3 × width)
//! - `AV_PIX_FMT_BGR48LE`  → [`Bgr48Frame`]  (B, G, R; stride in u16 elements ≥ 3 × width)
//! - `AV_PIX_FMT_RGBA64LE` → [`Rgba64Frame`] (R, G, B, A; stride in u16 elements ≥ 4 × width)
//! - `AV_PIX_FMT_BGRA64LE` → [`Bgra64Frame`] (B, G, R, A; stride in u16 elements ≥ 4 × width)
//!
//! Stride is in **u16 elements** (not bytes). Plane slice is `&[u16]`.
//! On little-endian hosts the slice can be cast directly from the FFmpeg byte buffer.

use derive_more::IsVariant;
use thiserror::Error;

// ---- Rgb48Frame --------------------------------------------------------------

/// Errors returned by [`Rgb48Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Rgb48FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 3 * width` (in u16 elements).
  #[error("stride ({stride}) is smaller than 3 * width ({min_stride}) u16 elements")]
  StrideTooSmall {
    /// Required minimum stride in u16 elements.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` u16 elements.
  #[error("RGB48 plane has {actual} u16 elements but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum u16 elements required.
    expected: usize,
    /// Actual u16 elements supplied.
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

/// A validated packed **RGB48** frame (`AV_PIX_FMT_RGB48LE`) — three `u16`
/// samples per pixel in `R, G, B` order. Each `u16` is a native little-endian
/// sample; the caller is responsible for casting the raw FFmpeg byte buffer.
///
/// `stride` is in **u16 elements** (≥ `3 * width`).
#[derive(Debug, Clone, Copy)]
pub struct Rgb48Frame<'a> {
  rgb48: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Rgb48Frame<'a> {
  /// Constructs a new [`Rgb48Frame`], validating dimensions and plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    rgb48: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Rgb48FrameError> {
    if width == 0 || height == 0 {
      return Err(Rgb48FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(3) {
      Some(v) => v,
      None => return Err(Rgb48FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Rgb48FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Rgb48FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if rgb48.len() < plane_min {
      return Err(Rgb48FrameError::PlaneTooShort {
        expected: plane_min,
        actual: rgb48.len(),
      });
    }
    Ok(Self {
      rgb48,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Rgb48Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(rgb48: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(rgb48, width, height, stride) {
      Ok(f) => f,
      Err(_) => panic!("invalid Rgb48Frame dimensions or plane length"),
    }
  }

  /// Packed RGB48 plane — `width * 3` u16 elements per row (`R, G, B` per pixel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgb48(&self) -> &'a [u16] {
    self.rgb48
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
  /// Stride in u16 elements (≥ `3 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

// ---- Bgr48Frame --------------------------------------------------------------

/// Errors returned by [`Bgr48Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Bgr48FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 3 * width` (in u16 elements).
  #[error("stride ({stride}) is smaller than 3 * width ({min_stride}) u16 elements")]
  StrideTooSmall {
    /// Required minimum stride in u16 elements.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` u16 elements.
  #[error("BGR48 plane has {actual} u16 elements but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum u16 elements required.
    expected: usize,
    /// Actual u16 elements supplied.
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

/// A validated packed **BGR48** frame (`AV_PIX_FMT_BGR48LE`) — three `u16`
/// samples per pixel in `B, G, R` order. Channel order is reversed relative
/// to [`Rgb48Frame`]; stride convention and element type are identical.
///
/// `stride` is in **u16 elements** (≥ `3 * width`).
#[derive(Debug, Clone, Copy)]
pub struct Bgr48Frame<'a> {
  bgr48: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Bgr48Frame<'a> {
  /// Constructs a new [`Bgr48Frame`], validating dimensions and plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    bgr48: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Bgr48FrameError> {
    if width == 0 || height == 0 {
      return Err(Bgr48FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(3) {
      Some(v) => v,
      None => return Err(Bgr48FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Bgr48FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Bgr48FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if bgr48.len() < plane_min {
      return Err(Bgr48FrameError::PlaneTooShort {
        expected: plane_min,
        actual: bgr48.len(),
      });
    }
    Ok(Self {
      bgr48,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Bgr48Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(bgr48: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(bgr48, width, height, stride) {
      Ok(f) => f,
      Err(_) => panic!("invalid Bgr48Frame dimensions or plane length"),
    }
  }

  /// Packed BGR48 plane — `width * 3` u16 elements per row (`B, G, R` per pixel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bgr48(&self) -> &'a [u16] {
    self.bgr48
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
  /// Stride in u16 elements (≥ `3 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

// ---- Rgba64Frame -------------------------------------------------------------

/// Errors returned by [`Rgba64Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Rgba64FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 4 * width` (in u16 elements).
  #[error("stride ({stride}) is smaller than 4 * width ({min_stride}) u16 elements")]
  StrideTooSmall {
    /// Required minimum stride in u16 elements.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` u16 elements.
  #[error("RGBA64 plane has {actual} u16 elements but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum u16 elements required.
    expected: usize,
    /// Actual u16 elements supplied.
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
  /// `4 * width` overflows `u32`.
  #[error("4 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **RGBA64** frame (`AV_PIX_FMT_RGBA64LE`) — four `u16`
/// samples per pixel in `R, G, B, A` order. The alpha channel is real
/// (not padding) and is passed through by `with_rgba` / `with_rgba_u16`.
///
/// `stride` is in **u16 elements** (≥ `4 * width`).
#[derive(Debug, Clone, Copy)]
pub struct Rgba64Frame<'a> {
  rgba64: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Rgba64Frame<'a> {
  /// Constructs a new [`Rgba64Frame`], validating dimensions and plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    rgba64: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Rgba64FrameError> {
    if width == 0 || height == 0 {
      return Err(Rgba64FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(Rgba64FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Rgba64FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Rgba64FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if rgba64.len() < plane_min {
      return Err(Rgba64FrameError::PlaneTooShort {
        expected: plane_min,
        actual: rgba64.len(),
      });
    }
    Ok(Self {
      rgba64,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Rgba64Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(rgba64: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(rgba64, width, height, stride) {
      Ok(f) => f,
      Err(_) => panic!("invalid Rgba64Frame dimensions or plane length"),
    }
  }

  /// Packed RGBA64 plane — `width * 4` u16 elements per row (`R, G, B, A` per pixel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn rgba64(&self) -> &'a [u16] {
    self.rgba64
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
  /// Stride in u16 elements (≥ `4 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

// ---- Bgra64Frame -------------------------------------------------------------

/// Errors returned by [`Bgra64Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Bgra64FrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < 4 * width` (in u16 elements).
  #[error("stride ({stride}) is smaller than 4 * width ({min_stride}) u16 elements")]
  StrideTooSmall {
    /// Required minimum stride in u16 elements.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` u16 elements.
  #[error("BGRA64 plane has {actual} u16 elements but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum u16 elements required.
    expected: usize,
    /// Actual u16 elements supplied.
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
  /// `4 * width` overflows `u32`.
  #[error("4 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **BGRA64** frame (`AV_PIX_FMT_BGRA64LE`) — four `u16`
/// samples per pixel in `B, G, R, A` order. Channel order is reversed on the
/// first three elements relative to [`Rgba64Frame`]; alpha at position 3 is
/// real and is passed through by `with_rgba` / `with_rgba_u16`.
///
/// `stride` is in **u16 elements** (≥ `4 * width`).
#[derive(Debug, Clone, Copy)]
pub struct Bgra64Frame<'a> {
  bgra64: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Bgra64Frame<'a> {
  /// Constructs a new [`Bgra64Frame`], validating dimensions and plane length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    bgra64: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Bgra64FrameError> {
    if width == 0 || height == 0 {
      return Err(Bgra64FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(Bgra64FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Bgra64FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Bgra64FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if bgra64.len() < plane_min {
      return Err(Bgra64FrameError::PlaneTooShort {
        expected: plane_min,
        actual: bgra64.len(),
      });
    }
    Ok(Self {
      bgra64,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Bgra64Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(bgra64: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(bgra64, width, height, stride) {
      Ok(f) => f,
      Err(_) => panic!("invalid Bgra64Frame dimensions or plane length"),
    }
  }

  /// Packed BGRA64 plane — `width * 4` u16 elements per row (`B, G, R, A` per pixel).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bgra64(&self) -> &'a [u16] {
    self.bgra64
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
  /// Stride in u16 elements (≥ `4 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
