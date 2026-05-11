//! Packed 10-bit RGB source frames:
//! - `AV_PIX_FMT_X2RGB10{LE,BE}` → [`X2Rgb10Frame`] — 32-bit word
//!   `(MSB) 2X | 10R | 10G | 10B (LSB)`.
//! - `AV_PIX_FMT_X2BGR10{LE,BE}` → [`X2Bgr10Frame`] — 32-bit word
//!   `(MSB) 2X | 10B | 10G | 10R (LSB)`.
//!
//! # Endian contract — `<const BE: bool = false>`
//!
//! Each frame type carries a `<const BE: bool>` parameter that defaults to
//! `false` (LE-encoded `u32` words, matching the FFmpeg `*LE` suffix). Set
//! `BE = true` to consume `*BE`-encoded plane bytes; row kernels perform the
//! byte-swap (or no-op) under the hood — callers do **not** pre-swap.
//!
//! `stride` is in **bytes** (not `u32` words). Plane length must be at least
//! `stride * height` bytes.

use derive_more::{IsVariant, TryUnwrap, Unwrap};
use thiserror::Error;

/// Errors returned by [`X2Rgb10Frame::try_new`]. Variant shape mirrors
/// the [`RgbaFrameError`] family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, TryUnwrap, Unwrap, Error)]
#[non_exhaustive]
pub enum X2Rgb10FrameError {
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `stride < 4 * width`.
  #[error("stride ({stride}) is smaller than 4 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// Plane is shorter than `stride * height` bytes.
  #[error("X2RGB10 plane has {actual} bytes but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `stride * height` overflows `usize`.
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride that overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `4 * width` overflows `u32`.
  #[error("4 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **X2RGB10** frame
/// (`AV_PIX_FMT_X2RGB10{LE,BE}`) — 4 bytes per pixel, 32-bit
/// word with `(MSB) 2X | 10R | 10G | 10B (LSB)`. The top 2 bits
/// are **ignored padding**; each colour channel carries 10 active
/// bits.
///
/// The `<const BE: bool>` parameter selects the plane byte order:
/// `false` (default) → LE-encoded words (`AV_PIX_FMT_X2RGB10LE`),
/// `true` → BE-encoded words (`AV_PIX_FMT_X2RGB10BE`).
///
/// `stride` is in **bytes** (≥ `4 * width`). No width parity
/// constraint.
///
/// # Aliases
/// - [`X2Rgb10LeFrame`] = `X2Rgb10Frame<'a, false>`.
/// - [`X2Rgb10BeFrame`] = `X2Rgb10Frame<'a, true>`.
#[derive(Debug, Clone, Copy)]
pub struct X2Rgb10Frame<'a, const BE: bool = false> {
  x2rgb10: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

/// LE-encoded `X2Rgb10Frame` (`AV_PIX_FMT_X2RGB10LE`).
pub type X2Rgb10LeFrame<'a> = X2Rgb10Frame<'a, false>;

/// BE-encoded `X2Rgb10Frame` (`AV_PIX_FMT_X2RGB10BE`).
pub type X2Rgb10BeFrame<'a> = X2Rgb10Frame<'a, true>;

impl<'a, const BE: bool> X2Rgb10Frame<'a, BE> {
  /// Constructs a new [`X2Rgb10Frame`], validating dimensions and
  /// plane length. `<const BE: bool>` selects LE (`false`, default)
  /// vs BE (`true`) plane byte order; row kernels perform the
  /// byte-swap.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    x2rgb10: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, X2Rgb10FrameError> {
    if width == 0 || height == 0 {
      return Err(X2Rgb10FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(X2Rgb10FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(X2Rgb10FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(X2Rgb10FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if x2rgb10.len() < plane_min {
      return Err(X2Rgb10FrameError::PlaneTooShort {
        expected: plane_min,
        actual: x2rgb10.len(),
      });
    }
    Ok(Self {
      x2rgb10,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`X2Rgb10Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(x2rgb10: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(x2rgb10, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid X2Rgb10Frame dimensions or plane length"),
    }
  }

  /// Packed X2RGB10 plane bytes — each 4-byte group is one
  /// `u32` word in the byte order selected by `<const BE: bool>`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn x2rgb10(&self) -> &'a [u8] {
    self.x2rgb10
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
  /// Byte stride (`>= 4 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
  /// Runtime mirror of the `<const BE: bool>` type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}

/// Errors returned by [`X2Bgr10Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, TryUnwrap, Unwrap, Error)]
#[non_exhaustive]
pub enum X2Bgr10FrameError {
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `stride < 4 * width`.
  #[error("stride ({stride}) is smaller than 4 * width ({min_stride})")]
  StrideTooSmall {
    /// Required minimum stride.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// Plane is shorter than `stride * height` bytes.
  #[error("X2BGR10 plane has {actual} bytes but at least {expected} are required")]
  PlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `stride * height` overflows `usize`.
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride that overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `4 * width` overflows `u32`.
  #[error("4 * width overflows u32 ({width} too large)")]
  WidthOverflow {
    /// The supplied width.
    width: u32,
  },
}

/// A validated packed **X2BGR10** frame
/// (`AV_PIX_FMT_X2BGR10{LE,BE}`) — 4 bytes per pixel, 32-bit
/// word with `(MSB) 2X | 10B | 10G | 10R (LSB)`. Channel positions
/// are reversed relative to [`X2Rgb10Frame`].
///
/// The `<const BE: bool>` parameter selects the plane byte order; see
/// [`X2Rgb10Frame`] for the contract.
///
/// # Aliases
/// - [`X2Bgr10LeFrame`] = `X2Bgr10Frame<'a, false>`.
/// - [`X2Bgr10BeFrame`] = `X2Bgr10Frame<'a, true>`.
#[derive(Debug, Clone, Copy)]
pub struct X2Bgr10Frame<'a, const BE: bool = false> {
  x2bgr10: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

/// LE-encoded `X2Bgr10Frame` (`AV_PIX_FMT_X2BGR10LE`).
pub type X2Bgr10LeFrame<'a> = X2Bgr10Frame<'a, false>;

/// BE-encoded `X2Bgr10Frame` (`AV_PIX_FMT_X2BGR10BE`).
pub type X2Bgr10BeFrame<'a> = X2Bgr10Frame<'a, true>;

impl<'a, const BE: bool> X2Bgr10Frame<'a, BE> {
  /// Constructs a new [`X2Bgr10Frame`], validating dimensions and
  /// plane length. `<const BE: bool>` selects LE (`false`, default)
  /// vs BE (`true`) plane byte order; row kernels perform the
  /// byte-swap.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    x2bgr10: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, X2Bgr10FrameError> {
    if width == 0 || height == 0 {
      return Err(X2Bgr10FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(v) => v,
      None => return Err(X2Bgr10FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(X2Bgr10FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(X2Bgr10FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if x2bgr10.len() < plane_min {
      return Err(X2Bgr10FrameError::PlaneTooShort {
        expected: plane_min,
        actual: x2bgr10.len(),
      });
    }
    Ok(Self {
      x2bgr10,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`X2Bgr10Frame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(x2bgr10: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(x2bgr10, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid X2Bgr10Frame dimensions or plane length"),
    }
  }

  /// Packed X2BGR10 plane bytes — each 4-byte group is one
  /// `u32` word in the byte order selected by `<const BE: bool>`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn x2bgr10(&self) -> &'a [u8] {
    self.x2bgr10
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
  /// Byte stride (`>= 4 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
  /// Runtime mirror of the `<const BE: bool>` type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}
