//! Validated source frame for the 8-bit indexed-color (`AV_PIX_FMT_PAL8`) format.
//!
//! Each pixel is a `u8` index into a 256-entry BGRA palette carried
//! alongside the pixel buffer. See [`Pal8Frame::try_new`] for layout details.

use derive_more::IsVariant;
use thiserror::Error;

/// Error returned by [`Pal8Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error, IsVariant)]
pub enum Pal8FrameError {
  /// `width` or `height` was zero.
  #[error("zero width or height: {width}×{height}")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `stride < width`.
  #[error("stride {stride} < width {width}")]
  StrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied plane stride.
    stride: u32,
  },
  /// `stride * height` overflows `usize` (can only fire on 32-bit targets).
  #[error("geometry overflow: stride {stride} × rows {rows} exceeds usize")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
  /// Pixel data is shorter than `stride * height` bytes.
  #[error("pixel data too short: expected >= {expected} bytes, got {actual}")]
  PlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
}

/// A validated 8-bit indexed-color (`AV_PIX_FMT_PAL8`) source frame.
///
/// `data` holds one `u8` index per pixel (row-major, with `stride`
/// bytes per row). `palette` is the 256-entry BGRA lookup table:
/// each entry is `[B, G, R, A]` per FFmpeg's `AV_PIX_FMT_PAL8`
/// convention.
#[derive(Debug, Clone, Copy)]
pub struct Pal8Frame<'a> {
  data: &'a [u8],
  palette: &'a [[u8; 4]; 256],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a> Pal8Frame<'a> {
  /// Constructs and validates a [`Pal8Frame`].
  ///
  /// Returns [`Pal8FrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - `stride < width`,
  /// - `stride * height` overflows `usize`, or
  /// - `data.len() < stride * height`.
  ///
  /// The `palette` slice is always exactly 256 entries — Rust's type
  /// system enforces this; no runtime palette-length check is needed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    data: &'a [u8],
    palette: &'a [[u8; 4]; 256],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Pal8FrameError> {
    if width == 0 || height == 0 {
      return Err(Pal8FrameError::ZeroDimension { width, height });
    }
    if stride < width {
      return Err(Pal8FrameError::StrideTooSmall { width, stride });
    }
    let min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Pal8FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if data.len() < min {
      return Err(Pal8FrameError::PlaneTooShort {
        expected: min,
        actual: data.len(),
      });
    }
    Ok(Self {
      data,
      palette,
      width,
      height,
      stride,
    })
  }

  /// The pixel index buffer. Row `r` starts at byte offset `r * stride()`.
  pub fn data(&self) -> &'a [u8] {
    self.data
  }

  /// The 256-entry BGRA palette. Each entry is `[B, G, R, A]` per
  /// FFmpeg's `AV_PIX_FMT_PAL8` convention.
  pub fn palette(&self) -> &'a [[u8; 4]; 256] {
    self.palette
  }

  /// Frame width in pixels.
  pub const fn width(&self) -> u32 {
    self.width
  }

  /// Frame height in pixels.
  pub const fn height(&self) -> u32 {
    self.height
  }

  /// Byte stride of the pixel plane (`>= width`).
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}
