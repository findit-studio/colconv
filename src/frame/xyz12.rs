//! Tier 12 — Packed 12-bit CIE XYZ source frames (`AV_PIX_FMT_XYZ12LE` /
//! `AV_PIX_FMT_XYZ12BE`).
//!
//! Despite the inventory doc note "3 planes (X / Y / Z)", FFmpeg's actual
//! `AV_PIX_FMT_XYZ12LE` descriptor is **packed**: one stream of `u16`
//! triples in `X, Y, Z` order, low-packed (samples in low 12 bits, upper
//! 4 bits zero). This matches the DCDM JPEG2000 cinema container format
//! that decoders like OpenJPEG expand into.
//!
//! # Stride semantics
//!
//! **Stride is in samples (`u16` elements)**, not bytes. Each row needs
//! at least `3 * width` u16 samples. Callers with a raw FFmpeg byte
//! buffer should cast via [`bytemuck::cast_slice`] (which checks
//! alignment) and divide `linesize[0]` by 2.
//!
//! # Sample-value validation
//!
//! `try_new` validates geometry only. Out-of-range samples (upper 4
//! bits set) are masked by `& 0x0FFF` inside every kernel — matches
//! `Yuv420pFrame16` / `GbrpHighBitFrame` precedent (scanning every
//! sample at video rates is prohibitive).
//!
//! # Endianness
//!
//! The const-generic `BE: bool` parameter selects whether the wire-format
//! u16 samples are little-endian (`BE = false`) or big-endian (`BE =
//! true`). Type aliases [`Xyz12LeFrame`] and [`Xyz12BeFrame`] cover the
//! two FFmpeg variants. The byte-swap is a compile-time const branch in
//! every row kernel; the `BE = false` path is a no-op.

use derive_more::IsVariant;
use thiserror::Error;

/// Errors returned by [`Xyz12Frame::try_new`].
///
/// Variant shape mirrors [`super::Rgbf32FrameError`] but with all sizes
/// expressed in **samples** (`u16` elements) instead of bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Xyz12FrameError {
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
    /// Required minimum stride (`3 * width`) in u16 elements.
    min_stride: u32,
    /// The supplied stride.
    stride: u32,
  },
  /// Plane is shorter than `stride * height` u16 elements.
  #[error("XYZ12 plane has {actual} u16 elements but at least {expected} are required")]
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

/// A validated packed **XYZ12** frame (`AV_PIX_FMT_XYZ12LE` /
/// `AV_PIX_FMT_XYZ12BE`).
///
/// Each pixel occupies 3 × `u16` (six bytes), in **`X, Y, Z`** order.
/// Samples are 12-bit values stored in the low 12 bits of each `u16`
/// (upper 4 bits zero per the SMPTE ST 428-1 spec). Out-of-range
/// samples are tolerated at construction time and masked inside every
/// row kernel.
///
/// `stride` is in **u16 elements** (≥ `3 * width`), matching the
/// per-format convention that stride aligns with the underlying slice
/// element type. No width parity constraint.
///
/// The `BE: bool` const parameter selects little-endian (`false`) or
/// big-endian (`true`) wire-format encoding of each `u16`. Use the
/// type aliases [`Xyz12LeFrame`] / [`Xyz12BeFrame`] at call sites.
#[derive(Debug, Clone, Copy)]
pub struct Xyz12Frame<'a, const BE: bool = false> {
  xyz: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

impl<'a, const BE: bool> Xyz12Frame<'a, BE> {
  /// Constructs a new [`Xyz12Frame`], validating dimensions and plane
  /// length.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    xyz: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Xyz12FrameError> {
    if width == 0 || height == 0 {
      return Err(Xyz12FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(3) {
      Some(v) => v,
      None => return Err(Xyz12FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Xyz12FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(Xyz12FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if xyz.len() < plane_min {
      return Err(Xyz12FrameError::PlaneTooShort {
        expected: plane_min,
        actual: xyz.len(),
      });
    }
    Ok(Self {
      xyz,
      width,
      height,
      stride,
    })
  }

  /// Constructs a new [`Xyz12Frame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(xyz: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(xyz, width, height, stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid Xyz12Frame dimensions or plane length"),
    }
  }

  /// Packed `X, Y, Z` plane — `width * 3` u16 elements per row.
  /// Samples are in the low 12 bits of each `u16`; upper bits are
  /// masked by every row kernel.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn xyz(&self) -> &'a [u16] {
    self.xyz
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
  /// Stride in **u16 elements** (≥ `3 * width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
  /// Returns whether wire-format `u16` samples are big-endian (`true`)
  /// or little-endian (`false`). Mirrors the const-generic parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn big_endian(&self) -> bool {
    BE
  }
}

/// Type alias for a validated packed XYZ12 frame, **little-endian**
/// wire format (`AV_PIX_FMT_XYZ12LE`).
pub type Xyz12LeFrame<'a> = Xyz12Frame<'a, false>;

/// Type alias for a validated packed XYZ12 frame, **big-endian**
/// wire format (`AV_PIX_FMT_XYZ12BE`).
pub type Xyz12BeFrame<'a> = Xyz12Frame<'a, true>;
