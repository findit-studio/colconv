//! 8-bit planar GBR (`AV_PIX_FMT_GBRP`) and planar GBRA
//! (`AV_PIX_FMT_GBRAP`) source frames.
//!
//! Both formats are *planar RGB* — three (or four) full-resolution byte
//! planes, one per channel, in **G, B, R** order (FFmpeg convention).
//! `Gbrap` adds a fourth full-resolution alpha plane.
//!
//! Per-row sizing: each plane is `width × height` bytes, with byte
//! stride ≥ width per plane. No chroma subsampling (planar RGB has none
//! by definition), so widths and heights have no parity constraint.

use derive_more::IsVariant;
use thiserror::Error;

/// A validated 8-bit planar GBR frame (`AV_PIX_FMT_GBRP`).
///
/// Three planes, all full-size, in **G, B, R** order:
/// - `g` — green plane.
/// - `b` — blue plane.
/// - `r` — red plane.
///
/// Plane order matches FFmpeg's `AV_PIX_FMT_GBRP` convention. Each
/// plane requires `*_stride >= width` and length `>= *_stride * height`.
/// No width / height parity constraint (planar RGB has no chroma
/// subsampling).
///
/// Canonical for screen-codec workflows (libx264 with `-pix_fmt gbrp`),
/// some VFX intermediates, and the *RGB-source* output of the JPEG2000
/// decoder.
#[derive(Debug, Clone, Copy)]
pub struct GbrpFrame<'a> {
  g: &'a [u8],
  b: &'a [u8],
  r: &'a [u8],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
}

impl<'a> GbrpFrame<'a> {
  /// Constructs a new [`GbrpFrame`], validating dimensions and plane
  /// lengths. Returns [`GbrpFrameError`] if any of:
  /// - `width` or `height` is zero,
  /// - any stride is smaller than `width`,
  /// - any plane is too short to cover its declared rows,
  /// - declared geometry overflows `usize` (32-bit targets only).
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    g: &'a [u8],
    b: &'a [u8],
    r: &'a [u8],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
  ) -> Result<Self, GbrpFrameError> {
    if width == 0 || height == 0 {
      return Err(GbrpFrameError::ZeroDimension { width, height });
    }
    if g_stride < width {
      return Err(GbrpFrameError::GStrideTooSmall { width, g_stride });
    }
    if b_stride < width {
      return Err(GbrpFrameError::BStrideTooSmall { width, b_stride });
    }
    if r_stride < width {
      return Err(GbrpFrameError::RStrideTooSmall { width, r_stride });
    }

    let g_min = match (g_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrpFrameError::GeometryOverflow {
          stride: g_stride,
          rows: height,
        });
      }
    };
    if g.len() < g_min {
      return Err(GbrpFrameError::GPlaneTooShort {
        expected: g_min,
        actual: g.len(),
      });
    }
    let b_min = match (b_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrpFrameError::GeometryOverflow {
          stride: b_stride,
          rows: height,
        });
      }
    };
    if b.len() < b_min {
      return Err(GbrpFrameError::BPlaneTooShort {
        expected: b_min,
        actual: b.len(),
      });
    }
    let r_min = match (r_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrpFrameError::GeometryOverflow {
          stride: r_stride,
          rows: height,
        });
      }
    };
    if r.len() < r_min {
      return Err(GbrpFrameError::RPlaneTooShort {
        expected: r_min,
        actual: r.len(),
      });
    }

    Ok(Self {
      g,
      b,
      r,
      width,
      height,
      g_stride,
      b_stride,
      r_stride,
    })
  }

  /// Constructs a new [`GbrpFrame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    g: &'a [u8],
    b: &'a [u8],
    r: &'a [u8],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
  ) -> Self {
    match Self::try_new(g, b, r, width, height, g_stride, b_stride, r_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid GbrpFrame dimensions or plane lengths"),
    }
  }

  /// Green plane bytes. Row `r` starts at byte offset `r * g_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> &'a [u8] {
    self.g
  }
  /// Blue plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> &'a [u8] {
    self.b
  }
  /// Red plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> &'a [u8] {
    self.r
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
  /// Byte stride of the green plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g_stride(&self) -> u32 {
    self.g_stride
  }
  /// Byte stride of the blue plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b_stride(&self) -> u32 {
    self.b_stride
  }
  /// Byte stride of the red plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r_stride(&self) -> u32 {
    self.r_stride
  }

  // ---- crate-internal Y/U/V aliases ------------------------------------
  //
  // The shared `walker!` macro uses fixed `y/u/v` field-name conventions
  // when emitting walker bodies (`src.y()`, `src.u_stride()`, etc.). To
  // reuse the macro verbatim for planar GBR — whose externally-correct
  // accessor names are `g/b/r` — we expose `pub(crate)` aliases here:
  // `y == g`, `u == b`, `v == r` (matching the walker macro's planar3
  // contract). The aliases stay crate-private so the public API remains
  // the externally-correct GBR names.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn y(&self) -> &'a [u8] {
    self.g
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn u(&self) -> &'a [u8] {
    self.b
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn v(&self) -> &'a [u8] {
    self.r
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn y_stride(&self) -> u32 {
    self.g_stride
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn u_stride(&self) -> u32 {
    self.b_stride
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn v_stride(&self) -> u32 {
    self.r_stride
  }
}

/// Errors returned by [`GbrpFrame::try_new`].
///
/// Variant shape mirrors [`super::Yuv444pFrameError`] — same full-width
/// per-plane validation, no width-parity constraint — but with `G` /
/// `B` / `R` plane names instead of `Y` / `U` / `V`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum GbrpFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `g_stride < width`.
  #[error("g_stride ({g_stride}) is smaller than width ({width})")]
  GStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied G-plane stride.
    g_stride: u32,
  },
  /// `b_stride < width`.
  #[error("b_stride ({b_stride}) is smaller than width ({width})")]
  BStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied B-plane stride.
    b_stride: u32,
  },
  /// `r_stride < width`.
  #[error("r_stride ({r_stride}) is smaller than width ({width})")]
  RStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied R-plane stride.
    r_stride: u32,
  },
  /// G plane is shorter than `g_stride * height` bytes.
  #[error("G plane has {actual} bytes but at least {expected} are required")]
  GPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// B plane is shorter than `b_stride * height` bytes.
  #[error("B plane has {actual} bytes but at least {expected} are required")]
  BPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// R plane is shorter than `r_stride * height` bytes.
  #[error("R plane has {actual} bytes but at least {expected} are required")]
  RPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize` (32-bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}

/// A validated 8-bit planar GBR frame with alpha (`AV_PIX_FMT_GBRAP`).
///
/// Four planes, all full-size, in **G, B, R, A** order:
/// - `g` / `b` / `r` — colour planes.
/// - `a` — alpha plane (1:1 with G; real per-pixel alpha, not padding).
///
/// Plane order and structure match FFmpeg's `AV_PIX_FMT_GBRAP`. Each
/// plane requires `*_stride >= width` and length `>= *_stride * height`.
/// No width / height parity constraint.
#[derive(Debug, Clone, Copy)]
pub struct GbrapFrame<'a> {
  g: &'a [u8],
  b: &'a [u8],
  r: &'a [u8],
  a: &'a [u8],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
  a_stride: u32,
}

impl<'a> GbrapFrame<'a> {
  /// Constructs a new [`GbrapFrame`], validating dimensions and plane
  /// lengths.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    g: &'a [u8],
    b: &'a [u8],
    r: &'a [u8],
    a: &'a [u8],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
    a_stride: u32,
  ) -> Result<Self, GbrapFrameError> {
    if width == 0 || height == 0 {
      return Err(GbrapFrameError::ZeroDimension { width, height });
    }
    if g_stride < width {
      return Err(GbrapFrameError::GStrideTooSmall { width, g_stride });
    }
    if b_stride < width {
      return Err(GbrapFrameError::BStrideTooSmall { width, b_stride });
    }
    if r_stride < width {
      return Err(GbrapFrameError::RStrideTooSmall { width, r_stride });
    }
    if a_stride < width {
      return Err(GbrapFrameError::AStrideTooSmall { width, a_stride });
    }

    let g_min = match (g_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrapFrameError::GeometryOverflow {
          stride: g_stride,
          rows: height,
        });
      }
    };
    if g.len() < g_min {
      return Err(GbrapFrameError::GPlaneTooShort {
        expected: g_min,
        actual: g.len(),
      });
    }
    let b_min = match (b_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrapFrameError::GeometryOverflow {
          stride: b_stride,
          rows: height,
        });
      }
    };
    if b.len() < b_min {
      return Err(GbrapFrameError::BPlaneTooShort {
        expected: b_min,
        actual: b.len(),
      });
    }
    let r_min = match (r_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrapFrameError::GeometryOverflow {
          stride: r_stride,
          rows: height,
        });
      }
    };
    if r.len() < r_min {
      return Err(GbrapFrameError::RPlaneTooShort {
        expected: r_min,
        actual: r.len(),
      });
    }
    let a_min = match (a_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrapFrameError::GeometryOverflow {
          stride: a_stride,
          rows: height,
        });
      }
    };
    if a.len() < a_min {
      return Err(GbrapFrameError::APlaneTooShort {
        expected: a_min,
        actual: a.len(),
      });
    }

    Ok(Self {
      g,
      b,
      r,
      a,
      width,
      height,
      g_stride,
      b_stride,
      r_stride,
      a_stride,
    })
  }

  /// Constructs a new [`GbrapFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    g: &'a [u8],
    b: &'a [u8],
    r: &'a [u8],
    a: &'a [u8],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
    a_stride: u32,
  ) -> Self {
    match Self::try_new(
      g, b, r, a, width, height, g_stride, b_stride, r_stride, a_stride,
    ) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid GbrapFrame dimensions or plane lengths"),
    }
  }

  /// Green plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> &'a [u8] {
    self.g
  }
  /// Blue plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> &'a [u8] {
    self.b
  }
  /// Red plane bytes.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> &'a [u8] {
    self.r
  }
  /// Alpha plane bytes — full-width × full-height (1:1 with G).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [u8] {
    self.a
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
  /// Byte stride of the green plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g_stride(&self) -> u32 {
    self.g_stride
  }
  /// Byte stride of the blue plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b_stride(&self) -> u32 {
    self.b_stride
  }
  /// Byte stride of the red plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r_stride(&self) -> u32 {
    self.r_stride
  }
  /// Byte stride of the alpha plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }

  // ---- crate-internal Y/U/V aliases ------------------------------------
  //
  // See [`GbrpFrame`] for the rationale. Same pattern: `y == g`,
  // `u == b`, `v == r`; alpha already has the right name.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn y(&self) -> &'a [u8] {
    self.g
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn u(&self) -> &'a [u8] {
    self.b
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn v(&self) -> &'a [u8] {
    self.r
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn y_stride(&self) -> u32 {
    self.g_stride
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn u_stride(&self) -> u32 {
    self.b_stride
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn v_stride(&self) -> u32 {
    self.r_stride
  }
}

/// Errors returned by [`GbrapFrame::try_new`].
///
/// Variant shape mirrors [`GbrpFrameError`] extended with `A`-plane
/// variants (matching the YUVA-pattern from
/// [`super::Yuva444pFrameError`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum GbrapFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `g_stride < width`.
  #[error("g_stride ({g_stride}) is smaller than width ({width})")]
  GStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied G-plane stride.
    g_stride: u32,
  },
  /// `b_stride < width`.
  #[error("b_stride ({b_stride}) is smaller than width ({width})")]
  BStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied B-plane stride.
    b_stride: u32,
  },
  /// `r_stride < width`.
  #[error("r_stride ({r_stride}) is smaller than width ({width})")]
  RStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied R-plane stride.
    r_stride: u32,
  },
  /// `a_stride < width`.
  #[error("a_stride ({a_stride}) is smaller than width ({width})")]
  AStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied A-plane stride.
    a_stride: u32,
  },
  /// G plane is shorter than `g_stride * height` bytes.
  #[error("G plane has {actual} bytes but at least {expected} are required")]
  GPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// B plane is shorter than `b_stride * height` bytes.
  #[error("B plane has {actual} bytes but at least {expected} are required")]
  BPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// R plane is shorter than `r_stride * height` bytes.
  #[error("R plane has {actual} bytes but at least {expected} are required")]
  RPlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// A plane is shorter than `a_stride * height` bytes.
  #[error("A plane has {actual} bytes but at least {expected} are required")]
  APlaneTooShort {
    /// Minimum bytes required.
    expected: usize,
    /// Actual bytes supplied.
    actual: usize,
  },
  /// `stride * rows` does not fit in `usize` (32-bit targets only).
  #[error("declared geometry overflows usize: stride={stride} * rows={rows}")]
  GeometryOverflow {
    /// Stride of the plane whose size overflowed.
    stride: u32,
    /// Row count that overflowed against the stride.
    rows: u32,
  },
}
