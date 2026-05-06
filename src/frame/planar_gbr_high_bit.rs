//! High-bit-depth planar GBR (`AV_PIX_FMT_GBRP{9,10,12,14,16}LE`) and
//! planar GBR+A (`AV_PIX_FMT_GBRAP{10,12,14,16}LE`) source frames.
//!
//! Both formats are *planar RGB* — three (or four) full-resolution `u16`
//! planes, one per channel, in **G, B, R** order (FFmpeg convention).
//! `GbrapHighBitFrame` adds a fourth full-resolution alpha plane.
//!
//! Samples are stored in the **low `BITS` bits** of each `u16` element
//! (upper `16 − BITS` bits zero), matching FFmpeg's little-endian
//! `gbrp{9,10,12,14,16}le` / `gbrap{10,12,14,16}le` conventions.
//! (FFmpeg has no `gbrap9le` — only the 3-plane `gbrp9le` exists at 9
//! bits, so `GbrapHighBitFrame` accepts only `BITS ∈ {10, 12, 14, 16}`.)
//! Callers with byte buffers from FFmpeg must cast via
//! [`bytemuck::cast_slice`] and divide `linesize[i]` by 2 before
//! construction.
//!
//! # Stride semantics
//!
//! **Stride is in samples (`u16` elements)**, not bytes. Each plane row
//! `r` starts at sample offset `r * *_stride`.
//!
//! # Sample-value range
//!
//! `try_new` validates geometry only. Out-of-range samples (upper bits
//! set) are masked by `& ((1 << BITS) - 1)` inside every kernel, giving
//! stable deterministic output. Scanning every sample at video rates is
//! prohibitive — same rationale as `Yuv420pFrame16`.

use derive_more::IsVariant;
use thiserror::Error;

/// A validated planar GBR frame at high bit depth (`AV_PIX_FMT_GBRP{9,10,12,14,16}LE`).
///
/// Three full-resolution `u16` planes in **G, B, R** order:
/// - `g` — green plane.
/// - `b` — blue plane.
/// - `r` — red plane.
///
/// `BITS ∈ {9, 10, 12, 14, 16}` — validated by a compile-time
/// `const` assertion at construction. Stride is in **samples** (`u16`
/// elements); each plane requires `*_stride >= width` and
/// `len >= *_stride * height`. No width/height parity constraint.
///
/// Use the per-depth type aliases [`Gbrp9Frame`], [`Gbrp10Frame`],
/// [`Gbrp12Frame`], [`Gbrp14Frame`], [`Gbrp16Frame`] at call sites.
#[derive(Debug, Clone, Copy)]
pub struct GbrpHighBitFrame<'a, const BITS: u32> {
  g: &'a [u16],
  b: &'a [u16],
  r: &'a [u16],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
}

impl<'a, const BITS: u32> GbrpHighBitFrame<'a, BITS> {
  /// Constructs a new [`GbrpHighBitFrame`], validating dimensions and
  /// plane lengths. Returns [`GbrpHighBitFrameError`] if any of:
  /// - `BITS ∉ {9, 10, 12, 14, 16}` — caught at compile time via
  ///   `const { assert!(…) }`, so misuse is a compile error rather than
  ///   a runtime error,
  /// - `width` or `height` is zero,
  /// - any stride is smaller than `width` (in samples),
  /// - `stride * height` overflows `usize` (32-bit targets only),
  /// - any plane is shorter than `stride * height` samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    g: &'a [u16],
    b: &'a [u16],
    r: &'a [u16],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
  ) -> Result<Self, GbrpHighBitFrameError> {
    const {
      assert!(
        matches!(BITS, 9 | 10 | 12 | 14 | 16),
        "BITS must be one of 9, 10, 12, 14, or 16",
      );
    }

    if width == 0 || height == 0 {
      return Err(GbrpHighBitFrameError::ZeroDimension { width, height });
    }
    if g_stride < width {
      return Err(GbrpHighBitFrameError::GStrideTooSmall { width, g_stride });
    }
    if b_stride < width {
      return Err(GbrpHighBitFrameError::BStrideTooSmall { width, b_stride });
    }
    if r_stride < width {
      return Err(GbrpHighBitFrameError::RStrideTooSmall { width, r_stride });
    }

    let g_min = match (g_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrpHighBitFrameError::GeometryOverflow {
          stride: g_stride,
          rows: height,
        });
      }
    };
    if g.len() < g_min {
      return Err(GbrpHighBitFrameError::GPlaneTooShort {
        expected: g_min,
        actual: g.len(),
      });
    }

    let b_min = match (b_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrpHighBitFrameError::GeometryOverflow {
          stride: b_stride,
          rows: height,
        });
      }
    };
    if b.len() < b_min {
      return Err(GbrpHighBitFrameError::BPlaneTooShort {
        expected: b_min,
        actual: b.len(),
      });
    }

    let r_min = match (r_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrpHighBitFrameError::GeometryOverflow {
          stride: r_stride,
          rows: height,
        });
      }
    };
    if r.len() < r_min {
      return Err(GbrpHighBitFrameError::RPlaneTooShort {
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

  /// Constructs a new [`GbrpHighBitFrame`], panicking on invalid inputs.
  /// Prefer [`Self::try_new`] when inputs may be invalid at runtime.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    g: &'a [u16],
    b: &'a [u16],
    r: &'a [u16],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
  ) -> Self {
    match Self::try_new(g, b, r, width, height, g_stride, b_stride, r_stride) {
      Ok(frame) => frame,
      Err(_) => panic!("invalid GbrpHighBitFrame dimensions or plane lengths"),
    }
  }

  /// Green plane samples. Row `r` starts at sample offset `r * g_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> &'a [u16] {
    self.g
  }
  /// Blue plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> &'a [u16] {
    self.b
  }
  /// Red plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> &'a [u16] {
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
  /// Sample stride of the green plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g_stride(&self) -> u32 {
    self.g_stride
  }
  /// Sample stride of the blue plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b_stride(&self) -> u32 {
    self.b_stride
  }
  /// Sample stride of the red plane (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r_stride(&self) -> u32 {
    self.r_stride
  }
  /// Active bit depth — one of 9, 10, 12, 14, or 16. Mirrors the `BITS`
  /// const parameter so generic code can read it without naming the type.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }

  // ---- crate-internal Y/U/V aliases ------------------------------------
  //
  // The shared `walker!` macro uses fixed `y/u/v` field-name conventions
  // (`src.y()`, `src.u_stride()`, etc.). To reuse the macro verbatim for
  // planar GBR — whose externally-correct accessor names are `g/b/r` —
  // we expose `pub(crate)` aliases: `y == g`, `u == b`, `v == r`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn y(&self) -> &'a [u16] {
    self.g
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn u(&self) -> &'a [u16] {
    self.b
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn v(&self) -> &'a [u16] {
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

/// Errors returned by [`GbrpHighBitFrame::try_new`].
///
/// Variant shape mirrors [`super::GbrpFrameError`] but with all sizes
/// expressed in **samples** (`u16` elements) instead of bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum GbrpHighBitFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `g_stride < width` (in samples).
  #[error("g_stride ({g_stride}) is smaller than width ({width})")]
  GStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied G-plane stride (samples).
    g_stride: u32,
  },
  /// `b_stride < width` (in samples).
  #[error("b_stride ({b_stride}) is smaller than width ({width})")]
  BStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied B-plane stride (samples).
    b_stride: u32,
  },
  /// `r_stride < width` (in samples).
  #[error("r_stride ({r_stride}) is smaller than width ({width})")]
  RStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied R-plane stride (samples).
    r_stride: u32,
  },
  /// G plane is shorter than `g_stride * height` samples.
  #[error("G plane has {actual} samples but at least {expected} are required")]
  GPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// B plane is shorter than `b_stride * height` samples.
  #[error("B plane has {actual} samples but at least {expected} are required")]
  BPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// R plane is shorter than `r_stride * height` samples.
  #[error("R plane has {actual} samples but at least {expected} are required")]
  RPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
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

/// Type alias for a validated planar GBR 9-bit frame
/// (`AV_PIX_FMT_GBRP9LE`). Samples in the low 9 bits of each `u16`.
pub type Gbrp9Frame<'a> = GbrpHighBitFrame<'a, 9>;
/// Type alias for a validated planar GBR 10-bit frame
/// (`AV_PIX_FMT_GBRP10LE`). Samples in the low 10 bits of each `u16`.
pub type Gbrp10Frame<'a> = GbrpHighBitFrame<'a, 10>;
/// Type alias for a validated planar GBR 12-bit frame
/// (`AV_PIX_FMT_GBRP12LE`). Samples in the low 12 bits of each `u16`.
pub type Gbrp12Frame<'a> = GbrpHighBitFrame<'a, 12>;
/// Type alias for a validated planar GBR 14-bit frame
/// (`AV_PIX_FMT_GBRP14LE`). Samples in the low 14 bits of each `u16`.
pub type Gbrp14Frame<'a> = GbrpHighBitFrame<'a, 14>;
/// Type alias for a validated planar GBR 16-bit frame
/// (`AV_PIX_FMT_GBRP16LE`). Full `u16` range — all 16 bits active.
pub type Gbrp16Frame<'a> = GbrpHighBitFrame<'a, 16>;

// ---------------------------------------------------------------------------

/// A validated planar GBR+A frame at high bit depth
/// (`AV_PIX_FMT_GBRAP{9,10,12,14,16}LE`).
///
/// Four full-resolution `u16` planes in **G, B, R, A** order:
/// - `g` / `b` / `r` — colour planes.
/// - `a` — alpha plane (1:1 with G; real per-pixel alpha).
///
/// `BITS ∈ {9, 10, 12, 14, 16}` — validated at compile time. Stride is
/// in **samples** (`u16` elements); each plane requires
/// `*_stride >= width` and `len >= *_stride * height`.
///
/// Use the per-depth aliases [`Gbrap10Frame`] through [`Gbrap16Frame`].
/// (FFmpeg has no GBRAP9 variant — only the 3-plane GBRP9 exists at 9 bits.)
#[derive(Debug, Clone, Copy)]
pub struct GbrapHighBitFrame<'a, const BITS: u32> {
  g: &'a [u16],
  b: &'a [u16],
  r: &'a [u16],
  a: &'a [u16],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
  a_stride: u32,
}

impl<'a, const BITS: u32> GbrapHighBitFrame<'a, BITS> {
  /// Constructs a new [`GbrapHighBitFrame`], validating dimensions and
  /// plane lengths.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    g: &'a [u16],
    b: &'a [u16],
    r: &'a [u16],
    a: &'a [u16],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
    a_stride: u32,
  ) -> Result<Self, GbrapHighBitFrameError> {
    const {
      assert!(
        matches!(BITS, 10 | 12 | 14 | 16),
        "BITS must be one of 10, 12, 14, or 16 (FFmpeg has no GBRAP9 variant)",
      );
    }

    if width == 0 || height == 0 {
      return Err(GbrapHighBitFrameError::ZeroDimension { width, height });
    }
    if g_stride < width {
      return Err(GbrapHighBitFrameError::GStrideTooSmall { width, g_stride });
    }
    if b_stride < width {
      return Err(GbrapHighBitFrameError::BStrideTooSmall { width, b_stride });
    }
    if r_stride < width {
      return Err(GbrapHighBitFrameError::RStrideTooSmall { width, r_stride });
    }
    if a_stride < width {
      return Err(GbrapHighBitFrameError::AStrideTooSmall { width, a_stride });
    }

    let g_min = match (g_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrapHighBitFrameError::GeometryOverflow {
          stride: g_stride,
          rows: height,
        });
      }
    };
    if g.len() < g_min {
      return Err(GbrapHighBitFrameError::GPlaneTooShort {
        expected: g_min,
        actual: g.len(),
      });
    }

    let b_min = match (b_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrapHighBitFrameError::GeometryOverflow {
          stride: b_stride,
          rows: height,
        });
      }
    };
    if b.len() < b_min {
      return Err(GbrapHighBitFrameError::BPlaneTooShort {
        expected: b_min,
        actual: b.len(),
      });
    }

    let r_min = match (r_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrapHighBitFrameError::GeometryOverflow {
          stride: r_stride,
          rows: height,
        });
      }
    };
    if r.len() < r_min {
      return Err(GbrapHighBitFrameError::RPlaneTooShort {
        expected: r_min,
        actual: r.len(),
      });
    }

    let a_min = match (a_stride as usize).checked_mul(height as usize) {
      Some(v) => v,
      None => {
        return Err(GbrapHighBitFrameError::GeometryOverflow {
          stride: a_stride,
          rows: height,
        });
      }
    };
    if a.len() < a_min {
      return Err(GbrapHighBitFrameError::APlaneTooShort {
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

  /// Constructs a new [`GbrapHighBitFrame`], panicking on invalid inputs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn new(
    g: &'a [u16],
    b: &'a [u16],
    r: &'a [u16],
    a: &'a [u16],
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
      Err(_) => panic!("invalid GbrapHighBitFrame dimensions or plane lengths"),
    }
  }

  /// Green plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> &'a [u16] {
    self.g
  }
  /// Blue plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> &'a [u16] {
    self.b
  }
  /// Red plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> &'a [u16] {
    self.r
  }
  /// Alpha plane samples — full-width × full-height (1:1 with G).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [u16] {
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
  /// Sample stride of the green plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g_stride(&self) -> u32 {
    self.g_stride
  }
  /// Sample stride of the blue plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b_stride(&self) -> u32 {
    self.b_stride
  }
  /// Sample stride of the red plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r_stride(&self) -> u32 {
    self.r_stride
  }
  /// Sample stride of the alpha plane.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }
  /// Active bit depth — one of 9, 10, 12, 14, or 16.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn bits(&self) -> u32 {
    BITS
  }

  // ---- crate-internal Y/U/V aliases ------------------------------------
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn y(&self) -> &'a [u16] {
    self.g
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn u(&self) -> &'a [u16] {
    self.b
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn v(&self) -> &'a [u16] {
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
  // `a_stride` already has the right name — no alias needed.
}

/// Errors returned by [`GbrapHighBitFrame::try_new`].
///
/// Mirrors [`GbrpHighBitFrameError`] extended with `A`-plane variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum GbrapHighBitFrameError {
  /// `width` or `height` was zero.
  #[error("width ({width}) or height ({height}) is zero")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },
  /// `g_stride < width` (in samples).
  #[error("g_stride ({g_stride}) is smaller than width ({width})")]
  GStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied G-plane stride (samples).
    g_stride: u32,
  },
  /// `b_stride < width` (in samples).
  #[error("b_stride ({b_stride}) is smaller than width ({width})")]
  BStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied B-plane stride (samples).
    b_stride: u32,
  },
  /// `r_stride < width` (in samples).
  #[error("r_stride ({r_stride}) is smaller than width ({width})")]
  RStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied R-plane stride (samples).
    r_stride: u32,
  },
  /// `a_stride < width` (in samples).
  #[error("a_stride ({a_stride}) is smaller than width ({width})")]
  AStrideTooSmall {
    /// Declared frame width in pixels.
    width: u32,
    /// The supplied A-plane stride (samples).
    a_stride: u32,
  },
  /// G plane is shorter than `g_stride * height` samples.
  #[error("G plane has {actual} samples but at least {expected} are required")]
  GPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// B plane is shorter than `b_stride * height` samples.
  #[error("B plane has {actual} samples but at least {expected} are required")]
  BPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// R plane is shorter than `r_stride * height` samples.
  #[error("R plane has {actual} samples but at least {expected} are required")]
  RPlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
    actual: usize,
  },
  /// A plane is shorter than `a_stride * height` samples.
  #[error("A plane has {actual} samples but at least {expected} are required")]
  APlaneTooShort {
    /// Minimum samples required.
    expected: usize,
    /// Actual samples supplied.
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

/// Type alias for a validated planar GBR+A 10-bit frame
/// (`AV_PIX_FMT_GBRAP10LE`). Samples in the low 10 bits of each `u16`.
pub type Gbrap10Frame<'a> = GbrapHighBitFrame<'a, 10>;
/// Type alias for a validated planar GBR+A 12-bit frame
/// (`AV_PIX_FMT_GBRAP12LE`). Samples in the low 12 bits of each `u16`.
pub type Gbrap12Frame<'a> = GbrapHighBitFrame<'a, 12>;
/// Type alias for a validated planar GBR+A 14-bit frame
/// (`AV_PIX_FMT_GBRAP14LE`). Samples in the low 14 bits of each `u16`.
pub type Gbrap14Frame<'a> = GbrapHighBitFrame<'a, 14>;
/// Type alias for a validated planar GBR+A 16-bit frame
/// (`AV_PIX_FMT_GBRAP16LE`). Full `u16` range — all 16 bits active.
pub type Gbrap16Frame<'a> = GbrapHighBitFrame<'a, 16>;
