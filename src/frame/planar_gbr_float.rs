//! Float-domain planar GBR source frames:
//! - `AV_PIX_FMT_GBRPF32{LE,BE}`  → [`Gbrpf32Frame`]  (G, B, R planes, `f32` elements)
//! - `AV_PIX_FMT_GBRAPF32{LE,BE}` → [`Gbrapf32Frame`] (G, B, R, A planes, `f32` elements)
//! - `AV_PIX_FMT_GBRPF16{LE,BE}`  → [`Gbrpf16Frame`]  (G, B, R planes, `half::f16`)
//! - `AV_PIX_FMT_GBRAPF16{LE,BE}` → [`Gbrapf16Frame`] (G, B, R, A planes, `half::f16`)
//!
//! Stride is in **elements** (not bytes). Sample range nominal `[0, 1]`; HDR > 1.0
//! is permitted on every accessor that documents it. NaN / Inf are preserved on
//! lossless pass-through paths and clamped on integer-output paths via
//! IEEE `min(max(x, 0.0), 1.0)`.
//!
//! # Endian contract — `<const BE: bool = false>`
//!
//! Each frame type carries a `<const BE: bool>` parameter that defaults to
//! `false` (LE-encoded plane bytes, matching the FFmpeg `*LE` suffix). Set
//! `BE = true` to consume `*BE`-encoded plane bytes; row kernels perform the
//! byte-swap (or no-op on host-native bit pattern) under the hood — callers
//! do **not** pre-swap. The `BE` parameter on `Frame` propagates through the
//! walker (`gbrpfXX_to::<BE>(...)`) into the sinker dispatch, which
//! monomorphizes the kernel call as `gbrpfXX_to_*_row::<BE>(...)`.

use derive_more::IsVariant;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type shared by all four frame constructors
// ---------------------------------------------------------------------------

/// Errors returned by the `try_new` constructors on the four float-domain
/// planar GBR frame types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum GbrFloatFrameError {
  /// `width` or `height` was zero.
  #[error("zero width or height: {width}×{height}")]
  ZeroDimension {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },

  /// `width × height` exceeds `i32::MAX` (the FFmpeg plane-size ceiling).
  #[error("dimension overflow: {width}×{height} exceeds i32::MAX")]
  DimensionOverflow {
    /// The supplied width.
    width: u32,
    /// The supplied height.
    height: u32,
  },

  /// A plane slice is shorter than `stride * (height - 1) + width`.
  #[error("plane '{plane}' too short: expected >= {expected}, got {actual}")]
  PlaneTooShort {
    /// Which plane was short (`"g"`, `"b"`, `"r"`, or `"a"`).
    plane: &'static str,
    /// Minimum elements required.
    expected: usize,
    /// Actual elements supplied.
    actual: usize,
  },

  /// A plane's stride is less than `width` (in elements).
  #[error("stride for plane '{plane}' must be >= width: stride={stride}, width={width}")]
  StrideBelowWidth {
    /// Which plane's stride was too small.
    plane: &'static str,
    /// The supplied stride (in elements).
    stride: u32,
    /// The declared frame width (in elements).
    width: u32,
  },

  /// `stride * (height - 1) + width` overflows `usize` (32-bit targets only).
  #[error("plane '{plane}' geometry overflows usize: stride={stride}, height={height}")]
  GeometryOverflow {
    /// Which plane's geometry overflowed.
    plane: &'static str,
    /// Stride of the plane that overflowed.
    stride: u32,
    /// Height of the frame.
    height: u32,
  },
}

// ---------------------------------------------------------------------------
// Helper: validate shared geometry checks
// ---------------------------------------------------------------------------

/// Returns `(width as usize, height as usize)` after confirming both are
/// non-zero and their product fits in `i32::MAX`.
#[inline(always)]
const fn check_dims(width: u32, height: u32) -> Result<(usize, usize), GbrFloatFrameError> {
  if width == 0 || height == 0 {
    return Err(GbrFloatFrameError::ZeroDimension { width, height });
  }
  if (width as i64) * (height as i64) > i32::MAX as i64 {
    return Err(GbrFloatFrameError::DimensionOverflow { width, height });
  }
  Ok((width as usize, height as usize))
}

/// Validates a single plane's stride and length.
#[inline(always)]
const fn check_plane(
  name: &'static str,
  plane_len: usize,
  stride: u32,
  w: usize,
  h: usize,
  height: u32,
) -> Result<(), GbrFloatFrameError> {
  if (stride as usize) < w {
    return Err(GbrFloatFrameError::StrideBelowWidth {
      plane: name,
      stride,
      width: w as u32,
    });
  }
  let stride_times_hm1 = match (stride as usize).checked_mul(h - 1) {
    Some(v) => v,
    None => {
      return Err(GbrFloatFrameError::GeometryOverflow {
        plane: name,
        stride,
        height,
      });
    }
  };
  let needed = match stride_times_hm1.checked_add(w) {
    Some(v) => v,
    None => {
      return Err(GbrFloatFrameError::GeometryOverflow {
        plane: name,
        stride,
        height,
      });
    }
  };
  if plane_len < needed {
    return Err(GbrFloatFrameError::PlaneTooShort {
      plane: name,
      expected: needed,
      actual: plane_len,
    });
  }
  Ok(())
}

// ---------------------------------------------------------------------------
// Gbrpf32Frame — three f32 planes, no alpha
// ---------------------------------------------------------------------------

/// A validated planar GBR float-32 frame (`AV_PIX_FMT_GBRPF32{LE,BE}`).
///
/// Three full-resolution `f32` planes in **G, B, R** order. Stride is in
/// `f32` elements. Nominal range `[0.0, 1.0]`; HDR values > 1.0 are
/// preserved bit-exact on lossless pass-through outputs and clamped to
/// `[0.0, 1.0]` on integer-output paths.
///
/// The `<const BE: bool>` parameter selects the plane byte order: `false`
/// (default) → LE-encoded bytes (`AV_PIX_FMT_GBRPF32LE`), `true` → BE-encoded
/// bytes (`AV_PIX_FMT_GBRPF32BE`). Downstream row kernels handle the
/// byte-swap of the float bit pattern (or no-op) under the hood — callers
/// do **not** pre-swap.
///
/// Stride is in **f32 elements** (not bytes). Callers holding byte buffers
/// from FFmpeg should cast via `bytemuck::cast_slice` and divide each
/// `linesize[i]` by 4 before constructing.
///
/// # Aliases
/// - [`Gbrpf32LeFrame`] = `Gbrpf32Frame<'a, false>`.
/// - [`Gbrpf32BeFrame`] = `Gbrpf32Frame<'a, true>`.
#[derive(Debug, Clone, Copy)]
pub struct Gbrpf32Frame<'a, const BE: bool = false> {
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
}

/// LE-encoded `Gbrpf32Frame` (`AV_PIX_FMT_GBRPF32LE`).
pub type Gbrpf32LeFrame<'a> = Gbrpf32Frame<'a, false>;
/// BE-encoded `Gbrpf32Frame` (`AV_PIX_FMT_GBRPF32BE`).
pub type Gbrpf32BeFrame<'a> = Gbrpf32Frame<'a, true>;

impl<'a, const BE: bool> Gbrpf32Frame<'a, BE> {
  /// Constructs a new [`Gbrpf32Frame`], validating dimensions and plane
  /// lengths. Returns [`GbrFloatFrameError`] if any precondition fails.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    g: &'a [f32],
    b: &'a [f32],
    r: &'a [f32],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
  ) -> Result<Self, GbrFloatFrameError> {
    let (w, h) = match check_dims(width, height) {
      Ok(v) => v,
      Err(e) => return Err(e),
    };
    if let Err(e) = check_plane("g", g.len(), g_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("b", b.len(), b_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("r", r.len(), r_stride, w, h, height) {
      return Err(e);
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
  /// Green plane samples. Row `n` starts at element offset `n * g_stride()`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> &'a [f32] {
    self.g
  }
  /// Green-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g_stride(&self) -> u32 {
    self.g_stride
  }
  /// Blue plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> &'a [f32] {
    self.b
  }
  /// Blue-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b_stride(&self) -> u32 {
    self.b_stride
  }
  /// Red plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> &'a [f32] {
    self.r
  }
  /// Red-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r_stride(&self) -> u32 {
    self.r_stride
  }
  /// Returns the compile-time BE flag — `true` if plane bytes are BE-encoded
  /// (`AV_PIX_FMT_GBRPF32BE`), `false` if LE-encoded (`AV_PIX_FMT_GBRPF32LE`).
  /// Runtime mirror of the `<const BE: bool>` type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}

// ---------------------------------------------------------------------------
// Gbrapf32Frame — four f32 planes, with alpha
// ---------------------------------------------------------------------------

/// A validated planar GBR+A float-32 frame (`AV_PIX_FMT_GBRAPF32{LE,BE}`).
///
/// Four full-resolution `f32` planes in **G, B, R, A** order. Alpha is
/// real per-pixel; nominal range `[0.0, 1.0]` (opaque = 1.0). Stride is
/// in `f32` elements.
///
/// The `<const BE: bool>` parameter selects the plane byte order; see
/// [`Gbrpf32Frame`] for the contract.
///
/// # Aliases
/// - [`Gbrapf32LeFrame`] = `Gbrapf32Frame<'a, false>`.
/// - [`Gbrapf32BeFrame`] = `Gbrapf32Frame<'a, true>`.
#[derive(Debug, Clone, Copy)]
pub struct Gbrapf32Frame<'a, const BE: bool = false> {
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  a: &'a [f32],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
  a_stride: u32,
}

/// LE-encoded `Gbrapf32Frame` (`AV_PIX_FMT_GBRAPF32LE`).
pub type Gbrapf32LeFrame<'a> = Gbrapf32Frame<'a, false>;
/// BE-encoded `Gbrapf32Frame` (`AV_PIX_FMT_GBRAPF32BE`).
pub type Gbrapf32BeFrame<'a> = Gbrapf32Frame<'a, true>;

impl<'a, const BE: bool> Gbrapf32Frame<'a, BE> {
  /// Constructs a new [`Gbrapf32Frame`], validating dimensions and plane
  /// lengths.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    g: &'a [f32],
    b: &'a [f32],
    r: &'a [f32],
    a: &'a [f32],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
    a_stride: u32,
  ) -> Result<Self, GbrFloatFrameError> {
    let (w, h) = match check_dims(width, height) {
      Ok(v) => v,
      Err(e) => return Err(e),
    };
    if let Err(e) = check_plane("g", g.len(), g_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("b", b.len(), b_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("r", r.len(), r_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("a", a.len(), a_stride, w, h, height) {
      return Err(e);
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
  /// Green plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> &'a [f32] {
    self.g
  }
  /// Green-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g_stride(&self) -> u32 {
    self.g_stride
  }
  /// Blue plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> &'a [f32] {
    self.b
  }
  /// Blue-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b_stride(&self) -> u32 {
    self.b_stride
  }
  /// Red plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> &'a [f32] {
    self.r
  }
  /// Red-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r_stride(&self) -> u32 {
    self.r_stride
  }
  /// Alpha plane samples (real per-pixel; opaque = 1.0).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [f32] {
    self.a
  }
  /// Alpha-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }
  /// Returns the compile-time BE flag — `true` if plane bytes are BE-encoded
  /// (`AV_PIX_FMT_GBRAPF32BE`), `false` if LE-encoded
  /// (`AV_PIX_FMT_GBRAPF32LE`). Runtime mirror of the `<const BE: bool>`
  /// type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}

// ---------------------------------------------------------------------------
// Gbrpf16Frame — three half::f16 planes, no alpha
// ---------------------------------------------------------------------------

/// A validated planar GBR float-16 frame (`AV_PIX_FMT_GBRPF16{LE,BE}`).
///
/// Three full-resolution [`half::f16`] planes in **G, B, R** order. Stride
/// is in `f16` elements. Nominal range `[0.0, 1.0]`; HDR values > 1.0 are
/// permitted (saturation to `+Inf` occurs on f16→f32 narrowing paths).
///
/// The `<const BE: bool>` parameter selects the plane byte order; see
/// [`Gbrpf32Frame`] for the contract.
///
/// # Aliases
/// - [`Gbrpf16LeFrame`] = `Gbrpf16Frame<'a, false>`.
/// - [`Gbrpf16BeFrame`] = `Gbrpf16Frame<'a, true>`.
#[derive(Debug, Clone, Copy)]
pub struct Gbrpf16Frame<'a, const BE: bool = false> {
  g: &'a [half::f16],
  b: &'a [half::f16],
  r: &'a [half::f16],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
}

/// LE-encoded `Gbrpf16Frame` (`AV_PIX_FMT_GBRPF16LE`).
pub type Gbrpf16LeFrame<'a> = Gbrpf16Frame<'a, false>;
/// BE-encoded `Gbrpf16Frame` (`AV_PIX_FMT_GBRPF16BE`).
pub type Gbrpf16BeFrame<'a> = Gbrpf16Frame<'a, true>;

impl<'a, const BE: bool> Gbrpf16Frame<'a, BE> {
  /// Constructs a new [`Gbrpf16Frame`], validating dimensions and plane
  /// lengths.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    g: &'a [half::f16],
    b: &'a [half::f16],
    r: &'a [half::f16],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
  ) -> Result<Self, GbrFloatFrameError> {
    let (w, h) = match check_dims(width, height) {
      Ok(v) => v,
      Err(e) => return Err(e),
    };
    if let Err(e) = check_plane("g", g.len(), g_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("b", b.len(), b_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("r", r.len(), r_stride, w, h, height) {
      return Err(e);
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
  /// Green plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> &'a [half::f16] {
    self.g
  }
  /// Green-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g_stride(&self) -> u32 {
    self.g_stride
  }
  /// Blue plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> &'a [half::f16] {
    self.b
  }
  /// Blue-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b_stride(&self) -> u32 {
    self.b_stride
  }
  /// Red plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> &'a [half::f16] {
    self.r
  }
  /// Red-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r_stride(&self) -> u32 {
    self.r_stride
  }
  /// Returns the compile-time BE flag — `true` if plane bytes are BE-encoded
  /// (`AV_PIX_FMT_GBRPF16BE`), `false` if LE-encoded
  /// (`AV_PIX_FMT_GBRPF16LE`). Runtime mirror of the `<const BE: bool>`
  /// type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}

// ---------------------------------------------------------------------------
// Gbrapf16Frame — four half::f16 planes, with alpha
// ---------------------------------------------------------------------------

/// A validated planar GBR+A float-16 frame (`AV_PIX_FMT_GBRAPF16{LE,BE}`).
///
/// Four full-resolution [`half::f16`] planes in **G, B, R, A** order.
/// Alpha is real per-pixel; nominal range `[0.0, 1.0]`. Stride is in
/// `f16` elements.
///
/// The `<const BE: bool>` parameter selects the plane byte order; see
/// [`Gbrpf32Frame`] for the contract.
///
/// # Aliases
/// - [`Gbrapf16LeFrame`] = `Gbrapf16Frame<'a, false>`.
/// - [`Gbrapf16BeFrame`] = `Gbrapf16Frame<'a, true>`.
#[derive(Debug, Clone, Copy)]
pub struct Gbrapf16Frame<'a, const BE: bool = false> {
  g: &'a [half::f16],
  b: &'a [half::f16],
  r: &'a [half::f16],
  a: &'a [half::f16],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
  a_stride: u32,
}

/// LE-encoded `Gbrapf16Frame` (`AV_PIX_FMT_GBRAPF16LE`).
pub type Gbrapf16LeFrame<'a> = Gbrapf16Frame<'a, false>;
/// BE-encoded `Gbrapf16Frame` (`AV_PIX_FMT_GBRAPF16BE`).
pub type Gbrapf16BeFrame<'a> = Gbrapf16Frame<'a, true>;

impl<'a, const BE: bool> Gbrapf16Frame<'a, BE> {
  /// Constructs a new [`Gbrapf16Frame`], validating dimensions and plane
  /// lengths.
  #[cfg_attr(not(tarpaulin), inline(always))]
  #[allow(clippy::too_many_arguments)]
  pub const fn try_new(
    g: &'a [half::f16],
    b: &'a [half::f16],
    r: &'a [half::f16],
    a: &'a [half::f16],
    width: u32,
    height: u32,
    g_stride: u32,
    b_stride: u32,
    r_stride: u32,
    a_stride: u32,
  ) -> Result<Self, GbrFloatFrameError> {
    let (w, h) = match check_dims(width, height) {
      Ok(v) => v,
      Err(e) => return Err(e),
    };
    if let Err(e) = check_plane("g", g.len(), g_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("b", b.len(), b_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("r", r.len(), r_stride, w, h, height) {
      return Err(e);
    }
    if let Err(e) = check_plane("a", a.len(), a_stride, w, h, height) {
      return Err(e);
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
  /// Green plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g(&self) -> &'a [half::f16] {
    self.g
  }
  /// Green-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn g_stride(&self) -> u32 {
    self.g_stride
  }
  /// Blue plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b(&self) -> &'a [half::f16] {
    self.b
  }
  /// Blue-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn b_stride(&self) -> u32 {
    self.b_stride
  }
  /// Red plane samples.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r(&self) -> &'a [half::f16] {
    self.r
  }
  /// Red-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn r_stride(&self) -> u32 {
    self.r_stride
  }
  /// Alpha plane samples (real per-pixel; opaque = 1.0).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a(&self) -> &'a [half::f16] {
    self.a
  }
  /// Alpha-plane element stride (`>= width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn a_stride(&self) -> u32 {
    self.a_stride
  }
  /// Returns the compile-time BE flag — `true` if plane bytes are BE-encoded
  /// (`AV_PIX_FMT_GBRAPF16BE`), `false` if LE-encoded
  /// (`AV_PIX_FMT_GBRAPF16LE`). Runtime mirror of the `<const BE: bool>`
  /// type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}
