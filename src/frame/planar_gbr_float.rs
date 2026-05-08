//! Float-domain planar GBR source frames:
//! - `AV_PIX_FMT_GBRPF32LE`  → [`Gbrpf32Frame`]  (G, B, R planes, `f32` elements)
//! - `AV_PIX_FMT_GBRAPF32LE` → [`Gbrapf32Frame`] (G, B, R, A planes, `f32` elements)
//! - `AV_PIX_FMT_GBRPF16LE`  → [`Gbrpf16Frame`]  (G, B, R planes, `half::f16`)
//! - `AV_PIX_FMT_GBRAPF16LE` → [`Gbrapf16Frame`] (G, B, R, A planes, `half::f16`)
//!
//! Stride is in **elements** (not bytes). Sample range nominal `[0, 1]`; HDR > 1.0
//! is permitted on every accessor that documents it. NaN / Inf are preserved on
//! lossless pass-through paths and clamped on integer-output paths via
//! IEEE `min(max(x, 0.0), 1.0)`.

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

/// A validated planar GBR float-32 frame (`AV_PIX_FMT_GBRPF32LE`).
///
/// Three full-resolution `f32` planes in **G, B, R** order. Stride is in
/// `f32` elements. Nominal range `[0.0, 1.0]`; HDR values > 1.0 are
/// preserved bit-exact on lossless pass-through outputs and clamped to
/// `[0.0, 1.0]` on integer-output paths.
///
/// # Endian contract — **LE-encoded bytes**
///
/// The three `&[f32]` planes are the **LE-encoded byte layout** reinterpreted
/// as `f32`, matching the FFmpeg `*LE` pixel-format suffix in the format
/// name. On a little-endian host (every CI runner today) LE bytes _are_
/// host-native, so the slices are also host-native float slices; on a
/// big-endian host the bytes have to be byte-swapped back to host-native
/// before arithmetic. Downstream row kernels handle this byte-swap (or
/// no-op on LE) under the hood — callers do **not** pre-swap.
///
/// Stride is in **f32 elements** (not bytes). Callers holding byte buffers
/// from FFmpeg should cast via `bytemuck::cast_slice` and divide each
/// `linesize[i]` by 4 before constructing.
#[derive(Debug, Clone, Copy)]
pub struct Gbrpf32Frame<'a> {
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
}

impl<'a> Gbrpf32Frame<'a> {
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
}

// ---------------------------------------------------------------------------
// Gbrapf32Frame — four f32 planes, with alpha
// ---------------------------------------------------------------------------

/// A validated planar GBR+A float-32 frame (`AV_PIX_FMT_GBRAPF32LE`).
///
/// Four full-resolution `f32` planes in **G, B, R, A** order. Alpha is
/// real per-pixel; nominal range `[0.0, 1.0]` (opaque = 1.0). Stride is
/// in `f32` elements.
///
/// # Endian contract — **LE-encoded bytes**
///
/// The four `&[f32]` planes are the **LE-encoded byte layout** reinterpreted
/// as `f32`, matching the FFmpeg `*LE` pixel-format suffix in the format
/// name. On a little-endian host (every CI runner today) LE bytes _are_
/// host-native, so the slices are also host-native float slices; on a
/// big-endian host the bytes have to be byte-swapped back to host-native
/// before arithmetic. Downstream row kernels handle this byte-swap (or
/// no-op on LE) under the hood — callers do **not** pre-swap.
///
/// Stride is in **f32 elements** (not bytes). Callers holding byte buffers
/// from FFmpeg should cast via `bytemuck::cast_slice` and divide each
/// `linesize[i]` by 4 before constructing.
#[derive(Debug, Clone, Copy)]
pub struct Gbrapf32Frame<'a> {
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

impl<'a> Gbrapf32Frame<'a> {
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
}

// ---------------------------------------------------------------------------
// Gbrpf16Frame — three half::f16 planes, no alpha
// ---------------------------------------------------------------------------

/// A validated planar GBR float-16 frame (`AV_PIX_FMT_GBRPF16LE`).
///
/// Three full-resolution [`half::f16`] planes in **G, B, R** order. Stride
/// is in `f16` elements. Nominal range `[0.0, 1.0]`; HDR values > 1.0 are
/// permitted (saturation to `+Inf` occurs on f16→f32 narrowing paths).
///
/// # Endian contract — **LE-encoded bytes**
///
/// The three `&[half::f16]` planes are the **LE-encoded byte layout**
/// reinterpreted as `f16`, matching the FFmpeg `*LE` pixel-format suffix in
/// the format name. On a little-endian host (every CI runner today) LE
/// bytes _are_ host-native, so the slices are also host-native f16 slices;
/// on a big-endian host the bytes have to be byte-swapped back to
/// host-native before arithmetic. Downstream row kernels handle this
/// byte-swap (or no-op on LE) under the hood — callers do **not** pre-swap.
///
/// Stride is in **f16 elements** (not bytes). Callers holding byte buffers
/// from FFmpeg should cast via `bytemuck::cast_slice` and divide each
/// `linesize[i]` by 2 before constructing.
#[derive(Debug, Clone, Copy)]
pub struct Gbrpf16Frame<'a> {
  g: &'a [half::f16],
  b: &'a [half::f16],
  r: &'a [half::f16],
  width: u32,
  height: u32,
  g_stride: u32,
  b_stride: u32,
  r_stride: u32,
}

impl<'a> Gbrpf16Frame<'a> {
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
}

// ---------------------------------------------------------------------------
// Gbrapf16Frame — four half::f16 planes, with alpha
// ---------------------------------------------------------------------------

/// A validated planar GBR+A float-16 frame (`AV_PIX_FMT_GBRAPF16LE`).
///
/// Four full-resolution [`half::f16`] planes in **G, B, R, A** order.
/// Alpha is real per-pixel; nominal range `[0.0, 1.0]`. Stride is in
/// `f16` elements.
///
/// # Endian contract — **LE-encoded bytes**
///
/// The four `&[half::f16]` planes are the **LE-encoded byte layout**
/// reinterpreted as `f16`, matching the FFmpeg `*LE` pixel-format suffix in
/// the format name. On a little-endian host (every CI runner today) LE
/// bytes _are_ host-native, so the slices are also host-native f16 slices;
/// on a big-endian host the bytes have to be byte-swapped back to
/// host-native before arithmetic. Downstream row kernels handle this
/// byte-swap (or no-op on LE) under the hood — callers do **not** pre-swap.
///
/// Stride is in **f16 elements** (not bytes). Callers holding byte buffers
/// from FFmpeg should cast via `bytemuck::cast_slice` and divide each
/// `linesize[i]` by 2 before constructing.
#[derive(Debug, Clone, Copy)]
pub struct Gbrapf16Frame<'a> {
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

impl<'a> Gbrapf16Frame<'a> {
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
}
