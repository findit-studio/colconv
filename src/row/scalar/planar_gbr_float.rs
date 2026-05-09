//! Scalar f32-source and f16-source kernels for planar GBR float formats.
//!
//! Source planes are `&[f32]` (or `&[half::f16]` widened to f32 at dispatch).
//! Nominal sample range `[0.0, 1.0]`; HDR values > 1.0 are permitted and
//! handled as follows:
//!
//! - **Integer-output paths** (`*_u8`, `*_u16`): clamped via
//!   `.clamp(0.0, 1.0)` before scaling. NaN is not clamped to a valid value —
//!   `f32::clamp(NaN, 0.0, 1.0)` returns NaN (Rust 1.50+), and the subsequent
//!   saturating `as u8` / `as u16` cast gives 0. Callers must not rely on a
//!   specific NaN result.
//! - **Lossless float-output paths** (`*_f32`): HDR, NaN,
//!   and Inf are preserved bit-exact (`gbrpf32_to_rgb_f32_row`,
//!   `gbrpf32_to_rgba_f32_row`).
//! - **f16-output paths** (`*_f16`): HDR values exceeding the f16 maximum
//!   (~65504) saturate to `f16::INFINITY` / `f16::NEG_INFINITY`. This is the
//!   documented caller-visible behaviour; callers needing full HDR range use
//!   the f32 pass-through accessors.
//!
//! # Endian support
//!
//! All `<const BE: bool>` kernels take the source planes as raw byte slices
//! (`&[u8]` reinterpreted as `&[u32]` / `&[u16]` with byte-swap when
//! `BE = true`). The `BE = false` path is identical to the original LE kernels
//! — the compiler eliminates the dead branch.
//!
//! # Rounding (float → integer)
//!
//! `(y.clamp(0.0, 1.0) * scale + 0.5) as T`
//!
//! Adding 0.5 before truncation gives round-to-nearest (ties round up),
//! MXCSR-independent. Matches the Grayf32 and Rgbf32 scalar contracts.
//!
//! # f32 → f16 rounding
//!
//! IEEE-754 round-to-nearest-even via `half::f16::from_f32` (the `half`
//! crate default). No override needed.
//!
//! # Channel reorder
//!
//! FFmpeg planar GBR stores planes in **G, B, R** order, but the packed
//! output convention is **R, G, B** (matching `AV_PIX_FMT_RGB24`). Every
//! kernel performs this reorder.

// Kernels are not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#![cfg_attr(not(test), allow(dead_code))]

use crate::ColorMatrix;

// ---- shared helpers --------------------------------------------------------

/// Round-to-nearest f32 → u8, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 255, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u8(y: f32) -> u8 {
  (y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Round-to-nearest f32 → u16, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 65535, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u16(y: f32) -> u16 {
  (y.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

/// f32 → half::f16 via IEEE-754 round-to-nearest-even.
/// HDR values exceeding f16 max (~65504) saturate to ±Inf.
#[inline(always)]
fn f32_to_f16(y: f32) -> half::f16 {
  half::f16::from_f32(y)
}

/// Load a single f32 sample from a `&[f32]` plane, target-endian aware.
///
/// The source plane is the raw on-disk / on-wire byte stream reinterpreted
/// as `&[f32]`. Each f32 read therefore picks up four bytes in **host-native**
/// order. We then convert that host-native u32 to the value the encoded
/// stream represents:
///
/// - `BE = true`: bytes on disk are big-endian → `u32::from_be` is a no-op
///   on BE hosts and a byte-swap on LE hosts.
/// - `BE = false`: bytes on disk are little-endian → `u32::from_le` is a
///   no-op on LE hosts and a byte-swap on BE hosts.
///
/// **Both** branches go through `from_be` / `from_le` so the
/// LE-data-on-BE-host case is handled correctly too. An unconditional
/// `swap_bytes` would corrupt rows on big-endian hosts (e.g. s390x).
#[inline(always)]
fn load_f32<const BE: bool>(plane: &[f32], i: usize) -> f32 {
  let raw = plane[i];
  if BE {
    f32::from_bits(u32::from_be(raw.to_bits()))
  } else {
    f32::from_bits(u32::from_le(raw.to_bits()))
  }
}

// ---- Gbrpf32 → u8 RGB ------------------------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **bytes**.
///
/// Each f32 sample is clamped to `[0.0, 1.0]` and scaled to `[0, 255]`
/// with round-half-up. Output order is **R, G, B** per pixel.
///
/// When `BE = true` each f32 element is loaded as a big-endian u32 bit
/// pattern (4-byte swap before reinterpret). `BE = false` is LE / host-native.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = f32_to_u8(load_f32::<BE>(r, x));
    rgb_out[dst + 1] = f32_to_u8(load_f32::<BE>(g, x));
    rgb_out[dst + 2] = f32_to_u8(load_f32::<BE>(b, x));
  }
}

// ---- Gbrpf32 → u8 RGBA (opaque α) -----------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **bytes**
/// with constant opaque α = `0xFF`. Used for `Gbrpf32` sources (no α plane).
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u8(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_u8(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_u8(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = 0xFF;
  }
}

// ---- Gbrpf32 → u16 RGB -----------------------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **`u16`**.
///
/// Each f32 sample is clamped to `[0.0, 1.0]` and scaled to `[0, 65535]`
/// with round-half-up (full-range).
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = f32_to_u16(load_f32::<BE>(r, x));
    rgb_out[dst + 1] = f32_to_u16(load_f32::<BE>(g, x));
    rgb_out[dst + 2] = f32_to_u16(load_f32::<BE>(b, x));
  }
}

// ---- Gbrpf32 → u16 RGBA (opaque α) ----------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **`u16`**
/// with constant opaque α = `0xFFFF`.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u16(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_u16(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_u16(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = 0xFFFF;
  }
}

// ---- Gbrpf32 → f32 RGB (lossless) ------------------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **`f32`**.
///
/// Lossless interleave — no clamping, no rounding. HDR values > 1.0,
/// NaN, and Inf are preserved bit-exact.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap)
/// before being written to the output. The output is always host-native f32.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = load_f32::<BE>(r, x);
    rgb_out[dst + 1] = load_f32::<BE>(g, x);
    rgb_out[dst + 2] = load_f32::<BE>(b, x);
  }
}

// ---- Gbrpf32 → f32 RGBA (lossless, α = 1.0) --------------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **`f32`**
/// with α = `1.0` (opaque). Lossless — HDR, NaN, and Inf preserved.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = load_f32::<BE>(r, x);
    rgba_out[dst + 1] = load_f32::<BE>(g, x);
    rgba_out[dst + 2] = load_f32::<BE>(b, x);
    rgba_out[dst + 3] = 1.0;
  }
}

// ---- Gbrpf32 → f16 RGB (fused narrow + interleave) -------------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B` **`half::f16`**.
///
/// Fused planar-gather, IEEE-754 round-to-nearest-even f32→f16 narrow, and
/// interleave in a single pass. HDR values exceeding the f16 maximum (~65504)
/// saturate to `half::f16::INFINITY`. Callers needing full HDR range use
/// `gbrpf32_to_rgb_f32_row` instead.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgb_f16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = f32_to_f16(load_f32::<BE>(r, x));
    rgb_out[dst + 1] = f32_to_f16(load_f32::<BE>(g, x));
    rgb_out[dst + 2] = f32_to_f16(load_f32::<BE>(b, x));
  }
}

// ---- Gbrpf32 → f16 RGBA (fused narrow, α = f16(1.0)) ----------------------

/// Interleaves planar G/B/R `f32` rows into packed `R, G, B, A` **`half::f16`**
/// with α = `half::f16::from_f32(1.0)`. HDR > ~65504 saturates to f16 ±Inf.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_rgba_f16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let one_f16 = half::f16::from_f32(1.0);
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_f16(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_f16(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_f16(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = one_f16;
  }
}

// ---- Gbrpf32 → u8 luma (staged via RGB scratch) ----------------------------

/// Derives luma (Y') from planar G/B/R `f32` rows by staging through an
/// 8-bit packed-RGB scratch buffer in chunks of up to 64 pixels.
///
/// The intermediate u8 RGB uses round-half-up clamping; luma is then computed
/// by `rgb_to_luma_row`. `matrix` and `full_range` control the luma weighting.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  luma_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    gbrpf32_to_rgb_row::<BE>(
      &g[offset..],
      &b[offset..],
      &r[offset..],
      &mut scratch[..n * 3],
      n,
    );
    super::rgb_to_luma_row(
      &scratch[..n * 3],
      &mut luma_out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → u16 luma (staged via RGB scratch) ---------------------------

/// Derives luma (Y') in `u16` from planar G/B/R `f32` rows by staging through
/// an 8-bit packed-RGB scratch buffer in chunks of up to 64 pixels.
///
/// The u16 luma value has the same dynamic range as the u8 path (0–255), zero-
/// extended into the u16 carrier — matching the convention of packed-YUV
/// `*_to_luma_u16_row` kernels for 8-bit-equivalent sources.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gbrpf32_to_luma_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  luma_out: &mut [u16],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    gbrpf32_to_rgb_row::<BE>(
      &g[offset..],
      &b[offset..],
      &r[offset..],
      &mut scratch[..n * 3],
      n,
    );
    super::rgb_to_luma_u16_row(
      &scratch[..n * 3],
      &mut luma_out[offset..offset + n],
      n,
      matrix,
      full_range,
    );
    offset += n;
  }
}

// ---- Gbrpf32 → HSV (staged via RGB scratch) --------------------------------

/// Converts planar G/B/R `f32` rows to planar HSV **bytes** by staging
/// through an 8-bit packed-RGB scratch buffer in chunks of up to 64 pixels.
///
/// Matches OpenCV `cv2.COLOR_RGB2HSV` semantics: `H ∈ [0, 179]`, `S, V ∈
/// [0, 255]`. f32 values are clamped via `f32_to_u8` before the RGB→HSV step.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf32_to_hsv_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");
  const CHUNK: usize = 64;
  let mut scratch = [0u8; CHUNK * 3];
  let mut offset = 0;
  while offset < width {
    let n = (width - offset).min(CHUNK);
    gbrpf32_to_rgb_row::<BE>(
      &g[offset..],
      &b[offset..],
      &r[offset..],
      &mut scratch[..n * 3],
      n,
    );
    super::rgb_to_hsv_row(
      &scratch[..n * 3],
      &mut h_out[offset..offset + n],
      &mut s_out[offset..offset + n],
      &mut v_out[offset..offset + n],
      n,
    );
    offset += n;
  }
}

// ---- Gbrapf32 → u8 RGBA (source α) ----------------------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A` **bytes**.
///
/// α is sourced from the `a` plane: clamped to `[0.0, 1.0]` and scaled by 255
/// with round-half-up.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u8(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_u8(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_u8(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = f32_to_u8(load_f32::<BE>(a, x));
  }
}

// ---- Gbrapf32 → u16 RGBA (source α) ----------------------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A` **`u16`**.
///
/// α is sourced from the `a` plane: clamped to `[0.0, 1.0]` and scaled by
/// 65535 with round-half-up.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_u16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_u16(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_u16(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_u16(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = f32_to_u16(load_f32::<BE>(a, x));
  }
}

// ---- Gbrapf32 → f32 RGBA (lossless source α) --------------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A` **`f32`**.
///
/// Lossless — HDR, NaN, and Inf are preserved bit-exact in all four channels
/// including α.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f32_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = load_f32::<BE>(r, x);
    rgba_out[dst + 1] = load_f32::<BE>(g, x);
    rgba_out[dst + 2] = load_f32::<BE>(b, x);
    rgba_out[dst + 3] = load_f32::<BE>(a, x);
  }
}

// ---- Gbrapf32 → f16 RGBA (fused narrow, source α) ---------------------------

/// Interleaves planar G/B/R/A `f32` rows into packed `R, G, B, A`
/// **`half::f16`** with source α.
///
/// Fused narrow: all four channels converted via IEEE-754 round-to-nearest-even
/// in a single pass. HDR > ~65504 saturates to f16 ±Inf.
///
/// `BE = true`: each f32 loaded as big-endian u32 bit pattern (4-byte swap).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf32_to_rgba_f16_row<const BE: bool>(
  g: &[f32],
  b: &[f32],
  r: &[f32],
  a: &[f32],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = f32_to_f16(load_f32::<BE>(r, x));
    rgba_out[dst + 1] = f32_to_f16(load_f32::<BE>(g, x));
    rgba_out[dst + 2] = f32_to_f16(load_f32::<BE>(b, x));
    rgba_out[dst + 3] = f32_to_f16(load_f32::<BE>(a, x));
  }
}

// ---- Unit tests ------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // ---- helper: byte-swap a slice of f32 to simulate BE source ----------------

  fn be_encode(src: &[f32]) -> std::vec::Vec<f32> {
    src
      .iter()
      .map(|v| f32::from_bits(v.to_bits().swap_bytes()))
      .collect()
  }

  /// Re-encode a host-native f32 slice as LE-encoded f32 storage. Kernels
  /// called with `BE = false` recover the intended host-native value via
  /// `u32::from_le` on both LE (no-op) and BE (byte-swap) hosts.
  fn as_le_f32(host: &[f32]) -> std::vec::Vec<f32> {
    host
      .iter()
      .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
      .collect()
  }

  // ---- gbrpf32_to_rgb_row --------------------------------------------------

  #[test]
  fn gbrpf32_to_rgb_clamps_and_scales() {
    // Values: 0.0, 0.5, 1.0, 1.5, -0.1, NaN → 0, 128, 255, 255, 0, 0
    // NaN passes through f32::clamp unchanged (Rust 1.50+); `NaN as u8` saturates to 0.
    // All three channels use the same value for simplicity.
    let vals = [0.0f32, 0.5, 1.0, 1.5, -0.1, f32::NAN];
    let expected = [0u8, 128, 255, 255, 0, 0];
    for (v, e) in vals.iter().zip(expected.iter()) {
      let g = as_le_f32(&[*v]);
      let b = as_le_f32(&[*v]);
      let r = as_le_f32(&[*v]);
      let mut out = [0u8; 3];
      gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out, 1);
      assert_eq!(out[0], *e, "R: v={v}, expected={e}");
      assert_eq!(out[1], *e, "G: v={v}, expected={e}");
      assert_eq!(out[2], *e, "B: v={v}, expected={e}");
    }
  }

  #[test]
  fn gbrpf32_to_rgb_channel_reorder() {
    // G=0.0, B=0.5, R=1.0 → packed R=255, G=0, B=128
    let g = as_le_f32(&[0.0f32]);
    let b = as_le_f32(&[0.5f32]);
    let r = as_le_f32(&[1.0f32]);
    let mut out = [0u8; 3];
    gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], 255, "R");
    assert_eq!(out[1], 0, "G");
    assert_eq!(out[2], 128, "B");
  }

  #[test]
  fn gbrpf32_to_rgb_be_parity() {
    // BE-encoded source must decode to same output as LE source.
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = std::vec![0u8; 4 * 3];
    let mut be_out = std::vec![0u8; 4 * 3];
    gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf32_to_rgb_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf32_to_rgb_row must match LE");
  }

  // ---- gbrpf32_to_rgba_row -------------------------------------------------

  #[test]
  fn gbrpf32_to_rgba_fills_alpha_max() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let mut out = [0u8; 4];
    gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], 0xFF, "alpha must be 0xFF");
  }

  #[test]
  fn gbrpf32_to_rgba_clamps_and_scales() {
    let vals = [0.0f32, 0.5, 1.0, 1.5, -0.1];
    let expected = [0u8, 128, 255, 255, 0];
    for (v, e) in vals.iter().zip(expected.iter()) {
      let g = as_le_f32(&[*v]);
      let b = as_le_f32(&[*v]);
      let r = as_le_f32(&[*v]);
      let mut out = [0u8; 4];
      gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut out, 1);
      assert_eq!(out[0], *e, "R: v={v}");
      assert_eq!(out[3], 0xFF, "alpha must remain 0xFF");
    }
  }

  #[test]
  fn gbrpf32_to_rgba_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = std::vec![0u8; 4 * 4];
    let mut be_out = std::vec![0u8; 4 * 4];
    gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf32_to_rgba_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf32_to_rgba_row must match LE");
  }

  // ---- gbrpf32_to_rgb_u16_row ----------------------------------------------

  #[test]
  fn gbrpf32_to_rgb_u16_clamps_and_scales() {
    // NaN passes through f32::clamp unchanged (Rust 1.50+); `NaN as u16` saturates to 0.
    let vals = [0.0f32, 0.5, 1.0, 1.5, -0.1, f32::NAN];
    // 0.5 → (0.5 * 65535 + 0.5) as u16 = 32768; NaN → 0
    let expected = [0u16, 32768, 65535, 65535, 0, 0];
    for (v, e) in vals.iter().zip(expected.iter()) {
      let g = as_le_f32(&[*v]);
      let b = as_le_f32(&[*v]);
      let r = as_le_f32(&[*v]);
      let mut out = [0u16; 3];
      gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut out, 1);
      assert_eq!(out[0], *e, "R u16: v={v}");
      assert_eq!(out[1], *e, "G u16: v={v}");
      assert_eq!(out[2], *e, "B u16: v={v}");
    }
  }

  #[test]
  fn gbrpf32_to_rgb_u16_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = std::vec![0u16; 4 * 3];
    let mut be_out = std::vec![0u16; 4 * 3];
    gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf32_to_rgb_u16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf32_to_rgb_u16_row must match LE");
  }

  // ---- gbrpf32_to_rgba_u16_row ---------------------------------------------

  #[test]
  fn gbrpf32_to_rgba_u16_fills_alpha_max() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let mut out = [0u16; 4];
    gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }

  #[test]
  fn gbrpf32_to_rgba_u16_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = std::vec![0u16; 4 * 4];
    let mut be_out = std::vec![0u16; 4 * 4];
    gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf32_to_rgba_u16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf32_to_rgba_u16_row must match LE");
  }

  // ---- gbrpf32_to_rgb_f32_row (lossless) ------------------------------------

  #[test]
  fn gbrpf32_to_rgb_f32_lossless_passthrough() {
    // HDR 2.5, NaN, Inf, negative all preserved bit-exact. Output is host-
    // native f32 after the kernel's `from_le` decode; compare against the
    // original host-native values rather than the LE-encoded inputs.
    let host_g = [2.5f32, f32::NAN, f32::INFINITY, -1.0];
    let host_b = [0.1f32, 0.2, 0.3, 0.4];
    let host_r = [0.5f32, 0.6, 0.7, 0.8];
    let g = as_le_f32(&host_g);
    let b = as_le_f32(&host_b);
    let r = as_le_f32(&host_r);
    let mut out = [0.0f32; 12];
    gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut out, 4);
    // Check R channel (index 0, 3, 6, 9 in RGBA interleave = index 0, 3, 6, 9)
    assert_eq!(out[0], host_r[0]);
    assert_eq!(out[3], host_r[1]);
    assert_eq!(out[6], host_r[2]);
    assert_eq!(out[9], host_r[3]);
    // Check G channel (index 1, 4, 7, 10)
    assert_eq!(out[1], host_g[0], "G HDR preserved");
    assert!(out[4].is_nan(), "G NaN preserved");
    assert!(out[7].is_infinite() && out[7] > 0.0, "G +Inf preserved");
    assert_eq!(out[10], host_g[3], "G negative preserved");
  }

  #[test]
  fn gbrpf32_to_rgb_f32_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = std::vec![0.0f32; 4 * 3];
    let mut be_out = std::vec![0.0f32; 4 * 3];
    gbrpf32_to_rgb_f32_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf32_to_rgb_f32_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf32_to_rgb_f32_row must match LE");
  }

  // ---- gbrpf32_to_rgba_f32_row (lossless, α = 1.0) -------------------------

  #[test]
  fn gbrpf32_to_rgba_f32_alpha_is_one() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let mut out = [0.0f32; 4];
    gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], 1.0, "alpha must be 1.0");
  }

  #[test]
  fn gbrpf32_to_rgba_f32_lossless_passthrough() {
    let r = as_le_f32(&[2.5f32]);
    let g = as_le_f32(&[f32::NAN]);
    let b = as_le_f32(&[f32::NEG_INFINITY]);
    let mut out = [0.0f32; 4];
    gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], 2.5, "R HDR preserved");
    assert!(out[1].is_nan(), "G NaN preserved");
    assert!(out[2].is_infinite() && out[2] < 0.0, "B -Inf preserved");
    assert_eq!(out[3], 1.0, "alpha = 1.0");
  }

  #[test]
  fn gbrpf32_to_rgba_f32_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = std::vec![0.0f32; 4 * 4];
    let mut be_out = std::vec![0.0f32; 4 * 4];
    gbrpf32_to_rgba_f32_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf32_to_rgba_f32_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf32_to_rgba_f32_row must match LE");
  }

  // ---- gbrpf32_to_rgb_f16_row ----------------------------------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf32_to_rgb_f16_normal_values() {
    let g = [0.0f32, 0.5, 1.0];
    let b = [0.25f32, 0.75, 0.0];
    let r = [1.0f32, 0.0, 0.5];
    let mut out = vec![half::f16::ZERO; 9];
    gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut out, 3);
    assert_eq!(out[0], half::f16::from_f32(1.0), "R[0]");
    assert_eq!(out[1], half::f16::from_f32(0.0), "G[0]");
    assert_eq!(out[2], half::f16::from_f32(0.25), "B[0]");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf32_to_rgb_f16_hdr_saturates_to_inf() {
    // Input 70000.0 > f16 max (~65504) → +Inf
    let g = [70_000.0f32];
    let b = [-70_000.0f32];
    let r = [0.5f32];
    let mut out = vec![half::f16::ZERO; 3];
    gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut out, 1);
    // G maps to index 1
    assert!(out[1].is_infinite() && out[1].to_f32() > 0.0, "G +Inf");
    // B maps to index 2
    assert!(out[2].is_infinite() && out[2].to_f32() < 0.0, "B -Inf");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf32_to_rgb_f16_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = vec![half::f16::ZERO; 4 * 3];
    let mut be_out = vec![half::f16::ZERO; 4 * 3];
    gbrpf32_to_rgb_f16_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf32_to_rgb_f16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf32_to_rgb_f16_row must match LE");
  }

  // ---- gbrpf32_to_rgba_f16_row ---------------------------------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf32_to_rgba_f16_alpha_is_one() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let mut out = vec![half::f16::ZERO; 4];
    gbrpf32_to_rgba_f16_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], half::f16::from_f32(1.0), "alpha must be f16(1.0)");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf32_to_rgba_f16_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = vec![half::f16::ZERO; 4 * 4];
    let mut be_out = vec![half::f16::ZERO; 4 * 4];
    gbrpf32_to_rgba_f16_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf32_to_rgba_f16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf32_to_rgba_f16_row must match LE");
  }

  // ---- gbrpf32_to_luma_row -------------------------------------------------

  #[test]
  fn gbrpf32_to_luma_zero_gives_zero() {
    let g = [0.0f32];
    let b = [0.0f32];
    let r = [0.0f32];
    let mut out = [0xFFu8; 1];
    gbrpf32_to_luma_row::<false>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert_eq!(out[0], 0);
  }

  #[test]
  fn gbrpf32_to_luma_max_gives_255() {
    let g = as_le_f32(&[1.0f32]);
    let b = as_le_f32(&[1.0f32]);
    let r = as_le_f32(&[1.0f32]);
    let mut out = [0u8; 1];
    gbrpf32_to_luma_row::<false>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert_eq!(out[0], 255);
  }

  #[test]
  fn gbrpf32_to_luma_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = std::vec![0u8; 4];
    let mut be_out = std::vec![0u8; 4];
    gbrpf32_to_luma_row::<false>(&g, &b, &r, &mut le_out, 4, ColorMatrix::Bt709, true);
    gbrpf32_to_luma_row::<true>(
      &g_be,
      &b_be,
      &r_be,
      &mut be_out,
      4,
      ColorMatrix::Bt709,
      true,
    );
    assert_eq!(be_out, le_out, "BE gbrpf32_to_luma_row must match LE");
  }

  // ---- gbrpf32_to_luma_u16_row ---------------------------------------------

  #[test]
  fn gbrpf32_to_luma_u16_zero_gives_zero() {
    let g = [0.0f32];
    let b = [0.0f32];
    let r = [0.0f32];
    let mut out = [0xFFFFu16; 1];
    gbrpf32_to_luma_u16_row::<false>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert_eq!(out[0], 0);
  }

  #[test]
  fn gbrpf32_to_luma_u16_max_gives_255_zero_extended() {
    let g = as_le_f32(&[1.0f32]);
    let b = as_le_f32(&[1.0f32]);
    let r = as_le_f32(&[1.0f32]);
    let mut out = [0u16; 1];
    gbrpf32_to_luma_u16_row::<false>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert_eq!(out[0], 255, "luma_u16 is zero-extended u8 luma");
  }

  #[test]
  fn gbrpf32_to_luma_u16_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_out = std::vec![0u16; 4];
    let mut be_out = std::vec![0u16; 4];
    gbrpf32_to_luma_u16_row::<false>(&g, &b, &r, &mut le_out, 4, ColorMatrix::Bt709, true);
    gbrpf32_to_luma_u16_row::<true>(
      &g_be,
      &b_be,
      &r_be,
      &mut be_out,
      4,
      ColorMatrix::Bt709,
      true,
    );
    assert_eq!(be_out, le_out, "BE gbrpf32_to_luma_u16_row must match LE");
  }

  // ---- gbrpf32_to_hsv_row --------------------------------------------------

  #[test]
  fn gbrpf32_to_hsv_achromatic_black() {
    let g = [0.0f32];
    let b = [0.0f32];
    let r = [0.0f32];
    let mut h = [0xFFu8; 1];
    let mut s = [0xFFu8; 1];
    let mut v = [0xFFu8; 1];
    gbrpf32_to_hsv_row::<false>(&g, &b, &r, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 0, "V must be 0 for black");
    assert_eq!(s[0], 0, "S must be 0 for achromatic");
  }

  #[test]
  fn gbrpf32_to_hsv_achromatic_white() {
    let g = as_le_f32(&[1.0f32]);
    let b = as_le_f32(&[1.0f32]);
    let r = as_le_f32(&[1.0f32]);
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    gbrpf32_to_hsv_row::<false>(&g, &b, &r, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 255, "V must be 255 for white");
    assert_eq!(s[0], 0, "S must be 0 for achromatic");
  }

  #[test]
  fn gbrpf32_to_hsv_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let mut le_h = std::vec![0u8; 4];
    let mut le_s = std::vec![0u8; 4];
    let mut le_v = std::vec![0u8; 4];
    let mut be_h = std::vec![0u8; 4];
    let mut be_s = std::vec![0u8; 4];
    let mut be_v = std::vec![0u8; 4];
    gbrpf32_to_hsv_row::<false>(&g, &b, &r, &mut le_h, &mut le_s, &mut le_v, 4);
    gbrpf32_to_hsv_row::<true>(&g_be, &b_be, &r_be, &mut be_h, &mut be_s, &mut be_v, 4);
    assert_eq!(be_h, le_h, "BE hsv H must match LE");
    assert_eq!(be_s, le_s, "BE hsv S must match LE");
    assert_eq!(be_v, le_v, "BE hsv V must match LE");
  }

  // ---- gbrapf32_to_rgba_row ------------------------------------------------

  #[test]
  fn gbrapf32_to_rgba_source_alpha_passthrough() {
    let g = as_le_f32(&[0.5f32]);
    let b = as_le_f32(&[0.5f32]);
    let r = as_le_f32(&[0.5f32]);
    let a = as_le_f32(&[0.5f32]);
    let mut out = [0u8; 4];
    gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut out, 1);
    // 0.5 → (0.5 * 255 + 0.5) as u8 = 128
    assert_eq!(out[3], 128, "alpha from source plane");
  }

  #[test]
  fn gbrapf32_to_rgba_source_alpha_clamps() {
    let g = as_le_f32(&[0.5f32]);
    let b = as_le_f32(&[0.5f32]);
    let r = as_le_f32(&[0.5f32]);
    // Test α > 1.0 → 255 and α < 0.0 → 0
    let a_high = as_le_f32(&[1.5f32]);
    let a_low = as_le_f32(&[-0.1f32]);
    let mut out_high = [0u8; 4];
    let mut out_low = [0u8; 4];
    gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a_high, &mut out_high, 1);
    gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a_low, &mut out_low, 1);
    assert_eq!(out_high[3], 255, "alpha HDR clamps to 255");
    assert_eq!(out_low[3], 0, "alpha negative clamps to 0");
  }

  #[test]
  fn gbrapf32_to_rgba_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let a = [0.2f32, 0.4, 0.6, 0.8];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let a_be = be_encode(&a);
    let mut le_out = std::vec![0u8; 4 * 4];
    let mut be_out = std::vec![0u8; 4 * 4];
    gbrapf32_to_rgba_row::<false>(&g, &b, &r, &a, &mut le_out, 4);
    gbrapf32_to_rgba_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrapf32_to_rgba_row must match LE");
  }

  // ---- gbrapf32_to_rgba_u16_row --------------------------------------------

  #[test]
  fn gbrapf32_to_rgba_u16_source_alpha_passthrough() {
    let g = as_le_f32(&[0.5f32]);
    let b = as_le_f32(&[0.5f32]);
    let r = as_le_f32(&[0.5f32]);
    let a = as_le_f32(&[0.5f32]);
    let mut out = [0u16; 4];
    gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut out, 1);
    // 0.5 → (0.5 * 65535 + 0.5) as u16 = 32768
    assert_eq!(out[3], 32768, "u16 alpha from source plane");
  }

  #[test]
  fn gbrapf32_to_rgba_u16_source_alpha_clamps() {
    let g = as_le_f32(&[0.5f32]);
    let b = as_le_f32(&[0.5f32]);
    let r = as_le_f32(&[0.5f32]);
    let a_high = as_le_f32(&[1.5f32]);
    let a_low = as_le_f32(&[-0.1f32]);
    let mut out_high = [0u16; 4];
    let mut out_low = [0u16; 4];
    gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a_high, &mut out_high, 1);
    gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a_low, &mut out_low, 1);
    assert_eq!(out_high[3], 65535, "u16 alpha HDR clamps to 65535");
    assert_eq!(out_low[3], 0, "u16 alpha negative clamps to 0");
  }

  #[test]
  fn gbrapf32_to_rgba_u16_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let a = [0.2f32, 0.4, 0.6, 0.8];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let a_be = be_encode(&a);
    let mut le_out = std::vec![0u16; 4 * 4];
    let mut be_out = std::vec![0u16; 4 * 4];
    gbrapf32_to_rgba_u16_row::<false>(&g, &b, &r, &a, &mut le_out, 4);
    gbrapf32_to_rgba_u16_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrapf32_to_rgba_u16_row must match LE");
  }

  // ---- gbrapf32_to_rgba_f32_row (lossless source α) -------------------------

  #[test]
  fn gbrapf32_to_rgba_f32_lossless_passthrough() {
    // HDR 2.5, NaN, Inf, negative all preserved — including in α
    let g = as_le_f32(&[0.5f32]);
    let b = as_le_f32(&[0.5f32]);
    let r = as_le_f32(&[0.5f32]);
    let a = as_le_f32(&[2.5f32]);
    let mut out = [0.0f32; 4];
    gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[3], 2.5, "HDR alpha preserved bit-exact");
  }

  #[test]
  fn gbrapf32_to_rgba_f32_nan_alpha_preserved() {
    let g = as_le_f32(&[0.5f32]);
    let b = as_le_f32(&[0.5f32]);
    let r = as_le_f32(&[0.5f32]);
    let a = as_le_f32(&[f32::NAN]);
    let mut out = [0.0f32; 4];
    gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut out, 1);
    assert!(out[3].is_nan(), "NaN alpha preserved");
  }

  #[test]
  fn gbrapf32_to_rgba_f32_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let a = [0.2f32, 0.4, 0.6, 0.8];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let a_be = be_encode(&a);
    let mut le_out = std::vec![0.0f32; 4 * 4];
    let mut be_out = std::vec![0.0f32; 4 * 4];
    gbrapf32_to_rgba_f32_row::<false>(&g, &b, &r, &a, &mut le_out, 4);
    gbrapf32_to_rgba_f32_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrapf32_to_rgba_f32_row must match LE");
  }

  // ---- gbrapf32_to_rgba_f16_row --------------------------------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrapf32_to_rgba_f16_source_alpha_passthrough() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let a = [0.75f32];
    let mut out = vec![half::f16::ZERO; 4];
    gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[3], half::f16::from_f32(0.75), "f16 alpha from source");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrapf32_to_rgba_f16_hdr_alpha_saturates() {
    let g = [0.5f32];
    let b = [0.5f32];
    let r = [0.5f32];
    let a = [70_000.0f32];
    let mut out = vec![half::f16::ZERO; 4];
    gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut out, 1);
    assert!(
      out[3].is_infinite() && out[3].to_f32() > 0.0,
      "HDR alpha saturates to +Inf"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrapf32_to_rgba_f16_be_parity() {
    let g = [0.0f32, 0.25, 0.5, 1.0];
    let b = [0.1f32, 0.3, 0.7, 0.9];
    let r = [0.5f32, 0.8, 0.2, 0.6];
    let a = [0.2f32, 0.4, 0.6, 0.8];
    let g_be = be_encode(&g);
    let b_be = be_encode(&b);
    let r_be = be_encode(&r);
    let a_be = be_encode(&a);
    let mut le_out = vec![half::f16::ZERO; 4 * 4];
    let mut be_out = vec![half::f16::ZERO; 4 * 4];
    gbrapf32_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut le_out, 4);
    gbrapf32_to_rgba_f16_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrapf32_to_rgba_f16_row must match LE");
  }
}
