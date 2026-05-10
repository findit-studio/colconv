//! Scalar reference kernels for 16-bit packed RGB sources (Tier 8 finish).
//!
//! Input planes are `&[u16]`. Each u16 sample is either LE- or BE-encoded on
//! disk/wire; the `<const BE: bool>` const-generic parameter selects the
//! interpretation.  When `BE = false` the input is LE-encoded; when `BE = true`
//! the input is BE-encoded.  In both cases each element is converted to
//! host-native byte order on load via `u16::from_le` / `u16::from_be`, which
//! are no-ops when the source byte order already matches the host.  This
//! mirrors the SIMD `load_endian_u16x*` helpers and keeps the scalar reference
//! correct on big-endian hosts (s390x).
//!
//! # Format layouts
//!
//! | Format  | Elements per pixel | Channel order |
//! |---------|--------------------|---------------|
//! | Rgb48   | 3                  | R, G, B       |
//! | Bgr48   | 3                  | B, G, R       |
//! | Rgba64  | 4                  | R, G, B, A    |
//! | Bgra64  | 4                  | B, G, R, A    |
//!
//! # Depth-conversion convention
//!
//! - u16 → u8: `(v >> 8) as u8` (high-byte extraction, matching Y216 / Ship 11d).
//! - u16 → u16: identity copy (no scaling).

// ---- Endian load helper ------------------------------------------------------

/// Load one u16 element from a source whose byte order is selected by `BE`,
/// returning the value in host-native byte order.
///
/// `u16::from_be` / `u16::from_le` are target-endian aware: each is a no-op
/// when the source byte order matches the host, and a `swap_bytes` otherwise.
/// This matches the SIMD `load_endian_u16x*` helpers and keeps the scalar
/// reference correct on big-endian hosts (s390x).
///
/// The `if BE` branch is evaluated at compile time (monomorphization), so the
/// unused branch is entirely eliminated from the generated binary.
#[inline(always)]
fn load_u16<const BE: bool>(v: u16) -> u16 {
  if BE { u16::from_be(v) } else { u16::from_le(v) }
}

// ---- Rgb48 family (3 u16 elements per pixel: R, G, B) ----------------------

/// Rgb48 → packed u8 RGB: narrow each 16-bit channel via `>> 8`.
///
/// When `BE = true` each u16 element is byte-swapped on load so the channel
/// value is in host-native order before narrowing.
///
/// Input stride: `width * 3` u16 elements, output: `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb48_to_rgb_row<const BE: bool>(rgb48: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 3;
    rgb_out[dst] = (load_u16::<BE>(rgb48[src]) >> 8) as u8;
    rgb_out[dst + 1] = (load_u16::<BE>(rgb48[src + 1]) >> 8) as u8;
    rgb_out[dst + 2] = (load_u16::<BE>(rgb48[src + 2]) >> 8) as u8;
  }
}

/// Rgb48 → packed u16 RGB: copy with optional byte-swap (already R, G, B order).
///
/// When `BE = true` each element is byte-swapped so the output contains
/// host-native u16 values.
///
/// Input and output stride: `width * 3` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb48_to_rgb_u16_row<const BE: bool>(
  rgb48: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  if BE {
    for i in 0..width * 3 {
      rgb_u16_out[i] = u16::from_be(rgb48[i]);
    }
  } else {
    // LE source: use the target-endian-aware load on each element so big-endian
    // hosts also receive host-native u16 output.
    for i in 0..width * 3 {
      rgb_u16_out[i] = u16::from_le(rgb48[i]);
    }
  }
}

/// Rgb48 → packed u8 RGBA: narrow each 16-bit channel via `>> 8`, force alpha = 0xFF.
///
/// When `BE = true` each u16 element is byte-swapped on load.
///
/// Input stride: `width * 3` u16 elements, output: `width * 4` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb48_to_rgba_row<const BE: bool>(rgb48: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_out[dst] = (load_u16::<BE>(rgb48[src]) >> 8) as u8;
    rgba_out[dst + 1] = (load_u16::<BE>(rgb48[src + 1]) >> 8) as u8;
    rgba_out[dst + 2] = (load_u16::<BE>(rgb48[src + 2]) >> 8) as u8;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Rgb48 → packed u16 RGBA: copy R/G/B (with optional byte-swap), force alpha = 0xFFFF.
///
/// When `BE = true` each element is byte-swapped to produce host-native output.
///
/// Input stride: `width * 3` u16 elements, output: `width * 4` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb48_to_rgba_u16_row<const BE: bool>(
  rgb48: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_u16_out[dst] = load_u16::<BE>(rgb48[src]);
    rgba_u16_out[dst + 1] = load_u16::<BE>(rgb48[src + 1]);
    rgba_u16_out[dst + 2] = load_u16::<BE>(rgb48[src + 2]);
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- Bgr48 family (3 u16 elements per pixel: B, G, R) ----------------------

/// Bgr48 → packed u8 RGB: narrow via `>> 8`, swap B↔R on output.
///
/// When `BE = true` each u16 element is byte-swapped on load.
///
/// Source layout `[B, G, R]` → output layout `[R, G, B]`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr48_to_rgb_row<const BE: bool>(bgr48: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 3;
    rgb_out[dst] = (load_u16::<BE>(bgr48[src + 2]) >> 8) as u8; // R (from B-G-R position 2)
    rgb_out[dst + 1] = (load_u16::<BE>(bgr48[src + 1]) >> 8) as u8; // G (unchanged)
    rgb_out[dst + 2] = (load_u16::<BE>(bgr48[src]) >> 8) as u8; // B (from B-G-R position 0)
  }
}

/// Bgr48 → packed u16 RGB: copy with B↔R swap (and optional byte-swap).
///
/// When `BE = true` each element is byte-swapped to produce host-native output.
///
/// Source layout `[B, G, R]` → output layout `[R, G, B]`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr48_to_rgb_u16_row<const BE: bool>(
  bgr48: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 3;
    rgb_u16_out[dst] = load_u16::<BE>(bgr48[src + 2]); // R
    rgb_u16_out[dst + 1] = load_u16::<BE>(bgr48[src + 1]); // G
    rgb_u16_out[dst + 2] = load_u16::<BE>(bgr48[src]); // B
  }
}

/// Bgr48 → packed u8 RGBA: narrow + B↔R swap + force alpha = 0xFF.
///
/// When `BE = true` each u16 element is byte-swapped on load.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr48_to_rgba_row<const BE: bool>(bgr48: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_out[dst] = (load_u16::<BE>(bgr48[src + 2]) >> 8) as u8; // R
    rgba_out[dst + 1] = (load_u16::<BE>(bgr48[src + 1]) >> 8) as u8; // G
    rgba_out[dst + 2] = (load_u16::<BE>(bgr48[src]) >> 8) as u8; // B
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Bgr48 → packed u16 RGBA: B↔R swap (+ optional byte-swap) + force alpha = 0xFFFF.
///
/// When `BE = true` each element is byte-swapped to produce host-native output.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr48_to_rgba_u16_row<const BE: bool>(
  bgr48: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_u16_out[dst] = load_u16::<BE>(bgr48[src + 2]); // R
    rgba_u16_out[dst + 1] = load_u16::<BE>(bgr48[src + 1]); // G
    rgba_u16_out[dst + 2] = load_u16::<BE>(bgr48[src]); // B
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- Rgba64 family (4 u16 elements per pixel: R, G, B, A) ------------------

/// Rgba64 → packed u8 RGB: drop alpha, narrow R/G/B via `>> 8`.
///
/// When `BE = true` each u16 element is byte-swapped on load.
///
/// Input stride: `width * 4` u16 elements, output: `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba64_to_rgb_row<const BE: bool>(rgba64: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_out[dst] = (load_u16::<BE>(rgba64[src]) >> 8) as u8;
    rgb_out[dst + 1] = (load_u16::<BE>(rgba64[src + 1]) >> 8) as u8;
    rgb_out[dst + 2] = (load_u16::<BE>(rgba64[src + 2]) >> 8) as u8;
  }
}

/// Rgba64 → packed u16 RGB: drop alpha, copy R/G/B (with optional byte-swap).
///
/// When `BE = true` each element is byte-swapped to produce host-native output.
///
/// Input stride: `width * 4` u16 elements, output: `width * 3` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba64_to_rgb_u16_row<const BE: bool>(
  rgba64: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_u16_out[dst] = load_u16::<BE>(rgba64[src]);
    rgb_u16_out[dst + 1] = load_u16::<BE>(rgba64[src + 1]);
    rgb_u16_out[dst + 2] = load_u16::<BE>(rgba64[src + 2]);
  }
}

/// Rgba64 → packed u8 RGBA: narrow all 4 channels via `>> 8` (source alpha passes through).
///
/// When `BE = true` each u16 element is byte-swapped on load.
///
/// Input and output stride: `width * 4` elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba64_to_rgba_row<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = (load_u16::<BE>(rgba64[i]) >> 8) as u8;
    rgba_out[i + 1] = (load_u16::<BE>(rgba64[i + 1]) >> 8) as u8;
    rgba_out[i + 2] = (load_u16::<BE>(rgba64[i + 2]) >> 8) as u8;
    rgba_out[i + 3] = (load_u16::<BE>(rgba64[i + 3]) >> 8) as u8;
  }
}

/// Rgba64 → packed u16 RGBA: copy all 4 channels (with optional byte-swap).
///
/// When `BE = true` each element is byte-swapped to produce host-native output.
///
/// Input and output stride: `width * 4` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba64_to_rgba_u16_row<const BE: bool>(
  rgba64: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  if BE {
    for i in 0..width * 4 {
      rgba_u16_out[i] = u16::from_be(rgba64[i]);
    }
  } else {
    // LE source: use the target-endian-aware load on each element so big-endian
    // hosts also receive host-native u16 output.
    for i in 0..width * 4 {
      rgba_u16_out[i] = u16::from_le(rgba64[i]);
    }
  }
}

// ---- Bgra64 family (4 u16 elements per pixel: B, G, R, A) ------------------

/// Bgra64 → packed u8 RGB: drop alpha, narrow via `>> 8`, swap B↔R on output.
///
/// When `BE = true` each u16 element is byte-swapped on load.
///
/// Source layout `[B, G, R, A]` → output layout `[R, G, B]`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra64_to_rgb_row<const BE: bool>(bgra64: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_out[dst] = (load_u16::<BE>(bgra64[src + 2]) >> 8) as u8; // R (from position 2)
    rgb_out[dst + 1] = (load_u16::<BE>(bgra64[src + 1]) >> 8) as u8; // G (unchanged)
    rgb_out[dst + 2] = (load_u16::<BE>(bgra64[src]) >> 8) as u8; // B (from position 0)
  }
}

/// Bgra64 → packed u16 RGB: drop alpha, B↔R swap (+ optional byte-swap).
///
/// When `BE = true` each element is byte-swapped to produce host-native output.
///
/// Source layout `[B, G, R, A]` → output layout `[R, G, B]`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra64_to_rgb_u16_row<const BE: bool>(
  bgra64: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_u16_out[dst] = load_u16::<BE>(bgra64[src + 2]); // R
    rgb_u16_out[dst + 1] = load_u16::<BE>(bgra64[src + 1]); // G
    rgb_u16_out[dst + 2] = load_u16::<BE>(bgra64[src]); // B
  }
}

/// Bgra64 → packed u8 RGBA: narrow via `>> 8`, swap B↔R, pass through source alpha.
///
/// When `BE = true` each u16 element is byte-swapped on load.
///
/// Source layout `[B, G, R, A]` → output layout `[R, G, B, A]` (all narrowed `>> 8`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra64_to_rgba_row<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 4;
    rgba_out[dst] = (load_u16::<BE>(bgra64[src + 2]) >> 8) as u8; // R
    rgba_out[dst + 1] = (load_u16::<BE>(bgra64[src + 1]) >> 8) as u8; // G
    rgba_out[dst + 2] = (load_u16::<BE>(bgra64[src]) >> 8) as u8; // B
    rgba_out[dst + 3] = (load_u16::<BE>(bgra64[src + 3]) >> 8) as u8; // A
  }
}

/// Bgra64 → packed u16 RGBA: B↔R swap (+ optional byte-swap), pass through source alpha.
///
/// When `BE = true` each element is byte-swapped to produce host-native output.
///
/// Source layout `[B, G, R, A]` → output layout `[R, G, B, A]` (all native u16).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra64_to_rgba_u16_row<const BE: bool>(
  bgra64: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let src = x * 4;
    let dst = x * 4;
    rgba_u16_out[dst] = load_u16::<BE>(bgra64[src + 2]); // R
    rgba_u16_out[dst + 1] = load_u16::<BE>(bgra64[src + 1]); // G
    rgba_u16_out[dst + 2] = load_u16::<BE>(bgra64[src]); // B
    rgba_u16_out[dst + 3] = load_u16::<BE>(bgra64[src + 3]); // A (byte-order corrected)
  }
}

// ---- Unit tests -------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests;
