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
mod tests {
  use super::*;

  // ---- Rgb48 ---------------------------------------------------------------

  /// All-white input: u16 passthrough should produce all-0xFFFF.
  #[test]
  fn rgb48_to_rgb_u16_all_white_passthrough() {
    let src = std::vec![0xFFFFu16; 3 * 4];
    let mut out = std::vec![0u16; 3 * 4];
    rgb48_to_rgb_u16_row::<false>(&src, &mut out, 4);
    assert!(
      out.iter().all(|&v| v == 0xFFFF),
      "expected all 0xFFFF, got {out:?}"
    );
  }

  /// All-white input narrowed to u8 should produce all-0xFF.
  #[test]
  fn rgb48_to_rgb_all_white_narrow() {
    let src = std::vec![0xFFFFu16; 3 * 4];
    let mut out = std::vec![0u8; 3 * 4];
    rgb48_to_rgb_row::<false>(&src, &mut out, 4);
    assert!(
      out.iter().all(|&v| v == 0xFF),
      "expected all 0xFF, got {out:?}"
    );
  }

  /// Known value: 0x1234 >> 8 = 0x12.
  #[test]
  fn rgb48_to_rgb_narrow_known_value() {
    let src = [0x1234u16, 0x5678, 0x9ABC];
    let mut out = [0u8; 3];
    rgb48_to_rgb_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x12, "R channel");
    assert_eq!(out[1], 0x56, "G channel");
    assert_eq!(out[2], 0x9A, "B channel");
  }

  /// rgba output forces alpha = 0xFF.
  #[test]
  fn rgb48_to_rgba_forces_alpha_0xff() {
    let src = [0xAAAAu16, 0xBBBB, 0xCCCC];
    let mut out = [0u8; 4];
    rgb48_to_rgba_row::<false>(&src, &mut out, 1);
    assert_eq!(out[3], 0xFF, "alpha must be 0xFF");
    assert_eq!(out[0], 0xAA, "R");
    assert_eq!(out[1], 0xBB, "G");
    assert_eq!(out[2], 0xCC, "B");
  }

  /// rgba_u16 output forces alpha = 0xFFFF.
  #[test]
  fn rgb48_to_rgba_u16_forces_alpha_0xffff() {
    let src = [0xAAAAu16, 0xBBBB, 0xCCCC];
    let mut out = [0u16; 4];
    rgb48_to_rgba_u16_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0xAAAA, "R");
    assert_eq!(out[1], 0xBBBB, "G");
    assert_eq!(out[2], 0xCCCC, "B");
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }

  // ---- Bgr48 ---------------------------------------------------------------

  /// All-white input: u16 passthrough should produce all-0xFFFF (order unchanged since all equal).
  #[test]
  fn bgr48_to_rgb_u16_all_white_passthrough() {
    let src = std::vec![0xFFFFu16; 3 * 3];
    let mut out = std::vec![0u16; 3 * 3];
    bgr48_to_rgb_u16_row::<false>(&src, &mut out, 3);
    assert!(out.iter().all(|&v| v == 0xFFFF), "expected all 0xFFFF");
  }

  /// All-white input narrowed to u8 should produce all-0xFF.
  #[test]
  fn bgr48_to_rgb_all_white_narrow() {
    let src = std::vec![0xFFFFu16; 3 * 3];
    let mut out = std::vec![0u8; 3 * 3];
    bgr48_to_rgb_row::<false>(&src, &mut out, 3);
    assert!(out.iter().all(|&v| v == 0xFF), "expected all 0xFF");
  }

  /// Channel-order swap: Bgr48 `[B=0x1234, G=0x5678, R=0x9ABC]`
  /// → `with_rgb_u16` → `[R=0x9ABC, G=0x5678, B=0x1234]`.
  #[test]
  fn bgr48_to_rgb_u16_channel_order_swapped() {
    // Source pixel in BGR order: B=0x1234, G=0x5678, R=0x9ABC
    let src = [0x1234u16, 0x5678, 0x9ABC];
    let mut out = [0u16; 3];
    bgr48_to_rgb_u16_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x9ABC, "R (was at src[2])");
    assert_eq!(out[1], 0x5678, "G (unchanged)");
    assert_eq!(out[2], 0x1234, "B (was at src[0])");
  }

  /// u8 RGB output: same swap + narrow.
  #[test]
  fn bgr48_to_rgb_channel_order_and_narrow() {
    let src = [0x1200u16, 0x5600, 0x9A00];
    let mut out = [0u8; 3];
    bgr48_to_rgb_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x9A, "R");
    assert_eq!(out[1], 0x56, "G");
    assert_eq!(out[2], 0x12, "B");
  }

  /// rgba output: swapped channels + forced alpha = 0xFF.
  #[test]
  fn bgr48_to_rgba_channel_order_and_alpha() {
    let src = [0x1100u16, 0x2200, 0x3300];
    let mut out = [0u8; 4];
    bgr48_to_rgba_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x33, "R");
    assert_eq!(out[1], 0x22, "G");
    assert_eq!(out[2], 0x11, "B");
    assert_eq!(out[3], 0xFF, "alpha must be 0xFF");
  }

  /// rgba_u16 output: swapped channels + forced alpha = 0xFFFF.
  #[test]
  fn bgr48_to_rgba_u16_channel_order_and_alpha() {
    let src = [0x1111u16, 0x2222, 0x3333];
    let mut out = [0u16; 4];
    bgr48_to_rgba_u16_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x3333, "R");
    assert_eq!(out[1], 0x2222, "G");
    assert_eq!(out[2], 0x1111, "B");
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }

  // ---- Rgba64 --------------------------------------------------------------

  /// All-white input: u16 identity copy produces all-0xFFFF.
  #[test]
  fn rgba64_to_rgba_u16_all_white_passthrough() {
    let src = std::vec![0xFFFFu16; 4 * 3];
    let mut out = std::vec![0u16; 4 * 3];
    rgba64_to_rgba_u16_row::<false>(&src, &mut out, 3);
    assert!(out.iter().all(|&v| v == 0xFFFF), "expected all 0xFFFF");
  }

  /// All-white narrowed to u8 produces all-0xFF.
  #[test]
  fn rgba64_to_rgba_all_white_narrow() {
    let src = std::vec![0xFFFFu16; 4 * 3];
    let mut out = std::vec![0u8; 4 * 3];
    rgba64_to_rgba_row::<false>(&src, &mut out, 3);
    assert!(out.iter().all(|&v| v == 0xFF), "expected all 0xFF");
  }

  /// Source alpha is preserved in u16 passthrough at position 3.
  #[test]
  fn rgba64_to_rgba_u16_source_alpha_preserved() {
    // R=0x1111, G=0x2222, B=0x3333, A=0xABCD
    let src = [0x1111u16, 0x2222, 0x3333, 0xABCD];
    let mut out = [0u16; 4];
    rgba64_to_rgba_u16_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x1111, "R");
    assert_eq!(out[1], 0x2222, "G");
    assert_eq!(out[2], 0x3333, "B");
    assert_eq!(out[3], 0xABCD, "alpha must be preserved as-is");
  }

  /// Source alpha is depth-converted (>> 8) in u8 rgba output.
  #[test]
  fn rgba64_to_rgba_source_alpha_depth_converted() {
    let src = [0x1100u16, 0x2200, 0x3300, 0xABCD];
    let mut out = [0u8; 4];
    rgba64_to_rgba_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x11, "R");
    assert_eq!(out[1], 0x22, "G");
    assert_eq!(out[2], 0x33, "B");
    assert_eq!(out[3], 0xAB, "alpha narrowed >> 8");
  }

  /// rgb path drops alpha, narrows.
  #[test]
  fn rgba64_to_rgb_drops_alpha() {
    let src = [0x1100u16, 0x2200, 0x3300, 0xDEAD];
    let mut out = [0u8; 3];
    rgba64_to_rgb_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x11, "R");
    assert_eq!(out[1], 0x22, "G");
    assert_eq!(out[2], 0x33, "B");
  }

  /// rgb_u16 path drops alpha, copies native u16.
  #[test]
  fn rgba64_to_rgb_u16_drops_alpha() {
    let src = [0x1111u16, 0x2222, 0x3333, 0xDEAD];
    let mut out = [0u16; 3];
    rgba64_to_rgb_u16_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x1111, "R");
    assert_eq!(out[1], 0x2222, "G");
    assert_eq!(out[2], 0x3333, "B");
  }

  // ---- Bgra64 --------------------------------------------------------------

  /// All-white input: u16 identity copy (swap is no-op for all-equal channels).
  #[test]
  fn bgra64_to_rgba_u16_all_white_passthrough() {
    let src = std::vec![0xFFFFu16; 4 * 2];
    let mut out = std::vec![0u16; 4 * 2];
    bgra64_to_rgba_u16_row::<false>(&src, &mut out, 2);
    assert!(out.iter().all(|&v| v == 0xFFFF), "expected all 0xFFFF");
  }

  /// All-white narrowed to u8 produces all-0xFF.
  #[test]
  fn bgra64_to_rgba_all_white_narrow() {
    let src = std::vec![0xFFFFu16; 4 * 2];
    let mut out = std::vec![0u8; 4 * 2];
    bgra64_to_rgba_row::<false>(&src, &mut out, 2);
    assert!(out.iter().all(|&v| v == 0xFF), "expected all 0xFF");
  }

  /// Channel order swap + alpha preserved: Bgra64 `[B, G, R, A]` → `[R, G, B, A]`.
  #[test]
  fn bgra64_to_rgba_u16_channel_order_and_alpha_preserved() {
    // Source in BGRA order: B=0x1111, G=0x2222, R=0x3333, A=0x4444
    let src = [0x1111u16, 0x2222, 0x3333, 0x4444];
    let mut out = [0u16; 4];
    bgra64_to_rgba_u16_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x3333, "R (from src[2])");
    assert_eq!(out[1], 0x2222, "G (unchanged)");
    assert_eq!(out[2], 0x1111, "B (from src[0])");
    assert_eq!(out[3], 0x4444, "A preserved as-is");
  }

  /// u8 rgba output: swap + narrow + source alpha depth-converted.
  #[test]
  fn bgra64_to_rgba_channel_order_and_alpha_narrowed() {
    let src = [0x1100u16, 0x2200, 0x3300, 0xAB00];
    let mut out = [0u8; 4];
    bgra64_to_rgba_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x33, "R");
    assert_eq!(out[1], 0x22, "G");
    assert_eq!(out[2], 0x11, "B");
    assert_eq!(out[3], 0xAB, "alpha narrowed >> 8");
  }

  /// rgb path drops alpha and swaps channels.
  #[test]
  fn bgra64_to_rgb_drops_alpha_and_swaps() {
    let src = [0x1100u16, 0x2200, 0x3300, 0xDEAD];
    let mut out = [0u8; 3];
    bgra64_to_rgb_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x33, "R");
    assert_eq!(out[1], 0x22, "G");
    assert_eq!(out[2], 0x11, "B");
  }

  /// rgb_u16 path drops alpha, swaps, native copy.
  #[test]
  fn bgra64_to_rgb_u16_drops_alpha_and_swaps() {
    let src = [0x1111u16, 0x2222, 0x3333, 0xDEAD];
    let mut out = [0u16; 3];
    bgra64_to_rgb_u16_row::<false>(&src, &mut out, 1);
    assert_eq!(out[0], 0x3333, "R");
    assert_eq!(out[1], 0x2222, "G");
    assert_eq!(out[2], 0x1111, "B");
  }

  // ---- Multi-pixel width tests ---------------------------------------------

  /// Width=3 Rgb48→rgb: verify correct stride indexing.
  #[test]
  fn rgb48_to_rgb_multi_pixel_width() {
    // 3 pixels: [R0=0x1100, G0=0x2200, B0=0x3300], [R1=0x4400, G1=0x5500, B1=0x6600],
    //           [R2=0x7700, G2=0x8800, B2=0x9900]
    let src = [
      0x1100u16, 0x2200, 0x3300, 0x4400, 0x5500, 0x6600, 0x7700, 0x8800, 0x9900,
    ];
    let mut out = [0u8; 9];
    rgb48_to_rgb_row::<false>(&src, &mut out, 3);
    assert_eq!(out[0], 0x11);
    assert_eq!(out[1], 0x22);
    assert_eq!(out[2], 0x33);
    assert_eq!(out[3], 0x44);
    assert_eq!(out[4], 0x55);
    assert_eq!(out[5], 0x66);
    assert_eq!(out[6], 0x77);
    assert_eq!(out[7], 0x88);
    assert_eq!(out[8], 0x99);
  }

  /// Width=2 Rgba64→rgba_u16: identity copy preserves layout.
  #[test]
  fn rgba64_to_rgba_u16_multi_pixel_identity() {
    let src = [
      0x1111u16, 0x2222, 0x3333, 0x4444, // pixel 0
      0x5555, 0x6666, 0x7777, 0x8888, // pixel 1
    ];
    let mut out = [0u16; 8];
    rgba64_to_rgba_u16_row::<false>(&src, &mut out, 2);
    assert_eq!(&out, &src, "identity copy must be byte-exact");
  }

  /// Bgr48 and Rgb48 on mirrored input produce same rgb output.
  #[test]
  fn bgr48_rgb_output_matches_rgb48_with_swapped_input() {
    // RGB input: R=0xAAAA, G=0xBBBB, B=0xCCCC
    let rgb48_src = [0xAAAAu16, 0xBBBB, 0xCCCC];
    // BGR input: B=0xCCCC, G=0xBBBB, R=0xAAAA
    let bgr48_src = [0xCCCCu16, 0xBBBB, 0xAAAA];

    let mut rgb48_out = [0u8; 3];
    let mut bgr48_out = [0u8; 3];
    rgb48_to_rgb_row::<false>(&rgb48_src, &mut rgb48_out, 1);
    bgr48_to_rgb_row::<false>(&bgr48_src, &mut bgr48_out, 1);

    assert_eq!(
      rgb48_out, bgr48_out,
      "RGB48 and BGR48 mirrored inputs must produce same RGB output"
    );
  }
}
