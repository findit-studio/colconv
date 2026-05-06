//! Scalar reference kernels for 16-bit packed RGB sources (Tier 8 finish).
//!
//! Input planes are `&[u16]`. Each u16 sample is the native channel value
//! (range [0, 65535]). No endian conversion — caller deserialises LE bytes
//! to `&[u16]` before constructing the frame.
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
// Kernels are wired into the dispatcher (Task 9) and sinker (Task 10) in later
// commits — suppress dead_code until then.
#![allow(dead_code)]

// ---- Rgb48 family (3 u16 elements per pixel: R, G, B) ----------------------

/// Rgb48 → packed u8 RGB: narrow each 16-bit channel via `>> 8`.
///
/// Input stride: `width * 3` u16 elements, output: `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb48_to_rgb_row(rgb48: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 3;
    rgb_out[dst] = (rgb48[src] >> 8) as u8;
    rgb_out[dst + 1] = (rgb48[src + 1] >> 8) as u8;
    rgb_out[dst + 2] = (rgb48[src + 2] >> 8) as u8;
  }
}

/// Rgb48 → packed u16 RGB: identity copy (already R, G, B order).
///
/// Input and output stride: `width * 3` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb48_to_rgb_u16_row(rgb48: &[u16], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  rgb_u16_out[..width * 3].copy_from_slice(&rgb48[..width * 3]);
}

/// Rgb48 → packed u8 RGBA: narrow each 16-bit channel via `>> 8`, force alpha = 0xFF.
///
/// Input stride: `width * 3` u16 elements, output: `width * 4` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb48_to_rgba_row(rgb48: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_out[dst] = (rgb48[src] >> 8) as u8;
    rgba_out[dst + 1] = (rgb48[src + 1] >> 8) as u8;
    rgba_out[dst + 2] = (rgb48[src + 2] >> 8) as u8;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Rgb48 → packed u16 RGBA: copy R/G/B as-is, force alpha = 0xFFFF.
///
/// Input stride: `width * 3` u16 elements, output: `width * 4` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb48_to_rgba_u16_row(rgb48: &[u16], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_u16_out[dst] = rgb48[src];
    rgba_u16_out[dst + 1] = rgb48[src + 1];
    rgba_u16_out[dst + 2] = rgb48[src + 2];
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- Bgr48 family (3 u16 elements per pixel: B, G, R) ----------------------

/// Bgr48 → packed u8 RGB: narrow via `>> 8`, swap B↔R on output.
///
/// Source layout `[B, G, R]` → output layout `[R, G, B]`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr48_to_rgb_row(bgr48: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 3;
    rgb_out[dst] = (bgr48[src + 2] >> 8) as u8; // R (from B-G-R position 2)
    rgb_out[dst + 1] = (bgr48[src + 1] >> 8) as u8; // G (unchanged)
    rgb_out[dst + 2] = (bgr48[src] >> 8) as u8; // B (from B-G-R position 0)
  }
}

/// Bgr48 → packed u16 RGB: copy with B↔R swap.
///
/// Source layout `[B, G, R]` → output layout `[R, G, B]`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr48_to_rgb_u16_row(bgr48: &[u16], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 3;
    rgb_u16_out[dst] = bgr48[src + 2]; // R
    rgb_u16_out[dst + 1] = bgr48[src + 1]; // G
    rgb_u16_out[dst + 2] = bgr48[src]; // B
  }
}

/// Bgr48 → packed u8 RGBA: narrow + B↔R swap + force alpha = 0xFF.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr48_to_rgba_row(bgr48: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_out[dst] = (bgr48[src + 2] >> 8) as u8; // R
    rgba_out[dst + 1] = (bgr48[src + 1] >> 8) as u8; // G
    rgba_out[dst + 2] = (bgr48[src] >> 8) as u8; // B
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Bgr48 → packed u16 RGBA: B↔R swap + force alpha = 0xFFFF.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr48_to_rgba_u16_row(bgr48: &[u16], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let src = x * 3;
    let dst = x * 4;
    rgba_u16_out[dst] = bgr48[src + 2]; // R
    rgba_u16_out[dst + 1] = bgr48[src + 1]; // G
    rgba_u16_out[dst + 2] = bgr48[src]; // B
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- Rgba64 family (4 u16 elements per pixel: R, G, B, A) ------------------

/// Rgba64 → packed u8 RGB: drop alpha, narrow R/G/B via `>> 8`.
///
/// Input stride: `width * 4` u16 elements, output: `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba64_to_rgb_row(rgba64: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_out[dst] = (rgba64[src] >> 8) as u8;
    rgb_out[dst + 1] = (rgba64[src + 1] >> 8) as u8;
    rgb_out[dst + 2] = (rgba64[src + 2] >> 8) as u8;
  }
}

/// Rgba64 → packed u16 RGB: drop alpha, copy R/G/B as-is.
///
/// Input stride: `width * 4` u16 elements, output: `width * 3` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba64_to_rgb_u16_row(rgba64: &[u16], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_u16_out[dst] = rgba64[src];
    rgb_u16_out[dst + 1] = rgba64[src + 1];
    rgb_u16_out[dst + 2] = rgba64[src + 2];
  }
}

/// Rgba64 → packed u8 RGBA: narrow all 4 channels via `>> 8` (source alpha passes through).
///
/// Input and output stride: `width * 4` elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba64_to_rgba_row(rgba64: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let i = x * 4;
    rgba_out[i] = (rgba64[i] >> 8) as u8;
    rgba_out[i + 1] = (rgba64[i + 1] >> 8) as u8;
    rgba_out[i + 2] = (rgba64[i + 2] >> 8) as u8;
    rgba_out[i + 3] = (rgba64[i + 3] >> 8) as u8;
  }
}

/// Rgba64 → packed u16 RGBA: identity copy of all 4 channels.
///
/// Input and output stride: `width * 4` u16 elements.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgba64_to_rgba_u16_row(rgba64: &[u16], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  rgba_u16_out[..width * 4].copy_from_slice(&rgba64[..width * 4]);
}

// ---- Bgra64 family (4 u16 elements per pixel: B, G, R, A) ------------------

/// Bgra64 → packed u8 RGB: drop alpha, narrow via `>> 8`, swap B↔R on output.
///
/// Source layout `[B, G, R, A]` → output layout `[R, G, B]`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra64_to_rgb_row(bgra64: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_out[dst] = (bgra64[src + 2] >> 8) as u8; // R (from position 2)
    rgb_out[dst + 1] = (bgra64[src + 1] >> 8) as u8; // G (unchanged)
    rgb_out[dst + 2] = (bgra64[src] >> 8) as u8; // B (from position 0)
  }
}

/// Bgra64 → packed u16 RGB: drop alpha, B↔R swap.
///
/// Source layout `[B, G, R, A]` → output layout `[R, G, B]`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra64_to_rgb_u16_row(bgra64: &[u16], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 3;
    rgb_u16_out[dst] = bgra64[src + 2]; // R
    rgb_u16_out[dst + 1] = bgra64[src + 1]; // G
    rgb_u16_out[dst + 2] = bgra64[src]; // B
  }
}

/// Bgra64 → packed u8 RGBA: narrow via `>> 8`, swap B↔R, pass through source alpha.
///
/// Source layout `[B, G, R, A]` → output layout `[R, G, B, A]` (all narrowed `>> 8`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra64_to_rgba_row(bgra64: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let src = x * 4;
    let dst = x * 4;
    rgba_out[dst] = (bgra64[src + 2] >> 8) as u8; // R
    rgba_out[dst + 1] = (bgra64[src + 1] >> 8) as u8; // G
    rgba_out[dst + 2] = (bgra64[src] >> 8) as u8; // B
    rgba_out[dst + 3] = (bgra64[src + 3] >> 8) as u8; // A
  }
}

/// Bgra64 → packed u16 RGBA: B↔R swap, pass through source alpha unchanged.
///
/// Source layout `[B, G, R, A]` → output layout `[R, G, B, A]` (all native u16).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgra64_to_rgba_u16_row(bgra64: &[u16], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let src = x * 4;
    let dst = x * 4;
    rgba_u16_out[dst] = bgra64[src + 2]; // R
    rgba_u16_out[dst + 1] = bgra64[src + 1]; // G
    rgba_u16_out[dst + 2] = bgra64[src]; // B
    rgba_u16_out[dst + 3] = bgra64[src + 3]; // A (unchanged)
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
    rgb48_to_rgb_u16_row(&src, &mut out, 4);
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
    rgb48_to_rgb_row(&src, &mut out, 4);
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
    rgb48_to_rgb_row(&src, &mut out, 1);
    assert_eq!(out[0], 0x12, "R channel");
    assert_eq!(out[1], 0x56, "G channel");
    assert_eq!(out[2], 0x9A, "B channel");
  }

  /// rgba output forces alpha = 0xFF.
  #[test]
  fn rgb48_to_rgba_forces_alpha_0xff() {
    let src = [0xAAAAu16, 0xBBBB, 0xCCCC];
    let mut out = [0u8; 4];
    rgb48_to_rgba_row(&src, &mut out, 1);
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
    rgb48_to_rgba_u16_row(&src, &mut out, 1);
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
    bgr48_to_rgb_u16_row(&src, &mut out, 3);
    assert!(out.iter().all(|&v| v == 0xFFFF), "expected all 0xFFFF");
  }

  /// All-white input narrowed to u8 should produce all-0xFF.
  #[test]
  fn bgr48_to_rgb_all_white_narrow() {
    let src = std::vec![0xFFFFu16; 3 * 3];
    let mut out = std::vec![0u8; 3 * 3];
    bgr48_to_rgb_row(&src, &mut out, 3);
    assert!(out.iter().all(|&v| v == 0xFF), "expected all 0xFF");
  }

  /// Channel-order swap: Bgr48 `[B=0x1234, G=0x5678, R=0x9ABC]`
  /// → `with_rgb_u16` → `[R=0x9ABC, G=0x5678, B=0x1234]`.
  #[test]
  fn bgr48_to_rgb_u16_channel_order_swapped() {
    // Source pixel in BGR order: B=0x1234, G=0x5678, R=0x9ABC
    let src = [0x1234u16, 0x5678, 0x9ABC];
    let mut out = [0u16; 3];
    bgr48_to_rgb_u16_row(&src, &mut out, 1);
    assert_eq!(out[0], 0x9ABC, "R (was at src[2])");
    assert_eq!(out[1], 0x5678, "G (unchanged)");
    assert_eq!(out[2], 0x1234, "B (was at src[0])");
  }

  /// u8 RGB output: same swap + narrow.
  #[test]
  fn bgr48_to_rgb_channel_order_and_narrow() {
    let src = [0x1200u16, 0x5600, 0x9A00];
    let mut out = [0u8; 3];
    bgr48_to_rgb_row(&src, &mut out, 1);
    assert_eq!(out[0], 0x9A, "R");
    assert_eq!(out[1], 0x56, "G");
    assert_eq!(out[2], 0x12, "B");
  }

  /// rgba output: swapped channels + forced alpha = 0xFF.
  #[test]
  fn bgr48_to_rgba_channel_order_and_alpha() {
    let src = [0x1100u16, 0x2200, 0x3300];
    let mut out = [0u8; 4];
    bgr48_to_rgba_row(&src, &mut out, 1);
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
    bgr48_to_rgba_u16_row(&src, &mut out, 1);
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
    rgba64_to_rgba_u16_row(&src, &mut out, 3);
    assert!(out.iter().all(|&v| v == 0xFFFF), "expected all 0xFFFF");
  }

  /// All-white narrowed to u8 produces all-0xFF.
  #[test]
  fn rgba64_to_rgba_all_white_narrow() {
    let src = std::vec![0xFFFFu16; 4 * 3];
    let mut out = std::vec![0u8; 4 * 3];
    rgba64_to_rgba_row(&src, &mut out, 3);
    assert!(out.iter().all(|&v| v == 0xFF), "expected all 0xFF");
  }

  /// Source alpha is preserved in u16 passthrough at position 3.
  #[test]
  fn rgba64_to_rgba_u16_source_alpha_preserved() {
    // R=0x1111, G=0x2222, B=0x3333, A=0xABCD
    let src = [0x1111u16, 0x2222, 0x3333, 0xABCD];
    let mut out = [0u16; 4];
    rgba64_to_rgba_u16_row(&src, &mut out, 1);
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
    rgba64_to_rgba_row(&src, &mut out, 1);
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
    rgba64_to_rgb_row(&src, &mut out, 1);
    assert_eq!(out[0], 0x11, "R");
    assert_eq!(out[1], 0x22, "G");
    assert_eq!(out[2], 0x33, "B");
  }

  /// rgb_u16 path drops alpha, copies native u16.
  #[test]
  fn rgba64_to_rgb_u16_drops_alpha() {
    let src = [0x1111u16, 0x2222, 0x3333, 0xDEAD];
    let mut out = [0u16; 3];
    rgba64_to_rgb_u16_row(&src, &mut out, 1);
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
    bgra64_to_rgba_u16_row(&src, &mut out, 2);
    assert!(out.iter().all(|&v| v == 0xFFFF), "expected all 0xFFFF");
  }

  /// All-white narrowed to u8 produces all-0xFF.
  #[test]
  fn bgra64_to_rgba_all_white_narrow() {
    let src = std::vec![0xFFFFu16; 4 * 2];
    let mut out = std::vec![0u8; 4 * 2];
    bgra64_to_rgba_row(&src, &mut out, 2);
    assert!(out.iter().all(|&v| v == 0xFF), "expected all 0xFF");
  }

  /// Channel order swap + alpha preserved: Bgra64 `[B, G, R, A]` → `[R, G, B, A]`.
  #[test]
  fn bgra64_to_rgba_u16_channel_order_and_alpha_preserved() {
    // Source in BGRA order: B=0x1111, G=0x2222, R=0x3333, A=0x4444
    let src = [0x1111u16, 0x2222, 0x3333, 0x4444];
    let mut out = [0u16; 4];
    bgra64_to_rgba_u16_row(&src, &mut out, 1);
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
    bgra64_to_rgba_row(&src, &mut out, 1);
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
    bgra64_to_rgb_row(&src, &mut out, 1);
    assert_eq!(out[0], 0x33, "R");
    assert_eq!(out[1], 0x22, "G");
    assert_eq!(out[2], 0x11, "B");
  }

  /// rgb_u16 path drops alpha, swaps, native copy.
  #[test]
  fn bgra64_to_rgb_u16_drops_alpha_and_swaps() {
    let src = [0x1111u16, 0x2222, 0x3333, 0xDEAD];
    let mut out = [0u16; 3];
    bgra64_to_rgb_u16_row(&src, &mut out, 1);
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
    rgb48_to_rgb_row(&src, &mut out, 3);
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
    rgba64_to_rgba_u16_row(&src, &mut out, 2);
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
    rgb48_to_rgb_row(&rgb48_src, &mut rgb48_out, 1);
    bgr48_to_rgb_row(&bgr48_src, &mut bgr48_out, 1);

    assert_eq!(
      rgb48_out, bgr48_out,
      "RGB48 and BGR48 mirrored inputs must produce same RGB output"
    );
  }
}
