//! Tests for `crate::row::scalar::packed_rgb_16bit`.

use super::*;

/// Re-encode a host-native u16 slice as LE-encoded byte storage. Kernels
/// called with `BE = false` recover the intended logical values via
/// `u16::from_le` on both LE (no-op) and BE (byte-swap) hosts.
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

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
  let src = as_le_u16(&[0x1234u16, 0x5678, 0x9ABC]);
  let mut out = [0u8; 3];
  rgb48_to_rgb_row::<false>(&src, &mut out, 1);
  assert_eq!(out[0], 0x12, "R channel");
  assert_eq!(out[1], 0x56, "G channel");
  assert_eq!(out[2], 0x9A, "B channel");
}

/// rgba output forces alpha = 0xFF.
#[test]
fn rgb48_to_rgba_forces_alpha_0xff() {
  let src = as_le_u16(&[0xAAAAu16, 0xBBBB, 0xCCCC]);
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
  let src = as_le_u16(&[0xAAAAu16, 0xBBBB, 0xCCCC]);
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
  let src = as_le_u16(&[0x1234u16, 0x5678, 0x9ABC]);
  let mut out = [0u16; 3];
  bgr48_to_rgb_u16_row::<false>(&src, &mut out, 1);
  assert_eq!(out[0], 0x9ABC, "R (was at src[2])");
  assert_eq!(out[1], 0x5678, "G (unchanged)");
  assert_eq!(out[2], 0x1234, "B (was at src[0])");
}

/// u8 RGB output: same swap + narrow.
#[test]
fn bgr48_to_rgb_channel_order_and_narrow() {
  let src = as_le_u16(&[0x1200u16, 0x5600, 0x9A00]);
  let mut out = [0u8; 3];
  bgr48_to_rgb_row::<false>(&src, &mut out, 1);
  assert_eq!(out[0], 0x9A, "R");
  assert_eq!(out[1], 0x56, "G");
  assert_eq!(out[2], 0x12, "B");
}

/// rgba output: swapped channels + forced alpha = 0xFF.
#[test]
fn bgr48_to_rgba_channel_order_and_alpha() {
  let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300]);
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
  let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333]);
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
  let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333, 0xABCD]);
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
  let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300, 0xABCD]);
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
  let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300, 0xDEAD]);
  let mut out = [0u8; 3];
  rgba64_to_rgb_row::<false>(&src, &mut out, 1);
  assert_eq!(out[0], 0x11, "R");
  assert_eq!(out[1], 0x22, "G");
  assert_eq!(out[2], 0x33, "B");
}

/// rgb_u16 path drops alpha, copies native u16.
#[test]
fn rgba64_to_rgb_u16_drops_alpha() {
  let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333, 0xDEAD]);
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
  let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333, 0x4444]);
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
  let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300, 0xAB00]);
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
  let src = as_le_u16(&[0x1100u16, 0x2200, 0x3300, 0xDEAD]);
  let mut out = [0u8; 3];
  bgra64_to_rgb_row::<false>(&src, &mut out, 1);
  assert_eq!(out[0], 0x33, "R");
  assert_eq!(out[1], 0x22, "G");
  assert_eq!(out[2], 0x11, "B");
}

/// rgb_u16 path drops alpha, swaps, native copy.
#[test]
fn bgra64_to_rgb_u16_drops_alpha_and_swaps() {
  let src = as_le_u16(&[0x1111u16, 0x2222, 0x3333, 0xDEAD]);
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
  let src = as_le_u16(&[
    0x1100u16, 0x2200, 0x3300, 0x4400, 0x5500, 0x6600, 0x7700, 0x8800, 0x9900,
  ]);
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

/// Width=2 Rgba64→rgba_u16: identity copy preserves layout. Output is
/// host-native u16 after the kernel's `from_le` decode; compare against
/// the original host-native intended values.
#[test]
fn rgba64_to_rgba_u16_multi_pixel_identity() {
  let intended: [u16; 8] = [
    0x1111u16, 0x2222, 0x3333, 0x4444, // pixel 0
    0x5555, 0x6666, 0x7777, 0x8888, // pixel 1
  ];
  let src = as_le_u16(&intended);
  let mut out = [0u16; 8];
  rgba64_to_rgba_u16_row::<false>(&src, &mut out, 2);
  assert_eq!(out, intended, "identity copy must be byte-exact");
}

/// Bgr48 and Rgb48 on mirrored input produce same rgb output.
///
/// Uses non-byte-palindromic values (low byte ≠ high byte) AND
/// LE-encodes both fixtures through `as_le_u16`. With palindromic
/// values like 0xAAAA / 0xCCCC the host-native and LE byte storage
/// happen to coincide, which would let a missing `as_le_u16` wrap
/// silently produce a passing parity check on BE — masking the very
/// byte-order bug the test should catch. Asserting the absolute
/// post-shift output bytes pins down the channel order against the
/// intended host-native samples instead of just self-comparing two
/// `<false>` outputs.
#[test]
fn bgr48_rgb_output_matches_rgb48_with_swapped_input() {
  // RGB input: R=0xAB12, G=0xCD34, B=0xEF56
  let rgb48_src = as_le_u16(&[0xAB12u16, 0xCD34, 0xEF56]);
  // BGR input: B=0xEF56, G=0xCD34, R=0xAB12
  let bgr48_src = as_le_u16(&[0xEF56u16, 0xCD34, 0xAB12]);

  let mut rgb48_out = [0u8; 3];
  let mut bgr48_out = [0u8; 3];
  rgb48_to_rgb_row::<false>(&rgb48_src, &mut rgb48_out, 1);
  bgr48_to_rgb_row::<false>(&bgr48_src, &mut bgr48_out, 1);

  assert_eq!(
    rgb48_out, bgr48_out,
    "RGB48 and BGR48 mirrored inputs must produce same RGB output"
  );
  // Independent expected-output assertion: rgb48_to_rgb_row extracts
  // the high byte of each u16 channel. R=0xAB12→0xAB, G=0xCD34→0xCD,
  // B=0xEF56→0xEF. A BE host that fails to LE-encode the fixtures
  // would decode swapped bytes (R=0x12AB) and emit 0x12 instead.
  assert_eq!(rgb48_out, [0xAB, 0xCD, 0xEF], "RGB48 high-byte extract");
}
