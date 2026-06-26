//! Tests for `crate::row::scalar::packed_rgb_32bit`.

use super::*;

/// Re-encode a host-native u32 slice as LE-encoded byte storage. Kernels
/// called with `BE = false` recover the intended logical values via
/// `u32::from_le` on both LE (no-op) and BE (byte-swap) hosts.
fn as_le_u32(host: &[u32]) -> std::vec::Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Mirror of [`as_le_u32`] for kernels invoked with `BE = true`.
fn as_be_u32(host: &[u32]) -> std::vec::Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

// ---- Rgb96 ---------------------------------------------------------------

/// All-white input narrowed to u16 should produce all-0xFFFF.
#[test]
fn rgb96_to_rgb_u16_all_white_narrow() {
  let src = std::vec![0xFFFF_FFFFu32; 3 * 4];
  let mut out = std::vec![0u16; 3 * 4];
  rgb96_to_rgb_u16_row::<false>(&src, &mut out, 4);
  assert!(
    out.iter().all(|&v| v == 0xFFFF),
    "expected all 0xFFFF, got {out:?}"
  );
}

/// All-white input narrowed to u8 should produce all-0xFF.
#[test]
fn rgb96_to_rgb_all_white_narrow() {
  let src = std::vec![0xFFFF_FFFFu32; 3 * 4];
  let mut out = std::vec![0u8; 3 * 4];
  rgb96_to_rgb_row::<false>(&src, &mut out, 4);
  assert!(
    out.iter().all(|&v| v == 0xFF),
    "expected all 0xFF, got {out:?}"
  );
}

/// Known value: u8 = `v >> 24`.
#[test]
fn rgb96_to_rgb_narrow_known_value() {
  let src = as_le_u32(&[0x1234_5678u32, 0x5678_9ABC, 0x9ABC_DEF0]);
  let mut out = [0u8; 3];
  rgb96_to_rgb_row::<false>(&src, &mut out, 1);
  assert_eq!(out[0], 0x12, "R channel");
  assert_eq!(out[1], 0x56, "G channel");
  assert_eq!(out[2], 0x9A, "B channel");
}

/// Known value: u16 = `v >> 16`.
#[test]
fn rgb96_to_rgb_u16_narrow_known_value() {
  let src = as_le_u32(&[0x1234_5678u32, 0x5678_9ABC, 0x9ABC_DEF0]);
  let mut out = [0u16; 3];
  rgb96_to_rgb_u16_row::<false>(&src, &mut out, 1);
  assert_eq!(out[0], 0x1234, "R channel");
  assert_eq!(out[1], 0x5678, "G channel");
  assert_eq!(out[2], 0x9ABC, "B channel");
}

/// rgba output forces alpha = 0xFF and narrows `>> 24`.
#[test]
fn rgb96_to_rgba_forces_alpha_0xff() {
  let src = as_le_u32(&[0xAAAA_0000u32, 0xBBBB_1111, 0xCCCC_2222]);
  let mut out = [0u8; 4];
  rgb96_to_rgba_row::<false>(&src, &mut out, 1);
  assert_eq!(out[0], 0xAA, "R");
  assert_eq!(out[1], 0xBB, "G");
  assert_eq!(out[2], 0xCC, "B");
  assert_eq!(out[3], 0xFF, "alpha must be 0xFF");
}

/// rgba_u16 output forces alpha = 0xFFFF and narrows `>> 16`.
#[test]
fn rgb96_to_rgba_u16_forces_alpha_0xffff() {
  let src = as_le_u32(&[0xAAAA_0000u32, 0xBBBB_1111, 0xCCCC_2222]);
  let mut out = [0u16; 4];
  rgb96_to_rgba_u16_row::<false>(&src, &mut out, 1);
  assert_eq!(out[0], 0xAAAA, "R");
  assert_eq!(out[1], 0xBBBB, "G");
  assert_eq!(out[2], 0xCCCC, "B");
  assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
}

/// BE parity: build LE / BE source buffers from one host-native `intended`
/// fixture and assert both decode the same logical values, pinned against an
/// absolute scalar reference (so the parity check cannot pass on two equally
/// corrupt byte-reversed decodes on a BE host).
#[test]
fn rgb96_to_rgb_be_parity_with_swapped_buffer() {
  let intended: std::vec::Vec<u32> = std::vec![
    0x1234_5678,
    0x5678_9ABC,
    0x9ABC_DEF0,
    0x0011_2233,
    0x4455_6677,
    0x8899_AABB
  ];
  let src_le = as_le_u32(&intended);
  let src_be = as_be_u32(&intended);
  let mut out_le = std::vec![0u8; 6];
  let mut out_be = std::vec![0u8; 6];
  rgb96_to_rgb_row::<false>(&src_le, &mut out_le, 2);
  rgb96_to_rgb_row::<true>(&src_be, &mut out_be, 2);
  let expected: std::vec::Vec<u8> = intended.iter().map(|&v| (v >> 24) as u8).collect();
  assert_eq!(out_le, expected, "LE path must match scalar reference");
  assert_eq!(out_be, expected, "BE path must match scalar reference");
  assert_eq!(out_le, out_be, "BE and LE outputs must agree");
}

/// BE parity for the u16 RGB path.
#[test]
fn rgb96_to_rgb_u16_be_parity_with_swapped_buffer() {
  let intended: std::vec::Vec<u32> = std::vec![
    0x1234_5678,
    0x5678_9ABC,
    0x9ABC_DEF0,
    0x0011_2233,
    0x4455_6677,
    0x8899_AABB
  ];
  let src_le = as_le_u32(&intended);
  let src_be = as_be_u32(&intended);
  let mut out_le = std::vec![0u16; 6];
  let mut out_be = std::vec![0u16; 6];
  rgb96_to_rgb_u16_row::<false>(&src_le, &mut out_le, 2);
  rgb96_to_rgb_u16_row::<true>(&src_be, &mut out_be, 2);
  let expected: std::vec::Vec<u16> = intended.iter().map(|&v| (v >> 16) as u16).collect();
  assert_eq!(out_le, expected, "LE path must match scalar reference");
  assert_eq!(out_be, expected, "BE path must match scalar reference");
  assert_eq!(out_le, out_be, "BE and LE outputs must agree");
}

// ---- Rgba128 -------------------------------------------------------------

/// Rgba128 → u8 RGB drops alpha and narrows R/G/B `>> 24`.
#[test]
fn rgba128_to_rgb_drops_alpha() {
  let src = as_le_u32(&[0x1100_0000, 0x2200_0000, 0x3300_0000, 0x4400_0000]);
  let mut out = [0u8; 3];
  rgba128_to_rgb_row::<false>(&src, &mut out, 1);
  assert_eq!(out, [0x11, 0x22, 0x33], "alpha dropped, R/G/B narrowed");
}

/// Rgba128 → u8 RGBA passes alpha through, narrowed `>> 24`.
#[test]
fn rgba128_to_rgba_passes_alpha() {
  let src = as_le_u32(&[0x1100_0000, 0x2200_0000, 0x3300_0000, 0x4400_0000]);
  let mut out = [0u8; 4];
  rgba128_to_rgba_row::<false>(&src, &mut out, 1);
  assert_eq!(out, [0x11, 0x22, 0x33, 0x44], "source alpha passes through");
}

/// Rgba128 → u16 RGBA passes alpha through, narrowed `>> 16`.
#[test]
fn rgba128_to_rgba_u16_passes_alpha() {
  let src = as_le_u32(&[0x1122_0000, 0x3344_0000, 0x5566_0000, 0x7788_0000]);
  let mut out = [0u16; 4];
  rgba128_to_rgba_u16_row::<false>(&src, &mut out, 1);
  assert_eq!(out, [0x1122, 0x3344, 0x5566, 0x7788]);
}

/// Rgba128 → u16 RGB drops alpha, narrows R/G/B `>> 16`.
#[test]
fn rgba128_to_rgb_u16_drops_alpha() {
  let src = as_le_u32(&[0x1122_0000, 0x3344_0000, 0x5566_0000, 0x7788_0000]);
  let mut out = [0u16; 3];
  rgba128_to_rgb_u16_row::<false>(&src, &mut out, 1);
  assert_eq!(out, [0x1122, 0x3344, 0x5566]);
}

/// BE parity for Rgba128 → u8 RGBA (alpha pass-through).
#[test]
fn rgba128_to_rgba_be_parity_with_swapped_buffer() {
  let intended: std::vec::Vec<u32> = std::vec![0x1234_5678, 0x5678_9ABC, 0x9ABC_DEF0, 0x0011_2233];
  let src_le = as_le_u32(&intended);
  let src_be = as_be_u32(&intended);
  let mut out_le = std::vec![0u8; 4];
  let mut out_be = std::vec![0u8; 4];
  rgba128_to_rgba_row::<false>(&src_le, &mut out_le, 1);
  rgba128_to_rgba_row::<true>(&src_be, &mut out_be, 1);
  let expected: std::vec::Vec<u8> = intended.iter().map(|&v| (v >> 24) as u8).collect();
  assert_eq!(out_le, expected, "LE path must match scalar reference");
  assert_eq!(out_be, expected, "BE path must match scalar reference");
  assert_eq!(out_le, out_be, "BE and LE outputs must agree");
}
