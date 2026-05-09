//! Tests for `crate::row::scalar::gray`.

use super::*;

// ---- Host-independent BE-fixture helpers ----------------------------------
//
// Build a `Vec<u16>` whose in-memory byte layout is the LE (resp. BE) byte
// encoding of `intended`. On a host with matching endianness the result
// equals `intended` element-wise; on the opposite-endian host every element
// is byte-swapped relative to `intended`. Together these let the
// `gray*_be_parity_*` tests below feed the LE kernel (`<false>`) and the BE
// kernel (`<true>`) byte-encoded inputs that decode to the same logical
// value on every host — replacing the earlier `swap_bytes()`-on-host-native
// pattern, which silently passed on BE hosts because both kernels decoded
// wrong but matched. See the `le_encoded_u16_buf` helpers in
// `src/frame/tests/*_high_bit.rs` for the same idea.

fn as_le_u16(intended: &[u16]) -> std::vec::Vec<u16> {
  let bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect()
}

fn as_be_u16(intended: &[u16]) -> std::vec::Vec<u16> {
  let bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
  bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect()
}

#[test]
fn gray8_to_rgb_broadcasts() {
  let y = [0u8, 128, 255];
  let mut out = [0u8; 9];
  gray8_to_rgb_row(&y, &mut out, 3, true);
  assert_eq!(&out[0..3], &[0, 0, 0]);
  assert_eq!(&out[3..6], &[128, 128, 128]);
  assert_eq!(&out[6..9], &[255, 255, 255]);
}

#[test]
fn gray8_to_rgba_broadcasts_opaque() {
  let y = [100u8, 200];
  let mut out = [0u8; 8];
  gray8_to_rgba_row(&y, &mut out, 2, true);
  assert_eq!(&out[0..4], &[100, 100, 100, 0xFF]);
  assert_eq!(&out[4..8], &[200, 200, 200, 0xFF]);
}

#[test]
fn gray8_to_hsv_h0_s0_v_y() {
  let y = [0u8, 128, 255];
  let mut h = [0xFFu8; 3];
  let mut s = [0xFFu8; 3];
  let mut v = [0u8; 3];
  gray8_to_hsv_row(&y, &mut h, &mut s, &mut v, 3, true);
  assert_eq!(h, [0, 0, 0]);
  assert_eq!(s, [0, 0, 0]);
  assert_eq!(v, [0, 128, 255]);
}

// ---- LE-host gating rationale (BE-tier11 follow-up) -----------------------
//
// The Gray9/10/12/14/16 limited-range / full-range / mask / luma / HSV /
// identity / opaque tests below construct fixtures as host-native
// `Vec<u16>` literals (e.g. `std::vec![512u16]`) and call the kernels with
// `::<false>`, which means "input is LE-encoded — decode to host-native by
// applying `u16::from_le`".
//
// On a little-endian host, host-native u16 bits and LE-encoded bits are the
// same byte sequence, so `u16::from_le` is a no-op and the assertions hold.
//
// On a big-endian host (powerpc64 / s390x / aarch64-be / mips), host-native
// u16 bits do NOT lay out little-endian, so the kernel's `from_le`
// byte-swap correctly reinterprets the host-native fixture as if it were
// an LE-encoded payload — producing a different (corrupted) value than the
// test expects.  The kernel itself is correct; this is purely a
// fixture-vs-kernel byte-order mismatch on BE hosts (same class as the
// PR #82 alpha_extract / planar_gbr_high_bit gates in `8f2e329` and the
// PR #83 Rgbf16 gates in `56342c0`).
//
// Kernel BE-host correctness is locked down separately by the dedicated
// `gray*_be_parity_*` tests further down in this module, which build the
// BE-encoded input via `swap_bytes()` and assert that BE+`<true>` matches
// LE+`<false>`. Those tests are intentionally NOT gated.
//
// Byte-symmetric value tests (`0x0000`, `0xFFFF`, `u16::MAX`) are also NOT
// gated — their bytes lay out the same in either order, so the kernel's
// `from_le` swap is a true no-op on every host.

// ---- limited-range tests: Gray8 ----

#[test]
fn gray8_limited_range_black() {
  // Y=16 → full-range black → RGB(0,0,0)
  let y = [16u8];
  let mut out = [0u8; 3];
  gray8_to_rgb_row(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[0, 0, 0]);
}

#[test]
fn gray8_limited_range_white() {
  // Y=235 → full-range white → RGB(255,255,255)
  let y = [235u8];
  let mut out = [0u8; 3];
  gray8_to_rgb_row(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[255, 255, 255]);
}

#[test]
fn gray8_limited_range_midpoint() {
  // Y=125 (8-bit ref mid of 16..235 span) → approx 127
  let y = [125u8]; // (125-16)*255/219 ≈ 127
  let mut out = [0u8; 3];
  gray8_to_rgb_row(&y, &mut out, 1, false);
  // Allow ±1 for rounding
  assert!(
    out[0] >= 126 && out[0] <= 128,
    "expected ~127 got {}",
    out[0]
  );
  assert_eq!(out[0], out[1]);
  assert_eq!(out[0], out[2]);
}

#[test]
fn gray8_limited_range_rgba() {
  let y = [16u8, 235];
  let mut out = [0u8; 8];
  gray8_to_rgba_row(&y, &mut out, 2, false);
  assert_eq!(&out[0..4], &[0, 0, 0, 0xFF]);
  assert_eq!(&out[4..8], &[255, 255, 255, 0xFF]);
}

#[test]
fn gray8_limited_range_hsv() {
  let y = [16u8, 235];
  let mut h = [0xFFu8; 2];
  let mut s = [0xFFu8; 2];
  let mut v = [0u8; 2];
  gray8_to_hsv_row(&y, &mut h, &mut s, &mut v, 2, false);
  assert_eq!(h, [0, 0]);
  assert_eq!(s, [0, 0]);
  assert_eq!(v, [0, 255]);
}

// ---- Gray10 limited-range tests ----

#[test]
fn gray10_limited_range_black() {
  // 10-bit black = 16 << 2 = 64
  let y = as_le_u16(&[64u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<10, false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[0, 0, 0]);
}

#[test]
fn gray10_limited_range_white() {
  // 10-bit white = 235 << 2 = 940
  let y = as_le_u16(&[940u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<10, false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[255, 255, 255]);
}

#[test]
fn gray10_limited_range_midpoint() {
  // 10-bit mid: 125 << 2 = 500 → approx 127
  let y = as_le_u16(&[500u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<10, false>(&y, &mut out, 1, false);
  assert!(
    out[0] >= 126 && out[0] <= 128,
    "expected ~127 got {}",
    out[0]
  );
}

#[test]
fn gray10_full_range_pass_through() {
  // 10-bit full range: value 512 >> 2 = 128
  let y = as_le_u16(&[512u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<10, false>(&y, &mut out, 1, true);
  assert_eq!(&out[0..3], &[128, 128, 128]);
}

// ---- Gray12 limited-range tests ----

#[test]
fn gray12_limited_range_black() {
  // 12-bit black = 16 << 4 = 256
  let y = as_le_u16(&[256u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<12, false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[0, 0, 0]);
}

#[test]
fn gray12_limited_range_white() {
  // 12-bit white = 235 << 4 = 3760
  let y = as_le_u16(&[3760u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<12, false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[255, 255, 255]);
}

// ---- Gray14 limited-range tests ----

#[test]
fn gray14_limited_range_black() {
  // 14-bit black = 16 << 6 = 1024
  let y = as_le_u16(&[1024u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<14, false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[0, 0, 0]);
}

#[test]
fn gray14_limited_range_white() {
  // 14-bit white = 235 << 6 = 15040
  let y = as_le_u16(&[15040u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<14, false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[255, 255, 255]);
}

// ---- Gray16 limited-range tests ----

#[test]
fn gray16_limited_range_black() {
  // 16-bit black = 16 << 8 = 4096
  let y = as_le_u16(&[4096u16]);
  let mut out = std::vec![0u8; 3];
  gray16_to_rgb_row::<false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[0, 0, 0]);
}

#[test]
fn gray16_limited_range_white() {
  // 16-bit white = 235 << 8 = 60160
  let y = as_le_u16(&[60160u16]);
  let mut out = std::vec![0u8; 3];
  gray16_to_rgb_row::<false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[255, 255, 255]);
}

#[test]
fn gray16_limited_range_midpoint() {
  // 16-bit mid: 125 << 8 = 32000 → approx 127
  let y = as_le_u16(&[32000u16]);
  let mut out = std::vec![0u8; 3];
  gray16_to_rgb_row::<false>(&y, &mut out, 1, false);
  assert!(
    out[0] >= 126 && out[0] <= 128,
    "expected ~127 got {}",
    out[0]
  );
}

#[test]
fn gray16_full_range_pass_through() {
  // 16-bit full range: 0x8000 >> 8 = 128
  let y = as_le_u16(&[0x8000u16]);
  let mut out = std::vec![0u8; 3];
  gray16_to_rgb_row::<false>(&y, &mut out, 1, true);
  assert_eq!(&out[0..3], &[128, 128, 128]);
}

// ---- Gray16 u16-output limited-range tests (i32 overflow regression) ----
//
// The native-u16 limited-range rescale `(y - black) * max_native` overflows
// i32 at BITS=16: `(60160 - 4096) * 65535 ≈ 3.67e9` > `i32::MAX`. Math runs
// in i64 to keep the product safe. These tests exercise the boundary
// values (black, white, over-white) end-to-end.

#[test]
fn gray16_to_rgb_u16_limited_range_black() {
  let y = as_le_u16(&[4096u16]); // limited-range black
  let mut out = std::vec![0u16; 3];
  gray16_to_rgb_u16_row::<false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[0, 0, 0]);
}

#[test]
fn gray16_to_rgb_u16_limited_range_white() {
  let y = as_le_u16(&[60160u16]); // limited-range white
  let mut out = std::vec![0u16; 3];
  gray16_to_rgb_u16_row::<false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[65535, 65535, 65535]);
}

#[test]
fn gray16_to_rgb_u16_limited_range_over_white_clamps() {
  // Over-white (Y > 60160) is clamped to max_native=65535.
  let y = as_le_u16(&[65535u16]);
  let mut out = std::vec![0u16; 3];
  gray16_to_rgb_u16_row::<false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[65535, 65535, 65535]);
}

#[test]
fn gray16_to_rgba_u16_limited_range_black_and_white() {
  let y = as_le_u16(&[4096u16, 60160u16]);
  let mut out = std::vec![0u16; 8];
  gray16_to_rgba_u16_row::<false>(&y, &mut out, 2, false);
  assert_eq!(&out[0..3], &[0, 0, 0]);
  assert_eq!(out[3], 0xFFFF);
  assert_eq!(&out[4..7], &[65535, 65535, 65535]);
  assert_eq!(out[7], 0xFFFF);
}

// ---- Original tests (now with full_range=true) ----

#[test]
fn gray_n_to_rgb_10bit_downshifts() {
  // 10-bit: 1023 >> 2 = 255; 0 >> 2 = 0; 512 >> 2 = 128
  let y = as_le_u16(&[0, 512, 1023]);
  let mut out = std::vec![0u8; 9];
  gray_n_to_rgb_row::<10, false>(&y, &mut out, 3, true);
  assert_eq!(&out[0..3], &[0, 0, 0]);
  assert_eq!(&out[3..6], &[128, 128, 128]);
  assert_eq!(&out[6..9], &[255, 255, 255]);
}

#[test]
fn gray_n_to_rgb_u16_10bit_masks() {
  // Upper bits should be masked out: 0xFFFF & 0x03FF = 0x03FF = 1023
  let y = as_le_u16(&[0xFFFF, 512, 0]);
  let mut out = std::vec![0u16; 9];
  gray_n_to_rgb_u16_row::<10, false>(&y, &mut out, 3, true);
  assert_eq!(&out[0..3], &[1023, 1023, 1023]);
  assert_eq!(&out[3..6], &[512, 512, 512]);
  assert_eq!(&out[6..9], &[0, 0, 0]);
}

#[test]
fn gray_n_to_hsv_h0_s0() {
  let y = as_le_u16(&[512u16]); // 512 >> 2 = 128
  let mut h = std::vec![0xFFu8; 1];
  let mut s = std::vec![0xFFu8; 1];
  let mut v = std::vec![0u8; 1];
  gray_n_to_hsv_row::<10, false>(&y, &mut h, &mut s, &mut v, 1, true);
  assert_eq!(h[0], 0);
  assert_eq!(s[0], 0);
  assert_eq!(v[0], 128);
}

#[test]
fn gray16_to_rgb_downshifts_8() {
  let y = as_le_u16(&[0, 0x8000, 0xFFFF]);
  let mut out = std::vec![0u8; 9];
  gray16_to_rgb_row::<false>(&y, &mut out, 3, true);
  assert_eq!(&out[0..3], &[0, 0, 0]);
  assert_eq!(&out[3..6], &[0x80, 0x80, 0x80]);
  assert_eq!(&out[6..9], &[0xFF, 0xFF, 0xFF]);
}

#[test]
fn gray16_to_luma_u16_identity() {
  let y = as_le_u16(&[0, 1000, 65535]);
  let mut out = std::vec![0u16; 3];
  gray16_to_luma_u16_row::<false>(&y, &mut out, 3);
  assert_eq!(out.as_slice(), &[0, 1000, 65535]);
}

#[test]
fn gray16_to_rgba_u16_opaque() {
  let y = as_le_u16(&[12345u16]);
  let mut out = std::vec![0u16; 4];
  gray16_to_rgba_u16_row::<false>(&y, &mut out, 1, true);
  assert_eq!(&out[0..4], &[12345, 12345, 12345, 0xFFFF]);
}

#[test]
fn gray_n_to_luma_u16_10bit_masks() {
  let y: std::vec::Vec<u16> = std::vec![0xFFFF]; // should mask to 1023
  let mut out = std::vec![0u16; 1];
  gray_n_to_luma_u16_row::<10, false>(&y, &mut out, 1);
  assert_eq!(out[0], 1023);
}

// ---- Gray9 limited-range tests ----

#[test]
fn gray9_limited_range_black() {
  // 9-bit black = 16 << 1 = 32
  let y = as_le_u16(&[32u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<9, false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[0, 0, 0]);
}

#[test]
fn gray9_limited_range_white() {
  // 9-bit white = 235 << 1 = 470
  let y = as_le_u16(&[470u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<9, false>(&y, &mut out, 1, false);
  assert_eq!(&out[0..3], &[255, 255, 255]);
}

#[test]
fn gray9_full_range_pass_through() {
  // 9-bit full range: value 256 >> 1 = 128
  let y = as_le_u16(&[256u16]);
  let mut out = std::vec![0u8; 3];
  gray_n_to_rgb_row::<9, false>(&y, &mut out, 1, true);
  assert_eq!(&out[0..3], &[128, 128, 128]);
}

// ---- BE parity tests: gray_n (Gray9-14) -----------------------------------
// Pattern: build LE-encoded and BE-encoded byte storage from the same
// logical `intended` value via `as_le_u16` / `as_be_u16` (host-independent),
// then assert the LE kernel on LE-encoded input and the BE kernel on
// BE-encoded input produce the same output. The earlier
// `swap_bytes()`-on-host-native pattern was vacuous on BE hosts: both
// kernels decoded incorrectly but to matching values.

#[test]
fn gray10_be_parity_rgb() {
  // Logical 10-bit value 512 → 512 >> 2 = 128 in the high 8 bits.
  let intended: std::vec::Vec<u16> = std::vec![512u16];
  let le_in = as_le_u16(&intended);
  let be_in = as_be_u16(&intended);
  let mut out_le = std::vec![0u8; 3];
  let mut out_be = std::vec![0u8; 3];
  gray_n_to_rgb_row::<10, false>(&le_in, &mut out_le, 1, true);
  gray_n_to_rgb_row::<10, true>(&be_in, &mut out_be, 1, true);
  assert_eq!(out_le, out_be, "BE and LE gray10 rgb outputs must match");
  assert_eq!(&out_le[..], &[128, 128, 128]);
}

#[test]
fn gray10_be_parity_rgba() {
  // Logical 10-bit value 768 → 768 >> 2 = 192.
  let intended: std::vec::Vec<u16> = std::vec![768u16];
  let le_in = as_le_u16(&intended);
  let be_in = as_be_u16(&intended);
  let mut out_le = std::vec![0u8; 4];
  let mut out_be = std::vec![0u8; 4];
  gray_n_to_rgba_row::<10, false>(&le_in, &mut out_le, 1, true);
  gray_n_to_rgba_row::<10, true>(&be_in, &mut out_be, 1, true);
  assert_eq!(out_le, out_be, "BE and LE gray10 rgba outputs must match");
  assert_eq!(&out_le[..], &[192, 192, 192, 0xFF]);
}

#[test]
fn gray10_be_parity_luma() {
  // Logical 10-bit value 256 → 256 >> 2 = 64.
  let intended: std::vec::Vec<u16> = std::vec![256u16];
  let le_in = as_le_u16(&intended);
  let be_in = as_be_u16(&intended);
  let mut out_le = std::vec![0u8; 1];
  let mut out_be = std::vec![0u8; 1];
  gray_n_to_luma_row::<10, false>(&le_in, &mut out_le, 1);
  gray_n_to_luma_row::<10, true>(&be_in, &mut out_be, 1);
  assert_eq!(out_le, out_be, "BE and LE gray10 luma outputs must match");
  assert_eq!(out_le[0], 64);
}

#[test]
fn gray10_be_parity_luma_u16() {
  let intended: std::vec::Vec<u16> = std::vec![512u16];
  let le_in = as_le_u16(&intended);
  let be_in = as_be_u16(&intended);
  let mut out_le = std::vec![0u16; 1];
  let mut out_be = std::vec![0u16; 1];
  gray_n_to_luma_u16_row::<10, false>(&le_in, &mut out_le, 1);
  gray_n_to_luma_u16_row::<10, true>(&be_in, &mut out_be, 1);
  assert_eq!(
    out_le, out_be,
    "BE and LE gray10 luma_u16 outputs must match"
  );
  assert_eq!(out_le[0], 512);
}

// ---- BE parity tests: gray16 -----------------------------------------------

#[test]
fn gray16_be_parity_rgb() {
  // Logical 16-bit value 0x8000 → high byte 0x80 = 128.
  let intended: std::vec::Vec<u16> = std::vec![0x8000u16];
  let le_in = as_le_u16(&intended);
  let be_in = as_be_u16(&intended);
  let mut out_le = std::vec![0u8; 3];
  let mut out_be = std::vec![0u8; 3];
  gray16_to_rgb_row::<false>(&le_in, &mut out_le, 1, true);
  gray16_to_rgb_row::<true>(&be_in, &mut out_be, 1, true);
  assert_eq!(out_le, out_be, "BE and LE gray16 rgb outputs must match");
  assert_eq!(&out_le[..], &[128, 128, 128]);
}

#[test]
fn gray16_be_parity_rgba() {
  // Logical 16-bit value 0xC000 → high byte 0xC0 = 192.
  let intended: std::vec::Vec<u16> = std::vec![0xC000u16];
  let le_in = as_le_u16(&intended);
  let be_in = as_be_u16(&intended);
  let mut out_le = std::vec![0u8; 4];
  let mut out_be = std::vec![0u8; 4];
  gray16_to_rgba_row::<false>(&le_in, &mut out_le, 1, true);
  gray16_to_rgba_row::<true>(&be_in, &mut out_be, 1, true);
  assert_eq!(out_le, out_be, "BE and LE gray16 rgba outputs must match");
  assert_eq!(&out_le[..], &[192, 192, 192, 0xFF]);
}

#[test]
fn gray16_be_parity_luma() {
  // Logical 16-bit value 0x4000 → high byte 0x40 = 64.
  let intended: std::vec::Vec<u16> = std::vec![0x4000u16];
  let le_in = as_le_u16(&intended);
  let be_in = as_be_u16(&intended);
  let mut out_le = std::vec![0u8; 1];
  let mut out_be = std::vec![0u8; 1];
  gray16_to_luma_row::<false>(&le_in, &mut out_le, 1);
  gray16_to_luma_row::<true>(&be_in, &mut out_be, 1);
  assert_eq!(out_le, out_be, "BE and LE gray16 luma outputs must match");
  assert_eq!(out_le[0], 64);
}

#[test]
fn gray16_be_parity_luma_u16() {
  // Logical 16-bit value 0x1234 — both kernels must recover it.
  let intended: std::vec::Vec<u16> = std::vec![0x1234u16];
  let le_in = as_le_u16(&intended);
  let be_in = as_be_u16(&intended);
  let mut out_le = std::vec![0u16; 1];
  let mut out_be = std::vec![0u16; 1];
  gray16_to_luma_u16_row::<false>(&le_in, &mut out_le, 1);
  gray16_to_luma_u16_row::<true>(&be_in, &mut out_be, 1);
  assert_eq!(
    out_le, out_be,
    "BE and LE gray16 luma_u16 outputs must match"
  );
  assert_eq!(out_le[0], 0x1234);
}
