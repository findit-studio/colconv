//! Tests for `crate::row::scalar::ya16`.

use super::*;

// Helper: make packed [Y, A, Y, A, ...] from pairs, in LE-encoded byte form
// so kernels with `BE = false` recover the intended logical values via
// `u16::from_le` on both LE (no-op) and BE (byte-swap) hosts.
fn packed_ya(pairs: &[(u16, u16)]) -> std::vec::Vec<u16> {
  pairs
    .iter()
    .flat_map(|&(y, a)| [y, a])
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as LE-encoded byte storage. Mirror of
/// the alpha_extract `as_le_u16` helper so a single host-native `intended`
/// fixture drives the `BE = false` path on every host.
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as BE-encoded byte storage. Combined
/// with `as_le_u16`, lets a single host-native `intended` fixture drive
/// both `<false>` and `<true>` kernel paths so they decode the same logical
/// values on every host.
fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

// -- Scalar references for the BE-parity tests --
//
// Walk host-native `intended` buffers (laid out as `[Y0, A0, Y1, A1, ...]`)
// and reproduce each kernel's documented behaviour without going through
// any byte-order conversion. Pinning the LE / BE outputs against these
// absolute references prevents the parity assertion from passing in
// lock-step on two equally corrupt decode paths.

/// Reference for `ya16_to_rgb_row`: Y `>> 8` broadcast to R = G = B; α dropped.
fn ref_ya16_to_rgb(intended: &[u16], width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width * 3];
  for x in 0..width {
    let y8 = (intended[x * 2] >> 8) as u8;
    out[x * 3] = y8;
    out[x * 3 + 1] = y8;
    out[x * 3 + 2] = y8;
  }
  out
}

/// Reference for `ya16_to_rgba_row`: Y `>> 8` broadcast, A `>> 8` from slot 1.
fn ref_ya16_to_rgba(intended: &[u16], width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width * 4];
  for x in 0..width {
    let y8 = (intended[x * 2] >> 8) as u8;
    let a8 = (intended[x * 2 + 1] >> 8) as u8;
    out[x * 4] = y8;
    out[x * 4 + 1] = y8;
    out[x * 4 + 2] = y8;
    out[x * 4 + 3] = a8;
  }
  out
}

/// Reference for `ya16_to_luma_row`: Y `>> 8` from slot 0 of every pair.
fn ref_ya16_to_luma(intended: &[u16], width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width];
  for x in 0..width {
    out[x] = (intended[x * 2] >> 8) as u8;
  }
  out
}

// ---- ya16_to_rgb_row -------------------------------------------------------

#[test]
fn ya16_to_rgb_downshifts_y_drops_alpha() {
  // Y=0x8000, A=0x4000 → rgb [0x80, 0x80, 0x80]
  let p = packed_ya(&[(0x8000, 0x4000)]);
  let mut out = [0u8; 3];
  ya16_to_rgb_row::<false>(&p, &mut out, 1);
  assert_eq!(out, [0x80, 0x80, 0x80]);
}

#[test]
fn ya16_to_rgb_zero_pixel() {
  let p = packed_ya(&[(0, 0)]);
  let mut out = [0xFFu8; 3];
  ya16_to_rgb_row::<false>(&p, &mut out, 1);
  assert_eq!(out, [0, 0, 0]);
}

#[test]
fn ya16_to_rgb_max_y() {
  let p = packed_ya(&[(0xFFFF, 0)]);
  let mut out = [0u8; 3];
  ya16_to_rgb_row::<false>(&p, &mut out, 1);
  assert_eq!(out, [0xFF, 0xFF, 0xFF]);
}

// ---- ya16_to_rgba_row -----------------------------------------------------

#[test]
fn ya16_to_rgba_downshifts_y_and_alpha() {
  // Y=0x8000, A=0x4000 → rgba [0x80, 0x80, 0x80, 0x40]
  let p = packed_ya(&[(0x8000, 0x4000)]);
  let mut out = [0u8; 4];
  ya16_to_rgba_row::<false>(&p, &mut out, 1);
  assert_eq!(out, [0x80, 0x80, 0x80, 0x40]);
}

#[test]
fn ya16_to_rgba_two_pixels() {
  let p = packed_ya(&[(0x8000, 0x4000), (0x1000, 0x0800)]);
  let mut out = [0u8; 8];
  ya16_to_rgba_row::<false>(&p, &mut out, 2);
  assert_eq!(&out[0..4], &[0x80, 0x80, 0x80, 0x40]);
  assert_eq!(&out[4..8], &[0x10, 0x10, 0x10, 0x08]);
}

// ---- ya16_to_rgb_u16_row --------------------------------------------------

#[test]
fn ya16_to_rgb_u16_native_y_broadcast() {
  // Y=0x8000 native, broadcast
  let p = packed_ya(&[(0x8000, 0x4000)]);
  let mut out = [0u16; 3];
  ya16_to_rgb_u16_row::<false>(&p, &mut out, 1);
  assert_eq!(out, [0x8000, 0x8000, 0x8000]);
}

#[test]
fn ya16_to_rgb_u16_zero() {
  let p = packed_ya(&[(0, 0)]);
  let mut out = [0xFFFFu16; 3];
  ya16_to_rgb_u16_row::<false>(&p, &mut out, 1);
  assert_eq!(out, [0, 0, 0]);
}

// ---- ya16_to_rgba_u16_row -------------------------------------------------

#[test]
fn ya16_to_rgba_u16_native_y_and_alpha() {
  // Y=0x8000, A=0x4000 → rgba_u16 [0x8000, 0x8000, 0x8000, 0x4000]
  let p = packed_ya(&[(0x8000, 0x4000)]);
  let mut out = [0u16; 4];
  ya16_to_rgba_u16_row::<false>(&p, &mut out, 1);
  assert_eq!(out, [0x8000, 0x8000, 0x8000, 0x4000]);
}

// ---- ya16_to_luma_row -----------------------------------------------------

#[test]
fn ya16_to_luma_downshifts() {
  let p = packed_ya(&[(0x8000, 0x4000), (0x0000, 0xFFFF)]);
  let mut out = [0u8; 2];
  ya16_to_luma_row::<false>(&p, &mut out, 2);
  assert_eq!(out, [0x80, 0x00]);
}

// ---- ya16_to_luma_u16_row -------------------------------------------------

#[test]
fn ya16_to_luma_u16_native_passthrough() {
  let p = packed_ya(&[(0x8000, 0x0000)]);
  let mut out = [0u16; 1];
  ya16_to_luma_u16_row::<false>(&p, &mut out, 1);
  assert_eq!(out[0], 0x8000);
}

// ---- ya16_to_hsv_row -------------------------------------------------------

#[test]
fn ya16_to_hsv_h0_s0_v_y8_drops_alpha() {
  // Y=0x8000 → V = 0x80; α dropped
  let p = packed_ya(&[(0x8000, 0x4000)]);
  let mut h = [0xFFu8; 1];
  let mut s = [0xFFu8; 1];
  let mut v = [0u8; 1];
  ya16_to_hsv_row::<false>(&p, &mut h, &mut s, &mut v, 1);
  assert_eq!(h[0], 0);
  assert_eq!(s[0], 0);
  assert_eq!(v[0], 0x80);
}

#[test]
fn ya16_to_hsv_zero_luma() {
  let p = packed_ya(&[(0, 0xFFFF)]);
  let mut h = [0u8; 1];
  let mut s = [0u8; 1];
  let mut v = [0xFFu8; 1];
  ya16_to_hsv_row::<false>(&p, &mut h, &mut s, &mut v, 1);
  assert_eq!(v[0], 0);
}

#[test]
fn ya16_to_hsv_max_luma() {
  let p = packed_ya(&[(0xFFFF, 0)]);
  let mut h = [0u8; 1];
  let mut s = [0u8; 1];
  let mut v = [0u8; 1];
  ya16_to_hsv_row::<false>(&p, &mut h, &mut s, &mut v, 1);
  assert_eq!(v[0], 0xFF);
}

// ---- BE parity tests: ya16 -------------------------------------------------
// Pattern: build a single host-native `intended` fixture, materialise it as
// LE-encoded bytes via `as_le_u16` and BE-encoded bytes via `as_be_u16`,
// run both `<false>` and `<true>` kernels, and pin each output against an
// absolute scalar reference so the parity assertion cannot pass on two
// equally corrupt decodes.

#[test]
fn ya16_be_parity_rgb() {
  // Y=0x8000, A=0x4000 → RGB [0x80, 0x80, 0x80]
  let intended: std::vec::Vec<u16> = std::vec![0x8000, 0x4000];
  let le = as_le_u16(&intended);
  let be = as_be_u16(&intended);
  let mut out_le = [0u8; 3];
  let mut out_be = [0u8; 3];
  ya16_to_rgb_row::<false>(&le, &mut out_le, 1);
  ya16_to_rgb_row::<true>(&be, &mut out_be, 1);
  let expected = ref_ya16_to_rgb(&intended, 1);
  assert_eq!(
    out_le.as_slice(),
    expected,
    "LE path must match scalar reference"
  );
  assert_eq!(
    out_be.as_slice(),
    expected,
    "BE path must match scalar reference"
  );
  assert_eq!(out_le, out_be, "BE and LE ya16 rgb outputs must agree");
}

#[test]
fn ya16_be_parity_rgba() {
  // Y=0x8000, A=0x4000 → RGBA [0x80, 0x80, 0x80, 0x40]
  let intended: std::vec::Vec<u16> = std::vec![0x8000, 0x4000];
  let le = as_le_u16(&intended);
  let be = as_be_u16(&intended);
  let mut out_le = [0u8; 4];
  let mut out_be = [0u8; 4];
  ya16_to_rgba_row::<false>(&le, &mut out_le, 1);
  ya16_to_rgba_row::<true>(&be, &mut out_be, 1);
  let expected = ref_ya16_to_rgba(&intended, 1);
  assert_eq!(
    out_le.as_slice(),
    expected,
    "LE path must match scalar reference"
  );
  assert_eq!(
    out_be.as_slice(),
    expected,
    "BE path must match scalar reference"
  );
  assert_eq!(out_le, out_be, "BE and LE ya16 rgba outputs must agree");
}

#[test]
fn ya16_be_parity_luma() {
  let intended: std::vec::Vec<u16> = std::vec![0xC000, 0x0000];
  let le = as_le_u16(&intended);
  let be = as_be_u16(&intended);
  let mut out_le = [0u8; 1];
  let mut out_be = [0u8; 1];
  ya16_to_luma_row::<false>(&le, &mut out_le, 1);
  ya16_to_luma_row::<true>(&be, &mut out_be, 1);
  let expected = ref_ya16_to_luma(&intended, 1);
  assert_eq!(
    out_le.as_slice(),
    expected,
    "LE path must match scalar reference"
  );
  assert_eq!(
    out_be.as_slice(),
    expected,
    "BE path must match scalar reference"
  );
  assert_eq!(out_le, out_be, "BE and LE ya16 luma outputs must agree");
}
