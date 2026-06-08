//! Tests for `crate::row::scalar::grayf32`.

use super::*;

/// Re-encode a host-native f32 slice as LE-encoded byte storage (each element
/// stored with LE u32-bit byte layout). Kernels called with `BE = false`
/// recover the intended host-native value via `u32::from_le` on both LE
/// (no-op) and BE (byte-swap) hosts.
fn as_le_f32(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

// ---- grayf32_to_rgb_row --------------------------------------------------

#[test]
fn grayf32_to_rgb_zero() {
  let plane = [0.0f32];
  let mut out = [0xFFu8; 3];
  grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0, 0, 0]);
}

#[test]
fn grayf32_to_rgb_max() {
  let plane = as_le_f32(&[1.0f32]);
  let mut out = [0u8; 3];
  grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [255, 255, 255]);
}

#[test]
fn grayf32_to_rgb_mid() {
  // Mid-gray Y=0.5 with round-half-up:
  //   0.5 * 255      = 127.5  (pure truncation would give 127)
  //   127.5 + 0.5    = 128.0  (round-half-up adds 0.5 first)
  //   trunc(128.0)   = 128
  // See module-level "Rounding (float → integer)" doc — `+ 0.5 then
  // truncate` is the contract this crate uses across scalar + SIMD.
  let plane = as_le_f32(&[0.5f32]);
  let mut out = [0u8; 3];
  grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [128, 128, 128]);
}

#[test]
fn grayf32_to_rgb_saturates_high() {
  let plane = as_le_f32(&[1.5f32]);
  let mut out = [0u8; 3];
  grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [255, 255, 255]);
}

#[test]
fn grayf32_to_rgb_saturates_low() {
  let plane = [-0.1f32];
  let mut out = [0xFFu8; 3];
  grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0, 0, 0]);
}

// ---- grayf32_to_rgba_row -------------------------------------------------

#[test]
fn grayf32_to_rgba_zero_alpha_opaque() {
  let plane = [0.0f32];
  let mut out = [0u8; 4];
  grayf32_to_rgba_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0, 0, 0, 0xFF]);
}

#[test]
fn grayf32_to_rgba_max_alpha_opaque() {
  let plane = as_le_f32(&[1.0f32]);
  let mut out = [0u8; 4];
  grayf32_to_rgba_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [255, 255, 255, 0xFF]);
}

// ---- grayf32_to_rgb_u16_row ----------------------------------------------

#[test]
fn grayf32_to_rgb_u16_zero() {
  let plane = [0.0f32];
  let mut out = [0xFFFFu16; 3];
  grayf32_to_rgb_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0, 0, 0]);
}

#[test]
fn grayf32_to_rgb_u16_max() {
  let plane = as_le_f32(&[1.0f32]);
  let mut out = [0u16; 3];
  grayf32_to_rgb_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [65535, 65535, 65535]);
}

#[test]
fn grayf32_to_rgb_u16_saturates_high() {
  let plane = as_le_f32(&[2.0f32]);
  let mut out = [0u16; 3];
  grayf32_to_rgb_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [65535, 65535, 65535]);
}

// ---- grayf32_to_rgba_u16_row ---------------------------------------------

#[test]
fn grayf32_to_rgba_u16_opaque() {
  let plane = as_le_f32(&[1.0f32]);
  let mut out = [0u16; 4];
  grayf32_to_rgba_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [65535, 65535, 65535, 0xFFFF]);
}

// ---- grayf32_to_rgb_f32_row ----------------------------------------------

#[test]
fn grayf32_to_rgb_f32_lossless_replicate() {
  // Non-clamped value preserved exactly. Output is host-native f32.
  let plane = as_le_f32(&[1.5f32]);
  let mut out = [0.0f32; 3];
  grayf32_to_rgb_f32_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [1.5, 1.5, 1.5]);
}

#[test]
fn grayf32_to_rgb_f32_negative_preserved() {
  let plane = as_le_f32(&[-0.5f32]);
  let mut out = [0.0f32; 3];
  grayf32_to_rgb_f32_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [-0.5, -0.5, -0.5]);
}

// ---- grayf32_to_luma_row -------------------------------------------------

#[test]
fn grayf32_to_luma_zero() {
  let plane = [0.0f32];
  let mut out = [0xFFu8; 1];
  grayf32_to_luma_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [0]);
}

#[test]
fn grayf32_to_luma_max() {
  let plane = as_le_f32(&[1.0f32]);
  let mut out = [0u8; 1];
  grayf32_to_luma_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [255]);
}

// ---- grayf32_to_luma_u16_row ---------------------------------------------

#[test]
fn grayf32_to_luma_u16_max() {
  let plane = as_le_f32(&[1.0f32]);
  let mut out = [0u16; 1];
  grayf32_to_luma_u16_row::<false>(&plane, &mut out, 1);
  assert_eq!(out, [65535]);
}

// ---- grayf32_to_luma_f32_row ---------------------------------------------

#[test]
fn grayf32_to_luma_f32_identity() {
  let plane = as_le_f32(&[0.0f32, 0.5, 1.0, 1.5, -0.1]);
  let mut out = [99.0f32; 5];
  grayf32_to_luma_f32_row::<false>(&plane, &mut out, 5);
  // Lossless pass-through — exact bit equality. Output is host-native f32.
  assert_eq!(out, [0.0, 0.5, 1.0, 1.5, -0.1]);
}

// ---- grayf32_to_hsv_row --------------------------------------------------

#[test]
fn grayf32_to_hsv_zero() {
  let plane = [0.0f32];
  let mut h = [0xFFu8; 1];
  let mut s = [0xFFu8; 1];
  let mut v = [0u8; 1];
  grayf32_to_hsv_row::<false>(&plane, &mut h, &mut s, &mut v, 1);
  assert_eq!(h[0], 0, "H must be 0 for achromatic source");
  assert_eq!(s[0], 0, "S must be 0 for achromatic source");
  assert_eq!(v[0], 0);
}

#[test]
fn grayf32_to_hsv_max() {
  let plane = as_le_f32(&[1.0f32]);
  let mut h = [0u8; 1];
  let mut s = [0u8; 1];
  let mut v = [0u8; 1];
  grayf32_to_hsv_row::<false>(&plane, &mut h, &mut s, &mut v, 1);
  assert_eq!(h[0], 0);
  assert_eq!(s[0], 0);
  assert_eq!(v[0], 255);
}

#[test]
fn grayf32_to_hsv_mid() {
  // 0.5 → (0.5 * 255 + 0.5) as u8 = 128
  let plane = as_le_f32(&[0.5f32]);
  let mut h = [0u8; 1];
  let mut s = [0u8; 1];
  let mut v = [0u8; 1];
  grayf32_to_hsv_row::<false>(&plane, &mut h, &mut s, &mut v, 1);
  assert_eq!(h[0], 0);
  assert_eq!(s[0], 0);
  assert_eq!(v[0], 128);
}

#[test]
fn grayf32_to_hsv_clamps_hdr() {
  // HDR value > 1.0 saturates to V=255.
  let plane = as_le_f32(&[2.0f32]);
  let mut h = [0u8; 1];
  let mut s = [0u8; 1];
  let mut v = [0u8; 1];
  grayf32_to_hsv_row::<false>(&plane, &mut h, &mut s, &mut v, 1);
  assert_eq!(v[0], 255);
}

#[test]
fn grayf32_to_rgb_multi_pixel() {
  let plane = as_le_f32(&[0.0f32, 1.0, 0.5]);
  let mut out = [0u8; 9];
  grayf32_to_rgb_row::<false>(&plane, &mut out, 3);
  assert_eq!(&out[0..3], &[0, 0, 0]);
  assert_eq!(&out[3..6], &[255, 255, 255]);
  assert_eq!(&out[6..9], &[128, 128, 128]); // 0.5 → 128
}

// ---- BE parity tests: grayf32 ---------------------------------------------
// Pattern: build a single host-native `intended` fixture, materialise it as
// LE-encoded bytes via `as_le_f32` and BE-encoded bytes via `as_be_f32`, run
// both `<false>` and `<true>` kernels, and pin each output against an
// absolute scalar reference so the parity assertion cannot pass on two
// equally corrupt decodes.

/// Mirror of `as_le_f32` for kernels invoked with `BE = true`. Combined
/// with `as_le_f32`, lets a single host-native `intended` fixture drive
/// both `<false>` and `<true>` kernel paths so they decode the same logical
/// bit pattern on every host.
fn as_be_f32(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect()
}

// -- Scalar references for the BE-parity tests --
//
// Walk host-native `intended` planes and reproduce each kernel's documented
// behaviour without going through any byte-order conversion. Pinning the
// LE / BE outputs against these absolute references prevents the parity
// assertion from passing in lock-step on two equally corrupt decode paths.

fn ref_grayf32_to_rgb(intended: &[f32], width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width * 3];
  for (x, &y) in intended[..width].iter().enumerate() {
    let v = f32_to_u8(y);
    out[x * 3] = v;
    out[x * 3 + 1] = v;
    out[x * 3 + 2] = v;
  }
  out
}

fn ref_grayf32_to_luma(intended: &[f32], width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec![0u8; width];
  for (x, &y) in intended[..width].iter().enumerate() {
    out[x] = f32_to_u8(y);
  }
  out
}

/// Lossless f32 → f32 reference: pure host-native pass-through.
fn ref_grayf32_to_luma_f32(intended: &[f32], width: usize) -> std::vec::Vec<f32> {
  intended[..width].to_vec()
}

#[test]
fn grayf32_be_parity_rgb() {
  let intended = [0.5f32];
  let le = as_le_f32(&intended);
  let be = as_be_f32(&intended);
  let mut out_le = [0u8; 3];
  let mut out_be = [0u8; 3];
  grayf32_to_rgb_row::<false>(&le, &mut out_le, 1);
  grayf32_to_rgb_row::<true>(&be, &mut out_be, 1);
  let expected = ref_grayf32_to_rgb(&intended, 1);
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
  assert_eq!(out_le, out_be, "BE and LE grayf32 rgb outputs must agree");
}

#[test]
fn grayf32_be_parity_luma() {
  let intended = [0.25f32];
  let le = as_le_f32(&intended);
  let be = as_be_f32(&intended);
  let mut out_le = [0u8; 1];
  let mut out_be = [0u8; 1];
  grayf32_to_luma_row::<false>(&le, &mut out_le, 1);
  grayf32_to_luma_row::<true>(&be, &mut out_be, 1);
  let expected = ref_grayf32_to_luma(&intended, 1);
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
  assert_eq!(out_le, out_be, "BE and LE grayf32 luma outputs must agree");
}

/// The integer-output `grayf32_be_parity_*` tests don't reach the lossless
/// `grayf32_to_luma_f32_row::<true>` (BE-encoded f32 → host-native f32)
/// fast/slow paths. Build a single host-native `intended` fixture,
/// materialise it as LE / BE byte storage, run both kernels, pin both
/// outputs against an absolute scalar reference (compared bitwise to be
/// NaN-safe), and additionally assert bit-for-bit LE↔BE parity.
///
/// Path coverage by host:
///   LE host: LE kernel = memcpy fast path; BE kernel = slow swap path.
///   BE host: LE kernel = slow swap path; BE kernel = memcpy fast path.
/// Either way both outputs must agree bit-for-bit, exercising the
/// `BE == HOST_NATIVE_BE` gate from both directions.
#[test]
fn grayf32_to_luma_f32_row_be_le_parity_lossless() {
  // Mix of normal, HDR, negative, subnormal, and exact-zero values to
  // ensure non-symmetric byte layouts in every position.
  let intended: std::vec::Vec<f32> = std::vec![0.25f32, 1.5, -0.5, 1e-5, 0.0, 65504.0];
  let width = intended.len();
  let le = as_le_f32(&intended);
  let be = as_be_f32(&intended);
  let mut out_le = std::vec![0.0f32; width];
  let mut out_be = std::vec![0.0f32; width];
  grayf32_to_luma_f32_row::<false>(&le, &mut out_le, width);
  grayf32_to_luma_f32_row::<true>(&be, &mut out_be, width);
  let expected = ref_grayf32_to_luma_f32(&intended, width);
  // Bitwise equality is NaN-safe and confirms no rounding/clamping was
  // applied along either path.
  let bits_le: std::vec::Vec<u32> = out_le.iter().map(|v| v.to_bits()).collect();
  let bits_be: std::vec::Vec<u32> = out_be.iter().map(|v| v.to_bits()).collect();
  let bits_expected: std::vec::Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
  assert_eq!(
    bits_le, bits_expected,
    "LE path must match scalar reference"
  );
  assert_eq!(
    bits_be, bits_expected,
    "BE path must match scalar reference"
  );
  assert_eq!(
    bits_le, bits_be,
    "BE and LE grayf32 luma_f32 outputs must match bit-for-bit"
  );
}
