//! Tests for `crate::row::scalar::yaf32`.

use super::*;

/// Re-encode a host-native f32 slice as LE-encoded byte storage. Kernels called
/// with `BE = false` recover the intended host-native value via `u32::from_le`
/// on both LE (no-op) and BE (byte-swap) hosts.
fn as_le(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
    .collect()
}

/// Re-encode a host-native f32 slice as BE-encoded byte storage, for the
/// `BE = true` kernels.
fn as_be(host: &[f32]) -> std::vec::Vec<f32> {
  host
    .iter()
    .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_be_bytes())))
    .collect()
}

// ---- yaf32_to_rgb_row (Y broadcast, alpha dropped) -----------------------

#[test]
fn rgb_broadcasts_y_drops_alpha() {
  // packed [Y0=0.5, A0=0.0, Y1=1.0, A1=0.25]
  let packed = as_le(&[0.5, 0.0, 1.0, 0.25]);
  let mut out = [0u8; 6];
  yaf32_to_rgb_row::<false>(&packed, &mut out, 2);
  // 0.5*255+0.5=128; 1.0 -> 255. Alpha not present in RGB.
  assert_eq!(out, [128, 128, 128, 255, 255, 255]);
}

#[test]
fn rgb_saturates_and_clamps() {
  let packed = as_le(&[1.5, 0.0, -0.1, 0.0]);
  let mut out = [9u8; 6];
  yaf32_to_rgb_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [255, 255, 255, 0, 0, 0]);
}

#[test]
fn rgb_be_matches_le() {
  let host = [0.25f32, 0.9, 0.5, 0.1];
  let mut le_out = [0u8; 6];
  let mut be_out = [0u8; 6];
  yaf32_to_rgb_row::<false>(&as_le(&host), &mut le_out, 2);
  yaf32_to_rgb_row::<true>(&as_be(&host), &mut be_out, 2);
  assert_eq!(le_out, be_out);
}

// ---- yaf32_to_rgba_row (Y broadcast + real alpha) ------------------------

#[test]
fn rgba_carries_source_alpha() {
  // Y=0.5, A=0.5 -> [128,128,128,128]; Y=1.0, A=0.0 -> [255,255,255,0]
  let packed = as_le(&[0.5, 0.5, 1.0, 0.0]);
  let mut out = [9u8; 8];
  yaf32_to_rgba_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [128, 128, 128, 128, 255, 255, 255, 0]);
}

#[test]
fn rgba_alpha_saturates() {
  // A=2.0 clamps to 255; A=-0.5 clamps to 0.
  let packed = as_le(&[1.0, 2.0, 0.0, -0.5]);
  let mut out = [9u8; 8];
  yaf32_to_rgba_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [255, 255, 255, 255, 0, 0, 0, 0]);
}

#[test]
fn rgba_be_matches_le() {
  let host = [0.5f32, 0.25, 0.75, 1.0];
  let mut le_out = [0u8; 8];
  let mut be_out = [0u8; 8];
  yaf32_to_rgba_row::<false>(&as_le(&host), &mut le_out, 2);
  yaf32_to_rgba_row::<true>(&as_be(&host), &mut be_out, 2);
  assert_eq!(le_out, be_out);
}

// ---- u16 outputs ----------------------------------------------------------

#[test]
fn rgb_u16_broadcasts() {
  let packed = as_le(&[1.0, 0.0]);
  let mut out = [0u16; 3];
  yaf32_to_rgb_u16_row::<false>(&packed, &mut out, 1);
  assert_eq!(out, [65535, 65535, 65535]);
}

#[test]
fn rgba_u16_carries_alpha() {
  let packed = as_le(&[1.0, 0.5]);
  let mut out = [0u16; 4];
  yaf32_to_rgba_u16_row::<false>(&packed, &mut out, 1);
  // 0.5*65535+0.5 = 32768.
  assert_eq!(out, [65535, 65535, 65535, 32768]);
}

// ---- yaf32_to_rgb_f32_row (lossless Y replicate, alpha dropped) ----------

#[test]
fn rgb_f32_lossless_replicate() {
  // HDR + negative preserved; alpha dropped.
  let packed = as_le(&[2.5, 0.3, -1.0, 0.7]);
  let mut out = [0.0f32; 6];
  yaf32_to_rgb_f32_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [2.5, 2.5, 2.5, -1.0, -1.0, -1.0]);
}

// ---- luma --------------------------------------------------------------

#[test]
fn luma_clamps_and_rounds() {
  let packed = as_le(&[0.5, 0.9, 1.5, 0.0, -0.2, 0.0]);
  let mut out = [9u8; 3];
  yaf32_to_luma_row::<false>(&packed, &mut out, 3);
  assert_eq!(out, [128, 255, 0]);
}

#[test]
fn luma_u16_clamps() {
  let packed = as_le(&[0.5, 0.0]);
  let mut out = [0u16; 1];
  yaf32_to_luma_u16_row::<false>(&packed, &mut out, 1);
  assert_eq!(out, [32768]);
}

#[test]
fn luma_f32_lossless() {
  let packed = as_le(&[3.0, 0.1, -2.0, 0.9]);
  let mut out = [0.0f32; 2];
  yaf32_to_luma_f32_row::<false>(&packed, &mut out, 2);
  assert_eq!(out, [3.0, -2.0]);
}

#[test]
fn luma_f32_be_matches_le() {
  let host = [0.5f32, 0.0, 0.25, 0.0];
  let mut le_out = [0.0f32; 2];
  let mut be_out = [0.0f32; 2];
  yaf32_to_luma_f32_row::<false>(&as_le(&host), &mut le_out, 2);
  yaf32_to_luma_f32_row::<true>(&as_be(&host), &mut be_out, 2);
  assert_eq!(le_out, be_out);
}

// ---- hsv (gray achromatic, alpha dropped) -------------------------------

#[test]
fn hsv_gray_fast_path() {
  let packed = as_le(&[0.5, 0.9, 1.0, 0.0]);
  let mut h = [9u8; 2];
  let mut s = [9u8; 2];
  let mut v = [9u8; 2];
  yaf32_to_hsv_row::<false>(&packed, &mut h, &mut s, &mut v, 2);
  assert_eq!(h, [0, 0]);
  assert_eq!(s, [0, 0]);
  assert_eq!(v, [128, 255]);
}
