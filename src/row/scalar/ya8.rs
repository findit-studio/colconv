//! Scalar Ya8 → {RGB, RGBA, RGB-u16, RGBA-u16, luma, luma-u16, HSV} kernels.
//!
//! Source is a `&[u8]` packed plane in `[Y0, A0, Y1, A1, ...]` order.
//! Each pixel occupies 2 bytes: Y at offset `2*x`, A at offset `2*x + 1`.
//!
//! # RGB / RGBA extraction
//!
//! Y is broadcast to R=G=B. α is either dropped (RGB) or passed through
//! from source slot 1 (RGBA).
//!
//! # u16 outputs
//!
//! Y and A are zero-extended: `y as u16`, `a as u16`.
//!
//! # HSV gray fast-path
//!
//! Gray sources are achromatic (S = 0 identically). H=0, S=0, V=Y.
//! α is dropped for HSV output.

#![allow(dead_code)]

/// Ya8 → packed u8 RGB. Broadcast Y to R=G=B; α dropped.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya8_to_rgb_row(packed: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let y = packed[x * 2];
    let i = x * 3;
    rgb_out[i] = y;
    rgb_out[i + 1] = y;
    rgb_out[i + 2] = y;
  }
}

/// Ya8 → packed u8 RGBA. Broadcast Y to R=G=B; α from source slot 1.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya8_to_rgba_row(packed: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let y = packed[x * 2];
    let a = packed[x * 2 + 1];
    let i = x * 4;
    rgba_out[i] = y;
    rgba_out[i + 1] = y;
    rgba_out[i + 2] = y;
    rgba_out[i + 3] = a;
  }
}

/// Ya8 → packed u16 RGB. Zero-extend Y to u16, broadcast R=G=B=Y; α dropped.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya8_to_rgb_u16_row(packed: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let y = packed[x * 2] as u16;
    let i = x * 3;
    rgb_u16_out[i] = y;
    rgb_u16_out[i + 1] = y;
    rgb_u16_out[i + 2] = y;
  }
}

/// Ya8 → packed u16 RGBA. Zero-extend Y and A, broadcast Y to R=G=B; α from source.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya8_to_rgba_u16_row(packed: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let y = packed[x * 2] as u16;
    let a = packed[x * 2 + 1] as u16;
    let i = x * 4;
    rgba_u16_out[i] = y;
    rgba_u16_out[i + 1] = y;
    rgba_u16_out[i + 2] = y;
    rgba_u16_out[i + 3] = a;
  }
}

/// Ya8 → luma u8. Extract Y bytes (`out[x] = packed[2*x]`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya8_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(luma_out.len() >= width, "luma_out too short");
  for x in 0..width {
    luma_out[x] = packed[x * 2];
  }
}

/// Ya8 → luma u16. Zero-extend Y → u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya8_to_luma_u16_row(packed: &[u8], luma_u16_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(luma_u16_out.len() >= width, "luma_u16_out too short");
  for x in 0..width {
    luma_u16_out[x] = packed[x * 2] as u16;
  }
}

/// Ya8 → HSV u8. Gray fast-path: H=0, S=0, V=Y. α dropped.
///
/// See [`super::gray::gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya8_to_hsv_row(
  packed: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(h_out.len() >= width, "h_out too short");
  debug_assert!(s_out.len() >= width, "s_out too short");
  debug_assert!(v_out.len() >= width, "v_out too short");
  for x in 0..width {
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = packed[x * 2];
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  // Helper: make packed [Y, A, Y, A, ...] from pairs.
  fn packed_ya(pairs: &[(u8, u8)]) -> std::vec::Vec<u8> {
    pairs.iter().flat_map(|&(y, a)| [y, a]).collect()
  }

  // ---- ya8_to_rgb_row -------------------------------------------------------

  #[test]
  fn ya8_to_rgb_broadcasts_y_drops_alpha() {
    // Y=100, A=200 → rgb [100, 100, 100]
    let p = packed_ya(&[(100, 200)]);
    let mut out = [0u8; 3];
    ya8_to_rgb_row(&p, &mut out, 1);
    assert_eq!(out, [100, 100, 100]);
  }

  #[test]
  fn ya8_to_rgb_zero_pixel() {
    let p = packed_ya(&[(0, 0)]);
    let mut out = [0xFFu8; 3];
    ya8_to_rgb_row(&p, &mut out, 1);
    assert_eq!(out, [0, 0, 0]);
  }

  #[test]
  fn ya8_to_rgb_max_pixel() {
    let p = packed_ya(&[(255, 0)]);
    let mut out = [0u8; 3];
    ya8_to_rgb_row(&p, &mut out, 1);
    assert_eq!(out, [255, 255, 255]);
  }

  // ---- ya8_to_rgba_row ------------------------------------------------------

  #[test]
  fn ya8_to_rgba_broadcasts_y_passes_alpha() {
    // Y=100, A=200 → rgba [100, 100, 100, 200]
    let p = packed_ya(&[(100, 200)]);
    let mut out = [0u8; 4];
    ya8_to_rgba_row(&p, &mut out, 1);
    assert_eq!(out, [100, 100, 100, 200]);
  }

  #[test]
  fn ya8_to_rgba_two_pixels() {
    // Pixel 0: Y=50, A=150; Pixel 1: Y=200, A=255
    let p = packed_ya(&[(50, 150), (200, 255)]);
    let mut out = [0u8; 8];
    ya8_to_rgba_row(&p, &mut out, 2);
    assert_eq!(&out[0..4], &[50, 50, 50, 150]);
    assert_eq!(&out[4..8], &[200, 200, 200, 255]);
  }

  // ---- ya8_to_rgb_u16_row ---------------------------------------------------

  #[test]
  fn ya8_to_rgb_u16_zero_extends() {
    // Y=100 → u16 100
    let p = packed_ya(&[(100, 0)]);
    let mut out = [0u16; 3];
    ya8_to_rgb_u16_row(&p, &mut out, 1);
    assert_eq!(out, [100, 100, 100]);
  }

  #[test]
  fn ya8_to_rgb_u16_max_y() {
    let p = packed_ya(&[(255, 0)]);
    let mut out = [0u16; 3];
    ya8_to_rgb_u16_row(&p, &mut out, 1);
    assert_eq!(out, [255, 255, 255]);
  }

  // ---- ya8_to_rgba_u16_row --------------------------------------------------

  #[test]
  fn ya8_to_rgba_u16_passes_alpha_zero_extended() {
    // Y=100, A=200 → rgba_u16 [100, 100, 100, 200]
    let p = packed_ya(&[(100, 200)]);
    let mut out = [0u16; 4];
    ya8_to_rgba_u16_row(&p, &mut out, 1);
    assert_eq!(out, [100, 100, 100, 200]);
  }

  // ---- ya8_to_luma_row ------------------------------------------------------

  #[test]
  fn ya8_to_luma_extracts_y_bytes() {
    let p = packed_ya(&[(100, 200), (50, 25)]);
    let mut out = [0u8; 2];
    ya8_to_luma_row(&p, &mut out, 2);
    assert_eq!(out, [100, 50]);
  }

  // ---- ya8_to_luma_u16_row --------------------------------------------------

  #[test]
  fn ya8_to_luma_u16_zero_extends_y() {
    let p = packed_ya(&[(128, 0)]);
    let mut out = [0u16; 1];
    ya8_to_luma_u16_row(&p, &mut out, 1);
    assert_eq!(out[0], 128);
  }

  // ---- ya8_to_hsv_row -------------------------------------------------------

  #[test]
  fn ya8_to_hsv_h0_s0_v_y_drops_alpha() {
    // Y=100, A=200 → H=0, S=0, V=100
    let p = packed_ya(&[(100, 200)]);
    let mut h = [0xFFu8; 1];
    let mut s = [0xFFu8; 1];
    let mut v = [0u8; 1];
    ya8_to_hsv_row(&p, &mut h, &mut s, &mut v, 1);
    assert_eq!(h[0], 0);
    assert_eq!(s[0], 0);
    assert_eq!(v[0], 100);
  }

  #[test]
  fn ya8_to_hsv_zero_luma() {
    let p = packed_ya(&[(0, 255)]);
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0xFFu8; 1];
    ya8_to_hsv_row(&p, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 0);
  }

  #[test]
  fn ya8_to_hsv_max_luma() {
    let p = packed_ya(&[(255, 0)]);
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    ya8_to_hsv_row(&p, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 255);
  }
}
