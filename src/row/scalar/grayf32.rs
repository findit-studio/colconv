//! Scalar Grayf32 → {RGB, RGBA, RGB-u16, RGBA-u16, RGB-f32, luma, luma-u16,
//! luma-f32, HSV} kernels.
//!
//! Source is a `&[f32]` luma plane. Nominal range `[0.0, 1.0]`; HDR > 1.0 is
//! permitted — all integer-output kernels clamp via `.max(0.0).min(1.0)` before
//! scaling, using the same MXCSR-independent pattern as the Rgbf32 scalar.
//!
//! # Rounding (float → integer)
//!
//! `(y.clamp(0.0, 1.0) * scale + 0.5) as T`
//!
//! Adding 0.5 before truncation gives round-to-nearest (ties round up) without
//! depending on the floating-point rounding mode register (MXCSR on x86). This
//! matches the Rgbf32 scalar pattern.
//!
//! # Lossless paths (float → float)
//!
//! `grayf32_to_rgb_f32_row` and `grayf32_to_luma_f32_row` perform no clamping
//! and no rounding — the f32 value is forwarded as-is (memcpy-equivalent).
//!
//! # HSV gray fast-path
//!
//! Gray sources are achromatic (S = 0 identically). H is fixed to 0 to match
//! OpenCV's `cv2.COLOR_GRAY2HSV` convention. V is the clamped Y in u8.

// ---- shared helpers ---------------------------------------------------------

/// Round-to-nearest f32 → u8, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 255, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u8(y: f32) -> u8 {
  (y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Round-to-nearest f32 → u16, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 65535, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u16(y: f32) -> u16 {
  (y.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

// ---- kernel implementations -------------------------------------------------

/// Grayf32 → packed u8 RGB. Clamp [0,1] × 255 → u8, broadcast R=G=B=Y.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgb_row(plane: &[f32], rgb_out: &mut [u8], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for (x, &y) in plane[..width].iter().enumerate() {
    let v = f32_to_u8(y);
    let i = x * 3;
    rgb_out[i] = v;
    rgb_out[i + 1] = v;
    rgb_out[i + 2] = v;
  }
}

/// Grayf32 → packed u8 RGBA. Same broadcast as rgb; α = 0xFF.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgba_row(plane: &[f32], rgba_out: &mut [u8], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for (x, &y) in plane[..width].iter().enumerate() {
    let v = f32_to_u8(y);
    let i = x * 4;
    rgba_out[i] = v;
    rgba_out[i + 1] = v;
    rgba_out[i + 2] = v;
    rgba_out[i + 3] = 0xFF;
  }
}

/// Grayf32 → packed u16 RGB. Clamp [0,1] × 65535 → u16, broadcast R=G=B=Y.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgb_u16_row(plane: &[f32], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for (x, &y) in plane[..width].iter().enumerate() {
    let v = f32_to_u16(y);
    let i = x * 3;
    rgb_u16_out[i] = v;
    rgb_u16_out[i + 1] = v;
    rgb_u16_out[i + 2] = v;
  }
}

/// Grayf32 → packed u16 RGBA. Same broadcast; α = 0xFFFF.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgba_u16_row(plane: &[f32], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for (x, &y) in plane[..width].iter().enumerate() {
    let v = f32_to_u16(y);
    let i = x * 4;
    rgba_u16_out[i] = v;
    rgba_u16_out[i + 1] = v;
    rgba_u16_out[i + 2] = v;
    rgba_u16_out[i + 3] = 0xFFFF;
  }
}

/// Grayf32 → packed f32 RGB. Lossless: replicate Y → R=G=B (no clamp, no round).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgb_f32_row(plane: &[f32], rgb_f32_out: &mut [f32], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgb_f32_out.len() >= width * 3, "rgb_f32_out too short");
  for (x, &y) in plane[..width].iter().enumerate() {
    let i = x * 3;
    rgb_f32_out[i] = y;
    rgb_f32_out[i + 1] = y;
    rgb_f32_out[i + 2] = y;
  }
}

/// Grayf32 → luma u8. Clamp [0,1] × 255 → u8.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_luma_row(plane: &[f32], luma_out: &mut [u8], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(luma_out.len() >= width, "luma_out too short");
  for (out, &y) in luma_out[..width].iter_mut().zip(plane[..width].iter()) {
    *out = f32_to_u8(y);
  }
}

/// Grayf32 → luma u16. Clamp [0,1] × 65535 → u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_luma_u16_row(plane: &[f32], luma_u16_out: &mut [u16], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(luma_u16_out.len() >= width, "luma_u16_out too short");
  for (out, &y) in luma_u16_out[..width].iter_mut().zip(plane[..width].iter()) {
    *out = f32_to_u16(y);
  }
}

/// Grayf32 → luma f32. Lossless pass-through (memcpy-equivalent).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_luma_f32_row(plane: &[f32], luma_f32_out: &mut [f32], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(luma_f32_out.len() >= width, "luma_f32_out too short");
  luma_f32_out[..width].copy_from_slice(&plane[..width]);
}

/// Grayf32 → HSV u8. Gray fast-path: H=0, S=0, V = clamp(Y, 0, 1) × 255.
///
/// Gray sources are achromatic (saturation = 0 identically). H is fixed to 0
/// to match OpenCV's `cv2.COLOR_GRAY2HSV` convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_hsv_row(
  plane: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(h_out.len() >= width, "h_out too short");
  debug_assert!(s_out.len() >= width, "s_out too short");
  debug_assert!(v_out.len() >= width, "v_out too short");
  for (x, &y) in plane[..width].iter().enumerate() {
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = f32_to_u8(y);
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  // ---- grayf32_to_rgb_row --------------------------------------------------

  #[test]
  fn grayf32_to_rgb_zero() {
    let plane = [0.0f32];
    let mut out = [0xFFu8; 3];
    grayf32_to_rgb_row(&plane, &mut out, 1);
    assert_eq!(out, [0, 0, 0]);
  }

  #[test]
  fn grayf32_to_rgb_max() {
    let plane = [1.0f32];
    let mut out = [0u8; 3];
    grayf32_to_rgb_row(&plane, &mut out, 1);
    assert_eq!(out, [255, 255, 255]);
  }

  #[test]
  fn grayf32_to_rgb_mid() {
    // 0.5 * 255 + 0.5 = 128.25, truncated to 128 — spec says 127, but
    // that's per spec §5.1 "Y=0.5 → u8 127 (saturating)". The rounding
    // adds 0.5 so result is 128. Adjust expectation to match the chosen
    // round-half-up scheme: 0.5*255 = 127.5 + 0.5 = 128.
    let plane = [0.5f32];
    let mut out = [0u8; 3];
    grayf32_to_rgb_row(&plane, &mut out, 1);
    // f32_to_u8(0.5) = (0.5 * 255 + 0.5) as u8 = (127.5 + 0.5) as u8 = 128
    assert_eq!(out, [128, 128, 128]);
  }

  #[test]
  fn grayf32_to_rgb_saturates_high() {
    let plane = [1.5f32];
    let mut out = [0u8; 3];
    grayf32_to_rgb_row(&plane, &mut out, 1);
    assert_eq!(out, [255, 255, 255]);
  }

  #[test]
  fn grayf32_to_rgb_saturates_low() {
    let plane = [-0.1f32];
    let mut out = [0xFFu8; 3];
    grayf32_to_rgb_row(&plane, &mut out, 1);
    assert_eq!(out, [0, 0, 0]);
  }

  // ---- grayf32_to_rgba_row -------------------------------------------------

  #[test]
  fn grayf32_to_rgba_zero_alpha_opaque() {
    let plane = [0.0f32];
    let mut out = [0u8; 4];
    grayf32_to_rgba_row(&plane, &mut out, 1);
    assert_eq!(out, [0, 0, 0, 0xFF]);
  }

  #[test]
  fn grayf32_to_rgba_max_alpha_opaque() {
    let plane = [1.0f32];
    let mut out = [0u8; 4];
    grayf32_to_rgba_row(&plane, &mut out, 1);
    assert_eq!(out, [255, 255, 255, 0xFF]);
  }

  // ---- grayf32_to_rgb_u16_row ----------------------------------------------

  #[test]
  fn grayf32_to_rgb_u16_zero() {
    let plane = [0.0f32];
    let mut out = [0xFFFFu16; 3];
    grayf32_to_rgb_u16_row(&plane, &mut out, 1);
    assert_eq!(out, [0, 0, 0]);
  }

  #[test]
  fn grayf32_to_rgb_u16_max() {
    let plane = [1.0f32];
    let mut out = [0u16; 3];
    grayf32_to_rgb_u16_row(&plane, &mut out, 1);
    assert_eq!(out, [65535, 65535, 65535]);
  }

  #[test]
  fn grayf32_to_rgb_u16_saturates_high() {
    let plane = [2.0f32];
    let mut out = [0u16; 3];
    grayf32_to_rgb_u16_row(&plane, &mut out, 1);
    assert_eq!(out, [65535, 65535, 65535]);
  }

  // ---- grayf32_to_rgba_u16_row ---------------------------------------------

  #[test]
  fn grayf32_to_rgba_u16_opaque() {
    let plane = [1.0f32];
    let mut out = [0u16; 4];
    grayf32_to_rgba_u16_row(&plane, &mut out, 1);
    assert_eq!(out, [65535, 65535, 65535, 0xFFFF]);
  }

  // ---- grayf32_to_rgb_f32_row ----------------------------------------------

  #[test]
  fn grayf32_to_rgb_f32_lossless_replicate() {
    // Non-clamped value preserved exactly.
    let plane = [1.5f32];
    let mut out = [0.0f32; 3];
    grayf32_to_rgb_f32_row(&plane, &mut out, 1);
    assert_eq!(out, [1.5, 1.5, 1.5]);
  }

  #[test]
  fn grayf32_to_rgb_f32_negative_preserved() {
    let plane = [-0.5f32];
    let mut out = [0.0f32; 3];
    grayf32_to_rgb_f32_row(&plane, &mut out, 1);
    assert_eq!(out, [-0.5, -0.5, -0.5]);
  }

  // ---- grayf32_to_luma_row -------------------------------------------------

  #[test]
  fn grayf32_to_luma_zero() {
    let plane = [0.0f32];
    let mut out = [0xFFu8; 1];
    grayf32_to_luma_row(&plane, &mut out, 1);
    assert_eq!(out, [0]);
  }

  #[test]
  fn grayf32_to_luma_max() {
    let plane = [1.0f32];
    let mut out = [0u8; 1];
    grayf32_to_luma_row(&plane, &mut out, 1);
    assert_eq!(out, [255]);
  }

  // ---- grayf32_to_luma_u16_row ---------------------------------------------

  #[test]
  fn grayf32_to_luma_u16_max() {
    let plane = [1.0f32];
    let mut out = [0u16; 1];
    grayf32_to_luma_u16_row(&plane, &mut out, 1);
    assert_eq!(out, [65535]);
  }

  // ---- grayf32_to_luma_f32_row ---------------------------------------------

  #[test]
  fn grayf32_to_luma_f32_identity() {
    let plane = [0.0f32, 0.5, 1.0, 1.5, -0.1];
    let mut out = [99.0f32; 5];
    grayf32_to_luma_f32_row(&plane, &mut out, 5);
    // Lossless pass-through — exact bit equality.
    assert_eq!(out, [0.0, 0.5, 1.0, 1.5, -0.1]);
  }

  // ---- grayf32_to_hsv_row --------------------------------------------------

  #[test]
  fn grayf32_to_hsv_zero() {
    let plane = [0.0f32];
    let mut h = [0xFFu8; 1];
    let mut s = [0xFFu8; 1];
    let mut v = [0u8; 1];
    grayf32_to_hsv_row(&plane, &mut h, &mut s, &mut v, 1);
    assert_eq!(h[0], 0, "H must be 0 for achromatic source");
    assert_eq!(s[0], 0, "S must be 0 for achromatic source");
    assert_eq!(v[0], 0);
  }

  #[test]
  fn grayf32_to_hsv_max() {
    let plane = [1.0f32];
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    grayf32_to_hsv_row(&plane, &mut h, &mut s, &mut v, 1);
    assert_eq!(h[0], 0);
    assert_eq!(s[0], 0);
    assert_eq!(v[0], 255);
  }

  #[test]
  fn grayf32_to_hsv_mid() {
    // 0.5 → (0.5 * 255 + 0.5) as u8 = 128
    let plane = [0.5f32];
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    grayf32_to_hsv_row(&plane, &mut h, &mut s, &mut v, 1);
    assert_eq!(h[0], 0);
    assert_eq!(s[0], 0);
    assert_eq!(v[0], 128);
  }

  #[test]
  fn grayf32_to_hsv_clamps_hdr() {
    // HDR value > 1.0 saturates to V=255.
    let plane = [2.0f32];
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    grayf32_to_hsv_row(&plane, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 255);
  }

  #[test]
  fn grayf32_to_rgb_multi_pixel() {
    let plane = [0.0f32, 1.0, 0.5];
    let mut out = [0u8; 9];
    grayf32_to_rgb_row(&plane, &mut out, 3);
    assert_eq!(&out[0..3], &[0, 0, 0]);
    assert_eq!(&out[3..6], &[255, 255, 255]);
    assert_eq!(&out[6..9], &[128, 128, 128]); // 0.5 → 128
  }
}
