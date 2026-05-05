//! Scalar Ya16 → {RGB, RGBA, RGB-u16, RGBA-u16, luma, luma-u16, HSV} kernels.
//!
//! Source is a `&[u16]` packed plane in `[Y0, A0, Y1, A1, ...]` order.
//! Each pixel occupies 2 u16 elements: Y at offset `2*x`, A at offset `2*x + 1`.
//!
//! # u8 outputs — downshift `>> 8`
//!
//! Y and A are narrowed from 16-bit to 8-bit via truncating right-shift:
//! `(sample >> 8) as u8`. This matches FFmpeg's `swscale` behavior for
//! big-depth-to-u8 conversions (consistent downward-bias truncation).
//!
//! # u16 outputs — native pass-through
//!
//! Y and A are forwarded as-is (native 16-bit depth).
//!
//! # HSV gray fast-path
//!
//! Gray sources are achromatic (S = 0 identically). H=0, S=0, V = Y >> 8.
//! α is dropped for HSV output.

/// Ya16 → packed u8 RGB. Y `>> 8`, broadcast R=G=B; α dropped.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgb_row(packed: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let y8 = (packed[x * 2] >> 8) as u8;
    let i = x * 3;
    rgb_out[i] = y8;
    rgb_out[i + 1] = y8;
    rgb_out[i + 2] = y8;
  }
}

/// Ya16 → packed u8 RGBA. Y `>> 8`, broadcast R=G=B; A `>> 8` from source slot 1.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgba_row(packed: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let y8 = (packed[x * 2] >> 8) as u8;
    let a8 = (packed[x * 2 + 1] >> 8) as u8;
    let i = x * 4;
    rgba_out[i] = y8;
    rgba_out[i + 1] = y8;
    rgba_out[i + 2] = y8;
    rgba_out[i + 3] = a8;
  }
}

/// Ya16 → packed u16 RGB. Y native u16, broadcast R=G=B=Y; α dropped.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgb_u16_row(packed: &[u16], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let y = packed[x * 2];
    let i = x * 3;
    rgb_u16_out[i] = y;
    rgb_u16_out[i + 1] = y;
    rgb_u16_out[i + 2] = y;
  }
}

/// Ya16 → packed u16 RGBA. Y native u16, broadcast; A native u16 from source slot 1.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgba_u16_row(packed: &[u16], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let y = packed[x * 2];
    let a = packed[x * 2 + 1];
    let i = x * 4;
    rgba_u16_out[i] = y;
    rgba_u16_out[i + 1] = y;
    rgba_u16_out[i + 2] = y;
    rgba_u16_out[i + 3] = a;
  }
}

/// Ya16 → luma u8. Y `>> 8`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_luma_row(packed: &[u16], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(luma_out.len() >= width, "luma_out too short");
  for x in 0..width {
    luma_out[x] = (packed[x * 2] >> 8) as u8;
  }
}

/// Ya16 → luma u16. Y native u16 pass-through.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_luma_u16_row(packed: &[u16], luma_u16_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(luma_u16_out.len() >= width, "luma_u16_out too short");
  for x in 0..width {
    luma_u16_out[x] = packed[x * 2];
  }
}

/// Ya16 → HSV u8. Gray fast-path: H=0, S=0, V = Y `>> 8`. α dropped.
///
/// See [`super::gray::gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_hsv_row(
  packed: &[u16],
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
    v_out[x] = (packed[x * 2] >> 8) as u8;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  // Helper: make packed [Y, A, Y, A, ...] from pairs.
  fn packed_ya(pairs: &[(u16, u16)]) -> std::vec::Vec<u16> {
    pairs.iter().flat_map(|&(y, a)| [y, a]).collect()
  }

  // ---- ya16_to_rgb_row -------------------------------------------------------

  #[test]
  fn ya16_to_rgb_downshifts_y_drops_alpha() {
    // Y=0x8000, A=0x4000 → rgb [0x80, 0x80, 0x80]
    let p = packed_ya(&[(0x8000, 0x4000)]);
    let mut out = [0u8; 3];
    ya16_to_rgb_row(&p, &mut out, 1);
    assert_eq!(out, [0x80, 0x80, 0x80]);
  }

  #[test]
  fn ya16_to_rgb_zero_pixel() {
    let p = packed_ya(&[(0, 0)]);
    let mut out = [0xFFu8; 3];
    ya16_to_rgb_row(&p, &mut out, 1);
    assert_eq!(out, [0, 0, 0]);
  }

  #[test]
  fn ya16_to_rgb_max_y() {
    let p = packed_ya(&[(0xFFFF, 0)]);
    let mut out = [0u8; 3];
    ya16_to_rgb_row(&p, &mut out, 1);
    assert_eq!(out, [0xFF, 0xFF, 0xFF]);
  }

  // ---- ya16_to_rgba_row -----------------------------------------------------

  #[test]
  fn ya16_to_rgba_downshifts_y_and_alpha() {
    // Y=0x8000, A=0x4000 → rgba [0x80, 0x80, 0x80, 0x40]
    let p = packed_ya(&[(0x8000, 0x4000)]);
    let mut out = [0u8; 4];
    ya16_to_rgba_row(&p, &mut out, 1);
    assert_eq!(out, [0x80, 0x80, 0x80, 0x40]);
  }

  #[test]
  fn ya16_to_rgba_two_pixels() {
    let p = packed_ya(&[(0x8000, 0x4000), (0x1000, 0x0800)]);
    let mut out = [0u8; 8];
    ya16_to_rgba_row(&p, &mut out, 2);
    assert_eq!(&out[0..4], &[0x80, 0x80, 0x80, 0x40]);
    assert_eq!(&out[4..8], &[0x10, 0x10, 0x10, 0x08]);
  }

  // ---- ya16_to_rgb_u16_row --------------------------------------------------

  #[test]
  fn ya16_to_rgb_u16_native_y_broadcast() {
    // Y=0x8000 native, broadcast
    let p = packed_ya(&[(0x8000, 0x4000)]);
    let mut out = [0u16; 3];
    ya16_to_rgb_u16_row(&p, &mut out, 1);
    assert_eq!(out, [0x8000, 0x8000, 0x8000]);
  }

  #[test]
  fn ya16_to_rgb_u16_zero() {
    let p = packed_ya(&[(0, 0)]);
    let mut out = [0xFFFFu16; 3];
    ya16_to_rgb_u16_row(&p, &mut out, 1);
    assert_eq!(out, [0, 0, 0]);
  }

  // ---- ya16_to_rgba_u16_row -------------------------------------------------

  #[test]
  fn ya16_to_rgba_u16_native_y_and_alpha() {
    // Y=0x8000, A=0x4000 → rgba_u16 [0x8000, 0x8000, 0x8000, 0x4000]
    let p = packed_ya(&[(0x8000, 0x4000)]);
    let mut out = [0u16; 4];
    ya16_to_rgba_u16_row(&p, &mut out, 1);
    assert_eq!(out, [0x8000, 0x8000, 0x8000, 0x4000]);
  }

  // ---- ya16_to_luma_row -----------------------------------------------------

  #[test]
  fn ya16_to_luma_downshifts() {
    let p = packed_ya(&[(0x8000, 0x4000), (0x0000, 0xFFFF)]);
    let mut out = [0u8; 2];
    ya16_to_luma_row(&p, &mut out, 2);
    assert_eq!(out, [0x80, 0x00]);
  }

  // ---- ya16_to_luma_u16_row -------------------------------------------------

  #[test]
  fn ya16_to_luma_u16_native_passthrough() {
    let p = packed_ya(&[(0x8000, 0x0000)]);
    let mut out = [0u16; 1];
    ya16_to_luma_u16_row(&p, &mut out, 1);
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
    ya16_to_hsv_row(&p, &mut h, &mut s, &mut v, 1);
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
    ya16_to_hsv_row(&p, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 0);
  }

  #[test]
  fn ya16_to_hsv_max_luma() {
    let p = packed_ya(&[(0xFFFF, 0)]);
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    ya16_to_hsv_row(&p, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 0xFF);
  }
}
