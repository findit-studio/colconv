//! Scalar gray → {RGB, RGBA, HSV, luma, luma_u16} kernels.
//!
//! # Gray8 (u8 source)
//!
//! `gray8_to_rgb_row`  — broadcast Y to each R/G/B channel.
//! `gray8_to_rgba_row` — broadcast Y + force α = 0xFF.
//! `gray8_to_hsv_row`  — H=0, S=0, V=Y (S=0 convention: H is fixed to 0).
//! `gray8_to_luma_u16_row` — zero-extend Y to u16 (`out[x] = y[x] as u16`).
//!   (luma u8 identity copy is handled by the sinker directly via
//!   `copy_from_slice`, no dedicated kernel needed.)
//!
//! # GrayN / Gray16 (u16 source)
//!
//! `gray_n_to_rgb_row<BITS>`       — mask → downshift to u8, broadcast.
//! `gray_n_to_rgba_row<BITS>`      — same + α = 0xFF.
//! `gray_n_to_rgb_u16_row<BITS>`   — mask, keep native depth, broadcast.
//! `gray_n_to_rgba_u16_row<BITS>`  — same + α = full-range max for BITS.
//! `gray_n_to_luma_row<BITS>`      — mask → downshift to u8.
//! `gray_n_to_luma_u16_row<BITS>`  — mask → identity (samples already u16).
//! `gray_n_to_hsv_row<BITS>`       — mask → downshift to u8 → H=0 S=0 V=Y8.
//!
//! `gray16_to_rgb_row`    — `>> 8` to u8, broadcast.
//! `gray16_to_rgba_row`   — same + α = 0xFF.
//! `gray16_to_rgb_u16_row`  — native u16, broadcast.
//! `gray16_to_rgba_u16_row` — native u16 + α = 0xFFFF.
//! `gray16_to_luma_row`   — `>> 8` to u8.
//! `gray16_to_luma_u16_row` — identity copy to u16.
//! `gray16_to_hsv_row`    — `>> 8` to u8 → H=0 S=0 V=Y8.
//!
//! # HSV S=0 convention
//!
//! When S=0 (which is always for gray sources — delta = 0), H is set to 0.
//! This matches OpenCV `cv2.COLOR_GRAY2HSV` behavior.

use super::bits_mask;

// ---- helpers ----------------------------------------------------------------

/// Broadcasts a `u8` gray value to packed RGB (3 bytes: R=G=B=y).
#[inline(always)]
fn broadcast_u8_to_rgb(y: u8, out: &mut [u8], x: usize) {
  let i = x * 3;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
}

/// Broadcasts a `u8` gray value to packed RGBA (4 bytes: R=G=B=y, A=0xFF).
#[inline(always)]
fn broadcast_u8_to_rgba(y: u8, out: &mut [u8], x: usize) {
  let i = x * 4;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
  out[i + 3] = 0xFF;
}

/// Broadcasts a `u16` gray value to packed u16 RGB (3 u16: R=G=B=y).
#[inline(always)]
fn broadcast_u16_to_rgb(y: u16, out: &mut [u16], x: usize) {
  let i = x * 3;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
}

/// Broadcasts a `u16` gray value to packed u16 RGBA (4 u16: R=G=B=y, A=alpha).
#[inline(always)]
fn broadcast_u16_to_rgba(y: u16, alpha: u16, out: &mut [u16], x: usize) {
  let i = x * 4;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
  out[i + 3] = alpha;
}

// ---- Gray8 ------------------------------------------------------------------

/// Broadcasts each `u8` gray sample to packed RGB (`R = G = B = Y`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_rgb_row(y_plane: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &y) in y_plane[..width].iter().enumerate() {
    broadcast_u8_to_rgb(y, out, x);
  }
}

/// Broadcasts each `u8` gray sample to packed RGBA (`R = G = B = Y`, `A = 0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_rgba_row(y_plane: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &y) in y_plane[..width].iter().enumerate() {
    broadcast_u8_to_rgba(y, out, x);
  }
}

/// Gray8 → HSV row. Convention: S=0, H=0, V=Y.
///
/// Gray sources are achromatic (saturation = 0). When S=0, H is
/// undefined in the continuous HSV model; this crate fixes H=0 to
/// match OpenCV's `cv2.COLOR_GRAY2HSV` convention and avoid
/// non-deterministic hue output.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_hsv_row(
  y_plane: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  for (x, &y) in y_plane[..width].iter().enumerate() {
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = y;
  }
}

// ---- GrayN (u16 low-bit-packed, BITS in {9,10,12,14}) ----------------------

/// GrayN → packed RGB u8. Masks to BITS bits, downshifts `BITS - 8` to u8,
/// broadcasts.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgb_row<const BITS: u32>(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let y8 = ((raw & mask) >> shift) as u8;
    broadcast_u8_to_rgb(y8, out, x);
  }
}

/// GrayN → packed RGBA u8. Masks to BITS bits, downshifts to u8, broadcasts,
/// α = 0xFF.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgba_row<const BITS: u32>(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let y8 = ((raw & mask) >> shift) as u8;
    broadcast_u8_to_rgba(y8, out, x);
  }
}

/// GrayN → packed u16 RGB. Masks to BITS bits, broadcasts at native depth.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgb_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  let mask = bits_mask::<BITS>();
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    broadcast_u16_to_rgb(raw & mask, out, x);
  }
}

/// GrayN → packed u16 RGBA. Masks to BITS bits, broadcasts, α = `(1 << BITS) - 1`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgba_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  let mask = bits_mask::<BITS>();
  let alpha = mask; // full-range max for BITS
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    broadcast_u16_to_rgba(raw & mask, alpha, out, x);
  }
}

/// GrayN → luma u8. Masks to BITS bits, downshifts `BITS - 8`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_luma_row<const BITS: u32>(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (out_byte, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    *out_byte = ((raw & mask) >> shift) as u8;
  }
}

/// GrayN → luma u16. Masks to BITS bits, identity copy.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_luma_u16_row<const BITS: u32>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  let mask = bits_mask::<BITS>();
  for (out_el, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    *out_el = raw & mask;
  }
}

/// GrayN → HSV u8. Masks to BITS bits, downshifts to u8, H=0 S=0 V=Y8.
///
/// See [`gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_hsv_row<const BITS: u32>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = ((raw & mask) >> shift) as u8;
  }
}

// ---- Gray16 (u16, all 16 bits active) ----------------------------------------

/// Gray16 → packed RGB u8. Downshifts `>> 8` to u8, broadcasts.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgb_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    broadcast_u8_to_rgb((raw >> 8) as u8, out, x);
  }
}

/// Gray16 → packed RGBA u8. Downshifts `>> 8`, broadcasts, α = 0xFF.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgba_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    broadcast_u8_to_rgba((raw >> 8) as u8, out, x);
  }
}

/// Gray16 → packed u16 RGB. Identity broadcast, native 16-bit depth.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgb_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    broadcast_u16_to_rgb(raw, out, x);
  }
}

/// Gray16 → packed u16 RGBA. Identity broadcast, α = 0xFFFF.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgba_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    broadcast_u16_to_rgba(raw, 0xFFFF, out, x);
  }
}

/// Gray16 → luma u8. Downshifts `>> 8`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_luma_row(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  for (out_byte, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    *out_byte = (raw >> 8) as u8;
  }
}

/// Gray16 → luma u16. Identity copy.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_luma_u16_row(y_plane: &[u16], out: &mut [u16], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  out[..width].copy_from_slice(&y_plane[..width]);
}

/// Gray16 → HSV u8. `>> 8` to u8, H=0 S=0 V=Y8.
///
/// See [`gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_hsv_row(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = (raw >> 8) as u8;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  #[test]
  fn gray8_to_rgb_broadcasts() {
    let y = [0u8, 128, 255];
    let mut out = [0u8; 9];
    gray8_to_rgb_row(&y, &mut out, 3);
    assert_eq!(&out[0..3], &[0, 0, 0]);
    assert_eq!(&out[3..6], &[128, 128, 128]);
    assert_eq!(&out[6..9], &[255, 255, 255]);
  }

  #[test]
  fn gray8_to_rgba_broadcasts_opaque() {
    let y = [100u8, 200];
    let mut out = [0u8; 8];
    gray8_to_rgba_row(&y, &mut out, 2);
    assert_eq!(&out[0..4], &[100, 100, 100, 0xFF]);
    assert_eq!(&out[4..8], &[200, 200, 200, 0xFF]);
  }

  #[test]
  fn gray8_to_hsv_h0_s0_v_y() {
    let y = [0u8, 128, 255];
    let mut h = [0xFFu8; 3];
    let mut s = [0xFFu8; 3];
    let mut v = [0u8; 3];
    gray8_to_hsv_row(&y, &mut h, &mut s, &mut v, 3);
    assert_eq!(h, [0, 0, 0]);
    assert_eq!(s, [0, 0, 0]);
    assert_eq!(v, [0, 128, 255]);
  }

  #[test]
  fn gray_n_to_rgb_10bit_downshifts() {
    // 10-bit: 1023 >> 2 = 255; 0 >> 2 = 0; 512 >> 2 = 128
    let y: Vec<u16> = std::vec![0, 512, 1023];
    let mut out = std::vec![0u8; 9];
    gray_n_to_rgb_row::<10>(&y, &mut out, 3);
    assert_eq!(&out[0..3], &[0, 0, 0]);
    assert_eq!(&out[3..6], &[128, 128, 128]);
    assert_eq!(&out[6..9], &[255, 255, 255]);
  }

  #[test]
  fn gray_n_to_rgb_u16_10bit_masks() {
    // Upper bits should be masked out: 0xFFFF & 0x03FF = 0x03FF = 1023
    let y: Vec<u16> = std::vec![0xFFFF, 512, 0];
    let mut out = std::vec![0u16; 9];
    gray_n_to_rgb_u16_row::<10>(&y, &mut out, 3);
    assert_eq!(&out[0..3], &[1023, 1023, 1023]);
    assert_eq!(&out[3..6], &[512, 512, 512]);
    assert_eq!(&out[6..9], &[0, 0, 0]);
  }

  #[test]
  fn gray_n_to_hsv_h0_s0() {
    let y: Vec<u16> = std::vec![512u16]; // 512 >> 2 = 128
    let mut h = std::vec![0xFFu8; 1];
    let mut s = std::vec![0xFFu8; 1];
    let mut v = std::vec![0u8; 1];
    gray_n_to_hsv_row::<10>(&y, &mut h, &mut s, &mut v, 1);
    assert_eq!(h[0], 0);
    assert_eq!(s[0], 0);
    assert_eq!(v[0], 128);
  }

  #[test]
  fn gray16_to_rgb_downshifts_8() {
    let y: Vec<u16> = std::vec![0, 0x8000, 0xFFFF];
    let mut out = std::vec![0u8; 9];
    gray16_to_rgb_row(&y, &mut out, 3);
    assert_eq!(&out[0..3], &[0, 0, 0]);
    assert_eq!(&out[3..6], &[0x80, 0x80, 0x80]);
    assert_eq!(&out[6..9], &[0xFF, 0xFF, 0xFF]);
  }

  #[test]
  fn gray16_to_luma_u16_identity() {
    let y: Vec<u16> = std::vec![0, 1000, 65535];
    let mut out = std::vec![0u16; 3];
    gray16_to_luma_u16_row(&y, &mut out, 3);
    assert_eq!(out.as_slice(), &[0, 1000, 65535]);
  }

  #[test]
  fn gray16_to_rgba_u16_opaque() {
    let y: Vec<u16> = std::vec![12345u16];
    let mut out = std::vec![0u16; 4];
    gray16_to_rgba_u16_row(&y, &mut out, 1);
    assert_eq!(&out[0..4], &[12345, 12345, 12345, 0xFFFF]);
  }

  #[test]
  fn gray_n_to_luma_u16_10bit_masks() {
    let y: Vec<u16> = std::vec![0xFFFF]; // should mask to 1023
    let mut out = std::vec![0u16; 1];
    gray_n_to_luma_u16_row::<10>(&y, &mut out, 1);
    assert_eq!(out[0], 1023);
  }
}
