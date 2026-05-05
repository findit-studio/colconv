//! Shared scalar `y_plane_to_luma_u16_row` — extracts Y as native-depth
//! u16 for any 8-bit source whose Y is a contiguous u8 plane (planar
//! Yuv*p, semi-planar Nv*).
//!
//! Output is `plane[x] as u16` — pure zero-extension, no shift.
//! Used by the 8-bit `with_luma_u16` accessor wired across 9 sinkers.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y_plane_to_luma_u16_row(plane: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(out.len() >= width, "out too short");
  for x in 0..width {
    out[x] = plane[x] as u16;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  #[test]
  fn y_plane_to_luma_u16_zero_extends() {
    let plane = [0u8, 1, 127, 128, 255];
    let mut out = std::vec![0u16; 5];
    y_plane_to_luma_u16_row(&plane, &mut out, 5);
    assert_eq!(out, std::vec![0, 1, 127, 128, 255]);
  }

  #[test]
  fn y_plane_to_luma_u16_only_writes_first_width_elements() {
    let plane = std::vec![0xABu8; 8];
    let mut out = std::vec![0xFFFFu16; 8];
    y_plane_to_luma_u16_row(&plane, &mut out, 4);
    assert_eq!(&out[..4], &[0xAB; 4]);
    assert_eq!(&out[4..], &[0xFFFF; 4]); // tail untouched
  }
}
