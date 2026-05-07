//! Parity tests: NEON pal8 kernels vs. scalar reference.
//!
//! Each test exercises boundary widths to catch main-loop vs. tail
//! split edge cases: 1, 8, 15, 16, 17, 32, 33, 128, 130.

use super::super::pal8::{
  pal8_to_rgb_row, pal8_to_rgb_u16_row, pal8_to_rgba_row, pal8_to_rgba_u16_row,
};
use crate::row::scalar::pal8 as scalar_pal8;

// ---- Test helpers -----------------------------------------------------------

/// A deterministic 256-entry palette where each entry is a pseudo-random
/// BGRA quad. Seeded from index so all 256 entries are distinct and
/// representative of real palettes.
fn make_test_palette() -> [[u8; 4]; 256] {
  let mut p = [[0u8; 4]; 256];
  for (i, entry) in p.iter_mut().enumerate() {
    let i = i as u32;
    // Simple LCG-derived values — different for B, G, R, A channels.
    entry[0] = ((i.wrapping_mul(73) ^ 0xA5) & 0xFF) as u8; // B
    entry[1] = ((i.wrapping_mul(131) ^ 0x5A) & 0xFF) as u8; // G
    entry[2] = ((i.wrapping_mul(197) ^ 0x3C) & 0xFF) as u8; // R
    entry[3] = ((i.wrapping_mul(251) ^ 0xF0) & 0xFF) as u8; // A
  }
  p
}

/// Generates `width` pseudo-random index values in [0, 255].
fn make_indices(width: usize) -> std::vec::Vec<u8> {
  let mut out = std::vec::Vec::with_capacity(width);
  let mut state: u32 = 0xDEAD_BEEF;
  for _ in 0..width {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    out.push((state >> 24) as u8);
  }
  out
}

// ---- pal8_to_rgb_row --------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgb_neon_matches_scalar_boundary_widths() {
  let palette = make_test_palette();
  for &w in &[1usize, 8, 15, 16, 17, 32, 33, 128, 130] {
    let indices = make_indices(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar_pal8::pal8_to_rgb_row(&indices, &palette, &mut out_scalar);
    unsafe { pal8_to_rgb_row(&indices, &palette, &mut out_neon) };
    assert_eq!(out_scalar, out_neon, "pal8_to_rgb width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgb_neon_bgra_to_rgb_reorder() {
  // Verify that BGRA→RGB reorder is correct: palette[0] = [B=10,G=20,R=30,A=40]
  // must produce out = [30, 20, 10] (R, G, B).
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [10, 20, 30, 40]; // [B, G, R, A]
  palette[1] = [50, 100, 200, 255];
  let indices = [0u8, 1u8];
  let mut out = [0u8; 6];
  unsafe { pal8_to_rgb_row(&indices, &palette, &mut out) };
  assert_eq!(out[0..3], [30, 20, 10], "entry 0 RGB");
  assert_eq!(out[3..6], [200, 100, 50], "entry 1 RGB");
}

// ---- pal8_to_rgba_row -------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgba_neon_matches_scalar_boundary_widths() {
  let palette = make_test_palette();
  for &w in &[1usize, 8, 15, 16, 17, 32, 33, 128, 130] {
    let indices = make_indices(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar_pal8::pal8_to_rgba_row(&indices, &palette, &mut out_scalar);
    unsafe { pal8_to_rgba_row(&indices, &palette, &mut out_neon) };
    assert_eq!(out_scalar, out_neon, "pal8_to_rgba width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgba_neon_preserves_alpha() {
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [10, 20, 30, 40]; // [B, G, R, A=40]
  let indices = [0u8];
  let mut out = [0u8; 4];
  unsafe { pal8_to_rgba_row(&indices, &palette, &mut out) };
  assert_eq!(out, [30, 20, 10, 40], "RGBA output including alpha");
}

// ---- pal8_to_rgb_u16_row ----------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgb_u16_neon_matches_scalar_boundary_widths() {
  let palette = make_test_palette();
  for &w in &[1usize, 8, 15, 16, 17, 32, 33, 128, 130] {
    let indices = make_indices(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar_pal8::pal8_to_rgb_u16_row(&indices, &palette, &mut out_scalar);
    unsafe { pal8_to_rgb_u16_row(&indices, &palette, &mut out_neon) };
    assert_eq!(out_scalar, out_neon, "pal8_to_rgb_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgb_u16_neon_full_range_expansion() {
  // R=255 → 0xFFFF, G=0 → 0x0000, B=0 → 0x0000
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [0, 0, 255, 255]; // [B=0, G=0, R=255, A=255]
  let indices = [0u8];
  let mut out = [0u16; 3];
  unsafe { pal8_to_rgb_u16_row(&indices, &palette, &mut out) };
  assert_eq!(out[0], 0xFFFF, "R=255 → 0xFFFF");
  assert_eq!(out[1], 0x0000, "G=0   → 0x0000");
  assert_eq!(out[2], 0x0000, "B=0   → 0x0000");
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgb_u16_neon_midpoint_expansion() {
  // v=128 → 0x8080
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [128, 128, 128, 128];
  let indices = [0u8];
  let mut out = [0u16; 3];
  unsafe { pal8_to_rgb_u16_row(&indices, &palette, &mut out) };
  assert_eq!(out[0], 0x8080, "R=128 → 0x8080");
  assert_eq!(out[1], 0x8080, "G=128 → 0x8080");
  assert_eq!(out[2], 0x8080, "B=128 → 0x8080");
}

// ---- pal8_to_rgba_u16_row ---------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgba_u16_neon_matches_scalar_boundary_widths() {
  let palette = make_test_palette();
  for &w in &[1usize, 8, 15, 16, 17, 32, 33, 128, 130] {
    let indices = make_indices(w);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar_pal8::pal8_to_rgba_u16_row(&indices, &palette, &mut out_scalar);
    unsafe { pal8_to_rgba_u16_row(&indices, &palette, &mut out_neon) };
    assert_eq!(out_scalar, out_neon, "pal8_to_rgba_u16 width={w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgba_u16_neon_alpha_expansion() {
  let mut palette = [[0u8; 4]; 256];
  palette[0] = [0, 0, 0, 128]; // A=128 → 0x8080
  let indices = [0u8];
  let mut out = [0u16; 4];
  unsafe { pal8_to_rgba_u16_row(&indices, &palette, &mut out) };
  assert_eq!(out[3], 0x8080, "A=128 → 0x8080");
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_to_rgba_u16_neon_all_channels_boundary() {
  // Verify min/max expansion across all 4 channels.
  let mut palette = [[0u8; 4]; 256];
  palette[5] = [0, 255, 0, 255]; // B=0, G=255, R=0, A=255
  let indices = [5u8];
  let mut out = [0u16; 4];
  unsafe { pal8_to_rgba_u16_row(&indices, &palette, &mut out) };
  assert_eq!(out[0], 0x0000, "R=0   → 0x0000");
  assert_eq!(out[1], 0xFFFF, "G=255 → 0xFFFF");
  assert_eq!(out[2], 0x0000, "B=0   → 0x0000");
  assert_eq!(out[3], 0xFFFF, "A=255 → 0xFFFF");
}

// ---- 16-pixel-exact boundary (main loop only, no tail) ----------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_neon_exact_16px_no_tail() {
  let palette = make_test_palette();
  let indices = make_indices(16);
  // RGB
  let mut s = std::vec![0u8; 48];
  let mut n = std::vec![0u8; 48];
  scalar_pal8::pal8_to_rgb_row(&indices, &palette, &mut s);
  unsafe { pal8_to_rgb_row(&indices, &palette, &mut n) };
  assert_eq!(s, n, "exact 16px RGB");
  // RGBA
  let mut s = std::vec![0u8; 64];
  let mut n = std::vec![0u8; 64];
  scalar_pal8::pal8_to_rgba_row(&indices, &palette, &mut s);
  unsafe { pal8_to_rgba_row(&indices, &palette, &mut n) };
  assert_eq!(s, n, "exact 16px RGBA");
}

// ---- 17-pixel (main loop + 1 tail pixel) ------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn pal8_neon_17px_main_plus_tail() {
  let palette = make_test_palette();
  let indices = make_indices(17);
  let mut s = std::vec![0u8; 17 * 3];
  let mut n = std::vec![0u8; 17 * 3];
  scalar_pal8::pal8_to_rgb_row(&indices, &palette, &mut s);
  unsafe { pal8_to_rgb_row(&indices, &palette, &mut n) };
  assert_eq!(s, n, "17px RGB (16 main + 1 tail)");
}
