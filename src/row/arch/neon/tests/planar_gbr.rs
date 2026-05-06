//! NEON parity tests for the planar-GBR kernels (Tier 10).
//!
//! Each test seeds three (or four) planar rows with deterministic
//! pseudo-random bytes, runs both the scalar reference and the NEON
//! kernel, and asserts byte-identical output. Widths span the
//! per-iteration SIMD step (16) and a mix of small / boundary /
//! large widths.

use super::*;

fn pseudo_random_plane(width: usize, seed: u32) -> std::vec::Vec<u8> {
  let mut state = seed;
  (0..width)
    .map(|_| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (state >> 8) as u8
    })
    .collect()
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn gbr_to_rgb_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let g = pseudo_random_plane(w, 0x6CCD_5C7B);
    let b = pseudo_random_plane(w, 0x12AB_34CD);
    let r = pseudo_random_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_row(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_row(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Direct semantic check: every output triple = R, G, B per pixel.
    for x in 0..w {
      assert_eq!(out_neon[x * 3], r[x], "R width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 1], g[x], "G width {w} px {x}");
      assert_eq!(out_neon[x * 3 + 2], b[x], "B width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn gbra_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let g = pseudo_random_plane(w, 0x6CCD_5C7B);
    let b = pseudo_random_plane(w, 0x12AB_34CD);
    let r = pseudo_random_plane(w, 0xDEAD_BEEF);
    let a = pseudo_random_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_row(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_row(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    for x in 0..w {
      assert_eq!(out_neon[x * 4], r[x], "R width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 1], g[x], "G width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 2], b[x], "B width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 3], a[x], "A width {w} px {x}");
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn gbr_to_rgba_opaque_neon_matches_scalar_widths() {
  for w in [1usize, 7, 15, 16, 17, 31, 32, 33, 1920, 1921] {
    let g = pseudo_random_plane(w, 0x6CCD_5C7B);
    let b = pseudo_random_plane(w, 0x12AB_34CD);
    let r = pseudo_random_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_row(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_row(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    for x in 0..w {
      assert_eq!(out_neon[x * 4], r[x], "R width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 1], g[x], "G width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 2], b[x], "B width {w} px {x}");
      assert_eq!(out_neon[x * 4 + 3], 0xFF, "A width {w} px {x}");
    }
  }
}
