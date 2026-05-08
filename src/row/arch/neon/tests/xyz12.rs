//! NEON parity tests for the Tier 12 (DCP / Xyz12) kernels.
//!
//! Every test asserts SIMD output is **byte-identical** to the scalar
//! reference. The 0-ULP parity contract relies on:
//!
//! 1. The per-lane scalar `smpte428_inverse_oetf` and `oetf_srgb`
//!    being *the same scalar function* both paths call, and
//! 2. The vectorized matmul + clamp + narrow being agnostic to lane
//!    arrangement (per-lane f32 ops, not horizontal reductions).
//!
//! Test matrix (per output × per gamut):
//! - widths: `[1, 4, 7, 16, 33, 1920]` (mix of small / SIMD-aligned /
//!   non-aligned / row-realistic).
//! - gamuts: `DciP3`, `Rec709`, `Rec2020` — at least one per output.
//! - BE: `false` and `true` parity test for the rgb-u8 path.

use super::*;
use crate::{
  DcpTargetGamut,
  row::arch::neon::xyz12::{
    xyz12_to_rgb_f16_row, xyz12_to_rgb_f32_row, xyz12_to_rgb_row, xyz12_to_rgb_u16_row,
    xyz12_to_rgba_f16_row, xyz12_to_rgba_row, xyz12_to_rgba_u16_row, xyz12_to_xyz_f32_row,
  },
};

const WIDTHS: &[usize] = &[1, 4, 7, 16, 33, 1920];

fn xyz12_plane(width: usize, seed: u32) -> std::vec::Vec<u16> {
  let mut state = seed;
  (0..width * 3)
    .map(|_| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (state & 0x0FFF) as u16
    })
    .collect()
}

/// Mirrors the dirty-input pattern from the planar-gbr tests: the
/// upper bits should be ignored by the kernel's `SAMPLE_MASK`, so a
/// value with bit 13 / 15 set must produce the same output as its
/// `& 0x0FFF` clean counterpart.
fn xyz12_plane_dirty(width: usize, seed: u32) -> std::vec::Vec<u16> {
  let mut state = seed;
  (0..width * 3)
    .map(|i| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      let clean = (state & 0x0FFF) as u16;
      let dirty = if i % 2 == 0 { 0xF000 } else { 0xA000 };
      clean | dirty
    })
    .collect()
}

fn byte_swap_vec(v: &[u16]) -> std::vec::Vec<u16> {
  v.iter().map(|x| x.swap_bytes()).collect()
}

// ---- u8 RGB --------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_matches_scalar() {
  for &w in WIDTHS {
    for gamut in [
      DcpTargetGamut::DciP3,
      DcpTargetGamut::Rec709,
      DcpTargetGamut::Rec2020,
    ] {
      let xyz = xyz12_plane(w, 0xC0FE_BABE);
      let mut out_scalar = std::vec![0u8; w * 3];
      let mut out_neon = std::vec![0u8; w * 3];
      scalar::xyz12::xyz12_to_rgb_row::<false>(&xyz, &mut out_scalar, w, gamut);
      unsafe {
        xyz12_to_rgb_row::<false>(&xyz, &mut out_neon, w, gamut);
      }
      assert_eq!(
        out_scalar, out_neon,
        "NEON xyz12_to_rgb diverges (w={w}, gamut={gamut:?})"
      );
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_dirty_input_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane_dirty(w, 0xDEAD_F00D);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::xyz12::xyz12_to_rgb_row::<false>(&xyz, &mut out_scalar, w, DcpTargetGamut::DciP3);
    unsafe {
      xyz12_to_rgb_row::<false>(&xyz, &mut out_neon, w, DcpTargetGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgb dirty-input diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_be_matches_le() {
  for &w in WIDTHS {
    let xyz_le = xyz12_plane(w, 0x1234_5678);
    let xyz_be = byte_swap_vec(&xyz_le);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      xyz12_to_rgb_row::<false>(&xyz_le, &mut out_le, w, DcpTargetGamut::Rec709);
      xyz12_to_rgb_row::<true>(&xyz_be, &mut out_be, w, DcpTargetGamut::Rec709);
    }
    assert_eq!(out_le, out_be, "NEON xyz12_to_rgb BE/LE mismatch (w={w})");
  }
}

// ---- u8 RGBA -------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgba_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xAFAF_AFAF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::xyz12::xyz12_to_rgba_row::<false>(&xyz, &mut out_scalar, w, DcpTargetGamut::Rec2020);
    unsafe {
      xyz12_to_rgba_row::<false>(&xyz, &mut out_neon, w, DcpTargetGamut::Rec2020);
    }
    assert_eq!(out_scalar, out_neon, "NEON xyz12_to_rgba diverges (w={w})");
  }
}

// ---- u16 RGB / RGBA -----------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_u16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xFEED_FACE);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::xyz12::xyz12_to_rgb_u16_row::<false>(&xyz, &mut out_scalar, w, DcpTargetGamut::DciP3);
    unsafe {
      xyz12_to_rgb_u16_row::<false>(&xyz, &mut out_neon, w, DcpTargetGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgb_u16 diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgba_u16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::xyz12::xyz12_to_rgba_u16_row::<false>(&xyz, &mut out_scalar, w, DcpTargetGamut::Rec709);
    unsafe {
      xyz12_to_rgba_u16_row::<false>(&xyz, &mut out_neon, w, DcpTargetGamut::Rec709);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgba_u16 diverges (w={w})"
    );
  }
}

// ---- f32 outputs --------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_f32_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0x600D_C0DE);
    let mut out_scalar = std::vec![0.0_f32; w * 3];
    let mut out_neon = std::vec![0.0_f32; w * 3];
    scalar::xyz12::xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_scalar, w, DcpTargetGamut::Rec2020);
    unsafe {
      xyz12_to_rgb_f32_row::<false>(&xyz, &mut out_neon, w, DcpTargetGamut::Rec2020);
    }
    // 0-ULP parity contract: SIMD matmul uses plain mul+add (not FMA),
    // matching the scalar's exact rounding schedule, so f32 output
    // must be bit-exact identical lane-by-lane.
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgb_f32 diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_xyz_f32_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xD00D_F00D);
    let mut out_scalar = std::vec![0.0_f32; w * 3];
    let mut out_neon = std::vec![0.0_f32; w * 3];
    scalar::xyz12::xyz12_to_xyz_f32_row::<false>(&xyz, &mut out_scalar, w);
    unsafe {
      xyz12_to_xyz_f32_row::<false>(&xyz, &mut out_neon, w);
    }
    // No matmul on this path — every f32 lane comes from the same
    // scalar `smpte428_inverse_oetf`, so equality is bit-exact.
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_xyz_f32 diverges (w={w})"
    );
  }
}

// ---- f16 outputs --------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgb_f16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xBEEF_CAFE);
    let zero_f16 = half::f16::from_f32(0.0);
    let mut out_scalar = std::vec![zero_f16; w * 3];
    let mut out_neon = std::vec![zero_f16; w * 3];
    scalar::xyz12::xyz12_to_rgb_f16_row::<false>(&xyz, &mut out_scalar, w, DcpTargetGamut::DciP3);
    unsafe {
      xyz12_to_rgb_f16_row::<false>(&xyz, &mut out_neon, w, DcpTargetGamut::DciP3);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgb_f16 diverges (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_xyz12_to_rgba_f16_matches_scalar() {
  for &w in WIDTHS {
    let xyz = xyz12_plane(w, 0xF1F1_F1F1);
    let zero_f16 = half::f16::from_f32(0.0);
    let mut out_scalar = std::vec![zero_f16; w * 4];
    let mut out_neon = std::vec![zero_f16; w * 4];
    scalar::xyz12::xyz12_to_rgba_f16_row::<false>(&xyz, &mut out_scalar, w, DcpTargetGamut::Rec709);
    unsafe {
      xyz12_to_rgba_f16_row::<false>(&xyz, &mut out_neon, w, DcpTargetGamut::Rec709);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON xyz12_to_rgba_f16 diverges (w={w})"
    );
  }
}
