//! NEON parity tests for legacy 16-bit packed-RGB kernels (Tier 7).
//!
//! Each test runs the NEON kernel and the scalar reference at widths
//! [1, 7, 8, 15, 16, 17, 32, 33, 64, 65] and asserts byte-identical output.
//! The pseudo-random plane generator produces LE u16 pixel words; mask limits
//! pixel values to the valid bit range for each format family.

use super::*;
use crate::row::arch::neon::legacy_rgb::*;

/// Pseudo-random LE u16 pixel plane for legacy RGB formats.
///
/// Each pixel word is `(LCG state & mask) as u16` in little-endian byte order.
/// `mask` should cover all bits of the format (e.g., `0xFFFF` for RGB565,
/// `0x7FFF` for RGB555, `0x0FFF` for RGB444).
fn legacy_rgb_plane(width: usize, seed: u32, mask: u16) -> std::vec::Vec<u8> {
  let mut state = seed;
  let mut out = std::vec::Vec::with_capacity(width * 2);
  for _ in 0..width {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let px = ((state & mask as u32) as u16).to_le_bytes();
    out.extend_from_slice(&px);
  }
  out
}

// RGB565.
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb565_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF, 0xFFFF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::legacy_rgb::rgb565_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      rgb565_to_rgb_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb565_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb565_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF, 0xFFFF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::legacy_rgb::rgb565_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      rgb565_to_rgba_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb565_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb565_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF, 0xFFFF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::legacy_rgb::rgb565_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb565_to_rgb_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb565_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb565_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF, 0xFFFF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::legacy_rgb::rgb565_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb565_to_rgba_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb565_to_rgba_u16 diverges (width={w})"
    );
  }
}

// BGR565.
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr565_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D, 0xFFFF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::legacy_rgb::bgr565_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      bgr565_to_rgb_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr565_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr565_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D, 0xFFFF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::legacy_rgb::bgr565_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      bgr565_to_rgba_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr565_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr565_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D, 0xFFFF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::legacy_rgb::bgr565_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr565_to_rgb_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr565_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr565_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D, 0xFFFF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::legacy_rgb::bgr565_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr565_to_rgba_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr565_to_rgba_u16 diverges (width={w})"
    );
  }
}

// RGB555.
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb555_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    // Mask 0x7FFF: bit 15 is padding (keep it clear for a clean signal,
    // though the kernel ignores it either way)
    let src = legacy_rgb_plane(w, 0x1234_5678, 0x7FFF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::legacy_rgb::rgb555_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      rgb555_to_rgb_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb555_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb555_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678, 0x7FFF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::legacy_rgb::rgb555_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      rgb555_to_rgba_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb555_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb555_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678, 0x7FFF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::legacy_rgb::rgb555_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb555_to_rgb_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb555_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb555_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678, 0x7FFF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::legacy_rgb::rgb555_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb555_to_rgba_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb555_to_rgba_u16 diverges (width={w})"
    );
  }
}

// BGR555.
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr555_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01, 0x7FFF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::legacy_rgb::bgr555_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      bgr555_to_rgb_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr555_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr555_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01, 0x7FFF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::legacy_rgb::bgr555_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      bgr555_to_rgba_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr555_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr555_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01, 0x7FFF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::legacy_rgb::bgr555_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr555_to_rgb_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr555_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr555_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01, 0x7FFF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::legacy_rgb::bgr555_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr555_to_rgba_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr555_to_rgba_u16 diverges (width={w})"
    );
  }
}

// RGB444.
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb444_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    // Mask 0x0FFF: bits [15:12] are padding
    let src = legacy_rgb_plane(w, 0xFEED_BABE, 0x0FFF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::legacy_rgb::rgb444_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      rgb444_to_rgb_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb444_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb444_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xFEED_BABE, 0x0FFF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::legacy_rgb::rgb444_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      rgb444_to_rgba_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb444_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb444_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xFEED_BABE, 0x0FFF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::legacy_rgb::rgb444_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb444_to_rgb_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb444_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgb444_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xFEED_BABE, 0x0FFF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::legacy_rgb::rgb444_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      rgb444_to_rgba_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON rgb444_to_rgba_u16 diverges (width={w})"
    );
  }
}

// BGR444.
#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr444_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x0BAD_C0DE, 0x0FFF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::legacy_rgb::bgr444_to_rgb_row(&src, &mut out_scalar, w);
    unsafe {
      bgr444_to_rgb_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr444_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr444_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x0BAD_C0DE, 0x0FFF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::legacy_rgb::bgr444_to_rgba_row(&src, &mut out_scalar, w);
    unsafe {
      bgr444_to_rgba_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr444_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr444_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x0BAD_C0DE, 0x0FFF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::legacy_rgb::bgr444_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr444_to_rgb_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr444_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_bgr444_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x0BAD_C0DE, 0x0FFF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::legacy_rgb::bgr444_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe {
      bgr444_to_rgba_u16_row(&src, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON bgr444_to_rgba_u16 diverges (width={w})"
    );
  }
}
