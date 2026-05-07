//! wasm-simd128 parity tests for legacy 16-bit packed-RGB kernels (Tier 7).
//!
//! These tests compile and run only when targeting `wasm32` with
//! `target-feature=+simd128`. The entire `wasm_simd128` backend is
//! `#[cfg(target_arch = "wasm32")]` — on native aarch64 / x86_64
//! hosts, this module is `cfg`-out and these tests are not built.
//! Run via `cargo test --target wasm32-unknown-unknown` with a
//! wasm-bindgen-test runtime, or via the project's `test-wasm-simd128`
//! CI job (which uses wasmtime).
//!
//! All tests carry `#[cfg_attr(miri, ignore = "...")]`.
//! Tests use `Vec` / `assert_eq!` — not index-looping patterns.
//!
//! Widths exercised: [1, 7, 8, 15, 16, 17, 32, 33, 64, 65] — covers all
//! boundary cases around the 8-pixel loop stride.

use super::super::legacy_rgb::*;
use crate::row::scalar;

// ---- Shared pseudo-random helper -------------------------------------------

fn legacy_rgb_plane(width: usize, seed: u32) -> std::vec::Vec<u8> {
  let mut state = seed;
  let mut out = std::vec::Vec::with_capacity(width * 2);
  for _ in 0..width {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let px = (state as u16).to_le_bytes();
    out.extend_from_slice(&px);
  }
  out
}

// ============================================================================
// RGB565
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb565_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::legacy_rgb::rgb565_to_rgb_row(&src, &mut out_scalar, w);
    unsafe { rgb565_to_rgb_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb565_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb565_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::legacy_rgb::rgb565_to_rgba_row(&src, &mut out_scalar, w);
    unsafe { rgb565_to_rgba_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb565_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb565_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::legacy_rgb::rgb565_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe { rgb565_to_rgb_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb565_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb565_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::legacy_rgb::rgb565_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe { rgb565_to_rgba_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb565_to_rgba_u16 diverges (width={w})"
    );
  }
}

// ============================================================================
// BGR565
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr565_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::legacy_rgb::bgr565_to_rgb_row(&src, &mut out_scalar, w);
    unsafe { bgr565_to_rgb_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr565_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr565_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::legacy_rgb::bgr565_to_rgba_row(&src, &mut out_scalar, w);
    unsafe { bgr565_to_rgba_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr565_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr565_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::legacy_rgb::bgr565_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe { bgr565_to_rgb_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr565_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr565_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::legacy_rgb::bgr565_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe { bgr565_to_rgba_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr565_to_rgba_u16 diverges (width={w})"
    );
  }
}

// ============================================================================
// RGB555
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb555_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::legacy_rgb::rgb555_to_rgb_row(&src, &mut out_scalar, w);
    unsafe { rgb555_to_rgb_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb555_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb555_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::legacy_rgb::rgb555_to_rgba_row(&src, &mut out_scalar, w);
    unsafe { rgb555_to_rgba_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb555_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb555_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::legacy_rgb::rgb555_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe { rgb555_to_rgb_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb555_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb555_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::legacy_rgb::rgb555_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe { rgb555_to_rgba_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb555_to_rgba_u16 diverges (width={w})"
    );
  }
}

// ============================================================================
// BGR555
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr555_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::legacy_rgb::bgr555_to_rgb_row(&src, &mut out_scalar, w);
    unsafe { bgr555_to_rgb_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr555_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr555_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::legacy_rgb::bgr555_to_rgba_row(&src, &mut out_scalar, w);
    unsafe { bgr555_to_rgba_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr555_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr555_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::legacy_rgb::bgr555_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe { bgr555_to_rgb_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr555_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr555_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::legacy_rgb::bgr555_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe { bgr555_to_rgba_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr555_to_rgba_u16 diverges (width={w})"
    );
  }
}

// ============================================================================
// RGB444
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb444_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::legacy_rgb::rgb444_to_rgb_row(&src, &mut out_scalar, w);
    unsafe { rgb444_to_rgb_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb444_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb444_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::legacy_rgb::rgb444_to_rgba_row(&src, &mut out_scalar, w);
    unsafe { rgb444_to_rgba_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb444_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb444_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::legacy_rgb::rgb444_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe { rgb444_to_rgb_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb444_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_rgb444_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::legacy_rgb::rgb444_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe { rgb444_to_rgba_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm rgb444_to_rgba_u16 diverges (width={w})"
    );
  }
}

// ============================================================================
// BGR444
// ============================================================================

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr444_to_rgb_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_wasm = std::vec![0u8; w * 3];
    scalar::legacy_rgb::bgr444_to_rgb_row(&src, &mut out_scalar, w);
    unsafe { bgr444_to_rgb_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr444_to_rgb diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr444_to_rgba_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_wasm = std::vec![0u8; w * 4];
    scalar::legacy_rgb::bgr444_to_rgba_row(&src, &mut out_scalar, w);
    unsafe { bgr444_to_rgba_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr444_to_rgba diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr444_to_rgb_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0x1234_5678);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_wasm = std::vec![0u16; w * 3];
    scalar::legacy_rgb::bgr444_to_rgb_u16_row(&src, &mut out_scalar, w);
    unsafe { bgr444_to_rgb_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr444_to_rgb_u16 diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "wasm SIMD intrinsics unsupported by Miri")]
fn wasm_bgr444_to_rgba_u16_matches_scalar() {
  for w in [1usize, 7, 8, 15, 16, 17, 32, 33, 64, 65] {
    let src = legacy_rgb_plane(w, 0xABCD_EF01);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_wasm = std::vec![0u16; w * 4];
    scalar::legacy_rgb::bgr444_to_rgba_u16_row(&src, &mut out_scalar, w);
    unsafe { bgr444_to_rgba_u16_row(&src, &mut out_wasm, w) };
    assert_eq!(
      out_scalar, out_wasm,
      "wasm bgr444_to_rgba_u16 diverges (width={w})"
    );
  }
}
