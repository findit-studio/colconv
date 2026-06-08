//! SIMD vs scalar equivalence tests for NEON packed 16-bit RGB kernels.
//!
//! Each test uses `width = 17` (2 SIMD iterations of 8 + 1 scalar tail pixel)
//! to exercise both the SIMD hot-loop and the scalar tail simultaneously.
//! All tests are gated on `target_arch = "aarch64"` and ignored under Miri.

use super::*;

// ---- helpers ----------------------------------------------------------------

/// Build a `width`-pixel Rgb48 / Bgr48 row with a pseudo-random pattern.
fn make_rgb48_src(width: usize, seed: u16) -> std::vec::Vec<u16> {
  (0..width * 3)
    .map(|i| (i as u16).wrapping_mul(seed).wrapping_add(0x1357))
    .collect()
}

/// Build a `width`-pixel Rgba64 / Bgra64 row with a pseudo-random pattern.
fn make_rgba64_src(width: usize, seed: u16) -> std::vec::Vec<u16> {
  (0..width * 4)
    .map(|i| (i as u16).wrapping_mul(seed).wrapping_add(0x2468))
    .collect()
}

// Rgb48.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgb48_to_rgb_matches_scalar_width17() {
  let src = make_rgb48_src(17, 0x0101);
  let mut simd_out = std::vec![0u8; 17 * 3];
  let mut scalar_out = std::vec![0u8; 17 * 3];
  unsafe { neon_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(simd_out, scalar_out, "rgb48→rgb: SIMD vs scalar mismatch");
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgb48_to_rgba_matches_scalar_width17() {
  let src = make_rgb48_src(17, 0x0303);
  let mut simd_out = std::vec![0u8; 17 * 4];
  let mut scalar_out = std::vec![0u8; 17 * 4];
  unsafe { neon_rgb48_to_rgba_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgb48_to_rgba_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(simd_out, scalar_out, "rgb48→rgba: SIMD vs scalar mismatch");
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgb48_to_rgb_u16_matches_scalar_width17() {
  let src = make_rgb48_src(17, 0x0505);
  let mut simd_out = std::vec![0u16; 17 * 3];
  let mut scalar_out = std::vec![0u16; 17 * 3];
  unsafe { neon_rgb48_to_rgb_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgb48_to_rgb_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb_u16: SIMD vs scalar mismatch"
  );
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgb48_to_rgba_u16_matches_scalar_width17() {
  let src = make_rgb48_src(17, 0x0707);
  let mut simd_out = std::vec![0u16; 17 * 4];
  let mut scalar_out = std::vec![0u16; 17 * 4];
  unsafe { neon_rgb48_to_rgba_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgb48_to_rgba_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgba_u16: SIMD vs scalar mismatch"
  );
}

// Bgr48.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgr48_to_rgb_matches_scalar_width17() {
  let src = make_rgb48_src(17, 0x1111);
  let mut simd_out = std::vec![0u8; 17 * 3];
  let mut scalar_out = std::vec![0u8; 17 * 3];
  unsafe { neon_bgr48_to_rgb_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgr48_to_rgb_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(simd_out, scalar_out, "bgr48→rgb: SIMD vs scalar mismatch");
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgr48_to_rgba_matches_scalar_width17() {
  let src = make_rgb48_src(17, 0x2222);
  let mut simd_out = std::vec![0u8; 17 * 4];
  let mut scalar_out = std::vec![0u8; 17 * 4];
  unsafe { neon_bgr48_to_rgba_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgr48_to_rgba_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(simd_out, scalar_out, "bgr48→rgba: SIMD vs scalar mismatch");
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgr48_to_rgb_u16_matches_scalar_width17() {
  let src = make_rgb48_src(17, 0x3333);
  let mut simd_out = std::vec![0u16; 17 * 3];
  let mut scalar_out = std::vec![0u16; 17 * 3];
  unsafe { neon_bgr48_to_rgb_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgr48_to_rgb_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgb_u16: SIMD vs scalar mismatch"
  );
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgr48_to_rgba_u16_matches_scalar_width17() {
  let src = make_rgb48_src(17, 0x4444);
  let mut simd_out = std::vec![0u16; 17 * 4];
  let mut scalar_out = std::vec![0u16; 17 * 4];
  unsafe { neon_bgr48_to_rgba_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgr48_to_rgba_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgr48→rgba_u16: SIMD vs scalar mismatch"
  );
}

// Rgba64.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgba64_to_rgb_matches_scalar_width17() {
  let src = make_rgba64_src(17, 0xAAAA);
  let mut simd_out = std::vec![0u8; 17 * 3];
  let mut scalar_out = std::vec![0u8; 17 * 3];
  unsafe { neon_rgba64_to_rgb_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgba64_to_rgb_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(simd_out, scalar_out, "rgba64→rgb: SIMD vs scalar mismatch");
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgba64_to_rgba_matches_scalar_width17() {
  let src = make_rgba64_src(17, 0xBBBB);
  let mut simd_out = std::vec![0u8; 17 * 4];
  let mut scalar_out = std::vec![0u8; 17 * 4];
  unsafe { neon_rgba64_to_rgba_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgba64_to_rgba_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(simd_out, scalar_out, "rgba64→rgba: SIMD vs scalar mismatch");
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgba64_to_rgb_u16_matches_scalar_width17() {
  let src = make_rgba64_src(17, 0xCCCC);
  let mut simd_out = std::vec![0u16; 17 * 3];
  let mut scalar_out = std::vec![0u16; 17 * 3];
  unsafe { neon_rgba64_to_rgb_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgba64_to_rgb_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgb_u16: SIMD vs scalar mismatch"
  );
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgba64_to_rgba_u16_matches_scalar_width17() {
  let src = make_rgba64_src(17, 0xDDDD);
  let mut simd_out = std::vec![0u16; 17 * 4];
  let mut scalar_out = std::vec![0u16; 17 * 4];
  unsafe { neon_rgba64_to_rgba_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::rgba64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba_u16: SIMD vs scalar mismatch"
  );
}

// Bgra64.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgra64_to_rgb_matches_scalar_width17() {
  let src = make_rgba64_src(17, 0x1234);
  let mut simd_out = std::vec![0u8; 17 * 3];
  let mut scalar_out = std::vec![0u8; 17 * 3];
  unsafe { neon_bgra64_to_rgb_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgra64_to_rgb_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(simd_out, scalar_out, "bgra64→rgb: SIMD vs scalar mismatch");
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgra64_to_rgba_matches_scalar_width17() {
  let src = make_rgba64_src(17, 0x5678);
  let mut simd_out = std::vec![0u8; 17 * 4];
  let mut scalar_out = std::vec![0u8; 17 * 4];
  unsafe { neon_bgra64_to_rgba_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgra64_to_rgba_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(simd_out, scalar_out, "bgra64→rgba: SIMD vs scalar mismatch");
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgra64_to_rgb_u16_matches_scalar_width17() {
  let src = make_rgba64_src(17, 0x9ABC);
  let mut simd_out = std::vec![0u16; 17 * 3];
  let mut scalar_out = std::vec![0u16; 17 * 3];
  unsafe { neon_bgra64_to_rgb_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgra64_to_rgb_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgb_u16: SIMD vs scalar mismatch"
  );
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgra64_to_rgba_u16_matches_scalar_width17() {
  let src = make_rgba64_src(17, 0xDEF0);
  let mut simd_out = std::vec![0u16; 17 * 4];
  let mut scalar_out = std::vec![0u16; 17 * 4];
  unsafe { neon_bgra64_to_rgba_u16_row::<false>(&src, &mut simd_out, 17) };
  scalar::bgra64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 17);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba_u16: SIMD vs scalar mismatch"
  );
}

// Exact-8 width: verify no-tail path works correctly.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgb48_to_rgb_exact8_matches_scalar() {
  let src = make_rgb48_src(8, 0xF0F0);
  let mut simd_out = std::vec![0u8; 8 * 3];
  let mut scalar_out = std::vec![0u8; 8 * 3];
  unsafe { neon_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 8) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 8);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb exact-8: SIMD vs scalar mismatch"
  );
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgba64_to_rgba_exact8_matches_scalar() {
  let src = make_rgba64_src(8, 0x0F0F);
  let mut simd_out = std::vec![0u8; 8 * 4];
  let mut scalar_out = std::vec![0u8; 8 * 4];
  unsafe { neon_rgba64_to_rgba_row::<false>(&src, &mut simd_out, 8) };
  scalar::rgba64_to_rgba_row::<false>(&src, &mut scalar_out, 8);
  assert_eq!(
    simd_out, scalar_out,
    "rgba64→rgba exact-8: SIMD vs scalar mismatch"
  );
}

// Tail-only width=1: no SIMD path, scalar only.
#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgb48_to_rgb_width1_scalar_tail_only() {
  let src = [0x1234u16, 0x5678, 0x9ABC];
  let mut simd_out = [0u8; 3];
  let mut scalar_out = [0u8; 3];
  unsafe { neon_rgb48_to_rgb_row::<false>(&src, &mut simd_out, 1) };
  scalar::rgb48_to_rgb_row::<false>(&src, &mut scalar_out, 1);
  assert_eq!(
    simd_out, scalar_out,
    "rgb48→rgb width=1: tail-only mismatch"
  );
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgra64_to_rgba_u16_width1_scalar_tail_only() {
  let src = [0x1111u16, 0x2222, 0x3333, 0x4444]; // B, G, R, A
  let mut simd_out = [0u16; 4];
  let mut scalar_out = [0u16; 4];
  unsafe { neon_bgra64_to_rgba_u16_row::<false>(&src, &mut simd_out, 1) };
  scalar::bgra64_to_rgba_u16_row::<false>(&src, &mut scalar_out, 1);
  assert_eq!(
    simd_out, scalar_out,
    "bgra64→rgba_u16 width=1: tail-only mismatch"
  );
}

// SIMD-level BE-vs-LE parity tests.
//
// These probe the `bswap_u16x8_if_be<BE>` gate (`BE != HOST_NATIVE_BE`) at
// the SIMD layer. Existing tests above use `BE=false` only and never exercise
// the swap path. The fix in this commit replaces the broken
// `if BE { ... }` gate (which corrupted output on the BE host × LE wire and
// BE host × BE wire quadrants) with the canonical helper from `super::`.
//
// Buffers are constructed via `to_le_bytes` / `to_be_bytes` so semantics are
// host-independent: on every host, `le_buf` carries the intended values as
// LE-encoded bytes and `be_buf` carries the same values as BE-encoded bytes.
// Both `kernel<BE=false>(le_buf)` and `kernel<BE=true>(be_buf)` should
// therefore decode to the same intended host-native u16 values and produce
// identical RGB output.
//
// Width 17 = 2 × 8-lane SIMD body + 1 scalar tail, ensuring the SIMD body
// is exercised (not just the scalar tail).

fn make_le_be_pair_u16(intended: &[u16]) -> (std::vec::Vec<u16>, std::vec::Vec<u16>) {
  let le_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  let be_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
  let le: std::vec::Vec<u16> = le_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  let be: std::vec::Vec<u16> = be_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  (le, be)
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgb48_be_le_simd_parity_width17() {
  let intended = make_rgb48_src(17, 0xACE1);
  let (le, be) = make_le_be_pair_u16(&intended);

  let mut out_le = std::vec![0u8; 17 * 3];
  let mut out_be = std::vec![0u8; 17 * 3];
  unsafe {
    neon_rgb48_to_rgb_row::<false>(&le, &mut out_le, 17);
    neon_rgb48_to_rgb_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(out_le, out_be, "rgb48→rgb SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u8; 17 * 4];
  let mut out_be = std::vec![0u8; 17 * 4];
  unsafe {
    neon_rgb48_to_rgba_row::<false>(&le, &mut out_le, 17);
    neon_rgb48_to_rgba_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(out_le, out_be, "rgb48→rgba SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u16; 17 * 3];
  let mut out_be = std::vec![0u16; 17 * 3];
  unsafe {
    neon_rgb48_to_rgb_u16_row::<false>(&le, &mut out_le, 17);
    neon_rgb48_to_rgb_u16_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "rgb48→rgb_u16 SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 17 * 4];
  let mut out_be = std::vec![0u16; 17 * 4];
  unsafe {
    neon_rgb48_to_rgba_u16_row::<false>(&le, &mut out_le, 17);
    neon_rgb48_to_rgba_u16_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "rgb48→rgba_u16 SIMD BE/LE parity (endian gate)"
  );
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgr48_be_le_simd_parity_width17() {
  let intended = make_rgb48_src(17, 0xBEEF);
  let (le, be) = make_le_be_pair_u16(&intended);

  let mut out_le = std::vec![0u8; 17 * 3];
  let mut out_be = std::vec![0u8; 17 * 3];
  unsafe {
    neon_bgr48_to_rgb_row::<false>(&le, &mut out_le, 17);
    neon_bgr48_to_rgb_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(out_le, out_be, "bgr48→rgb SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u8; 17 * 4];
  let mut out_be = std::vec![0u8; 17 * 4];
  unsafe {
    neon_bgr48_to_rgba_row::<false>(&le, &mut out_le, 17);
    neon_bgr48_to_rgba_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(out_le, out_be, "bgr48→rgba SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u16; 17 * 3];
  let mut out_be = std::vec![0u16; 17 * 3];
  unsafe {
    neon_bgr48_to_rgb_u16_row::<false>(&le, &mut out_le, 17);
    neon_bgr48_to_rgb_u16_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "bgr48→rgb_u16 SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 17 * 4];
  let mut out_be = std::vec![0u16; 17 * 4];
  unsafe {
    neon_bgr48_to_rgba_u16_row::<false>(&le, &mut out_le, 17);
    neon_bgr48_to_rgba_u16_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "bgr48→rgba_u16 SIMD BE/LE parity (endian gate)"
  );
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_rgba64_be_le_simd_parity_width17() {
  let intended = make_rgba64_src(17, 0xCAFE);
  let (le, be) = make_le_be_pair_u16(&intended);

  let mut out_le = std::vec![0u8; 17 * 3];
  let mut out_be = std::vec![0u8; 17 * 3];
  unsafe {
    neon_rgba64_to_rgb_row::<false>(&le, &mut out_le, 17);
    neon_rgba64_to_rgb_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(out_le, out_be, "rgba64→rgb SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u8; 17 * 4];
  let mut out_be = std::vec![0u8; 17 * 4];
  unsafe {
    neon_rgba64_to_rgba_row::<false>(&le, &mut out_le, 17);
    neon_rgba64_to_rgba_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "rgba64→rgba SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 17 * 3];
  let mut out_be = std::vec![0u16; 17 * 3];
  unsafe {
    neon_rgba64_to_rgb_u16_row::<false>(&le, &mut out_le, 17);
    neon_rgba64_to_rgb_u16_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "rgba64→rgb_u16 SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 17 * 4];
  let mut out_be = std::vec![0u16; 17 * 4];
  unsafe {
    neon_rgba64_to_rgba_u16_row::<false>(&le, &mut out_le, 17);
    neon_rgba64_to_rgba_u16_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "rgba64→rgba_u16 SIMD BE/LE parity (endian gate)"
  );
}

#[cfg(target_arch = "aarch64")]
#[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
#[test]
fn neon_bgra64_be_le_simd_parity_width17() {
  let intended = make_rgba64_src(17, 0xF00D);
  let (le, be) = make_le_be_pair_u16(&intended);

  let mut out_le = std::vec![0u8; 17 * 3];
  let mut out_be = std::vec![0u8; 17 * 3];
  unsafe {
    neon_bgra64_to_rgb_row::<false>(&le, &mut out_le, 17);
    neon_bgra64_to_rgb_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(out_le, out_be, "bgra64→rgb SIMD BE/LE parity (endian gate)");

  let mut out_le = std::vec![0u8; 17 * 4];
  let mut out_be = std::vec![0u8; 17 * 4];
  unsafe {
    neon_bgra64_to_rgba_row::<false>(&le, &mut out_le, 17);
    neon_bgra64_to_rgba_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "bgra64→rgba SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 17 * 3];
  let mut out_be = std::vec![0u16; 17 * 3];
  unsafe {
    neon_bgra64_to_rgb_u16_row::<false>(&le, &mut out_le, 17);
    neon_bgra64_to_rgb_u16_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "bgra64→rgb_u16 SIMD BE/LE parity (endian gate)"
  );

  let mut out_le = std::vec![0u16; 17 * 4];
  let mut out_be = std::vec![0u16; 17 * 4];
  unsafe {
    neon_bgra64_to_rgba_u16_row::<false>(&le, &mut out_le, 17);
    neon_bgra64_to_rgba_u16_row::<true>(&be, &mut out_be, 17);
  }
  assert_eq!(
    out_le, out_be,
    "bgra64→rgba_u16 SIMD BE/LE parity (endian gate)"
  );
}
