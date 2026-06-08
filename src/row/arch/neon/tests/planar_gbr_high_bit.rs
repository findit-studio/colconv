//! NEON parity tests for the high-bit-depth planar-GBR kernels (Tier 10b).
//!
//! Covers BITS=10 and BITS=16 at widths [1, 7, 8, 16, 17, 32, 33, 64, 128, 130].
//! Each test asserts SIMD output == scalar output.

use super::*;

fn gbr_plane_u16<const BITS: u32>(width: usize, seed: u32) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  let mut state = seed;
  (0..width)
    .map(|_| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (state & mask) as u16
    })
    .collect()
}

/// Generate a plane where every sample has upper bits set (dirty inputs).
fn gbr_plane_u16_dirty<const BITS: u32>(width: usize, dirty_upper: u16) -> std::vec::Vec<u16> {
  // Combine a clean in-range value with the dirty upper bits.
  let clean_mask = ((1u32 << BITS) - 1) as u16;
  // Use a simple pattern: pixel i gets value (i as u16 & clean_mask) | dirty_upper.
  (0..width)
    .map(|i| (i as u16 & clean_mask) | dirty_upper)
    .collect()
}

// ---- u8 output -----------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<16, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<10, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgba_opaque_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<16, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgba_opaque_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_high_bit<16> diverges (width={w})"
    );
  }
}

// ---- u16 output ----------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_u16_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_u16_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_u16_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_u16_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgba_opaque_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_u16_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::gbr_to_rgba_opaque_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<16, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgba_opaque_u16_high_bit<16> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_u16_high_bit_matches_scalar_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_u16_high_bit<10> diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_u16_high_bit_matches_scalar_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<16, false>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_u16_high_bit<16> diverges (width={w})"
    );
  }
}

// ---- Upper-bits masking: SIMD must match scalar for dirty inputs ----------

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_high_bit_upper_bits_masked_bits10() {
  // dirty_upper = 0x0400 sets bit 10 which is out of range for BITS=10.
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let b = gbr_plane_u16_dirty::<10>(w, 0x0800);
    let r = gbr_plane_u16_dirty::<10>(w, 0x0400);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_high_bit_row::<10, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_high_bit<10> dirty-input diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_high_bit_upper_bits_masked_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let b = gbr_plane_u16_dirty::<10>(w, 0x0800);
    let r = gbr_plane_u16_dirty::<10>(w, 0x0400);
    let a = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_high_bit<10> dirty-input diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_u16_high_bit_upper_bits_masked_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let b = gbr_plane_u16_dirty::<10>(w, 0x0800);
    let r = gbr_plane_u16_dirty::<10>(w, 0x0400);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<10, false>(&g, &b, &r, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbr_to_rgb_u16_high_bit<10> dirty-input diverges (width={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_u16_high_bit_upper_bits_masked_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let b = gbr_plane_u16_dirty::<10>(w, 0x0800);
    let r = gbr_plane_u16_dirty::<10>(w, 0x0400);
    let a = gbr_plane_u16_dirty::<10>(w, 0x0C00);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<10, false>(&g, &b, &r, &a, &mut out_neon, w);
    }
    assert_eq!(
      out_scalar, out_neon,
      "NEON gbra_to_rgba_u16_high_bit<10> dirty-input diverges (width={w})"
    );
  }
}

// BE parity: NEON<BITS, true> on BE-encoded storage must match
// NEON<BITS, false> on LE-encoded storage. Each plane is built from
// the same host-native `intended` buffer, then re-encoded with
// `to_le_bytes` / `to_be_bytes` so the kernels' `from_le` / `from_be`
// decode it back to the same logical values on every host. A naive
// `swap_bytes` of host-native data is vacuous: on BE the `<false>` path
// would byte-swap host-native into wrong logical values while the
// `<true>` path on the swapped buffer produced the same wrong values,
// so equality could pass on a corrupted decode. Each test also pins
// the LE output to an absolute expected value computed independently
// from the host-native intended planes.

/// Re-encode host-native `u16` samples as LE byte storage. On a LE host this
/// is identity; on a BE host each element is byte-swapped so the kernel's
/// `from_le` recovers the original logical value.
fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode host-native `u16` samples as BE byte storage. Mirror of
/// `as_le_u16` for the `<true>` (BE) kernel path.
fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Independent reference for `gbr_to_rgb_high_bit_row`: reorders the planes
/// to packed R, G, B and applies `>> (BITS - 8)` on host-native logical
/// samples. Pins the LE path's output absolutely so equality cannot pass on
/// equally corrupted decodes.
fn ref_gbr_to_rgb_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  width: usize,
) -> std::vec::Vec<u8> {
  let shift = BITS - 8;
  let mut out = std::vec![0u8; width * 3];
  for x in 0..width {
    out[x * 3] = (r[x] >> shift) as u8;
    out[x * 3 + 1] = (g[x] >> shift) as u8;
    out[x * 3 + 2] = (b[x] >> shift) as u8;
  }
  out
}

/// Independent reference for `gbr_to_rgba_opaque_high_bit_row`.
fn ref_gbr_to_rgba_opaque_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  width: usize,
) -> std::vec::Vec<u8> {
  let shift = BITS - 8;
  let mut out = std::vec![0u8; width * 4];
  for x in 0..width {
    out[x * 4] = (r[x] >> shift) as u8;
    out[x * 4 + 1] = (g[x] >> shift) as u8;
    out[x * 4 + 2] = (b[x] >> shift) as u8;
    out[x * 4 + 3] = 0xFF;
  }
  out
}

/// Independent reference for `gbra_to_rgba_high_bit_row`.
fn ref_gbra_to_rgba_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  width: usize,
) -> std::vec::Vec<u8> {
  let shift = BITS - 8;
  let mut out = std::vec![0u8; width * 4];
  for x in 0..width {
    out[x * 4] = (r[x] >> shift) as u8;
    out[x * 4 + 1] = (g[x] >> shift) as u8;
    out[x * 4 + 2] = (b[x] >> shift) as u8;
    out[x * 4 + 3] = (a[x] >> shift) as u8;
  }
  out
}

/// Independent reference for `gbr_to_rgb_u16_high_bit_row`.
fn ref_gbr_to_rgb_u16_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  width: usize,
) -> std::vec::Vec<u16> {
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let mut out = std::vec![0u16; width * 3];
  for x in 0..width {
    out[x * 3] = r[x] & mask;
    out[x * 3 + 1] = g[x] & mask;
    out[x * 3 + 2] = b[x] & mask;
  }
  out
}

/// Independent reference for `gbr_to_rgba_opaque_u16_high_bit_row`.
fn ref_gbr_to_rgba_opaque_u16_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  width: usize,
) -> std::vec::Vec<u16> {
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let mut out = std::vec![0u16; width * 4];
  for x in 0..width {
    out[x * 4] = r[x] & mask;
    out[x * 4 + 1] = g[x] & mask;
    out[x * 4 + 2] = b[x] & mask;
    out[x * 4 + 3] = mask;
  }
  out
}

/// Independent reference for `gbra_to_rgba_u16_high_bit_row`.
fn ref_gbra_to_rgba_u16_high_bit<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  width: usize,
) -> std::vec::Vec<u16> {
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let mut out = std::vec![0u16; width * 4];
  for x in 0..width {
    out[x * 4] = r[x] & mask;
    out[x * 4 + 1] = g[x] & mask;
    out[x * 4 + 2] = b[x] & mask;
    out[x * 4 + 3] = a[x] & mask;
  }
  out
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_high_bit_be_matches_le_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      gbr_to_rgb_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &mut out_le, w);
      gbr_to_rgb_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    let expected = ref_gbr_to_rgb_high_bit::<10>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "NEON gbr_to_rgb_high_bit<10> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbr_to_rgb_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_high_bit_be_matches_le_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      gbr_to_rgb_high_bit_row::<16, false>(&g_le, &b_le, &r_le, &mut out_le, w);
      gbr_to_rgb_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    let expected = ref_gbr_to_rgb_high_bit::<16>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "NEON gbr_to_rgb_high_bit<16> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbr_to_rgb_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_high_bit_be_matches_le_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &mut out_le, w);
      gbr_to_rgba_opaque_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    let expected = ref_gbr_to_rgba_opaque_high_bit::<10>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "NEON gbr_to_rgba_opaque_high_bit<10> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbr_to_rgba_opaque_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_high_bit_be_matches_le_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      gbr_to_rgba_opaque_high_bit_row::<16, false>(&g_le, &b_le, &r_le, &mut out_le, w);
      gbr_to_rgba_opaque_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    let expected = ref_gbr_to_rgba_opaque_high_bit::<16>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "NEON gbr_to_rgba_opaque_high_bit<16> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbr_to_rgba_opaque_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_high_bit_be_matches_le_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let a_le = as_le_u16(&a);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let a_be = as_be_u16(&a);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      gbra_to_rgba_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &a_le, &mut out_le, w);
      gbra_to_rgba_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    }
    let expected = ref_gbra_to_rgba_high_bit::<10>(&g, &b, &r, &a, w);
    assert_eq!(
      out_le, expected,
      "NEON gbra_to_rgba_high_bit<10> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbra_to_rgba_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_high_bit_be_matches_le_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let a_le = as_le_u16(&a);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let a_be = as_be_u16(&a);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      gbra_to_rgba_high_bit_row::<16, false>(&g_le, &b_le, &r_le, &a_le, &mut out_le, w);
      gbra_to_rgba_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    }
    let expected = ref_gbra_to_rgba_high_bit::<16>(&g, &b, &r, &a, w);
    assert_eq!(
      out_le, expected,
      "NEON gbra_to_rgba_high_bit<16> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbra_to_rgba_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_u16_high_bit_be_matches_le_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u16; w * 3];
    let mut out_be = std::vec![0u16; w * 3];
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &mut out_le, w);
      gbr_to_rgb_u16_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    let expected = ref_gbr_to_rgb_u16_high_bit::<10>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "NEON gbr_to_rgb_u16_high_bit<10> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbr_to_rgb_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgb_u16_high_bit_be_matches_le_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u16; w * 3];
    let mut out_be = std::vec![0u16; w * 3];
    unsafe {
      gbr_to_rgb_u16_high_bit_row::<16, false>(&g_le, &b_le, &r_le, &mut out_le, w);
      gbr_to_rgb_u16_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    let expected = ref_gbr_to_rgb_u16_high_bit::<16>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "NEON gbr_to_rgb_u16_high_bit<16> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbr_to_rgb_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_u16_high_bit_be_matches_le_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &mut out_le, w);
      gbr_to_rgba_opaque_u16_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    let expected = ref_gbr_to_rgba_opaque_u16_high_bit::<10>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "NEON gbr_to_rgba_opaque_u16_high_bit<10> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbr_to_rgba_opaque_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbr_to_rgba_opaque_u16_high_bit_be_matches_le_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      gbr_to_rgba_opaque_u16_high_bit_row::<16, false>(&g_le, &b_le, &r_le, &mut out_le, w);
      gbr_to_rgba_opaque_u16_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &mut out_be, w);
    }
    let expected = ref_gbr_to_rgba_opaque_u16_high_bit::<16>(&g, &b, &r, w);
    assert_eq!(
      out_le, expected,
      "NEON gbr_to_rgba_opaque_u16_high_bit<16> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbr_to_rgba_opaque_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_u16_high_bit_be_matches_le_bits10() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<10>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<10>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<10>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<10>(w, 0xCAFE_F00D);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let a_le = as_le_u16(&a);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let a_be = as_be_u16(&a);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<10, false>(&g_le, &b_le, &r_le, &a_le, &mut out_le, w);
      gbra_to_rgba_u16_high_bit_row::<10, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    }
    let expected = ref_gbra_to_rgba_u16_high_bit::<10>(&g, &b, &r, &a, w);
    assert_eq!(
      out_le, expected,
      "NEON gbra_to_rgba_u16_high_bit<10> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbra_to_rgba_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_gbra_to_rgba_u16_high_bit_be_matches_le_bits16() {
  for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
    let g = gbr_plane_u16::<16>(w, 0x6CCD_5C7B);
    let b = gbr_plane_u16::<16>(w, 0x12AB_34CD);
    let r = gbr_plane_u16::<16>(w, 0xDEAD_BEEF);
    let a = gbr_plane_u16::<16>(w, 0xCAFE_F00D);
    let g_le = as_le_u16(&g);
    let b_le = as_le_u16(&b);
    let r_le = as_le_u16(&r);
    let a_le = as_le_u16(&a);
    let g_be = as_be_u16(&g);
    let b_be = as_be_u16(&b);
    let r_be = as_be_u16(&r);
    let a_be = as_be_u16(&a);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      gbra_to_rgba_u16_high_bit_row::<16, false>(&g_le, &b_le, &r_le, &a_le, &mut out_le, w);
      gbra_to_rgba_u16_high_bit_row::<16, true>(&g_be, &b_be, &r_be, &a_be, &mut out_be, w);
    }
    let expected = ref_gbra_to_rgba_u16_high_bit::<16>(&g, &b, &r, &a, w);
    assert_eq!(
      out_le, expected,
      "NEON gbra_to_rgba_u16_high_bit<16> LE output != reference (w={w})"
    );
    assert_eq!(
      out_le, out_be,
      "NEON gbra_to_rgba_u16_high_bit BE/LE mismatch (w={w})"
    );
  }
}
