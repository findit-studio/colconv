//! Strategy A+ α-extract helpers: overwrite the α slot of RGBA buffers
//! from a separate α source, after `expand_rgb_to_rgba_row` has produced
//! a `[R, G, B, max]`-padded RGBA buffer.
//!
//! These helpers exist to recover the chroma cost in source-α formats'
//! `with_rgb + with_rgba` combo case. See spec
//! `docs/superpowers/specs/2026-05-04-pr4-strategy-a-plus-design.md` § 3.1.
//!
//! All helpers operate on `width` pixels and write **only** the α slot
//! (offset 3 of every 4-element tuple) of `rgba_out`. Other slots are
//! untouched.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

/// VUYA → u8 RGBA: gather α from `packed[3 + 4*n]` into `rgba_out[3 + 4*n]`.
///
/// VUYA layout per pixel: `[V(8), U(8), Y(8), A(8)]` — α is at slot 3.
pub(crate) fn copy_alpha_packed_u8x4_at_3(packed: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = packed[n * 4 + 3];
  }
}

/// AYUV64 → u8 RGBA: gather α from `packed[0 + 4*n]` (u16 element)
/// into `rgba_out[3 + 4*n]` (u8 element) with depth-conv `>> 8`.
///
/// AYUV64 layout per pixel: `[A(16), Y(16), U(16), V(16)]` — α is at slot 0.
pub(crate) fn copy_alpha_packed_u16x4_to_u8_at_0(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = (packed[n * 4] >> 8) as u8;
  }
}

/// AYUV64 → u16 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
pub(crate) fn copy_alpha_packed_u16x4_at_0(packed: &[u16], rgba_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = packed[n * 4];
  }
}

/// Yuva420p / 422p / 444p u8 → u8 RGBA: scatter α plane into
/// `rgba_out[3 + 4*n]`.
pub(crate) fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = alpha[n];
  }
}

/// Yuva*p9/10/12/14 → u8 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> (BITS - 8)`.
///
/// `BITS` is the source α bit depth (9, 10, 12, or 14).
///
/// α is masked with `(1 << BITS) - 1` BEFORE the shift to canonicalize
/// over-range source samples. Frame constructors admit raw u16 input
/// (e.g., p010-style buffers store the 10 active bits in the HIGH bits
/// of u16), so an unmasked over-range value would otherwise leak through
/// the shift and produce divergent output between scalar and SIMD paths.
/// See sibling inline-α kernels (`yuva_4_*` row impls) for the same
/// pattern with comment "silently turning over-range alpha into
/// transparent output".
pub(crate) fn copy_alpha_plane_u16_to_u8<const BITS: u32>(
  alpha: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(BITS >= 8 && BITS <= 16, "BITS must be in [8, 16]");
  }
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let shift = BITS - 8;
  for n in 0..width {
    rgba_out[n * 4 + 3] = ((alpha[n] & mask) >> shift) as u8;
  }
}

/// Yuva*p9/10/12/14/16 → u16 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u16). The α value is written at source bit
/// depth, masked to `(1 << BITS) - 1` so over-range source samples
/// don't leak through (parity with the inline-α kernels — frame
/// constructors admit raw u16 input above the BITS-bit native range).
pub(crate) fn copy_alpha_plane_u16<const BITS: u32>(
  alpha: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(BITS > 0 && BITS <= 16, "BITS must be in [1, 16]");
  }
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  for n in 0..width {
    rgba_out[n * 4 + 3] = alpha[n] & mask;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  #[test]
  fn copy_alpha_packed_u8x4_at_3_overwrites_only_alpha_slots() {
    let packed = [10, 20, 30, 99, 11, 21, 31, 88, 12, 22, 32, 77];
    let mut rgba = std::vec![1u8; 12];
    copy_alpha_packed_u8x4_at_3(&packed, &mut rgba, 3);
    assert_eq!(rgba, std::vec![1, 1, 1, 99, 1, 1, 1, 88, 1, 1, 1, 77]);
  }

  #[test]
  fn copy_alpha_packed_u16x4_to_u8_at_0_depth_converts_correctly() {
    let packed: std::vec::Vec<u16> = std::vec![0x1234, 100, 200, 300, 0xABCD, 101, 201, 301,];
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_packed_u16x4_to_u8_at_0(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0x12, 1, 1, 1, 0xAB]);
  }

  #[test]
  fn copy_alpha_packed_u16x4_at_0_preserves_native_u16() {
    let packed: std::vec::Vec<u16> = std::vec![0x1234, 100, 200, 300, 0xABCD, 101, 201, 301,];
    let mut rgba = std::vec![1u16; 8];
    copy_alpha_packed_u16x4_at_0(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0x1234, 1, 1, 1, 0xABCD]);
  }

  #[test]
  fn copy_alpha_plane_u8_scatters_into_rgba_alpha_slot() {
    let alpha = std::vec![50u8, 60, 70, 80];
    let mut rgba = std::vec![1u8; 16];
    copy_alpha_plane_u8(&alpha, &mut rgba, 4);
    assert_eq!(
      rgba,
      std::vec![1, 1, 1, 50, 1, 1, 1, 60, 1, 1, 1, 70, 1, 1, 1, 80]
    );
  }

  #[test]
  fn copy_alpha_plane_u16_to_u8_depth_converts_at_each_bits_value() {
    // BITS=10
    let alpha: std::vec::Vec<u16> = std::vec![0x3FF, 0x1FF];
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_plane_u16_to_u8::<10>(&alpha, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xFF, 1, 1, 1, 0x7F]);

    // BITS=12
    let alpha: std::vec::Vec<u16> = std::vec![0xFFF, 0x800];
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_plane_u16_to_u8::<12>(&alpha, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xFF, 1, 1, 1, 0x80]);

    // BITS=16
    let alpha: std::vec::Vec<u16> = std::vec![0xFFFF, 0x8000];
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_plane_u16_to_u8::<16>(&alpha, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xFF, 1, 1, 1, 0x80]);
  }

  #[test]
  fn copy_alpha_plane_u16_preserves_native_u16_within_bits_range() {
    // In-range values pass through unchanged.
    let alpha: std::vec::Vec<u16> = std::vec![0x3FF, 0x1FF, 0x000];
    let mut rgba = std::vec![1u16; 12];
    copy_alpha_plane_u16::<10>(&alpha, &mut rgba, 3);
    assert_eq!(
      rgba,
      std::vec![1, 1, 1, 0x3FF, 1, 1, 1, 0x1FF, 1, 1, 1, 0x000]
    );
  }

  #[test]
  fn copy_alpha_plane_u16_masks_overrange_to_bits_range() {
    // Over-range α (e.g., 0xFFFF at BITS=10) must be masked to low BITS.
    // Without the mask, raw u16 0xFFFF would leak straight to output and
    // exceed the documented [0, (1 << BITS) - 1] native-depth range,
    // diverging from the inline-α scalar reference.
    let alpha: std::vec::Vec<u16> = std::vec![0xFFFF, 0x0500, 0x07FF];
    let mut rgba = std::vec![1u16; 12];
    copy_alpha_plane_u16::<10>(&alpha, &mut rgba, 3);
    assert_eq!(
      rgba,
      std::vec![1, 1, 1, 0x3FF, 1, 1, 1, 0x100, 1, 1, 1, 0x3FF]
    );
  }

  #[test]
  fn copy_alpha_plane_u16_to_u8_masks_overrange_then_shifts() {
    // Without the BITS mask, 0x0500 at BITS=10 would shift `>> 2` to
    // 320 and either narrow as u8 to 64 (scalar `as u8`) or saturate to
    // 255 (some SIMD narrow-with-saturation paths). With masking, 0x0500
    // & 0x3FF = 0x100 → 0x100 >> 2 = 64 consistently across all paths.
    let alpha: std::vec::Vec<u16> = std::vec![0x0500, 0xFFFF, 0x03FF];
    let mut rgba = std::vec![1u8; 12];
    copy_alpha_plane_u16_to_u8::<10>(&alpha, &mut rgba, 3);
    assert_eq!(rgba, std::vec![1, 1, 1, 64, 1, 1, 1, 0xFF, 1, 1, 1, 0xFF]);
  }
}
