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

// Functions are integrated in Task 7; suppress dead_code until then.
#![allow(dead_code)]

/// VUYA → u8 RGBA: gather α from `packed[3 + 4*n]` into `rgba_out[3 + 4*n]`.
///
/// VUYA layout per pixel: `[V(8), U(8), Y(8), A(8)]` — α is at slot 3.
pub(crate) fn copy_alpha_packed_u8x4_at_3(packed: &[u8], rgba_out: &mut [u8], width: usize) {
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
  for n in 0..width {
    rgba_out[n * 4 + 3] = (packed[n * 4] >> 8) as u8;
  }
}

/// AYUV64 → u16 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
pub(crate) fn copy_alpha_packed_u16x4_at_0(packed: &[u16], rgba_out: &mut [u16], width: usize) {
  for n in 0..width {
    rgba_out[n * 4 + 3] = packed[n * 4];
  }
}

/// Yuva420p / 422p / 444p u8 → u8 RGBA: scatter α plane into
/// `rgba_out[3 + 4*n]`.
pub(crate) fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize) {
  for n in 0..width {
    rgba_out[n * 4 + 3] = alpha[n];
  }
}

/// Yuva*p9/10/12/14 → u8 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> (BITS - 8)`.
///
/// `BITS` is the source α bit depth (9, 10, 12, or 14).
pub(crate) fn copy_alpha_plane_u16_to_u8<const BITS: u32>(
  alpha: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  let shift = BITS - 8;
  for n in 0..width {
    rgba_out[n * 4 + 3] = (alpha[n] >> shift) as u8;
  }
}

/// Yuva*p9/10/12/14/16 → u16 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u16). `BITS` is unused at runtime — kept as a
/// const generic for symmetry with the `_to_u8` sibling and future
/// validation. The α value is written natively at source bit depth.
pub(crate) fn copy_alpha_plane_u16<const BITS: u32>(
  alpha: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  // BITS is informational — no shift applied, output preserves source depth.
  let _ = BITS;
  for n in 0..width {
    rgba_out[n * 4 + 3] = alpha[n];
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
  fn copy_alpha_plane_u16_preserves_native_u16() {
    let alpha: std::vec::Vec<u16> = std::vec![0x3FF, 0x1FF, 0xFFFF];
    let mut rgba = std::vec![1u16; 12];
    copy_alpha_plane_u16::<10>(&alpha, &mut rgba, 3);
    assert_eq!(
      rgba,
      std::vec![1, 1, 1, 0x3FF, 1, 1, 1, 0x1FF, 1, 1, 1, 0xFFFF]
    );
  }
}
