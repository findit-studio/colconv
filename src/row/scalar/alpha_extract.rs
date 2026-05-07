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

/// Rgba64 / Bgra64 → u8 RGBA: copy α from slot 3 of each 4-element u16
/// pixel tuple into `rgba_out[3 + 4*n]` (u8) with `>> 8` depth conversion.
///
/// Rgba64 / Bgra64 layout per pixel: `[R(16), G(16), B(16), A(16)]` — α is
/// at slot 3 (trailing position). Contrast with AYUV64's at-slot-0 variant.
///
/// Used in Strategy A+: after `expand_rgb_to_rgba_row` fills the RGBA buffer
/// with a forced-opaque alpha, this helper overwrites only the α slot with the
/// real source alpha, depth-converted to u8.
#[allow(dead_code)] // wired in sinker Task 10
pub(crate) fn copy_alpha_packed_u16x4_to_u8_at_3(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = (packed[n * 4 + 3] >> 8) as u8;
  }
}

/// Rgba64 / Bgra64 → u16 RGBA: copy α from slot 3 of each 4-element u16
/// pixel tuple into `rgba_u16_out[3 + 4*n]` (u16). No depth conversion.
///
/// Used in Strategy A+: after `expand_rgb_u16_to_rgba_u16_row` fills the
/// RGBA buffer, this helper overwrites only the α slot with the real source
/// alpha at native 16-bit depth.
#[allow(dead_code)] // wired in sinker Task 10
pub(crate) fn copy_alpha_packed_u16x4_at_3(packed: &[u16], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for n in 0..width {
    rgba_u16_out[n * 4 + 3] = packed[n * 4 + 3];
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
/// `BITS` is the source α bit depth (9, 10, 12, or 14). `BE` selects the
/// **byte order** of the encoded source α plane: `false` = LE on disk/wire
/// (e.g., AV `Yuva420p10le`), `true` = BE on disk/wire (e.g., `Yuva420p10be`).
///
/// Each raw u16 sample is converted from its disk byte order into host-native
/// order via `u16::from_le` / `u16::from_be` BEFORE the BITS-mask + shift.
/// On a host whose endianness matches the data, the conversion compiles to a
/// no-op; otherwise it is a `swap_bytes`. This mirrors the
/// `load_endian_u16x*::<BE>` SIMD pattern from #81 so scalar tails and SIMD
/// paths stay byte-for-byte equivalent on every host. Without this, a
/// big-endian host (e.g., s390x) processing LE source data would emit a
/// byte-reversed α plane.
///
/// α is masked with `(1 << BITS) - 1` AFTER the endian conversion to
/// canonicalize over-range source samples. Frame constructors admit raw u16
/// input (e.g., p010-style buffers store the 10 active bits in the HIGH bits
/// of u16), so an unmasked over-range value would otherwise leak through
/// the shift and produce divergent output between scalar and SIMD paths.
/// See sibling inline-α kernels (`yuva_4_*` row impls) for the same
/// pattern with comment "silently turning over-range alpha into
/// transparent output".
pub(crate) fn copy_alpha_plane_u16_to_u8<const BITS: u32, const BE: bool>(
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
    let raw = if BE {
      u16::from_be(alpha[n])
    } else {
      u16::from_le(alpha[n])
    };
    rgba_out[n * 4 + 3] = ((raw & mask) >> shift) as u8;
  }
}

/// Yuva*p9/10/12/14/16 → u16 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u16). The α value is written at source bit
/// depth, masked to `(1 << BITS) - 1` so over-range source samples
/// don't leak through (parity with the inline-α kernels — frame
/// constructors admit raw u16 input above the BITS-bit native range).
///
/// `BE` selects the **byte order** of the encoded source α plane:
/// `false` = LE on disk/wire, `true` = BE on disk/wire. Each raw u16
/// sample is converted to host-native order via `u16::from_le` /
/// `u16::from_be` BEFORE masking. On a host whose endianness matches
/// the data, the conversion compiles to a no-op; otherwise it is a
/// `swap_bytes`. Mirrors the `load_endian_u16x*::<BE>` SIMD pattern
/// from #81 so scalar and SIMD stay byte-for-byte equivalent on every
/// host. Without this, a BE host processing LE source data would emit
/// a byte-reversed α plane.
pub(crate) fn copy_alpha_plane_u16<const BITS: u32, const BE: bool>(
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
    let raw = if BE {
      u16::from_be(alpha[n])
    } else {
      u16::from_le(alpha[n])
    };
    rgba_out[n * 4 + 3] = raw & mask;
  }
}

/// Ya8 → u8 RGBA: gather A from `packed[1 + 2*n]` into `rgba_out[3 + 4*n]`.
///
/// Ya8 layout per pixel: `[Y(8), A(8)]` — α is at odd byte offsets (slot 1).
pub(crate) fn copy_alpha_ya_u8(packed: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = packed[n * 2 + 1];
  }
}

/// Ya16 → u8 RGBA: gather A from `packed[1 + 2*n]` (u16), depth-conv `>> 8`,
/// into `rgba_out[3 + 4*n]` (u8).
///
/// Ya16 layout per pixel: `[Y(16), A(16)]` — α is at odd u16 offsets (slot 1).
pub(crate) fn copy_alpha_ya_u16_to_u8(packed: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = (packed[n * 2 + 1] >> 8) as u8;
  }
}

/// Ya16 → u16 RGBA: gather A from `packed[1 + 2*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
pub(crate) fn copy_alpha_ya_u16(packed: &[u16], rgba_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = packed[n * 2 + 1];
  }
}

/// Gbrapf32 → u8 RGBA: scatter α plane (f32) into `rgba_out[3 + 4*n]` (u8).
///
/// Each α sample is clamped to `[0.0, 1.0]`, multiplied by 255, and rounded
/// with round-half-up (`+ 0.5` then truncate). Only slot 3 of every 4-element
/// tuple is written; R, G, B slots are untouched.
// Not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#[allow(dead_code)]
pub(crate) fn copy_alpha_plane_f32_to_u8(alpha: &[f32], rgba_out: &mut [u8], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = (alpha[n].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
  }
}

/// Gbrapf32 → u16 RGBA: scatter α plane (f32) into `rgba_out[3 + 4*n]` (u16).
///
/// Each α sample is clamped to `[0.0, 1.0]`, multiplied by 65535, and rounded
/// with round-half-up. Only slot 3 of every 4-element tuple is written.
// Not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#[allow(dead_code)]
pub(crate) fn copy_alpha_plane_f32_to_u16(alpha: &[f32], rgba_out: &mut [u16], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = (alpha[n].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
  }
}

/// Gbrapf32 → f32 RGBA: lossless scatter α plane (f32) into
/// `rgba_out[3 + 4*n]` (f32).
///
/// No clamping, no rounding — HDR values, NaN, and Inf in the α plane are
/// preserved bit-exact. Only slot 3 of every 4-element tuple is written.
// Not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#[allow(dead_code)]
pub(crate) fn copy_alpha_plane_f32(alpha: &[f32], rgba_out: &mut [f32], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
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
    copy_alpha_plane_u16_to_u8::<10, false>(&alpha, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xFF, 1, 1, 1, 0x7F]);

    // BITS=12
    let alpha: std::vec::Vec<u16> = std::vec![0xFFF, 0x800];
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_plane_u16_to_u8::<12, false>(&alpha, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xFF, 1, 1, 1, 0x80]);

    // BITS=16
    let alpha: std::vec::Vec<u16> = std::vec![0xFFFF, 0x8000];
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_plane_u16_to_u8::<16, false>(&alpha, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xFF, 1, 1, 1, 0x80]);
  }

  #[test]
  fn copy_alpha_plane_u16_preserves_native_u16_within_bits_range() {
    // In-range values pass through unchanged.
    let alpha: std::vec::Vec<u16> = std::vec![0x3FF, 0x1FF, 0x000];
    let mut rgba = std::vec![1u16; 12];
    copy_alpha_plane_u16::<10, false>(&alpha, &mut rgba, 3);
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
    copy_alpha_plane_u16::<10, false>(&alpha, &mut rgba, 3);
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
    copy_alpha_plane_u16_to_u8::<10, false>(&alpha, &mut rgba, 3);
    assert_eq!(rgba, std::vec![1, 1, 1, 64, 1, 1, 1, 0xFF, 1, 1, 1, 0xFF]);
  }

  /// BE parity: byte-swapping the source α plane and toggling the `BE`
  /// flag must yield byte-for-byte identical output. Locks down the
  /// codex-flagged corruption where a BE host processing LE input
  /// would otherwise emit a byte-reversed α slot. The synthesized
  /// "BE-encoded" buffer is built by host-side `swap_bytes` on the LE
  /// fixture; both `from_le` (LE flag) and `from_be` (BE flag with the
  /// swapped buffer) recover the same logical u16 values, so the
  /// outputs match on every host.
  #[test]
  fn copy_alpha_plane_u16_to_u8_be_parity_with_swapped_buffer() {
    let alpha_le: std::vec::Vec<u16> = std::vec![0x3FF, 0x1FF, 0x0500, 0xFFFF, 0x07FF, 0x0123];
    let alpha_be: std::vec::Vec<u16> = alpha_le.iter().map(|x| x.swap_bytes()).collect();
    let mut rgba_le = std::vec![1u8; 24];
    let mut rgba_be = std::vec![1u8; 24];
    copy_alpha_plane_u16_to_u8::<10, false>(&alpha_le, &mut rgba_le, 6);
    copy_alpha_plane_u16_to_u8::<10, true>(&alpha_be, &mut rgba_be, 6);
    assert_eq!(
      rgba_le, rgba_be,
      "BE flag + byte-swapped buffer must match LE path"
    );
  }

  /// BE parity for the u16-output variant.
  #[test]
  fn copy_alpha_plane_u16_be_parity_with_swapped_buffer() {
    let alpha_le: std::vec::Vec<u16> = std::vec![0xFFFF, 0x0500, 0x07FF, 0x0123, 0x3FF, 0x000];
    let alpha_be: std::vec::Vec<u16> = alpha_le.iter().map(|x| x.swap_bytes()).collect();
    let mut rgba_le = std::vec![7u16; 24];
    let mut rgba_be = std::vec![7u16; 24];
    copy_alpha_plane_u16::<10, false>(&alpha_le, &mut rgba_le, 6);
    copy_alpha_plane_u16::<10, true>(&alpha_be, &mut rgba_be, 6);
    assert_eq!(
      rgba_le, rgba_be,
      "BE flag + byte-swapped buffer must match LE path"
    );
  }

  #[test]
  fn copy_alpha_ya_u8_extracts_alpha_from_odd_byte_slots() {
    // Ya8 packed layout: [Y0, A0, Y1, A1, Y2, A2]
    let packed = std::vec![10u8, 99, 20, 88, 30, 77];
    let mut rgba = std::vec![1u8; 12];
    copy_alpha_ya_u8(&packed, &mut rgba, 3);
    assert_eq!(rgba, std::vec![1, 1, 1, 99, 1, 1, 1, 88, 1, 1, 1, 77]);
  }

  #[test]
  fn copy_alpha_ya_u16_to_u8_depth_converts_via_high_byte() {
    // Ya16 packed → u8 RGBA: α >> 8 selects the high byte.
    let packed: std::vec::Vec<u16> = std::vec![0x1234, 0xABCD, 0x5678, 0xFF00];
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_ya_u16_to_u8(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xAB, 1, 1, 1, 0xFF]);
  }

  #[test]
  fn copy_alpha_ya_u16_preserves_native_u16() {
    let packed: std::vec::Vec<u16> = std::vec![0x1234, 0xABCD, 0x5678, 0x9ABC];
    let mut rgba = std::vec![1u16; 8];
    copy_alpha_ya_u16(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xABCD, 1, 1, 1, 0x9ABC]);
  }

  #[test]
  fn copy_alpha_plane_f32_to_u8_clamps_and_scales() {
    // Values [0.0, 0.5, 1.0, 1.5, -0.1] → [0, 128, 255, 255, 0] in slot 3.
    let alpha = vec![0.0f32, 0.5, 1.0, 1.5, -0.1];
    let mut rgba = vec![1u8; 20];
    copy_alpha_plane_f32_to_u8(&alpha, &mut rgba, 5);
    // R, G, B slots (0, 1, 2) must be untouched; slot 3 has the alpha.
    assert_eq!(rgba[3], 0, "alpha[0]=0.0 → 0");
    assert_eq!(rgba[7], 128, "alpha[1]=0.5 → 128");
    assert_eq!(rgba[11], 255, "alpha[2]=1.0 → 255");
    assert_eq!(rgba[15], 255, "alpha[3]=1.5 → clamped to 255");
    assert_eq!(rgba[19], 0, "alpha[4]=-0.1 → clamped to 0");
    // Non-alpha slots unchanged.
    assert_eq!(rgba[0], 1);
    assert_eq!(rgba[1], 1);
    assert_eq!(rgba[2], 1);
  }

  #[test]
  fn copy_alpha_plane_f32_to_u16_clamps_and_scales() {
    // Values [0.0, 0.5, 1.0, 1.5, -0.1] → [0, 32768, 65535, 65535, 0] in slot 3.
    let alpha = vec![0.0f32, 0.5, 1.0, 1.5, -0.1];
    let mut rgba = vec![1u16; 20];
    copy_alpha_plane_f32_to_u16(&alpha, &mut rgba, 5);
    assert_eq!(rgba[3], 0, "alpha[0]=0.0 → 0");
    assert_eq!(rgba[7], 32768, "alpha[1]=0.5 → 32768");
    assert_eq!(rgba[11], 65535, "alpha[2]=1.0 → 65535");
    assert_eq!(rgba[15], 65535, "alpha[3]=1.5 → clamped to 65535");
    assert_eq!(rgba[19], 0, "alpha[4]=-0.1 → clamped to 0");
    // Non-alpha slots unchanged.
    assert_eq!(rgba[0], 1);
    assert_eq!(rgba[1], 1);
    assert_eq!(rgba[2], 1);
  }

  #[test]
  fn copy_alpha_plane_f32_lossless_passthrough() {
    // HDR (2.5), NaN, Inf, negative all preserved bit-exact.
    let alpha = vec![2.5f32, f32::NAN, f32::INFINITY, -1.0];
    let mut rgba = vec![0.0f32; 16];
    copy_alpha_plane_f32(&alpha, &mut rgba, 4);
    assert_eq!(rgba[3], 2.5, "HDR 2.5 preserved");
    assert!(rgba[7].is_nan(), "NaN preserved");
    assert!(rgba[11].is_infinite() && rgba[11] > 0.0, "+Inf preserved");
    assert_eq!(rgba[15], -1.0, "negative preserved");
    // Non-alpha slots untouched (still 0.0).
    assert_eq!(rgba[0], 0.0);
    assert_eq!(rgba[1], 0.0);
    assert_eq!(rgba[2], 0.0);
  }

  // ---- copy_alpha_packed_u16x4_to_u8_at_3 / copy_alpha_packed_u16x4_at_3 --

  /// Alpha at slot 3 is depth-converted >> 8 and written to rgba_out[3 + 4*n].
  #[test]
  fn copy_alpha_packed_u16x4_to_u8_at_3_narrows_correctly() {
    let packed: std::vec::Vec<u16> = std::vec![100, 200, 300, 0xABFF, 101, 201, 301, 0x1234];
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_packed_u16x4_to_u8_at_3(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xAB, 1, 1, 1, 0x12]);
  }

  /// Alpha at slot 3 is copied verbatim (no depth conversion).
  #[test]
  fn copy_alpha_packed_u16x4_at_3_copies_verbatim() {
    let packed: std::vec::Vec<u16> = std::vec![100, 200, 300, 0xABFF, 101, 201, 301, 0x1234];
    let mut rgba_u16 = std::vec![1u16; 8];
    copy_alpha_packed_u16x4_at_3(&packed, &mut rgba_u16, 2);
    assert_eq!(rgba_u16, std::vec![1, 1, 1, 0xABFF, 1, 1, 1, 0x1234]);
  }

  /// Only the alpha slot (index 3) is overwritten; RGB slots [0..3] are untouched.
  #[test]
  fn copy_alpha_packed_u16x4_to_u8_at_3_touches_only_alpha_slot() {
    let packed: std::vec::Vec<u16> = std::vec![0, 0, 0, 0xFFFF];
    let mut rgba = std::vec![42u8; 4];
    copy_alpha_packed_u16x4_to_u8_at_3(&packed, &mut rgba, 1);
    assert_eq!(rgba[..3], [42, 42, 42]);
    assert_eq!(rgba[3], 0xFF);
  }

  /// Only the alpha slot (index 3) is overwritten; RGB slots [0..3] are untouched.
  #[test]
  fn copy_alpha_packed_u16x4_at_3_touches_only_alpha_slot() {
    let packed: std::vec::Vec<u16> = std::vec![0, 0, 0, 0xBEEF];
    let mut rgba_u16 = std::vec![99u16; 4];
    copy_alpha_packed_u16x4_at_3(&packed, &mut rgba_u16, 1);
    assert_eq!(rgba_u16[..3], [99, 99, 99]);
    assert_eq!(rgba_u16[3], 0xBEEF);
  }
}
