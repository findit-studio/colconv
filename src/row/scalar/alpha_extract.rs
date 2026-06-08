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
#[cfg(feature = "yuv-444-packed")]
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
///
/// `BE` selects the **byte order** of the encoded source `packed` plane
/// (`false` = LE on disk/wire, e.g. `AV_PIX_FMT_AYUV64LE` per the Frame
/// contract; `true` = BE on disk/wire). Each raw u16 is normalised to
/// host-native order via `u16::from_le` / `u16::from_be` before the
/// `>> 8` depth conversion. On a host whose endianness matches the
/// source the conversion compiles to a no-op; otherwise it is a
/// `swap_bytes`. Without this a BE host (e.g., s390x) processing the
/// LE-encoded Frame would emit a byte-reversed α byte.
#[cfg(feature = "yuv-444-packed")]
pub(crate) fn copy_alpha_packed_u16x4_to_u8_at_0<const BE: bool>(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    let raw = if BE {
      u16::from_be(packed[n * 4])
    } else {
      u16::from_le(packed[n * 4])
    };
    rgba_out[n * 4 + 3] = (raw >> 8) as u8;
  }
}

/// AYUV64 → u16 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// `BE` selects the **byte order** of the encoded source `packed` plane.
/// See [`copy_alpha_packed_u16x4_to_u8_at_0`] for the full rationale.
#[cfg(feature = "yuv-444-packed")]
pub(crate) fn copy_alpha_packed_u16x4_at_0<const BE: bool>(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    let raw = if BE {
      u16::from_be(packed[n * 4])
    } else {
      u16::from_le(packed[n * 4])
    };
    rgba_out[n * 4 + 3] = raw;
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
///
/// `BE` selects the **byte order** of the encoded source `packed` plane
/// (`false` = LE on disk/wire, e.g. `AV_PIX_FMT_RGBA64LE` /
/// `AV_PIX_FMT_BGRA64LE` per the Frame contract; `true` = BE). Each raw
/// u16 is normalised to host-native order via `u16::from_le` /
/// `u16::from_be` before the `>> 8` depth conversion. Without this a BE
/// host processing the LE-encoded Frame would emit a byte-reversed α byte.
#[allow(dead_code)] // wired in sinker Task 10
pub(crate) fn copy_alpha_packed_u16x4_to_u8_at_3<const BE: bool>(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    let raw = if BE {
      u16::from_be(packed[n * 4 + 3])
    } else {
      u16::from_le(packed[n * 4 + 3])
    };
    rgba_out[n * 4 + 3] = (raw >> 8) as u8;
  }
}

/// Rgba64 / Bgra64 → u16 RGBA: copy α from slot 3 of each 4-element u16
/// pixel tuple into `rgba_u16_out[3 + 4*n]` (u16). No depth conversion.
///
/// Used in Strategy A+: after `expand_rgb_u16_to_rgba_u16_row` fills the
/// RGBA buffer, this helper overwrites only the α slot with the real source
/// alpha at native 16-bit depth.
///
/// `BE` selects the **byte order** of the encoded source `packed` plane.
/// See [`copy_alpha_packed_u16x4_to_u8_at_3`] for the full rationale.
#[allow(dead_code)] // wired in sinker Task 10
pub(crate) fn copy_alpha_packed_u16x4_at_3<const BE: bool>(
  packed: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for n in 0..width {
    let raw = if BE {
      u16::from_be(packed[n * 4 + 3])
    } else {
      u16::from_le(packed[n * 4 + 3])
    };
    rgba_u16_out[n * 4 + 3] = raw;
  }
}

/// Yuva420p / 422p / 444p u8 → u8 RGBA: scatter α plane into
/// `rgba_out[3 + 4*n]`. Consumed by Yuva planar (`yuva`) and Gbrap
/// (`gbr`) source families.
#[cfg(any(feature = "gbr", feature = "yuva"))]
pub(crate) fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = alpha[n];
  }
}

/// Yuva*p9/10/12/14/16 + Gbrap10/12/14/16 → u8 RGBA: scatter α plane
/// (u16) into `rgba_out[3 + 4*n]` (u8) with depth-conv `>> (BITS - 8)`.
///
/// `BITS` is the source α bit depth (any value in `[8, 16]`; the runtime
/// `assert!` enforces the range). In practice callers pass 9, 10, 12, 14,
/// or 16. `BE` selects the **byte order** of the encoded source α plane:
/// `false` = LE on disk/wire (e.g., AV `Yuva420p10le`, `Gbrap10le`),
/// `true` = BE on disk/wire (e.g., `Yuva420p10be`, `Gbrap10be`).
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
#[cfg(any(feature = "gbr", feature = "yuva"))]
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
#[cfg(any(feature = "gbr", feature = "yuva"))]
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
#[cfg(feature = "gray")]
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
///
/// `BE` selects the **byte order** of the encoded source `packed` plane
/// (`false` = LE on disk/wire, e.g. `AV_PIX_FMT_YA16LE` per the
/// `Ya16Frame` contract; `true` = BE). Each raw u16 is normalised to
/// host-native order via `u16::from_le` / `u16::from_be` before the
/// `>> 8` depth conversion. Without this a BE host processing the
/// LE-encoded Frame would emit a byte-reversed α byte.
#[cfg(feature = "gray")]
pub(crate) fn copy_alpha_ya_u16_to_u8<const BE: bool>(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    let raw = if BE {
      u16::from_be(packed[n * 2 + 1])
    } else {
      u16::from_le(packed[n * 2 + 1])
    };
    rgba_out[n * 4 + 3] = (raw >> 8) as u8;
  }
}

/// Ya16 → u16 RGBA: gather A from `packed[1 + 2*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// `BE` selects the **byte order** of the encoded source `packed` plane.
/// See [`copy_alpha_ya_u16_to_u8`] for the full rationale.
#[cfg(feature = "gray")]
pub(crate) fn copy_alpha_ya_u16<const BE: bool>(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    let raw = if BE {
      u16::from_be(packed[n * 2 + 1])
    } else {
      u16::from_le(packed[n * 2 + 1])
    };
    rgba_out[n * 4 + 3] = raw;
  }
}

/// Gbrapf32 → u8 RGBA: scatter α plane (f32) into `rgba_out[3 + 4*n]` (u8).
///
/// Each α sample is clamped to `[0.0, 1.0]`, multiplied by 255, and rounded
/// with round-half-up (`+ 0.5` then truncate). Only slot 3 of every 4-element
/// tuple is written; R, G, B slots are untouched.
///
/// `BE` selects the **byte order** of the encoded source α plane:
/// `false` = LE on disk/wire (e.g., `AV_PIX_FMT_GBRAPF32LE` per the
/// `Gbrapf32Frame` contract; this also matches the case where the f32
/// scratch is already host-native and the host is little-endian);
/// `true` = BE on disk/wire (or host-native scratch on a BE host). Each
/// raw f32 is bit-normalised to host-native order via
/// `f32::from_bits(u32::from_le(bits))` (or `from_be`) BEFORE the clamp /
/// scale / round-half-up. Without this a BE host (e.g., s390x) processing
/// the LE-encoded Frame would clamp byte-swapped garbage values, typically
/// producing α = 0 or α = 255 regardless of intent. Mirrors the
/// `copy_alpha_plane_u16_to_u8::<BITS, BE>` endian pattern.
///
/// Routing pattern at the sinker layer:
/// - **Direct-Frame paths** (e.g., `Gbrapf32Frame` → α plane consumed directly)
///   pass `BE = false` (data is LE-encoded per the unified Frame contract).
/// - **Post-widen paths** (e.g., `Gbrapf16Frame` widened-to-f32 scratch) pass
///   `BE = HOST_NATIVE_BE` (scratch is host-native f32 after widen).
// Not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#[allow(dead_code)]
pub(crate) fn copy_alpha_plane_f32_to_u8<const BE: bool>(
  alpha: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    let bits = alpha[n].to_bits();
    let host_bits = if BE {
      u32::from_be(bits)
    } else {
      u32::from_le(bits)
    };
    let v = f32::from_bits(host_bits);
    rgba_out[n * 4 + 3] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
  }
}

/// Gbrapf32 → u16 RGBA: scatter α plane (f32) into `rgba_out[3 + 4*n]` (u16).
///
/// Each α sample is clamped to `[0.0, 1.0]`, multiplied by 65535, and rounded
/// with round-half-up. Only slot 3 of every 4-element tuple is written.
///
/// `BE` selects the **byte order** of the encoded source α plane.
/// See [`copy_alpha_plane_f32_to_u8`] for the full rationale and the
/// direct-Frame vs post-widen routing pattern.
// Not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#[allow(dead_code)]
pub(crate) fn copy_alpha_plane_f32_to_u16<const BE: bool>(
  alpha: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    let bits = alpha[n].to_bits();
    let host_bits = if BE {
      u32::from_be(bits)
    } else {
      u32::from_le(bits)
    };
    let v = f32::from_bits(host_bits);
    rgba_out[n * 4 + 3] = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
  }
}

/// Gbrapf32 → f32 RGBA: lossless scatter α plane (f32) into
/// `rgba_out[3 + 4*n]` (f32).
///
/// No clamping, no rounding — HDR values, NaN, and Inf in the α plane are
/// preserved bit-exact. Only slot 3 of every 4-element tuple is written.
/// The output α is always written in **host-native** byte order (the
/// downstream consumer of `&[f32]` expects host-native floats); this helper's
/// `BE` only describes the **input** plane.
///
/// `BE` selects the **byte order** of the encoded source α plane.
/// See [`copy_alpha_plane_f32_to_u8`] for the full rationale and the
/// direct-Frame vs post-widen routing pattern.
// Not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#[allow(dead_code)]
pub(crate) fn copy_alpha_plane_f32<const BE: bool>(
  alpha: &[f32],
  rgba_out: &mut [f32],
  width: usize,
) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    let bits = alpha[n].to_bits();
    let host_bits = if BE {
      u32::from_be(bits)
    } else {
      u32::from_le(bits)
    };
    rgba_out[n * 4 + 3] = f32::from_bits(host_bits);
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  /// Re-encode a host-native u16 slice as LE-encoded byte storage, packed back
  /// into `Vec<u16>`. On LE host this is a no-op; on BE host every u16 is byte-
  /// swapped relative to its host-native representation. Kernels called with
  /// `BE = false` recover the intended logical values via `u16::from_le` on
  /// both LE and BE hosts.
  fn as_le_u16(host: &[u16]) -> std::vec::Vec<u16> {
    host
      .iter()
      .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
      .collect()
  }

  /// Re-encode a host-native u16 slice as BE-encoded byte storage. Mirror of
  /// `as_le_u16` for kernels invoked with `BE = true`. On a BE host this is a
  /// no-op; on a LE host every u16 is byte-swapped relative to its host-native
  /// representation. Combined with `as_le_u16`, this lets a single host-native
  /// `intended` fixture drive both `<false>` and `<true>` kernel paths so they
  /// must decode the same logical values on every host.
  fn as_be_u16(host: &[u16]) -> std::vec::Vec<u16> {
    host
      .iter()
      .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
      .collect()
  }

  /// Re-encode a host-native f32 slice as LE-encoded bit storage, packed back
  /// into `Vec<f32>`. The f32 bits are byte-swapped on a BE host so that the
  /// kernel's `u32::from_le(bits)` recovers the original logical bit pattern.
  fn as_le_f32(host: &[f32]) -> std::vec::Vec<f32> {
    host
      .iter()
      .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
      .collect()
  }

  /// Mirror of `as_le_f32` for kernels invoked with `BE = true`.
  fn as_be_f32(host: &[f32]) -> std::vec::Vec<f32> {
    host
      .iter()
      .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_be_bytes())))
      .collect()
  }

  // Scalar references for the BE-parity tests.
  //
  // These walk host-native `intended` buffers and reproduce the kernel's
  // documented behaviour without going through any byte-order conversion.
  // Pinning the LE / BE outputs against these absolute references prevents
  // the parity assertion from passing in lock-step on two equally corrupt
  // decode paths (a vacuous `swap_bytes` construction otherwise lets both
  // `<false>` and `<true>` agree on byte-reversed garbage on a big-endian
  // host).

  /// Reference for `copy_alpha_packed_u16x4_to_u8_at_0`: gather α from
  /// slot 0 of every 4-element u16 tuple, depth-convert `>> 8`, scatter to
  /// slot 3 of the u8 RGBA quad. Untouched slots stay at the input fill.
  fn ref_copy_alpha_packed_u16x4_to_u8_at_0(
    intended: &[u16],
    fill: u8,
    width: usize,
  ) -> std::vec::Vec<u8> {
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = (intended[n * 4] >> 8) as u8;
    }
    out
  }

  /// Reference for `copy_alpha_packed_u16x4_at_0` (u16 output, no depth conv).
  fn ref_copy_alpha_packed_u16x4_at_0(
    intended: &[u16],
    fill: u16,
    width: usize,
  ) -> std::vec::Vec<u16> {
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = intended[n * 4];
    }
    out
  }

  /// Reference for `copy_alpha_plane_u16_to_u8::<BITS, _>`: mask with
  /// `(1 << BITS) - 1`, shift `>> (BITS - 8)`, narrow to u8, scatter to
  /// slot 3 of the u8 RGBA quad.
  fn ref_copy_alpha_plane_u16_to_u8<const BITS: u32>(
    intended: &[u16],
    fill: u8,
    width: usize,
  ) -> std::vec::Vec<u8> {
    let mask: u16 = ((1u32 << BITS) - 1) as u16;
    let shift = BITS - 8;
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = ((intended[n] & mask) >> shift) as u8;
    }
    out
  }

  /// Reference for `copy_alpha_plane_u16::<BITS, _>` (u16 output, masked).
  fn ref_copy_alpha_plane_u16<const BITS: u32>(
    intended: &[u16],
    fill: u16,
    width: usize,
  ) -> std::vec::Vec<u16> {
    let mask: u16 = ((1u32 << BITS) - 1) as u16;
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = intended[n] & mask;
    }
    out
  }

  /// Reference for `copy_alpha_ya_u16_to_u8`: gather α from slot 1 of every
  /// 2-element u16 tuple, depth-convert `>> 8`.
  fn ref_copy_alpha_ya_u16_to_u8(intended: &[u16], fill: u8, width: usize) -> std::vec::Vec<u8> {
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = (intended[n * 2 + 1] >> 8) as u8;
    }
    out
  }

  /// Reference for `copy_alpha_ya_u16` (u16 output, no depth conv).
  fn ref_copy_alpha_ya_u16(intended: &[u16], fill: u16, width: usize) -> std::vec::Vec<u16> {
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = intended[n * 2 + 1];
    }
    out
  }

  /// Reference for `copy_alpha_packed_u16x4_to_u8_at_3`: gather α from slot 3
  /// of every 4-element u16 tuple, depth-convert `>> 8`.
  fn ref_copy_alpha_packed_u16x4_to_u8_at_3(
    intended: &[u16],
    fill: u8,
    width: usize,
  ) -> std::vec::Vec<u8> {
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = (intended[n * 4 + 3] >> 8) as u8;
    }
    out
  }

  /// Reference for `copy_alpha_packed_u16x4_at_3` (u16 output, no depth conv).
  fn ref_copy_alpha_packed_u16x4_at_3(
    intended: &[u16],
    fill: u16,
    width: usize,
  ) -> std::vec::Vec<u16> {
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = intended[n * 4 + 3];
    }
    out
  }

  /// Reference for `copy_alpha_plane_f32_to_u8`: clamp to `[0, 1]`, scale by
  /// 255, round-half-up.
  fn ref_copy_alpha_plane_f32_to_u8(intended: &[f32], fill: u8, width: usize) -> std::vec::Vec<u8> {
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = (intended[n].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    out
  }

  /// Reference for `copy_alpha_plane_f32_to_u16`.
  fn ref_copy_alpha_plane_f32_to_u16(
    intended: &[f32],
    fill: u16,
    width: usize,
  ) -> std::vec::Vec<u16> {
    let mut out = std::vec![fill; width * 4];
    for n in 0..width {
      out[n * 4 + 3] = (intended[n].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    }
    out
  }

  #[test]
  fn copy_alpha_packed_u8x4_at_3_overwrites_only_alpha_slots() {
    let packed = [10, 20, 30, 99, 11, 21, 31, 88, 12, 22, 32, 77];
    let mut rgba = std::vec![1u8; 12];
    copy_alpha_packed_u8x4_at_3(&packed, &mut rgba, 3);
    assert_eq!(rgba, std::vec![1, 1, 1, 99, 1, 1, 1, 88, 1, 1, 1, 77]);
  }

  #[test]
  fn copy_alpha_packed_u16x4_to_u8_at_0_depth_converts_correctly() {
    let packed = as_le_u16(&[0x1234, 100, 200, 300, 0xABCD, 101, 201, 301]);
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_packed_u16x4_to_u8_at_0::<false>(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0x12, 1, 1, 1, 0xAB]);
  }

  #[test]
  fn copy_alpha_packed_u16x4_at_0_preserves_native_u16() {
    let packed = as_le_u16(&[0x1234, 100, 200, 300, 0xABCD, 101, 201, 301]);
    let mut rgba = std::vec![1u16; 8];
    copy_alpha_packed_u16x4_at_0::<false>(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0x1234, 1, 1, 1, 0xABCD]);
  }

  /// BE parity for AYUV64 alpha-at-slot-0 → u8 RGBA. Build LE / BE source
  /// buffers from a single host-native `intended` fixture via
  /// `as_le_u16` / `as_be_u16`, so the kernel's `from_le` / `from_be`
  /// recovers the same logical u16 values on every host. Pin both outputs
  /// against an absolute scalar reference so the parity assertion cannot
  /// pass on two equally corrupt decodes.
  #[test]
  fn copy_alpha_packed_u16x4_to_u8_at_0_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<u16> = std::vec![0x1234, 100, 200, 300, 0xABCD, 101, 201, 301];
    let packed_le = as_le_u16(&intended);
    let packed_be = as_be_u16(&intended);
    let mut rgba_le = std::vec![1u8; 8];
    let mut rgba_be = std::vec![1u8; 8];
    copy_alpha_packed_u16x4_to_u8_at_0::<false>(&packed_le, &mut rgba_le, 2);
    copy_alpha_packed_u16x4_to_u8_at_0::<true>(&packed_be, &mut rgba_be, 2);
    let expected = ref_copy_alpha_packed_u16x4_to_u8_at_0(&intended, 1, 2);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
  }

  /// BE parity for AYUV64 alpha-at-slot-0 → u16 RGBA. Same host-independent
  /// fixture pattern + absolute reference assertion as the u8-output variant.
  #[test]
  fn copy_alpha_packed_u16x4_at_0_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<u16> = std::vec![0x1234, 100, 200, 300, 0xABCD, 101, 201, 301];
    let packed_le = as_le_u16(&intended);
    let packed_be = as_be_u16(&intended);
    let mut rgba_le = std::vec![7u16; 8];
    let mut rgba_be = std::vec![7u16; 8];
    copy_alpha_packed_u16x4_at_0::<false>(&packed_le, &mut rgba_le, 2);
    copy_alpha_packed_u16x4_at_0::<true>(&packed_be, &mut rgba_be, 2);
    let expected = ref_copy_alpha_packed_u16x4_at_0(&intended, 7, 2);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
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
    let alpha = as_le_u16(&[0x3FF, 0x1FF]);
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_plane_u16_to_u8::<10, false>(&alpha, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xFF, 1, 1, 1, 0x7F]);

    // BITS=12
    let alpha = as_le_u16(&[0xFFF, 0x800]);
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_plane_u16_to_u8::<12, false>(&alpha, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xFF, 1, 1, 1, 0x80]);

    // BITS=16
    let alpha = as_le_u16(&[0xFFFF, 0x8000]);
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_plane_u16_to_u8::<16, false>(&alpha, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xFF, 1, 1, 1, 0x80]);
  }

  #[test]
  fn copy_alpha_plane_u16_preserves_native_u16_within_bits_range() {
    // In-range values pass through unchanged.
    let alpha = as_le_u16(&[0x3FF, 0x1FF, 0x000]);
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
    let alpha = as_le_u16(&[0xFFFF, 0x0500, 0x07FF]);
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
    let alpha = as_le_u16(&[0x0500, 0xFFFF, 0x03FF]);
    let mut rgba = std::vec![1u8; 12];
    copy_alpha_plane_u16_to_u8::<10, false>(&alpha, &mut rgba, 3);
    assert_eq!(rgba, std::vec![1, 1, 1, 64, 1, 1, 1, 0xFF, 1, 1, 1, 0xFF]);
  }

  /// BE parity: build LE / BE source buffers from a single host-native
  /// `intended` fixture via `as_le_u16` / `as_be_u16` so each kernel decodes
  /// the same logical u16 values on every host. Pin both outputs against an
  /// absolute scalar reference so the parity assertion cannot pass on two
  /// equally corrupt decodes (a `swap_bytes` host-side construction is
  /// vacuous on a BE host because both flags decode byte-reversed values
  /// that happen to match).
  #[test]
  fn copy_alpha_plane_u16_to_u8_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<u16> = std::vec![0x3FF, 0x1FF, 0x0500, 0xFFFF, 0x07FF, 0x0123];
    let alpha_le = as_le_u16(&intended);
    let alpha_be = as_be_u16(&intended);
    let mut rgba_le = std::vec![1u8; 24];
    let mut rgba_be = std::vec![1u8; 24];
    copy_alpha_plane_u16_to_u8::<10, false>(&alpha_le, &mut rgba_le, 6);
    copy_alpha_plane_u16_to_u8::<10, true>(&alpha_be, &mut rgba_be, 6);
    let expected = ref_copy_alpha_plane_u16_to_u8::<10>(&intended, 1, 6);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
  }

  /// BE parity for the u16-output variant. Host-independent fixture +
  /// absolute reference assertion.
  #[test]
  fn copy_alpha_plane_u16_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<u16> = std::vec![0xFFFF, 0x0500, 0x07FF, 0x0123, 0x3FF, 0x000];
    let alpha_le = as_le_u16(&intended);
    let alpha_be = as_be_u16(&intended);
    let mut rgba_le = std::vec![7u16; 24];
    let mut rgba_be = std::vec![7u16; 24];
    copy_alpha_plane_u16::<10, false>(&alpha_le, &mut rgba_le, 6);
    copy_alpha_plane_u16::<10, true>(&alpha_be, &mut rgba_be, 6);
    let expected = ref_copy_alpha_plane_u16::<10>(&intended, 7, 6);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
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
    let packed = as_le_u16(&[0x1234, 0xABCD, 0x5678, 0xFF00]);
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_ya_u16_to_u8::<false>(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xAB, 1, 1, 1, 0xFF]);
  }

  #[test]
  fn copy_alpha_ya_u16_preserves_native_u16() {
    let packed = as_le_u16(&[0x1234, 0xABCD, 0x5678, 0x9ABC]);
    let mut rgba = std::vec![1u16; 8];
    copy_alpha_ya_u16::<false>(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xABCD, 1, 1, 1, 0x9ABC]);
  }

  /// BE parity for Ya16 → u8 RGBA. Build LE / BE source buffers from a
  /// single host-native `intended` fixture via `as_le_u16` / `as_be_u16`
  /// and pin both outputs against an absolute scalar reference (a
  /// `swap_bytes` construction is vacuous on a BE host because both flags
  /// then decode byte-reversed values that match).
  #[test]
  fn copy_alpha_ya_u16_to_u8_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<u16> = std::vec![0x1234, 0xABCD, 0x5678, 0xFF00, 0x0001, 0x00FF];
    let packed_le = as_le_u16(&intended);
    let packed_be = as_be_u16(&intended);
    let mut rgba_le = std::vec![1u8; 12];
    let mut rgba_be = std::vec![1u8; 12];
    copy_alpha_ya_u16_to_u8::<false>(&packed_le, &mut rgba_le, 3);
    copy_alpha_ya_u16_to_u8::<true>(&packed_be, &mut rgba_be, 3);
    let expected = ref_copy_alpha_ya_u16_to_u8(&intended, 1, 3);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
  }

  /// BE parity for Ya16 → u16 RGBA (16-bit α path). Host-independent fixture
  /// + absolute reference assertion.
  #[test]
  fn copy_alpha_ya_u16_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<u16> = std::vec![0x1234, 0xABCD, 0x5678, 0x9ABC, 0x0001, 0x00FF];
    let packed_le = as_le_u16(&intended);
    let packed_be = as_be_u16(&intended);
    let mut rgba_le = std::vec![7u16; 12];
    let mut rgba_be = std::vec![7u16; 12];
    copy_alpha_ya_u16::<false>(&packed_le, &mut rgba_le, 3);
    copy_alpha_ya_u16::<true>(&packed_be, &mut rgba_be, 3);
    let expected = ref_copy_alpha_ya_u16(&intended, 7, 3);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
  }

  #[test]
  fn copy_alpha_plane_f32_to_u8_clamps_and_scales() {
    // Values [0.0, 0.5, 1.0, 1.5, -0.1] → [0, 128, 255, 255, 0] in slot 3.
    let alpha = as_le_f32(&[0.0f32, 0.5, 1.0, 1.5, -0.1]);
    let mut rgba = vec![1u8; 20];
    copy_alpha_plane_f32_to_u8::<false>(&alpha, &mut rgba, 5);
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
    let alpha = as_le_f32(&[0.0f32, 0.5, 1.0, 1.5, -0.1]);
    let mut rgba = vec![1u16; 20];
    copy_alpha_plane_f32_to_u16::<false>(&alpha, &mut rgba, 5);
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
    let alpha = as_le_f32(&[2.5f32, f32::NAN, f32::INFINITY, -1.0]);
    let mut rgba = vec![0.0f32; 16];
    copy_alpha_plane_f32::<false>(&alpha, &mut rgba, 4);
    assert_eq!(rgba[3], 2.5, "HDR 2.5 preserved");
    assert!(rgba[7].is_nan(), "NaN preserved");
    assert!(rgba[11].is_infinite() && rgba[11] > 0.0, "+Inf preserved");
    assert_eq!(rgba[15], -1.0, "negative preserved");
    // Non-alpha slots untouched (still 0.0).
    assert_eq!(rgba[0], 0.0);
    assert_eq!(rgba[1], 0.0);
    assert_eq!(rgba[2], 0.0);
  }

  /// BE parity for Gbrapf32 → u8 RGBA. Build LE / BE source buffers from a
  /// single host-native `intended` fixture via `as_le_f32` / `as_be_f32` so
  /// each kernel's `u32::from_le` / `u32::from_be` recovers the same logical
  /// bit pattern on every host. Pin both outputs against an absolute scalar
  /// reference (a `to_bits().swap_bytes()` construction is vacuous on a BE
  /// host because both flags decode byte-reversed bits that match after
  /// clamp+scale).
  #[test]
  fn copy_alpha_plane_f32_to_u8_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<f32> = std::vec![0.0f32, 0.25, 0.5, 0.75, 1.0, 1.5, -0.1, 0.123];
    let alpha_le = as_le_f32(&intended);
    let alpha_be = as_be_f32(&intended);
    let mut rgba_le = std::vec![1u8; 32];
    let mut rgba_be = std::vec![1u8; 32];
    copy_alpha_plane_f32_to_u8::<false>(&alpha_le, &mut rgba_le, 8);
    copy_alpha_plane_f32_to_u8::<true>(&alpha_be, &mut rgba_be, 8);
    let expected = ref_copy_alpha_plane_f32_to_u8(&intended, 1, 8);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
  }

  /// BE parity for Gbrapf32 → u16 RGBA. Host-independent fixture + absolute
  /// reference assertion.
  #[test]
  fn copy_alpha_plane_f32_to_u16_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<f32> = std::vec![0.0f32, 0.25, 0.5, 0.75, 1.0, 1.5, -0.1, 0.123];
    let alpha_le = as_le_f32(&intended);
    let alpha_be = as_be_f32(&intended);
    let mut rgba_le = std::vec![7u16; 32];
    let mut rgba_be = std::vec![7u16; 32];
    copy_alpha_plane_f32_to_u16::<false>(&alpha_le, &mut rgba_le, 8);
    copy_alpha_plane_f32_to_u16::<true>(&alpha_be, &mut rgba_be, 8);
    let expected = ref_copy_alpha_plane_f32_to_u16(&intended, 7, 8);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
  }

  /// BE parity for Gbrapf32 → f32 RGBA (lossless α pass-through).
  /// Host-independent fixture: LE / BE source buffers built from a single
  /// `intended` host-native f32 sequence via `as_le_f32` / `as_be_f32`. The
  /// kernel writes the recovered f32 in host-native order, so output α must
  /// equal the host-native bit-pattern of `intended` on every host.
  /// NaN bit-patterns may differ across hardware after a `from_bits → to_bits`
  /// round-trip, so we compare on the bit representation of finite, non-NaN
  /// samples only — pin against an absolute reference, not just LE-vs-BE
  /// parity.
  #[test]
  fn copy_alpha_plane_f32_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<f32> =
      std::vec![0.0f32, 0.25, 0.5, 0.75, 1.0, 2.5, -1.0, f32::INFINITY];
    let alpha_le = as_le_f32(&intended);
    let alpha_be = as_be_f32(&intended);
    let mut rgba_le = std::vec![0.0f32; 32];
    let mut rgba_be = std::vec![0.0f32; 32];
    copy_alpha_plane_f32::<false>(&alpha_le, &mut rgba_le, 8);
    copy_alpha_plane_f32::<true>(&alpha_be, &mut rgba_be, 8);
    let bits_le: std::vec::Vec<u32> = rgba_le.iter().map(|v| v.to_bits()).collect();
    let bits_be: std::vec::Vec<u32> = rgba_be.iter().map(|v| v.to_bits()).collect();
    // Absolute reference: only slot 3 of every 4-tuple is written; the rest
    // remain at the 0.0 fill. Slot 3 must equal the host-native `intended`
    // bit pattern (the kernel writes f32 in host-native byte order).
    let mut expected_bits = std::vec![0u32; 32];
    for (n, v) in intended.iter().enumerate() {
      expected_bits[n * 4 + 3] = v.to_bits();
    }
    assert_eq!(
      bits_le, expected_bits,
      "LE path must match scalar reference"
    );
    assert_eq!(
      bits_be, expected_bits,
      "BE path must match scalar reference"
    );
    assert_eq!(bits_le, bits_be, "BE and LE outputs must agree bit-for-bit");
  }

  // ---- copy_alpha_packed_u16x4_to_u8_at_3 / copy_alpha_packed_u16x4_at_3 --

  /// Alpha at slot 3 is depth-converted >> 8 and written to rgba_out[3 + 4*n].
  #[test]
  fn copy_alpha_packed_u16x4_to_u8_at_3_narrows_correctly() {
    let packed = as_le_u16(&[100, 200, 300, 0xABFF, 101, 201, 301, 0x1234]);
    let mut rgba = std::vec![1u8; 8];
    copy_alpha_packed_u16x4_to_u8_at_3::<false>(&packed, &mut rgba, 2);
    assert_eq!(rgba, std::vec![1, 1, 1, 0xAB, 1, 1, 1, 0x12]);
  }

  /// Alpha at slot 3 is copied verbatim (no depth conversion).
  #[test]
  fn copy_alpha_packed_u16x4_at_3_copies_verbatim() {
    let packed = as_le_u16(&[100, 200, 300, 0xABFF, 101, 201, 301, 0x1234]);
    let mut rgba_u16 = std::vec![1u16; 8];
    copy_alpha_packed_u16x4_at_3::<false>(&packed, &mut rgba_u16, 2);
    assert_eq!(rgba_u16, std::vec![1, 1, 1, 0xABFF, 1, 1, 1, 0x1234]);
  }

  /// Only the alpha slot (index 3) is overwritten; RGB slots [0..3] are untouched.
  #[test]
  fn copy_alpha_packed_u16x4_to_u8_at_3_touches_only_alpha_slot() {
    let packed = as_le_u16(&[0, 0, 0, 0xFFFF]);
    let mut rgba = std::vec![42u8; 4];
    copy_alpha_packed_u16x4_to_u8_at_3::<false>(&packed, &mut rgba, 1);
    assert_eq!(rgba[..3], [42, 42, 42]);
    assert_eq!(rgba[3], 0xFF);
  }

  /// Only the alpha slot (index 3) is overwritten; RGB slots [0..3] are untouched.
  #[test]
  fn copy_alpha_packed_u16x4_at_3_touches_only_alpha_slot() {
    let packed = as_le_u16(&[0, 0, 0, 0xBEEF]);
    let mut rgba_u16 = std::vec![99u16; 4];
    copy_alpha_packed_u16x4_at_3::<false>(&packed, &mut rgba_u16, 1);
    assert_eq!(rgba_u16[..3], [99, 99, 99]);
    assert_eq!(rgba_u16[3], 0xBEEF);
  }

  /// BE parity for Rgba64 / Bgra64 alpha-at-slot-3 → u8 RGBA.
  /// Host-independent fixture + absolute reference assertion.
  #[test]
  fn copy_alpha_packed_u16x4_to_u8_at_3_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<u16> = std::vec![100, 200, 300, 0xABFF, 101, 201, 301, 0x1234];
    let packed_le = as_le_u16(&intended);
    let packed_be = as_be_u16(&intended);
    let mut rgba_le = std::vec![1u8; 8];
    let mut rgba_be = std::vec![1u8; 8];
    copy_alpha_packed_u16x4_to_u8_at_3::<false>(&packed_le, &mut rgba_le, 2);
    copy_alpha_packed_u16x4_to_u8_at_3::<true>(&packed_be, &mut rgba_be, 2);
    let expected = ref_copy_alpha_packed_u16x4_to_u8_at_3(&intended, 1, 2);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
  }

  /// BE parity for Rgba64 / Bgra64 alpha-at-slot-3 → u16 RGBA.
  /// Host-independent fixture + absolute reference assertion.
  #[test]
  fn copy_alpha_packed_u16x4_at_3_be_parity_with_swapped_buffer() {
    let intended: std::vec::Vec<u16> = std::vec![100, 200, 300, 0xABFF, 101, 201, 301, 0x1234];
    let packed_le = as_le_u16(&intended);
    let packed_be = as_be_u16(&intended);
    let mut rgba_le = std::vec![7u16; 8];
    let mut rgba_be = std::vec![7u16; 8];
    copy_alpha_packed_u16x4_at_3::<false>(&packed_le, &mut rgba_le, 2);
    copy_alpha_packed_u16x4_at_3::<true>(&packed_be, &mut rgba_be, 2);
    let expected = ref_copy_alpha_packed_u16x4_at_3(&intended, 7, 2);
    assert_eq!(rgba_le, expected, "LE path must match scalar reference");
    assert_eq!(rgba_be, expected, "BE path must match scalar reference");
    assert_eq!(rgba_le, rgba_be, "BE and LE outputs must agree");
  }
}
