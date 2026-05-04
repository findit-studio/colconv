//! NEON α-extract helpers — SIMD parity of `crate::row::scalar::alpha_extract`.
//!
//! Each fn matches its scalar counterpart byte-for-byte (verified by
//! `*_matches_scalar_widths` tests in this file).
//!
//! # Strategy
//!
//! Every helper writes **only** the α slot of `rgba_out` — slots 0/1/2
//! must be preserved. The natural NEON pattern is:
//!
//! 1. Deinterleave the destination via `vld4q_u8` / `vld4q_u16` /
//!    `vld4_u8` (lane-wise structured load) into a `*x4_t` tuple.
//! 2. Replace the `.3` (or `.0` for AYUV64) field with the α vector
//!    sourced from the input.
//! 3. Re-interleave and store via the matching `vst4*` intrinsic.
//!
//! This keeps the non-α channels round-trip identical to the input
//! buffer (within the structured-load semantics) and avoids any
//! gather/scatter or per-pixel masking. The cost is a load + store
//! per block, which still beats the scalar element-by-element write.
//!
//! # BITS dispatch in `copy_alpha_plane_u16_to_u8`
//!
//! `vshrn_n_u16` requires a literal const-generic shift, but the per-
//! call shift is `BITS - 8`, which is not a stable const expression on
//! a const generic in current Rust. The crate-wide convention (see
//! `yuv_planar_high_bit.rs` and `subsampled_high_bit_pn_4_4_4.rs`) is
//! to use `vshlq_u16` with a *negative* count vector built once outside
//! the loop. NEON treats a negative shift as a logical right shift, so
//! this matches the `vshrn_n_u16` semantics without needing per-BITS
//! monomorphization or a match dispatch. The narrow step uses
//! `vmovn_u16` (truncating) — not `vqmovn_u16` (saturating) — to match
//! the scalar `as u8` truncation byte-for-byte at over-range inputs.

#![allow(dead_code)]

use core::arch::aarch64::*;

use crate::row::scalar::alpha_extract as scalar;

/// VUYA → u8 RGBA: gather α from `packed[3 + 4*n]` into `rgba_out[3 + 4*n]`.
///
/// VUYA layout: `[V(8), U(8), Y(8), A(8)]` per pixel. The α byte is
/// already at offset 3 in the packed buffer, so we deinterleave both
/// `packed` and `rgba_out` with `vld4q_u8` and rebuild `rgba_out` with
/// `packed.3` substituted into the α slot.
///
/// Block size: 16 px / iter (one `vld4q_u8` = 64 bytes).
pub(crate) fn copy_alpha_packed_u8x4_at_3(packed: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  let mut x = 0usize;
  // SAFETY: bounds verified by the loop guard `x + 16 <= width` and the
  // debug_asserts above; both buffers are u8 so any byte alignment works.
  // NEON intrinsics are unsafe by definition in std, but they touch only
  // the slices' valid ranges.
  unsafe {
    while x + 16 <= width {
      let off = x * 4;
      let src = vld4q_u8(packed.as_ptr().add(off));
      let dst = vld4q_u8(rgba_out.as_ptr().add(off));
      let merged = uint8x16x4_t(dst.0, dst.1, dst.2, src.3);
      vst4q_u8(rgba_out.as_mut_ptr().add(off), merged);
      x += 16;
    }
  }

  if x < width {
    scalar::copy_alpha_packed_u8x4_at_3(
      &packed[x * 4..width * 4],
      &mut rgba_out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AYUV64 → u8 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> 8`.
///
/// AYUV64 layout: `[A(16), Y(16), U(16), V(16)]`. We use `vld4q_u16`
/// to pull 8 px × 4 ch in 64-bit lanes, take `.0` as the α u16 vector,
/// narrow to u8 via `vshrn_n_u16::<8>`, then load the 8-px RGBA via
/// the 64-bit `vld4_u8` (32 bytes), substitute `.3`, and store back
/// via `vst4_u8`.
///
/// Block size: 8 px / iter (one `vld4q_u16` for src = 64 bytes,
/// one `vld4_u8` / `vst4_u8` for dst = 32 bytes).
pub(crate) fn copy_alpha_packed_u16x4_to_u8_at_0(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  let mut x = 0usize;
  // SAFETY: loop guard `x + 8 <= width` plus the debug_asserts above
  // guarantee the 8-px reads/writes stay in bounds for both buffers.
  unsafe {
    while x + 8 <= width {
      let src_off = x * 4;
      let dst_off = x * 4;
      let src = vld4q_u16(packed.as_ptr().add(src_off));
      // Narrow α (u16) to u8 by taking the high byte (>> 8).
      let a_u8 = vshrn_n_u16::<8>(src.0);
      let dst = vld4_u8(rgba_out.as_ptr().add(dst_off));
      let merged = uint8x8x4_t(dst.0, dst.1, dst.2, a_u8);
      vst4_u8(rgba_out.as_mut_ptr().add(dst_off), merged);
      x += 8;
    }
  }

  if x < width {
    scalar::copy_alpha_packed_u16x4_to_u8_at_0(
      &packed[x * 4..width * 4],
      &mut rgba_out[x * 4..width * 4],
      width - x,
    );
  }
}

/// AYUV64 → u16 RGBA: gather α from `packed[0 + 4*n]` (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion.
///
/// Block size: 8 px / iter (`vld4q_u16` × 2 = 128 bytes round-trip).
pub(crate) fn copy_alpha_packed_u16x4_at_0(packed: &[u16], rgba_out: &mut [u16], width: usize) {
  debug_assert!(packed.len() >= width * 4, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  let mut x = 0usize;
  // SAFETY: loop guard `x + 8 <= width` plus debug_asserts above
  // guarantee both 8-px reads/writes stay in bounds.
  unsafe {
    while x + 8 <= width {
      let off = x * 4;
      let src = vld4q_u16(packed.as_ptr().add(off));
      let dst = vld4q_u16(rgba_out.as_ptr().add(off));
      // RGBA layout is (.0=R, .1=G, .2=B, .3=A); replace .3 with packed.0 (=A).
      let merged = uint16x8x4_t(dst.0, dst.1, dst.2, src.0);
      vst4q_u16(rgba_out.as_mut_ptr().add(off), merged);
      x += 8;
    }
  }

  if x < width {
    scalar::copy_alpha_packed_u16x4_at_0(
      &packed[x * 4..width * 4],
      &mut rgba_out[x * 4..width * 4],
      width - x,
    );
  }
}

/// Yuva420p / 422p / 444p u8 → u8 RGBA: scatter α plane into
/// `rgba_out[3 + 4*n]`.
///
/// Block size: 16 px / iter. `vld1q_u8` loads 16 contiguous α bytes;
/// `vld4q_u8` deinterleaves 16 px of RGBA; we substitute `.3` and
/// `vst4q_u8` writes back.
pub(crate) fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");

  let mut x = 0usize;
  // SAFETY: loop guard `x + 16 <= width` and debug_asserts ensure both
  // the 16-byte α load and the 64-byte RGBA round-trip stay in bounds.
  unsafe {
    while x + 16 <= width {
      let a_vec = vld1q_u8(alpha.as_ptr().add(x));
      let off = x * 4;
      let dst = vld4q_u8(rgba_out.as_ptr().add(off));
      let merged = uint8x16x4_t(dst.0, dst.1, dst.2, a_vec);
      vst4q_u8(rgba_out.as_mut_ptr().add(off), merged);
      x += 16;
    }
  }

  if x < width {
    scalar::copy_alpha_plane_u8(&alpha[x..width], &mut rgba_out[x * 4..width * 4], width - x);
  }
}

/// Yuva*p9/10/12/14 → u8 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u8) with depth-conv `>> (BITS - 8)`.
///
/// `vshrn_n_u16` requires a literal const shift, so we use the crate's
/// established pattern: `vshlq_u16` with a negative-count splat acts
/// as a logical right shift by `BITS - 8`. Then `vmovn_u16` truncates
/// to u8 — matching scalar `as u8` for over-range inputs (saturating
/// `vqmovn_u16` would diverge there).
///
/// Block size: 8 px / iter.
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

  let mut x = 0usize;
  // SAFETY: loop guard `x + 8 <= width` plus debug_asserts; the negative
  // shift count is loop-invariant and built before the loop.
  unsafe {
    let shr_count = vdupq_n_s16(-((BITS as i16) - 8));
    while x + 8 <= width {
      let a_u16 = vld1q_u16(alpha.as_ptr().add(x));
      // Right shift by (BITS - 8) via vshlq_u16 with negative count.
      let a_shifted = vshlq_u16(a_u16, shr_count);
      // Truncating narrow to u8 — matches scalar `as u8` (no saturation).
      let a_u8 = vmovn_u16(a_shifted);
      let off = x * 4;
      let dst = vld4_u8(rgba_out.as_ptr().add(off));
      let merged = uint8x8x4_t(dst.0, dst.1, dst.2, a_u8);
      vst4_u8(rgba_out.as_mut_ptr().add(off), merged);
      x += 8;
    }
  }

  if x < width {
    scalar::copy_alpha_plane_u16_to_u8::<BITS>(
      &alpha[x..width],
      &mut rgba_out[x * 4..width * 4],
      width - x,
    );
  }
}

/// Yuva*p9/10/12/14/16 → u16 RGBA: scatter α plane (u16) into
/// `rgba_out[3 + 4*n]` (u16). No depth conversion. `BITS` is
/// informational at the SIMD layer (preserved for monomorphization
/// symmetry with the scalar API).
///
/// Block size: 8 px / iter.
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
  // BITS is informational — no shift applied (matches scalar).
  let _ = BITS;

  let mut x = 0usize;
  // SAFETY: loop guard `x + 8 <= width` plus debug_asserts cover both
  // the 16-byte α load and the 64-byte RGBA round-trip.
  unsafe {
    while x + 8 <= width {
      let a_vec = vld1q_u16(alpha.as_ptr().add(x));
      let off = x * 4;
      let dst = vld4q_u16(rgba_out.as_ptr().add(off));
      let merged = uint16x8x4_t(dst.0, dst.1, dst.2, a_vec);
      vst4q_u16(rgba_out.as_mut_ptr().add(off), merged);
      x += 8;
    }
  }

  if x < width {
    scalar::copy_alpha_plane_u16::<BITS>(
      &alpha[x..width],
      &mut rgba_out[x * 4..width * 4],
      width - x,
    );
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use crate::row::scalar::alpha_extract as scalar;

  fn pseudo_random_u8(out: &mut [u8], seed: u32) {
    let mut state = seed;
    for v in out.iter_mut() {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (state >> 16) as u8;
    }
  }

  fn pseudo_random_u16(out: &mut [u16], seed: u32) {
    let mut state = seed;
    for v in out.iter_mut() {
      state = state.wrapping_mul(1664525).wrapping_add(1013904223);
      *v = (state >> 8) as u16;
    }
  }

  // Cover both 8-px and 16-px main loops + tail.
  const WIDTHS: &[usize] = &[1, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 128, 130];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_copy_alpha_packed_u8x4_at_3_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut packed = std::vec![0u8; w * 4];
      pseudo_random_u8(&mut packed, 0xC0FFEE);
      let mut rgba_simd = std::vec![1u8; w * 4];
      let mut rgba_scalar = std::vec![1u8; w * 4];
      // Seed RGB slots with a non-1 pattern so we'd notice if SIMD
      // accidentally clobbered non-α bytes.
      pseudo_random_u8(&mut rgba_simd, 0xDECAF);
      rgba_scalar.copy_from_slice(&rgba_simd);
      super::copy_alpha_packed_u8x4_at_3(&packed, &mut rgba_simd, w);
      scalar::copy_alpha_packed_u8x4_at_3(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_copy_alpha_packed_u16x4_to_u8_at_0_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut packed, 0xCAB00D);
      let mut rgba_simd = std::vec![1u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0xFEED);
      let mut rgba_scalar = rgba_simd.clone();
      super::copy_alpha_packed_u16x4_to_u8_at_0(&packed, &mut rgba_simd, w);
      scalar::copy_alpha_packed_u16x4_to_u8_at_0(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_copy_alpha_packed_u16x4_at_0_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut packed = std::vec![0u16; w * 4];
      pseudo_random_u16(&mut packed, 0xBEEF11);
      let mut rgba_simd = std::vec![1u16; w * 4];
      pseudo_random_u16(&mut rgba_simd, 0x1337);
      let mut rgba_scalar = rgba_simd.clone();
      super::copy_alpha_packed_u16x4_at_0(&packed, &mut rgba_simd, w);
      scalar::copy_alpha_packed_u16x4_at_0(&packed, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_copy_alpha_plane_u8_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut alpha = std::vec![0u8; w];
      pseudo_random_u8(&mut alpha, 0xABCDEF);
      let mut rgba_simd = std::vec![1u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0x123456);
      let mut rgba_scalar = rgba_simd.clone();
      super::copy_alpha_plane_u8(&alpha, &mut rgba_simd, w);
      scalar::copy_alpha_plane_u8(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits10() {
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xC0DE);
      // Mask to 10-bit input range to simulate well-formed sources.
      for v in alpha.iter_mut() {
        *v &= 0x03FF;
      }
      let mut rgba_simd = std::vec![1u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0xBABE);
      let mut rgba_scalar = rgba_simd.clone();
      super::copy_alpha_plane_u16_to_u8::<10>(&alpha, &mut rgba_simd, w);
      scalar::copy_alpha_plane_u16_to_u8::<10>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_copy_alpha_plane_u16_to_u8_matches_scalar_widths_bits12() {
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xF00BAA);
      for v in alpha.iter_mut() {
        *v &= 0x0FFF;
      }
      let mut rgba_simd = std::vec![1u8; w * 4];
      pseudo_random_u8(&mut rgba_simd, 0x5EED);
      let mut rgba_scalar = rgba_simd.clone();
      super::copy_alpha_plane_u16_to_u8::<12>(&alpha, &mut rgba_simd, w);
      scalar::copy_alpha_plane_u16_to_u8::<12>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn neon_copy_alpha_plane_u16_matches_scalar_widths() {
    for &w in WIDTHS {
      let mut alpha = std::vec![0u16; w];
      pseudo_random_u16(&mut alpha, 0xDEADBE);
      let mut rgba_simd = std::vec![1u16; w * 4];
      pseudo_random_u16(&mut rgba_simd, 0xFADE);
      let mut rgba_scalar = rgba_simd.clone();
      super::copy_alpha_plane_u16::<10>(&alpha, &mut rgba_simd, w);
      scalar::copy_alpha_plane_u16::<10>(&alpha, &mut rgba_scalar, w);
      assert_eq!(rgba_simd, rgba_scalar, "width={w}");
    }
  }
}
