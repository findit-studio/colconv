//! aarch64 NEON backend for the row primitives.
//!
//! Selected by [`crate::row`]'s dispatcher after
//! `is_aarch64_feature_detected!("neon")` returns true (runtime,
//! std‑gated) or `cfg!(target_feature = "neon")` evaluates true
//! (compile‑time, no‑std). The kernel itself carries
//! `#[target_feature(enable = "neon")]` so its intrinsics execute in
//! an explicitly NEON‑enabled context rather than one merely inherited
//! from the aarch64 target's default feature set.
//!
//! # Numerical contract
//!
//! The kernel uses i32 widening multiplies and the same
//! `(prod + (1 << 14)) >> 15` Q15 rounding as
//! [`crate::row::scalar::yuv_420_to_rgb_row`], so output is
//! **byte‑identical** to the scalar reference for every input. This is
//! asserted by the equivalence tests below.
//!
//! # Pipeline (per 16 Y pixels / 8 chroma samples)
//!
//! 1. Load 16 Y (`vld1q_u8`) + 8 U (`vld1_u8`) + 8 V (`vld1_u8`).
//! 2. Widen U/V to i16, subtract 128 → `u_i16`, `v_i16`.
//! 3. Widen to i32 and apply `c_scale` (Q15) → `u_d`, `v_d` (i32x4 × 2).
//! 4. Per channel C ∈ {R, G, B}:
//!    `C_chroma = (C_u * u_d + C_v * v_d + RND) >> 15` in i32,
//!    narrow‑saturate to i16x8 (8 lanes = 8 chroma pairs).
//! 5. Duplicate each chroma lane into its Y‑pair slot with
//!    `vzip1q_s16` / `vzip2q_s16` → 16 i16 chroma lanes matching the
//!    16 Y lanes (nearest‑neighbor upsample in registers, no memory
//!    traffic).
//! 6. Y path: `(Y - y_off) * y_scale + RND >> 15` in i32, narrow to i16.
//! 7. Saturating add Y + chroma per channel → i16x16.
//! 8. Saturate‑narrow to u8x16 and interleave with `vst3q_u8`.

use core::arch::aarch64::*;

#[allow(unused_imports)]
pub(super) use crate::{ColorMatrix, row::scalar};

pub(crate) mod alpha_extract;
mod ayuv64;
pub(crate) mod endian;
mod gray;
mod hsv;
pub(crate) mod legacy_rgb;
pub(crate) mod mono1bit;
mod packed_rgb;
mod packed_rgb_16bit;
mod packed_rgb_float;
mod packed_yuv_8bit;
pub(crate) mod pal8;
mod planar_gbr;
mod planar_gbr_float;
mod planar_gbr_high_bit;
mod semi_planar_8bit;
mod subsampled_high_bit_pn_4_2_0;
mod subsampled_high_bit_pn_4_4_4;
mod v210;
mod v30x;
mod v410;
mod vuya;
mod xv36;
mod y216;
mod y2xx;
pub(crate) mod y_plane_to_luma_u16;
mod yuv_planar_16bit;
mod yuv_planar_8bit;
mod yuv_planar_high_bit;

pub(crate) use alpha_extract::*;
pub(crate) use ayuv64::*;
pub(crate) use gray::*;
pub(crate) use hsv::*;
pub(crate) use packed_rgb::*;
#[allow(unused_imports)]
pub(crate) use packed_rgb_16bit::*;
pub(crate) use packed_rgb_float::*;
pub(crate) use packed_yuv_8bit::*;
pub(crate) use planar_gbr::*;
pub(crate) use planar_gbr_float::*;
pub(crate) use planar_gbr_high_bit::*;
pub(crate) use semi_planar_8bit::*;
pub(crate) use subsampled_high_bit_pn_4_2_0::*;
pub(crate) use subsampled_high_bit_pn_4_4_4::*;
pub(crate) use v30x::*;
pub(crate) use v210::*;
pub(crate) use v410::*;
pub(crate) use vuya::*;
pub(crate) use xv36::*;
pub(crate) use y_plane_to_luma_u16::*;
pub(crate) use y2xx::*;
pub(crate) use y216::*;
pub(crate) use yuv_planar_8bit::*;
pub(crate) use yuv_planar_16bit::*;
pub(crate) use yuv_planar_high_bit::*;

// ---- Shared helpers (used across submodules) -------------------------

/// Clamps an i16x8 vector to `[0, max]` and reinterprets to u16x8.
/// Used by native-depth u16 output paths (10/12/14 bit) to avoid
/// `vqmovun_s16`'s u8 saturation.
#[inline(always)]
pub(super) fn clamp_u16_max(v: int16x8_t, zero_v: int16x8_t, max_v: int16x8_t) -> uint16x8_t {
  unsafe { vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(v, zero_v), max_v)) }
}

// The helpers below wrap NEON register‑only intrinsics (shifts, adds,
// multiplies, narrowing conversions, lane movers). None of them touch
// memory or take pointers, so there is no safety invariant to hoist to
// the caller — the functions themselves are safe. The `unsafe { ... }`
// blocks inside are only required because `core::arch::aarch64`
// intrinsics are marked `unsafe fn` in the standard library.
//
// `#[inline(always)]` guarantees these are inlined into the NEON‑
// enabled caller (`yuv_420_to_rgb_row` has
// `#[target_feature(enable = "neon")]`), so the intrinsics execute in
// a context where NEON is explicitly enabled — not just implicitly
// via the aarch64 target's default feature set.

/// `>>_a 15` shift (arithmetic, sign‑extending).
#[inline(always)]
pub(super) fn q15_shift(v: int32x4_t) -> int32x4_t {
  unsafe { vshrq_n_s32::<15>(v) }
}

/// Build an i16x8 channel chroma vector from the 8 paired i32 chroma
/// samples. Mirrors the scalar
/// `(coeff_u * u_d + coeff_v * v_d + RND) >> 15`.
#[inline(always)]
pub(super) fn chroma_i16x8(
  cu: int32x4_t,
  cv: int32x4_t,
  u_d_lo: int32x4_t,
  v_d_lo: int32x4_t,
  u_d_hi: int32x4_t,
  v_d_hi: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_lo), vmulq_s32(cv, v_d_lo)),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vaddq_s32(vmulq_s32(cu, u_d_hi), vmulq_s32(cv, v_d_hi)),
      rnd,
    ));
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
  }
}

/// `(Y - y_off) * y_scale + RND >> 15` returned as i16x8 (8 Y pixels).
#[inline(always)]
pub(super) fn scale_y(
  y_i16: int16x8_t,
  y_off_v: int16x8_t,
  y_scale_v: int32x4_t,
  rnd: int32x4_t,
) -> int16x8_t {
  unsafe {
    let shifted = vsubq_s16(y_i16, y_off_v);
    let lo = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_low_s16(shifted)), y_scale_v),
      rnd,
    ));
    let hi = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vmovl_s16(vget_high_s16(shifted)), y_scale_v),
      rnd,
    ));
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
  }
}

// ===== 16-bit helpers =====================================================

/// Scale 8 u16 Y pixels to i16x8 for the 16-bit u8-output path.
///
/// Unsigned-widens via `vmovl_u16`, subtracts `y_off` in i32, multiplies
/// by `y_scale` (small for u8 output — no i32 overflow), Q15-shifts, and
/// narrows to i16x8 with `vqmovn_s32`.
#[inline(always)]
pub(super) fn scale_y_u16_to_i16(
  y_vec: uint16x8_t,
  y_off_v: int32x4_t,
  y_scale_v: int32x4_t,
  rnd_v: int32x4_t,
) -> int16x8_t {
  unsafe {
    let lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(y_vec)));
    let hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(y_vec)));
    let lo_s = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vsubq_s32(lo, y_off_v), y_scale_v),
      rnd_v,
    ));
    let hi_s = vshrq_n_s32::<15>(vaddq_s32(
      vmulq_s32(vsubq_s32(hi, y_off_v), y_scale_v),
      rnd_v,
    ));
    vcombine_s16(vqmovn_s32(lo_s), vqmovn_s32(hi_s))
  }
}

/// `(cu*u_d + cv*v_d + RND) >> 15` in i64 for 4 chroma values → i32x4.
///
/// Used by the 16-bit u16-output path where `coeff * u_d` exceeds i32.
/// `vmull_s32` widens each 32×32 product to 64 bits, avoiding overflow.
#[inline(always)]
pub(super) fn chroma_i64x4(
  cu: int32x4_t,
  cv: int32x4_t,
  u_d: int32x4_t,
  v_d: int32x4_t,
  rnd64: int64x2_t,
) -> int32x4_t {
  unsafe {
    let sum_lo = vshrq_n_s64::<15>(vaddq_s64(
      vaddq_s64(
        vmull_s32(vget_low_s32(cu), vget_low_s32(u_d)),
        vmull_s32(vget_low_s32(cv), vget_low_s32(v_d)),
      ),
      rnd64,
    ));
    let sum_hi = vshrq_n_s64::<15>(vaddq_s64(
      vaddq_s64(
        vmull_s32(vget_high_s32(cu), vget_high_s32(u_d)),
        vmull_s32(vget_high_s32(cv), vget_high_s32(v_d)),
      ),
      rnd64,
    ));
    vcombine_s32(vmovn_s64(sum_lo), vmovn_s64(sum_hi))
  }
}

/// Scale 4 u16 Y pixels via i64 widening for the 16-bit u16-output path.
///
/// `(y - y_off) * y_scale` can reach ~2.35×10⁹ at 16-bit limited range,
/// overflowing i32. `vmull_s32` widens to i64 before the Q15 shift.
/// Input `y_u32` is already unsigned-widened and reinterpreted as i32.
#[inline(always)]
pub(super) fn scale_y_u16_i64(
  y_i32: int32x4_t,
  y_off_v: int32x4_t,
  y_scale_d: int32x2_t,
  rnd64: int64x2_t,
) -> int32x4_t {
  unsafe {
    let sub = vsubq_s32(y_i32, y_off_v);
    let lo = vshrq_n_s64::<15>(vaddq_s64(vmull_s32(vget_low_s32(sub), y_scale_d), rnd64));
    let hi = vshrq_n_s64::<15>(vaddq_s64(vmull_s32(vget_high_s32(sub), y_scale_d), rnd64));
    vcombine_s32(vmovn_s64(lo), vmovn_s64(hi))
  }
}

// ---- BE helpers ----------------------------------------------------------

/// Compile-time host endianness. `true` on BE targets (e.g. `s390x`,
/// `powerpc`-BE), `false` on LE targets (e.g. `aarch64-apple-darwin`,
/// `x86_64`).
///
/// Used by the conditional byte-swap helpers below to decide whether a raw
/// NEON load already matches the wire endian. Without this, the helpers
/// would only correctly handle two of the four `host × wire` quadrants.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Conditionally byte-swap 8 u16 lanes in a NEON register so that the
/// returned value is in **host-native** byte order, regardless of the
/// host endianness.
///
/// The gate is `BE != HOST_NATIVE_BE`:
///
/// | wire `BE` | host       | gate    | action            |
/// |-----------|------------|---------|-------------------|
/// | `false`   | LE         | `false` | no swap (LE→LE)   |
/// | `false`   | BE         | `true`  | swap (LE→BE)      |
/// | `true`    | LE         | `true`  | swap (BE→LE)      |
/// | `true`    | BE         | `false` | no swap (BE→BE)   |
///
/// The unused branch is eliminated by the compiler — `BE` and
/// `HOST_NATIVE_BE` are both compile-time constants, so the gate folds.
///
/// Used by the packed YUV 4:4:4 kernels (XV36, AYUV64) after `vld4q_u16`
/// to correct samples loaded from a wire-encoded buffer.
///
/// Mirrors PR #82's `9c7d533` dispatcher routing fix and PR #85's
/// `9e678b0` Ya16 SIMD gate — both addressed the same bug class
/// (only swapping on `BE = true` rather than `BE != HOST_NATIVE_BE`).
#[inline(always)]
pub(super) unsafe fn bswap_u16x8_if_be<const BE: bool>(v: uint16x8_t) -> uint16x8_t {
  if BE != HOST_NATIVE_BE {
    unsafe { vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v))) }
  } else {
    v
  }
}

/// Conditionally byte-swap 4 u32 lanes in a NEON register so that the
/// returned value is in **host-native** byte order, regardless of the
/// host endianness.
///
/// Same `BE != HOST_NATIVE_BE` gate as [`bswap_u16x8_if_be`] — see that
/// helper for the truth table.
///
/// Used by the V410 kernel after `vld1q_u32` to correct u32 words loaded
/// from a wire-encoded buffer.
#[inline(always)]
pub(super) unsafe fn bswap_u32x4_if_be<const BE: bool>(v: uint32x4_t) -> uint32x4_t {
  if BE != HOST_NATIVE_BE {
    unsafe { vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(v))) }
  } else {
    v
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
