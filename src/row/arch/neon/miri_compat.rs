//! `cfg(miri)` fallbacks for the specialized NEON intrinsics Miri cannot
//! execute.
//!
//! Miri shims the common load/store/lane NEON intrinsics but does **not**
//! provide the LLVM foreign functions behind the horizontal-reduce
//! (`vaddvq_*`), float→int convert (`vcvt{a,n}q_u32_f32`) and
//! saturating-narrow (`vqmov{u}n_*`) families — calling them under Miri
//! aborts with `unsupported operation: can't call foreign function
//! llvm.aarch64.neon.*`. The feature-sharded Miri matrix (#253) exercises
//! every format kernel, so these all need a Miri-only path.
//!
//! Each `*_compat` helper has two branches:
//!
//! * `#[cfg(not(miri))]` — the **real** intrinsic. Production is
//!   byte-identical to calling the intrinsic directly.
//! * `#[cfg(miri)]` — a **bit-exact** scalar emulation built only from
//!   load/store/lane intrinsics Miri *does* shim (`vst1q_*`, `vld1_*`,
//!   `vld1q_*`). It runs under Miri only, but must reproduce the
//!   intrinsic's exact semantics or the SIMD-vs-scalar parity tests fail
//!   under Miri.
//!
//! The emulations mirror the ARM semantics precisely:
//! `vaddvq_u32`/`vaddvq_u64` are modular (wrapping) horizontal adds;
//! `vcvtnq_u32_f32` is round-to-nearest-**even** (FCVTNU),
//! `vcvtaq_u32_f32` is round-to-nearest-**ties-away** (FCVTAU), both with
//! the saturating unsigned cast Rust's `f32 as u32` already performs
//! (NaN→0, <0→0, ≥2³²→u32::MAX); and the `vqmov{u}n_*` family clamps to
//! the destination range before the narrowing cast.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

/// Horizontal add of a `float64x2_t` — `lane0 + lane1`.
///
/// Miri does not shim the `llvm.aarch64.neon.faddv.f64` foreign function
/// behind `vaddvq_f64`, so under Miri this stores the two lanes and sums
/// them in the same lane order (`vst1q_f64` is shimmed). The result is the
/// exact `f64` sum of the two lanes either way, so the real SIMD build is
/// byte-identical to the original `vaddvq_f64`.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vaddvq_f64_compat(v: float64x2_t) -> f64 {
  #[cfg(not(miri))]
  {
    vaddvq_f64(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0f64; 2];
    // SAFETY: `a` holds two `f64`; `vst1q_f64` writes exactly two lanes.
    unsafe { vst1q_f64(a.as_mut_ptr(), v) };
    a[0] + a[1]
  }
}

/// Horizontal add of a `uint32x4_t` — `lane0 + lane1 + lane2 + lane3`,
/// **modular** (`u32` wrapping), matching ARM `ADDV`/`UADDLV`-free
/// `vaddvq_u32` (it truncates to 32 bits).
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vaddvq_u32_compat(v: uint32x4_t) -> u32 {
  #[cfg(not(miri))]
  {
    vaddvq_u32(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0u32; 4];
    // SAFETY: `a` holds four `u32`; `vst1q_u32` writes exactly four lanes.
    unsafe { vst1q_u32(a.as_mut_ptr(), v) };
    a[0]
      .wrapping_add(a[1])
      .wrapping_add(a[2])
      .wrapping_add(a[3])
  }
}

/// Horizontal add of a `uint64x2_t` — `lane0 + lane1`, **modular**
/// (`u64` wrapping), matching ARM `vaddvq_u64`.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vaddvq_u64_compat(v: uint64x2_t) -> u64 {
  #[cfg(not(miri))]
  {
    vaddvq_u64(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0u64; 2];
    // SAFETY: `a` holds two `u64`; `vst1q_u64` writes exactly two lanes.
    unsafe { vst1q_u64(a.as_mut_ptr(), v) };
    a[0].wrapping_add(a[1])
  }
}

/// IEEE-754 `maximum` of one lane pair — the exact ARM `FMAX` semantics:
/// NaN-propagating (NaN if either operand is NaN) and `+0.0 > -0.0`. This
/// differs from `f32::max`, which suppresses NaN and ignores zero sign, so
/// it is **not** interchangeable with it.
#[cfg(miri)]
#[inline]
fn fmax_lane(a: f32, b: f32) -> f32 {
  if a.is_nan() || b.is_nan() {
    f32::NAN
  } else if a == b {
    // Equal magnitudes incl. `+0.0`/`-0.0`: FMAX yields the positively
    // signed value (`+0.0`).
    if a.is_sign_positive() { a } else { b }
  } else if a > b {
    a
  } else {
    b
  }
}

/// IEEE-754 `minimum` of one lane pair — the exact ARM `FMIN` semantics:
/// NaN-propagating and `-0.0 < +0.0`. Not interchangeable with `f32::min`.
#[cfg(miri)]
#[inline]
fn fmin_lane(a: f32, b: f32) -> f32 {
  if a.is_nan() || b.is_nan() {
    f32::NAN
  } else if a == b {
    // Equal magnitudes incl. `+0.0`/`-0.0`: FMIN yields the negatively
    // signed value (`-0.0`).
    if a.is_sign_negative() { a } else { b }
  } else if a < b {
    a
  } else {
    b
  }
}

/// Per-lane float maximum — ARM `FMAX` (`vmaxq_f32`), NaN-propagating with
/// `+0.0 > -0.0`. Miri does not shim `llvm.aarch64.neon.fmax.v4f32`, so
/// under Miri this stores both vectors and reduces lane-wise; production
/// (`not(miri)`) is the raw intrinsic, byte-identical.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vmaxq_f32_compat(a: float32x4_t, b: float32x4_t) -> float32x4_t {
  #[cfg(not(miri))]
  {
    vmaxq_f32(a, b)
  }
  #[cfg(miri)]
  {
    let mut av = [0f32; 4];
    let mut bv = [0f32; 4];
    // SAFETY: each array holds four `f32`; `vst1q_f32` writes four lanes.
    unsafe {
      vst1q_f32(av.as_mut_ptr(), a);
      vst1q_f32(bv.as_mut_ptr(), b);
    }
    let out = [
      fmax_lane(av[0], bv[0]),
      fmax_lane(av[1], bv[1]),
      fmax_lane(av[2], bv[2]),
      fmax_lane(av[3], bv[3]),
    ];
    // SAFETY: `out` holds four `f32`; `vld1q_f32` reads four lanes.
    unsafe { vld1q_f32(out.as_ptr()) }
  }
}

/// Per-lane float minimum — ARM `FMIN` (`vminq_f32`), NaN-propagating with
/// `-0.0 < +0.0`. Miri does not shim `llvm.aarch64.neon.fmin.v4f32`, so
/// under Miri this stores both vectors and reduces lane-wise; production
/// (`not(miri)`) is the raw intrinsic, byte-identical.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vminq_f32_compat(a: float32x4_t, b: float32x4_t) -> float32x4_t {
  #[cfg(not(miri))]
  {
    vminq_f32(a, b)
  }
  #[cfg(miri)]
  {
    let mut av = [0f32; 4];
    let mut bv = [0f32; 4];
    // SAFETY: each array holds four `f32`; `vst1q_f32` writes four lanes.
    unsafe {
      vst1q_f32(av.as_mut_ptr(), a);
      vst1q_f32(bv.as_mut_ptr(), b);
    }
    let out = [
      fmin_lane(av[0], bv[0]),
      fmin_lane(av[1], bv[1]),
      fmin_lane(av[2], bv[2]),
      fmin_lane(av[3], bv[3]),
    ];
    // SAFETY: `out` holds four `f32`; `vld1q_f32` reads four lanes.
    unsafe { vld1q_f32(out.as_ptr()) }
  }
}

/// Per-lane signed-16-bit **max** — ARM `SMAX` (`vmaxq_s16`). miri does not
/// shim the `llvm.aarch64.neon.smax.v8i16` foreign call, so emulate it with
/// the same store→lane-wise→load shape as the float helpers. Integer max has
/// no NaN/signed-zero subtlety, so `i16::max` is exact by construction.
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vmaxq_s16_compat(a: int16x8_t, b: int16x8_t) -> int16x8_t {
  #[cfg(not(miri))]
  {
    vmaxq_s16(a, b)
  }
  #[cfg(miri)]
  {
    let mut av = [0i16; 8];
    let mut bv = [0i16; 8];
    // SAFETY: each array holds eight `i16`; `vst1q_s16` writes eight lanes.
    unsafe {
      vst1q_s16(av.as_mut_ptr(), a);
      vst1q_s16(bv.as_mut_ptr(), b);
    }
    let out = [
      av[0].max(bv[0]),
      av[1].max(bv[1]),
      av[2].max(bv[2]),
      av[3].max(bv[3]),
      av[4].max(bv[4]),
      av[5].max(bv[5]),
      av[6].max(bv[6]),
      av[7].max(bv[7]),
    ];
    // SAFETY: `out` holds eight `i16`; `vld1q_s16` reads eight lanes.
    unsafe { vld1q_s16(out.as_ptr()) }
  }
}

/// Per-lane signed-16-bit **min** — ARM `SMIN` (`vminq_s16`). miri does not
/// shim `llvm.aarch64.neon.smin.v8i16`; emulate with lane-wise `i16::min`
/// (exact — no NaN/signed-zero subtlety).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vminq_s16_compat(a: int16x8_t, b: int16x8_t) -> int16x8_t {
  #[cfg(not(miri))]
  {
    vminq_s16(a, b)
  }
  #[cfg(miri)]
  {
    let mut av = [0i16; 8];
    let mut bv = [0i16; 8];
    // SAFETY: each array holds eight `i16`; `vst1q_s16` writes eight lanes.
    unsafe {
      vst1q_s16(av.as_mut_ptr(), a);
      vst1q_s16(bv.as_mut_ptr(), b);
    }
    let out = [
      av[0].min(bv[0]),
      av[1].min(bv[1]),
      av[2].min(bv[2]),
      av[3].min(bv[3]),
      av[4].min(bv[4]),
      av[5].min(bv[5]),
      av[6].min(bv[6]),
      av[7].min(bv[7]),
    ];
    // SAFETY: `out` holds eight `i16`; `vld1q_s16` reads eight lanes.
    unsafe { vld1q_s16(out.as_ptr()) }
  }
}

/// Per-lane float→u32 convert with **round-to-nearest-even** then a
/// saturating unsigned cast — ARM `FCVTNU` (`vcvtnq_u32_f32`).
///
/// Rust's `f32 as u32` already saturates exactly as FCVTNU does (NaN→0,
/// negatives→0, `≥2³²`→`u32::MAX`); `round_ties_even` supplies the
/// round-half-to-even step the cast (which truncates) would otherwise
/// skip.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vcvtnq_u32_f32_compat(v: float32x4_t) -> uint32x4_t {
  #[cfg(not(miri))]
  {
    vcvtnq_u32_f32(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0f32; 4];
    // SAFETY: `a` holds four `f32`; `vst1q_f32` writes exactly four lanes.
    unsafe { vst1q_f32(a.as_mut_ptr(), v) };
    let out = [
      a[0].round_ties_even() as u32,
      a[1].round_ties_even() as u32,
      a[2].round_ties_even() as u32,
      a[3].round_ties_even() as u32,
    ];
    // SAFETY: `out` holds four `u32`; `vld1q_u32` reads exactly four lanes.
    unsafe { vld1q_u32(out.as_ptr()) }
  }
}

/// Per-lane float→u32 convert with **round-to-nearest-ties-away** then a
/// saturating unsigned cast — ARM `FCVTAU` (`vcvtaq_u32_f32`).
///
/// Rust's `f32::round` is ties-away-from-zero (FCVTAU's rounding) and the
/// `as u32` cast supplies the same saturation as FCVTAU (NaN→0,
/// negatives→0, `≥2³²`→`u32::MAX`).
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vcvtaq_u32_f32_compat(v: float32x4_t) -> uint32x4_t {
  #[cfg(not(miri))]
  {
    vcvtaq_u32_f32(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0f32; 4];
    // SAFETY: `a` holds four `f32`; `vst1q_f32` writes exactly four lanes.
    unsafe { vst1q_f32(a.as_mut_ptr(), v) };
    let out = [
      a[0].round() as u32,
      a[1].round() as u32,
      a[2].round() as u32,
      a[3].round() as u32,
    ];
    // SAFETY: `out` holds four `u32`; `vld1q_u32` reads exactly four lanes.
    unsafe { vld1q_u32(out.as_ptr()) }
  }
}

/// Saturating narrow `int16x8_t` → `uint8x8_t` — ARM `SQXTUN`
/// (`vqmovun_s16`): each signed 16-bit lane clamps to `[0, 255]`.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vqmovun_s16_compat(v: int16x8_t) -> uint8x8_t {
  #[cfg(not(miri))]
  {
    vqmovun_s16(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0i16; 8];
    // SAFETY: `a` holds eight `i16`; `vst1q_s16` writes exactly eight lanes.
    unsafe { vst1q_s16(a.as_mut_ptr(), v) };
    let out = [
      a[0].clamp(0, 255) as u8,
      a[1].clamp(0, 255) as u8,
      a[2].clamp(0, 255) as u8,
      a[3].clamp(0, 255) as u8,
      a[4].clamp(0, 255) as u8,
      a[5].clamp(0, 255) as u8,
      a[6].clamp(0, 255) as u8,
      a[7].clamp(0, 255) as u8,
    ];
    // SAFETY: `out` holds eight `u8`; `vld1_u8` reads exactly eight lanes.
    unsafe { vld1_u8(out.as_ptr()) }
  }
}

/// Saturating narrow `uint16x8_t` → `uint8x8_t` — ARM `UQXTN`
/// (`vqmovn_u16`): each unsigned 16-bit lane saturates to `255`.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vqmovn_u16_compat(v: uint16x8_t) -> uint8x8_t {
  #[cfg(not(miri))]
  {
    vqmovn_u16(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0u16; 8];
    // SAFETY: `a` holds eight `u16`; `vst1q_u16` writes exactly eight lanes.
    unsafe { vst1q_u16(a.as_mut_ptr(), v) };
    let out = [
      a[0].min(255) as u8,
      a[1].min(255) as u8,
      a[2].min(255) as u8,
      a[3].min(255) as u8,
      a[4].min(255) as u8,
      a[5].min(255) as u8,
      a[6].min(255) as u8,
      a[7].min(255) as u8,
    ];
    // SAFETY: `out` holds eight `u8`; `vld1_u8` reads exactly eight lanes.
    unsafe { vld1_u8(out.as_ptr()) }
  }
}

/// Saturating narrow `int32x4_t` → `uint16x4_t` — ARM `SQXTUN`
/// (`vqmovun_s32`): each signed 32-bit lane clamps to `[0, 65535]`.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vqmovun_s32_compat(v: int32x4_t) -> uint16x4_t {
  #[cfg(not(miri))]
  {
    vqmovun_s32(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0i32; 4];
    // SAFETY: `a` holds four `i32`; `vst1q_s32` writes exactly four lanes.
    unsafe { vst1q_s32(a.as_mut_ptr(), v) };
    let out = [
      a[0].clamp(0, 65535) as u16,
      a[1].clamp(0, 65535) as u16,
      a[2].clamp(0, 65535) as u16,
      a[3].clamp(0, 65535) as u16,
    ];
    // SAFETY: `out` holds four `u16`; `vld1_u16` reads exactly four lanes.
    unsafe { vld1_u16(out.as_ptr()) }
  }
}

/// Saturating narrow `uint32x4_t` → `uint16x4_t` — ARM `UQXTN`
/// (`vqmovn_u32`): each unsigned 32-bit lane saturates to `65535`.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vqmovn_u32_compat(v: uint32x4_t) -> uint16x4_t {
  #[cfg(not(miri))]
  {
    vqmovn_u32(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0u32; 4];
    // SAFETY: `a` holds four `u32`; `vst1q_u32` writes exactly four lanes.
    unsafe { vst1q_u32(a.as_mut_ptr(), v) };
    let out = [
      a[0].min(65535) as u16,
      a[1].min(65535) as u16,
      a[2].min(65535) as u16,
      a[3].min(65535) as u16,
    ];
    // SAFETY: `out` holds four `u16`; `vld1_u16` reads exactly four lanes.
    unsafe { vld1_u16(out.as_ptr()) }
  }
}

/// Saturating narrow `int32x4_t` → `int16x4_t` — ARM `SQXTN`
/// (`vqmovn_s32`): each signed 32-bit lane clamps to `[-32768, 32767]`.
///
/// # Safety
///
/// NEON must be available (baseline on aarch64).
#[inline]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub(crate) unsafe fn vqmovn_s32_compat(v: int32x4_t) -> int16x4_t {
  #[cfg(not(miri))]
  {
    vqmovn_s32(v)
  }
  #[cfg(miri)]
  {
    let mut a = [0i32; 4];
    // SAFETY: `a` holds four `i32`; `vst1q_s32` writes exactly four lanes.
    unsafe { vst1q_s32(a.as_mut_ptr(), v) };
    let out = [
      a[0].clamp(-32768, 32767) as i16,
      a[1].clamp(-32768, 32767) as i16,
      a[2].clamp(-32768, 32767) as i16,
      a[3].clamp(-32768, 32767) as i16,
    ];
    // SAFETY: `out` holds four `i16`; `vld1_s16` reads exactly four lanes.
    unsafe { vld1_s16(out.as_ptr()) }
  }
}
