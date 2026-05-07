//! Runtime SIMD dispatcher for Strategy A+ α-extract helpers. Selects
//! the highest-available SIMD backend per platform; falls back to scalar.
//!
//! Each dispatcher function has the same signature as its scalar
//! counterpart plus a `use_simd: bool` flag, and is `pub(crate)` so the
//! sinker impls can call it without going through the public row API.
//! The `use_simd` flag mirrors `MixedSinker::with_simd(false)` — when
//! `false`, the dispatcher skips feature detection and calls scalar
//! directly. This is required for benchmarking, fuzzing, and
//! differential-testing parity with the rest of the kernel call sites
//! that already accept `use_simd`.
//!
//! Dispatch follows the standard `cfg_select!` pattern used everywhere
//! in `dispatch::*`: the platform arm is selected at compile time, and
//! the best available backend is selected at runtime via the
//! `*_available()` helpers in [`crate::row`].

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use crate::row::scalar::alpha_extract as scalar;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};

// ---------------------------------------------------------------------------
// Helper 1: VUYA u8 → u8 RGBA  (α at packed slot 3)
// ---------------------------------------------------------------------------

/// Runtime-dispatched α-extract for VUYA: gather α from
/// `packed[3 + 4*n]` into `rgba_out[3 + 4*n]`.
///
/// Selects the highest available SIMD backend; falls back to scalar.
/// When `use_simd` is `false` (`MixedSinker::with_simd(false)`), the
/// SIMD cascade is bypassed and scalar runs directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_packed_u8x4_at_3(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  if !use_simd {
    return scalar::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime.
        unsafe { arch::neon::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe { arch::x86_avx512::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe { arch::x86_avx2::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe { arch::x86_sse41::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe { arch::wasm_simd128::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width); }
        return;
      }
    },
    _ => {}
  }
  scalar::copy_alpha_packed_u8x4_at_3(packed, rgba_out, width);
}

// ---------------------------------------------------------------------------
// Helper 2: AYUV64 u16 → u8 RGBA  (α at packed slot 0, depth >> 8)
// ---------------------------------------------------------------------------

/// Runtime-dispatched α-extract for AYUV64 → u8 RGBA: gather α from
/// `packed[0 + 4*n]` (u16) into `rgba_out[3 + 4*n]` (u8) via `>> 8`.
///
/// Selects the highest available SIMD backend; falls back to scalar.
/// When `use_simd` is `false`, calls scalar directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_packed_u16x4_to_u8_at_0(
  packed: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  if !use_simd {
    return scalar::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe { arch::neon::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe { arch::x86_avx512::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe { arch::x86_avx2::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe { arch::x86_sse41::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe { arch::wasm_simd128::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width); }
        return;
      }
    },
    _ => {}
  }
  scalar::copy_alpha_packed_u16x4_to_u8_at_0(packed, rgba_out, width);
}

// ---------------------------------------------------------------------------
// Helper 3: AYUV64 u16 → u16 RGBA  (α at packed slot 0, no depth conv)
// ---------------------------------------------------------------------------

/// Runtime-dispatched α-extract for AYUV64 → u16 RGBA: gather α from
/// `packed[0 + 4*n]` (u16) into `rgba_out[3 + 4*n]` (u16). No depth
/// conversion.
///
/// Selects the highest available SIMD backend; falls back to scalar.
/// When `use_simd` is `false`, calls scalar directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_packed_u16x4_at_0(
  packed: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  if !use_simd {
    return scalar::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe { arch::neon::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe { arch::x86_avx512::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe { arch::x86_avx2::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe { arch::x86_sse41::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe { arch::wasm_simd128::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width); }
        return;
      }
    },
    _ => {}
  }
  scalar::copy_alpha_packed_u16x4_at_0(packed, rgba_out, width);
}

// ---------------------------------------------------------------------------
// Helper 4: α plane u8 → u8 RGBA
// ---------------------------------------------------------------------------

/// Runtime-dispatched α-extract for planar Yuva u8: scatter α plane
/// into `rgba_out[3 + 4*n]`.
///
/// Selects the highest available SIMD backend; falls back to scalar.
/// When `use_simd` is `false`, calls scalar directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_plane_u8(alpha: &[u8], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  if !use_simd {
    return scalar::copy_alpha_plane_u8(alpha, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe { arch::neon::copy_alpha_plane_u8(alpha, rgba_out, width); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe { arch::x86_avx512::copy_alpha_plane_u8(alpha, rgba_out, width); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe { arch::x86_avx2::copy_alpha_plane_u8(alpha, rgba_out, width); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe { arch::x86_sse41::copy_alpha_plane_u8(alpha, rgba_out, width); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe { arch::wasm_simd128::copy_alpha_plane_u8(alpha, rgba_out, width); }
        return;
      }
    },
    _ => {}
  }
  scalar::copy_alpha_plane_u8(alpha, rgba_out, width);
}

// ---------------------------------------------------------------------------
// Helper 5: α plane u16 → u8 RGBA  (depth-conv >> (BITS-8))
// ---------------------------------------------------------------------------

/// Runtime-dispatched α-extract for planar Yuva*p high-bit → u8 RGBA:
/// scatter α plane (u16) into `rgba_out[3 + 4*n]` (u8) with
/// depth-conv `>> (BITS - 8)`.
///
/// `BE` selects the source α plane byte order (`false` = LE on disk/wire,
/// `true` = BE on disk/wire). The SIMD α-extract backends use raw
/// native-u16 loads (`vld1q_u16` / `_mm_loadu_si128` / `v128_load64_zero`)
/// and have no byte-swap path, so SIMD is only correct when the source
/// byte order matches the host CPU's native byte order. The dispatcher
/// computes `need_swap = BE != cfg!(target_endian = "big")` and routes
/// to scalar whenever a swap would be required (LE-on-BE-host or
/// BE-on-LE-host). Per the codex review of #82 the scalar helper is
/// target-endian-aware via `u16::from_be` / `u16::from_le`, so this
/// scalar fallback emits the correct α plane on every host. Phase 4 will
/// plumb BE through SIMD if a real BE-input sinker hot-path lands.
///
/// Truth table (`need_swap = BE != target_endian == "big"`):
/// - LE data, LE host: `false != false = false` → SIMD (host-native LE u16 loads correct)
/// - LE data, BE host: `false != true  = true`  → scalar (uses `u16::from_le`)
/// - BE data, LE host: `true  != false = true`  → scalar (uses `u16::from_be`)
/// - BE data, BE host: `true  != true  = false` → SIMD (host-native BE u16 loads correct)
///
/// Selects the highest available SIMD backend when host endian == data
/// endian; falls back to scalar otherwise. When `use_simd` is `false`,
/// calls scalar directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_plane_u16_to_u8<const BITS: u32, const BE: bool>(
  alpha: &[u16],
  rgba_out: &mut [u8],
  width: usize,
  use_simd: bool,
) {
  // Need a byte-swap if data byte order differs from host CPU's native order.
  let need_swap = BE != cfg!(target_endian = "big");
  if need_swap || !use_simd {
    return scalar::copy_alpha_plane_u16_to_u8::<BITS, BE>(alpha, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe { arch::neon::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe { arch::x86_avx512::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe { arch::x86_avx2::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe { arch::x86_sse41::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe { arch::wasm_simd128::copy_alpha_plane_u16_to_u8::<BITS>(alpha, rgba_out, width); }
        return;
      }
    },
    _ => {}
  }
  scalar::copy_alpha_plane_u16_to_u8::<BITS, BE>(alpha, rgba_out, width);
}

// ---------------------------------------------------------------------------
// Helper 6: α plane u16 → u16 RGBA  (no depth conv)
// ---------------------------------------------------------------------------

/// Runtime-dispatched α-extract for planar Yuva*p high-bit → u16 RGBA:
/// scatter α plane (u16) into `rgba_out[3 + 4*n]` (u16). No depth
/// conversion.
///
/// `BE` selects the source α plane byte order (`false` = LE on disk/wire,
/// `true` = BE on disk/wire). The dispatcher computes
/// `need_swap = BE != cfg!(target_endian = "big")` and routes to scalar
/// whenever a swap would be required: see `copy_alpha_plane_u16_to_u8`
/// above for the truth table and rationale (SIMD α-extract uses
/// host-native u16 loads; scalar is target-endian-aware).
///
/// Selects the highest available SIMD backend when host endian == data
/// endian; falls back to scalar otherwise. When `use_simd` is `false`,
/// calls scalar directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_plane_u16<const BITS: u32, const BE: bool>(
  alpha: &[u16],
  rgba_out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  // Need a byte-swap if data byte order differs from host CPU's native order.
  let need_swap = BE != cfg!(target_endian = "big");
  if need_swap || !use_simd {
    return scalar::copy_alpha_plane_u16::<BITS, BE>(alpha, rgba_out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON verified at runtime.
        unsafe { arch::neon::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512BW verified at runtime.
        unsafe { arch::x86_avx512::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe { arch::x86_avx2::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe { arch::x86_sse41::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe { arch::wasm_simd128::copy_alpha_plane_u16::<BITS>(alpha, rgba_out, width); }
        return;
      }
    },
    _ => {}
  }
  scalar::copy_alpha_plane_u16::<BITS, BE>(alpha, rgba_out, width);
}
