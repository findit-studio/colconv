//! Endian-aware u16/u32 SIMD loaders for x86_64 AVX2.
#![allow(dead_code)] // tier kernels (Phase 2 rollout PRs) will consume these
//!
//! Each helper takes a raw byte pointer to LE-encoded (or BE-encoded) data
//! and returns a `__m256i` vector containing the elements in **host-native**
//! byte order, ready for native u16/u32 SIMD math.
//!
//! The host-native conversion is monomorphized at compile time via
//! `cfg(target_endian = ...)`:
//!   - `load_le_*` is a no-op on LE targets (all real x86), byte-swap on BE
//!   - `load_be_*` is byte-swap on LE targets, no-op on BE targets
//!
//! Byte-swap is implemented with `_mm256_shuffle_epi8` (AVX2) using
//! compile-time shuffle masks.  The masks replicate the 128-bit SSE pattern
//! across both 128-bit lanes of the 256-bit register — AVX2's `vpshufb`
//! operates per-lane, so the same within-lane byte permutation is applied to
//! both lanes independently.

use core::arch::x86_64::*;

// ---- Byte-swap shuffle masks -----------------------------------------------

/// AVX2 `_mm256_shuffle_epi8` mask that swaps bytes within every 2-byte (u16)
/// lane across both 128-bit halves.
const BYTESWAP_MASK_U16: __m256i = unsafe {
  core::mem::transmute([
    // low 128-bit lane
    1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    // high 128-bit lane (identical pattern)
    1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
  ])
};

/// AVX2 `_mm256_shuffle_epi8` mask that swaps bytes within every 4-byte (u32)
/// lane across both 128-bit halves.
const BYTESWAP_MASK_U32: __m256i = unsafe {
  core::mem::transmute([
    // low 128-bit lane
    3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    // high 128-bit lane (identical pattern)
    3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
  ])
};

// ---- u16x16 loaders --------------------------------------------------------

/// Loads 16 × u16 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes.  Caller must have AVX2
/// enabled via `#[target_feature(enable = "avx2")]`.
#[inline(always)]
pub(crate) unsafe fn load_le_u16x16(ptr: *const u8) -> __m256i {
  let v = unsafe { _mm256_loadu_si256(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { _mm256_shuffle_epi8(v, BYTESWAP_MASK_U16) };
  v
}

/// Loads 16 × u16 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes.  Caller must have AVX2
/// enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u16x16(ptr: *const u8) -> __m256i {
  let v = unsafe { _mm256_loadu_si256(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { _mm256_shuffle_epi8(v, BYTESWAP_MASK_U16) };
  v
}

/// Generic dispatcher: routes to `load_le_u16x16` or `load_be_u16x16` based
/// on the compile-time `BE` const parameter.
///
/// # Safety
///
/// Same as `load_le_u16x16` / `load_be_u16x16`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u16x16<const BE: bool>(ptr: *const u8) -> __m256i {
  if BE {
    unsafe { load_be_u16x16(ptr) }
  } else {
    unsafe { load_le_u16x16(ptr) }
  }
}

// ---- u32x8 loaders ---------------------------------------------------------

/// Loads 8 × u32 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes.  Caller must have AVX2
/// enabled.
#[inline(always)]
pub(crate) unsafe fn load_le_u32x8(ptr: *const u8) -> __m256i {
  let v = unsafe { _mm256_loadu_si256(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { _mm256_shuffle_epi8(v, BYTESWAP_MASK_U32) };
  v
}

/// Loads 8 × u32 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes.  Caller must have AVX2
/// enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u32x8(ptr: *const u8) -> __m256i {
  let v = unsafe { _mm256_loadu_si256(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { _mm256_shuffle_epi8(v, BYTESWAP_MASK_U32) };
  v
}

/// Generic dispatcher: routes to `load_le_u32x8` or `load_be_u32x8` based on
/// the compile-time `BE` const parameter.
///
/// # Safety
///
/// Same as `load_le_u32x8` / `load_be_u32x8`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u32x8<const BE: bool>(ptr: *const u8) -> __m256i {
  if BE {
    unsafe { load_be_u32x8(ptr) }
  } else {
    unsafe { load_le_u32x8(ptr) }
  }
}
