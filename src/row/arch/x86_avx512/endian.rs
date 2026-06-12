//! Endian-aware u16/u32 SIMD loaders for x86_64 AVX-512 (F + BW).
#![allow(dead_code)] // tier kernels (Phase 2 rollout PRs) will consume these
//!
//! Each helper takes a raw byte pointer to LE-encoded (or BE-encoded) data
//! and returns a `__m512i` vector containing the elements in **host-native**
//! byte order, ready for native u16/u32 SIMD math.
//!
//! The host-native conversion is monomorphized at compile time via
//! `cfg(target_endian = ...)`:
//!   - `load_le_*` is a no-op on LE targets (all real x86), byte-swap on BE
//!   - `load_be_*` is byte-swap on LE targets, no-op on BE targets
//!
//! Byte-swap is implemented with `_mm512_shuffle_epi8` (AVX-512BW) using
//! compile-time shuffle masks.  AVX-512's `vpshufb` operates per 128-bit
//! lane, so the mask replicates the same within-lane byte permutation across
//! all four 128-bit lanes of the 512-bit register.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::x86_64::*;

// ---- Byte-swap shuffle masks -----------------------------------------------

/// AVX-512BW `_mm512_shuffle_epi8` mask that swaps bytes within every 2-byte
/// (u16) lane across all four 128-bit lanes.
const BYTESWAP_MASK_U16: __m512i = unsafe {
  core::mem::transmute([
    // lane 0
    1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, // lane 1
    1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, // lane 2
    1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, // lane 3
    1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
  ])
};

/// AVX-512BW `_mm512_shuffle_epi8` mask that swaps bytes within every 4-byte
/// (u32) lane across all four 128-bit lanes.
const BYTESWAP_MASK_U32: __m512i = unsafe {
  core::mem::transmute([
    // lane 0
    3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, // lane 1
    3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, // lane 2
    3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, // lane 3
    3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
  ])
};

// ---- u16x32 loaders --------------------------------------------------------

/// Loads 32 × u16 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes.  Caller must have AVX-512F
/// and AVX-512BW enabled via
/// `#[target_feature(enable = "avx512f,avx512bw")]`.
#[inline(always)]
pub(crate) unsafe fn load_le_u16x32(ptr: *const u8) -> __m512i {
  let v = unsafe { _mm512_loadu_si512(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { _mm512_shuffle_epi8(v, BYTESWAP_MASK_U16) };
  v
}

/// Loads 32 × u16 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes.  Caller must have AVX-512F
/// and AVX-512BW enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u16x32(ptr: *const u8) -> __m512i {
  let v = unsafe { _mm512_loadu_si512(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { _mm512_shuffle_epi8(v, BYTESWAP_MASK_U16) };
  v
}

/// Generic dispatcher: routes to `load_le_u16x32` or `load_be_u16x32` based
/// on the compile-time `BE` const parameter.
///
/// # Safety
///
/// Same as `load_le_u16x32` / `load_be_u16x32`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u16x32<const BE: bool>(ptr: *const u8) -> __m512i {
  if BE {
    unsafe { load_be_u16x32(ptr) }
  } else {
    unsafe { load_le_u16x32(ptr) }
  }
}

// ---- u16x16 loaders (via _mm256_loadu_si256, for f16 widening) -------------
//
// AVX-512 kernels widen 16 × f16 using `_mm512_cvtph_ps(__m256i)`, which
// requires a 256-bit lane load.  The helpers below provide endian-aware
// loading of that 32-byte (16 × u16) block.

/// AVX2 `_mm256_shuffle_epi8` mask that swaps bytes within every 2-byte (u16)
/// lane across both 128-bit halves.
const BYTESWAP_MASK_U16X16: __m256i = unsafe {
  core::mem::transmute([
    // low 128-bit lane
    1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, // high 128-bit lane
    1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
  ])
};

/// Loads 16 × u16 (32 bytes) from `ptr` (LE-encoded) into a `__m256i`,
/// host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes. Caller must have AVX2
/// (implied by AVX-512) enabled.
#[inline(always)]
pub(crate) unsafe fn load_le_u16x16(ptr: *const u8) -> __m256i {
  let v = unsafe { _mm256_loadu_si256(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { _mm256_shuffle_epi8(v, BYTESWAP_MASK_U16X16) };
  v
}

/// Loads 16 × u16 (32 bytes) from `ptr` (BE-encoded) into a `__m256i`,
/// host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 32 readable bytes. Caller must have AVX2
/// (implied by AVX-512) enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u16x16(ptr: *const u8) -> __m256i {
  let v = unsafe { _mm256_loadu_si256(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { _mm256_shuffle_epi8(v, BYTESWAP_MASK_U16X16) };
  v
}

/// Generic dispatcher: routes to `load_le_u16x16` or `load_be_u16x16`.
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

// ---- u32x16 loaders --------------------------------------------------------

/// Loads 16 × u32 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes.  Caller must have AVX-512F
/// and AVX-512BW enabled.
#[inline(always)]
pub(crate) unsafe fn load_le_u32x16(ptr: *const u8) -> __m512i {
  let v = unsafe { _mm512_loadu_si512(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { _mm512_shuffle_epi8(v, BYTESWAP_MASK_U32) };
  v
}

/// Loads 16 × u32 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 64 readable bytes.  Caller must have AVX-512F
/// and AVX-512BW enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u32x16(ptr: *const u8) -> __m512i {
  let v = unsafe { _mm512_loadu_si512(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { _mm512_shuffle_epi8(v, BYTESWAP_MASK_U32) };
  v
}

/// Generic dispatcher: routes to `load_le_u32x16` or `load_be_u32x16` based
/// on the compile-time `BE` const parameter.
///
/// # Safety
///
/// Same as `load_le_u32x16` / `load_be_u32x16`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u32x16<const BE: bool>(ptr: *const u8) -> __m512i {
  if BE {
    unsafe { load_be_u32x16(ptr) }
  } else {
    unsafe { load_le_u32x16(ptr) }
  }
}
