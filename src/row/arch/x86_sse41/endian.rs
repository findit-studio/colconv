//! Endian-aware u16/u32 SIMD loaders for x86_64 SSE4.1.
#![allow(dead_code)] // tier kernels (Phase 2 rollout PRs) will consume these
//!
//! Each helper takes a raw byte pointer to LE-encoded (or BE-encoded) data
//! and returns an `__m128i` vector containing the elements in **host-native**
//! byte order, ready for native u16/u32 SIMD math.
//!
//! The host-native conversion is monomorphized at compile time via
//! `cfg(target_endian = ...)`:
//!   - `load_le_*` is a no-op on LE targets (all real x86), byte-swap on BE
//!   - `load_be_*` is byte-swap on LE targets, no-op on BE targets
//!
//! Byte-swap is implemented with `_mm_shuffle_epi8` (SSSE3, a subset of
//! SSE4.1) using compile-time shuffle masks.  The mask constants use
//! `core::mem::transmute` because `__m128i` has no `const` constructor in
//! stable Rust; the transmutes are always safe — `__m128i` is a plain 128-bit
//! bag of bits.

use core::arch::x86_64::*;

// ---- Byte-swap shuffle masks -----------------------------------------------

/// SSSE3 `_mm_shuffle_epi8` mask that swaps bytes within every 2-byte (u16)
/// lane: `[1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 13,12, 15,14]`.
const BYTESWAP_MASK_U16: __m128i =
  unsafe { core::mem::transmute([1u8, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]) };

/// SSSE3 `_mm_shuffle_epi8` mask that swaps bytes within every 4-byte (u32)
/// lane: `[3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12]`.
const BYTESWAP_MASK_U32: __m128i =
  unsafe { core::mem::transmute([3u8, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12]) };

// ---- u16x8 loaders ---------------------------------------------------------

/// Loads 8 × u16 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes.  Caller must have SSE4.1
/// (and SSSE3) enabled via `#[target_feature(enable = "sse4.1")]`.
#[inline(always)]
pub(crate) unsafe fn load_le_u16x8(ptr: *const u8) -> __m128i {
  let v = unsafe { _mm_loadu_si128(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { _mm_shuffle_epi8(v, BYTESWAP_MASK_U16) };
  v
}

/// Loads 8 × u16 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes.  Caller must have SSE4.1
/// (and SSSE3) enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u16x8(ptr: *const u8) -> __m128i {
  let v = unsafe { _mm_loadu_si128(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { _mm_shuffle_epi8(v, BYTESWAP_MASK_U16) };
  v
}

/// Generic dispatcher: routes to `load_le_u16x8` or `load_be_u16x8` based on
/// the compile-time `BE` const parameter.
///
/// # Safety
///
/// Same as `load_le_u16x8` / `load_be_u16x8`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u16x8<const BE: bool>(ptr: *const u8) -> __m128i {
  if BE {
    unsafe { load_be_u16x8(ptr) }
  } else {
    unsafe { load_le_u16x8(ptr) }
  }
}

// ---- u16x4 loaders (via _mm_loadl_epi64, low 64 bits only) ----------------

/// SSSE3 `_mm_shuffle_epi8` mask that swaps bytes within every 2-byte (u16)
/// lane in the LOW 8 bytes of a 128-bit register. Upper bytes are zeroed.
const BYTESWAP_MASK_U16X4: __m128i = unsafe {
  core::mem::transmute([
    1u8, 0, 3, 2, 5, 4, 7, 6, 0x80u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
  ])
};

/// Loads 4 × u16 (8 bytes) from `ptr` (LE-encoded) into the low 64 bits of
/// `__m128i`, host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 8 readable bytes. Caller must have SSE4.1
/// (and SSSE3) enabled.
#[inline(always)]
pub(crate) unsafe fn load_le_u16x4(ptr: *const u8) -> __m128i {
  let v = unsafe { _mm_loadl_epi64(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { _mm_shuffle_epi8(v, BYTESWAP_MASK_U16X4) };
  v
}

/// Loads 4 × u16 (8 bytes) from `ptr` (BE-encoded) into the low 64 bits of
/// `__m128i`, host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 8 readable bytes. Caller must have SSE4.1
/// (and SSSE3) enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u16x4(ptr: *const u8) -> __m128i {
  let v = unsafe { _mm_loadl_epi64(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { _mm_shuffle_epi8(v, BYTESWAP_MASK_U16X4) };
  v
}

/// Generic dispatcher: routes to `load_le_u16x4` or `load_be_u16x4`.
///
/// # Safety
///
/// Same as `load_le_u16x4` / `load_be_u16x4`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u16x4<const BE: bool>(ptr: *const u8) -> __m128i {
  if BE {
    unsafe { load_be_u16x4(ptr) }
  } else {
    unsafe { load_le_u16x4(ptr) }
  }
}

// ---- u32x4 loaders ---------------------------------------------------------

/// Loads 4 × u32 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes.  Caller must have SSE4.1
/// (and SSSE3) enabled.
#[inline(always)]
pub(crate) unsafe fn load_le_u32x4(ptr: *const u8) -> __m128i {
  let v = unsafe { _mm_loadu_si128(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { _mm_shuffle_epi8(v, BYTESWAP_MASK_U32) };
  v
}

/// Loads 4 × u32 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes.  Caller must have SSE4.1
/// (and SSSE3) enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u32x4(ptr: *const u8) -> __m128i {
  let v = unsafe { _mm_loadu_si128(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { _mm_shuffle_epi8(v, BYTESWAP_MASK_U32) };
  v
}

/// Generic dispatcher: routes to `load_le_u32x4` or `load_be_u32x4` based on
/// the compile-time `BE` const parameter.
///
/// # Safety
///
/// Same as `load_le_u32x4` / `load_be_u32x4`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u32x4<const BE: bool>(ptr: *const u8) -> __m128i {
  if BE {
    unsafe { load_be_u32x4(ptr) }
  } else {
    unsafe { load_le_u32x4(ptr) }
  }
}

// (SSE4.1 u16x4 8-byte loaders `load_le_u16x4` / `load_be_u16x4` /
// `load_endian_u16x4` are now provided by PR #83's be-tier9 branch
// — see definitions earlier in this file.)
