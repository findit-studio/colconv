//! Endian-aware u16/u32 SIMD loaders for WebAssembly simd128.
#![allow(dead_code)] // tier kernels (Phase 2 rollout PRs) will consume these
//!
//! Each helper takes a raw byte pointer to LE-encoded (or BE-encoded) data
//! and returns a `v128` vector containing the elements in **host-native** byte
//! order, ready for native u16/u32 SIMD math.
//!
//! The host-native conversion is monomorphized at compile time via
//! `cfg(target_endian = ...)`:
//!   - `load_le_*` is a no-op on LE targets (wasm32 is LE), byte-swap on BE
//!   - `load_be_*` is byte-swap on LE targets, no-op on BE targets
//!
//! Byte-swap is implemented with `u8x16_swizzle`, which has the same
//! semantics as SSSE3 `_mm_shuffle_epi8`: indices ≥ 16 zero the output lane.
//! The shuffle indices are expressed as `i8x16` constants (negative values
//! zero-out lanes, but all our indices are 0..15 so we use non-negative
//! values cast to i8).

use core::arch::wasm32::*;

// ---- u16x8 loaders ---------------------------------------------------------

/// Loads 8 x u16 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes.  Caller must have simd128
/// enabled via `#[target_feature(enable = "simd128")]`.
#[inline(always)]
pub(crate) unsafe fn load_le_u16x8(ptr: *const u8) -> v128 {
  let v = unsafe { v128_load(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = {
    // swap bytes within each u16 lane: [1,0, 3,2, 5,4, 7,6, 9,8, 11,10, 13,12, 15,14]
    let mask = i8x16(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    u8x16_swizzle(v, mask)
  };
  v
}

/// Loads 8 x u16 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes.  Caller must have simd128
/// enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u16x8(ptr: *const u8) -> v128 {
  let v = unsafe { v128_load(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = {
    let mask = i8x16(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    u8x16_swizzle(v, mask)
  };
  v
}

/// Generic dispatcher: routes to `load_le_u16x8` or `load_be_u16x8` based on
/// the compile-time `BE` const parameter.
///
/// # Safety
///
/// Same as `load_le_u16x8` / `load_be_u16x8`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u16x8<const BE: bool>(ptr: *const u8) -> v128 {
  if BE {
    unsafe { load_be_u16x8(ptr) }
  } else {
    unsafe { load_le_u16x8(ptr) }
  }
}

// ---- u32x4 loaders ---------------------------------------------------------

/// Loads 4 x u32 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes.  Caller must have simd128
/// enabled.
#[inline(always)]
pub(crate) unsafe fn load_le_u32x4(ptr: *const u8) -> v128 {
  let v = unsafe { v128_load(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = {
    // swap bytes within each u32 lane: [3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12]
    let mask = i8x16(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    u8x16_swizzle(v, mask)
  };
  v
}

/// Loads 4 x u32 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes.  Caller must have simd128
/// enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u32x4(ptr: *const u8) -> v128 {
  let v = unsafe { v128_load(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = {
    let mask = i8x16(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    u8x16_swizzle(v, mask)
  };
  v
}

/// Generic dispatcher: routes to `load_le_u32x4` or `load_be_u32x4` based on
/// the compile-time `BE` const parameter.
///
/// # Safety
///
/// Same as `load_le_u32x4` / `load_be_u32x4`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u32x4<const BE: bool>(ptr: *const u8) -> v128 {
  if BE {
    unsafe { load_be_u32x4(ptr) }
  } else {
    unsafe { load_le_u32x4(ptr) }
  }
}
