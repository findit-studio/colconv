//! Endian-aware u16/u32 SIMD loaders for AArch64 NEON.
#![allow(dead_code)] // tier kernels (Phase 2 rollout PRs) will consume these
//!
//! Each helper takes a raw byte pointer to LE-encoded (or BE-encoded) data
//! and returns a NEON vector containing the elements in **host-native** byte
//! order, ready for native u16/u32 SIMD math.
//!
//! The host-native conversion is monomorphized at compile time via
//! `cfg(target_endian = ...)`:
//!   - `load_le_*` is a no-op on LE targets, byte-swap on BE targets
//!   - `load_be_*` is byte-swap on LE targets, no-op on BE targets
//!
//! Tier kernels call the generic dispatchers `load_endian_u16x8::<BE>` and
//! `load_endian_u32x4::<BE>` from their own `<const BE: bool>` contexts.
//! The `if BE { ... } else { ... }` in the dispatcher is eliminated by the
//! compiler — each monomorphization sees only one branch.

use core::arch::aarch64::*;

// ---- u16x4 loaders ---------------------------------------------------------

/// Loads 4 x u16 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 8 readable bytes, aligned to at least 1 byte.
/// Caller must have NEON enabled.
#[inline(always)]
pub(crate) unsafe fn load_le_u16x4(ptr: *const u8) -> uint16x4_t {
  let v = unsafe { vld1_u16(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(v))) };
  v
}

/// Loads 4 x u16 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 8 readable bytes, aligned to at least 1 byte.
/// Caller must have NEON enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u16x4(ptr: *const u8) -> uint16x4_t {
  let v = unsafe { vld1_u16(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(v))) };
  v
}

/// Generic dispatcher: routes to `load_le_u16x4` or `load_be_u16x4`.
///
/// # Safety
///
/// Same as `load_le_u16x4` / `load_be_u16x4`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u16x4<const BE: bool>(ptr: *const u8) -> uint16x4_t {
  if BE {
    unsafe { load_be_u16x4(ptr) }
  } else {
    unsafe { load_le_u16x4(ptr) }
  }
}

// ---- u16x8 loaders ---------------------------------------------------------

/// Loads 8 x u16 from `ptr` (LE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes, aligned to at least 1 byte.
/// Caller must have NEON enabled (via `#[target_feature(enable = "neon")]`).
#[inline(always)]
pub(crate) unsafe fn load_le_u16x8(ptr: *const u8) -> uint16x8_t {
  let v = unsafe { vld1q_u16(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v))) };
  v
}

/// Loads 8 x u16 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes, aligned to at least 1 byte.
/// Caller must have NEON enabled (via `#[target_feature(enable = "neon")]`).
#[inline(always)]
pub(crate) unsafe fn load_be_u16x8(ptr: *const u8) -> uint16x8_t {
  let v = unsafe { vld1q_u16(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v))) };
  v
}

/// Generic dispatcher: routes to `load_le_u16x8` or `load_be_u16x8` based on
/// the compile-time `BE` const parameter.  The unused branch is eliminated by
/// the compiler when the caller is monomorphized.
///
/// # Safety
///
/// Same as `load_le_u16x8` / `load_be_u16x8`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u16x8<const BE: bool>(ptr: *const u8) -> uint16x8_t {
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
/// `ptr` must point to at least 16 readable bytes, aligned to at least 1 byte.
/// Caller must have NEON enabled.
#[inline(always)]
pub(crate) unsafe fn load_le_u32x4(ptr: *const u8) -> uint32x4_t {
  let v = unsafe { vld1q_u32(ptr.cast()) };
  #[cfg(target_endian = "big")]
  let v = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(v))) };
  v
}

/// Loads 4 x u32 from `ptr` (BE-encoded on disk/wire) into host-native order.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes, aligned to at least 1 byte.
/// Caller must have NEON enabled.
#[inline(always)]
pub(crate) unsafe fn load_be_u32x4(ptr: *const u8) -> uint32x4_t {
  let v = unsafe { vld1q_u32(ptr.cast()) };
  #[cfg(target_endian = "little")]
  let v = unsafe { vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(v))) };
  v
}

/// Generic dispatcher: routes to `load_le_u32x4` or `load_be_u32x4` based on
/// the compile-time `BE` const parameter.
///
/// # Safety
///
/// Same as `load_le_u32x4` / `load_be_u32x4`.
#[inline(always)]
pub(crate) unsafe fn load_endian_u32x4<const BE: bool>(ptr: *const u8) -> uint32x4_t {
  if BE {
    unsafe { load_be_u32x4(ptr) }
  } else {
    unsafe { load_le_u32x4(ptr) }
  }
}
