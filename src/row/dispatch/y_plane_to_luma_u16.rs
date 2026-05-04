//! Runtime SIMD dispatcher for `y_plane_to_luma_u16_row`. Mirrors the
//! pattern used by other crate-internal dispatchers (e.g.
//! `dispatch::alpha_extract`).

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use crate::row::scalar::y_plane_to_luma_u16 as scalar;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};

/// Runtime-dispatched zero-extension of a u8 Y plane to u16.
///
/// Selects the highest available SIMD backend; falls back to scalar.
/// When `use_simd` is `false` (`MixedSinker::with_simd(false)`), the
/// SIMD cascade is bypassed and scalar runs directly.
// Task 2 will wire the 9 sinkers; allow until then.
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn y_plane_to_luma_u16_row(plane: &[u8], out: &mut [u16], width: usize, use_simd: bool) {
  if !use_simd {
    return scalar::y_plane_to_luma_u16_row(plane, out, width);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() {
        // SAFETY: NEON is baseline on aarch64 and verified at runtime.
        unsafe { arch::neon::y_plane_to_luma_u16_row(plane, out, width); }
        return;
      }
    },
    target_arch = "x86_64" => {
      if avx512_available() {
        // SAFETY: AVX-512F + AVX-512BW verified at runtime.
        unsafe { arch::x86_avx512::y_plane_to_luma_u16_row(plane, out, width); }
        return;
      }
      if avx2_available() {
        // SAFETY: AVX2 verified at runtime.
        unsafe { arch::x86_avx2::y_plane_to_luma_u16_row(plane, out, width); }
        return;
      }
      if sse41_available() {
        // SAFETY: SSE4.1 verified at runtime.
        unsafe { arch::x86_sse41::y_plane_to_luma_u16_row(plane, out, width); }
        return;
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() {
        // SAFETY: simd128 enabled at compile time.
        unsafe { arch::wasm_simd128::y_plane_to_luma_u16_row(plane, out, width); }
        return;
      }
    },
    _ => {}
  }
  scalar::y_plane_to_luma_u16_row(plane, out, width);
}
