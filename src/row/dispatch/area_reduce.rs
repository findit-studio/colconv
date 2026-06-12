//! Runtime SIMD dispatcher for the fused-downscale H-pass. Mirrors
//! the crate's dispatcher pattern (`dispatch::y_plane_to_luma_u16`).

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use crate::row::scalar::area_reduce as scalar;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::sse41_available;

/// Runtime-dispatched per-span weighted reduction of one source row
/// into `h_tmp` (`starts.len() * channels` slots).
///
/// `(w16, w16_off)` is the plan-time zero-padded u16 weight arena
/// (every span padded to a multiple of 8). The engine leaves it empty
/// when the geometry exceeds the u16 weight bound, which — like
/// channel counts other than 1 and 3, and `use_simd == false` —
/// routes to the scalar reference; every backend matches it
/// bit-for-bit.
///
/// x86 dispatches at the SSE4.1 tier only: spans chunk in 8 taps, so
/// 128 bits is the kernel's natural width. AVX2/AVX-512 tiers would
/// pay only on 16-tap-plus spans (16x-plus downscale factors) and are
/// deferred until profiling demands them.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn area_h_reduce_row(
  row: &[u8],
  channels: usize,
  starts: &[usize],
  offsets: &[usize],
  weights: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
  use_simd: bool,
) {
  // Release-mode safety boundary before any unsafe SIMD dispatch —
  // the per-arch helpers only debug_assert these.
  let out = starts.len();
  assert!(offsets.len() == out + 1, "offsets shape");
  assert!(h_tmp.len() >= out * channels, "h_tmp too short");
  assert!(weights.len() >= offsets[out], "weights arena too short");

  #[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
  ))]
  let padded = use_simd
    && !w16.is_empty()
    && w16_off.len() == out + 1
    && w16_off[out] <= w16.len()
    && w16_off[out].is_multiple_of(8);

  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && padded {
        if channels == 1 {
          // SAFETY: NEON availability checked; arena shape gated via
          // `padded`; row/h_tmp bounds asserted above.
          unsafe { arch::neon::area_reduce::area_h_reduce_row_c1(row, starts, w16, w16_off, h_tmp); }
          return;
        }
        if channels == 3 {
          // SAFETY: as above, 3-channel variant.
          unsafe { arch::neon::area_reduce::area_h_reduce_row_c3(row, starts, w16, w16_off, h_tmp); }
          return;
        }
      }
    },
    target_arch = "x86_64" => {
      if sse41_available() && padded {
        if channels == 1 {
          // SAFETY: SSE4.1 verified at runtime; arena shape gated via
          // `padded`; row/h_tmp bounds asserted above.
          unsafe { arch::x86_sse41::area_reduce::area_h_reduce_row_c1(row, starts, w16, w16_off, h_tmp); }
          return;
        }
        if channels == 3 {
          // SAFETY: as above, 3-channel variant.
          unsafe { arch::x86_sse41::area_reduce::area_h_reduce_row_c3(row, starts, w16, w16_off, h_tmp); }
          return;
        }
      }
    },
    target_arch = "wasm32" => {
      if simd128_available() && padded {
        if channels == 1 {
          // SAFETY: simd128 enabled at compile time; arena shape gated
          // via `padded`; row/h_tmp bounds asserted above.
          unsafe { arch::wasm_simd128::area_reduce::area_h_reduce_row_c1(row, starts, w16, w16_off, h_tmp); }
          return;
        }
        if channels == 3 {
          // SAFETY: as above, 3-channel variant.
          unsafe { arch::wasm_simd128::area_reduce::area_h_reduce_row_c3(row, starts, w16, w16_off, h_tmp); }
          return;
        }
      }
    },
    _ => {}
  }
  scalar::area_h_reduce_row(row, channels, starts, offsets, weights, h_tmp);
}
