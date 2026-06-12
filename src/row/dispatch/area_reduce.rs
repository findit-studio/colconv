//! Runtime SIMD dispatcher for the fused-downscale H-pass. Mirrors
//! the crate's dispatcher pattern (`dispatch::y_plane_to_luma_u16`).

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(target_arch = "aarch64")]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use crate::row::scalar::area_reduce as scalar;

/// Runtime-dispatched per-span weighted reduction of one source row
/// into `h_tmp` (`starts.len() * channels` slots).
///
/// `(w16, w16_off)` is the plan-time zero-padded u16 weight arena
/// (every span padded to a multiple of 8). The engine leaves it empty
/// when the geometry exceeds the u16 weight bound, which routes 1-
/// and 3-channel rows to the SIMD backends; everything else — and
/// `use_simd == false` — takes the scalar reference, which every
/// backend matches bit-for-bit.
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

  #[cfg(target_arch = "aarch64")]
  let padded = !w16.is_empty()
    && w16_off.len() == out + 1
    && w16_off[out] <= w16.len()
    && w16_off[out].is_multiple_of(8);

  if !use_simd {
    return scalar::area_h_reduce_row(row, channels, starts, offsets, weights, h_tmp);
  }
  cfg_select! {
    target_arch = "aarch64" => {
      if neon_available() && padded && channels == 1 {
        // SAFETY: NEON availability checked; arena shape asserted via
        // `padded`; row/h_tmp bounds asserted above.
        return unsafe {
          arch::neon::area_reduce::area_h_reduce_row_c1(row, starts, w16, w16_off, h_tmp)
        };
      }
      if neon_available() && padded && channels == 3 {
        // SAFETY: as above, 3-channel variant.
        return unsafe {
          arch::neon::area_reduce::area_h_reduce_row_c3(row, starts, w16, w16_off, h_tmp)
        };
      }
      scalar::area_h_reduce_row(row, channels, starts, offsets, weights, h_tmp)
    }
    _ => {
      // x86_64 / wasm backends follow per the backend-symmetry
      // pattern; scalar in the meantime.
      scalar::area_h_reduce_row(row, channels, starts, offsets, weights, h_tmp)
    }
  }
}
