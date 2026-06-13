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
use std::vec::Vec;

use crate::row::scalar::area_reduce as scalar;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};

/// Plan-time SIMD staging for the fused-downscale H-pass: the span
/// starts, a zero-padded u16 copy of the weight arena (every span
/// padded to a multiple of 8 so kernels run pure wide loads, with the
/// zero lanes annihilating samples past a span's last tap), and the
/// furthest source cell any real span touches (a padded span may
/// overhang the row by up to 7 cells; kernels stage those final
/// chunks through stack copies).
///
/// Invalid states are unrepresentable: the fields are private and
/// only [`Self::build`] populates them, so a value existing proves
/// per-span 8-multiple padding, monotonic offsets, u16-bounded
/// weights, and an accurate `max_reach` — the dispatcher then needs
/// only O(1) binds per row (shape vs the plan, reach vs the row)
/// before entering an unsafe kernel.
#[derive(Debug)]
pub(crate) struct PaddedSpans {
  starts: Vec<usize>,
  w16: Vec<u16>,
  off: Vec<usize>,
  max_reach: usize,
}

impl PaddedSpans {
  /// Builds the staging arena from a plan's H spans. Returns `None`
  /// — routing the dispatcher to the scalar reference — when any
  /// weight cannot fit u16 (for planner spans that means an output
  /// dimension past `u16::MAX`, pre-screened by the span-count gate),
  /// on arithmetic overflow, or when an allocation is refused: the
  /// arena is an optional accelerator, never a reason to fail stream
  /// creation.
  pub(crate) fn build(starts: &[usize], offsets: &[usize], weights: &[usize]) -> Option<Self> {
    let out = starts.len();
    if offsets.len() != out + 1 || out > usize::from(u16::MAX) {
      return None;
    }
    let mut total = 0usize;
    let mut max_reach = 0usize;
    for j in 0..out {
      let k = offsets[j + 1].checked_sub(offsets[j])?;
      let k_pad = k.div_ceil(8).checked_mul(8)?;
      total = total.checked_add(k_pad)?;
      max_reach = max_reach.max(starts[j].checked_add(k)?);
    }
    if weights.len() < offsets[out] {
      return None;
    }
    let mut w16 = Vec::new();
    let mut off = Vec::new();
    let mut starts_own = Vec::new();
    if w16.try_reserve_exact(total).is_err()
      || off.try_reserve_exact(out + 1).is_err()
      || starts_own.try_reserve_exact(out).is_err()
    {
      return None;
    }
    starts_own.extend_from_slice(starts);
    off.push(0);
    for j in 0..out {
      let span = &weights[offsets[j]..offsets[j + 1]];
      for &w in span {
        // Real validation, not a planner-shape assumption: the proof
        // must hold for arbitrary builder inputs.
        w16.push(u16::try_from(w).ok()?);
      }
      w16.resize(w16.len() + (span.len().div_ceil(8) * 8 - span.len()), 0);
      off.push(w16.len());
    }
    Some(Self {
      starts: starts_own,
      w16,
      off,
      max_reach,
    })
  }
}

/// Runtime-dispatched per-span weighted reduction of one source row
/// into `h_tmp` (`starts.len() * channels` slots).
///
/// `padded` is the plan-time staging arena; kernels consume its
/// internally-proven spans exclusively. `None` — like channel counts
/// other than 1 and 3, and `use_simd == false` — routes to the scalar
/// reference, which every backend matches bit-for-bit.
///
/// x86 dispatches at the SSE4.1 tier only: spans chunk in 8 taps, so
/// 128 bits is the kernel's natural width. AVX2/AVX-512 tiers would
/// pay only on 16-tap-plus spans (16x-plus downscale factors) and are
/// deferred until profiling demands them.
///
/// # Panics
///
/// When a supplied arena does not bind to this call: span count
/// differing from the plan's, or reach past the row — both are
/// internal contract bugs, not fallback cases.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn area_h_reduce_row(
  row: &[u8],
  channels: usize,
  starts: &[usize],
  offsets: &[usize],
  weights: &[usize],
  padded: Option<&PaddedSpans>,
  h_tmp: &mut [u32],
  use_simd: bool,
) {
  // Release-mode safety boundary before any unsafe SIMD dispatch —
  // the per-arch helpers only debug_assert these.
  let out = starts.len();
  assert!(offsets.len() == out + 1, "offsets shape");
  assert!(h_tmp.len() >= out * channels, "h_tmp too short");
  assert!(weights.len() >= offsets[out], "weights arena too short");

  if use_simd && let Some(p) = padded {
    // O(1) binds completing the construction proof for this call:
    // same span count as the plan (so the h_tmp bound covers kernel
    // writes) and every real span inside the row (so kernel index
    // arithmetic cannot wrap; the padded overhang past a span's last
    // tap stages through guarded stack copies).
    assert!(p.starts.len() == out, "padded arena shape");
    assert!(
      p.off.len() == out + 1 && p.off[out] <= p.w16.len(),
      "padded arena layout"
    );
    assert!(
      p.max_reach <= row.len() / channels,
      "padded arena exceeds row"
    );
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() && channels == 1 {
          // SAFETY: NEON availability checked; arena coherence proven
          // by construction plus the binds above; h_tmp bound asserted.
          unsafe { arch::neon::area_reduce::area_h_reduce_row_c1(row, &p.starts, &p.w16, &p.off, h_tmp); }
          return;
        }
        if neon_available() && channels == 3 {
          // SAFETY: as above, 3-channel variant.
          unsafe { arch::neon::area_reduce::area_h_reduce_row_c3(row, &p.starts, &p.w16, &p.off, h_tmp); }
          return;
        }
      },
      target_arch = "x86_64" => {
        // Highest available tier wins; all consume the same arena and
        // are bit-identical. AVX2/AVX-512 widen within a span (16/32
        // taps per step), so they outrun SSE4.1 only past 16-tap
        // spans (16x-plus downscales); narrower spans fall to the
        // shared 128-bit step inside each kernel.
        if channels == 1 {
          if avx512_available() {
            // SAFETY: AVX-512F+BW verified; arena coherence proven by
            // construction plus the binds above; h_tmp bound asserted.
            unsafe { arch::x86_avx512::area_reduce::area_h_reduce_row_c1(row, &p.starts, &p.w16, &p.off, h_tmp); }
            return;
          }
          if avx2_available() {
            // SAFETY: AVX2 verified; same arena/bind guarantees.
            unsafe { arch::x86_avx2::area_reduce::area_h_reduce_row_c1(row, &p.starts, &p.w16, &p.off, h_tmp); }
            return;
          }
          if sse41_available() {
            // SAFETY: SSE4.1 verified; same arena/bind guarantees.
            unsafe { arch::x86_sse41::area_reduce::area_h_reduce_row_c1(row, &p.starts, &p.w16, &p.off, h_tmp); }
            return;
          }
        }
        if channels == 3 {
          if avx512_available() {
            // SAFETY: AVX-512F+BW verified; 3-channel variant.
            unsafe { arch::x86_avx512::area_reduce::area_h_reduce_row_c3(row, &p.starts, &p.w16, &p.off, h_tmp); }
            return;
          }
          if avx2_available() {
            // SAFETY: AVX2 verified; 3-channel variant.
            unsafe { arch::x86_avx2::area_reduce::area_h_reduce_row_c3(row, &p.starts, &p.w16, &p.off, h_tmp); }
            return;
          }
          if sse41_available() {
            // SAFETY: SSE4.1 verified; 3-channel variant.
            unsafe { arch::x86_sse41::area_reduce::area_h_reduce_row_c3(row, &p.starts, &p.w16, &p.off, h_tmp); }
            return;
          }
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() && channels == 1 {
          // SAFETY: simd128 enabled at compile time; arena coherence
          // proven by construction plus the binds above; h_tmp bound
          // asserted.
          unsafe { arch::wasm_simd128::area_reduce::area_h_reduce_row_c1(row, &p.starts, &p.w16, &p.off, h_tmp); }
          return;
        }
        if simd128_available() && channels == 3 {
          // SAFETY: as above, 3-channel variant.
          unsafe { arch::wasm_simd128::area_reduce::area_h_reduce_row_c3(row, &p.starts, &p.w16, &p.off, h_tmp); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::area_h_reduce_row(row, channels, starts, offsets, weights, h_tmp);
}

/// Runtime-dispatched V-pass AXPY: `acc[i] += w * h_tmp[i]` over the
/// H-reduced row. The kernels broadcast the weight in u32 lanes; a
/// wider weight (output dimension past `u32::MAX`) — like
/// `use_simd == false` — takes the scalar reference, which every
/// backend matches bit-for-bit.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn area_v_accumulate(acc: &mut [u64], h_tmp: &[u32], w: u64, use_simd: bool) {
  // Release-mode safety boundary before any unsafe SIMD dispatch —
  // the per-arch helpers only debug_assert this.
  assert!(h_tmp.len() >= acc.len(), "h_tmp too short");

  if !use_simd {
    return scalar::area_v_accumulate(acc, h_tmp, w);
  }
  #[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
  ))]
  if let Ok(w32) = u32::try_from(w) {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON availability checked; bounds asserted above.
          unsafe { arch::neon::area_reduce::area_v_accumulate(acc, h_tmp, w32); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512F verified at runtime; bounds asserted above.
          unsafe { arch::x86_avx512::area_reduce::area_v_accumulate(acc, h_tmp, w32); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified at runtime; bounds asserted above.
          unsafe { arch::x86_avx2::area_reduce::area_v_accumulate(acc, h_tmp, w32); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified at runtime; bounds asserted above.
          unsafe { arch::x86_sse41::area_reduce::area_v_accumulate(acc, h_tmp, w32); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 enabled at compile time; bounds asserted above.
          unsafe { arch::wasm_simd128::area_reduce::area_v_accumulate(acc, h_tmp, w32); }
          return;
        }
      },
      _ => {}
    }
  }
  scalar::area_v_accumulate(acc, h_tmp, w);
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  fn arena(starts: &[usize], offsets: &[usize], weights: &[usize]) -> PaddedSpans {
    PaddedSpans::build(starts, offsets, weights).expect("valid spans")
  }

  #[test]
  fn build_proves_padding_and_reach() {
    // A 4-tap and a 9-tap span pad to 8 and 16 entries; the reach is
    // the furthest padded end. Non-monotonic offsets and non-8-multiple
    // spans are unrepresentable: only this builder populates the
    // fields.
    let p = arena(&[0, 4], &[0, 4, 13], &[1usize; 13]);
    assert_eq!(p.off, std::vec![0, 8, 24]);
    assert_eq!(p.w16.len(), 24);
    assert_eq!(p.max_reach, 13);
    assert_eq!(p.starts, std::vec![0, 4]);
    assert_eq!(&p.w16[..4], [1, 1, 1, 1]);
    assert_eq!(&p.w16[4..8], [0, 0, 0, 0]);
  }

  #[test]
  fn build_rejects_an_overweight_weight() {
    // 65536 truncates to 0 as u16; the builder must refuse, not
    // silently zero a tap.
    assert!(PaddedSpans::build(&[0], &[0, 1], &[65_536]).is_none());
  }

  #[test]
  fn build_rejects_outputs_past_the_u16_weight_bound() {
    let out = usize::from(u16::MAX) + 1;
    let starts = std::vec![0usize; out];
    let offsets: std::vec::Vec<usize> = (0..=out).collect();
    let weights = std::vec![1usize; out];
    assert!(PaddedSpans::build(&starts, &offsets, &weights).is_none());
  }

  #[test]
  #[should_panic(expected = "padded arena shape")]
  fn dispatcher_panics_on_plan_mismatch() {
    // An arena built for a one-span plan must not bind to a two-span
    // call: kernel h_tmp writes would no longer be covered by the
    // outer bound assert.
    let p = arena(&[0], &[0, 4], &[1usize; 4]);
    let row = [0u8; 16];
    let weights = [1usize; 8];
    let mut h_tmp = [0u32; 2];
    area_h_reduce_row(
      &row,
      1,
      &[0, 4],
      &[0, 4, 8],
      &weights,
      Some(&p),
      &mut h_tmp,
      true,
    );
  }

  #[test]
  #[should_panic(expected = "padded arena exceeds row")]
  fn dispatcher_panics_on_row_shorter_than_reach() {
    // A 4-tap span against a 3-cell row: rejected before any kernel
    // index arithmetic could wrap.
    let p = arena(&[0], &[0, 4], &[1usize; 4]);
    let row = [0u8; 3];
    let weights = [1usize; 4];
    let mut h_tmp = [0u32; 1];
    area_h_reduce_row(&row, 1, &[0], &[0, 4], &weights, Some(&p), &mut h_tmp, true);
  }
}
