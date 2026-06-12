//! NEON fused-downscale H-pass: per output span, widening
//! multiply-accumulate over a zero-padded u16 weight arena.
//!
//! The arena (built once per plan by the resample engine) pads every
//! span's weights to a multiple of 8 with zeros, so the hot loop is
//! pure wide loads: 8 source samples widen `u8 -> u16` and meet 8
//! arena weights in `vmull_u16`/`vmull_high_u16` exact u32 lanes.
//! Padding lanes multiply by zero, so samples past a span's last tap
//! contribute nothing — sample loads only stage through a stack copy
//! at the row-end boundary, where a direct 8-byte load would cross
//! the slice end. Bit-identical to the scalar reference by integer
//! associativity.

#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

/// # Safety
///
/// NEON must be available (baseline on aarch64). Caller guarantees:
/// `w16_off.len() == starts.len() + 1` with monotonic entries bounded
/// by `w16.len()`, every span's padded length a multiple of 8,
/// `row.len() >= cells`, and `h_tmp.len() >= starts.len()`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_h_reduce_row_c1(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 8-byte sample load is either fully in-bounds
    // (guarded against row.len()) or staged through a zero-filled
    // stack copy; weight loads come from the 8-multiple arena slice.
    unsafe {
      let mut acc = vdupq_n_u32(0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let s8 = if base + 8 <= row.len() {
          vld1_u8(row.as_ptr().add(base))
        } else {
          let mut sbuf = [0u8; 8];
          let take = row.len() - base;
          sbuf[..take].copy_from_slice(&row[base..]);
          vld1_u8(sbuf.as_ptr())
        };
        let s16 = vmovl_u8(s8);
        let w = vld1q_u16(chunk.as_ptr());
        acc = vaddq_u32(acc, vmull_u16(vget_low_u16(s16), vget_low_u16(w)));
        acc = vaddq_u32(acc, vmull_high_u16(s16, w));
      }
      h_tmp[j] = vaddvq_u32(acc);
    }
  }
}

/// 3-channel (interleaved RGB) variant: `vld3_u8` deinterleaves eight
/// pixels per iteration into per-channel lanes sharing one weight
/// vector.
///
/// # Safety
///
/// As [`area_h_reduce_row_c1`], with `row.len() >= cells * 3` and
/// `h_tmp.len() >= starts.len() * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn area_h_reduce_row_c3(
  row: &[u8],
  starts: &[usize],
  w16: &[u16],
  w16_off: &[usize],
  h_tmp: &mut [u32],
) {
  for (j, &start) in starts.iter().enumerate() {
    let span = &w16[w16_off[j]..w16_off[j + 1]];
    // SAFETY: each 24-byte vld3_u8 load is either fully in-bounds
    // (guarded against row.len()) or staged through a zero-filled
    // stack copy; weight loads come from the 8-multiple arena slice.
    unsafe {
      let mut acc0 = vdupq_n_u32(0);
      let mut acc1 = vdupq_n_u32(0);
      let mut acc2 = vdupq_n_u32(0);
      for (ci, chunk) in span.chunks_exact(8).enumerate() {
        let base = start + ci * 8;
        let px = if (base + 8) * 3 <= row.len() {
          vld3_u8(row.as_ptr().add(base * 3))
        } else {
          let mut sbuf = [0u8; 24];
          let take = row.len() - base * 3;
          sbuf[..take].copy_from_slice(&row[base * 3..]);
          vld3_u8(sbuf.as_ptr())
        };
        let w = vld1q_u16(chunk.as_ptr());
        let wl = vget_low_u16(w);
        let s0 = vmovl_u8(px.0);
        acc0 = vaddq_u32(acc0, vmull_u16(vget_low_u16(s0), wl));
        acc0 = vaddq_u32(acc0, vmull_high_u16(s0, w));
        let s1 = vmovl_u8(px.1);
        acc1 = vaddq_u32(acc1, vmull_u16(vget_low_u16(s1), wl));
        acc1 = vaddq_u32(acc1, vmull_high_u16(s1, w));
        let s2 = vmovl_u8(px.2);
        acc2 = vaddq_u32(acc2, vmull_u16(vget_low_u16(s2), wl));
        acc2 = vaddq_u32(acc2, vmull_high_u16(s2, w));
      }
      h_tmp[j * 3] = vaddvq_u32(acc0);
      h_tmp[j * 3 + 1] = vaddvq_u32(acc1);
      h_tmp[j * 3 + 2] = vaddvq_u32(acc2);
    }
  }
}
