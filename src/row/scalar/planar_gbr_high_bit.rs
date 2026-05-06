//! Scalar reference kernels for high-bit-depth planar GBR sources
//! (Tier 10b — `AV_PIX_FMT_GBRP{9,10,12,14,16}LE` /
//! `AV_PIX_FMT_GBRAP{9,10,12,14,16}LE`).
//!
//! All functions are const-generic over `BITS ∈ {9, 10, 12, 14, 16}`.
//! No runtime branching on `BITS` — every `BITS - 8` shift is a
//! const-eval expression resolved at monomorphisation.
//!
//! # Output variants
//!
//! | Suffix             | Element type | Alpha         |
//! |--------------------|-------------|---------------|
//! | `rgb_high_bit`     | `u8`        | none          |
//! | `rgb_u16_high_bit` | `u16`       | none          |
//! | `rgba_opaque_*`    | `u8`/`u16`  | opaque const  |
//! | `gbra_to_rgba_*`   | `u8`/`u16`  | source plane  |
//!
//! # Channel reorder
//!
//! FFmpeg planar GBR stores planes in **G, B, R** order, but the
//! packed output convention is **R, G, B** (matching FFmpeg
//! `AV_PIX_FMT_RGB24`). Every kernel performs this reorder.
//!
//! # u8 downshift
//!
//! u8-output kernels apply `>> (BITS - 8)` per sample (plain truncation,
//! matching FFmpeg `swscale` behaviour). For `BITS == 16` this is `>> 8`;
//! for `BITS == 9` it is `>> 1`.
//!
//! # Opaque alpha constants
//!
//! - u8: `0xFF`
//! - u16: `(1u16 << BITS) - 1` (i.e., `511`, `1023`, `4095`, …)

/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B`
/// **bytes**, downshifting each sample by `BITS - 8`.
///
/// Output order is **R, G, B** per pixel (FFmpeg `RGB24` convention).
///
/// # Panics (debug builds)
///
/// Asserts that `g`, `b`, `r` each have at least `width` samples and
/// `rgb_out` has at least `width * 3` bytes.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgb_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let shift = BITS - 8;
  for x in 0..width {
    let r_val = r[x] & mask;
    let g_val = g[x] & mask;
    let b_val = b[x] & mask;
    let dst = x * 3;
    rgb_out[dst] = (r_val >> shift) as u8;
    rgb_out[dst + 1] = (g_val >> shift) as u8;
    rgb_out[dst + 2] = (b_val >> shift) as u8;
  }
}

/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B`
/// **`u16`** samples. Copies samples directly without shifting —
/// output values are in `[0, (1 << BITS) - 1]`.
///
/// # Panics (debug builds)
///
/// Asserts that `g`, `b`, `r` each have at least `width` samples and
/// `rgb_u16_out` has at least `width * 3` samples.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgb_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  for x in 0..width {
    let r_val = r[x] & mask;
    let g_val = g[x] & mask;
    let b_val = b[x] & mask;
    let dst = x * 3;
    rgb_u16_out[dst] = r_val;
    rgb_u16_out[dst + 1] = g_val;
    rgb_u16_out[dst + 2] = b_val;
  }
}

/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B, A`
/// **bytes** with a constant **opaque** alpha (`0xFF`). Used for
/// `Gbrp*` sources (no alpha plane) when `with_rgba` is requested.
///
/// Each sample is downshifted by `BITS - 8`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgba_opaque_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let shift = BITS - 8;
  for x in 0..width {
    let r_val = r[x] & mask;
    let g_val = g[x] & mask;
    let b_val = b[x] & mask;
    let dst = x * 4;
    rgba_out[dst] = (r_val >> shift) as u8;
    rgba_out[dst + 1] = (g_val >> shift) as u8;
    rgba_out[dst + 2] = (b_val >> shift) as u8;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Interleaves three planar G/B/R `u16` rows into packed `R, G, B, A`
/// **`u16`** samples with a constant **opaque** alpha
/// (`(1u16 << BITS) - 1`). Used for `Gbrp*` sources (no alpha plane)
/// when `with_rgba_u16` is requested. Copies samples directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgba_opaque_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let opaque: u16 = mask;
  for x in 0..width {
    let r_val = r[x] & mask;
    let g_val = g[x] & mask;
    let b_val = b[x] & mask;
    let dst = x * 4;
    rgba_u16_out[dst] = r_val;
    rgba_u16_out[dst + 1] = g_val;
    rgba_u16_out[dst + 2] = b_val;
    rgba_u16_out[dst + 3] = opaque;
  }
}

/// Interleaves four planar G/B/R/A `u16` rows into packed `R, G, B, A`
/// **bytes**. Alpha is sourced from the `a` plane (real per-pixel α).
/// Each sample (including α) is downshifted by `BITS - 8`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbra_to_rgba_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 10 | 12 | 14 | 16),
      "BITS must be one of 10, 12, 14, or 16 (FFmpeg has no GBRAP9)"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  let shift = BITS - 8;
  for x in 0..width {
    let r_val = r[x] & mask;
    let g_val = g[x] & mask;
    let b_val = b[x] & mask;
    let a_val = a[x] & mask;
    let dst = x * 4;
    rgba_out[dst] = (r_val >> shift) as u8;
    rgba_out[dst + 1] = (g_val >> shift) as u8;
    rgba_out[dst + 2] = (b_val >> shift) as u8;
    rgba_out[dst + 3] = (a_val >> shift) as u8;
  }
}

/// Interleaves four planar G/B/R/A `u16` rows into packed `R, G, B, A`
/// **`u16`** samples. Alpha is sourced from the `a` plane at native
/// depth (no shift). Copies all four channels directly.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbra_to_rgba_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  a: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  const {
    assert!(
      matches!(BITS, 10 | 12 | 14 | 16),
      "BITS must be one of 10, 12, 14, or 16 (FFmpeg has no GBRAP9)"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  let mask: u16 = ((1u32 << BITS) - 1) as u16;
  for x in 0..width {
    let r_val = r[x] & mask;
    let g_val = g[x] & mask;
    let b_val = b[x] & mask;
    let a_val = a[x] & mask;
    let dst = x * 4;
    rgba_u16_out[dst] = r_val;
    rgba_u16_out[dst + 1] = g_val;
    rgba_u16_out[dst + 2] = b_val;
    rgba_u16_out[dst + 3] = a_val;
  }
}

/// Derives luma (Y') from three planar G/B/R `u16` rows directly at
/// native bit depth, avoiding the 256-level banding that would result
/// from staging through u8.
///
/// Uses i64 intermediates throughout so the BITS=16 case
/// (`max R = 65535`, product ≈ 1.54 B) does not overflow. The
/// performance cost relative to a separate i32 path for lower
/// bit-depths is negligible at the per-row level.
///
/// `full_range = true` → Y' ∈ `[0, (1 << BITS) - 1]` (full).
/// `full_range = false` → Y' ∈ `[16 << (BITS - 8), 235 << (BITS - 8)]`
/// (limited / studio swing). The limited-range formula mirrors
/// `rgb_to_luma_row` but scaled to native depth.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_luma_u16_high_bit_row<const BITS: u32>(
  g: &[u16],
  b: &[u16],
  r: &[u16],
  luma_out: &mut [u16],
  width: usize,
  matrix: crate::ColorMatrix,
  full_range: bool,
) {
  const {
    assert!(
      matches!(BITS, 9 | 10 | 12 | 14 | 16),
      "BITS must be one of 9, 10, 12, 14, or 16"
    )
  };
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(luma_out.len() >= width, "luma row too short");

  let (k_r, k_g, k_b) = super::luma_coefficients_q15(matrix);
  let k_r = k_r as i64;
  let k_g = k_g as i64;
  let k_b = k_b as i64;
  const RND: i64 = 1 << 14;
  let native_max: u16 = ((1u32 << BITS) - 1) as u16;
  let mask: u16 = native_max;

  if full_range {
    for x in 0..width {
      let rv = (r[x] & mask) as i64;
      let gv = (g[x] & mask) as i64;
      let bv = (b[x] & mask) as i64;
      let y = ((k_r * rv + k_g * gv + k_b * bv + RND) >> 15) as i32;
      luma_out[x] = y.clamp(0, native_max as i32) as u16;
    }
  } else {
    // Limited-range luma at native depth:
    //   Y_lim = Y_off + Y_full_clamped * range / native_max
    // where:
    //   Y_off       = 16  << (BITS - 8)        (native limited black)
    //   range       = 219 << (BITS - 8)        (native limited span)
    //   native_max  = (1 << BITS) - 1          (full-range upper bound)
    //
    // The naive 8-bit `LIMITED_SCALE_Q15 = round(219/255 × 32768)` ratio
    // is wrong here because it scales Y_full by `219/255 ≈ 0.85882`
    // when the correct native ratio is `range / native_max ≈ 0.85546`
    // at BITS=16. The ~0.4% overshoot makes the top ~250 input codes
    // collapse onto the y_max clamp, destroying highlight gradation
    // (codex review). The exact form below uses i64 throughout —
    // `range × native_max < 2^32` for BITS ≤ 16 — and a +native_max/2
    // bias for round-half-up semantics.
    let y_off = (16i64) << (BITS - 8);
    let range = (219i64) << (BITS - 8);
    let native_max_i64 = native_max as i64;
    let y_max = (235i64) << (BITS - 8);
    let y_min = y_off;
    for x in 0..width {
      let rv = (r[x] & mask) as i64;
      let gv = (g[x] & mask) as i64;
      let bv = (b[x] & mask) as i64;
      let y_full = (k_r * rv + k_g * gv + k_b * bv + RND) >> 15;
      let y_full_clamped = y_full.clamp(0, native_max_i64);
      let y_lim = y_off + (y_full_clamped * range + native_max_i64 / 2) / native_max_i64;
      luma_out[x] = y_lim.clamp(y_min, y_max) as u16;
    }
  }
}

// ---- Unit tests -----------------------------------------------------------

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests {
  use super::*;
  use crate::ColorMatrix;

  // ---- gbr_to_rgb_high_bit_row: u8 output, downshift ----------------------

  #[test]
  fn rgb_high_bit_bits10_channel_reorder() {
    // G=0, B=100, R=1000 → packed R,G,B = 1000>>2, 0>>2, 100>>2 = 250, 0, 25
    let g = [0u16; 1];
    let b = [100u16; 1];
    let r = [1000u16; 1];
    let mut out = [0u8; 3];
    gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], 250); // R
    assert_eq!(out[1], 0); // G
    assert_eq!(out[2], 25); // B
  }

  #[test]
  fn rgb_high_bit_bits10_max_value_becomes_0xff() {
    let max = (1u16 << 10) - 1; // 1023
    let g = [max; 4];
    let b = [max; 4];
    let r = [max; 4];
    let mut out = [0u8; 12];
    gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out, 4);
    assert!(out.iter().all(|&v| v == 0xFF), "all pixels must be 0xFF");
  }

  #[test]
  fn rgb_high_bit_bits16_max_value_becomes_0xff() {
    let max = u16::MAX;
    let g = [max; 2];
    let b = [max; 2];
    let r = [max; 2];
    let mut out = [0u8; 6];
    gbr_to_rgb_high_bit_row::<16>(&g, &b, &r, &mut out, 2);
    assert!(out.iter().all(|&v| v == 0xFF));
  }

  #[test]
  fn rgb_high_bit_bits10_zero_becomes_zero() {
    let g = [0u16; 2];
    let b = [0u16; 2];
    let r = [0u16; 2];
    let mut out = [0xFFu8; 6];
    gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out, 2);
    assert!(out.iter().all(|&v| v == 0));
  }

  #[test]
  fn rgb_high_bit_bits9_downshift_by_1() {
    // BITS=9: shift = 1. Value 510 >> 1 = 255.
    let g = [510u16; 1];
    let b = [0u16; 1];
    let r = [0u16; 1];
    let mut out = [0u8; 3];
    gbr_to_rgb_high_bit_row::<9>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[1], 255); // G channel
  }

  #[test]
  fn rgb_high_bit_bits12_downshift_by_4() {
    // BITS=12: shift = 4. Value 4080 >> 4 = 255.
    let r = [4080u16; 1];
    let g = [0u16; 1];
    let b = [0u16; 1];
    let mut out = [0u8; 3];
    gbr_to_rgb_high_bit_row::<12>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], 255); // R channel
  }

  #[test]
  fn rgb_high_bit_multiple_pixels_correct_layout() {
    // 3 pixels: (R,G,B) = (100,200,300>>2=75), (200>>2=50,0,0), (0,150>>2=37,50>>2=12)
    // BITS=10, shift=2
    let r = [400u16, 200u16, 0u16];
    let g = [800u16, 0u16, 600u16];
    let b = [300u16, 0u16, 200u16];
    let mut out = [0u8; 9];
    gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out, 3);
    // pixel 0: R=400>>2=100, G=800>>2=200, B=300>>2=75
    assert_eq!(out[0], 100);
    assert_eq!(out[1], 200);
    assert_eq!(out[2], 75);
    // pixel 1: R=200>>2=50, G=0, B=0
    assert_eq!(out[3], 50);
    assert_eq!(out[4], 0);
    assert_eq!(out[5], 0);
    // pixel 2: R=0, G=600>>2=150, B=200>>2=50
    assert_eq!(out[6], 0);
    assert_eq!(out[7], 150);
    assert_eq!(out[8], 50);
  }

  // ---- gbr_to_rgb_u16_high_bit_row: u16 output, no shift ------------------

  #[test]
  fn rgb_u16_high_bit_channel_reorder() {
    let g = [111u16; 1];
    let b = [222u16; 1];
    let r = [333u16; 1];
    let mut out = [0u16; 3];
    gbr_to_rgb_u16_high_bit_row::<10>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], 333); // R
    assert_eq!(out[1], 111); // G
    assert_eq!(out[2], 222); // B
  }

  #[test]
  fn rgb_u16_high_bit_bits10_max_preserved() {
    let max = (1u16 << 10) - 1; // 1023
    let g = [max; 4];
    let b = [max; 4];
    let r = [max; 4];
    let mut out = [0u16; 12];
    gbr_to_rgb_u16_high_bit_row::<10>(&g, &b, &r, &mut out, 4);
    assert!(out.iter().all(|&v| v == max));
  }

  #[test]
  fn rgb_u16_high_bit_bits16_max_preserved() {
    let max = u16::MAX;
    let g = [max; 2];
    let b = [max; 2];
    let r = [max; 2];
    let mut out = [0u16; 6];
    gbr_to_rgb_u16_high_bit_row::<16>(&g, &b, &r, &mut out, 2);
    assert!(out.iter().all(|&v| v == max));
  }

  #[test]
  fn rgb_u16_high_bit_values_not_shifted() {
    // Verify that u16 output does NOT shift values (unlike u8 output).
    let g = [1000u16; 1];
    let b = [2000u16; 1];
    let r = [3000u16; 1];
    let mut out = [0u16; 3];
    gbr_to_rgb_u16_high_bit_row::<12>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], 3000); // R — unchanged
    assert_eq!(out[1], 1000); // G — unchanged
    assert_eq!(out[2], 2000); // B — unchanged
  }

  // ---- gbr_to_rgba_opaque_high_bit_row: u8 RGBA with constant alpha --------

  #[test]
  fn rgba_opaque_high_bit_bits10_alpha_is_0xff() {
    let max = (1u16 << 10) - 1;
    let g = [max; 4];
    let b = [max; 4];
    let r = [max; 4];
    let mut out = [0u8; 16];
    gbr_to_rgba_opaque_high_bit_row::<10>(&g, &b, &r, &mut out, 4);
    for i in 0..4 {
      assert_eq!(out[i * 4 + 3], 0xFF, "alpha must be 0xFF at pixel {i}");
      assert_eq!(out[i * 4], 0xFF, "R must be 0xFF at pixel {i}");
    }
  }

  #[test]
  fn rgba_opaque_high_bit_bits9_downshift_correct() {
    // BITS=9, shift=1. Value 510 >> 1 = 255.
    let g = [510u16; 1];
    let b = [0u16; 1];
    let r = [0u16; 1];
    let mut out = [0u8; 4];
    gbr_to_rgba_opaque_high_bit_row::<9>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[1], 255); // G
    assert_eq!(out[3], 0xFF); // alpha
  }

  // ---- gbr_to_rgba_opaque_u16_high_bit_row: u16 RGBA with constant alpha ---

  #[test]
  fn rgba_opaque_u16_high_bit_bits10_alpha_is_1023() {
    let g = [500u16; 2];
    let b = [200u16; 2];
    let r = [800u16; 2];
    let mut out = [0u16; 8];
    gbr_to_rgba_opaque_u16_high_bit_row::<10>(&g, &b, &r, &mut out, 2);
    let opaque = (1u16 << 10) - 1; // 1023
    assert_eq!(out[3], opaque); // pixel 0 alpha
    assert_eq!(out[7], opaque); // pixel 1 alpha
    assert_eq!(out[0], 800); // R
    assert_eq!(out[1], 500); // G
    assert_eq!(out[2], 200); // B
  }

  #[test]
  fn rgba_opaque_u16_high_bit_bits16_alpha_is_65535() {
    let g = [0u16; 1];
    let b = [0u16; 1];
    let r = [0u16; 1];
    let mut out = [0u16; 4];
    gbr_to_rgba_opaque_u16_high_bit_row::<16>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], u16::MAX);
  }

  #[test]
  fn rgba_opaque_u16_high_bit_bits9_alpha_is_511() {
    let g = [0u16; 1];
    let b = [0u16; 1];
    let r = [0u16; 1];
    let mut out = [0u16; 4];
    gbr_to_rgba_opaque_u16_high_bit_row::<9>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], (1u16 << 9) - 1); // 511
  }

  // ---- gbra_to_rgba_high_bit_row: u8 RGBA with source alpha ----------------

  #[test]
  fn gbra_rgba_high_bit_bits10_source_alpha_downshifted() {
    // BITS=10, shift=2. Alpha value 512 >> 2 = 128.
    let g = [0u16; 1];
    let b = [0u16; 1];
    let r = [0u16; 1];
    let a = [512u16; 1];
    let mut out = [0u8; 4];
    gbra_to_rgba_high_bit_row::<10>(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[3], 128); // alpha = 512 >> 2
  }

  #[test]
  fn gbra_rgba_high_bit_bits10_max_alpha_is_0xff() {
    let max = (1u16 << 10) - 1;
    let g = [max; 2];
    let b = [max; 2];
    let r = [max; 2];
    let a = [max; 2];
    let mut out = [0u8; 8];
    gbra_to_rgba_high_bit_row::<10>(&g, &b, &r, &a, &mut out, 2);
    for i in 0..2 {
      assert_eq!(out[i * 4 + 3], 0xFF, "alpha must be 0xFF at pixel {i}");
    }
  }

  #[test]
  fn gbra_rgba_high_bit_bits14_channel_reorder_and_shift() {
    // BITS=14, shift=6. R=16320 >> 6 = 255, G=0, B=0, A=8192 >> 6 = 128.
    let g = [0u16; 1];
    let b = [0u16; 1];
    let r = [16320u16; 1];
    let a = [8192u16; 1];
    let mut out = [0u8; 4];
    gbra_to_rgba_high_bit_row::<14>(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[0], 255); // R
    assert_eq!(out[1], 0); // G
    assert_eq!(out[2], 0); // B
    assert_eq!(out[3], 128); // A
  }

  // ---- gbra_to_rgba_u16_high_bit_row: u16 RGBA with source alpha -----------

  #[test]
  fn gbra_rgba_u16_high_bit_source_alpha_preserved() {
    let g = [100u16; 1];
    let b = [200u16; 1];
    let r = [300u16; 1];
    let a = [777u16; 1];
    let mut out = [0u16; 4];
    gbra_to_rgba_u16_high_bit_row::<10>(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[0], 300); // R
    assert_eq!(out[1], 100); // G
    assert_eq!(out[2], 200); // B
    assert_eq!(out[3], 777); // A — preserved as-is
  }

  #[test]
  fn gbra_rgba_u16_high_bit_bits16_all_channels_preserved() {
    let g = [10000u16; 2];
    let b = [20000u16; 2];
    let r = [30000u16; 2];
    let a = [40000u16; 2];
    let mut out = [0u16; 8];
    gbra_to_rgba_u16_high_bit_row::<16>(&g, &b, &r, &a, &mut out, 2);
    for i in 0..2 {
      assert_eq!(out[i * 4], 30000);
      assert_eq!(out[i * 4 + 1], 10000);
      assert_eq!(out[i * 4 + 2], 20000);
      assert_eq!(out[i * 4 + 3], 40000);
    }
  }

  // ---- Round-trip parity: high-bit u8 output matches 8-bit source ----------

  #[test]
  fn rgb_high_bit_bits10_parity_with_scaled_8bit() {
    // val=128 in 8-bit; in 10-bit: 128 << 2 = 512. 512 >> 2 = 128.
    let val: u16 = 128u16 << 2;
    let g = [val; 8];
    let b = [val; 8];
    let r = [val; 8];
    let mut out = [0u8; 24];
    gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out, 8);
    assert!(out.iter().all(|&v| v == 128));
  }

  #[test]
  fn rgb_high_bit_bits12_parity_with_scaled_8bit() {
    // val=200 in 8-bit; in 12-bit: 200 << 4 = 3200. 3200 >> 4 = 200.
    let val: u16 = 200u16 << 4;
    let g = [val; 4];
    let b = [val; 4];
    let r = [val; 4];
    let mut out = [0u8; 12];
    gbr_to_rgb_high_bit_row::<12>(&g, &b, &r, &mut out, 4);
    assert!(out.iter().all(|&v| v == 200));
  }

  // ---- Upper-bits masking tests --------------------------------------------
  // These tests verify that samples with bits above BITS set are masked
  // correctly before processing, ensuring scalar/SIMD produce identical output.

  #[test]
  fn gbr_to_rgb_high_bit_masks_upper_bits_bits10() {
    // BITS=10, mask=0x03FF. Input 0x0CFF has upper bits set.
    // masked = 0x0CFF & 0x03FF = 0x00FF = 255. 255 >> 2 = 63 as u8.
    let dirty: u16 = 0x0CFF;
    let clean = dirty & 0x03FF;
    let expected_u8 = (clean >> 2) as u8;
    let g = [dirty; 1];
    let b = [dirty; 1];
    let r = [dirty; 1];
    let mut out = [0u8; 3];
    gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out, 1);
    assert_eq!(
      out[0], expected_u8,
      "R must equal masked-then-shifted value"
    );
    assert_eq!(
      out[1], expected_u8,
      "G must equal masked-then-shifted value"
    );
    assert_eq!(
      out[2], expected_u8,
      "B must equal masked-then-shifted value"
    );
  }

  #[test]
  fn gbr_to_rgb_high_bit_masks_upper_bits_multiple_widths_bits10() {
    // Width sweep: [1, 7, 8, 16, 17, 32, 33, 64, 128, 130].
    let dirty: u16 = 0x0500; // BITS=10: mask&0x0500 = 0x0100=256; 256>>2=64.
    let clean = dirty & 0x03FF;
    let expected_u8 = (clean >> 2) as u8;
    for w in [1usize, 7, 8, 16, 17, 32, 33, 64, 128, 130] {
      let g = vec![dirty; w];
      let b = vec![dirty; w];
      let r = vec![dirty; w];
      let mut out = vec![0u8; w * 3];
      gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out, w);
      for i in 0..w {
        assert_eq!(out[i * 3], expected_u8, "R pixel {i} wrong at width {w}");
        assert_eq!(
          out[i * 3 + 1],
          expected_u8,
          "G pixel {i} wrong at width {w}"
        );
        assert_eq!(
          out[i * 3 + 2],
          expected_u8,
          "B pixel {i} wrong at width {w}"
        );
      }
    }
  }

  #[test]
  fn gbra_to_rgba_high_bit_masks_upper_bits_alpha_bits10() {
    // Verify that the alpha channel is also masked before shifting.
    // BITS=10: dirty_alpha = 0x0800 | 512 = 0x0A00 = 2560.
    // masked = 2560 & 0x03FF = 0x0200 = 512. 512 >> 2 = 128.
    let dirty_rgb: u16 = 0x0400; // masked = 0 (upper bit only). 0>>2=0.
    let dirty_alpha: u16 = 0x0A00; // masked = 0x0200 = 512. 512>>2=128.
    let g = [dirty_rgb; 1];
    let b = [dirty_rgb; 1];
    let r = [dirty_rgb; 1];
    let a = [dirty_alpha; 1];
    let mut out = [0u8; 4];
    gbra_to_rgba_high_bit_row::<10>(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[0], 0, "R (dirty, masked to 0)");
    assert_eq!(out[1], 0, "G (dirty, masked to 0)");
    assert_eq!(out[2], 0, "B (dirty, masked to 0)");
    assert_eq!(out[3], 128, "alpha must be masked then shifted");
  }

  #[test]
  fn gbr_to_rgb_u16_high_bit_masks_upper_bits_bits10() {
    // u16-output: verify that masked sample is in the output (not raw dirty value).
    let dirty: u16 = 0x0CFF;
    let clean = dirty & 0x03FF; // = 0x00FF = 255
    let g = [dirty; 1];
    let b = [dirty; 1];
    let r = [dirty; 1];
    let mut out = [0u16; 3];
    gbr_to_rgb_u16_high_bit_row::<10>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], clean, "R u16 must be masked value");
    assert_eq!(out[1], clean, "G u16 must be masked value");
    assert_eq!(out[2], clean, "B u16 must be masked value");
  }

  #[test]
  fn gbra_to_rgba_u16_high_bit_masks_upper_bits_bits10() {
    // u16 RGBA output: all channels masked.
    let dirty: u16 = 0x0555; // BITS=10: masked = 0x0555 & 0x03FF = 0x0155 = 341.
    let clean = dirty & 0x03FF;
    let g = [dirty; 1];
    let b = [dirty; 1];
    let r = [dirty; 1];
    let a = [dirty; 1];
    let mut out = [0u16; 4];
    gbra_to_rgba_u16_high_bit_row::<10>(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[0], clean, "R u16 must be masked");
    assert_eq!(out[1], clean, "G u16 must be masked");
    assert_eq!(out[2], clean, "B u16 must be masked");
    assert_eq!(out[3], clean, "A u16 must be masked");
  }

  #[test]
  fn gbr_to_rgba_opaque_high_bit_masks_upper_bits_bits10() {
    // u8 RGBA opaque: RGB channels masked, alpha always 0xFF.
    let dirty: u16 = 0x0CFF; // masked & 0x03FF = 0x00FF = 255. 255>>2=63.
    let clean = dirty & 0x03FF;
    let expected_u8 = (clean >> 2) as u8;
    let g = [dirty; 1];
    let b = [dirty; 1];
    let r = [dirty; 1];
    let mut out = [0u8; 4];
    gbr_to_rgba_opaque_high_bit_row::<10>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], expected_u8, "R must be masked");
    assert_eq!(out[1], expected_u8, "G must be masked");
    assert_eq!(out[2], expected_u8, "B must be masked");
    assert_eq!(out[3], 0xFF, "alpha must be 0xFF");
  }

  #[test]
  fn gbr_to_rgba_opaque_u16_high_bit_masks_upper_bits_bits10() {
    // u16 RGBA opaque: RGB masked, alpha is opaque mask value.
    let dirty: u16 = 0x0CFF; // masked = 0x00FF = 255.
    let clean = dirty & 0x03FF;
    let g = [dirty; 1];
    let b = [dirty; 1];
    let r = [dirty; 1];
    let mut out = [0u16; 4];
    gbr_to_rgba_opaque_u16_high_bit_row::<10>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], clean, "R u16 must be masked");
    assert_eq!(out[1], clean, "G u16 must be masked");
    assert_eq!(out[2], clean, "B u16 must be masked");
    assert_eq!(out[3], (1u16 << 10) - 1, "alpha must be opaque 1023");
  }

  #[test]
  fn gbr_to_rgb_high_bit_bits16_mask_is_noop() {
    // For BITS=16, mask = 0xFFFF. The AND is a no-op; verify that u16::MAX
    // samples pass through correctly (masked == original).
    let val = u16::MAX;
    let g = [val; 2];
    let b = [val; 2];
    let r = [val; 2];
    let mut out = [0u8; 6];
    gbr_to_rgb_high_bit_row::<16>(&g, &b, &r, &mut out, 2);
    assert!(
      out.iter().all(|&v| v == 0xFF),
      "BITS=16: max sample => 0xFF"
    );
  }

  // ---- Cross-path consistency: direct GBRA vs masked RGB + separate alpha ---

  #[test]
  fn gbra_to_rgba_high_bit_cross_path_consistency_bits10() {
    // With upper-bits-set alpha: direct gbra_to_rgba == manual masking.
    // BITS=10, dirty_alpha = 0x0800 | 0x0100 = 0x0900; masked=0x0100=256; 256>>2=64.
    let dirty_alpha: u16 = 0x0900;
    let clean_alpha = dirty_alpha & 0x03FF; // 256
    let expected_a_u8 = (clean_alpha >> 2) as u8; // 64

    let r = [400u16; 1]; // in-range sample: 400 >> 2 = 100
    let g = [200u16; 1];
    let b = [100u16; 1];
    let a = [dirty_alpha; 1];

    // Direct path
    let mut out_direct = [0u8; 4];
    gbra_to_rgba_high_bit_row::<10>(&g, &b, &r, &a, &mut out_direct, 1);

    // Manual path: apply mask to alpha, call with clean value
    let a_clean = [clean_alpha; 1];
    let mut out_manual = [0u8; 4];
    gbra_to_rgba_high_bit_row::<10>(&g, &b, &r, &a_clean, &mut out_manual, 1);

    assert_eq!(
      out_direct, out_manual,
      "direct GBRA path must match manually-masked alpha path"
    );
    assert_eq!(out_direct[3], expected_a_u8, "alpha channel value");
  }

  // ---- gbr_to_luma_u16_high_bit_row: native-depth luma --------------------

  #[test]
  fn luma_u16_high_bit_bits10_max_white_not_banded() {
    // BITS=10: max = 1023. Old path gave (255 as u16) << 2 = 1020, not 1023.
    // New kernel must produce a value near 1023 for all-white input.
    let max = (1u16 << 10) - 1; // 1023
    let g = [max; 1];
    let b = [max; 1];
    let r = [max; 1];
    let mut out = [0u16; 1];
    gbr_to_luma_u16_high_bit_row::<10>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    // For BT.709 full-range all-white: Y = round(Kr*max + Kg*max + Kb*max).
    // = round((6966 + 23436 + 2366) / 32768 * 1023) ≈ round(32768/32768 * 1023) = 1023.
    assert!(
      out[0] >= 1020,
      "max-white luma_u16 must be near 1023 (was {}, old banded path gives 1020)",
      out[0]
    );
    assert!(
      out[0] <= 1023,
      "max-white luma_u16 must not exceed native max"
    );
  }

  #[test]
  fn luma_u16_high_bit_bits12_max_white_not_banded() {
    // BITS=12: max = 4095. Old path: (255 as u16) << 4 = 4080.
    // New kernel should give a value in [4090, 4095].
    let max = (1u16 << 12) - 1; // 4095
    let g = [max; 1];
    let b = [max; 1];
    let r = [max; 1];
    let mut out = [0u16; 1];
    gbr_to_luma_u16_high_bit_row::<12>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt601, true);
    assert!(
      out[0] >= 4090,
      "max-white luma_u16 bits12 must be near 4095 (was {})",
      out[0]
    );
    assert!(out[0] <= 4095, "must not exceed native max");
  }

  #[test]
  fn luma_u16_high_bit_bits16_max_white_not_banded() {
    // BITS=16: max = 65535. Old path: (255 as u16) << 8 = 65280.
    // New kernel (i64 path) should give a value in [65520, 65535].
    let max = u16::MAX;
    let g = [max; 1];
    let b = [max; 1];
    let r = [max; 1];
    let mut out = [0u16; 1];
    gbr_to_luma_u16_high_bit_row::<16>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert!(
      out[0] >= 65520,
      "max-white luma_u16 bits16 must be near 65535 (was {}), old banded gives 65280",
      out[0]
    );
    // u16 is bounded by type; kernel clamp ensures value stays in [0, native_max].
  }

  #[test]
  fn luma_u16_high_bit_bits10_neutral_gray_midrange() {
    // BITS=10: mid = 512. Luma of neutral gray ≈ 512.
    let mid = 512u16;
    let g = [mid; 1];
    let b = [mid; 1];
    let r = [mid; 1];
    let mut out = [0u16; 1];
    gbr_to_luma_u16_high_bit_row::<10>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, true);
    assert!(
      out[0] >= 510 && out[0] <= 514,
      "neutral gray luma_u16 must be ~512 (was {})",
      out[0]
    );
  }

  #[test]
  fn luma_u16_high_bit_bits10_zero_gives_zero() {
    let g = [0u16; 2];
    let b = [0u16; 2];
    let r = [0u16; 2];
    let mut out = [0xFFFFu16; 2];
    gbr_to_luma_u16_high_bit_row::<10>(&g, &b, &r, &mut out, 2, ColorMatrix::Bt709, true);
    assert!(out.iter().all(|&v| v == 0), "all-black must give zero luma");
  }

  #[test]
  fn luma_u16_high_bit_bits10_full_range_vs_limited_range() {
    // For mid-gray input, limited-range luma should be in [16<<2, 235<<2] = [64, 940].
    let mid = 512u16;
    let g = [mid; 1];
    let b = [mid; 1];
    let r = [mid; 1];
    let mut out_full = [0u16; 1];
    let mut out_lim = [0u16; 1];
    gbr_to_luma_u16_high_bit_row::<10>(&g, &b, &r, &mut out_full, 1, ColorMatrix::Bt601, true);
    gbr_to_luma_u16_high_bit_row::<10>(&g, &b, &r, &mut out_lim, 1, ColorMatrix::Bt601, false);
    let y_off = 16u16 << 2; // 64
    let y_max = 235u16 << 2; // 940
    assert!(
      out_full[0] >= out_lim[0],
      "limited-range luma <= full-range luma for mid gray"
    );
    assert!(
      out_lim[0] >= y_off,
      "limited-range must be >= 64 (was {})",
      out_lim[0]
    );
    assert!(
      out_lim[0] <= y_max,
      "limited-range must be <= 940 (was {})",
      out_lim[0]
    );
  }

  #[test]
  fn luma_u16_high_bit_bits16_limited_range_black_gives_min_offset() {
    // BITS=16: all-black limited-range should give Y_off = 16 << 8 = 4096.
    let g = [0u16; 1];
    let b = [0u16; 1];
    let r = [0u16; 1];
    let mut out = [0u16; 1];
    gbr_to_luma_u16_high_bit_row::<16>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, false);
    let y_off = 16u16 << 8; // 4096
    assert_eq!(
      out[0], y_off,
      "all-black limited-range must give Y_off={y_off}"
    );
  }

  // ---- Limited-range native-depth scaling boundary regression ---------
  //
  // Codex review #4233323791 caught that an earlier 8-bit `219/255`
  // Q15 scale collapsed the top ~250 input codes onto the y_max clamp
  // at BITS=16 (e.g. y_full = 65300 → 60160 instead of ~59955),
  // destroying highlight gradation. The current implementation uses
  // native-depth scaling `(y_full × range) / native_max` where
  // `range = 219 << (BITS - 8)` and `native_max = (1 << BITS) - 1`.

  #[test]
  fn luma_u16_high_bit_bits16_limited_range_max_white_maps_to_y_max() {
    // BITS=16, all-white in: y_full clamps to native_max=65535;
    // y_lim = 4096 + 65535 × 56064 / 65535 = 60160 = 235 << 8.
    let g = [u16::MAX; 1];
    let b = [u16::MAX; 1];
    let r = [u16::MAX; 1];
    let mut out = [0u16; 1];
    gbr_to_luma_u16_high_bit_row::<16>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, false);
    let y_max = 235u16 << 8; // 60160
    assert_eq!(
      out[0], y_max,
      "all-white limited-range must give Y_max={y_max}"
    );
  }

  #[test]
  fn luma_u16_high_bit_bits16_limited_range_near_white_keeps_gradation() {
    // BITS=16, BT.709 luma weights ≈ Kr=0.2126, Kg=0.7152, Kb=0.0722.
    // Setting all 3 channels equal makes the matrix multiply produce
    // y_full ≈ input, so we can probe the limited-range scaling at
    // specific y_full values. y_full = 65000 / 65300 / 65500 must each
    // produce a distinct y_lim — the buggy 8-bit ratio would clamp the
    // top two onto y_max=60160, destroying the gradation codex flagged.
    for &v in &[65000u16, 65300, 65500] {
      let g = [v; 1];
      let b = [v; 1];
      let r = [v; 1];
      let mut out = [0u16; 1];
      gbr_to_luma_u16_high_bit_row::<16>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, false);
      // Native-depth limited-range: y_lim = 4096 + v × 56064 / 65535
      let expected = 4096 + ((v as u64 * 56064 + 65535 / 2) / 65535) as u16;
      // Allow ±1 LSB for matrix-multiply rounding (BT.709 weights aren't
      // exactly 1.0 even with all channels equal; tiny Q15 residue).
      let diff = (out[0] as i32 - expected as i32).abs();
      assert!(
        diff <= 1,
        "v={v} expected ≈{expected} got {} (diff {diff})",
        out[0]
      );
      // The bug-collapse value would be 60160 (y_max). Reject any
      // result above 60160 — that means we re-introduced the clamp.
      assert!(
        out[0] < 60160 || (v >= 65500),
        "limited-range must not clamp at v={v} (got {})",
        out[0]
      );
    }
  }

  #[test]
  fn luma_u16_high_bit_bits10_limited_range_endpoints() {
    // BITS=10: y_off=64 (=16<<2), y_max=940 (=235<<2), native_max=1023.
    // BT.709 luma at all-equal channels passes y_full ≈ input through.
    // Test endpoint values: 0 → 64, 1023 → 940.
    let cases: &[(u16, u16)] = &[(0, 64), (1023, 940)];
    for &(input, expected) in cases {
      let g = [input; 1];
      let b = [input; 1];
      let r = [input; 1];
      let mut out = [0u16; 1];
      gbr_to_luma_u16_high_bit_row::<10>(&g, &b, &r, &mut out, 1, ColorMatrix::Bt709, false);
      let diff = (out[0] as i32 - expected as i32).abs();
      assert!(
        diff <= 1,
        "BITS=10 input={input} expected ≈{expected} got {}",
        out[0]
      );
    }
  }
}
