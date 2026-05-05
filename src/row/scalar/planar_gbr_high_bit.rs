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
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  let shift = BITS - 8;
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = (r[x] >> shift) as u8;
    rgb_out[dst + 1] = (g[x] >> shift) as u8;
    rgb_out[dst + 2] = (b[x] >> shift) as u8;
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
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_u16_out[dst] = r[x];
    rgb_u16_out[dst + 1] = g[x];
    rgb_u16_out[dst + 2] = b[x];
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
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let shift = BITS - 8;
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = (r[x] >> shift) as u8;
    rgba_out[dst + 1] = (g[x] >> shift) as u8;
    rgba_out[dst + 2] = (b[x] >> shift) as u8;
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
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  let opaque: u16 = ((1u32 << BITS) - 1) as u16;
  for x in 0..width {
    let dst = x * 4;
    rgba_u16_out[dst] = r[x];
    rgba_u16_out[dst + 1] = g[x];
    rgba_u16_out[dst + 2] = b[x];
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
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let shift = BITS - 8;
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = (r[x] >> shift) as u8;
    rgba_out[dst + 1] = (g[x] >> shift) as u8;
    rgba_out[dst + 2] = (b[x] >> shift) as u8;
    rgba_out[dst + 3] = (a[x] >> shift) as u8;
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
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(
    rgba_u16_out.len() >= width * 4,
    "rgba_u16_out row too short"
  );
  for x in 0..width {
    let dst = x * 4;
    rgba_u16_out[dst] = r[x];
    rgba_u16_out[dst + 1] = g[x];
    rgba_u16_out[dst + 2] = b[x];
    rgba_u16_out[dst + 3] = a[x];
  }
}

// ---- Unit tests -----------------------------------------------------------

#[cfg(all(test, any(feature = "std", feature = "alloc")))]
mod tests {
  use super::*;

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
    let g = vec![val; 8];
    let b = vec![val; 8];
    let r = vec![val; 8];
    let mut out = [0u8; 24];
    gbr_to_rgb_high_bit_row::<10>(&g, &b, &r, &mut out, 8);
    assert!(out.iter().all(|&v| v == 128));
  }

  #[test]
  fn rgb_high_bit_bits12_parity_with_scaled_8bit() {
    // val=200 in 8-bit; in 12-bit: 200 << 4 = 3200. 3200 >> 4 = 200.
    let val: u16 = 200u16 << 4;
    let g = vec![val; 4];
    let b = vec![val; 4];
    let r = vec![val; 4];
    let mut out = [0u8; 12];
    gbr_to_rgb_high_bit_row::<12>(&g, &b, &r, &mut out, 4);
    assert!(out.iter().all(|&v| v == 200));
  }
}
