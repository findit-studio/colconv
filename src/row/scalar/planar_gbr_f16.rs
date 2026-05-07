//! f16-native lossless I/O kernels for planar GBR f16 sources.
//!
//! **Scope:** This file handles `half::f16` source planes for lossless
//! interleave and α-scatter output only. For f16-source → integer, luma, or
//! HSV outputs the dispatch layer widens each f16 plane to f32 in a scratch
//! buffer at row entry, then calls the corresponding `gbrpf32_to_*` kernel
//! from [`super::planar_gbr_float`]. No separate f16-source kernels are needed
//! for those paths.
//!
//! ## Endian support
//!
//! All `<const BE: bool>` kernels treat the source planes as opaque `u16`
//! bit-patterns (which they already are for the lossless f16 paths).
//! When `BE = true` each u16 element is byte-swapped before being written to
//! the interleaved output buffer — i.e. we load a big-endian f16 bit-pattern
//! and emit it as host-native f16.
//!
//! ## Kernels in this file
//!
//! | Kernel | In | Out | Notes |
//! |---|---|---|---|
//! | `gbrpf16_to_rgb_f16_row` | G, B, R f16 planes | `R, G, B` f16 | pure interleave, lossless |
//! | `gbrpf16_to_rgba_f16_row` | G, B, R f16 planes | `R, G, B, A` f16 | α = f16(1.0) |
//! | `gbrapf16_to_rgba_f16_row` | G, B, R, A f16 planes | `R, G, B, A` f16 | source α pass-through |
//! | `copy_alpha_plane_f16` | α f16 plane | slot 3 of `R,G,B,A` f16 buf | lossless α scatter |
//!
//! Output order is **R, G, B** per pixel (FFmpeg `AV_PIX_FMT_RGBA64` / packed
//! RGB convention). No arithmetic is performed — these are pure gather-scatter
//! kernels over opaque `u16` bit-patterns.

// Kernels are not yet consumed by any sinker (Task 8 wires MixedSinker impls).
#![cfg_attr(not(test), allow(dead_code))]

// ---- shared BE helper -------------------------------------------------------

/// Load a single `half::f16` sample with optional BE byte-swap.
///
/// When `BE = true` the two bytes of the f16 bit-pattern are reversed (i.e.
/// we load a big-endian f16 from disk and convert to host-native). When
/// `BE = false` the value is returned as-is. The dead branch is eliminated
/// by the compiler when the caller is monomorphized.
#[inline(always)]
fn load_f16<const BE: bool>(plane: &[half::f16], i: usize) -> half::f16 {
  if BE {
    half::f16::from_bits(plane[i].to_bits().swap_bytes())
  } else {
    plane[i]
  }
}

// ---- Gbrpf16 → f16 RGB (lossless interleave) --------------------------------

/// Interleaves planar G/B/R `half::f16` rows into packed `R, G, B`
/// **`half::f16`**.
///
/// Pure gather-scatter — no conversion. HDR values, NaN, and Inf are
/// preserved bit-exact. Output order is **R, G, B** per pixel.
///
/// `BE = true`: each f16 element is byte-swapped (BE → host-native) before
/// being written to the interleaved output.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgb_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = load_f16::<BE>(r, x);
    rgb_out[dst + 1] = load_f16::<BE>(g, x);
    rgb_out[dst + 2] = load_f16::<BE>(b, x);
  }
}

// ---- Gbrpf16 → f16 RGBA (opaque α = f16(1.0)) ------------------------------

/// Interleaves planar G/B/R `half::f16` rows into packed `R, G, B, A`
/// **`half::f16`** with constant opaque α = `half::f16::from_f32(1.0)`.
///
/// Used for `Gbrpf16` sources (no α plane) when `with_rgba_f16` is requested.
///
/// `BE = true`: each f16 element is byte-swapped (BE → host-native) before
/// being written. α is always host-native f16(1.0) regardless of `BE`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrpf16_to_rgba_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  let one_f16 = half::f16::from_f32(1.0);
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = load_f16::<BE>(r, x);
    rgba_out[dst + 1] = load_f16::<BE>(g, x);
    rgba_out[dst + 2] = load_f16::<BE>(b, x);
    rgba_out[dst + 3] = one_f16;
  }
}

// ---- Gbrapf16 → f16 RGBA (source α pass-through) ----------------------------

/// Interleaves planar G/B/R/A `half::f16` rows into packed `R, G, B, A`
/// **`half::f16`** with source α.
///
/// Pure gather-scatter. All four channels including α are copied losslessly —
/// HDR, NaN, and Inf preserved bit-exact.
///
/// `BE = true`: each f16 element (including α) is byte-swapped before write.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbrapf16_to_rgba_f16_row<const BE: bool>(
  g: &[half::f16],
  b: &[half::f16],
  r: &[half::f16],
  a: &[half::f16],
  rgba_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = load_f16::<BE>(r, x);
    rgba_out[dst + 1] = load_f16::<BE>(g, x);
    rgba_out[dst + 2] = load_f16::<BE>(b, x);
    rgba_out[dst + 3] = load_f16::<BE>(a, x);
  }
}

// ---- copy_alpha_plane_f16 (lossless α scatter) ------------------------------

/// Scatters a `half::f16` α plane into slot 3 of a packed `R, G, B, A`
/// **`half::f16`** output buffer.
///
/// Only slot 3 of every 4-element tuple is written; R, G, B slots are
/// untouched. Lossless — HDR, NaN, and Inf in the α plane are preserved
/// bit-exact.
// Only called from the `mod tests` block which is gated on `feature = "std"`.
// Under `cargo test --no-default-features` the test module is compiled out,
// leaving the function without callers; suppress the resulting lint there.
#[cfg_attr(not(feature = "std"), expect(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn copy_alpha_plane_f16(alpha: &[half::f16], rgba_out: &mut [half::f16], width: usize) {
  debug_assert!(alpha.len() >= width, "alpha plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for n in 0..width {
    rgba_out[n * 4 + 3] = alpha[n];
  }
}

// ---- Unit tests ------------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  // ---- helper: byte-swap a slice of f16 to simulate BE source ----------------

  fn be_encode_f16(src: &[half::f16]) -> std::vec::Vec<half::f16> {
    src.iter().map(|v| half::f16::from_bits(v.to_bits().swap_bytes())).collect()
  }

  // ---- gbrpf16_to_rgb_f16_row ----------------------------------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf16_to_rgb_f16_channel_reorder() {
    // G=0.25, B=0.5, R=1.0 → packed R=1.0, G=0.25, B=0.5
    let g = [half::f16::from_f32(0.25)];
    let b = [half::f16::from_f32(0.5)];
    let r = [half::f16::from_f32(1.0)];
    let mut out = vec![half::f16::ZERO; 3];
    gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[0], half::f16::from_f32(1.0), "R");
    assert_eq!(out[1], half::f16::from_f32(0.25), "G");
    assert_eq!(out[2], half::f16::from_f32(0.5), "B");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf16_to_rgb_f16_hdr_preserved() {
    // HDR value 2.5 passes through losslessly.
    let hdr = half::f16::from_f32(2.5);
    let g = [hdr];
    let b = [half::f16::from_f32(0.0)];
    let r = [half::f16::from_f32(0.0)];
    let mut out = vec![half::f16::ZERO; 3];
    gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[1], hdr, "HDR G preserved bit-exact");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf16_to_rgb_f16_be_parity() {
    // BE-encoded source must decode to same output as LE source.
    let g = [
      half::f16::from_f32(0.0),
      half::f16::from_f32(0.25),
      half::f16::from_f32(0.5),
      half::f16::from_f32(1.0),
    ];
    let b = [
      half::f16::from_f32(0.1),
      half::f16::from_f32(0.3),
      half::f16::from_f32(0.7),
      half::f16::from_f32(0.9),
    ];
    let r = [
      half::f16::from_f32(0.5),
      half::f16::from_f32(0.8),
      half::f16::from_f32(0.2),
      half::f16::from_f32(0.6),
    ];
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let mut le_out = vec![half::f16::ZERO; 4 * 3];
    let mut be_out = vec![half::f16::ZERO; 4 * 3];
    gbrpf16_to_rgb_f16_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf16_to_rgb_f16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf16_to_rgb_f16_row must match LE");
  }

  // ---- gbrpf16_to_rgba_f16_row ---------------------------------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf16_to_rgba_f16_alpha_is_one() {
    let g = [half::f16::from_f32(0.5)];
    let b = [half::f16::from_f32(0.5)];
    let r = [half::f16::from_f32(0.5)];
    let mut out = vec![half::f16::ZERO; 4];
    gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut out, 1);
    assert_eq!(out[3], half::f16::from_f32(1.0), "alpha must be f16(1.0)");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrpf16_to_rgba_f16_be_parity() {
    let g = [
      half::f16::from_f32(0.0),
      half::f16::from_f32(0.25),
      half::f16::from_f32(0.5),
      half::f16::from_f32(1.0),
    ];
    let b = [
      half::f16::from_f32(0.1),
      half::f16::from_f32(0.3),
      half::f16::from_f32(0.7),
      half::f16::from_f32(0.9),
    ];
    let r = [
      half::f16::from_f32(0.5),
      half::f16::from_f32(0.8),
      half::f16::from_f32(0.2),
      half::f16::from_f32(0.6),
    ];
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let mut le_out = vec![half::f16::ZERO; 4 * 4];
    let mut be_out = vec![half::f16::ZERO; 4 * 4];
    gbrpf16_to_rgba_f16_row::<false>(&g, &b, &r, &mut le_out, 4);
    gbrpf16_to_rgba_f16_row::<true>(&g_be, &b_be, &r_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrpf16_to_rgba_f16_row must match LE");
  }

  // ---- gbrapf16_to_rgba_f16_row --------------------------------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrapf16_to_rgba_f16_source_alpha_passthrough() {
    let g = [half::f16::from_f32(0.25)];
    let b = [half::f16::from_f32(0.5)];
    let r = [half::f16::from_f32(0.75)];
    let a = [half::f16::from_f32(0.9)];
    let mut out = vec![half::f16::ZERO; 4];
    gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut out, 1);
    assert_eq!(out[0], half::f16::from_f32(0.75), "R");
    assert_eq!(out[1], half::f16::from_f32(0.25), "G");
    assert_eq!(out[2], half::f16::from_f32(0.5), "B");
    assert_eq!(out[3], half::f16::from_f32(0.9), "A from source");
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn gbrapf16_to_rgba_f16_be_parity() {
    let g = [
      half::f16::from_f32(0.0),
      half::f16::from_f32(0.25),
      half::f16::from_f32(0.5),
      half::f16::from_f32(1.0),
    ];
    let b = [
      half::f16::from_f32(0.1),
      half::f16::from_f32(0.3),
      half::f16::from_f32(0.7),
      half::f16::from_f32(0.9),
    ];
    let r = [
      half::f16::from_f32(0.5),
      half::f16::from_f32(0.8),
      half::f16::from_f32(0.2),
      half::f16::from_f32(0.6),
    ];
    let a = [
      half::f16::from_f32(0.2),
      half::f16::from_f32(0.4),
      half::f16::from_f32(0.6),
      half::f16::from_f32(0.8),
    ];
    let g_be = be_encode_f16(&g);
    let b_be = be_encode_f16(&b);
    let r_be = be_encode_f16(&r);
    let a_be = be_encode_f16(&a);
    let mut le_out = vec![half::f16::ZERO; 4 * 4];
    let mut be_out = vec![half::f16::ZERO; 4 * 4];
    gbrapf16_to_rgba_f16_row::<false>(&g, &b, &r, &a, &mut le_out, 4);
    gbrapf16_to_rgba_f16_row::<true>(&g_be, &b_be, &r_be, &a_be, &mut be_out, 4);
    assert_eq!(be_out, le_out, "BE gbrapf16_to_rgba_f16_row must match LE");
  }

  // ---- copy_alpha_plane_f16 ------------------------------------------------

  #[test]
  #[cfg_attr(
    miri,
    ignore = "half::f16 uses inline assembly on aarch64 unsupported by Miri"
  )]
  fn copy_alpha_plane_f16_only_writes_alpha_slot() {
    let alpha = vec![half::f16::from_f32(0.7), half::f16::from_f32(0.3)];
    let sentinel = half::f16::from_f32(0.1);
    let mut rgba = vec![sentinel; 8];
    copy_alpha_plane_f16(&alpha, &mut rgba, 2);
    // Only slot 3 written; R, G, B slots (0, 1, 2) must be untouched.
    assert_eq!(rgba[0], sentinel, "R slot 0 untouched");
    assert_eq!(rgba[1], sentinel, "G slot 0 untouched");
    assert_eq!(rgba[2], sentinel, "B slot 0 untouched");
    assert_eq!(rgba[3], half::f16::from_f32(0.7), "A slot 0");
    assert_eq!(rgba[4], sentinel, "R slot 1 untouched");
    assert_eq!(rgba[5], sentinel, "G slot 1 untouched");
    assert_eq!(rgba[6], sentinel, "B slot 1 untouched");
    assert_eq!(rgba[7], half::f16::from_f32(0.3), "A slot 1");
  }
}
