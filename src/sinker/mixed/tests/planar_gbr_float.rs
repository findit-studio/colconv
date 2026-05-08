//! Task 9 — cross-format / HDR / α / round-half-up integration tests for
//! the Tier 10 float planar GBR source family.
//!
//! Covers:
//! - Cross-format planar parity (Gbrpf32 vs Rgbf32; Gbrpf16 vs Rgbf16).
//! - Gbrapf32 Strategy A+ byte-equivalence against independent-kernel path.
//! - HDR pass-through (NaN, Inf, values > 1.0 preserved on lossless paths).
//! - f32 α plane reaches RGBA output slot 3 untouched (memcpy semantics).
//! - f16-narrowing saturation: values > 65504 → +Inf; values < -65504 → -Inf.
//! - Round-half-up regression: {0.5/255, 2.5/255, 4.5/255} → {1, 3, 5} on
//!   every backend (scalar + NEON + SSE4.1 + AVX2 + AVX-512).
//! - Gbrapf32 Strategy A+ for rgba_u8, rgba_u16, rgba_f32, rgba_f16 outputs.
//! - 32-bit overflow guard (only on 32-bit targets).

use super::*;
use crate::sinker::MixedSinker;

// ---- Helpers ---------------------------------------------------------------

/// Build a Gbrpf32 frame with constant colour `(r, g, b)` across
/// all `width × height` pixels. Returns `(g_plane, b_plane, r_plane)`.
fn solid_gbrpf32_planes(
  width: usize,
  height: usize,
  r: f32,
  g: f32,
  b: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  let n = width * height;
  (std::vec![g; n], std::vec![b; n], std::vec![r; n])
}

/// Build a Gbrpf32 frame from random-ish per-pixel f32 values.
/// Values cycle through `vals` (repeated across G, B, R planes).
fn patterned_gbrpf32_planes(
  width: usize,
  height: usize,
  vals: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  let n = width * height;
  let g: Vec<f32> = (0..n).map(|i| vals[i % vals.len()]).collect();
  let b: Vec<f32> = (0..n).map(|i| vals[(i + 1) % vals.len()]).collect();
  let r: Vec<f32> = (0..n).map(|i| vals[(i + 2) % vals.len()]).collect();
  (g, b, r)
}

/// Build an α plane from `vals` (cycled), then build a Gbrapf32 frame.
fn patterned_alpha_f32(width: usize, height: usize, seed: u8) -> Vec<f32> {
  let n = width * height;
  let mut buf = std::vec![0u8; n];
  pseudo_random_u8(&mut buf, seed as u32);
  buf.iter().map(|&b| b as f32 / 255.0).collect()
}

/// Build packed Rgbf32 data from the same (r, g, b) values as
/// `patterned_gbrpf32_planes`, so the two frames contain identical RGB data.
fn patterned_rgbf32_packed(width: usize, height: usize, vals: &[f32]) -> Vec<f32> {
  let n = width * height;
  let mut buf = std::vec![0.0f32; n * 3];
  for i in 0..n {
    buf[i * 3] = vals[(i + 2) % vals.len()]; // R
    buf[i * 3 + 1] = vals[i % vals.len()]; // G
    buf[i * 3 + 2] = vals[(i + 1) % vals.len()]; // B
  }
  buf
}

// ---- Cross-format planar parity (6 tests) ----------------------------------

/// Gbrpf32 and Rgbf32 with the same per-pixel RGB values produce byte-identical
/// u8 RGB output.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgb_matches_rgbf32_rgb() {
  let w = 32usize;
  let h = 8usize;
  // Use a set that spans in-range, HDR-clamped, and negative-clamped.
  let vals = [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5];
  let (gp, bp, rp) = patterned_gbrpf32_planes(w, h, &vals);
  let packed = patterned_rgbf32_packed(w, h, &vals);

  let gbrp_src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();
  let rgbf32_src = Rgbf32Frame::try_new(&packed, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb_gbrp = std::vec![0u8; w * h * 3];
  let mut rgb_rgbf32 = std::vec![0u8; w * h * 3];

  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb(&mut rgb_gbrp)
      .unwrap();
    gbrpf32_to(&gbrp_src, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Rgbf32>::new(w, h)
      .with_rgb(&mut rgb_rgbf32)
      .unwrap();
    rgbf32_to(&rgbf32_src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(
    rgb_gbrp, rgb_rgbf32,
    "Gbrpf32 vs Rgbf32 RGB u8 output must be byte-identical"
  );
}

/// Gbrpf16 and Rgbf16 with the same per-pixel values produce byte-identical
/// u8 RGB output.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_rgb_matches_rgbf16_rgb() {
  let w = 32usize;
  let h = 8usize;
  let vals_f32 = [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5];
  let vals_f16: Vec<half::f16> = vals_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
  let n = w * h;

  // G, B, R planes (planar; same cycling as patterned_gbrpf32_planes)
  let gp: Vec<half::f16> = (0..n).map(|i| vals_f16[i % vals_f16.len()]).collect();
  let bp: Vec<half::f16> = (0..n).map(|i| vals_f16[(i + 1) % vals_f16.len()]).collect();
  let rp: Vec<half::f16> = (0..n).map(|i| vals_f16[(i + 2) % vals_f16.len()]).collect();

  // Packed Rgbf16: R=vals[(i+2)%len], G=vals[i%len], B=vals[(i+1)%len]
  let packed: Vec<half::f16> = (0..n * 3)
    .map(|j| {
      let i = j / 3;
      let ch = j % 3;
      match ch {
        0 => vals_f16[(i + 2) % vals_f16.len()], // R
        1 => vals_f16[i % vals_f16.len()],       // G
        _ => vals_f16[(i + 1) % vals_f16.len()], // B
      }
    })
    .collect();

  let gbrp_src = Gbrpf16Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();
  let rgbf16_src = Rgbf16Frame::try_new(&packed, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb_gbrp = std::vec![0u8; w * h * 3];
  let mut rgb_rgbf16 = std::vec![0u8; w * h * 3];

  {
    let mut sink = MixedSinker::<Gbrpf16>::new(w, h)
      .with_rgb(&mut rgb_gbrp)
      .unwrap();
    gbrpf16_to(&gbrp_src, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Rgbf16>::new(w, h)
      .with_rgb(&mut rgb_rgbf16)
      .unwrap();
    rgbf16_to(&rgbf16_src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(
    rgb_gbrp, rgb_rgbf16,
    "Gbrpf16 vs Rgbf16 RGB u8 output must be byte-identical"
  );
}

/// Gbrapf32 Strategy A+ (with_rgb + with_rgba) produces byte-identical output
/// to two independent-kernel runs.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let vals = [0.1f32, 0.3, 0.5, 0.7, 0.9, 1.1, -0.1];
  let (gp, bp, rp) = patterned_gbrpf32_planes(w, h, &vals);
  let ap = patterned_alpha_f32(w, h, 0xAB);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // Independent: two separate sinks.
  let mut rgb_ref = std::vec![0u8; w * h * 3];
  let mut rgba_ref = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // Strategy A+ combo.
  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  assert_eq!(
    rgb_combo, rgb_ref,
    "Strategy A+ RGB must match independent RGB"
  );
  assert_eq!(
    rgba_combo, rgba_ref,
    "Strategy A+ RGBA must match independent RGBA"
  );
}

/// Gbrapf32 with_rgba_f32: α plane passes through lossless (memcpy semantics
/// for the α slot; no scaling, no clamping on the f32-output path).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_f32_lossless_alpha() {
  let w = 16usize;
  let h = 4usize;
  // Use a mix including out-of-[0,1] α values to confirm lossless pass-through.
  let ap: Vec<f32> = (0..w * h)
    .map(|i| match i % 4 {
      0 => 0.0,
      1 => 0.5,
      2 => 1.0,
      _ => 1.5, // HDR α
    })
    .collect();
  let g = std::vec![0.5f32; w * h];
  let b = std::vec![0.5f32; w * h];
  let r = std::vec![0.5f32; w * h];
  let src = Gbrapf32Frame::try_new(
    &g, &b, &r, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_out = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_f32(&mut rgba_out)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // Slot 3 of each pixel must be the source α value, bit-exact.
  for i in 0..w * h {
    let got = rgba_out[i * 4 + 3];
    let want = ap[i];
    assert_eq!(
      got.to_bits(),
      want.to_bits(),
      "α at pixel {i}: got {got} want {want}"
    );
  }
}

/// Gbrpf32 with_rgb_f16 is byte-equivalent to with_rgb_f32 followed by
/// per-element `half::f16::from_f32`. Locks in the fused narrow correctness.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgb_f16_matches_caller_side_narrow() {
  let w = 64usize;
  let h = 4usize;
  let vals = [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, -0.5];
  let (gp, bp, rp) = patterned_gbrpf32_planes(w, h, &vals);
  let src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // Fused path.
  let mut rgb_f16_fused = std::vec![half::f16::ZERO; w * h * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb_f16(&mut rgb_f16_fused)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  // Caller-side: f32 first, then narrow.
  let mut rgb_f32 = std::vec![0.0f32; w * h * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  let rgb_f16_caller: Vec<half::f16> = rgb_f32.iter().map(|&v| half::f16::from_f32(v)).collect();

  assert_eq!(
    rgb_f16_fused, rgb_f16_caller,
    "fused with_rgb_f16 must equal caller-side narrow of with_rgb_f32"
  );
}

/// Gbrpf32 with_rgb_f16 saturates HDR values > 65504 to f16::INFINITY /
/// f16::NEG_INFINITY (regression guard for the narrowing path).
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgb_f16_saturates_hdr_to_f16_inf() {
  let w = 4usize;
  let h = 1usize;
  // 70000 > 65504 (f16 max representable finite) — must saturate to +Inf.
  // -70000 < -65504 — must saturate to -Inf.
  let r = std::vec![70000.0f32, -70000.0, 0.5, 1.0];
  let g = std::vec![70000.0f32, -70000.0, 0.5, 1.0];
  let b = std::vec![70000.0f32, -70000.0, 0.5, 1.0];
  let src =
    Gbrpf32Frame::try_new(&g, &b, &r, w as u32, h as u32, w as u32, w as u32, w as u32).unwrap();

  let mut out = std::vec![half::f16::ZERO; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb_f16(&mut out)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  // Pixel 0: R/G/B all +Inf.
  assert_eq!(out[0], half::f16::INFINITY, "px0 R must be +Inf");
  assert_eq!(out[1], half::f16::INFINITY, "px0 G must be +Inf");
  assert_eq!(out[2], half::f16::INFINITY, "px0 B must be +Inf");
  // Pixel 1: R/G/B all -Inf.
  assert_eq!(out[3], half::f16::NEG_INFINITY, "px1 R must be -Inf");
  assert_eq!(out[4], half::f16::NEG_INFINITY, "px1 G must be -Inf");
  assert_eq!(out[5], half::f16::NEG_INFINITY, "px1 B must be -Inf");
}

// ---- HDR pass-through (4 tests) --------------------------------------------

/// Gbrpf32 with_rgb_f32 preserves HDR > 1.0, NaN, and Inf bit-exact.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgb_f32_preserves_hdr_nan_inf() {
  let w = 8usize;
  let h = 1usize;
  let special: [f32; 8] = [
    1.5,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    100.0,
    -0.5,
    0.0,
    1.0,
  ];
  let gp: Vec<f32> = special.to_vec();
  let bp: Vec<f32> = special.to_vec();
  let rp: Vec<f32> = special.to_vec();
  let src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut out = std::vec![0.0f32; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgb_f32(&mut out)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  // Output R channel per pixel = rp[i] bit-exact (including NaN/Inf).
  for i in 0..w {
    let r_out = out[i * 3];
    let r_src = rp[i];
    // NaN != NaN, so compare bit patterns.
    assert_eq!(
      r_out.to_bits(),
      r_src.to_bits(),
      "R at pixel {i}: expected {r_src} (bits {:08x}) got {r_out} (bits {:08x})",
      r_src.to_bits(),
      r_out.to_bits()
    );
  }
}

/// Gbrapf32 with_rgba_f32 preserves α plane bit-exact through Strategy A+.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_f32_preserves_hdr_alpha() {
  let w = 8usize;
  let h = 2usize;
  let ap: Vec<f32> = (0..w * h)
    .map(|i| match i % 4 {
      0 => 0.0,
      1 => f32::INFINITY,
      2 => 1.5,
      _ => 0.75,
    })
    .collect();
  let g = std::vec![0.5f32; w * h];
  let b = std::vec![0.5f32; w * h];
  let r = std::vec![0.5f32; w * h];
  let src = Gbrapf32Frame::try_new(
    &g, &b, &r, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_out = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_f32(&mut rgba_out)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  for i in 0..w * h {
    let got = rgba_out[i * 4 + 3];
    let want = ap[i];
    assert_eq!(
      got.to_bits(),
      want.to_bits(),
      "α at pixel {i}: got {got} (bits {:08x}) want {want} (bits {:08x})",
      got.to_bits(),
      want.to_bits()
    );
  }
}

/// Gbrpf16 with_rgb_f16 is lossless — no conversion, bit-exact interleave.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_rgb_f16_lossless_passthrough() {
  let w = 16usize;
  let h = 4usize;
  let vals_f32 = [0.0f32, 0.25, 0.5, 1.0, 1.5, 100.0, -0.5, f32::INFINITY];
  let vals_f16: Vec<half::f16> = vals_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
  let n = w * h;
  let gp: Vec<half::f16> = (0..n).map(|i| vals_f16[i % vals_f16.len()]).collect();
  let bp: Vec<half::f16> = (0..n).map(|i| vals_f16[(i + 1) % vals_f16.len()]).collect();
  let rp: Vec<half::f16> = (0..n).map(|i| vals_f16[(i + 2) % vals_f16.len()]).collect();
  let src = Gbrpf16Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut out = std::vec![half::f16::ZERO; w * h * 3];
  {
    let mut sink = MixedSinker::<Gbrpf16>::new(w, h)
      .with_rgb_f16(&mut out)
      .unwrap();
    gbrpf16_to(&src, &mut sink).unwrap();
  }

  // Packed output order is R, G, B.
  for i in 0..n {
    assert_eq!(out[i * 3].to_bits(), rp[i].to_bits(), "R at pixel {i}");
    assert_eq!(out[i * 3 + 1].to_bits(), gp[i].to_bits(), "G at pixel {i}");
    assert_eq!(out[i * 3 + 2].to_bits(), bp[i].to_bits(), "B at pixel {i}");
  }
}

/// Gbrpf32 (no α) with_rgba_f32 fills slot 3 with α = 1.0 exactly.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_rgba_f32_alpha_max_filled_correctly() {
  let w = 16usize;
  let h = 4usize;
  let (gp, bp, rp) = solid_gbrpf32_planes(w, h, 0.5, 0.25, 0.75);
  let src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_out = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
      .with_rgba_f32(&mut rgba_out)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  for i in 0..w * h {
    let alpha = rgba_out[i * 4 + 3];
    assert_eq!(
      alpha.to_bits(),
      1.0f32.to_bits(),
      "α slot at pixel {i}: expected 1.0 got {alpha}"
    );
  }
}

// ---- Round-half-up regression (5 tests, one per backend) -------------------
//
// Feed inputs where banker's rounding would diverge from round-half-up:
//   0.5/255 → 1 (not 0)   [banker's rounds 0.5 to even = 0]
//   2.5/255 → 3 (not 2)   [banker's rounds 2.5 to even = 2 on some impls]
//   4.5/255 → 5 (not 4)
// All backends must produce {1, 3, 5}.

fn round_half_up_check(rgb_out: &[u8], prefix: &str) {
  // The three input pixels are replicated across R, G, B identically.
  // rgb_out[0..3] = pixel 0 (R,G,B) from 0.5/255; rgb_out[3..6] = pixel 1 (2.5/255); etc.
  assert_eq!(
    rgb_out[0], 1,
    "{prefix}: R[0] from 0.5/255 must be 1 (round-half-up)"
  );
  assert_eq!(rgb_out[1], 1, "{prefix}: G[0] from 0.5/255 must be 1");
  assert_eq!(rgb_out[2], 1, "{prefix}: B[0] from 0.5/255 must be 1");
  assert_eq!(rgb_out[3], 3, "{prefix}: R[1] from 2.5/255 must be 3");
  assert_eq!(rgb_out[4], 3, "{prefix}: G[1] from 2.5/255 must be 3");
  assert_eq!(rgb_out[5], 3, "{prefix}: B[1] from 2.5/255 must be 3");
  assert_eq!(rgb_out[6], 5, "{prefix}: R[2] from 4.5/255 must be 5");
  assert_eq!(rgb_out[7], 5, "{prefix}: G[2] from 4.5/255 must be 5");
  assert_eq!(rgb_out[8], 5, "{prefix}: B[2] from 4.5/255 must be 5");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_scalar() {
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(false); // scalar path
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "scalar");
}

#[test]
#[cfg(target_arch = "aarch64")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_neon() {
  if !crate::row::neon_available() {
    return;
  }
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(true);
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "neon");
}

#[test]
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_sse41() {
  if !is_x86_feature_detected!("sse4.1") {
    return;
  }
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(true);
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "sse41");
}

#[test]
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_avx2() {
  if !is_x86_feature_detected!("avx2") {
    return;
  }
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(true);
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "avx2");
}

#[test]
#[cfg(target_arch = "x86_64")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_round_half_up_at_boundaries_avx512() {
  if !is_x86_feature_detected!("avx512f") {
    return;
  }
  let inputs = [0.5f32 / 255.0, 2.5 / 255.0, 4.5 / 255.0];
  let w = inputs.len();
  let gp = inputs.to_vec();
  let bp = inputs.to_vec();
  let rp = inputs.to_vec();
  let src =
    Gbrpf32Frame::try_new(&gp, &bp, &rp, w as u32, 1, w as u32, w as u32, w as u32).unwrap();

  let mut rgb_out = std::vec![0u8; w * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(w, 1)
      .with_rgb(&mut rgb_out)
      .unwrap()
      .with_simd(true);
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  round_half_up_check(&rgb_out, "avx512");
}

// ---- Strategy A+ byte-equivalence tests (5 tests) --------------------------
//
// For Gbrapf32: rgba_u8, rgba_u16, rgba_f32, rgba_f16 — verify Strategy A+
// (when both with_rgb AND the rgba accessor are attached) produces byte-identical
// output to the independent-kernel path (only the rgba accessor attached).

fn make_gbrapf32_src(w: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
  let vals = [0.1f32, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8];
  let (gp, bp, rp) = patterned_gbrpf32_planes(w, h, &vals);
  let ap = patterned_alpha_f32(w, h, 0xC3);
  (gp, bp, rp, ap)
}

/// Strategy A+ for rgba (u8): combo path = independent path.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_u8_strategy_a_plus_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let (gp, bp, rp, ap) = make_gbrapf32_src(w, h);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_ref = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0u8; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combo, rgba_ref,
    "Strategy A+ rgba_u8 must match independent kernel"
  );
}

/// Strategy A+ for rgba_u16: combo path = independent path.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_u16_strategy_a_plus_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let (gp, bp, rp, ap) = make_gbrapf32_src(w, h);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_ref = std::vec![0u16; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_u16(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0u16; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba_u16(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combo, rgba_ref,
    "Strategy A+ rgba_u16 must match independent kernel"
  );
}

/// Strategy A+ for rgba_f32: combo path = independent path.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_f32_strategy_a_plus_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let (gp, bp, rp, ap) = make_gbrapf32_src(w, h);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_ref = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_f32(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![0.0f32; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba_f32(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  // f32: compare bit-patterns to catch NaN equality issues.
  let bits_ref: Vec<u32> = rgba_ref.iter().map(|v| v.to_bits()).collect();
  let bits_combo: Vec<u32> = rgba_combo.iter().map(|v| v.to_bits()).collect();
  assert_eq!(
    bits_combo, bits_ref,
    "Strategy A+ rgba_f32 must match independent kernel"
  );
}

/// Strategy A+ for rgba_f16: combo path = independent path.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_rgba_f16_strategy_a_plus_matches_independent_kernel() {
  let w = 32usize;
  let h = 8usize;
  let (gp, bp, rp, ap) = make_gbrapf32_src(w, h);
  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_ref = std::vec![half::f16::ZERO; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgba_f16(&mut rgba_ref)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let mut rgb_combo = std::vec![0u8; w * h * 3];
  let mut rgba_combo = std::vec![half::f16::ZERO; w * h * 4];
  {
    let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
      .with_rgb(&mut rgb_combo)
      .unwrap()
      .with_rgba_f16(&mut rgba_combo)
      .unwrap();
    gbrapf32_to(&src, &mut sink).unwrap();
  }

  let bits_ref: Vec<u16> = rgba_ref.iter().map(|v| v.to_bits()).collect();
  let bits_combo: Vec<u16> = rgba_combo.iter().map(|v| v.to_bits()).collect();
  assert_eq!(
    bits_combo, bits_ref,
    "Strategy A+ rgba_f16 must match independent kernel"
  );
}

// ---- HOST_NATIVE_BE routing parity (codex PR #84 Findings 1-3) -------------
//
// LE-host routing-equivalence and host-native sinker-contract tests for the
// `Gbrpf32` / `Gbrapf32` / `Gbrpf16` / `Gbrapf16` sinkers. Mirrors the
// `Rgbf32` / `Rgbf16` sinker tests added for PR #83's `dcf40a3` (sinker
// HOST_NATIVE_BE routing) and `c3a6478` (dispatch f16-widen HOST_NATIVE_BE
// routing).
//
// On a LE host `HOST_NATIVE_BE = false`, so the kernel-level test below is
// a routing sanity check (proving the dispatcher / sinker substitute the
// correct `BE` template parameter); BE-host correctness of the routing is
// verified by the existing row-kernel BE parity tests in
// `src/row/arch/*/tests/` and by the contract tests below (which assert
// host-native pass-through end-to-end on every host).

/// Kernel-level test: on a LE host, `gbrpf32_to_*::<false>` and
/// `gbrpf32_to_*::<HOST_NATIVE_BE>` must produce byte-identical output for
/// every Tier 10 float planar GBR dispatcher across every output type
/// (u8 RGB / u8 RGBA / u16 RGB / u16 RGBA / f32 lossless). Width 33 covers
/// SIMD main loop + scalar tail across every backend; width 5 covers tail-
/// only paths.
///
/// **LE-host-only**: gated on `target_endian = "little"`. On a BE host
/// `::<false>` decodes the host-native fixture as LE-encoded (byte-swap)
/// while `::<HOST_NATIVE_BE> == ::<true>` decodes as BE (no swap), so the
/// outputs diverge by design. This sinker-routing-equivalence claim is
/// specifically about the LE host pattern; BE-host correctness of the
/// routing change is verified by the contract tests below and the row-
/// kernel BE parity tests in `src/row/arch/*/tests/`.
#[test]
#[cfg(target_endian = "little")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_kernel_host_native_be_matches_false_on_le_host() {
  use crate::row::{
    gbrpf32_to_rgb_f32_row, gbrpf32_to_rgb_row, gbrpf32_to_rgb_u16_row, gbrpf32_to_rgba_row,
    gbrpf32_to_rgba_u16_row,
  };

  // Sinker-layer `HOST_NATIVE_BE` mirrors `cfg!(target_endian = "big")`; on
  // the LE-host gate this evaluates to `false`.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

  // Width 33: SIMD main loop + scalar tail. Width 5: tail-only path. Run both
  // to cover SIMD-tail-aware backends.
  for w in [5usize, 7usize, 33usize] {
    let mut gp = std::vec![0.0f32; w];
    let mut bp = std::vec![0.0f32; w];
    let mut rp = std::vec![0.0f32; w];
    for (i, (g, (b, r))) in gp
      .iter_mut()
      .zip(bp.iter_mut().zip(rp.iter_mut()))
      .enumerate()
    {
      *g = match i % 5 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.75,
        _ => -0.25,
      };
      *b = match i % 5 {
        0 => 0.25,
        1 => 0.75,
        2 => 1.5,
        3 => 0.0,
        _ => -0.5,
      };
      *r = match i % 5 {
        0 => 1.0,
        1 => 0.5,
        2 => 0.0,
        3 => -0.25,
        _ => 1.25,
      };
    }

    let mut rgb_false = std::vec![0u8; w * 3];
    let mut rgb_host = std::vec![0u8; w * 3];
    gbrpf32_to_rgb_row::<false>(&gp, &bp, &rp, &mut rgb_false, w, true);
    gbrpf32_to_rgb_row::<HOST_NATIVE_BE>(&gp, &bp, &rp, &mut rgb_host, w, true);
    assert_eq!(rgb_false, rgb_host, "u8 RGB diverges (w = {w})");

    let mut rgba_false = std::vec![0u8; w * 4];
    let mut rgba_host = std::vec![0u8; w * 4];
    gbrpf32_to_rgba_row::<false>(&gp, &bp, &rp, &mut rgba_false, w, true);
    gbrpf32_to_rgba_row::<HOST_NATIVE_BE>(&gp, &bp, &rp, &mut rgba_host, w, true);
    assert_eq!(rgba_false, rgba_host, "u8 RGBA diverges (w = {w})");

    let mut u16_false = std::vec![0u16; w * 3];
    let mut u16_host = std::vec![0u16; w * 3];
    gbrpf32_to_rgb_u16_row::<false>(&gp, &bp, &rp, &mut u16_false, w, true);
    gbrpf32_to_rgb_u16_row::<HOST_NATIVE_BE>(&gp, &bp, &rp, &mut u16_host, w, true);
    assert_eq!(u16_false, u16_host, "u16 RGB diverges (w = {w})");

    let mut u16a_false = std::vec![0u16; w * 4];
    let mut u16a_host = std::vec![0u16; w * 4];
    gbrpf32_to_rgba_u16_row::<false>(&gp, &bp, &rp, &mut u16a_false, w, true);
    gbrpf32_to_rgba_u16_row::<HOST_NATIVE_BE>(&gp, &bp, &rp, &mut u16a_host, w, true);
    assert_eq!(u16a_false, u16a_host, "u16 RGBA diverges (w = {w})");

    let mut f32_false = std::vec![0.0f32; w * 3];
    let mut f32_host = std::vec![0.0f32; w * 3];
    gbrpf32_to_rgb_f32_row::<false>(&gp, &bp, &rp, &mut f32_false, w, true);
    gbrpf32_to_rgb_f32_row::<HOST_NATIVE_BE>(&gp, &bp, &rp, &mut f32_host, w, true);
    assert_eq!(f32_false, f32_host, "f32 RGB diverges (w = {w})");
  }
}

/// Sinker contract test: feeding host-native `f32` planes through
/// [`MixedSinker<Gbrpf32>`] must produce the same output every other sinker
/// would expect from a host-native source — specifically, `with_rgb_f32`
/// must be bit-exact identical to the source on every host. Documents the
/// public-API contract that the [`HOST_NATIVE_BE`] routing fix preserves.
/// Pairs with the kernel-level test above; together they cover both the
/// dispatch boundary and the public sinker boundary.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_sinker_host_native_contract_lossless_passthrough() {
  let w = 16usize;
  let h = 4usize;
  // Mix HDR, in-range, and negative values — the f32 lossless path must
  // round-trip them bit-exact on every host.
  let mut gp = std::vec![0.0f32; w * h];
  let mut bp = std::vec![0.0f32; w * h];
  let mut rp = std::vec![0.0f32; w * h];
  for (i, (g, (b, r))) in gp
    .iter_mut()
    .zip(bp.iter_mut().zip(rp.iter_mut()))
    .enumerate()
  {
    *g = match i % 4 {
      0 => 0.5,
      1 => 1.5,
      2 => -0.25,
      _ => 100.0,
    };
    *b = match i % 4 {
      0 => 0.0,
      1 => 0.25,
      2 => 1.0,
      _ => f32::INFINITY,
    };
    *r = match i % 4 {
      0 => 1.0,
      1 => -1.0,
      2 => 65505.0,
      _ => 0.5,
    };
  }
  let src = Gbrpf32Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  // rgb_f32 lossless: each pixel `(R, G, B)` packed in source plane order.
  let mut rgb_f32 = std::vec![0.0f32; w * h * 3];
  let mut sink = MixedSinker::<Gbrpf32>::new(w, h)
    .with_rgb_f32(&mut rgb_f32)
    .unwrap();
  gbrpf32_to(&src, &mut sink).unwrap();

  // The lossless scatter writes `(R, G, B)` per pixel in plane-index order.
  // Bit-exact equality on every host. Buggy `::<false>` routing on a BE host
  // would byte-swap the output here; the fix keeps it bit-exact.
  for i in 0..(w * h) {
    assert_eq!(rgb_f32[i * 3], rp[i], "R mismatch at idx {i}");
    assert_eq!(rgb_f32[i * 3 + 1], gp[i], "G mismatch at idx {i}");
    assert_eq!(rgb_f32[i * 3 + 2], bp[i], "B mismatch at idx {i}");
  }
}

/// Same as [`gbrpf32_kernel_host_native_be_matches_false_on_le_host`] but
/// for the `Gbrpf16` family — covers both `use_simd = false` (dispatch's
/// scalar widen-fallback) and `use_simd = true` (SIMD widen path) at tail
/// widths 5, 7, 33 to exercise every backend's main loop + scalar tail.
#[test]
#[cfg(target_endian = "little")]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_kernel_host_native_be_matches_false_on_le_host() {
  use crate::row::{
    gbrpf16_to_rgb_row, gbrpf16_to_rgb_u16_row, gbrpf16_to_rgba_row, gbrpf16_to_rgba_u16_row,
  };

  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

  for w in [5usize, 7usize, 33usize] {
    let gp: Vec<half::f16> = (0..w)
      .map(|i| {
        half::f16::from_f32(match i % 5 {
          0 => 0.0,
          1 => 0.5,
          2 => 1.0,
          3 => 1.75,
          _ => -0.25,
        })
      })
      .collect();
    let bp: Vec<half::f16> = (0..w)
      .map(|i| {
        half::f16::from_f32(match i % 5 {
          0 => 0.25,
          1 => 0.75,
          2 => 1.5,
          3 => 0.0,
          _ => -0.5,
        })
      })
      .collect();
    let rp: Vec<half::f16> = (0..w)
      .map(|i| {
        half::f16::from_f32(match i % 5 {
          0 => 1.0,
          1 => 0.5,
          2 => 0.0,
          3 => -0.25,
          _ => 1.25,
        })
      })
      .collect();

    // Both `use_simd = false` and `use_simd = true` to cover dispatch's
    // scalar widen-fallback and the SIMD widen path on every backend.
    for use_simd in [false, true] {
      let mut rgb_false = std::vec![0u8; w * 3];
      let mut rgb_host = std::vec![0u8; w * 3];
      gbrpf16_to_rgb_row::<false>(&gp, &bp, &rp, &mut rgb_false, w, use_simd);
      gbrpf16_to_rgb_row::<HOST_NATIVE_BE>(&gp, &bp, &rp, &mut rgb_host, w, use_simd);
      assert_eq!(
        rgb_false, rgb_host,
        "u8 RGB diverges (w = {w}, use_simd = {use_simd})"
      );

      let mut rgba_false = std::vec![0u8; w * 4];
      let mut rgba_host = std::vec![0u8; w * 4];
      gbrpf16_to_rgba_row::<false>(&gp, &bp, &rp, &mut rgba_false, w, use_simd);
      gbrpf16_to_rgba_row::<HOST_NATIVE_BE>(&gp, &bp, &rp, &mut rgba_host, w, use_simd);
      assert_eq!(
        rgba_false, rgba_host,
        "u8 RGBA diverges (w = {w}, use_simd = {use_simd})"
      );

      let mut u16_false = std::vec![0u16; w * 3];
      let mut u16_host = std::vec![0u16; w * 3];
      gbrpf16_to_rgb_u16_row::<false>(&gp, &bp, &rp, &mut u16_false, w, use_simd);
      gbrpf16_to_rgb_u16_row::<HOST_NATIVE_BE>(&gp, &bp, &rp, &mut u16_host, w, use_simd);
      assert_eq!(
        u16_false, u16_host,
        "u16 RGB diverges (w = {w}, use_simd = {use_simd})"
      );

      let mut u16a_false = std::vec![0u16; w * 4];
      let mut u16a_host = std::vec![0u16; w * 4];
      gbrpf16_to_rgba_u16_row::<false>(&gp, &bp, &rp, &mut u16a_false, w, use_simd);
      gbrpf16_to_rgba_u16_row::<HOST_NATIVE_BE>(&gp, &bp, &rp, &mut u16a_host, w, use_simd);
      assert_eq!(
        u16a_false, u16a_host,
        "u16 RGBA diverges (w = {w}, use_simd = {use_simd})"
      );
    }
  }
}

/// Sinker contract: host-native `half::f16` source through [`MixedSinker<Gbrpf16>`]
/// `with_rgb_f16` must round-trip the planes bit-exact on every host. The
/// `::<HOST_NATIVE_BE>` routing keeps the lossless interleave a no-op in the
/// BE-load layer; the buggy `::<false>` routing on a BE host would byte-swap
/// every f16 element.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_sinker_host_native_contract_lossless_passthrough() {
  let w = 16usize;
  let h = 4usize;
  let gp: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.5,
        1 => 1.5,
        2 => -0.25,
        _ => 100.0,
      })
    })
    .collect();
  let bp: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 0.0,
        1 => 0.25,
        2 => 1.0,
        _ => 65000.0,
      })
    })
    .collect();
  let rp: Vec<half::f16> = (0..w * h)
    .map(|i| {
      half::f16::from_f32(match i % 4 {
        0 => 1.0,
        1 => -1.0,
        2 => 0.125,
        _ => 0.5,
      })
    })
    .collect();

  let src = Gbrpf16Frame::try_new(
    &gp, &bp, &rp, w as u32, h as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgb_f16 = std::vec![half::f16::ZERO; w * h * 3];
  let mut sink = MixedSinker::<Gbrpf16>::new(w, h)
    .with_rgb_f16(&mut rgb_f16)
    .unwrap();
  gbrpf16_to(&src, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(
      rgb_f16[i * 3].to_bits(),
      rp[i].to_bits(),
      "R mismatch at idx {i}"
    );
    assert_eq!(
      rgb_f16[i * 3 + 1].to_bits(),
      gp[i].to_bits(),
      "G mismatch at idx {i}"
    );
    assert_eq!(
      rgb_f16[i * 3 + 2].to_bits(),
      bp[i].to_bits(),
      "B mismatch at idx {i}"
    );
  }
}

/// Sinker contract: [`MixedSinker<Gbrapf16>`] `with_rgba_f16` must round-trip
/// the source α plane bit-exact alongside the G/B/R planes, on every host.
/// Validates Strategy A+ alpha consistency under the `HOST_NATIVE_BE`
/// routing — the previous mix-mode (LE-decoded RGB + host-native α) is gone.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf16_sinker_host_native_contract_lossless_passthrough_with_alpha() {
  let w = 16usize;
  let h = 4usize;
  let gp: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.1 + (i as f32) * 0.001))
    .collect();
  let bp: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.2 + (i as f32) * 0.002))
    .collect();
  let rp: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.3 + (i as f32) * 0.003))
    .collect();
  let ap: Vec<half::f16> = (0..w * h)
    .map(|i| half::f16::from_f32(0.5 + (i as f32) * 0.001))
    .collect();

  let src = Gbrapf16Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_f16 = std::vec![half::f16::ZERO; w * h * 4];
  let mut sink = MixedSinker::<Gbrapf16>::new(w, h)
    .with_rgba_f16(&mut rgba_f16)
    .unwrap();
  gbrapf16_to(&src, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(rgba_f16[i * 4].to_bits(), rp[i].to_bits(), "R idx {i}");
    assert_eq!(rgba_f16[i * 4 + 1].to_bits(), gp[i].to_bits(), "G idx {i}");
    assert_eq!(rgba_f16[i * 4 + 2].to_bits(), bp[i].to_bits(), "B idx {i}");
    assert_eq!(rgba_f16[i * 4 + 3].to_bits(), ap[i].to_bits(), "A idx {i}");
  }
}

/// Sinker contract: [`MixedSinker<Gbrapf32>`] `with_rgba_f32` lossless
/// pass-through plus α — confirms Strategy A+ alpha consistency when the
/// f32 RGB chain routes via `HOST_NATIVE_BE`. The α plane is host-native
/// f32, also routed via `HOST_NATIVE_BE`, eliminating any mix-mode.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrapf32_sinker_host_native_contract_lossless_passthrough_with_alpha() {
  let w = 16usize;
  let h = 4usize;
  let mut gp = std::vec![0.0f32; w * h];
  let mut bp = std::vec![0.0f32; w * h];
  let mut rp = std::vec![0.0f32; w * h];
  let mut ap = std::vec![0.0f32; w * h];
  for i in 0..(w * h) {
    gp[i] = 0.1 + (i as f32) * 0.001;
    bp[i] = 0.2 + (i as f32) * 0.002;
    rp[i] = 0.3 + (i as f32) * 0.003;
    ap[i] = 0.5 + (i as f32) * 0.0005;
  }

  let src = Gbrapf32Frame::try_new(
    &gp, &bp, &rp, &ap, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
  )
  .unwrap();

  let mut rgba_f32 = std::vec![0.0f32; w * h * 4];
  let mut sink = MixedSinker::<Gbrapf32>::new(w, h)
    .with_rgba_f32(&mut rgba_f32)
    .unwrap();
  gbrapf32_to(&src, &mut sink).unwrap();

  for i in 0..(w * h) {
    assert_eq!(rgba_f32[i * 4], rp[i], "R idx {i}");
    assert_eq!(rgba_f32[i * 4 + 1], gp[i], "G idx {i}");
    assert_eq!(rgba_f32[i * 4 + 2], bp[i], "B idx {i}");
    assert_eq!(rgba_f32[i * 4 + 3], ap[i], "A idx {i}");
  }
}

// ---- 32-bit overflow guards ------------------------------------------------
//
// Feeding width = usize::MAX / 2 + 1 to a dispatcher must panic with a message
// containing "overflows usize". These tests are only meaningful on 32-bit
// targets where usize = u32. On 64-bit targets the width would be ~2 GiB which
// won't trigger the overflow guard, so the tests are compiled-out.

#[cfg(target_pointer_width = "32")]
#[test]
#[should_panic(expected = "overflows usize")]
fn gbr_float_dispatch_panics_on_width_overflow_gbrpf32_rgb() {
  let bad_width = usize::MAX / 2 + 1;
  // Allocate 1-element planes — the overflow panic fires before the plane-len
  // check so the short planes won't be the cause of the panic.
  let g = [0.0f32; 1];
  let b = [0.0f32; 1];
  let r = [0.0f32; 1];
  let mut out = [0u8; 3];
  crate::row::gbrpf32_to_rgb_row::<false>(&g, &b, &r, &mut out, bad_width, false);
}

#[cfg(target_pointer_width = "32")]
#[test]
#[should_panic(expected = "overflows usize")]
fn gbr_float_dispatch_panics_on_width_overflow_gbrpf32_rgba() {
  let bad_width = usize::MAX / 2 + 1;
  let g = [0.0f32; 1];
  let b = [0.0f32; 1];
  let r = [0.0f32; 1];
  let mut out = [0u8; 4];
  crate::row::gbrpf32_to_rgba_row::<false>(&g, &b, &r, &mut out, bad_width, false);
}

#[cfg(target_pointer_width = "32")]
#[test]
#[should_panic(expected = "overflows usize")]
fn gbr_float_dispatch_panics_on_width_overflow_gbrpf32_rgb_u16() {
  let bad_width = usize::MAX / 2 + 1;
  let g = [0.0f32; 1];
  let b = [0.0f32; 1];
  let r = [0.0f32; 1];
  let mut out = [0u16; 3];
  crate::row::gbrpf32_to_rgb_u16_row::<false>(&g, &b, &r, &mut out, bad_width, false);
}

#[cfg(target_pointer_width = "32")]
#[test]
#[should_panic(expected = "overflows usize")]
fn gbr_float_dispatch_panics_on_width_overflow_gbrpf32_rgba_u16() {
  let bad_width = usize::MAX / 2 + 1;
  let g = [0.0f32; 1];
  let b = [0.0f32; 1];
  let r = [0.0f32; 1];
  let mut out = [0u16; 4];
  crate::row::gbrpf32_to_rgba_u16_row::<false>(&g, &b, &r, &mut out, bad_width, false);
}
