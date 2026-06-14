//! Fused-downscale coverage for the float planar-GBR family
//! ([`Gbrpf32`]).
//!
//! `Gbrpf32` scatters its G/B/R planes into a source-width packed
//! `R, G, B` f32 row and bins in float on the shared `AreaStream<f32>`.
//! Unlike `Rgbf32` (whose packed `rgbf32_*` clamp/scale kernels drive the
//! emit), the `gbr` build has no packed-float kernels, so the tail
//! de-interleaves each binned row back into G/B/R planes and runs the
//! exact direct `gbrpf32_*` kernels. Therefore:
//! - `rgb_f32` is the exact native 2x2 block mean (lossless),
//! - every output (rgb / rgba / rgb_u16 / rgba_u16 / rgba_f32 / rgb_f16 /
//!   rgba_f16 / luma / luma_u16 / hsv) matches a **direct**
//!   full-resolution `Gbrpf32` conversion of the pre-binned frame — the
//!   lossless f32 and half-float outputs at native precision, and
//!   `luma_u16` at the direct path's narrowed (8-bit-in-`u16`) precision
//!   (its kernel stages through u8) — full parity for the complete set.

use crate::{
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{Gbrpf32, gbrpf32_to},
};

const SRC: usize = 8;
const OUT: usize = 4;

/// LE-encode a host-native `f32` slice as the `*LE` Frame contract
/// requires, so a fixture reads back identically on LE (no-op) and BE
/// (byte-swap) hosts. Mirrors `as_le_rgbf32` in `resample_rgbf32`.
fn as_le_f32(host: &[f32]) -> Vec<f32> {
  host
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect()
}

/// Per-plane f32 ramps with **integer-valued** samples (so the 2x2 area
/// mean is exact in f32) that deliberately span HDR (> 1.0) and negative
/// values — the float path carries both losslessly into `rgb_f32`, while
/// the integer outputs saturate them per the direct path's clamp.
/// Returns `(g, b, r)` host-native planes.
fn gbr_planes() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  let n = SRC * SRC;
  let mut g = std::vec![0.0f32; n];
  let mut b = std::vec![0.0f32; n];
  let mut r = std::vec![0.0f32; n];
  for i in 0..n {
    let i = i as i32;
    // R: small in-range integers and HDR — saturates on the u8/u16
    // outputs, preserved exactly in rgb_f32.
    r[i as usize] = (i % 5) as f32;
    // G: large HDR values to exercise saturation.
    g[i as usize] = (100 - i) as f32;
    // B: negative samples — clamp to 0 on integer outputs, preserved
    // (with sign) in rgb_f32.
    b[i as usize] = -((i % 7) as f32);
  }
  (g, b, r)
}

fn frame<'a>(
  g: &'a [f32],
  b: &'a [f32],
  r: &'a [f32],
  w: usize,
  h: usize,
) -> crate::frame::Gbrpf32Frame<'a> {
  crate::frame::Gbrpf32Frame::try_new(g, b, r, w as u32, h as u32, w as u32, w as u32, w as u32)
    .unwrap()
}

/// Exact 2x2 block mean over host f32 values — integer-valued samples
/// divided by 4 (a power of two) are exactly representable.
fn block_mean_plane(plane: &[f32], ox: usize, oy: usize) -> f32 {
  let mut acc = 0.0f64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as f64;
    }
  }
  (acc / 4.0) as f32
}

/// De-interleave a packed `R, G, B` f32 row buffer into `(g, b, r)`
/// planes — the inverse of the source scatter, used to drive the oracle.
fn planes_from_packed(rgb: &[f32], n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
  let (mut g, mut b, mut r) = (
    std::vec![0.0f32; n],
    std::vec![0.0f32; n],
    std::vec![0.0f32; n],
  );
  for i in 0..n {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }
  (g, b, r)
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_downscale_rgb_f32_is_exact_area_mean() {
  let (g, b, r) = gbr_planes();
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrpf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  // Packed output order is R, G, B (per the source scatter).
  for oy in 0..OUT {
    for ox in 0..OUT {
      let base = (oy * OUT + ox) * 3;
      assert_eq!(rgb_f32[base], block_mean_plane(&r, ox, oy), "R ({ox},{oy})");
      assert_eq!(
        rgb_f32[base + 1],
        block_mean_plane(&g, ox, oy),
        "G ({ox},{oy})"
      );
      assert_eq!(
        rgb_f32[base + 2],
        block_mean_plane(&b, ox, oy),
        "B ({ox},{oy})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_downscale_rgb_f32_preserves_hdr_and_negative() {
  let (g, b, r) = gbr_planes();
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrpf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  assert!(
    rgb_f32.iter().any(|&v| v > 1.0),
    "binned rgb_f32 lost all HDR (> 1.0) values"
  );
  assert!(
    rgb_f32.iter().any(|&v| v < 0.0),
    "binned rgb_f32 clamped away negative values"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_all_outputs_match_direct_conversion_of_prebinned_frame() {
  // Resample SRC->OUT with every output attached, then compare against a
  // full-resolution direct Gbrpf32 conversion of the pre-binned frame —
  // the parity oracle. Every output matches the direct path, including
  // luma_u16 at the direct path's narrowed (8-bit-in-u16) precision.
  let (g, b, r) = gbr_planes();
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s_ = std::vec![0u8; OUT * OUT];
  let mut v_ = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gbrpf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  // Reference: the full-res sink over the (exact) binned f32 RGB, split
  // back into LE-wire G/B/R planes exactly as the source arrived.
  let (bg, bb, br) = planes_from_packed(&rgb_f32, OUT * OUT);
  let (bgw, bbw, brw) = (as_le_f32(&bg), as_le_f32(&bb), as_le_f32(&br));
  let mut ref_rgb = std::vec![0u8; OUT * OUT * 3];
  let mut ref_rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut ref_rgba = std::vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut ref_luma = std::vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = std::vec![0u16; OUT * OUT];
  let mut ref_h = std::vec![0u8; OUT * OUT];
  let mut ref_s = std::vec![0u8; OUT * OUT];
  let mut ref_v = std::vec![0u8; OUT * OUT];
  {
    let binned = frame(&bgw, &bbw, &brw, OUT, OUT);
    let mut sink = MixedSinker::<Gbrpf32>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gbrpf32_to(&binned, &mut sink).unwrap();
  }

  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(luma, ref_luma, "luma");
  // luma_u16 on the fused path is the direct path's narrowed (8-bit, in a
  // u16 carrier) value — `gbrpf32_to_luma_u16_row` stages through u8 — so
  // it is byte-identical to the direct Gbrpf32 `with_luma_u16`.
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16 (narrowed, full parity)");
  assert_eq!(h, ref_h, "hsv H");
  assert_eq!(s_, ref_s, "hsv S");
  assert_eq!(v_, ref_v, "hsv V");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_identity_plan_matches_new_sink() {
  let (g, b, r) = gbr_planes();
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut direct = std::vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Gbrpf32>::new(SRC, SRC)
      .with_rgb_f32(&mut direct)
      .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  let mut via_area = std::vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Gbrpf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_f32(&mut via_area)
        .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity-plan resample == direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_no_output_sink_is_a_noop() {
  // A resampled sink with no attached output neither allocates the
  // stream nor enforces sequencing — the documented legal no-op.
  let (g, b, r) = gbr_planes();
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut sink =
    MixedSinker::<Gbrpf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  gbrpf32_to(&src, &mut sink).unwrap();
  assert!(
    !sink.rgb_stream_f32_allocated(),
    "no-output sink allocated the f32 stream"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_f32_only_downscale_does_not_size_the_plane_scratch() {
  // An rgb_f32-only sink copies the binned row directly, so the G/B/R
  // plane scratch is never sized.
  let (g, b, r) = gbr_planes();
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Gbrpf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f32(&mut rgb_f32)
      .unwrap();
  gbrpf32_to(&src, &mut sink).unwrap();
  assert_eq!(
    sink.rgb_plane_scratch_capacity(),
    0,
    "rgb_f32-only sink grew the G/B/R plane scratch"
  );

  // Positive control: attaching luma_u16 (a plane-derived output) sizes
  // it to the out-width G/B/R planes.
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut sink2 =
    MixedSinker::<Gbrpf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  gbrpf32_to(&src, &mut sink2).unwrap();
  assert!(
    sink2.rgb_plane_scratch_capacity() >= OUT * 3,
    "luma_u16 output did not size the plane scratch"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_resample_reuses_stream_across_frames() {
  // begin_frame resets the f32 area stream + frozen output set, so frame
  // 2's row 0 is accepted (not rejected as out-of-sequence) and the
  // output reflects frame 2's input. Both frames share one output
  // buffer; only the input data changes.
  let (g1, b1, r1) = gbr_planes();
  // Frame 2: negate every plane so its block mean differs from frame 1.
  let g2: Vec<f32> = g1.iter().map(|&v| -v).collect();
  let b2: Vec<f32> = b1.iter().map(|&v| -v).collect();
  let r2: Vec<f32> = r1.iter().map(|&v| -v).collect();
  let (g1w, b1w, r1w) = (as_le_f32(&g1), as_le_f32(&b1), as_le_f32(&r1));
  let (g2w, b2w, r2w) = (as_le_f32(&g2), as_le_f32(&b2), as_le_f32(&r2));

  let mut out = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrpf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut out)
        .unwrap();
    gbrpf32_to(&frame(&g1w, &b1w, &r1w, SRC, SRC), &mut sink).unwrap();
    gbrpf32_to(&frame(&g2w, &b2w, &r2w, SRC, SRC), &mut sink).unwrap();
  }

  for oy in 0..OUT {
    for ox in 0..OUT {
      let base = (oy * OUT + ox) * 3;
      assert_eq!(
        out[base],
        block_mean_plane(&r2, ox, oy),
        "frame2 R ({ox},{oy})"
      );
      assert_eq!(
        out[base + 1],
        block_mean_plane(&g2, ox, oy),
        "frame2 G ({ox},{oy})"
      );
      assert_eq!(
        out[base + 2],
        block_mean_plane(&b2, ox, oy),
        "frame2 B ({ox},{oy})"
      );
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf32_float_and_f16_outputs_match_direct_conversion_of_prebinned_frame() {
  // The resample tail now produces the full output set: the lossless
  // `rgba_f32` and the half-float `rgb_f16` / `rgba_f16` derive from the
  // same de-interleaved G/B/R planes via the direct `gbrpf32_*` kernels,
  // so each is byte-identical to a full-resolution direct conversion of
  // the pre-binned frame (the parity oracle).
  let (g, b, r) = gbr_planes();
  let (gw, bw, rw) = (as_le_f32(&g), as_le_f32(&b), as_le_f32(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  // `rgb_f32` is attached only to recover the exact binned planes for the
  // oracle below; the assertions target the three newly routed outputs.
  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  let mut rgb_f16 = std::vec![half::f16::ZERO; OUT * OUT * 3];
  let mut rgba_f16 = std::vec![half::f16::ZERO; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gbrpf32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_rgba_f32(&mut rgba_f32)
        .unwrap()
        .with_rgb_f16(&mut rgb_f16)
        .unwrap()
        .with_rgba_f16(&mut rgba_f16)
        .unwrap();
    gbrpf32_to(&src, &mut sink).unwrap();
  }

  // Reference: the full-res direct sink over the (exact) binned f32 RGB,
  // split back into LE-wire G/B/R planes exactly as the source arrived.
  let (bg, bb, br) = planes_from_packed(&rgb_f32, OUT * OUT);
  let (bgw, bbw, brw) = (as_le_f32(&bg), as_le_f32(&bb), as_le_f32(&br));
  let mut ref_rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  let mut ref_rgb_f16 = std::vec![half::f16::ZERO; OUT * OUT * 3];
  let mut ref_rgba_f16 = std::vec![half::f16::ZERO; OUT * OUT * 4];
  {
    let binned = frame(&bgw, &bbw, &brw, OUT, OUT);
    let mut sink = MixedSinker::<Gbrpf32>::new(OUT, OUT)
      .with_rgba_f32(&mut ref_rgba_f32)
      .unwrap()
      .with_rgb_f16(&mut ref_rgb_f16)
      .unwrap()
      .with_rgba_f16(&mut ref_rgba_f16)
      .unwrap();
    gbrpf32_to(&binned, &mut sink).unwrap();
  }

  assert_eq!(rgba_f32, ref_rgba_f32, "rgba_f32 (lossless, full parity)");
  assert_eq!(rgb_f16, ref_rgb_f16, "rgb_f16 (full parity)");
  assert_eq!(rgba_f16, ref_rgba_f16, "rgba_f16 (full parity)");
}
