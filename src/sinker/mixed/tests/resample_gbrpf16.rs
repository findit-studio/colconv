//! Fused-downscale coverage for the half-float planar-GBR family
//! ([`Gbrpf16`]).
//!
//! There is no `AreaStream<f16>`, so `Gbrpf16` widens its G/B/R `half::f16`
//! planes to host-native f32, scatters them into a source-width packed
//! `R, G, B` f32 row, and bins in **float** on the shared
//! `AreaStream<f32>`. Per finalized output row the tail de-interleaves the
//! binned row into f32 planes, **rounds each element to `half::f16`**, and
//! runs the exact direct `gbrpf16_*` kernels. Therefore every output (rgb /
//! rgba / rgb_u16 / rgba_u16 / rgb_f32 / rgba_f32 / rgb_f16 / rgba_f16 /
//! luma / luma_u16 / hsv) is **byte-identical** to a direct full-resolution
//! `Gbrpf16` conversion of the pre-binned frame — the frame whose per-pixel
//! f16 G/B/R is the f32 area mean rounded to f16 (the oracle). For an
//! integer downscale ratio the area mean is the simple block average.

use crate::{
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{Gbrpf16, gbrpf16_to},
};

const SRC: usize = 8;
const OUT: usize = 4;

/// LE-encode a host-native `half::f16` slice as the `*LE` Frame contract
/// requires, so a fixture reads back identically on LE (no-op) and BE
/// (byte-swap) hosts. Mirrors `as_le_f32` in `resample_gbrpf32`.
fn as_le_f16(host: &[half::f16]) -> Vec<half::f16> {
  host
    .iter()
    .map(|&v| half::f16::from_bits(v.to_bits().to_le()))
    .collect()
}

/// Per-plane f16 ramps with **integer-valued** samples (so the f32 2x2
/// area mean is exact and its round to f16 is deterministic) that span
/// HDR (> 1.0) and negative values — the float path carries both into
/// the binned planes, then the round to f16 + direct `gbrpf16_*` kernels
/// reproduce the direct path's saturation on the integer/u8 outputs.
/// Returns `(g, b, r)` host-native f16 planes.
fn gbr_planes_f16() -> (Vec<half::f16>, Vec<half::f16>, Vec<half::f16>) {
  let n = SRC * SRC;
  let mut g = std::vec![half::f16::ZERO; n];
  let mut b = std::vec![half::f16::ZERO; n];
  let mut r = std::vec![half::f16::ZERO; n];
  for i in 0..n {
    let i = i as i32;
    // R: small in-range integers and HDR — saturates on u8/u16 outputs,
    // preserved (post-round) in rgb_f32.
    r[i as usize] = half::f16::from_f32((i % 5) as f32);
    // G: large HDR values to exercise saturation.
    g[i as usize] = half::f16::from_f32((100 - i) as f32);
    // B: negative samples — clamp to 0 on integer outputs, preserved
    // (with sign, post-round) in rgb_f32.
    b[i as usize] = half::f16::from_f32(-((i % 7) as f32));
  }
  (g, b, r)
}

fn frame<'a>(
  g: &'a [half::f16],
  b: &'a [half::f16],
  r: &'a [half::f16],
  w: usize,
  h: usize,
) -> crate::frame::Gbrpf16Frame<'a> {
  crate::frame::Gbrpf16Frame::try_new(g, b, r, w as u32, h as u32, w as u32, w as u32, w as u32)
    .unwrap()
}

/// Build the pre-binned f16 frame plane: average the f16 source **in f32**
/// over each 2x2 block, then round the mean to `half::f16`. This is the
/// oracle's pre-binned per-pixel value — the f32 block-mean rounded to f16.
fn prebinned_plane_f16(plane: &[half::f16]) -> Vec<half::f16> {
  let mut out = std::vec![half::f16::ZERO; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut acc = 0.0f64;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += plane[(oy * 2 + dy) * SRC + ox * 2 + dx].to_f32() as f64;
        }
      }
      out[oy * OUT + ox] = half::f16::from_f32((acc / 4.0) as f32);
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_downscale_rgb_f32_is_f16_rounded_area_mean() {
  // The direct `Gbrpf16` `rgb_f32` path widens the f16 source to f32, so
  // the fused `rgb_f32` is the f32 area mean rounded to f16, then widened
  // back to f32 — NOT the raw f32 bin. Assert exactly that.
  let (g, b, r) = gbr_planes_f16();
  let (gw, bw, rw) = (as_le_f16(&g), as_le_f16(&b), as_le_f16(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrpf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    gbrpf16_to(&src, &mut sink).unwrap();
  }
  let (pg, pb, pr) = (
    prebinned_plane_f16(&g),
    prebinned_plane_f16(&b),
    prebinned_plane_f16(&r),
  );
  // Packed output order is R, G, B.
  for oy in 0..OUT {
    for ox in 0..OUT {
      let i = oy * OUT + ox;
      let base = i * 3;
      assert_eq!(rgb_f32[base], pr[i].to_f32(), "R ({ox},{oy})");
      assert_eq!(rgb_f32[base + 1], pg[i].to_f32(), "G ({ox},{oy})");
      assert_eq!(rgb_f32[base + 2], pb[i].to_f32(), "B ({ox},{oy})");
    }
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_downscale_rgb_f32_preserves_hdr_and_negative() {
  let (g, b, r) = gbr_planes_f16();
  let (gw, bw, rw) = (as_le_f16(&g), as_le_f16(&b), as_le_f16(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrpf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap();
    gbrpf16_to(&src, &mut sink).unwrap();
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
fn gbrpf16_all_outputs_match_direct_conversion_of_prebinned_frame() {
  // Resample SRC->OUT with every output attached, then compare against a
  // full-resolution direct Gbrpf16 conversion of the pre-binned frame (the
  // f32 area mean rounded to f16) — the parity oracle. Every output is
  // byte-identical to the direct path, luma_u16 at the direct path's
  // narrowed (8-bit-in-u16) precision (its kernel stages through u8).
  let (g, b, r) = gbr_planes_f16();
  let (gw, bw, rw) = (as_le_f16(&g), as_le_f16(&b), as_le_f16(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut rgb = std::vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut rgb_f16 = std::vec![half::f16::ZERO; OUT * OUT * 3];
  let mut rgba = std::vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  let mut rgba_f16 = std::vec![half::f16::ZERO; OUT * OUT * 4];
  let mut luma = std::vec![0u8; OUT * OUT];
  let mut luma_u16 = std::vec![0u16; OUT * OUT];
  let mut h = std::vec![0u8; OUT * OUT];
  let mut s_ = std::vec![0u8; OUT * OUT];
  let mut v_ = std::vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gbrpf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgb_f32(&mut rgb_f32)
        .unwrap()
        .with_rgb_f16(&mut rgb_f16)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_rgba_f32(&mut rgba_f32)
        .unwrap()
        .with_rgba_f16(&mut rgba_f16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gbrpf16_to(&src, &mut sink).unwrap();
  }

  // Reference: the full-res direct sink over the pre-binned f16 planes
  // (f32 area mean rounded to f16), LE-encoded exactly as the source.
  let (pg, pb, pr) = (
    prebinned_plane_f16(&g),
    prebinned_plane_f16(&b),
    prebinned_plane_f16(&r),
  );
  let (pgw, pbw, prw) = (as_le_f16(&pg), as_le_f16(&pb), as_le_f16(&pr));
  let mut ref_rgb = std::vec![0u8; OUT * OUT * 3];
  let mut ref_rgb_u16 = std::vec![0u16; OUT * OUT * 3];
  let mut ref_rgb_f32 = std::vec![0.0f32; OUT * OUT * 3];
  let mut ref_rgb_f16 = std::vec![half::f16::ZERO; OUT * OUT * 3];
  let mut ref_rgba = std::vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = std::vec![0u16; OUT * OUT * 4];
  let mut ref_rgba_f32 = std::vec![0.0f32; OUT * OUT * 4];
  let mut ref_rgba_f16 = std::vec![half::f16::ZERO; OUT * OUT * 4];
  let mut ref_luma = std::vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = std::vec![0u16; OUT * OUT];
  let mut ref_h = std::vec![0u8; OUT * OUT];
  let mut ref_s = std::vec![0u8; OUT * OUT];
  let mut ref_v = std::vec![0u8; OUT * OUT];
  {
    let binned = frame(&pgw, &pbw, &prw, OUT, OUT);
    let mut sink = MixedSinker::<Gbrpf16>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgb_f32(&mut ref_rgb_f32)
      .unwrap()
      .with_rgb_f16(&mut ref_rgb_f16)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_rgba_f32(&mut ref_rgba_f32)
      .unwrap()
      .with_rgba_f16(&mut ref_rgba_f16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gbrpf16_to(&binned, &mut sink).unwrap();
  }

  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgb_f32, ref_rgb_f32, "rgb_f32 (f16-rounded, full parity)");
  assert_eq!(rgb_f16, ref_rgb_f16, "rgb_f16 (full parity)");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(
    rgba_f32, ref_rgba_f32,
    "rgba_f32 (f16-rounded, full parity)"
  );
  assert_eq!(rgba_f16, ref_rgba_f16, "rgba_f16 (full parity)");
  assert_eq!(luma, ref_luma, "luma");
  // luma_u16 on the fused path is the direct path's narrowed (8-bit, in a
  // u16 carrier) value — `gbrpf32_to_luma_u16_row` stages through u8 — so
  // it is byte-identical to the direct Gbrpf16 `with_luma_u16`.
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
fn gbrpf16_identity_plan_matches_new_sink() {
  // An identity (SRC->SRC) plan still routes through the f32 bin + f16
  // round emit; an identity bin is a copy, so each output must equal the
  // direct `new` sink (which widens the f16 source to f32). rgb_f32 is the
  // f16 source widened to f32 — i.e. the identity-binned-then-rounded
  // value equals the source f16 widened, since rounding an exact f16 value
  // is itself.
  let (g, b, r) = gbr_planes_f16();
  let (gw, bw, rw) = (as_le_f16(&g), as_le_f16(&b), as_le_f16(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut direct = std::vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Gbrpf16>::new(SRC, SRC)
      .with_rgb_f32(&mut direct)
      .unwrap();
    gbrpf16_to(&src, &mut sink).unwrap();
  }
  let mut via_area = std::vec![0.0f32; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Gbrpf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_f32(&mut via_area)
        .unwrap();
    gbrpf16_to(&src, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity-plan resample == direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_no_output_sink_is_a_noop() {
  // A resampled sink with no attached output neither allocates the
  // stream nor enforces sequencing — the documented legal no-op.
  let (g, b, r) = gbr_planes_f16();
  let (gw, bw, rw) = (as_le_f16(&g), as_le_f16(&b), as_le_f16(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut sink =
    MixedSinker::<Gbrpf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  gbrpf16_to(&src, &mut sink).unwrap();
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
fn gbrpf16_plane_scratch_sizing_is_f16_only() {
  // The f16 tail stages only the rounded f16 G/B/R planes; it never uses
  // the f32 plane scratch (that belongs to the Gbrpf32 tail). So a
  // no-output sink sizes neither, and an output-bearing sink sizes ONLY
  // the f16 plane scratch — the f32 one stays empty.
  let (g, b, r) = gbr_planes_f16();
  let (gw, bw, rw) = (as_le_f16(&g), as_le_f16(&b), as_le_f16(&r));
  let src = frame(&gw, &bw, &rw, SRC, SRC);

  let mut sink =
    MixedSinker::<Gbrpf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  gbrpf16_to(&src, &mut sink).unwrap();
  assert_eq!(
    sink.rgb_plane_scratch_f16_capacity(),
    0,
    "no-output sink grew the f16 G/B/R plane scratch"
  );

  // Attaching rgb_f16 (an f16-plane-derived output) sizes the f16 plane
  // scratch to the out-width G/B/R planes, while the f32 plane scratch —
  // unused by this tail — stays empty.
  let mut rgb_f16 = std::vec![half::f16::ZERO; OUT * OUT * 3];
  let mut sink2 =
    MixedSinker::<Gbrpf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_f16(&mut rgb_f16)
      .unwrap();
  gbrpf16_to(&src, &mut sink2).unwrap();
  assert!(
    sink2.rgb_plane_scratch_f16_capacity() >= OUT * 3,
    "rgb_f16 output did not size the f16 plane scratch"
  );
  assert_eq!(
    sink2.rgb_plane_scratch_capacity(),
    0,
    "f16 tail unexpectedly grew the f32 G/B/R plane scratch"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gbrpf16_resample_reuses_stream_across_frames() {
  // begin_frame resets the f32 area stream + frozen output set, so frame
  // 2's row 0 is accepted (not rejected as out-of-sequence) and the
  // output reflects frame 2's input. Both frames share one output buffer;
  // only the input data changes.
  let (g1, b1, r1) = gbr_planes_f16();
  // Frame 2: negate every plane so its block mean differs from frame 1.
  let g2: Vec<half::f16> = g1
    .iter()
    .map(|&v| half::f16::from_f32(-v.to_f32()))
    .collect();
  let b2: Vec<half::f16> = b1
    .iter()
    .map(|&v| half::f16::from_f32(-v.to_f32()))
    .collect();
  let r2: Vec<half::f16> = r1
    .iter()
    .map(|&v| half::f16::from_f32(-v.to_f32()))
    .collect();
  let (g1w, b1w, r1w) = (as_le_f16(&g1), as_le_f16(&b1), as_le_f16(&r1));
  let (g2w, b2w, r2w) = (as_le_f16(&g2), as_le_f16(&b2), as_le_f16(&r2));

  let mut out = std::vec![0.0f32; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Gbrpf16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_f32(&mut out)
        .unwrap();
    gbrpf16_to(&frame(&g1w, &b1w, &r1w, SRC, SRC), &mut sink).unwrap();
    gbrpf16_to(&frame(&g2w, &b2w, &r2w, SRC, SRC), &mut sink).unwrap();
  }

  let (pg2, pb2, pr2) = (
    prebinned_plane_f16(&g2),
    prebinned_plane_f16(&b2),
    prebinned_plane_f16(&r2),
  );
  for oy in 0..OUT {
    for ox in 0..OUT {
      let i = oy * OUT + ox;
      let base = i * 3;
      assert_eq!(out[base], pr2[i].to_f32(), "frame2 R ({ox},{oy})");
      assert_eq!(out[base + 1], pg2[i].to_f32(), "frame2 G ({ox},{oy})");
      assert_eq!(out[base + 2], pb2[i].to_f32(), "frame2 B ({ox},{oy})");
    }
  }
}
