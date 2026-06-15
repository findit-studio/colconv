//! Fused-downscale coverage for the 8-bit semi-planar YUV family
//! (NV12 / NV21 / NV16 / NV24 / NV42) — routed through the shared
//! row-stage planar resample. Each member bins the Y plane directly for
//! luma (the YUV luma contract) and bins a source-width RGB row
//! converted with the format's own fused `nv*_to_rgb_row` kernel for
//! colour. So RGB equals an `Rgb24` resample of the identity-converted
//! frame, luma equals the area-downscaled Y plane (not RGB-derived),
//! and — where the planar twin exists — every output is byte-identical
//! to the [`Yuv420p`] / [`Yuv422p`] / [`Yuv444p`] resample of the
//! de-interleaved planes.

use crate::{
  ColorMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{
    Nv12, Nv12Row, Nv16, Nv21, Nv24, Nv42, Rgb24, nv12_to, nv16_to, nv21_to, nv24_to, nv42_to,
    rgb24_to,
  },
};
use mediaframe::frame::{Nv12Frame, Nv16Frame, Nv21Frame, Nv24Frame, Nv42Frame, Rgb24Frame};

const SRC: usize = 8;
const OUT: usize = 4;

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid plane to
/// the `OUT` grid — the integer-ratio (2:1) area-downscale reference.
fn block_mean_2x2(plane: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u8;
    }
  }
  out
}

/// `Rgb24` resample of a full-res converted RGB frame — the colour
/// reference every member must match (convert-then-bin semantics).
fn rgb24_rgb_reference(converted: &[u8]) -> Vec<u8> {
  let src = Rgb24Frame::new(converted, SRC as u32, SRC as u32, (SRC * 3) as u32);
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    rgb24_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  rgb
}

// ---- Plane / interleaved-chroma fixtures --------------------------------

/// A deterministic Y ramp over the `SRC` grid.
fn y_ramp() -> Vec<u8> {
  (0..SRC * SRC)
    .map(|i| 40 + (i as u8).wrapping_mul(2))
    .collect()
}

/// Separate U / V planes at chroma width `cw`, chroma height `ch`.
fn uv_planes(cw: usize, ch: usize) -> (Vec<u8>, Vec<u8>) {
  let mut u = vec![0u8; cw * ch];
  let mut v = vec![0u8; cw * ch];
  for (i, p) in u.iter_mut().enumerate() {
    *p = 70 + ((i % cw) as u8).wrapping_mul(5);
  }
  for (i, p) in v.iter_mut().enumerate() {
    *p = 200u8.wrapping_sub(((i % cw) as u8).wrapping_mul(4));
  }
  (u, v)
}

/// Interleave separate U / V planes into a semi-planar chroma plane.
/// `swap_uv = false` writes `U0 V0 U1 V1 …` (NV12 / NV16 / NV24);
/// `swap_uv = true` writes `V0 U0 …` (NV21 / NV42).
fn interleave(u: &[u8], v: &[u8], swap_uv: bool) -> Vec<u8> {
  let mut out = vec![0u8; u.len() * 2];
  for (i, (&uu, &vv)) in u.iter().zip(v).enumerate() {
    let (a, b) = if swap_uv { (vv, uu) } else { (uu, vv) };
    out[i * 2] = a;
    out[i * 2 + 1] = b;
  }
  out
}

// =========================================================================
// NV12 (4:2:0, UV)
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv12>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv12_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    nv12_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv12 rgb: convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_luma_is_area_downscaled_y_plane() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    nv12_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  let y_ref = block_mean_2x2(&y);
  assert_eq!(luma, y_ref, "nv12 luma must be the area-downscaled Y plane");
  let y_ref_u16: Vec<u16> = y_ref.iter().map(|&b| b as u16).collect();
  assert_eq!(luma_u16, y_ref_u16, "nv12 luma_u16 = Y, zero-extended");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_luma_from_y_not_rgb_under_saturated_chroma() {
  let y = vec![16u8; SRC * SRC];
  let cw = SRC / 2;
  let ch = SRC / 2;
  let u = vec![240u8; cw * ch];
  let v = vec![16u8; cw * ch];
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    nv12_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert!(
    luma.iter().all(|&b| b == 16),
    "nv12 luma must be Y (16); got {luma:?}"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_identity_plan_matches_new_sink() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv12>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    nv12_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    nv12_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "nv12 identity plan == new sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_no_output_is_noop() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
  let mut sink =
    MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  // No attached outputs: every row is a legal no-op (no alloc, no
  // sequencing), even out of order.
  nv12_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_reuses_streams_across_frames() {
  let y1 = y_ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    let f1 = Nv12Frame::new(&y1, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let f2 = Nv12Frame::new(&y2, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    nv12_to(&f1, true, ColorMatrix::Bt601, &mut sink).unwrap();
    nv12_to(&f2, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(
    luma,
    block_mean_2x2(&y2),
    "frame 2 luma must area-downscale frame 2's Y"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_out_of_order_row_rejected() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  // Feed row 3 first: the luma stream expects row 0.
  let err = sink
    .process(Nv12Row::new(
      &y[3 * SRC..4 * SRC],
      &uv[SRC..2 * SRC],
      3,
      ColorMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(matches!(
    err,
    MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
  ));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_mid_frame_output_change_rejected() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma2 = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  sink
    .process(Nv12Row::new(
      &y[0..SRC],
      &uv[0..SRC],
      0,
      ColorMatrix::Bt601,
      true,
    ))
    .unwrap();
  // Swap the luma buffer mid-frame: the frozen-output check must reject.
  sink.set_luma(&mut luma2).unwrap();
  let err = sink
    .process(Nv12Row::new(
      &y[SRC..2 * SRC],
      &uv[0..SRC],
      1,
      ColorMatrix::Bt601,
      true,
    ))
    .unwrap_err();
  assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
}

// =========================================================================
// NV21 (4:2:0, VU) — must equal NV12 on the same logical chroma
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv21_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let vu = interleave(&u, &v, true);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv21Frame::new(&y, &vu, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv21>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv21_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv21Frame::new(&y, &vu, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv21, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    nv21_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv21 rgb: convert-then-bin"
  );
}

// =========================================================================
// NV16 (4:2:2, UV)
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC);
  let uv = interleave(&u, &v, false);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv16Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Nv16>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv16_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv16Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    nv16_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv16 rgb: convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv16_resample_luma_is_area_downscaled_y_plane() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  {
    let frame = Nv16Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Nv16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    nv16_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(luma, block_mean_2x2(&y), "nv16 luma = area-downscaled Y");
}

// =========================================================================
// NV24 (4:4:4, UV) and NV42 (4:4:4, VU)
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC, SRC);
  let uv = interleave(&u, &v, false);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv24Frame::new(
      &y,
      &uv,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink = MixedSinker::<Nv24>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv24_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv24Frame::new(
      &y,
      &uv,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink =
      MixedSinker::<Nv24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    nv24_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv24 rgb: convert-then-bin"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv24_resample_luma_is_area_downscaled_y_plane() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC, SRC);
  let uv = interleave(&u, &v, false);
  let mut luma = vec![0u8; OUT * OUT];
  {
    let frame = Nv24Frame::new(
      &y,
      &uv,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink =
      MixedSinker::<Nv24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    nv24_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(luma, block_mean_2x2(&y), "nv24 luma = area-downscaled Y");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv42_resample_rgb_matches_rgb24_of_converted_frame() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC, SRC);
  let vu = interleave(&u, &v, true);
  let mut full_rgb = vec![0u8; SRC * SRC * 3];
  {
    let frame = Nv42Frame::new(
      &y,
      &vu,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink = MixedSinker::<Nv42>::new(SRC, SRC)
      .with_rgb(&mut full_rgb)
      .unwrap();
    nv42_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let frame = Nv42Frame::new(
      &y,
      &vu,
      SRC as u32,
      SRC as u32,
      SRC as u32,
      (SRC * 2) as u32,
    );
    let mut sink =
      MixedSinker::<Nv42, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    nv42_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
  }
  assert_eq!(
    rgb,
    rgb24_rgb_reference(&full_rgb),
    "nv42 rgb: convert-then-bin"
  );
}

// =========================================================================
// All-outputs combo (rgb + rgba + luma + luma_u16 + hsv) in one frame —
// each output equals its standalone single-output resample.
// =========================================================================

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn nv12_resample_all_outputs_match_standalone() {
  let y = y_ramp();
  let (u, v) = uv_planes(SRC / 2, SRC / 2);
  let uv = interleave(&u, &v, false);
  let mk = || Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);

  // Combined sink: every output attached.
  let (mut rgb, mut rgba) = (vec![0u8; OUT * OUT * 3], vec![0u8; OUT * OUT * 4]);
  let (mut luma, mut luma_u16) = (vec![0u8; OUT * OUT], vec![0u16; OUT * OUT]);
  let (mut hh, mut ss, mut vv) = (
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
  );
  {
    let mut sink =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut hh, &mut ss, &mut vv)
        .unwrap();
    nv12_to(&mk(), true, ColorMatrix::Bt601, &mut sink).unwrap();
  }

  // Standalone references.
  let mut rgb_ref = vec![0u8; OUT * OUT * 3];
  {
    let mut s =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb_ref)
        .unwrap();
    nv12_to(&mk(), true, ColorMatrix::Bt601, &mut s).unwrap();
  }
  let mut rgba_ref = vec![0u8; OUT * OUT * 4];
  {
    let mut s =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba_ref)
        .unwrap();
    nv12_to(&mk(), true, ColorMatrix::Bt601, &mut s).unwrap();
  }
  let (mut h_ref, mut s_ref, mut v_ref) = (
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
    vec![0u8; OUT * OUT],
  );
  {
    let mut s =
      MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
        .unwrap();
    nv12_to(&mk(), true, ColorMatrix::Bt601, &mut s).unwrap();
  }

  assert_eq!(rgb, rgb_ref, "combined rgb == standalone");
  assert_eq!(rgba, rgba_ref, "combined rgba == standalone");
  assert_eq!(
    luma,
    block_mean_2x2(&y),
    "combined luma == area-downscaled Y"
  );
  let y_ref_u16: Vec<u16> = block_mean_2x2(&y).iter().map(|&b| b as u16).collect();
  assert_eq!(luma_u16, y_ref_u16, "combined luma_u16 == zero-extended Y");
  assert_eq!(
    (hh, ss, vv),
    (h_ref, s_ref, v_ref),
    "combined hsv == standalone"
  );
}

// =========================================================================
// Byte-identical parity with the planar twin on the de-interleaved planes
// (the strongest check — requires the yuv-planar family compiled in).
// =========================================================================

#[cfg(feature = "yuv-planar")]
mod twin_parity {
  use super::*;
  use crate::source::{Yuv420p, Yuv444p, yuv420p_to, yuv444p_to};
  use mediaframe::frame::{Yuv420pFrame, Yuv444pFrame};

  /// NV12 resample must be byte-identical to a Yuv420p resample of the
  /// de-interleaved planes, for every output.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv12_equals_yuv420p_on_deinterleaved_planes() {
    let y = y_ramp();
    let (u, v) = uv_planes(SRC / 2, SRC / 2);
    let uv = interleave(&u, &v, false);
    let cw = (SRC / 2) as u32;

    let mut nv_rgb = vec![0u8; OUT * OUT * 3];
    let mut nv_luma = vec![0u8; OUT * OUT];
    {
      let frame = Nv12Frame::new(&y, &uv, SRC as u32, SRC as u32, SRC as u32, SRC as u32);
      let mut sink =
        MixedSinker::<Nv12, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_rgb(&mut nv_rgb)
          .unwrap()
          .with_luma(&mut nv_luma)
          .unwrap();
      nv12_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
    }
    let mut p_rgb = vec![0u8; OUT * OUT * 3];
    let mut p_luma = vec![0u8; OUT * OUT];
    {
      let frame = Yuv420pFrame::new(&y, &u, &v, SRC as u32, SRC as u32, SRC as u32, cw, cw);
      let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          // Force the row-stage tier — the semi-planar path is row-stage.
          .with_native(false)
          .with_rgb(&mut p_rgb)
          .unwrap()
          .with_luma(&mut p_luma)
          .unwrap();
      yuv420p_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
    }
    assert_eq!(nv_rgb, p_rgb, "nv12 rgb == yuv420p(row-stage) rgb");
    assert_eq!(nv_luma, p_luma, "nv12 luma == yuv420p luma");
  }

  /// NV24 resample must be byte-identical to a Yuv444p resample of the
  /// de-interleaved planes.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn nv24_equals_yuv444p_on_deinterleaved_planes() {
    let y = y_ramp();
    let (u, v) = uv_planes(SRC, SRC);
    let uv = interleave(&u, &v, false);

    let mut nv_rgb = vec![0u8; OUT * OUT * 3];
    {
      let frame = Nv24Frame::new(
        &y,
        &uv,
        SRC as u32,
        SRC as u32,
        SRC as u32,
        (SRC * 2) as u32,
      );
      let mut sink =
        MixedSinker::<Nv24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
          .unwrap()
          .with_rgb(&mut nv_rgb)
          .unwrap();
      nv24_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
    }
    let mut p_rgb = vec![0u8; OUT * OUT * 3];
    {
      let frame = Yuv444pFrame::new(
        &y, &u, &v, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
      );
      let mut sink = MixedSinker::<Yuv444p, AreaResampler>::with_resampler(
        SRC,
        SRC,
        AreaResampler::to(OUT, OUT),
      )
      .unwrap()
      .with_rgb(&mut p_rgb)
      .unwrap();
      yuv444p_to(&frame, true, ColorMatrix::Bt601, &mut sink).unwrap();
    }
    assert_eq!(nv_rgb, p_rgb, "nv24 rgb == yuv444p rgb");
  }
}
