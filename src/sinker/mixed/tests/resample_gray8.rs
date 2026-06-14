//! Fused-downscale coverage for `Gray8` — routed through a single
//! 1-channel `AreaStream<u8>` that bins the source Y plane (Gray *is* a
//! luma plane). Every attached output then derives from the binned Y
//! exactly as the direct path does: `luma` copies it, `luma_u16`
//! zero-extends it, `rgb` broadcasts Y to `[Y, Y, Y]`, `rgba` broadcasts
//! and pads alpha to `0xFF`, and `hsv` is `H=0 / S=0 / V=Y`. So every
//! resampled output equals the direct Gray8 sink run over a frame that
//! already holds the binned Y plane.

use crate::{
  ColorMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Gray8, Gray8Row, gray8_to},
};
use mediaframe::frame::Gray8Frame;

const SRC: usize = 8;
const OUT: usize = 4;
// Gray is luma-only; the walker still threads a matrix / range through.
const FR: bool = true;
const M: ColorMatrix = ColorMatrix::Bt709;

fn gray8_frame(plane: &[u8]) -> Gray8Frame<'_> {
  Gray8Frame::new(plane, SRC as u32, SRC as u32, SRC as u32)
}

/// Interior Y ramp so the area mean sees real variation per 2x2 block.
fn ramp() -> Vec<u8> {
  let mut y = vec![0u8; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = 16 + (i % 200) as u8;
  }
  y
}

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

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_downscale_luma_is_exact_area_mean() {
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma,
    block_mean_2x2(&plane),
    "luma must be the exact 2x2 block mean of the Y plane"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_all_outputs_match_direct_over_binned_y() {
  // Every attached output — luma / luma_u16 / rgb / rgba / hsv — must be
  // exactly what the direct Gray8 sink produces over the (exact) binned
  // Y plane. The binned Y is the area mean, so we feed that mean as a
  // full-resolution `OUT`-grid Gray8 frame to the reference sink.
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }

  // Reference: the direct sink over the exact binned Y plane.
  let binned_y = block_mean_2x2(&plane);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Gray8Frame::new(&binned_y, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray8>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray8_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_standalone_rgba_matches_direct_over_binned_y() {
  // RGBA-only exercises the dedicated fast path (no RGB scratch).
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_y = block_mean_2x2(&plane);
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  {
    let binned = Gray8Frame::new(&binned_y, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray8>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap();
    gray8_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, ref_rgba, "standalone rgba");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_hsv_plus_rgba_matches_direct_over_binned_y() {
  // HSV + RGBA without RGB forces the RGB-kernel-into-scratch branch
  // (HSV derived from the scratch RGB, RGBA fanned out from it).
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_y = block_mean_2x2(&plane);
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Gray8Frame::new(&binned_y, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray8>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray8_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, ref_rgba, "hsv+rgba: rgba");
  assert_eq!(h, ref_h, "hsv+rgba: h");
  assert_eq!(s_, ref_s, "hsv+rgba: s");
  assert_eq!(v_, ref_v, "hsv+rgba: v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_identity_plan_matches_new_sink() {
  let plane = ramp();
  let src = gray8_frame(&plane);

  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Gray8>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    gray8_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray8_resample_reuses_luma_stream_across_frames() {
  // A reused sink must reset the luma stream each frame; without the
  // reset, frame 2's row 0 is rejected as out-of-sequence.
  let y1 = ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 255 - *p;
  }
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma(&mut luma)
        .unwrap();
    gray8_to(&gray8_frame(&y1), FR, M, &mut sink).unwrap();
    gray8_to(&gray8_frame(&y2), FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma,
    block_mean_2x2(&y2),
    "frame 2 luma must area-downscale frame 2's Y"
  );
}

#[test]
fn gray8_resample_no_outputs_is_a_no_op() {
  let plane = ramp();
  let src = gray8_frame(&plane);
  let mut sink =
    MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  // No outputs attached: a legal no-op, accepted without error.
  gray8_to(&src, FR, M, &mut sink).unwrap();
  // A no-output call has no stream to sequence and never allocates.
  assert!(
    !sink.luma_stream_allocated(),
    "no-output sink allocated a stream"
  );
}

#[test]
fn gray8_out_of_sequence_first_row_rejected_before_allocation() {
  let plane = ramp();
  let row3 = &plane[3 * SRC..4 * SRC];

  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Feed row 3 first — the stream expects strict sequencing from 0.
  let err = sink.process(Gray8Row::new(row3, 3, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  // The out-of-sequence first row must be rejected before the luma
  // stream is allocated.
  assert!(
    !sink.luma_stream_allocated(),
    "stream allocated for a rejected row"
  );
  assert!(luma.iter().all(|&b| b == 0), "rejected row mutated output");
}

#[test]
fn gray8_resample_rejects_mid_frame_out_of_sequence() {
  let plane = ramp();
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma(&mut luma)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Gray8Row::new(&plane[..SRC], 0, M, FR))
    .unwrap();
  // Skip row 1 — feeding row 2 next is out of sequence.
  let err = sink
    .process(Gray8Row::new(&plane[2 * SRC..3 * SRC], 2, M, FR))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
}

#[test]
fn gray8_resample_rejects_mid_frame_output_change() {
  let plane = ramp();
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray8, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink
    .process(Gray8Row::new(&plane[..SRC], 0, M, FR))
    .unwrap();
  // Attaching a new output mid-frame trips the frozen-output check.
  sink.set_luma(&mut luma).unwrap();
  let err = sink
    .process(Gray8Row::new(&plane[SRC..2 * SRC], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0),
    "rejected row mutated the new output"
  );
}
