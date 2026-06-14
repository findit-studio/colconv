//! Fused-downscale coverage for `Gray16` — routed through a single
//! 1-channel `AreaStream<u16>` that bins the source `u16` luma plane at
//! u16 precision (Gray16 *is* a u16 luma plane). The wire row converts
//! to a host-native u16 luma plane first (the same kernel the direct
//! `luma_u16` path uses), then every attached output derives from each
//! finalized binned u16 luma row exactly as the direct path does:
//! `luma_u16` is a host-native pass-through, `luma` is `>> 8`, `rgb` /
//! `rgba` broadcast the `>> 8` byte (α = 0xFF), `rgb_u16` / `rgba_u16`
//! broadcast the native u16 (α = 0xFFFF), and `hsv` is `H=0 / S=0 /
//! V=Y>>8`. So every resampled output equals the direct Gray16 sink run
//! over a frame that already holds the binned u16 luma plane.
//!
//! The binning is at full u16 precision: a u8 luma stream would bin the
//! high byte only, losing the low 8 bits of every sample.

use crate::{
  ColorMatrix, PixelSink,
  frame::Gray16Frame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Gray16, Gray16Row, gray16_to, gray16_to_endian},
};

const SRC: usize = 8;
const OUT: usize = 4;
// Gray is luma-only; the walker still threads a matrix / range through.
const FR: bool = true;
const M: ColorMatrix = ColorMatrix::Bt709;

/// Re-encode a host-native u16 slice as LE-encoded byte storage (the
/// `gray16le` plane contract). `gray16_to` recovers the logical values
/// via `u16::from_le` — a no-op on LE hosts, a byte-swap on BE.
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as BE-encoded byte storage (the
/// `gray16be` plane contract), recovered via `u16::from_be`.
fn as_be_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Interior 16-bit luma ramp so the area mean sees real variation per
/// 2x2 block, exercising the low byte the u16 stream must preserve.
fn ramp() -> Vec<u16> {
  let mut y = vec![0u16; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    // Spread across the full 16-bit range with a non-trivial low byte.
    *p = ((i * 1031) % 65536) as u16;
  }
  y
}

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid `u16`
/// plane to the `OUT` grid — the integer-ratio (2:1) area-downscale
/// reference, computed at u16 precision (sum of four u16 fits in u32).
fn block_mean_2x2(plane: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u16;
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_downscale_luma_u16_is_exact_area_mean() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma_u16,
    block_mean_2x2(&plane),
    "luma_u16 must be the exact 2x2 block mean of the u16 luma plane"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_all_outputs_match_direct_over_binned_luma() {
  // Every attached output — luma / luma_u16 / rgb / rgba / rgb_u16 /
  // rgba_u16 / hsv — must be exactly what the direct Gray16 sink
  // produces over the (exact) binned u16 luma plane. The binned luma is
  // the area mean, so we feed that mean as a full-resolution `OUT`-grid
  // Gray16 frame to the reference sink.
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }

  // Reference: the direct sink over the exact binned u16 luma plane.
  let binned = block_mean_2x2(&plane);
  let binned_pix = as_le_u16(&binned);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned_frame = Gray16Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray16>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray16_to(&binned_frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_le_be_resample_outputs_identical() {
  // The binned luma is host-native, so the derive kernels must run with
  // `HOST_NATIVE_BE`, not `<false>`. On an LE dev/CI host `<false>`
  // masks the bug; the LE-vs-BE parity check catches a wrong const on
  // either host (LE and BE wire encodings of the same logical plane must
  // resample to identical outputs).
  let plane = ramp();
  let pix_le = as_le_u16(&plane);
  let pix_be = as_be_u16(&plane);

  let mut le_luma_u16 = vec![0u16; OUT * OUT];
  let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut le_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Gray16Frame::new(&pix_le, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut le_luma_u16)
        .unwrap()
        .with_rgb_u16(&mut le_rgb_u16)
        .unwrap()
        .with_rgba(&mut le_rgba)
        .unwrap();
    gray16_to(&frame, FR, M, &mut sink).unwrap();
  }

  let mut be_luma_u16 = vec![0u16; OUT * OUT];
  let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut be_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Gray16Frame::<true>::new(&pix_be, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Gray16<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_luma_u16(&mut be_luma_u16)
    .unwrap()
    .with_rgb_u16(&mut be_rgb_u16)
    .unwrap()
    .with_rgba(&mut be_rgba)
    .unwrap();
    gray16_to_endian::<_, true>(&frame, FR, M, &mut sink).unwrap();
  }

  assert_eq!(le_luma_u16, be_luma_u16, "luma_u16 LE/BE diverge");
  assert_eq!(le_rgb_u16, be_rgb_u16, "rgb_u16 LE/BE diverge");
  assert_eq!(le_rgba, be_rgba, "rgba LE/BE diverge");
  // And the binned u16 luma is the exact area mean regardless of wire.
  assert_eq!(
    le_luma_u16,
    block_mean_2x2(&plane),
    "luma_u16 not area mean"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_standalone_rgba_matches_direct_over_binned_luma() {
  // u8 RGBA-only exercises the dedicated fast path (no RGB scratch).
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_pix = as_le_u16(&block_mean_2x2(&plane));
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  {
    let binned = Gray16Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray16>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap();
    gray16_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba, ref_rgba, "standalone rgba");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_standalone_rgba_u16_matches_direct_over_binned_luma() {
  // u16 RGBA-only exercises the native rgba_u16 fast path.
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  let binned_pix = as_le_u16(&block_mean_2x2(&plane));
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  {
    let binned = Gray16Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray16>::new(OUT, OUT)
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap();
    gray16_to(&binned, FR, M, &mut sink).unwrap();
  }
  assert_eq!(rgba_u16, ref_rgba_u16, "standalone rgba_u16");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_hsv_plus_rgba_matches_direct_over_binned_luma() {
  // HSV + u8 RGBA without RGB: both derive directly from the binned luma
  // (no RGB kernel, no RGB scratch — asserted below), byte-identical to
  // the direct path's output.
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
    // Regression: this case must not reserve RGB scratch — it derives
    // HSV+RGBA from luma and never reads the scratch, so reserving it
    // could spuriously AllocationFail under memory pressure.
    assert_eq!(
      sink.rgb_scratch_capacity(),
      0,
      "gray16 HSV+RGBA must not reserve RGB scratch"
    );
  }
  let binned_pix = as_le_u16(&block_mean_2x2(&plane));
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Gray16Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray16>::new(OUT, OUT)
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray16_to(&binned, FR, M, &mut sink).unwrap();
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
fn gray16_identity_plan_matches_new_sink() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut direct = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Gray16>::new(SRC, SRC)
      .with_rgb_u16(&mut direct)
      .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_u16(&mut via_area)
        .unwrap();
    gray16_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area, "identity plan must match the direct sink");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray16_resample_reuses_luma_stream_across_frames() {
  // A reused sink must reset the u16 luma stream each frame; without the
  // reset, frame 2's row 0 is rejected as out-of-sequence.
  let y1 = ramp();
  let mut y2 = y1.clone();
  for p in y2.iter_mut() {
    *p = 0xFFFF - *p;
  }
  let pix1 = as_le_u16(&y1);
  let pix2 = as_le_u16(&y2);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    gray16_to(
      &Gray16Frame::new(&pix1, SRC as u32, SRC as u32, SRC as u32),
      FR,
      M,
      &mut sink,
    )
    .unwrap();
    gray16_to(
      &Gray16Frame::new(&pix2, SRC as u32, SRC as u32, SRC as u32),
      FR,
      M,
      &mut sink,
    )
    .unwrap();
  }
  assert_eq!(
    luma_u16,
    block_mean_2x2(&y2),
    "frame 2 luma_u16 must area-downscale frame 2's luma"
  );
}

#[test]
fn gray16_resample_no_outputs_is_a_no_op() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let src = Gray16Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);
  let mut sink =
    MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  // No outputs attached: a legal no-op, accepted without error.
  gray16_to(&src, FR, M, &mut sink).unwrap();
  // A no-output call has no stream to sequence and never allocates.
  assert!(
    !sink.luma_stream_u16_allocated(),
    "no-output sink allocated a u16 luma stream"
  );
}

#[test]
fn gray16_out_of_sequence_first_row_rejected_before_allocation() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let row3 = &pix[3 * SRC..4 * SRC];

  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Feed row 3 first — the stream expects strict sequencing from 0.
  let err = sink.process(Gray16Row::new(row3, 3, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
  // The out-of-sequence first row must be rejected before the u16 luma
  // stream is allocated.
  assert!(
    !sink.luma_stream_u16_allocated(),
    "stream allocated for a rejected row"
  );
  assert!(
    luma_u16.iter().all(|&b| b == 0),
    "rejected row mutated output"
  );
}

#[test]
fn gray16_resample_rejects_mid_frame_out_of_sequence() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.process(Gray16Row::new(&pix[..SRC], 0, M, FR)).unwrap();
  // Skip row 1 — feeding row 2 next is out of sequence.
  let err = sink
    .process(Gray16Row::new(&pix[2 * SRC..3 * SRC], 2, M, FR))
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
fn gray16_resample_rejects_mid_frame_output_change() {
  let plane = ramp();
  let pix = as_le_u16(&plane);
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray16, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  sink.process(Gray16Row::new(&pix[..SRC], 0, M, FR)).unwrap();
  // Attaching a new output mid-frame trips the frozen-output check.
  sink.set_luma_u16(&mut luma_u16).unwrap();
  let err = sink
    .process(Gray16Row::new(&pix[SRC..2 * SRC], 1, M, FR))
    .unwrap_err();
  assert!(
    matches!(err, MixedSinkerError::ResampleOutputsChanged(_)),
    "expected ResampleOutputsChanged, got {err:?}"
  );
  assert!(
    luma_u16.iter().all(|&b| b == 0),
    "rejected row mutated the new output"
  );
}
