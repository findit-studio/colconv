//! Fused-downscale coverage for the packed RGB family: the source row
//! is already interleaved RGB, so binning IS the whole fused job and
//! output values are exact area means.

use crate::{
  ColorMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Rgb24, Rgb24Row, rgb24_to},
};
use mediaframe::frame::Rgb24Frame;

const SRC: usize = 8;
const OUT: usize = 4;

/// Per-channel ramps; every value interior so derived kernels see
/// real math.
fn packed_frame() -> Vec<u8> {
  let mut buf = vec![0u8; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px[0] = 40 + (i as u8) * 2;
    px[1] = 200 - (i as u8) * 2;
    px[2] = 60 + ((i % 8) as u8) * 10;
  }
  buf
}

/// Direct 2x2 block mean with round-half-up — the exact contract for
/// integer-ratio area downscale of packed RGB.
fn expected_block_mean(src: &[u8], ox: usize, oy: usize, c: usize) -> u8 {
  let mut acc = 0u32;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
    }
  }
  ((acc + 2) / 4) as u8
}

#[test]
fn rgb24_downscale_is_exact_area_mean() {
  let buf = packed_frame();
  let src = Rgb24Frame::new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          rgb[(oy * OUT + ox) * 3 + c],
          expected_block_mean(&buf, ox, oy, c),
          "({ox},{oy}) c{c}"
        );
      }
    }
  }
}

#[test]
fn rgb24_derived_outputs_come_from_binned_rgb() {
  // Luma and HSV must be the kernels applied to the (exact) binned
  // RGB — i.e. derived from the downscaled image.
  let buf = packed_frame();
  let src = Rgb24Frame::new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut luma = vec![0u8; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  // Reference: run the FULL-RES sink over a frame that already
  // contains the binned RGB.
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  {
    let binned = Rgb24Frame::new(&rgb, OUT as u32, OUT as u32, (OUT * 3) as u32);
    let mut sink = MixedSinker::<Rgb24>::new(OUT, OUT)
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    rgb24_to(&binned, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(luma, ref_luma);
  assert_eq!(h, ref_h);
  assert_eq!(s_, ref_s);
  assert_eq!(v_, ref_v);
  assert_eq!(rgba, ref_rgba);
}

#[test]
fn rgb24_identity_plan_matches_new_sink() {
  let buf = packed_frame();
  let src = Rgb24Frame::new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Rgb24>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
fn rgb24_contracts_hold_on_the_fused_path() {
  let buf = packed_frame();
  let row0 = &buf[..SRC * 3];

  // Out-of-order direct process.
  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    let err = sink
      .process(Rgb24Row::new(row0, 3, ColorMatrix::Bt709, true))
      .unwrap_err();
    assert!(matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ));
  }

  // Mid-frame output change.
  let mut luma = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    sink.begin_frame(SRC as u32, SRC as u32).unwrap();
    sink
      .process(Rgb24Row::new(row0, 0, ColorMatrix::Bt709, true))
      .unwrap();
    sink.set_luma(&mut luma).unwrap();
    let err = sink
      .process(Rgb24Row::new(
        &buf[SRC * 3..SRC * 6],
        1,
        ColorMatrix::Bt709,
        true,
      ))
      .unwrap_err();
    assert!(matches!(err, MixedSinkerError::ResampleOutputsChanged(_)));
  }
  assert!(luma.iter().all(|&l| l == 0));
}
