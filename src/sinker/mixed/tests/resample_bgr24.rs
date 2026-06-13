//! Fused-downscale coverage for `Bgr24` — the second format routed
//! through the shared packed-RGB resample tail. The source row is
//! swapped to RGB at source width in the shared scratch, then fed to
//! the same area stream and emit derivations as `Rgb24`, so these
//! tests pin both the swap and the shared tail.

use crate::{
  ColorMatrix, PixelSink,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Bgr24, Bgr24Row, Rgb24, bgr24_to, rgb24_to},
};
use mediaframe::frame::{Bgr24Frame, Rgb24Frame};

const SRC: usize = 8;
const OUT: usize = 4;

/// Per-channel ramps in `B, G, R` order; every value interior so the
/// derived kernels see real math.
fn packed_bgr_frame() -> Vec<u8> {
  let mut buf = vec![0u8; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px[0] = 60 + ((i % 8) as u8) * 10; // B
    px[1] = 200 - (i as u8) * 2; // G
    px[2] = 40 + (i as u8) * 2; // R
  }
  buf
}

/// The same pixels with channels swapped to `R, G, B` — the canonical
/// form the shared tail bins.
fn swapped_to_rgb(bgr: &[u8]) -> Vec<u8> {
  let mut rgb = bgr.to_vec();
  for px in rgb.chunks_exact_mut(3) {
    px.swap(0, 2);
  }
  rgb
}

/// Direct 2x2 block mean (round-half-up) of the swapped (RGB) source —
/// the exact contract for integer-ratio area downscale.
fn expected_block_mean(rgb: &[u8], ox: usize, oy: usize, c: usize) -> u8 {
  let mut acc = 0u32;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u32;
    }
  }
  ((acc + 2) / 4) as u8
}

#[test]
fn bgr24_downscale_is_exact_area_mean_of_swapped_source() {
  let buf = packed_bgr_frame();
  let rgb_ref = swapped_to_rgb(&buf);
  let src = Bgr24Frame::try_new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut rgb = vec![0u8; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap();
    bgr24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          rgb[(oy * OUT + ox) * 3 + c],
          expected_block_mean(&rgb_ref, ox, oy, c),
          "({ox},{oy}) c{c}"
        );
      }
    }
  }
}

#[test]
fn bgr24_resample_matches_rgb24_of_swapped_frame() {
  // The shared tail is format-agnostic past the swap: a Bgr24 frame
  // and the channel-swapped Rgb24 frame must produce identical RGB,
  // RGBA, luma and HSV outputs.
  let buf = packed_bgr_frame();
  let rgb_buf = swapped_to_rgb(&buf);
  let bgr_src = Bgr24Frame::try_new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();
  let rgb_src = Rgb24Frame::new(&rgb_buf, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let (mut rgb_a, mut rgba_a, mut luma_a) = (
    vec![0u8; OUT * OUT * 3],
    vec![0u8; OUT * OUT * 4],
    vec![0u8; OUT * OUT],
  );
  {
    let mut sink =
      MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb_a)
        .unwrap()
        .with_rgba(&mut rgba_a)
        .unwrap()
        .with_luma(&mut luma_a)
        .unwrap();
    bgr24_to(&bgr_src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  let (mut rgb_b, mut rgba_b, mut luma_b) = (
    vec![0u8; OUT * OUT * 3],
    vec![0u8; OUT * OUT * 4],
    vec![0u8; OUT * OUT],
  );
  {
    let mut sink =
      MixedSinker::<Rgb24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb_b)
        .unwrap()
        .with_rgba(&mut rgba_b)
        .unwrap()
        .with_luma(&mut luma_b)
        .unwrap();
    rgb24_to(&rgb_src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(rgb_a, rgb_b, "rgb");
  assert_eq!(rgba_a, rgba_b, "rgba");
  assert_eq!(luma_a, luma_b, "luma");
}

#[test]
fn bgr24_identity_plan_matches_new_sink() {
  let buf = packed_bgr_frame();
  let src = Bgr24Frame::try_new(&buf, SRC as u32, SRC as u32, (SRC * 3) as u32).unwrap();

  let mut direct = vec![0u8; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Bgr24>::new(SRC, SRC)
      .with_rgb(&mut direct)
      .unwrap();
    bgr24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  let mut via_area = vec![0u8; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb(&mut via_area)
        .unwrap();
    bgr24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

#[test]
fn bgr24_resample_no_outputs_is_a_no_op() {
  // Attaching no outputs is legal and must write nothing. The
  // resample route freezes the (empty) output set, then returns
  // before allocating a stream or enforcing sequencing — so even an
  // out-of-order row is accepted, which a stream-backed path would
  // reject. (Without the output-presence preflight this allocates and
  // sequence-checks for no observable output.)
  let buf = packed_bgr_frame();
  let mut sink =
    MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let row2 = &buf[SRC * 3 * 2..SRC * 3 * 3];
  sink
    .process(Bgr24Row::new(row2, 2, ColorMatrix::Bt709, true))
    .unwrap();
}

#[test]
fn bgr24_resample_rejects_out_of_sequence_before_staging() {
  // An out-of-sequence row must be rejected before the source-width
  // scratch is grown or the BGR->RGB swap runs — sequencing is
  // checked ahead of staging (matching Rgb24 / YUV). The scratch
  // capacity staying zero after the rejection proves no staging
  // allocation happened; without the ordering it would grow to
  // source-width * 3.
  let buf = packed_bgr_frame();
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let row2 = &buf[SRC * 3 * 2..SRC * 3 * 3];
  let err = sink
    .process(Bgr24Row::new(row2, 2, ColorMatrix::Bt709, true))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "got {err:?}"
  );
  assert_eq!(
    sink.rgb_scratch_capacity(),
    0,
    "out-of-sequence row must not stage into the scratch"
  );
}

#[test]
fn bgr24_resample_rejects_out_of_sequence_rows() {
  let buf = packed_bgr_frame();
  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut sink =
    MixedSinker::<Bgr24, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  // Skip row 0 — the stream expects strict sequencing from row 0.
  let row1 = &buf[SRC * 3..SRC * 3 * 2];
  let err = sink
    .process(Bgr24Row::new(row1, 1, ColorMatrix::Bt709, true))
    .unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "got {err:?}"
  );
}
