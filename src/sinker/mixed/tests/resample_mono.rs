//! Fused-downscale coverage for the 1-bit bilevel sources `Monoblack`
//! and `Monowhite` through the single-channel luma resample.
//!
//! Both formats are achromatic: each source bit expands to a 0/255 u8
//! luma (Monoblack: bit=1 → 255; Monowhite: bit=0 → 255), and every
//! output is a broadcast of that luma. The resample oracle is therefore
//! "expand the source bits to a 0/255 luma plane, area-bin that plane,
//! then derive each output from the binned luma" — exactly what the
//! sinker does, with the binned mean standing in for the per-pixel
//! expanded value of the direct path.

use crate::{
  ColorMatrix,
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{Monoblack, Monowhite, monoblack_to, monowhite_to},
};
use mediaframe::frame::{MonoblackFrame, MonowhiteFrame};

const SRC_W: usize = 8;
const SRC_H: usize = 4;
const OUT_W: usize = 4;
const OUT_H: usize = 2;
// One byte per row covers all 8 pixels (MSB first).
const STRIDE: usize = SRC_W / 8;

/// Bit pattern (one byte per row) chosen so the 2x2 block means span a
/// spread of values (0, 64, 128, 191, 255) rather than only the
/// saturated endpoints.
///
/// Rows (bit = Monoblack luma 255):
///   row0: 1111_0000
///   row1: 1100_0000
///   row2: 1010_1010
///   row3: 0000_1111
const PATTERN: [u8; SRC_H * STRIDE] = [0b1111_0000, 0b1100_0000, 0b1010_1010, 0b0000_1111];

/// Expand a 1-bit `PATTERN`-shaped buffer to a source-width 0/255 luma
/// plane, applying polarity (`invert` = Monowhite).
fn expand_luma(data: &[u8], invert: bool) -> Vec<u8> {
  let mut out = vec![0u8; SRC_W * SRC_H];
  for row in 0..SRC_H {
    let byte = data[row * STRIDE];
    for col in 0..SRC_W {
      let raw = (byte >> (7 - col)) & 1;
      let bit = if invert { 1 - raw } else { raw };
      out[row * SRC_W + col] = bit * 255;
    }
  }
  out
}

/// Exact 2x2-block area mean (round-half-up) of an `SRC`-grid plane to
/// the `OUT` grid — the integer-ratio (2:1) area-downscale reference.
fn block_mean_2x2(plane: &[u8]) -> Vec<u8> {
  let mut out = vec![0u8; OUT_W * OUT_H];
  for oy in 0..OUT_H {
    for ox in 0..OUT_W {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += plane[(oy * 2 + dy) * SRC_W + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT_W + ox] = ((s + 2) / 4) as u8;
    }
  }
  out
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_resample_luma_is_block_mean_of_expanded_bits() {
  let src = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();

  let mut luma = vec![0u8; OUT_W * OUT_H];
  {
    let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  // Expanded 0/255 luma plane (Monoblack: bit=1 → 255):
  //   row0: 255 255 255 255   0   0   0   0
  //   row1: 255 255   0   0   0   0   0   0
  //   row2: 255   0 255   0 255   0 255   0
  //   row3:   0   0   0   0 255 255 255 255
  // 2x2 block means (round-half-up), e.g.
  //   out(0,0)=mean(255,255,255,255)=255   out(0,1)=mean(255,255,0,0)=128
  //   out(1,0)=mean(255,255,255,0)... = rows{2,3} cols{0,1} = mean(255,0,0,0)=64
  //   out(1,2)= rows{2,3} cols{4,5}     = mean(255,0,255,255)=191
  let expected = block_mean_2x2(&expand_luma(&PATTERN, false));
  assert_eq!(expected, vec![255, 128, 0, 0, 64, 64, 191, 191]);
  assert_eq!(
    luma, expected,
    "luma must be the 2x2 block mean of the expanded bits"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_resample_all_outputs_match_direct_mono_over_binned_luma() {
  let src = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();

  let mut luma = vec![0u8; OUT_W * OUT_H];
  let mut luma_u16 = vec![0u16; OUT_W * OUT_H];
  let mut rgb = vec![0u8; OUT_W * OUT_H * 3];
  let mut rgba = vec![0u8; OUT_W * OUT_H * 4];
  let mut rgb_u16 = vec![0u16; OUT_W * OUT_H * 3];
  let mut rgba_u16 = vec![0u16; OUT_W * OUT_H * 4];
  let mut hp = vec![0u8; OUT_W * OUT_H];
  let mut sp = vec![0u8; OUT_W * OUT_H];
  let mut vp = vec![0u8; OUT_W * OUT_H];
  {
    let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_hsv(&mut hp, &mut sp, &mut vp)
    .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  // Oracle: the direct mono path broadcasts each (binned) luma to
  // R=G=B (alpha opaque), copies it to luma, zero-extends to u16, and
  // emits H=0/S=0/V=Y. Apply that to the area-binned luma reference.
  let y_ref = block_mean_2x2(&expand_luma(&PATTERN, false));

  assert_eq!(luma, y_ref);
  let y_ref_u16: Vec<u16> = y_ref.iter().map(|&y| y as u16).collect();
  assert_eq!(luma_u16, y_ref_u16);

  for (px, &y) in rgb.chunks_exact(3).zip(&y_ref) {
    assert_eq!(px, [y, y, y]);
  }
  for (px, &y) in rgba.chunks_exact(4).zip(&y_ref) {
    assert_eq!(px, [y, y, y, 0xFF]);
  }
  for (px, &y) in rgb_u16.chunks_exact(3).zip(&y_ref) {
    let y16 = y as u16;
    assert_eq!(px, [y16, y16, y16]);
  }
  for (px, &y) in rgba_u16.chunks_exact(4).zip(&y_ref) {
    let y16 = y as u16;
    assert_eq!(px, [y16, y16, y16, 0x00FF]);
  }
  assert_eq!(vp, y_ref, "HSV V must be the binned luma");
  assert!(hp.iter().all(|&h| h == 0), "HSV H must be 0 (achromatic)");
  assert!(sp.iter().all(|&s| s == 0), "HSV S must be 0 (achromatic)");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monowhite_resample_all_outputs_match_direct_mono_over_binned_luma() {
  let src = MonowhiteFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();

  let mut luma = vec![0u8; OUT_W * OUT_H];
  let mut rgb = vec![0u8; OUT_W * OUT_H * 3];
  let mut rgba = vec![0u8; OUT_W * OUT_H * 4];
  {
    let mut sink = MixedSinker::<Monowhite, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
    monowhite_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  // Monowhite inverts polarity: the expanded plane is the complement of
  // Monoblack's, so the binned reference differs.
  let y_ref = block_mean_2x2(&expand_luma(&PATTERN, true));
  assert_eq!(luma, y_ref);
  for (px, &y) in rgb.chunks_exact(3).zip(&y_ref) {
    assert_eq!(px, [y, y, y]);
  }
  for (px, &y) in rgba.chunks_exact(4).zip(&y_ref) {
    assert_eq!(px, [y, y, y, 0xFF]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_identity_plan_matches_new_sink() {
  let src = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();

  let mut direct_rgb = vec![0u8; SRC_W * SRC_H * 3];
  let mut direct_rgba = vec![0u8; SRC_W * SRC_H * 4];
  {
    let mut sink = MixedSinker::<Monoblack>::new(SRC_W, SRC_H)
      .with_rgb(&mut direct_rgb)
      .unwrap()
      .with_rgba(&mut direct_rgba)
      .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  let mut area_rgb = vec![0u8; SRC_W * SRC_H * 3];
  let mut area_rgba = vec![0u8; SRC_W * SRC_H * 4];
  {
    // An identity (no-op) resampler plan must take the direct path and
    // stay byte-identical.
    let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(SRC_W, SRC_H),
    )
    .unwrap()
    .with_rgb(&mut area_rgb)
    .unwrap()
    .with_rgba(&mut area_rgba)
    .unwrap();
    monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  assert_eq!(
    direct_rgb, area_rgb,
    "identity rgb must match the direct path"
  );
  assert_eq!(
    direct_rgba, area_rgba,
    "identity rgba must match the direct path"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_resample_reuses_luma_stream_across_frames() {
  // A reused sink must reset the luma stream each frame; without the
  // reset, frame 2's row 0 is rejected as out-of-sequence and the luma
  // never reflects frame 2.
  let frame1 = PATTERN;
  let frame2: [u8; SRC_H * STRIDE] = [!PATTERN[0], !PATTERN[1], !PATTERN[2], !PATTERN[3]];

  let mut luma = vec![0u8; OUT_W * OUT_H];
  {
    let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
      SRC_W,
      SRC_H,
      AreaResampler::to(OUT_W, OUT_H),
    )
    .unwrap()
    .with_luma(&mut luma)
    .unwrap();
    let f1 = MonoblackFrame::try_new(&frame1, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();
    monoblack_to(&f1, true, ColorMatrix::Bt709, &mut sink).unwrap();
    let f2 = MonoblackFrame::try_new(&frame2, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();
    monoblack_to(&f2, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  let y2_ref = block_mean_2x2(&expand_luma(&frame2, false));
  assert_eq!(
    luma, y2_ref,
    "frame 2 luma must area-downscale frame 2's bits"
  );
}

#[test]
fn monoblack_resample_no_outputs_is_a_no_op() {
  let src = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();
  let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap();
  // No outputs attached: a legal no-op, accepted without error. The
  // sequence guard's no-output branch returns before any stream is
  // created, so the luma stream is never allocated.
  monoblack_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  assert!(
    !sink.luma_stream_allocated(),
    "a no-output frame must not allocate the luma stream"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn monoblack_resample_begin_frame_resets_stream_sequencing() {
  // `MonoblackRow::new` is crate-private to mediaframe, so a misordered
  // first row cannot be hand-fed to `process` — the out-of-sequence
  // *rejection* (OutOfSequenceRow) is reachable only through a public
  // row ctor (the macro-generated formats, e.g. the Yuv411p suite, which
  // shares `mono_luma_resample`'s sequence-check body verbatim). What is
  // reachable for mono is the positive guarantee the rejection protects:
  // after a full frame advances the stream to `next_y == SRC_H`,
  // `begin_frame` must reset it so the next frame's row 0 is back in
  // sequence rather than rejected.
  let f1 = MonoblackFrame::try_new(&PATTERN, SRC_W as u32, SRC_H as u32, STRIDE as u32).unwrap();
  let mut luma = vec![0u8; OUT_W * OUT_H];
  let mut sink = MixedSinker::<Monoblack, AreaResampler>::with_resampler(
    SRC_W,
    SRC_H,
    AreaResampler::to(OUT_W, OUT_H),
  )
  .unwrap()
  .with_luma(&mut luma)
  .unwrap();
  monoblack_to(&f1, true, ColorMatrix::Bt709, &mut sink).unwrap();
  assert!(
    sink.luma_stream_allocated(),
    "the first frame must have created the luma stream"
  );
  // A second walk re-enters begin_frame (reset to next_y == 0) and must
  // succeed in strict order on the SAME stream.
  monoblack_to(&f1, true, ColorMatrix::Bt709, &mut sink).unwrap();
  let y_ref = block_mean_2x2(&expand_luma(&PATTERN, false));
  assert_eq!(
    luma, y_ref,
    "frame 2 must re-bin after the begin_frame reset"
  );
}
