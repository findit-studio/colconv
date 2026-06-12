//! Full-pipeline fused-downscale gate: 1920x1080 -> 336x189 (the
//! SigLIP2-NaFlex analysis geometry from the design issues).
//!
//! Compares, per frame:
//! - the NATIVE tier (bin Y/U/V at output res, convert once per
//!   output row),
//! - the ROW-STAGE tier (convert each row at source res, then bin),
//! - the full-res conversion baseline (what convert-then-resize
//!   pipelines pay before they even start resizing),
//! plus the packed-RGB fused path, and luma-only variants (which
//! never read chroma).

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{
  ColorMatrix, PixelSink,
  frame::{Rgb24Frame, Yuv420pFrame},
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{Rgb24, Yuv420p, rgb24_to, yuv420p_to},
};

const SRC_W: usize = 1920;
const SRC_H: usize = 1080;
const OUT_W: usize = 336;
const OUT_H: usize = 189;

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  let mut y = std::vec![0u8; SRC_W * SRC_H];
  let mut u = std::vec![0u8; SRC_W * SRC_H / 4];
  let mut v = std::vec![0u8; SRC_W * SRC_H / 4];
  fill_pseudo_random(&mut y, 0x1111);
  fill_pseudo_random(&mut u, 0x2222);
  fill_pseudo_random(&mut v, 0x3333);

  let mut group = c.benchmark_group("fused_downscale_1080p_to_336x189");
  group.sample_size(20);

  let n = OUT_W * OUT_H;
  let mut rgb = std::vec![0u8; n * 3];
  let mut h = std::vec![0u8; n];
  let mut s = std::vec![0u8; n];
  let mut vv = std::vec![0u8; n];
  let mut luma = std::vec![0u8; n];

  for (name, native) in [
    ("yuv420p_rgb_hsv_native", true),
    ("yuv420p_rgb_hsv_rowstage", false),
  ] {
    group.bench_function(name, |b| {
      b.iter(|| {
        let src = Yuv420pFrame::new(
          &y,
          &u,
          &v,
          SRC_W as u32,
          SRC_H as u32,
          SRC_W as u32,
          (SRC_W / 2) as u32,
          (SRC_W / 2) as u32,
        );
        let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
          SRC_W,
          SRC_H,
          AreaResampler::to(OUT_W, OUT_H),
        )
        .unwrap()
        .with_native(native)
        .with_rgb(&mut rgb)
        .unwrap()
        .with_hsv(&mut h, &mut s, &mut vv)
        .unwrap();
        yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
        black_box(&rgb);
      });
    });
  }

  for (name, native) in [
    ("yuv420p_luma_only_native", true),
    ("yuv420p_luma_only_rowstage", false),
  ] {
    group.bench_function(name, |b| {
      b.iter(|| {
        let src = Yuv420pFrame::new(
          &y,
          &u,
          &v,
          SRC_W as u32,
          SRC_H as u32,
          SRC_W as u32,
          (SRC_W / 2) as u32,
          (SRC_W / 2) as u32,
        );
        let mut sink = MixedSinker::<Yuv420p, AreaResampler>::with_resampler(
          SRC_W,
          SRC_H,
          AreaResampler::to(OUT_W, OUT_H),
        )
        .unwrap()
        .with_native(native)
        .with_luma(&mut luma)
        .unwrap();
        yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
        black_box(&luma);
      });
    });
  }

  // What a convert-then-resize pipeline pays for the convert step
  // alone, before any resizing.
  let mut full_rgb = std::vec![0u8; SRC_W * SRC_H * 3];
  group.bench_function("yuv420p_fullres_convert_baseline", |b| {
    b.iter(|| {
      let src = Yuv420pFrame::new(
        &y,
        &u,
        &v,
        SRC_W as u32,
        SRC_H as u32,
        SRC_W as u32,
        (SRC_W / 2) as u32,
        (SRC_W / 2) as u32,
      );
      let mut sink = MixedSinker::<Yuv420p>::new(SRC_W, SRC_H)
        .with_rgb(&mut full_rgb)
        .unwrap();
      yuv420p_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
      black_box(&full_rgb);
    });
  });

  let mut packed = std::vec![0u8; SRC_W * SRC_H * 3];
  fill_pseudo_random(&mut packed, 0x4444);
  group.bench_function("rgb24_rgb_hsv_fused", |b| {
    b.iter(|| {
      let src = Rgb24Frame::new(&packed, SRC_W as u32, SRC_H as u32, (SRC_W * 3) as u32);
      let mut sink = MixedSinker::<Rgb24, AreaResampler>::with_resampler(
        SRC_W,
        SRC_H,
        AreaResampler::to(OUT_W, OUT_H),
      )
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut vv)
      .unwrap();
      rgb24_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
      black_box(&rgb);
    });
  });

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
