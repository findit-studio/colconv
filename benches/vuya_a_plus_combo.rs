//! VUYA `with_rgb + with_rgba` combo benchmarks: Strategy A+ vs simulated 2-kernel.
//!
//! Compares the v0.18+ Strategy A+ flow (single chroma kernel → `expand_rgb_to_rgba_row`
//! → α-overwrite from source) against the simulated v0.17 baseline (two independent
//! chroma kernels — runs chroma math TWICE). Both flows are measured at:
//! - `use_simd = true`  — best available SIMD backend on the host
//! - `use_simd = false` — forced scalar reference path
//!
//! **Approach: Option A** — `expand_rgb_to_rgba_row` and `alpha_extract` helpers are
//! exposed as `#[doc(hidden)] pub` in `crate::row` so bench binaries can import them
//! directly. This lets us isolate per-row cost rather than measuring sinker overhead.
//!
//! Per-row throughput at 720p (1280), 1080p (1920), and 4K (3840).
//! Throughput metric: total output bytes (RGB + RGBA combined = `w * 7`).
//!
//! Bench paths:
//! - `a_plus_simd`      — rgb_row(simd) + expand + α_extract(simd)   ← current A+
//! - `a_plus_scalar`    — rgb_row(scalar) + expand + α_extract(scalar)
//! - `two_kernel_simd`  — rgb_row(simd) + rgba_row(simd)              ← pre-A+ baseline
//! - `two_kernel_scalar`— rgb_row(scalar) + rgba_row(scalar)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{
  ColorMatrix,
  row::{alpha_extract, expand_rgb_to_rgba_row, vuya_to_rgb_row, vuya_to_rgba_row},
};

fn fill_pseudo_random(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

fn bench(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("vuya_a_plus_combo");

  for &w in WIDTHS {
    let mut packed = std::vec![0u8; w * 4];
    fill_pseudo_random(&mut packed, 0x1111);
    let mut rgb = std::vec![0u8; w * 3];
    let mut rgba = std::vec![0u8; w * 4];

    // Throughput: combined RGB + RGBA output bytes written per row.
    group.throughput(Throughput::Bytes((w * 7) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      // ---- A+ combo: rgb kernel → expand → α-overwrite from source --------
      // Mirrors the v0.18+ MixedSinker Strategy A+ path for VUYA when both
      // with_rgb and with_rgba are attached. Chroma math runs ONCE.
      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &w| {
          b.iter(|| {
            vuya_to_rgb_row(
              black_box(&packed),
              black_box(&mut rgb),
              w,
              MATRIX,
              FULL_RANGE,
              use_simd,
            );
            // Expand RGB → RGBA with opaque α (scalar, L1-hot from prior write).
            expand_rgb_to_rgba_row(black_box(&rgb), black_box(&mut rgba), w);
            // Overwrite α channel from VUYA source bytes (SIMD-dispatched).
            alpha_extract::copy_alpha_packed_u8x4_at_3(
              black_box(&packed),
              black_box(&mut rgba),
              w,
              use_simd,
            );
            black_box((&rgb, &rgba));
          });
        },
      );

      // ---- 2-kernel combo: rgb kernel + rgba-with-α-src kernel -------------
      // Simulates the pre-A+ v0.17 path where each output format ran its own
      // full chroma kernel. Chroma math runs TWICE.
      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &w| {
          b.iter(|| {
            vuya_to_rgb_row(
              black_box(&packed),
              black_box(&mut rgb),
              w,
              MATRIX,
              FULL_RANGE,
              use_simd,
            );
            vuya_to_rgba_row(
              black_box(&packed),
              black_box(&mut rgba),
              w,
              MATRIX,
              FULL_RANGE,
              use_simd,
            );
            black_box((&rgb, &rgba));
          });
        },
      );
    }
  }

  group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
