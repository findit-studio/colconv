//! AYUV64 `with_rgb + with_rgba` combo benchmarks: Strategy A+ vs simulated 2-kernel.
//!
//! AYUV64 (FFmpeg `AV_PIX_FMT_AYUV64LE`) carries a 16-bit source α channel, so the
//! A+ combo must handle two output depths:
//! - **u8 RGBA**: α depth-converted u16 → u8 via `>> 8`
//! - **u16 RGBA**: α written direct at native 16-bit depth
//!
//! Two Criterion groups are used — `ayuv64_a_plus_combo_u8` and
//! `ayuv64_a_plus_combo_u16` — so Criterion's report separates the two paths.
//!
//! Compares the v0.18+ Strategy A+ flow (single chroma kernel → `expand_*_row`
//! → α-overwrite from source) against the simulated v0.17 baseline (two independent
//! chroma kernels — runs chroma math TWICE). Both paths at `use_simd = true/false`.
//!
//! **Approach: Option A** — `expand_rgb_to_rgba_row`, `expand_rgb_u16_to_rgba_u16_row`,
//! and `alpha_extract` helpers are exposed as `#[doc(hidden)] pub` in `crate::row` so
//! bench binaries can import them directly. This isolates per-row cost rather than
//! measuring sinker overhead.
//!
//! Per-row throughput at 720p (1280), 1080p (1920), and 4K (3840).
//!
//! Bench paths (per group):
//! - `a_plus_simd`      — rgb_row(simd) + expand + α_extract(simd)
//! - `a_plus_scalar`    — rgb_row(scalar) + expand + α_extract(scalar)
//! - `two_kernel_simd`  — rgb_row(simd) + rgba_row(simd)
//! - `two_kernel_scalar`— rgb_row(scalar) + rgba_row(scalar)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use colconv::{
  ColorMatrix,
  row::{
    alpha_extract, ayuv64_to_rgb_row, ayuv64_to_rgb_u16_row, ayuv64_to_rgba_row,
    ayuv64_to_rgba_u16_row, expand_rgb_to_rgba_row, expand_rgb_u16_to_rgba_u16_row,
  },
};

fn fill_pseudo_random_u16(buf: &mut [u16], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    // Produce values in 0..=65535 (full u16 range) — AYUV64 is 16-bit native.
    *b = (state >> 16) as u16;
  }
}

// ---- u8 RGBA group ----------------------------------------------------------

fn bench_u8(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  let mut group = c.benchmark_group("ayuv64_a_plus_combo_u8");

  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 4];
    fill_pseudo_random_u16(&mut packed, 0x1111);
    let mut rgb = std::vec![0u8; w * 3];
    let mut rgba = std::vec![0u8; w * 4];

    // Throughput: RGB (u8, 3 bytes/px) + RGBA (u8, 4 bytes/px) = 7 bytes/px.
    group.throughput(Throughput::Bytes((w * 7) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      // ---- A+ combo: rgb kernel → expand → α-overwrite (u16→u8 depth conv) --
      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &w| {
          b.iter(|| {
            ayuv64_to_rgb_row(
              black_box(&packed),
              black_box(&mut rgb),
              w,
              MATRIX,
              FULL_RANGE,
              use_simd,
            );
            // Expand u8 RGB → u8 RGBA with opaque α (scalar, L1-hot).
            expand_rgb_to_rgba_row(black_box(&rgb), black_box(&mut rgba), w);
            // Overwrite α channel: AYUV64 slot-0 u16 → u8 via >> 8.
            alpha_extract::copy_alpha_packed_u16x4_to_u8_at_0(
              black_box(&packed),
              black_box(&mut rgba),
              w,
              use_simd,
            );
            black_box((&rgb, &rgba));
          });
        },
      );

      // ---- 2-kernel combo: rgb kernel + rgba-with-α-src kernel ---------------
      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &w| {
          b.iter(|| {
            ayuv64_to_rgb_row(
              black_box(&packed),
              black_box(&mut rgb),
              w,
              MATRIX,
              FULL_RANGE,
              use_simd,
            );
            ayuv64_to_rgba_row(
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

// ---- u16 RGBA group ---------------------------------------------------------

fn bench_u16(c: &mut Criterion) {
  const WIDTHS: &[usize] = &[1280, 1920, 3840];
  const MATRIX: ColorMatrix = ColorMatrix::Bt709;
  const FULL_RANGE: bool = false;

  // AYUV64 native output depth is 16-bit; the A+ path uses BITS=16 for the α
  // max constant in expand_rgb_u16_to_rgba_u16_row.
  const BITS: u32 = 16;

  let mut group = c.benchmark_group("ayuv64_a_plus_combo_u16");

  for &w in WIDTHS {
    let mut packed = std::vec![0u16; w * 4];
    fill_pseudo_random_u16(&mut packed, 0x2222);
    let mut rgb = std::vec![0u16; w * 3];
    let mut rgba = std::vec![0u16; w * 4];

    // Throughput: u16 RGB (6 bytes/px) + u16 RGBA (8 bytes/px) = 14 bytes/px.
    group.throughput(Throughput::Bytes((w * 14) as u64));

    for use_simd in [false, true] {
      let simd_label = if use_simd { "simd" } else { "scalar" };

      // ---- A+ combo: u16 rgb kernel → u16 expand → u16 α-overwrite ----------
      group.bench_with_input(
        BenchmarkId::new(format!("a_plus_{simd_label}"), w),
        &w,
        |b, &w| {
          b.iter(|| {
            ayuv64_to_rgb_u16_row(
              black_box(&packed),
              black_box(&mut rgb),
              w,
              MATRIX,
              FULL_RANGE,
              use_simd,
            );
            // Expand u16 RGB → u16 RGBA with opaque α at BITS-bit max (scalar).
            expand_rgb_u16_to_rgba_u16_row::<BITS>(black_box(&rgb), black_box(&mut rgba), w);
            // Overwrite α channel: AYUV64 slot-0 u16 → u16 RGBA slot-3 (no conv).
            alpha_extract::copy_alpha_packed_u16x4_at_0(
              black_box(&packed),
              black_box(&mut rgba),
              w,
              use_simd,
            );
            black_box((&rgb, &rgba));
          });
        },
      );

      // ---- 2-kernel combo: u16 rgb kernel + u16 rgba-with-α-src kernel -------
      group.bench_with_input(
        BenchmarkId::new(format!("two_kernel_{simd_label}"), w),
        &w,
        |b, &w| {
          b.iter(|| {
            ayuv64_to_rgb_u16_row(
              black_box(&packed),
              black_box(&mut rgb),
              w,
              MATRIX,
              FULL_RANGE,
              use_simd,
            );
            ayuv64_to_rgba_u16_row(
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

criterion_group!(benches_u8, bench_u8);
criterion_group!(benches_u16, bench_u16);
criterion_main!(benches_u8, benches_u16);
