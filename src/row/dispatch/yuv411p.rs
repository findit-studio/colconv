//! 8-bit YUV 4:1:1 → RGB / RGBA dispatchers (`yuv_411_to_rgb_row`,
//! `yuv_411_to_rgba_row`).
//!
//! 4:1:1 is **legacy DV-NTSC** subsampling: chroma is quarter-width
//! and full-height. SIMD coverage today: NEON only. x86 (SSE4.1 /
//! AVX2 / AVX-512) and wasm32 simd128 backends fall through to the
//! scalar reference — 4:1:1 is rare enough to defer SIMD work on
//! those targets until a real workload demands it.

#[cfg(target_arch = "aarch64")]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use crate::{
  ColorMatrix,
  row::{rgb_row_bytes, rgba_row_bytes, scalar},
};

/// Converts one row of 4:1:1 YUV to packed RGB.
///
/// Dispatches to the best available backend for the current target.
/// See `scalar::yuv_411_to_rgb_row` for the full semantic
/// specification (range handling, matrix definitions, output layout).
///
/// `use_simd = false` forces the scalar reference path, bypassing any
/// SIMD backend. 4:1:1 currently has SIMD only on aarch64 / NEON;
/// every other target lands on the scalar fallback.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_411_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary. The unsafe NEON
  // kernel below relies on these invariants for pointer arithmetic,
  // so we validate in *release* builds too — not just under
  // `debug_assert!`. The kernel keeps its own `debug_assert!`s as
  // internal sanity checks. `rgb_min` uses `checked_mul` because
  // `3 * width` can wrap `usize` on 32-bit targets (wasm32, i686)
  // for extreme widths.
  assert_eq!(width & 3, 0, "YUV 4:1:1 requires width % 4 == 0");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: `neon_available()` verified NEON is present on this
          // CPU. Bounds / parity invariants are the caller's obligation
          // (asserted above); the kernel re-checks them with
          // `debug_assert` in debug builds.
          unsafe {
            arch::neon::yuv_411_to_rgb_row(
              y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      // 4:1:1 SIMD coverage on x86_64 / wasm32 deferred — DV-NTSC
      // legacy is rare enough that the scalar fallback's perf is
      // adequate for the foreseeable future.
      _ => {}
    }
  }

  scalar::yuv_411_to_rgb_row(y, u_quarter, v_quarter, rgb_out, width, matrix, full_range);
}

/// Converts one row of 4:1:1 YUV to packed **RGBA** (8-bit).
///
/// Same numerical contract as [`yuv_411_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque, for every pixel — sources without an alpha plane
/// produce opaque output).
///
/// `rgba_out.len() >= 4 * width`. `use_simd = false` forces the
/// scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_411_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary — see
  // [`yuv_411_to_rgb_row`] for rationale, including the checked
  // `width × 4` multiplication via [`rgba_row_bytes`].
  assert_eq!(width & 3, 0, "YUV 4:1:1 requires width % 4 == 0");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified present; bounds / parity asserted above.
          unsafe {
            arch::neon::yuv_411_to_rgba_row(
              y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
            );
          }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::yuv_411_to_rgba_row(y, u_quarter, v_quarter, rgba_out, width, matrix, full_range);
}
