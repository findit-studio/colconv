//! 8-bit YUV 4:1:0 → RGB / RGBA dispatchers (`yuv_410_to_rgb_row`,
//! `yuv_410_to_rgba_row`). Tier 1 P3 legacy (Cinepak / Sorenson).
//!
//! Backends: scalar (always) + NEON (aarch64). The other SIMD tiers
//! (SSE4.1 / AVX2 / AVX-512 / wasm32 simd128) intentionally fall
//! through to scalar for this format — 4:1:0 has 1/4 the chroma math
//! density of 4:2:0, modern decoders almost never produce it, and
//! the maintenance cost of four extra hand-rolled kernels for a
//! format with this usage profile isn't justified. Scalar is fast
//! enough for the legacy decode-side use case and the dispatcher
//! preserves the option to add more backends later.

#[cfg(target_arch = "aarch64")]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
use crate::{
  ColorMatrix,
  row::{rgb_row_bytes, rgba_row_bytes, scalar},
};

/// Converts one row of 4:1:0 YUV to packed RGB.
///
/// Dispatches to the best available backend for the current target.
/// See `scalar::yuv_410_to_rgb_row` for the full semantic
/// specification (range handling, matrix definitions, output layout).
///
/// `use_simd = false` forces the scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_410_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  // Runtime asserts at the dispatcher boundary — same rationale as the
  // 4:2:0 sibling. Unsafe SIMD kernels rely on these in release builds.
  assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  let rgb_min = rgb_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");

  if use_simd {
    #[cfg(target_arch = "aarch64")]
    if neon_available() {
      // SAFETY: `neon_available()` verified NEON is present.
      // Bounds / parity invariants are the caller's obligation.
      unsafe {
        arch::neon::yuv_410_to_rgb_row(y, u_quarter, v_quarter, rgb_out, width, matrix, full_range);
      }
      return;
    }
    // Other architectures (x86_64 / wasm32 / s390x / riscv) fall
    // through to scalar — see module docs for the rationale.
  }

  scalar::yuv_410_to_rgb_row(y, u_quarter, v_quarter, rgb_out, width, matrix, full_range);
}

/// Converts one row of 4:1:0 YUV to packed **RGBA** (8-bit).
///
/// Same numerical contract as [`yuv_410_to_rgb_row`]; the only
/// differences are the per-pixel stride (4 vs 3) and the alpha byte
/// (`0xFF`, opaque, for every pixel — sources without an alpha plane
/// produce opaque output). The first three bytes per pixel are
/// byte-identical to what [`yuv_410_to_rgb_row`] would write.
///
/// `rgba_out.len() >= 4 * width`. `use_simd = false` forces the
/// scalar reference path.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn yuv_410_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  let rgba_min = rgba_row_bytes(width);
  assert!(y.len() >= width, "y row too short");
  assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  assert!(rgba_out.len() >= rgba_min, "rgba_out row too short");

  if use_simd {
    #[cfg(target_arch = "aarch64")]
    if neon_available() {
      // SAFETY: `neon_available()` verified NEON is present.
      unsafe {
        arch::neon::yuv_410_to_rgba_row(
          y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
        );
      }
      return;
    }
  }

  scalar::yuv_410_to_rgba_row(y, u_quarter, v_quarter, rgba_out, width, matrix, full_range);
}
