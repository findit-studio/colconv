//! Packed YUV 4:1:1 (8-bit) dispatchers — Tier 5.25 source-side
//! support for `uyyvyy411` (DV legacy).
//!
//! Two row-conversion entries (RGB, RGBA) plus one luma-extraction
//! entry. Routes through the standard `cfg_select!` per-arch block;
//! `use_simd = false` forces scalar.
//!
//! Width must be a multiple of 4 — fast-path-asserted at every public
//! entry to gate entry into unsafe SIMD loads.

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::{avx2_available, avx512_available, sse41_available};
use crate::{
  ColorMatrix,
  row::{packed_yuv411_row_bytes, rgb_row_bytes, rgba_row_bytes, scalar},
};

/// Converts one row of UYYVYY411 to packed RGB. See
/// [`scalar::uyyvyy411_to_rgb_row`] for byte layout / numerical
/// contract. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn uyyvyy411_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  assert!(
    packed.len() >= packed_yuv411_row_bytes(width),
    "packed row too short"
  );
  assert!(
    rgb_out.len() >= rgb_row_bytes(width),
    "rgb_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe { arch::neon::uyyvyy411_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::uyyvyy411_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::uyyvyy411_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::uyyvyy411_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::uyyvyy411_to_rgb_row(packed, rgb_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::uyyvyy411_to_rgb_row(packed, rgb_out, width, matrix, full_range);
}

/// Converts one row of UYYVYY411 to packed RGBA (alpha = `0xFF`).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn uyyvyy411_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
  use_simd: bool,
) {
  assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  assert!(
    packed.len() >= packed_yuv411_row_bytes(width),
    "packed row too short"
  );
  assert!(
    rgba_out.len() >= rgba_row_bytes(width),
    "rgba_out row too short"
  );

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe { arch::neon::uyyvyy411_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::uyyvyy411_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::uyyvyy411_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::uyyvyy411_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::uyyvyy411_to_rgba_row(packed, rgba_out, width, matrix, full_range); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::uyyvyy411_to_rgba_row(packed, rgba_out, width, matrix, full_range);
}

/// Extracts one row of 8-bit luma from a packed UYYVYY411 buffer.
/// Y bytes live at offsets 1, 2, 4, 5 of each 6-byte / 4-pixel block.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn uyyvyy411_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize, use_simd: bool) {
  assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  assert!(
    packed.len() >= packed_yuv411_row_bytes(width),
    "packed row too short"
  );
  assert!(luma_out.len() >= width, "luma_out row too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe { arch::neon::uyyvyy411_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::uyyvyy411_to_luma_row(packed, luma_out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::uyyvyy411_to_luma_row(packed, luma_out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::uyyvyy411_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::uyyvyy411_to_luma_row(packed, luma_out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::uyyvyy411_to_luma_row(packed, luma_out, width);
}

/// Extracts one row of u16 luma (zero-extended Y bytes) from a packed
/// UYYVYY411 buffer. `use_simd = false` forces scalar.
#[cfg_attr(not(any(feature = "std", feature = "alloc")), allow(dead_code))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn uyyvyy411_to_luma_u16_row(
  packed: &[u8],
  out: &mut [u16],
  width: usize,
  use_simd: bool,
) {
  assert_eq!(
    width & 3,
    0,
    "packed YUV 4:1:1 requires width multiple of 4"
  );
  assert!(
    packed.len() >= packed_yuv411_row_bytes(width),
    "packed row too short"
  );
  assert!(out.len() >= width, "out too short");

  if use_simd {
    cfg_select! {
      target_arch = "aarch64" => {
        if neon_available() {
          // SAFETY: NEON verified at runtime.
          unsafe { arch::neon::uyyvyy411_to_luma_u16_row(packed, out, width); }
          return;
        }
      },
      target_arch = "x86_64" => {
        if avx512_available() {
          // SAFETY: AVX-512BW verified.
          unsafe { arch::x86_avx512::uyyvyy411_to_luma_u16_row(packed, out, width); }
          return;
        }
        if avx2_available() {
          // SAFETY: AVX2 verified.
          unsafe { arch::x86_avx2::uyyvyy411_to_luma_u16_row(packed, out, width); }
          return;
        }
        if sse41_available() {
          // SAFETY: SSE4.1 verified.
          unsafe { arch::x86_sse41::uyyvyy411_to_luma_u16_row(packed, out, width); }
          return;
        }
      },
      target_arch = "wasm32" => {
        if simd128_available() {
          // SAFETY: simd128 compile-time verified.
          unsafe { arch::wasm_simd128::uyyvyy411_to_luma_u16_row(packed, out, width); }
          return;
        }
      },
      _ => {}
    }
  }

  scalar::uyyvyy411_to_luma_u16_row(packed, out, width);
}
