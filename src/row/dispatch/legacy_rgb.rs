//! Runtime SIMD dispatchers for legacy 16-bit packed-RGB source formats (Tier 7).
//!
//! Six source formats, each with four output variants:
//! - [`rgb565_to_rgb_row`] / [`rgb565_to_rgba_row`] /
//!   [`rgb565_to_rgb_u16_row`] / [`rgb565_to_rgba_u16_row`]
//! - Likewise for `bgr565`, `rgb555`, `bgr555`, `rgb444`, `bgr444`.
//!
//! Input planes are `&[u8]` byte-strided at **2 bytes/pixel** (one LE `u16`
//! word per pixel). Buffer minimum checks use the crate helpers
//! [`packed_yuv422_row_bytes`](crate::row::packed_yuv422_row_bytes) (for the
//! 2-bytes/pixel input side) and [`rgb_row_bytes`] / [`rgba_row_bytes`] /
//! [`rgb_row_elems`] / [`rgba_row_elems`] (for the output side) to guard
//! 32-bit target overflow.
//!
//! **Task 3 (scalar-only):** all 24 entries fall through to the scalar reference
//! implementation. SIMD branches (NEON / SSE4.1 / AVX2 / AVX-512 / wasm-simd128)
//! will be wired in Tasks 4-8.

#[cfg(any(
  target_arch = "aarch64",
  target_arch = "x86_64",
  target_arch = "wasm32"
))]
use crate::row::arch;
#[cfg(target_arch = "x86_64")]
use crate::row::avx2_available;
#[cfg(target_arch = "x86_64")]
use crate::row::avx512_available;
#[cfg(target_arch = "aarch64")]
use crate::row::neon_available;
#[cfg(target_arch = "wasm32")]
use crate::row::simd128_available;
#[cfg(target_arch = "x86_64")]
use crate::row::sse41_available;
use crate::row::{
  packed_yuv422_row_bytes, rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems, scalar,
};

// RGB565 (R5 G6 B5 — bits [15:11] [10:5] [4:0]).
/// Dispatches RGB565 → packed `R, G, B` bytes to the best available backend.
///
/// Channels are expanded via bit-replication to full u8:
/// R5 → `(r5 << 3) | (r5 >> 2)`, G6 → `(g6 << 2) | (g6 >> 4)`,
/// B5 → `(b5 << 3) | (b5 >> 2)`.
///
/// `use_simd = false` forces scalar (useful for tests and benchmarks).
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgb_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    // SAFETY: NEON verified available.
    unsafe {
      return arch::neon::legacy_rgb::rgb565_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    // SAFETY: AVX-512BW verified available (implies F).
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb565_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    // SAFETY: AVX2 verified available.
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb565_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    // SAFETY: SSE4.1 verified available.
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb565_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    // SAFETY: simd128 compile-time enabled.
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb565_to_rgb_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb565_to_rgb_row(src, rgb_out, width);
}

/// Dispatches RGB565 → packed `R, G, B, A` bytes (α = `0xFF`) to the best
/// available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgba_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb565_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb565_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb565_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb565_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb565_to_rgba_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb565_to_rgba_row(src, rgba_out, width);
}

/// Dispatches RGB565 → packed `R, G, B` **u16** elements (native bit-width,
/// no expansion) to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb565_to_rgb_u16_row(src: &[u8], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgb_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb565_to_rgb_u16_row(src, rgb_out, width);
}

/// Dispatches RGB565 → packed `R, G, B, A` **u16** elements (α = `0xFFFF`)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb565_to_rgba_u16_row(src: &[u8], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgba_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb565_to_rgba_u16_row(src, rgba_out, width);
}

// BGR565 (B5 G6 R5 — bits [15:11]=B5, [10:5]=G6, [4:0]=R5).
/// Dispatches BGR565 → packed `R, G, B` bytes (output always R-first) to the
/// best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgb_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr565_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr565_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr565_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr565_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr565_to_rgb_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr565_to_rgb_row(src, rgb_out, width);
}

/// Dispatches BGR565 → packed `R, G, B, A` bytes (α = `0xFF`) to the best
/// available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgba_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr565_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr565_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr565_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr565_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr565_to_rgba_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr565_to_rgba_row(src, rgba_out, width);
}

/// Dispatches BGR565 → packed `R, G, B` **u16** elements (native bit-width)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr565_to_rgb_u16_row(src: &[u8], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgb_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr565_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr565_to_rgb_u16_row(src, rgb_out, width);
}

/// Dispatches BGR565 → packed `R, G, B, A` **u16** elements (α = `0xFFFF`)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr565_to_rgba_u16_row(src: &[u8], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgba_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr565_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr565_to_rgba_u16_row(src, rgba_out, width);
}

// RGB555 (1X R5 G5 B5 — bits [14:10]=R5, [9:5]=G5, [4:0]=B5, bit 15 ignored).
/// Dispatches RGB555 → packed `R, G, B` bytes to the best available backend.
/// `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgb_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb555_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb555_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb555_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb555_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb555_to_rgb_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb555_to_rgb_row(src, rgb_out, width);
}

/// Dispatches RGB555 → packed `R, G, B, A` bytes (α = `0xFF`) to the best
/// available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgba_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb555_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb555_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb555_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb555_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb555_to_rgba_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb555_to_rgba_row(src, rgba_out, width);
}

/// Dispatches RGB555 → packed `R, G, B` **u16** elements (native bit-width)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb555_to_rgb_u16_row(src: &[u8], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgb_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb555_to_rgb_u16_row(src, rgb_out, width);
}

/// Dispatches RGB555 → packed `R, G, B, A` **u16** elements (α = `0xFFFF`)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb555_to_rgba_u16_row(src: &[u8], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgba_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb555_to_rgba_u16_row(src, rgba_out, width);
}

// BGR555 (1X B5 G5 R5 — bits [14:10]=B5, [9:5]=G5, [4:0]=R5, bit 15 ignored).
/// Dispatches BGR555 → packed `R, G, B` bytes (output always R-first) to the
/// best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgb_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr555_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr555_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr555_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr555_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr555_to_rgb_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr555_to_rgb_row(src, rgb_out, width);
}

/// Dispatches BGR555 → packed `R, G, B, A` bytes (α = `0xFF`) to the best
/// available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgba_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr555_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr555_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr555_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr555_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr555_to_rgba_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr555_to_rgba_row(src, rgba_out, width);
}

/// Dispatches BGR555 → packed `R, G, B` **u16** elements (native bit-width)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr555_to_rgb_u16_row(src: &[u8], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgb_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr555_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr555_to_rgb_u16_row(src, rgb_out, width);
}

/// Dispatches BGR555 → packed `R, G, B, A` **u16** elements (α = `0xFFFF`)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr555_to_rgba_u16_row(src: &[u8], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgba_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr555_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr555_to_rgba_u16_row(src, rgba_out, width);
}

// RGB444 (4X R4 G4 B4 — bits [11:8]=R4, [7:4]=G4, [3:0]=B4, bits [15:12] ignored).
/// Dispatches RGB444 → packed `R, G, B` bytes to the best available backend.
/// `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgb_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb444_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb444_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb444_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb444_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb444_to_rgb_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb444_to_rgb_row(src, rgb_out, width);
}

/// Dispatches RGB444 → packed `R, G, B, A` bytes (α = `0xFF`) to the best
/// available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgba_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb444_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb444_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb444_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb444_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb444_to_rgba_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb444_to_rgba_row(src, rgba_out, width);
}

/// Dispatches RGB444 → packed `R, G, B` **u16** elements (native bit-width)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb444_to_rgb_u16_row(src: &[u8], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgb_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb444_to_rgb_u16_row(src, rgb_out, width);
}

/// Dispatches RGB444 → packed `R, G, B, A` **u16** elements (α = `0xFFFF`)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn rgb444_to_rgba_u16_row(src: &[u8], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgba_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::rgb444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::rgb444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::rgb444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::rgb444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::rgb444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::rgb444_to_rgba_u16_row(src, rgba_out, width);
}

// BGR444 (4X B4 G4 R4 — bits [11:8]=B4, [7:4]=G4, [3:0]=R4, bits [15:12] ignored).
/// Dispatches BGR444 → packed `R, G, B` bytes (output always R-first) to the
/// best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgb_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr444_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr444_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr444_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr444_to_rgb_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr444_to_rgb_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr444_to_rgb_row(src, rgb_out, width);
}

/// Dispatches BGR444 → packed `R, G, B, A` bytes (α = `0xFF`) to the best
/// available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize, use_simd: bool) {
  let out_min = rgba_row_bytes(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr444_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr444_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr444_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr444_to_rgba_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr444_to_rgba_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr444_to_rgba_row(src, rgba_out, width);
}

/// Dispatches BGR444 → packed `R, G, B` **u16** elements (native bit-width)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr444_to_rgb_u16_row(src: &[u8], rgb_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgb_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr444_to_rgb_u16_row(src, rgb_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr444_to_rgb_u16_row(src, rgb_out, width);
}

/// Dispatches BGR444 → packed `R, G, B, A` **u16** elements (α = `0xFFFF`)
/// to the best available backend. `use_simd = false` forces scalar.
#[cfg_attr(not(tarpaulin), inline(always))]
pub fn bgr444_to_rgba_u16_row(src: &[u8], rgba_out: &mut [u16], width: usize, use_simd: bool) {
  let out_min = rgba_row_elems(width);
  let src_min = packed_yuv422_row_bytes(width);
  assert!(src.len() >= src_min, "src row too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");
  #[cfg(target_arch = "aarch64")]
  if use_simd && neon_available() {
    unsafe {
      return arch::neon::legacy_rgb::bgr444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx512_available() {
    unsafe {
      return arch::x86_avx512::legacy_rgb::bgr444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && avx2_available() {
    unsafe {
      return arch::x86_avx2::legacy_rgb::bgr444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "x86_64")]
  if use_simd && sse41_available() {
    unsafe {
      return arch::x86_sse41::legacy_rgb::bgr444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  #[cfg(target_arch = "wasm32")]
  if use_simd && simd128_available() {
    unsafe {
      return arch::wasm_simd128::legacy_rgb::bgr444_to_rgba_u16_row(src, rgba_out, width);
    }
  }
  let _ = use_simd;
  scalar::legacy_rgb::bgr444_to_rgba_u16_row(src, rgba_out, width);
}

// =========================================================================
// Legacy bit-packed RGB/BGR (8bpp 3:3:2 + 1:2:1; 4bpp 1:2:1 two-per-byte)
// (AV_PIX_FMT_RGB8 / BGR8 / RGB4_BYTE / BGR4_BYTE / RGB4 / BGR4)
//
// Input planes are `&[u8]`: 1 byte/pixel for `Rgb8` / `Bgr8` / `Rgb4Byte` /
// `Bgr4Byte` (`src_min = width`); `width.div_ceil(2)` bytes for the 4-bpp
// `Rgb4` / `Bgr4` (two pixels per byte). Output-side minimums reuse
// [`rgb_row_bytes`] / [`rgba_row_bytes`] / [`rgb_row_elems`] /
// [`rgba_row_elems`] (32-bit overflow guard).
//
// Each entry routes to the best runtime-detected backend
// (NEON / AVX-512 / AVX2 / SSE4.1 / wasm-simd128) with a scalar fallback,
// exactly like the 16-bit packed entries above. The arch kernels share the
// dispatcher's function name, so the routing references
// `arch::<backend>::legacy_rgb::<name>` directly.
// =========================================================================

/// Emits one runtime-dispatched kernel for a legacy bit-packed RGB/BGR format.
/// `$src_kind` is `byte` (1 byte/pixel) or `nibble` (`width.div_ceil(2)`
/// bytes, two pixels per byte); `u8` / `u16` selects the output element type.
macro_rules! packed_rgb_lowbit_dispatch {
  ($name:ident, u8, $out_guard:path, $scalar:path, $src_kind:tt, $doc:expr) => {
    #[doc = $doc]
    ///
    /// `use_simd = false` forces scalar (useful for tests and benchmarks).
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn $name(src: &[u8], out: &mut [u8], width: usize, use_simd: bool) {
      let out_min = $out_guard(width);
      let src_min = packed_rgb_lowbit_dispatch!(@src $src_kind, width);
      assert!(src.len() >= src_min, "src row too short");
      assert!(out.len() >= out_min, "output row too short");
      packed_rgb_lowbit_dispatch!(@route $name, src, out, width, use_simd);
      scalar::legacy_rgb::$name(src, out, width);
    }
  };
  ($name:ident, u16, $out_guard:path, $scalar:path, $src_kind:tt, $doc:expr) => {
    #[doc = $doc]
    ///
    /// `use_simd = false` forces scalar (useful for tests and benchmarks).
    #[cfg_attr(not(tarpaulin), inline(always))]
    pub fn $name(src: &[u8], out: &mut [u16], width: usize, use_simd: bool) {
      let out_min = $out_guard(width);
      let src_min = packed_rgb_lowbit_dispatch!(@src $src_kind, width);
      assert!(src.len() >= src_min, "src row too short");
      assert!(out.len() >= out_min, "output row too short");
      packed_rgb_lowbit_dispatch!(@route $name, src, out, width, use_simd);
      scalar::legacy_rgb::$name(src, out, width);
    }
  };
  (@src byte, $w:expr) => {
    $w
  };
  (@src nibble, $w:expr) => {
    ($w).div_ceil(2)
  };
  // Runtime SIMD routing shared by both element types — each branch returns on
  // a hit; control falls through to the scalar tail otherwise.
  (@route $name:ident, $src:ident, $out:ident, $width:ident, $use_simd:ident) => {
    #[cfg(target_arch = "aarch64")]
    if $use_simd && neon_available() {
      // SAFETY: NEON verified available.
      unsafe {
        return arch::neon::legacy_rgb::$name($src, $out, $width);
      }
    }
    #[cfg(target_arch = "x86_64")]
    if $use_simd && avx512_available() {
      // SAFETY: AVX-512BW verified available (implies F).
      unsafe {
        return arch::x86_avx512::legacy_rgb::$name($src, $out, $width);
      }
    }
    #[cfg(target_arch = "x86_64")]
    if $use_simd && avx2_available() {
      // SAFETY: AVX2 verified available.
      unsafe {
        return arch::x86_avx2::legacy_rgb::$name($src, $out, $width);
      }
    }
    #[cfg(target_arch = "x86_64")]
    if $use_simd && sse41_available() {
      // SAFETY: SSE4.1 verified available.
      unsafe {
        return arch::x86_sse41::legacy_rgb::$name($src, $out, $width);
      }
    }
    #[cfg(target_arch = "wasm32")]
    if $use_simd && simd128_available() {
      // SAFETY: simd128 compile-time enabled.
      unsafe {
        return arch::wasm_simd128::legacy_rgb::$name($src, $out, $width);
      }
    }
    let _ = $use_simd;
  };
}

packed_rgb_lowbit_dispatch!(
  rgb8_to_rgb_row,
  u8,
  rgb_row_bytes,
  scalar::legacy_rgb::rgb8_to_rgb_row,
  byte,
  "Dispatches RGB8 (3:3:2) → packed `R, G, B` bytes (bit-replicated channels)."
);
packed_rgb_lowbit_dispatch!(
  rgb8_to_rgba_row,
  u8,
  rgba_row_bytes,
  scalar::legacy_rgb::rgb8_to_rgba_row,
  byte,
  "Dispatches RGB8 (3:3:2) → packed `R, G, B, A` bytes (α = `0xFF`)."
);
packed_rgb_lowbit_dispatch!(
  rgb8_to_rgb_u16_row,
  u16,
  rgb_row_elems,
  scalar::legacy_rgb::rgb8_to_rgb_u16_row,
  byte,
  "Dispatches RGB8 → packed `R, G, B` u16 (native 3/3/2-bit, no expansion)."
);
packed_rgb_lowbit_dispatch!(
  rgb8_to_rgba_u16_row,
  u16,
  rgba_row_elems,
  scalar::legacy_rgb::rgb8_to_rgba_u16_row,
  byte,
  "Dispatches RGB8 → packed `R, G, B, A` u16 (native, α = `0xFFFF`)."
);

packed_rgb_lowbit_dispatch!(
  bgr8_to_rgb_row,
  u8,
  rgb_row_bytes,
  scalar::legacy_rgb::bgr8_to_rgb_row,
  byte,
  "Dispatches BGR8 (3:3:2) → packed `R, G, B` bytes (output R-first)."
);
packed_rgb_lowbit_dispatch!(
  bgr8_to_rgba_row,
  u8,
  rgba_row_bytes,
  scalar::legacy_rgb::bgr8_to_rgba_row,
  byte,
  "Dispatches BGR8 (3:3:2) → packed `R, G, B, A` bytes (α = `0xFF`)."
);
packed_rgb_lowbit_dispatch!(
  bgr8_to_rgb_u16_row,
  u16,
  rgb_row_elems,
  scalar::legacy_rgb::bgr8_to_rgb_u16_row,
  byte,
  "Dispatches BGR8 → packed `R, G, B` u16 (native 3/3/2-bit, R-first)."
);
packed_rgb_lowbit_dispatch!(
  bgr8_to_rgba_u16_row,
  u16,
  rgba_row_elems,
  scalar::legacy_rgb::bgr8_to_rgba_u16_row,
  byte,
  "Dispatches BGR8 → packed `R, G, B, A` u16 (native, α = `0xFFFF`)."
);

packed_rgb_lowbit_dispatch!(
  rgb4_byte_to_rgb_row,
  u8,
  rgb_row_bytes,
  scalar::legacy_rgb::rgb4_byte_to_rgb_row,
  byte,
  "Dispatches RGB4_BYTE (1:2:1, low nibble) → packed `R, G, B` bytes."
);
packed_rgb_lowbit_dispatch!(
  rgb4_byte_to_rgba_row,
  u8,
  rgba_row_bytes,
  scalar::legacy_rgb::rgb4_byte_to_rgba_row,
  byte,
  "Dispatches RGB4_BYTE → packed `R, G, B, A` bytes (α = `0xFF`)."
);
packed_rgb_lowbit_dispatch!(
  rgb4_byte_to_rgb_u16_row,
  u16,
  rgb_row_elems,
  scalar::legacy_rgb::rgb4_byte_to_rgb_u16_row,
  byte,
  "Dispatches RGB4_BYTE → packed `R, G, B` u16 (native 1/2/1-bit)."
);
packed_rgb_lowbit_dispatch!(
  rgb4_byte_to_rgba_u16_row,
  u16,
  rgba_row_elems,
  scalar::legacy_rgb::rgb4_byte_to_rgba_u16_row,
  byte,
  "Dispatches RGB4_BYTE → packed `R, G, B, A` u16 (native, α = `0xFFFF`)."
);

packed_rgb_lowbit_dispatch!(
  bgr4_byte_to_rgb_row,
  u8,
  rgb_row_bytes,
  scalar::legacy_rgb::bgr4_byte_to_rgb_row,
  byte,
  "Dispatches BGR4_BYTE (1:2:1, low nibble) → packed `R, G, B` bytes (R-first)."
);
packed_rgb_lowbit_dispatch!(
  bgr4_byte_to_rgba_row,
  u8,
  rgba_row_bytes,
  scalar::legacy_rgb::bgr4_byte_to_rgba_row,
  byte,
  "Dispatches BGR4_BYTE → packed `R, G, B, A` bytes (α = `0xFF`)."
);
packed_rgb_lowbit_dispatch!(
  bgr4_byte_to_rgb_u16_row,
  u16,
  rgb_row_elems,
  scalar::legacy_rgb::bgr4_byte_to_rgb_u16_row,
  byte,
  "Dispatches BGR4_BYTE → packed `R, G, B` u16 (native 1/2/1-bit, R-first)."
);
packed_rgb_lowbit_dispatch!(
  bgr4_byte_to_rgba_u16_row,
  u16,
  rgba_row_elems,
  scalar::legacy_rgb::bgr4_byte_to_rgba_u16_row,
  byte,
  "Dispatches BGR4_BYTE → packed `R, G, B, A` u16 (native, α = `0xFFFF`)."
);

packed_rgb_lowbit_dispatch!(
  rgb4_to_rgb_row,
  u8,
  rgb_row_bytes,
  scalar::legacy_rgb::rgb4_to_rgb_row,
  nibble,
  "Dispatches RGB4 (4 bpp, two pixels/byte) → packed `R, G, B` bytes."
);
packed_rgb_lowbit_dispatch!(
  rgb4_to_rgba_row,
  u8,
  rgba_row_bytes,
  scalar::legacy_rgb::rgb4_to_rgba_row,
  nibble,
  "Dispatches RGB4 → packed `R, G, B, A` bytes (α = `0xFF`)."
);
packed_rgb_lowbit_dispatch!(
  rgb4_to_rgb_u16_row,
  u16,
  rgb_row_elems,
  scalar::legacy_rgb::rgb4_to_rgb_u16_row,
  nibble,
  "Dispatches RGB4 → packed `R, G, B` u16 (native 1/2/1-bit)."
);
packed_rgb_lowbit_dispatch!(
  rgb4_to_rgba_u16_row,
  u16,
  rgba_row_elems,
  scalar::legacy_rgb::rgb4_to_rgba_u16_row,
  nibble,
  "Dispatches RGB4 → packed `R, G, B, A` u16 (native, α = `0xFFFF`)."
);

packed_rgb_lowbit_dispatch!(
  bgr4_to_rgb_row,
  u8,
  rgb_row_bytes,
  scalar::legacy_rgb::bgr4_to_rgb_row,
  nibble,
  "Dispatches BGR4 (4 bpp, two pixels/byte) → packed `R, G, B` bytes (R-first)."
);
packed_rgb_lowbit_dispatch!(
  bgr4_to_rgba_row,
  u8,
  rgba_row_bytes,
  scalar::legacy_rgb::bgr4_to_rgba_row,
  nibble,
  "Dispatches BGR4 → packed `R, G, B, A` bytes (α = `0xFF`)."
);
packed_rgb_lowbit_dispatch!(
  bgr4_to_rgb_u16_row,
  u16,
  rgb_row_elems,
  scalar::legacy_rgb::bgr4_to_rgb_u16_row,
  nibble,
  "Dispatches BGR4 → packed `R, G, B` u16 (native 1/2/1-bit, R-first)."
);
packed_rgb_lowbit_dispatch!(
  bgr4_to_rgba_u16_row,
  u16,
  rgba_row_elems,
  scalar::legacy_rgb::bgr4_to_rgba_u16_row,
  nibble,
  "Dispatches BGR4 → packed `R, G, B, A` u16 (native, α = `0xFFFF`)."
);

// Overflow-guard tests — 32-bit target only.
#[cfg(all(test, feature = "std"))]
mod tests {
  //! Regression tests for the 32-bit overflow guards. Input side uses
  //! [`packed_yuv422_row_bytes`] (width x 2); output side uses
  //! [`rgb_row_bytes`] / [`rgba_row_bytes`] / [`rgb_row_elems`] /
  //! [`rgba_row_elems`]. Any of the four can overflow independently on a
  //! 32-bit target, so each has at least one dedicated test.

  #[cfg(target_pointer_width = "32")]
  use super::*;

  /// Smallest width whose `width x 2` overflows 32-bit `usize`.
  /// On a 32-bit target `usize::MAX == 2^32 - 1`, so `usize::MAX / 2 ==
  /// 2^31 - 1` (odd); adding 1 gives `2^31` (even). The parity fixup is
  /// a safety guard in case the arithmetic differs on a hypothetical platform.
  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_X2: usize = {
    let c = (usize::MAX / 2) + 1;
    c + (c & 1)
  };

  /// Smallest width whose `width x 3` overflows 32-bit `usize`.
  /// `usize::MAX / 3 == 1_431_655_765` (always odd on 32-bit), so `+ 1`
  /// gives an even value. The parity fixup keeps correctness on edge cases.
  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_X3: usize = {
    let c = (usize::MAX / 3) + 1;
    c + (c & 1)
  };

  /// Smallest width whose `width x 4` overflows 32-bit `usize`.
  /// `usize::MAX / 4 == 2^30 - 1` (odd on 32-bit), so `+ 1` is `2^30` (even).
  #[cfg(target_pointer_width = "32")]
  const OVERFLOW_WIDTH_X4: usize = {
    let c = (usize::MAX / 4) + 1;
    c + (c & 1)
  };

  /// Input-side `width x 2` overflow: `packed_yuv422_row_bytes` panics
  /// before the output guard is reached. Exercises the input path that is
  /// unique to legacy RGB (2 bytes/pixel) versus 3- or 4-channel outputs.
  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn rgb565_to_rgb_rejects_src_width_times_2_overflow() {
    let src: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    rgb565_to_rgb_row(&src, &mut rgb, OVERFLOW_WIDTH_X2, false);
  }

  /// Output-side `width x 3` overflow: the input fits (OVERFLOW_WIDTH_X3 <
  /// OVERFLOW_WIDTH_X2 on 32-bit), but `rgb_row_bytes` panics on the
  /// output-buffer calculation.
  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn rgb565_to_rgb_rejects_out_width_times_3_overflow() {
    let src: [u8; 0] = [];
    let mut rgb: [u8; 0] = [];
    rgb565_to_rgb_row(&src, &mut rgb, OVERFLOW_WIDTH_X3, false);
  }

  /// Output-side `width x 4` overflow via rgba output: `rgba_row_bytes` panics.
  /// OVERFLOW_WIDTH_X4 < OVERFLOW_WIDTH_X2 on 32-bit, so input guard passes.
  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn rgb565_to_rgba_rejects_out_width_times_4_overflow() {
    let src: [u8; 0] = [];
    let mut rgba: [u8; 0] = [];
    rgb565_to_rgba_row(&src, &mut rgba, OVERFLOW_WIDTH_X4, false);
  }

  /// u16 output-side `width x 3` element overflow: `rgb_row_elems` panics.
  /// Uses `bgr444_to_rgb_u16_row` to cover a different format and the u16 path.
  #[cfg(target_pointer_width = "32")]
  #[test]
  #[should_panic(expected = "overflows usize")]
  fn bgr444_to_rgb_u16_rejects_out_width_times_3_overflow() {
    let src: [u8; 0] = [];
    let mut rgb: [u16; 0] = [];
    bgr444_to_rgb_u16_row(&src, &mut rgb, OVERFLOW_WIDTH_X3, false);
  }
}

// SIMD-vs-scalar parity for the legacy bit-packed (8bpp + 4bpp) kernels.
#[cfg(all(test, feature = "std"))]
mod lowbit_simd_parity {
  use super::*;

  /// Deterministic pseudo-random byte plane of `len` bytes.
  fn rand_bytes(len: usize, seed: u32) -> std::vec::Vec<u8> {
    let mut state = seed;
    (0..len)
      .map(|_| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as u8
      })
      .collect()
  }

  /// For each width in `0..=40` plus a wide row, the SIMD (`use_simd = true`)
  /// and scalar (`use_simd = false`) outputs of every dispatcher must be
  /// byte-identical — exercising the vector body and the scalar tail. `$src_per`
  /// is the source bytes per pixel-equivalent (`1` byte-formats, or the
  /// nibble closure for the 4-bpp formats).
  macro_rules! parity {
    ($to_rgb:ident, $to_rgba:ident, $to_rgb_u16:ident, $to_rgba_u16:ident, $src_bytes:expr) => {{
      for width in (0..=40usize).chain([127, 256, 257]) {
        let src_bytes: usize = $src_bytes(width);
        let src = rand_bytes(src_bytes.max(1), 0x51A7_0001 ^ (width as u32));

        let mut a = std::vec![0u8; width * 3];
        let mut b = std::vec![0u8; width * 3];
        $to_rgb(&src, &mut a, width, true);
        $to_rgb(&src, &mut b, width, false);
        assert_eq!(a, b, "{} simd/scalar diverge @w={width}", stringify!($to_rgb));

        let mut a = std::vec![0u8; width * 4];
        let mut b = std::vec![0u8; width * 4];
        $to_rgba(&src, &mut a, width, true);
        $to_rgba(&src, &mut b, width, false);
        assert_eq!(a, b, "{} simd/scalar diverge @w={width}", stringify!($to_rgba));

        let mut a = std::vec![0u16; width * 3];
        let mut b = std::vec![0u16; width * 3];
        $to_rgb_u16(&src, &mut a, width, true);
        $to_rgb_u16(&src, &mut b, width, false);
        assert_eq!(a, b, "{} simd/scalar diverge @w={width}", stringify!($to_rgb_u16));

        let mut a = std::vec![0u16; width * 4];
        let mut b = std::vec![0u16; width * 4];
        $to_rgba_u16(&src, &mut a, width, true);
        $to_rgba_u16(&src, &mut b, width, false);
        assert_eq!(a, b, "{} simd/scalar diverge @w={width}", stringify!($to_rgba_u16));
      }
    }};
  }

  fn byte_bytes(w: usize) -> usize {
    w
  }
  fn nibble_bytes(w: usize) -> usize {
    w.div_ceil(2)
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn rgb8_simd_matches_scalar() {
    parity!(
      rgb8_to_rgb_row,
      rgb8_to_rgba_row,
      rgb8_to_rgb_u16_row,
      rgb8_to_rgba_u16_row,
      byte_bytes
    );
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn bgr8_simd_matches_scalar() {
    parity!(
      bgr8_to_rgb_row,
      bgr8_to_rgba_row,
      bgr8_to_rgb_u16_row,
      bgr8_to_rgba_u16_row,
      byte_bytes
    );
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn rgb4_byte_simd_matches_scalar() {
    parity!(
      rgb4_byte_to_rgb_row,
      rgb4_byte_to_rgba_row,
      rgb4_byte_to_rgb_u16_row,
      rgb4_byte_to_rgba_u16_row,
      byte_bytes
    );
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn bgr4_byte_simd_matches_scalar() {
    parity!(
      bgr4_byte_to_rgb_row,
      bgr4_byte_to_rgba_row,
      bgr4_byte_to_rgb_u16_row,
      bgr4_byte_to_rgba_u16_row,
      byte_bytes
    );
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn rgb4_simd_matches_scalar() {
    parity!(
      rgb4_to_rgb_row,
      rgb4_to_rgba_row,
      rgb4_to_rgb_u16_row,
      rgb4_to_rgba_u16_row,
      nibble_bytes
    );
  }

  #[test]
  #[cfg_attr(miri, ignore = "SIMD intrinsics unsupported by Miri")]
  fn bgr4_simd_matches_scalar() {
    parity!(
      bgr4_to_rgb_row,
      bgr4_to_rgba_row,
      bgr4_to_rgb_u16_row,
      bgr4_to_rgba_u16_row,
      nibble_bytes
    );
  }
}
