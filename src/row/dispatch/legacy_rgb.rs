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

// ============================================================================
// RGB565 (R5 G6 B5 — bits [15:11] [10:5] [4:0])
// ============================================================================

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

// ============================================================================
// BGR565 (B5 G6 R5 — bits [15:11]=B5, [10:5]=G6, [4:0]=R5)
// ============================================================================

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

// ============================================================================
// RGB555 (1X R5 G5 B5 — bits [14:10]=R5, [9:5]=G5, [4:0]=B5, bit 15 ignored)
// ============================================================================

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

// ============================================================================
// BGR555 (1X B5 G5 R5 — bits [14:10]=B5, [9:5]=G5, [4:0]=R5, bit 15 ignored)
// ============================================================================

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

// ============================================================================
// RGB444 (4X R4 G4 B4 — bits [11:8]=R4, [7:4]=G4, [3:0]=B4, bits [15:12] ignored)
// ============================================================================

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

// ============================================================================
// BGR444 (4X B4 G4 R4 — bits [11:8]=B4, [7:4]=G4, [3:0]=R4, bits [15:12] ignored)
// ============================================================================

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

// ============================================================================
// Overflow-guard tests — 32-bit target only
// ============================================================================

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
