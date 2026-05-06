//! Dispatcher for `AV_PIX_FMT_PAL8` scalar kernels.
//!
//! Palette gather is not SIMD-efficient on any supported ISA
//! (see spec § 4.4). The `use_simd` parameter is accepted for API
//! consistency but is always a no-op — every path calls the scalar
//! kernel.

use crate::row::scalar::pal8::{
  pal8_to_rgb_row as scalar_rgb, pal8_to_rgb_u16_row as scalar_rgb_u16,
  pal8_to_rgba_row as scalar_rgba, pal8_to_rgba_u16_row as scalar_rgba_u16,
};

/// Converts one row of `AV_PIX_FMT_PAL8` indices to packed `[R, G, B]` bytes.
///
/// Performs a palette lookup for each pixel index and packs the result as
/// `[R, G, B]` (palette entries are `[B, G, R, A]`; alpha is dropped).
///
/// `indices.len()` must equal the row's pixel width; `rgb_out.len()` must be
/// >= `3 * indices.len()`.
///
/// **`_use_simd` is a no-op.** Palette gather is not SIMD-efficient on any
/// supported ISA (NEON lacks gather, x86 gather has poor latency for 1 KB LUTs,
/// wasm has no gather). The scalar kernel is always called regardless of this flag.
pub fn pal8_to_rgb_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgb_out: &mut [u8],
  _use_simd: bool,
) {
  scalar_rgb(indices, palette, rgb_out);
}

/// Converts one row of `AV_PIX_FMT_PAL8` indices to packed `[R, G, B, A]` bytes.
///
/// Performs a palette lookup for each pixel index and packs the result as
/// `[R, G, B, A]` (palette entries are `[B, G, R, A]`; alpha is preserved).
///
/// `indices.len()` must equal the row's pixel width; `rgba_out.len()` must be
/// >= `4 * indices.len()`.
///
/// **`_use_simd` is a no-op** — see [`pal8_to_rgb_row`] for the rationale.
pub fn pal8_to_rgba_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgba_out: &mut [u8],
  _use_simd: bool,
) {
  scalar_rgba(indices, palette, rgba_out);
}

/// Converts one row of `AV_PIX_FMT_PAL8` indices to packed `[R, G, B]` u16.
///
/// Performs a palette lookup and widens each 8-bit channel to u16 via
/// `(x << 8) | x` (`0 → 0x0000`, `255 → 0xFFFF`). Alpha is dropped.
///
/// `indices.len()` must equal the row's pixel width; `rgb_u16_out.len()` must
/// be >= `3 * indices.len()`.
///
/// **`_use_simd` is a no-op** — see [`pal8_to_rgb_row`] for the rationale.
pub fn pal8_to_rgb_u16_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgb_u16_out: &mut [u16],
  _use_simd: bool,
) {
  scalar_rgb_u16(indices, palette, rgb_u16_out);
}

/// Converts one row of `AV_PIX_FMT_PAL8` indices to packed `[R, G, B, A]` u16.
///
/// Performs a palette lookup and widens each 8-bit channel (including alpha) to
/// u16 via `(x << 8) | x` (`0 → 0x0000`, `255 → 0xFFFF`). Alpha is preserved.
///
/// `indices.len()` must equal the row's pixel width; `rgba_u16_out.len()` must
/// be >= `4 * indices.len()`.
///
/// **`_use_simd` is a no-op** — see [`pal8_to_rgb_row`] for the rationale.
pub fn pal8_to_rgba_u16_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgba_u16_out: &mut [u16],
  _use_simd: bool,
) {
  scalar_rgba_u16(indices, palette, rgba_u16_out);
}
