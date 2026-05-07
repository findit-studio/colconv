//! Dispatcher for `AV_PIX_FMT_PAL8` row kernels.
//!
//! On aarch64 targets, selects the NEON backend when NEON is available
//! at runtime (std) or compile-time (no-std). The NEON backend uses a
//! hybrid strategy: scalar palette gather + NEON deinterleave/store, which
//! delivers measurable throughput gains over pure scalar at production widths.
//!
//! On all other targets the scalar reference kernel is used. The `_use_simd`
//! parameter is accepted for API consistency; on non-NEON targets it is a
//! no-op.

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
/// On aarch64 with NEON, delegates to the NEON hybrid backend (scalar gather +
/// NEON deinterleave/store). On all other targets, always uses the scalar kernel
/// regardless of `_use_simd`.
pub fn pal8_to_rgb_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgb_out: &mut [u8],
  _use_simd: bool,
) {
  #[cfg(target_arch = "aarch64")]
  {
    use crate::row::neon_available;
    if _use_simd && neon_available() {
      // SAFETY: neon_available() guarantees NEON is present; slice lengths
      // are the caller's responsibility (same contract as scalar).
      unsafe { crate::row::arch::neon::pal8::pal8_to_rgb_row(indices, palette, rgb_out) }
      return;
    }
  }
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
/// On aarch64 with NEON, delegates to the NEON hybrid backend. On all other
/// targets, always uses the scalar kernel regardless of `_use_simd`.
pub fn pal8_to_rgba_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgba_out: &mut [u8],
  _use_simd: bool,
) {
  #[cfg(target_arch = "aarch64")]
  {
    use crate::row::neon_available;
    if _use_simd && neon_available() {
      unsafe { crate::row::arch::neon::pal8::pal8_to_rgba_row(indices, palette, rgba_out) }
      return;
    }
  }
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
/// On aarch64 with NEON, delegates to the NEON hybrid backend. On all other
/// targets, always uses the scalar kernel regardless of `_use_simd`.
pub fn pal8_to_rgb_u16_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgb_u16_out: &mut [u16],
  _use_simd: bool,
) {
  #[cfg(target_arch = "aarch64")]
  {
    use crate::row::neon_available;
    if _use_simd && neon_available() {
      unsafe { crate::row::arch::neon::pal8::pal8_to_rgb_u16_row(indices, palette, rgb_u16_out) }
      return;
    }
  }
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
/// On aarch64 with NEON, delegates to the NEON hybrid backend. On all other
/// targets, always uses the scalar kernel regardless of `_use_simd`.
pub fn pal8_to_rgba_u16_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgba_u16_out: &mut [u16],
  _use_simd: bool,
) {
  #[cfg(target_arch = "aarch64")]
  {
    use crate::row::neon_available;
    if _use_simd && neon_available() {
      unsafe { crate::row::arch::neon::pal8::pal8_to_rgba_u16_row(indices, palette, rgba_u16_out) }
      return;
    }
  }
  scalar_rgba_u16(indices, palette, rgba_u16_out);
}
