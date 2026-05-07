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

use crate::row::{
  rgb_row_bytes, rgb_row_elems, rgba_row_bytes, rgba_row_elems,
  scalar::pal8::{
    pal8_to_rgb_row as scalar_rgb, pal8_to_rgb_u16_row as scalar_rgb_u16,
    pal8_to_rgba_row as scalar_rgba, pal8_to_rgba_u16_row as scalar_rgba_u16,
  },
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
  let width = indices.len();
  // Hoist checked multiply first so 32-bit overflow surfaces as a panic here
  // rather than silently accepting an undersized buffer downstream.
  let out_min = rgb_row_bytes(width);
  assert!(indices.len() >= width, "indices too short");
  assert!(rgb_out.len() >= out_min, "rgb_out too short");

  #[cfg(target_arch = "aarch64")]
  {
    use crate::row::neon_available;
    if _use_simd && neon_available() {
      // SAFETY: neon_available() guarantees NEON is present. The dispatcher
      // has already asserted `indices.len() >= width` and
      // `rgb_out.len() >= rgb_row_bytes(width)` above (release-mode checks),
      // satisfying the unsafe fn's preconditions 2 and 3.
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
  let width = indices.len();
  let out_min = rgba_row_bytes(width);
  assert!(indices.len() >= width, "indices too short");
  assert!(rgba_out.len() >= out_min, "rgba_out too short");

  #[cfg(target_arch = "aarch64")]
  {
    use crate::row::neon_available;
    if _use_simd && neon_available() {
      // SAFETY: neon_available() guarantees NEON is present. The dispatcher
      // has already asserted `indices.len() >= width` and
      // `rgba_out.len() >= rgba_row_bytes(width)` above (release-mode checks),
      // satisfying the unsafe fn's preconditions 2 and 3.
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
  let width = indices.len();
  let out_min = rgb_row_elems(width);
  assert!(indices.len() >= width, "indices too short");
  assert!(rgb_u16_out.len() >= out_min, "rgb_u16_out too short");

  #[cfg(target_arch = "aarch64")]
  {
    use crate::row::neon_available;
    if _use_simd && neon_available() {
      // SAFETY: neon_available() guarantees NEON is present. The dispatcher
      // has already asserted `indices.len() >= width` and
      // `rgb_u16_out.len() >= rgb_row_elems(width)` above (release-mode checks),
      // satisfying the unsafe fn's preconditions 2 and 3.
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
  let width = indices.len();
  let out_min = rgba_row_elems(width);
  assert!(indices.len() >= width, "indices too short");
  assert!(rgba_u16_out.len() >= out_min, "rgba_u16_out too short");

  #[cfg(target_arch = "aarch64")]
  {
    use crate::row::neon_available;
    if _use_simd && neon_available() {
      // SAFETY: neon_available() guarantees NEON is present. The dispatcher
      // has already asserted `indices.len() >= width` and
      // `rgba_u16_out.len() >= rgba_row_elems(width)` above (release-mode checks),
      // satisfying the unsafe fn's preconditions 2 and 3.
      unsafe { crate::row::arch::neon::pal8::pal8_to_rgba_u16_row(indices, palette, rgba_u16_out) }
      return;
    }
  }
  scalar_rgba_u16(indices, palette, rgba_u16_out);
}

#[cfg(test)]
mod tests {
  use super::*;

  fn make_palette() -> [[u8; 4]; 256] {
    let mut p = [[0u8; 4]; 256];
    for (i, entry) in p.iter_mut().enumerate() {
      let i = i as u32;
      entry[0] = ((i.wrapping_mul(73) ^ 0xA5) & 0xFF) as u8;
      entry[1] = ((i.wrapping_mul(131) ^ 0x5A) & 0xFF) as u8;
      entry[2] = ((i.wrapping_mul(197) ^ 0x3C) & 0xFF) as u8;
      entry[3] = ((i.wrapping_mul(251) ^ 0xF0) & 0xFF) as u8;
    }
    p
  }

  // ---- undersized output buffer panics (release-mode safety regression) ------
  //
  // These tests verify that a safe caller passing an undersized output slice
  // gets a panic from the release-mode assert BEFORE any unsafe SIMD code
  // executes. The `use_simd = true` flag exercises the NEON branch on aarch64
  // and the scalar-fallback branch on all other targets — both must assert.

  #[test]
  #[should_panic(expected = "rgb_out too short")]
  fn pal8_to_rgb_row_undersized_out_panics() {
    let palette = make_palette();
    let indices = [0u8; 16];
    // 16 pixels need 48 bytes; provide only 47.
    let mut out = [0u8; 47];
    pal8_to_rgb_row(&indices, &palette, &mut out, true);
  }

  #[test]
  #[should_panic(expected = "rgba_out too short")]
  fn pal8_to_rgba_row_undersized_out_panics() {
    let palette = make_palette();
    let indices = [0u8; 16];
    // 16 pixels need 64 bytes; provide only 63.
    let mut out = [0u8; 63];
    pal8_to_rgba_row(&indices, &palette, &mut out, true);
  }

  #[test]
  #[should_panic(expected = "rgb_u16_out too short")]
  fn pal8_to_rgb_u16_row_undersized_out_panics() {
    let palette = make_palette();
    let indices = [0u8; 16];
    // 16 pixels need 48 u16 elements; provide only 47.
    let mut out = [0u16; 47];
    pal8_to_rgb_u16_row(&indices, &palette, &mut out, true);
  }

  #[test]
  #[should_panic(expected = "rgba_u16_out too short")]
  fn pal8_to_rgba_u16_row_undersized_out_panics() {
    let palette = make_palette();
    let indices = [0u8; 16];
    // 16 pixels need 64 u16 elements; provide only 63.
    let mut out = [0u16; 63];
    pal8_to_rgba_u16_row(&indices, &palette, &mut out, true);
  }

  // ---- exact-size buffers succeed (no panic) ---------------------------------

  #[test]
  fn pal8_to_rgb_row_exact_size_ok() {
    let palette = make_palette();
    let indices = [0u8; 16];
    let mut out = [0u8; 48];
    pal8_to_rgb_row(&indices, &palette, &mut out, true);
  }

  #[test]
  fn pal8_to_rgba_row_exact_size_ok() {
    let palette = make_palette();
    let indices = [0u8; 16];
    let mut out = [0u8; 64];
    pal8_to_rgba_row(&indices, &palette, &mut out, true);
  }

  #[test]
  fn pal8_to_rgb_u16_row_exact_size_ok() {
    let palette = make_palette();
    let indices = [0u8; 16];
    let mut out = [0u16; 48];
    pal8_to_rgb_u16_row(&indices, &palette, &mut out, true);
  }

  #[test]
  fn pal8_to_rgba_u16_row_exact_size_ok() {
    let palette = make_palette();
    let indices = [0u8; 16];
    let mut out = [0u16; 64];
    pal8_to_rgba_u16_row(&indices, &palette, &mut out, true);
  }
}
