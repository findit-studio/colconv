//! Scalar kernels for `AV_PIX_FMT_PAL8` — 8-bit indexed color.
//!
//! Each kernel performs one palette lookup per pixel. The palette
//! entries are in FFmpeg's `[B, G, R, A]` byte order; every kernel
//! reorders to `[R, G, B, A]` on output.
//!
//! # SIMD note
//! Palette gather is not efficient on any current SIMD ISA (see
//! the design spec § 4.4 for the full rationale). These scalar
//! kernels are the only implementation; the dispatcher treats
//! `use_simd = true` as equivalent to `false`.
//!
//! # u16 scaling
//! The formula `((x as u16) << 8) | (x as u16)` maps `u8` values to `u16`
//! full-range: `0 → 0x0000`, `255 → 0xFFFF`. See [`expand_u8_to_u16`].

/// Maps `u8` value to `u16` full-range: `(v as u16) << 8 | v as u16`.
/// Guarantees `0 → 0x0000`, `255 → 0xFFFF`.
#[inline(always)]
fn expand_u8_to_u16(v: u8) -> u16 {
  let v16 = v as u16;
  (v16 << 8) | v16
}

/// Palette lookup → packed `[R, G, B]` output for one row.
///
/// `indices.len()` must equal `width`; `rgb_out.len()` must be >= `3 * width`.
/// Palette entries are in FFmpeg `[B, G, R, A]` order; output reorders to
/// `[R, G, B]` (alpha dropped).
pub(crate) fn pal8_to_rgb_row(indices: &[u8], palette: &[[u8; 4]; 256], rgb_out: &mut [u8]) {
  let w = indices.len();
  debug_assert!(rgb_out.len() >= 3 * w);
  for x in 0..w {
    let [b, g, r, _a] = palette[indices[x] as usize];
    rgb_out[3 * x] = r;
    rgb_out[3 * x + 1] = g;
    rgb_out[3 * x + 2] = b;
  }
}

/// Palette lookup → packed `[R, G, B, A]` output for one row.
///
/// `indices.len()` must equal `width`; `rgba_out.len()` must be >= `4 * width`.
/// Palette entries are in FFmpeg `[B, G, R, A]` order; output reorders to
/// `[R, G, B, A]` (alpha preserved).
pub(crate) fn pal8_to_rgba_row(indices: &[u8], palette: &[[u8; 4]; 256], rgba_out: &mut [u8]) {
  let w = indices.len();
  debug_assert!(rgba_out.len() >= 4 * w);
  for x in 0..w {
    let [b, g, r, a] = palette[indices[x] as usize];
    rgba_out[4 * x] = r;
    rgba_out[4 * x + 1] = g;
    rgba_out[4 * x + 2] = b;
    rgba_out[4 * x + 3] = a;
  }
}

/// Palette lookup → `[R, G, B]` u16 (`(x << 8) | x` scaling) for one row.
///
/// `indices.len()` must equal `width`; `rgb_u16_out.len()` must be >= `3 * width`.
/// Palette entries are in FFmpeg `[B, G, R, A]` order; output reorders to
/// `[R, G, B]` u16 (alpha dropped). Each 8-bit channel is widened via
/// `(x << 8) | x`: `0 → 0x0000`, `255 → 0xFFFF`.
pub(crate) fn pal8_to_rgb_u16_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgb_u16_out: &mut [u16],
) {
  let w = indices.len();
  debug_assert!(rgb_u16_out.len() >= 3 * w);
  for x in 0..w {
    let [b, g, r, _a] = palette[indices[x] as usize];
    rgb_u16_out[3 * x] = expand_u8_to_u16(r);
    rgb_u16_out[3 * x + 1] = expand_u8_to_u16(g);
    rgb_u16_out[3 * x + 2] = expand_u8_to_u16(b);
  }
}

/// Palette lookup → `[R, G, B, A]` u16 for one row.
///
/// `indices.len()` must equal `width`; `rgba_u16_out.len()` must be >= `4 * width`.
/// Palette entries are in FFmpeg `[B, G, R, A]` order; output reorders to
/// `[R, G, B, A]` u16 (alpha preserved). Each 8-bit channel is widened via
/// `(x << 8) | x`: `0 → 0x0000`, `255 → 0xFFFF`.
pub(crate) fn pal8_to_rgba_u16_row(
  indices: &[u8],
  palette: &[[u8; 4]; 256],
  rgba_u16_out: &mut [u16],
) {
  let w = indices.len();
  debug_assert!(rgba_u16_out.len() >= 4 * w);
  for x in 0..w {
    let [b, g, r, a] = palette[indices[x] as usize];
    rgba_u16_out[4 * x] = expand_u8_to_u16(r);
    rgba_u16_out[4 * x + 1] = expand_u8_to_u16(g);
    rgba_u16_out[4 * x + 2] = expand_u8_to_u16(b);
    rgba_u16_out[4 * x + 3] = expand_u8_to_u16(a);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn make_palette() -> [[u8; 4]; 256] {
    let mut p = [[0u8; 4]; 256];
    // Entry 0: B=10, G=20, R=30, A=40
    p[0] = [10, 20, 30, 40];
    // Entry 1: B=50, G=100, R=200, A=255
    p[1] = [50, 100, 200, 255];
    p
  }

  #[test]
  fn rgb_row_reorders_bgra_to_rgb() {
    let palette = make_palette();
    let indices = [0u8, 1u8];
    let mut out = [0u8; 6];
    pal8_to_rgb_row(&indices, &palette, &mut out);
    // Entry 0: [B=10,G=20,R=30,A=40] → [R=30,G=20,B=10]
    assert_eq!(out[0..3], [30, 20, 10]);
    // Entry 1: [B=50,G=100,R=200,A=255] → [R=200,G=100,B=50]
    assert_eq!(out[3..6], [200, 100, 50]);
  }

  #[test]
  fn rgba_row_passes_alpha() {
    let palette = make_palette();
    let indices = [0u8];
    let mut out = [0u8; 4];
    pal8_to_rgba_row(&indices, &palette, &mut out);
    assert_eq!(out, [30, 20, 10, 40]); // [R, G, B, A]
  }

  #[test]
  fn rgb_u16_row_expands_full_range() {
    let mut palette = [[0u8; 4]; 256];
    palette[0] = [0, 0, 255, 255]; // B=0, G=0, R=255, A=255
    let indices = [0u8];
    let mut out = [0u16; 3];
    pal8_to_rgb_u16_row(&indices, &palette, &mut out);
    assert_eq!(out[0], 0xFFFF); // R=255 → 0xFFFF
    assert_eq!(out[1], 0x0000); // G=0   → 0x0000
    assert_eq!(out[2], 0x0000); // B=0   → 0x0000
  }

  #[test]
  fn rgba_u16_row_expands_alpha() {
    let mut palette = [[0u8; 4]; 256];
    palette[0] = [0, 0, 0, 128];
    let indices = [0u8];
    let mut out = [0u16; 4];
    pal8_to_rgba_u16_row(&indices, &palette, &mut out);
    assert_eq!(out[3], expand_u8_to_u16(128)); // A=128 → 0x8080
  }

  #[test]
  fn expand_u8_to_u16_boundary_values() {
    assert_eq!(expand_u8_to_u16(0), 0x0000);
    assert_eq!(expand_u8_to_u16(1), 0x0101);
    assert_eq!(expand_u8_to_u16(128), 0x8080);
    assert_eq!(expand_u8_to_u16(255), 0xFFFF);
  }
}
