//! SIMD vs scalar equivalence tests for NEON packed 32-bit RGB kernels
//! (Rgb96).
//!
//! Each test uses `width = 19` — for the u8 paths that is two 8-pixel SIMD
//! iterations plus a 3-pixel scalar tail; for the u16 paths (4 px/iter) it is
//! four SIMD iterations plus a 3-pixel tail. Both `BE = false` and `BE = true`
//! are exercised so the per-u32-lane byte-swap path is covered. Gated on
//! `target_arch = "aarch64"` and ignored under Miri.

use super::*;

/// Build a `width`-pixel Rgb96 row with a pseudo-random u32 pattern.
fn make_rgb96_src(width: usize, seed: u32) -> std::vec::Vec<u32> {
  (0..width * 3)
    .map(|i| (i as u32).wrapping_mul(seed).wrapping_add(0x1357_9BDF))
    .collect()
}

macro_rules! neon_rgb96_parity {
  ($name:ident, $simd:ident, $scalar:ident, $out_ty:ty, $out_pp:expr, $be:literal, $seed:expr) => {
    #[cfg(target_arch = "aarch64")]
    #[cfg_attr(miri, ignore = "NEON intrinsics not supported under Miri")]
    #[test]
    fn $name() {
      let width = 19;
      let src = make_rgb96_src(width, $seed);
      let mut simd_out = std::vec![0 as $out_ty; width * $out_pp];
      let mut scalar_out = std::vec![0 as $out_ty; width * $out_pp];
      unsafe { $simd::<$be>(&src, &mut simd_out, width) };
      scalar::$scalar::<$be>(&src, &mut scalar_out, width);
      assert_eq!(simd_out, scalar_out, "SIMD vs scalar mismatch");
    }
  };
}

neon_rgb96_parity!(
  neon_rgb96_to_rgb_le,
  neon_rgb96_to_rgb_row,
  rgb96_to_rgb_row,
  u8,
  3,
  false,
  0x0101_0101
);
neon_rgb96_parity!(
  neon_rgb96_to_rgb_be,
  neon_rgb96_to_rgb_row,
  rgb96_to_rgb_row,
  u8,
  3,
  true,
  0x0202_0202
);
neon_rgb96_parity!(
  neon_rgb96_to_rgba_le,
  neon_rgb96_to_rgba_row,
  rgb96_to_rgba_row,
  u8,
  4,
  false,
  0x0303_0303
);
neon_rgb96_parity!(
  neon_rgb96_to_rgba_be,
  neon_rgb96_to_rgba_row,
  rgb96_to_rgba_row,
  u8,
  4,
  true,
  0x0404_0404
);
neon_rgb96_parity!(
  neon_rgb96_to_rgb_u16_le,
  neon_rgb96_to_rgb_u16_row,
  rgb96_to_rgb_u16_row,
  u16,
  3,
  false,
  0x0505_0505
);
neon_rgb96_parity!(
  neon_rgb96_to_rgb_u16_be,
  neon_rgb96_to_rgb_u16_row,
  rgb96_to_rgb_u16_row,
  u16,
  3,
  true,
  0x0606_0606
);
neon_rgb96_parity!(
  neon_rgb96_to_rgba_u16_le,
  neon_rgb96_to_rgba_u16_row,
  rgb96_to_rgba_u16_row,
  u16,
  4,
  false,
  0x0707_0707
);
neon_rgb96_parity!(
  neon_rgb96_to_rgba_u16_be,
  neon_rgb96_to_rgba_u16_row,
  rgb96_to_rgba_u16_row,
  u16,
  4,
  true,
  0x0808_0808
);
