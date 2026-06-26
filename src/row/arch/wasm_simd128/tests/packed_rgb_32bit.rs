//! wasm-simd128 vs scalar equivalence tests for packed 32-bit RGB kernels
//! (Rgb96).
//!
//! Compile-time gated via `#[cfg(target_feature = "simd128")]` — no runtime
//! guard needed. Covers `BE = false` and `BE = true` across a width sweep that
//! exercises the 8-pixel SIMD loop and the scalar tail.

use super::super::*;
use crate::row::scalar;

fn pseudo_random_u32(n: usize, seed: u64) -> std::vec::Vec<u32> {
  let mut v = std::vec::Vec::with_capacity(n);
  let mut s = seed;
  for _ in 0..n {
    s = s
      .wrapping_mul(6364136223846793005)
      .wrapping_add(1442695040888963407);
    v.push((s >> 32) as u32);
  }
  v
}

fn widths() -> &'static [usize] {
  &[1, 7, 8, 9, 15, 16, 17, 31, 32, 33]
}

macro_rules! wasm_rgb96_parity {
  ($name:ident, $simd:ident, $scalar:ident, $out_ty:ty, $out_pp:expr, $be:literal) => {
    #[cfg(target_feature = "simd128")]
    #[test]
    fn $name() {
      for &w in widths() {
        let src = pseudo_random_u32(w * 3, 0xDEAD_BEEF_1234_5678 ^ ($out_pp as u64) ^ ($be as u64));
        let mut scalar_out = std::vec![0 as $out_ty; w * $out_pp];
        let mut simd_out = std::vec![0 as $out_ty; w * $out_pp];
        scalar::$scalar::<$be>(&src, &mut scalar_out, w);
        unsafe { $simd::<$be>(&src, &mut simd_out, w) };
        assert_eq!(scalar_out, simd_out, "diverges (width={w})");
      }
    }
  };
}

wasm_rgb96_parity!(
  wasm_rgb96_to_rgb_le,
  wasm_rgb96_to_rgb_row,
  rgb96_to_rgb_row,
  u8,
  3,
  false
);
wasm_rgb96_parity!(
  wasm_rgb96_to_rgb_be,
  wasm_rgb96_to_rgb_row,
  rgb96_to_rgb_row,
  u8,
  3,
  true
);
wasm_rgb96_parity!(
  wasm_rgb96_to_rgba_le,
  wasm_rgb96_to_rgba_row,
  rgb96_to_rgba_row,
  u8,
  4,
  false
);
wasm_rgb96_parity!(
  wasm_rgb96_to_rgba_be,
  wasm_rgb96_to_rgba_row,
  rgb96_to_rgba_row,
  u8,
  4,
  true
);
wasm_rgb96_parity!(
  wasm_rgb96_to_rgb_u16_le,
  wasm_rgb96_to_rgb_u16_row,
  rgb96_to_rgb_u16_row,
  u16,
  3,
  false
);
wasm_rgb96_parity!(
  wasm_rgb96_to_rgb_u16_be,
  wasm_rgb96_to_rgb_u16_row,
  rgb96_to_rgb_u16_row,
  u16,
  3,
  true
);
wasm_rgb96_parity!(
  wasm_rgb96_to_rgba_u16_le,
  wasm_rgb96_to_rgba_u16_row,
  rgb96_to_rgba_u16_row,
  u16,
  4,
  false
);
wasm_rgb96_parity!(
  wasm_rgb96_to_rgba_u16_be,
  wasm_rgb96_to_rgba_u16_row,
  rgb96_to_rgba_u16_row,
  u16,
  4,
  true
);
