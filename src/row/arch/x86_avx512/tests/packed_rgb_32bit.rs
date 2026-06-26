//! AVX-512 vs scalar equivalence tests for packed 32-bit RGB kernels (Rgb96).
//!
//! Width 43 exercises the 32-pixel main loop, the 8-pixel cleanup loop, and a
//! 3-pixel scalar tail; both `BE = false` and `BE = true` are covered.
//! Runtime-gated on AVX-512F + AVX-512BW.

use super::super::*;
use crate::row::scalar;

fn make_rgb96_src(width: usize, seed: u32) -> std::vec::Vec<u32> {
  (0..width * 3)
    .map(|i| (i as u32).wrapping_mul(seed).wrapping_add(0x1357_9BDF))
    .collect()
}

macro_rules! avx512_rgb96_parity {
  ($name:ident, $simd:ident, $scalar:ident, $out_ty:ty, $out_pp:expr, $be:literal, $seed:expr) => {
    #[test]
    #[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
    fn $name() {
      if !(std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw"))
      {
        return;
      }
      let width = 43;
      let src = make_rgb96_src(width, $seed);
      let mut simd_out = std::vec![0 as $out_ty; width * $out_pp];
      let mut scalar_out = std::vec![0 as $out_ty; width * $out_pp];
      unsafe { $simd::<$be>(&src, &mut simd_out, width) };
      scalar::$scalar::<$be>(&src, &mut scalar_out, width);
      assert_eq!(simd_out, scalar_out, "SIMD vs scalar mismatch");
    }
  };
}

avx512_rgb96_parity!(
  avx512_rgb96_to_rgb_le,
  avx512_rgb96_to_rgb_row,
  rgb96_to_rgb_row,
  u8,
  3,
  false,
  0x0101_0101
);
avx512_rgb96_parity!(
  avx512_rgb96_to_rgb_be,
  avx512_rgb96_to_rgb_row,
  rgb96_to_rgb_row,
  u8,
  3,
  true,
  0x0202_0202
);
avx512_rgb96_parity!(
  avx512_rgb96_to_rgba_le,
  avx512_rgb96_to_rgba_row,
  rgb96_to_rgba_row,
  u8,
  4,
  false,
  0x0303_0303
);
avx512_rgb96_parity!(
  avx512_rgb96_to_rgba_be,
  avx512_rgb96_to_rgba_row,
  rgb96_to_rgba_row,
  u8,
  4,
  true,
  0x0404_0404
);
avx512_rgb96_parity!(
  avx512_rgb96_to_rgb_u16_le,
  avx512_rgb96_to_rgb_u16_row,
  rgb96_to_rgb_u16_row,
  u16,
  3,
  false,
  0x0505_0505
);
avx512_rgb96_parity!(
  avx512_rgb96_to_rgb_u16_be,
  avx512_rgb96_to_rgb_u16_row,
  rgb96_to_rgb_u16_row,
  u16,
  3,
  true,
  0x0606_0606
);
avx512_rgb96_parity!(
  avx512_rgb96_to_rgba_u16_le,
  avx512_rgb96_to_rgba_u16_row,
  rgb96_to_rgba_u16_row,
  u16,
  4,
  false,
  0x0707_0707
);
avx512_rgb96_parity!(
  avx512_rgb96_to_rgba_u16_be,
  avx512_rgb96_to_rgba_u16_row,
  rgb96_to_rgba_u16_row,
  u16,
  4,
  true,
  0x0808_0808
);

fn make_rgba128_src(width: usize, seed: u32) -> std::vec::Vec<u32> {
  (0..width * 4)
    .map(|i| (i as u32).wrapping_mul(seed).wrapping_add(0x2468_ACE0))
    .collect()
}

macro_rules! avx512_rgba128_parity {
  ($name:ident, $simd:ident, $scalar:ident, $out_ty:ty, $out_pp:expr, $be:literal, $seed:expr) => {
    #[test]
    #[cfg_attr(miri, ignore = "x86 SIMD intrinsics unsupported by Miri")]
    fn $name() {
      if !(std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512bw")) {
        return;
      }
      let width = 43;
      let src = make_rgba128_src(width, $seed);
      let mut simd_out = std::vec![0 as $out_ty; width * $out_pp];
      let mut scalar_out = std::vec![0 as $out_ty; width * $out_pp];
      unsafe { $simd::<$be>(&src, &mut simd_out, width) };
      scalar::$scalar::<$be>(&src, &mut scalar_out, width);
      assert_eq!(simd_out, scalar_out, "SIMD vs scalar mismatch");
    }
  };
}

avx512_rgba128_parity!(
  avx512_rgba128_to_rgb_le,
  avx512_rgba128_to_rgb_row,
  rgba128_to_rgb_row,
  u8,
  3,
  false,
  0x0909_0909
);
avx512_rgba128_parity!(
  avx512_rgba128_to_rgb_be,
  avx512_rgba128_to_rgb_row,
  rgba128_to_rgb_row,
  u8,
  3,
  true,
  0x0A0A_0A0A
);
avx512_rgba128_parity!(
  avx512_rgba128_to_rgba_le,
  avx512_rgba128_to_rgba_row,
  rgba128_to_rgba_row,
  u8,
  4,
  false,
  0x0B0B_0B0B
);
avx512_rgba128_parity!(
  avx512_rgba128_to_rgba_be,
  avx512_rgba128_to_rgba_row,
  rgba128_to_rgba_row,
  u8,
  4,
  true,
  0x0C0C_0C0C
);
avx512_rgba128_parity!(
  avx512_rgba128_to_rgb_u16_le,
  avx512_rgba128_to_rgb_u16_row,
  rgba128_to_rgb_u16_row,
  u16,
  3,
  false,
  0x0D0D_0D0D
);
avx512_rgba128_parity!(
  avx512_rgba128_to_rgb_u16_be,
  avx512_rgba128_to_rgb_u16_row,
  rgba128_to_rgb_u16_row,
  u16,
  3,
  true,
  0x0E0E_0E0E
);
avx512_rgba128_parity!(
  avx512_rgba128_to_rgba_u16_le,
  avx512_rgba128_to_rgba_u16_row,
  rgba128_to_rgba_u16_row,
  u16,
  4,
  false,
  0x0F0F_0F0F
);
avx512_rgba128_parity!(
  avx512_rgba128_to_rgba_u16_be,
  avx512_rgba128_to_rgba_u16_row,
  rgba128_to_rgba_u16_row,
  u16,
  4,
  true,
  0x1010_1010
);
