#[cfg(any(
  feature = "bayer",
  feature = "gbr",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
use super::*;
#[cfg(any(
  feature = "gbr",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
use crate::ColorMatrix;
// `frame::*` glob is consumed only by the families whose test files
// reference unqualified `*Frame` types; the Bayer tests import their
// frame types locally, so `bayer` is not in this set.
#[cfg(any(
  feature = "gbr",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
use crate::frame::*;
// `source::*` carries the source markers (`Bayer` / `Bayer16` / …) the
// test files name unqualified — Bayer included.
#[cfg(any(
  feature = "bayer",
  feature = "gbr",
  feature = "mono",
  feature = "rgb",
  feature = "rgb-float",
  feature = "rgb-legacy",
  feature = "v210",
  feature = "xyz",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
use crate::source::*;

// Per-format-family submodules. Each houses tests + format-local
// helpers (`solid_*_frame` builders); cross-cutting helpers
// (`pseudo_random_u8`, `pseudo_random_u16_low_n_bits`) live at
// module scope below and are re-exported as `pub(super)` so the
// submodules can pull them via `use super::*;`.
#[cfg(feature = "yuv-444-packed")]
mod ayuv64;
#[cfg(feature = "bayer")]
mod bayer;
#[cfg(feature = "gray")]
mod gray;
#[cfg(feature = "rgb-legacy")]
mod legacy_rgb;
#[cfg(feature = "mono")]
mod mono1bit;
#[cfg(feature = "rgb")]
mod packed_rgb_10bit;
#[cfg(feature = "rgb")]
mod packed_rgb_16bit;
#[cfg(feature = "rgb")]
mod packed_rgb_8bit;
#[cfg(feature = "rgb-float")]
mod packed_rgb_f16;
#[cfg(feature = "rgb-float")]
mod packed_rgb_float;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_4_1_1;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_8bit;
#[cfg(feature = "mono")]
mod pal8;
#[cfg(feature = "yuv-planar")]
mod phase4_yuv_hb_be_roundtrip;
#[cfg(feature = "gbr")]
mod planar_gbr;
#[cfg(feature = "gbr")]
mod planar_gbr_float;
#[cfg(feature = "gbr")]
mod planar_gbr_high_bit;
#[cfg(feature = "yuv-planar")]
mod planar_other_8bit_9bit;
#[cfg(feature = "yuv-444-packed")]
mod resample_ayuv64;
#[cfg(feature = "bayer")]
mod resample_bayer8;
#[cfg(feature = "rgb")]
mod resample_bgr24;
#[cfg(feature = "gbr")]
mod resample_gbr_high_bit;
#[cfg(feature = "gbr")]
mod resample_gbrap_8bit;
#[cfg(feature = "gbr")]
mod resample_gbrap_high_bit;
#[cfg(feature = "gbr")]
mod resample_gbrapf16;
#[cfg(feature = "gbr")]
mod resample_gbrapf16_filter;
#[cfg(feature = "gbr")]
mod resample_gbrapf32;
#[cfg(feature = "gbr")]
mod resample_gbrapf32_filter;
#[cfg(feature = "gbr")]
mod resample_gbrp;
#[cfg(feature = "gbr")]
mod resample_gbrpf16;
#[cfg(feature = "gbr")]
mod resample_gbrpf16_filter;
#[cfg(feature = "gbr")]
mod resample_gbrpf32;
#[cfg(feature = "gbr")]
mod resample_gbrpf32_filter;
#[cfg(feature = "yuv-planar")]
mod resample_geometry;
#[cfg(feature = "gray")]
mod resample_gray16;
#[cfg(feature = "gray")]
mod resample_gray8;
#[cfg(feature = "gray")]
mod resample_gray_n;
#[cfg(feature = "gray")]
mod resample_grayf32;
#[cfg(feature = "rgb-legacy")]
mod resample_legacy_rgb;
#[cfg(feature = "mono")]
mod resample_mono;
// The high-bit semi-planar P-format sinks live under the
// `yuv-planar`-gated `subsampled_4_*_high_bit` parent modules with a
// `yuv-semi-planar`-gated inner module, so they require BOTH features.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod resample_p0xx_high_bit;
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod resample_p0xx_high_bit_native;
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod resample_p2xx_high_bit;
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod resample_p4xx_high_bit;
#[cfg(feature = "rgb")]
mod resample_packed_rgb_10bit;
#[cfg(feature = "rgb")]
mod resample_packed_rgba_16bit;
#[cfg(feature = "rgb")]
mod resample_packed_rgba_8bit;
#[cfg(feature = "rgb")]
mod resample_packed_rgba_u16;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_packed_rgbf16;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_packed_rgbf32;
#[cfg(feature = "yuv-packed")]
mod resample_packed_yuv_8bit;
#[cfg(feature = "yuv-packed")]
mod resample_packed_yuv_8bit_filter;
#[cfg(feature = "rgb")]
mod resample_padding_byte;
#[cfg(feature = "mono")]
mod resample_pal8;
#[cfg(feature = "mono")]
mod resample_pal8_filter;
#[cfg(feature = "rgb")]
mod resample_rgb24;
#[cfg(feature = "rgb")]
mod resample_rgb48;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_rgbf16;
#[cfg(all(feature = "rgb-float", any(feature = "yuv-planar", feature = "rgb")))]
mod resample_rgbf32;
#[cfg(all(feature = "yuv-semi-planar", feature = "rgb"))]
mod resample_semi_planar;
#[cfg(feature = "yuv-semi-planar")]
mod resample_semi_planar_8bit_filter;
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod resample_subsampled_high_bit_p_filter;
#[cfg(feature = "yuv-packed")]
mod resample_uyyvyy411;
#[cfg(feature = "v210")]
mod resample_v210;
#[cfg(feature = "yuv-444-packed")]
mod resample_v30x;
#[cfg(feature = "yuv-444-packed")]
mod resample_v410;
#[cfg(feature = "yuv-444-packed")]
mod resample_v410_v30x_filter;
#[cfg(feature = "yuv-444-packed")]
mod resample_vuya;
#[cfg(feature = "yuv-444-packed")]
mod resample_vuya_vuyx_filter;
#[cfg(feature = "yuv-444-packed")]
mod resample_vuyx;
#[cfg(feature = "yuv-444-packed")]
mod resample_xv36;
#[cfg(feature = "xyz")]
mod resample_xyz12;
#[cfg(feature = "xyz")]
mod resample_xyz12_filter;
#[cfg(feature = "y2xx")]
mod resample_y2xx;
#[cfg(feature = "y2xx")]
mod resample_y2xx_filter;
#[cfg(feature = "gray")]
mod resample_ya16;
#[cfg(feature = "gray")]
mod resample_ya8;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_yuv410_440p;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_yuv411p;
#[cfg(feature = "yuv-planar")]
mod resample_yuv420p_high_bit;
#[cfg(feature = "yuv-planar")]
mod resample_yuv420p_high_bit_native;
#[cfg(all(feature = "yuv-planar", feature = "rgb"))]
mod resample_yuv422_444p;
#[cfg(feature = "yuv-planar")]
mod resample_yuv422p_high_bit;
#[cfg(feature = "yuv-planar")]
mod resample_yuv440p_high_bit;
#[cfg(feature = "yuv-planar")]
mod resample_yuv444p_high_bit;
#[cfg(feature = "yuv-planar")]
mod resample_yuv_planar_8bit_filter;
#[cfg(feature = "yuv-planar")]
mod resample_yuv_planar_high_bit_filter;
#[cfg(feature = "yuva")]
mod resample_yuva420p;
#[cfg(feature = "yuva")]
mod resample_yuva420p_high_bit;
#[cfg(feature = "yuva")]
mod resample_yuva422p;
#[cfg(feature = "yuva")]
mod resample_yuva422p_high_bit;
#[cfg(feature = "yuva")]
mod resample_yuva444p;
#[cfg(feature = "yuva")]
mod resample_yuva444p_high_bit;
#[cfg(feature = "yuva")]
mod resample_yuva_planar_8bit_filter;
#[cfg(feature = "yuv-semi-planar")]
mod semi_planar_8bit;
#[cfg(feature = "yuv-planar")]
mod subsampled_4_2_0_high_bit;
// Exercises the high-bit semi-planar P-format `MixedSinker` impls, which
// build on the planar high-bit sink infrastructure, and cross-checks them
// against the planar oracles — so it needs both families compiled.
#[cfg(all(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod subsampled_high_bit_pn;
#[cfg(feature = "v210")]
mod v210;
#[cfg(feature = "yuv-444-packed")]
mod v30x;
#[cfg(feature = "yuv-444-packed")]
mod v410;
#[cfg(feature = "yuv-444-packed")]
mod vuya;
#[cfg(feature = "yuv-444-packed")]
mod vuyx;
#[cfg(feature = "yuv-444-packed")]
mod xv36;
#[cfg(feature = "xyz")]
mod xyz12;
#[cfg(feature = "y2xx")]
mod y210;
#[cfg(feature = "y2xx")]
mod y212;
#[cfg(feature = "y2xx")]
mod y216;
#[cfg(feature = "yuv-planar")]
mod yuv410p_8bit;
#[cfg(feature = "yuv-planar")]
mod yuv411p_8bit;
#[cfg(feature = "yuv-planar")]
mod yuv420p_8bit;
#[cfg(feature = "yuva")]
mod yuva;

// `v210` is intentionally absent: V210 only consumes this helper inside
// its `yuv-planar`-gated planar-parity oracles (in `v210.rs` /
// `resample_v210.rs`), so the `yuv-planar` arm already covers it and a
// `v210`-solo `--tests` build would otherwise see it as dead.
#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "y2xx",
  feature = "yuv-444-packed",
  feature = "yuv-planar",
  feature = "yuva",
))]
pub(super) fn pseudo_random_u16_low_n_bits(buf: &mut [u16], seed: u32, bits: u32) {
  let mask = ((1u32 << bits) - 1) as u16;
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) as u16) & mask;
  }
}

#[cfg(any(
  feature = "gbr",
  feature = "gray",
  feature = "rgb",
  feature = "v210",
  feature = "yuv-444-packed",
  feature = "yuv-packed",
  feature = "yuv-planar",
  feature = "yuv-semi-planar",
  feature = "yuva",
))]
pub(super) fn pseudo_random_u8(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 16) as u8;
  }
}

// Host-independent **wire-byte** encoders for `&[u16]` / `&[u32]` test
// fixtures. Frames carry bytes, not numbers — so a sinker test that
// wants to feed an LE-wire `0x1234` u16 needs the underlying byte view
// to be `[0x34, 0x12]` regardless of host endianness; ditto BE wants
// `[0x12, 0x34]`. The pattern `T::from_ne_bytes(v.to_{le,be}_bytes())`
// achieves exactly that: on LE hosts `as_le_*` is identity and `as_be_*`
// byte-swaps; on BE hosts (e.g. s390x) the polarity flips. Centralising
// these here matches the `le_encoded_u16_buf` convention from the
// `frame/tests/` fixture builders and keeps the call sites in
// xv36/v410/ayuv64 sinker tests self-documenting.

/// Encode a logical `u16` as host-independent **LE-wire** byte storage.
#[cfg(all(
  feature = "std",
  any(feature = "gbr", feature = "rgb", feature = "yuv-444-packed")
))]
#[inline]
pub(super) fn as_le_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

/// Encode a logical `u16` as host-independent **BE-wire** byte storage.
#[cfg(all(
  feature = "std",
  any(feature = "gbr", feature = "rgb", feature = "yuv-444-packed")
))]
#[inline]
pub(super) fn as_be_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_be_bytes())
}

/// Encode a logical `u32` as host-independent **LE-wire** byte storage.
#[cfg(all(feature = "std", feature = "yuv-444-packed"))]
#[inline]
pub(super) fn as_le_u32(v: u32) -> u32 {
  u32::from_ne_bytes(v.to_le_bytes())
}

/// Encode a logical `u32` as host-independent **BE-wire** byte storage.
#[cfg(all(feature = "std", feature = "yuv-444-packed"))]
#[inline]
pub(super) fn as_be_u32(v: u32) -> u32 {
  u32::from_ne_bytes(v.to_be_bytes())
}
