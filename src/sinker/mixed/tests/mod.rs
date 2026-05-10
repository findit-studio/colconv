use super::*;
use crate::{ColorMatrix, frame::*, raw::*, source::*};

// Per-format-family submodules. Each houses tests + format-local
// helpers (`solid_*_frame` builders); cross-cutting helpers
// (`pseudo_random_u8`, `pseudo_random_u16_low_n_bits`) live at
// module scope below and are re-exported as `pub(super)` so the
// submodules can pull them via `use super::*;`.
mod ayuv64;
mod bayer;
mod gray;
mod legacy_rgb;
mod mono1bit;
mod packed_rgb_10bit;
mod packed_rgb_16bit;
mod packed_rgb_8bit;
mod packed_rgb_f16;
mod packed_rgb_float;
mod packed_yuv_4_1_1;
mod packed_yuv_8bit;
mod pal8;
mod phase4_yuv_hb_be_roundtrip;
mod planar_gbr;
mod planar_gbr_float;
mod planar_gbr_high_bit;
mod planar_other_8bit_9bit;
mod semi_planar_8bit;
mod subsampled_4_2_0_high_bit;
mod subsampled_high_bit_pn;
mod v210;
mod v30x;
mod v410;
mod vuya;
mod vuyx;
mod xv36;
mod xyz12;
mod y210;
mod y212;
mod y216;
mod yuv410p_8bit;
mod yuv411p_8bit;
mod yuv420p_8bit;
mod yuva;

pub(super) fn pseudo_random_u16_low_n_bits(buf: &mut [u16], seed: u32, bits: u32) {
  let mask = ((1u32 << bits) - 1) as u16;
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = ((state >> 8) as u16) & mask;
  }
}

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
// `frame/tests/` fixture builders (PR #95/#96) and keeps the call sites
// in xv36/v410/ayuv64 sinker tests self-documenting.

/// Encode a logical `u16` as host-independent **LE-wire** byte storage.
#[cfg(feature = "std")]
#[inline]
pub(super) fn as_le_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_le_bytes())
}

/// Encode a logical `u16` as host-independent **BE-wire** byte storage.
#[cfg(feature = "std")]
#[inline]
pub(super) fn as_be_u16(v: u16) -> u16 {
  u16::from_ne_bytes(v.to_be_bytes())
}

/// Encode a logical `u32` as host-independent **LE-wire** byte storage.
#[cfg(feature = "std")]
#[inline]
pub(super) fn as_le_u32(v: u32) -> u32 {
  u32::from_ne_bytes(v.to_le_bytes())
}

/// Encode a logical `u32` as host-independent **BE-wire** byte storage.
#[cfg(feature = "std")]
#[inline]
pub(super) fn as_be_u32(v: u32) -> u32 {
  u32::from_ne_bytes(v.to_be_bytes())
}
