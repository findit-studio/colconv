//! Dispatch layer for Monoblack and Monowhite kernels.
//! No SIMD backends (scalar-only).

use crate::row::scalar::mono1bit as scalar;

// ---- Monoblack dispatch
// -------------------------------------------------------

pub(crate) fn monoblack_to_rgb_or_rgba_row<const ALPHA: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
  _use_simd: bool, // unused; scalar only
) {
  if ALPHA {
    scalar::monoblack_to_rgba_row(data, out, width);
  } else {
    scalar::monoblack_to_rgb_row(data, out, width);
  }
}

pub(crate) fn monoblack_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
  _use_simd: bool,
) {
  if ALPHA {
    scalar::monoblack_to_rgba_u16_row(data, out, width);
  } else {
    scalar::monoblack_to_rgb_u16_row(data, out, width);
  }
}

pub(crate) fn monoblack_to_luma_row(data: &[u8], out: &mut [u8], width: usize, _use_simd: bool) {
  scalar::monoblack_to_luma_row(data, out, width);
}

pub(crate) fn monoblack_to_luma_u16_row(
  data: &[u8],
  out: &mut [u16],
  width: usize,
  _use_simd: bool,
) {
  scalar::monoblack_to_luma_u16_row(data, out, width);
}

pub(crate) fn monoblack_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
  _use_simd: bool,
) {
  scalar::monoblack_to_hsv_row(data, h, s, v, width);
}

// ---- Monowhite dispatch
// ------------------------------------------------

pub(crate) fn monowhite_to_rgb_or_rgba_row<const ALPHA: bool>(
  data: &[u8],
  out: &mut [u8],
  width: usize,
  _use_simd: bool,
) {
  if ALPHA {
    scalar::monowhite_to_rgba_row(data, out, width);
  } else {
    scalar::monowhite_to_rgb_row(data, out, width);
  }
}

pub(crate) fn monowhite_to_rgb_u16_or_rgba_u16_row<const ALPHA: bool>(
  data: &[u8],
  out: &mut [u16],
  width: usize,
  _use_simd: bool,
) {
  if ALPHA {
    scalar::monowhite_to_rgba_u16_row(data, out, width);
  } else {
    scalar::monowhite_to_rgb_u16_row(data, out, width);
  }
}

pub(crate) fn monowhite_to_luma_row(data: &[u8], out: &mut [u8], width: usize, _use_simd: bool) {
  scalar::monowhite_to_luma_row(data, out, width);
}

pub(crate) fn monowhite_to_luma_u16_row(
  data: &[u8],
  out: &mut [u16],
  width: usize,
  _use_simd: bool,
) {
  scalar::monowhite_to_luma_u16_row(data, out, width);
}

pub(crate) fn monowhite_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
  _use_simd: bool,
) {
  scalar::monowhite_to_hsv_row(data, h, s, v, width);
}
