//! Scalar kernels for 1-bit-per-pixel monochrome frames (Monoblack / Monowhite).
//!
//! Unpack logic: each byte carries 8 pixels, MSB first (bit 7 = pixel 0, bit 0 =
//! pixel 7). Polarity controlled by `const INVERT: bool`:
//! - INVERT=false (Monoblack): bit=0 → Y=0, bit=1 → Y=255
//! - INVERT=true (Monowhite): bit=0 → Y=255, bit=1 → Y=0

/// Unpack 1-bit-per-pixel buffer to u8 luma array.
/// MSB first: bit 7 of byte[0] is pixel 0, bit 0 of byte[0] is pixel 7, etc.
#[cfg_attr(not(tarpaulin), inline(always))]
fn mono1bit_to_luma_row_generic<const INVERT: bool>(data: &[u8], out: &mut [u8], width: usize) {
  let mut out_idx = 0;
  for byte_val in data.iter().take(width.div_ceil(8)) {
    for bit_pos in (0..8).rev() {
      if out_idx >= width {
        break;
      }
      let bit = (*byte_val >> bit_pos) & 1;
      let luma = if INVERT { (1 - bit) * 255 } else { bit * 255 };
      out[out_idx] = luma;
      out_idx += 1;
    }
  }
}

// ---- allocation-free pixel-expansion helpers --------------------------------

/// Extract the Y value for one pixel bit, applying polarity.
#[inline(always)]
fn pixel_y<const INVERT: bool>(byte: u8, bit: usize) -> u8 {
  let raw = (byte >> (7 - bit)) & 1;
  if INVERT { (1 - raw) * 255 } else { raw * 255 }
}

// ---- allocation-free RGB/RGBA/u16 generics ----------------------------------

#[cfg_attr(not(tarpaulin), inline(always))]
fn mono1bit_to_rgb_generic<const INVERT: bool>(data: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 3);
  let full_bytes = width / 8;
  for (byte_idx, &byte) in data[..full_bytes].iter().enumerate() {
    for bit in 0..8usize {
      let y = pixel_y::<INVERT>(byte, bit);
      let i = (byte_idx * 8 + bit) * 3;
      out[i] = y;
      out[i + 1] = y;
      out[i + 2] = y;
    }
  }
  let tail = width % 8;
  if tail > 0 {
    let byte = data[full_bytes];
    for bit in 0..tail {
      let y = pixel_y::<INVERT>(byte, bit);
      let i = (full_bytes * 8 + bit) * 3;
      out[i] = y;
      out[i + 1] = y;
      out[i + 2] = y;
    }
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn mono1bit_to_rgba_generic<const INVERT: bool>(data: &[u8], out: &mut [u8], width: usize) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 4);
  let full_bytes = width / 8;
  for (byte_idx, &byte) in data[..full_bytes].iter().enumerate() {
    for bit in 0..8usize {
      let y = pixel_y::<INVERT>(byte, bit);
      let i = (byte_idx * 8 + bit) * 4;
      out[i] = y;
      out[i + 1] = y;
      out[i + 2] = y;
      out[i + 3] = 0xFF;
    }
  }
  let tail = width % 8;
  if tail > 0 {
    let byte = data[full_bytes];
    for bit in 0..tail {
      let y = pixel_y::<INVERT>(byte, bit);
      let i = (full_bytes * 8 + bit) * 4;
      out[i] = y;
      out[i + 1] = y;
      out[i + 2] = y;
      out[i + 3] = 0xFF;
    }
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn mono1bit_to_rgb_u16_generic<const INVERT: bool>(data: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 3);
  let full_bytes = width / 8;
  for (byte_idx, &byte) in data[..full_bytes].iter().enumerate() {
    for bit in 0..8usize {
      let y8 = pixel_y::<INVERT>(byte, bit) as u16;
      let y16 = (y8 << 8) | y8;
      let i = (byte_idx * 8 + bit) * 3;
      out[i] = y16;
      out[i + 1] = y16;
      out[i + 2] = y16;
    }
  }
  let tail = width % 8;
  if tail > 0 {
    let byte = data[full_bytes];
    for bit in 0..tail {
      let y8 = pixel_y::<INVERT>(byte, bit) as u16;
      let y16 = (y8 << 8) | y8;
      let i = (full_bytes * 8 + bit) * 3;
      out[i] = y16;
      out[i + 1] = y16;
      out[i + 2] = y16;
    }
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn mono1bit_to_rgba_u16_generic<const INVERT: bool>(data: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width * 4);
  let full_bytes = width / 8;
  for (byte_idx, &byte) in data[..full_bytes].iter().enumerate() {
    for bit in 0..8usize {
      let y8 = pixel_y::<INVERT>(byte, bit) as u16;
      let y16 = (y8 << 8) | y8;
      let i = (byte_idx * 8 + bit) * 4;
      out[i] = y16;
      out[i + 1] = y16;
      out[i + 2] = y16;
      out[i + 3] = 0xFFFF;
    }
  }
  let tail = width % 8;
  if tail > 0 {
    let byte = data[full_bytes];
    for bit in 0..tail {
      let y8 = pixel_y::<INVERT>(byte, bit) as u16;
      let y16 = (y8 << 8) | y8;
      let i = (full_bytes * 8 + bit) * 4;
      out[i] = y16;
      out[i + 1] = y16;
      out[i + 2] = y16;
      out[i + 3] = 0xFFFF;
    }
  }
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn mono1bit_to_luma_u16_generic<const INVERT: bool>(data: &[u8], out: &mut [u16], width: usize) {
  debug_assert!(data.len() >= width.div_ceil(8));
  debug_assert!(out.len() >= width);
  let full_bytes = width / 8;
  for (byte_idx, &byte) in data[..full_bytes].iter().enumerate() {
    for bit in 0..8usize {
      let y8 = pixel_y::<INVERT>(byte, bit) as u16;
      out[byte_idx * 8 + bit] = (y8 << 8) | y8;
    }
  }
  let tail = width % 8;
  if tail > 0 {
    let byte = data[full_bytes];
    for bit in 0..tail {
      let y8 = pixel_y::<INVERT>(byte, bit) as u16;
      out[full_bytes * 8 + bit] = (y8 << 8) | y8;
    }
  }
}

// ---- Monoblack (INVERT=false) -----------------------------------------------

/// Monoblack → RGB u8 (broadcast Y).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  mono1bit_to_rgb_generic::<false>(data, out, width);
}

/// Monoblack → RGBA u8 (broadcast Y, α=0xFF).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  mono1bit_to_rgba_generic::<false>(data, out, width);
}

/// Monoblack → RGB u16 (broadcast Y, upshift (Y << 8) | Y).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  mono1bit_to_rgb_u16_generic::<false>(data, out, width);
}

/// Monoblack → RGBA u16 (broadcast Y, α=0xFFFF).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  mono1bit_to_rgba_u16_generic::<false>(data, out, width);
}

/// Monoblack → Luma u8 (pass-through).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  mono1bit_to_luma_row_generic::<false>(data, out, width);
}

/// Monoblack → Luma u16 (upshift (Y << 8) | Y).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  mono1bit_to_luma_u16_generic::<false>(data, out, width);
}

/// Monoblack → HSV (H=0, S=0, V=Y, achromatic).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  mono1bit_to_luma_row_generic::<false>(data, v, width);
  for i in 0..width {
    h[i] = 0;
    s[i] = 0;
  }
}

// ---- Monowhite (INVERT=true) ------------------------------------------------

/// Monowhite → RGB u8 (broadcast Y, inverted polarity).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  mono1bit_to_rgb_generic::<true>(data, out, width);
}

/// Monowhite → RGBA u8.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  mono1bit_to_rgba_generic::<true>(data, out, width);
}

/// Monowhite → RGB u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  mono1bit_to_rgb_u16_generic::<true>(data, out, width);
}

/// Monowhite → RGBA u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  mono1bit_to_rgba_u16_generic::<true>(data, out, width);
}

/// Monowhite → Luma u8.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  mono1bit_to_luma_row_generic::<true>(data, out, width);
}

/// Monowhite → Luma u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  mono1bit_to_luma_u16_generic::<true>(data, out, width);
}

/// Monowhite → HSV.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_hsv_row(
  data: &[u8],
  h: &mut [u8],
  s: &mut [u8],
  v: &mut [u8],
  width: usize,
) {
  mono1bit_to_luma_row_generic::<true>(data, v, width);
  for i in 0..width {
    h[i] = 0;
    s[i] = 0;
  }
}
