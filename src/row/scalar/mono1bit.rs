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

// ---- Monoblack (INVERT=false) -----------------------------------------------

/// Monoblack → RGB u8 (broadcast Y).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_rgb_row(data: &[u8], out: &mut [u8], width: usize) {
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<false>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i];
    out[i * 3] = y;
    out[i * 3 + 1] = y;
    out[i * 3 + 2] = y;
  }
}

/// Monoblack → RGBA u8 (broadcast Y, α=0xFF).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<false>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i];
    out[i * 4] = y;
    out[i * 4 + 1] = y;
    out[i * 4 + 2] = y;
    out[i * 4 + 3] = 0xFF;
  }
}

/// Monoblack → RGB u16 (broadcast Y, upshift (Y << 8) | Y).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<false>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i] as u16;
    let y16 = (y << 8) | y;
    out[i * 3] = y16;
    out[i * 3 + 1] = y16;
    out[i * 3 + 2] = y16;
  }
}

/// Monoblack → RGBA u16 (broadcast Y, α=0xFFFF).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<false>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i] as u16;
    let y16 = (y << 8) | y;
    out[i * 4] = y16;
    out[i * 4 + 1] = y16;
    out[i * 4 + 2] = y16;
    out[i * 4 + 3] = 0xFFFF;
  }
}

/// Monoblack → Luma u8 (pass-through).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  mono1bit_to_luma_row_generic::<false>(data, out, width);
}

/// Monoblack → Luma u16 (upshift (Y << 8) | Y).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monoblack_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<false>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i] as u16;
    out[i] = (y << 8) | y;
  }
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
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<true>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i];
    out[i * 3] = y;
    out[i * 3 + 1] = y;
    out[i * 3 + 2] = y;
  }
}

/// Monowhite → RGBA u8.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_rgba_row(data: &[u8], out: &mut [u8], width: usize) {
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<true>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i];
    out[i * 4] = y;
    out[i * 4 + 1] = y;
    out[i * 4 + 2] = y;
    out[i * 4 + 3] = 0xFF;
  }
}

/// Monowhite → RGB u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_rgb_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<true>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i] as u16;
    let y16 = (y << 8) | y;
    out[i * 3] = y16;
    out[i * 3 + 1] = y16;
    out[i * 3 + 2] = y16;
  }
}

/// Monowhite → RGBA u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_rgba_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<true>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i] as u16;
    let y16 = (y << 8) | y;
    out[i * 4] = y16;
    out[i * 4 + 1] = y16;
    out[i * 4 + 2] = y16;
    out[i * 4 + 3] = 0xFFFF;
  }
}

/// Monowhite → Luma u8.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_luma_row(data: &[u8], out: &mut [u8], width: usize) {
  mono1bit_to_luma_row_generic::<true>(data, out, width);
}

/// Monowhite → Luma u16.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn monowhite_to_luma_u16_row(data: &[u8], out: &mut [u16], width: usize) {
  let mut luma = vec![0u8; width];
  mono1bit_to_luma_row_generic::<true>(data, &mut luma, width);
  for i in 0..width {
    let y = luma[i] as u16;
    out[i] = (y << 8) | y;
  }
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
