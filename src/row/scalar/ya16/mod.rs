//! Scalar Ya16 → {RGB, RGBA, RGB-u16, RGBA-u16, luma, luma-u16, HSV} kernels.
//!
//! Source is a `&[u16]` packed plane in `[Y0, A0, Y1, A1, ...]` order.
//! Each pixel occupies 2 u16 elements: Y at offset `2*x`, A at offset `2*x + 1`.
//!
//! # u8 outputs — downshift `>> 8`
//!
//! Y and A are narrowed from 16-bit to 8-bit via truncating right-shift:
//! `(sample >> 8) as u8`. This matches FFmpeg's `swscale` behavior for
//! big-depth-to-u8 conversions (consistent downward-bias truncation).
//!
//! # u16 outputs — native pass-through
//!
//! Y and A are forwarded as-is (native 16-bit depth).
//!
//! # HSV gray fast-path
//!
//! Gray sources are achromatic (S = 0 identically). H=0, S=0, V = Y >> 8.
//! α is dropped for HSV output.

/// Ya16 → packed u8 RGB. Y `>> 8`, broadcast R=G=B; α dropped.
///
/// When `BE = true`, each u16 element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgb_row<const BE: bool>(packed: &[u16], rgb_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let y_raw = if BE {
      u16::from_be(packed[x * 2])
    } else {
      u16::from_le(packed[x * 2])
    };
    let y8 = (y_raw >> 8) as u8;
    let i = x * 3;
    rgb_out[i] = y8;
    rgb_out[i + 1] = y8;
    rgb_out[i + 2] = y8;
  }
}

/// Ya16 → packed u8 RGBA. Y `>> 8`, broadcast R=G=B; A `>> 8` from source slot 1.
///
/// When `BE = true`, each u16 element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgba_row<const BE: bool>(packed: &[u16], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let y_raw = if BE {
      u16::from_be(packed[x * 2])
    } else {
      u16::from_le(packed[x * 2])
    };
    let a_raw = if BE {
      u16::from_be(packed[x * 2 + 1])
    } else {
      u16::from_le(packed[x * 2 + 1])
    };
    let y8 = (y_raw >> 8) as u8;
    let a8 = (a_raw >> 8) as u8;
    let i = x * 4;
    rgba_out[i] = y8;
    rgba_out[i + 1] = y8;
    rgba_out[i + 2] = y8;
    rgba_out[i + 3] = a8;
  }
}

/// Ya16 → packed u16 RGB. Y native u16, broadcast R=G=B=Y; α dropped.
///
/// When `BE = true`, each u16 element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgb_u16_row<const BE: bool>(
  packed: &[u16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let y = if BE {
      u16::from_be(packed[x * 2])
    } else {
      u16::from_le(packed[x * 2])
    };
    let i = x * 3;
    rgb_u16_out[i] = y;
    rgb_u16_out[i + 1] = y;
    rgb_u16_out[i + 2] = y;
  }
}

/// Ya16 → packed u16 RGBA. Y native u16, broadcast; A native u16 from source slot 1.
///
/// When `BE = true`, each u16 element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_rgba_u16_row<const BE: bool>(
  packed: &[u16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let y = if BE {
      u16::from_be(packed[x * 2])
    } else {
      u16::from_le(packed[x * 2])
    };
    let a = if BE {
      u16::from_be(packed[x * 2 + 1])
    } else {
      u16::from_le(packed[x * 2 + 1])
    };
    let i = x * 4;
    rgba_u16_out[i] = y;
    rgba_u16_out[i + 1] = y;
    rgba_u16_out[i + 2] = y;
    rgba_u16_out[i + 3] = a;
  }
}

/// Ya16 → luma u8. Y `>> 8`.
///
/// When `BE = true`, each u16 element is byte-swapped before processing.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_luma_row<const BE: bool>(packed: &[u16], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(luma_out.len() >= width, "luma_out too short");
  for x in 0..width {
    let y = if BE {
      u16::from_be(packed[x * 2])
    } else {
      u16::from_le(packed[x * 2])
    };
    luma_out[x] = (y >> 8) as u8;
  }
}

/// Ya16 → luma u16. Y native u16 pass-through (or byte-swap for BE).
///
/// When `BE = true`, each u16 element is byte-swapped before output.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_luma_u16_row<const BE: bool>(
  packed: &[u16],
  luma_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(luma_u16_out.len() >= width, "luma_u16_out too short");
  for x in 0..width {
    luma_u16_out[x] = if BE {
      u16::from_be(packed[x * 2])
    } else {
      u16::from_le(packed[x * 2])
    };
  }
}

/// Ya16 → HSV u8. Gray fast-path: H=0, S=0, V = Y `>> 8`. α dropped.
///
/// When `BE = true`, each u16 element is byte-swapped before processing.
/// See [`super::gray::gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn ya16_to_hsv_row<const BE: bool>(
  packed: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(h_out.len() >= width, "h_out too short");
  debug_assert!(s_out.len() >= width, "s_out too short");
  debug_assert!(v_out.len() >= width, "v_out too short");
  for x in 0..width {
    let y = if BE {
      u16::from_be(packed[x * 2])
    } else {
      u16::from_le(packed[x * 2])
    };
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = (y >> 8) as u8;
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
