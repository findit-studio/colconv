//! Scalar Grayf32 → {RGB, RGBA, RGB-u16, RGBA-u16, RGB-f32, luma, luma-u16,
//! luma-f32, HSV} kernels.
//!
//! Source is a `&[f32]` luma plane. Nominal range `[0.0, 1.0]`; HDR > 1.0 is
//! permitted — all integer-output kernels clamp via `.max(0.0).min(1.0)` before
//! scaling, using the same MXCSR-independent pattern as the Rgbf32 scalar.
//!
//! # Rounding (float → integer)
//!
//! `(y.clamp(0.0, 1.0) * scale + 0.5) as T`
//!
//! Adding 0.5 before truncation gives round-to-nearest (ties round up) without
//! depending on the floating-point rounding mode register (MXCSR on x86). This
//! matches the Rgbf32 scalar pattern.
//!
//! # Lossless paths (float → float)
//!
//! `grayf32_to_rgb_f32_row` and `grayf32_to_luma_f32_row` perform no clamping
//! and no rounding — the f32 value is forwarded as-is (memcpy-equivalent).
//!
//! # HSV gray fast-path
//!
//! Gray sources are achromatic (S = 0 identically). H is fixed to 0 to match
//! OpenCV's `cv2.COLOR_GRAY2HSV` convention. V is the clamped Y in u8.

// ---- shared helpers ---------------------------------------------------------

/// Round-to-nearest f32 → u8, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 255, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u8(y: f32) -> u8 {
  (y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Round-to-nearest f32 → u16, MXCSR-independent.
/// Clamps `y` to `[0.0, 1.0]`, multiplies by 65535, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u16(y: f32) -> u16 {
  (y.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

// ---- kernel implementations -------------------------------------------------

/// Grayf32 → packed u8 RGB. Clamp [0,1] × 255 → u8, broadcast R=G=B=Y.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgb_row<const BE: bool>(plane: &[f32], rgb_out: &mut [u8], width: usize) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let y = if BE {
      f32::from_bits(u32::from_be(raw.to_bits()))
    } else {
      f32::from_bits(u32::from_le(raw.to_bits()))
    };
    let v = f32_to_u8(y);
    let i = x * 3;
    rgb_out[i] = v;
    rgb_out[i + 1] = v;
    rgb_out[i + 2] = v;
  }
}

/// Grayf32 → packed u8 RGBA. Same broadcast as rgb; α = 0xFF.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgba_row<const BE: bool>(
  plane: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let y = if BE {
      f32::from_bits(u32::from_be(raw.to_bits()))
    } else {
      f32::from_bits(u32::from_le(raw.to_bits()))
    };
    let v = f32_to_u8(y);
    let i = x * 4;
    rgba_out[i] = v;
    rgba_out[i + 1] = v;
    rgba_out[i + 2] = v;
    rgba_out[i + 3] = 0xFF;
  }
}

/// Grayf32 → packed u16 RGB. Clamp [0,1] × 65535 → u16, broadcast R=G=B=Y.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgb_u16_row<const BE: bool>(
  plane: &[f32],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let y = if BE {
      f32::from_bits(u32::from_be(raw.to_bits()))
    } else {
      f32::from_bits(u32::from_le(raw.to_bits()))
    };
    let v = f32_to_u16(y);
    let i = x * 3;
    rgb_u16_out[i] = v;
    rgb_u16_out[i + 1] = v;
    rgb_u16_out[i + 2] = v;
  }
}

/// Grayf32 → packed u16 RGBA. Same broadcast; α = 0xFFFF.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgba_u16_row<const BE: bool>(
  plane: &[f32],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let y = if BE {
      f32::from_bits(u32::from_be(raw.to_bits()))
    } else {
      f32::from_bits(u32::from_le(raw.to_bits()))
    };
    let v = f32_to_u16(y);
    let i = x * 4;
    rgba_u16_out[i] = v;
    rgba_u16_out[i + 1] = v;
    rgba_u16_out[i + 2] = v;
    rgba_u16_out[i + 3] = 0xFFFF;
  }
}

/// Grayf32 → packed f32 RGB. Lossless: replicate Y → R=G=B (no clamp, no round).
///
/// When `BE = true`, each f32 element is byte-swapped (treats stored bits as
/// BE-encoded IEEE 754 and converts to host-native before replication).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_rgb_f32_row<const BE: bool>(
  plane: &[f32],
  rgb_f32_out: &mut [f32],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgb_f32_out.len() >= width * 3, "rgb_f32_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let y = if BE {
      f32::from_bits(u32::from_be(raw.to_bits()))
    } else {
      f32::from_bits(u32::from_le(raw.to_bits()))
    };
    let i = x * 3;
    rgb_f32_out[i] = y;
    rgb_f32_out[i + 1] = y;
    rgb_f32_out[i + 2] = y;
  }
}

/// Grayf32 → luma u8. Clamp [0,1] × 255 → u8.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_luma_row<const BE: bool>(
  plane: &[f32],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(luma_out.len() >= width, "luma_out too short");
  for (out, &raw) in luma_out[..width].iter_mut().zip(plane[..width].iter()) {
    let y = if BE {
      f32::from_bits(u32::from_be(raw.to_bits()))
    } else {
      f32::from_bits(u32::from_le(raw.to_bits()))
    };
    *out = f32_to_u8(y);
  }
}

/// Grayf32 → luma u16. Clamp [0,1] × 65535 → u16.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_luma_u16_row<const BE: bool>(
  plane: &[f32],
  luma_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(luma_u16_out.len() >= width, "luma_u16_out too short");
  for (out, &raw) in luma_u16_out[..width].iter_mut().zip(plane[..width].iter()) {
    let y = if BE {
      f32::from_bits(u32::from_be(raw.to_bits()))
    } else {
      f32::from_bits(u32::from_le(raw.to_bits()))
    };
    *out = f32_to_u16(y);
  }
}

/// Grayf32 → luma f32. Lossless pass-through (or byte-swap copy when the
/// encoded byte order differs from the host).
///
/// `BE` selects the **encoded byte order** of the input buffer: `false` =
/// LE-encoded, `true` = BE-encoded. A swap happens only when the encoded
/// order differs from the host CPU's native order.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_luma_f32_row<const BE: bool>(
  plane: &[f32],
  luma_f32_out: &mut [f32],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(luma_f32_out.len() >= width, "luma_f32_out too short");
  // Fast path: encoded byte order matches host-native — pure memcpy.
  // (LE-encoded data on LE host, or BE-encoded data on BE host.) The
  // const-generic `BE == HOST_NATIVE_BE` branch is dead-code-eliminated
  // per monomorphization, so this becomes a single `copy_from_slice` call
  // with no swap loop. Mirrors the `rgbf32_to_rgb_f32_row` fast path
  // landed in PR #83 (`b915754`).
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");
  if BE == HOST_NATIVE_BE {
    luma_f32_out[..width].copy_from_slice(&plane[..width]);
    return;
  }
  // Slow path: encoded byte order differs from host — byte-swap each f32
  // element via `u32::from_be` / `u32::from_le` (the dead branch is
  // eliminated since `BE` is const). Output is always host-native.
  for (out, &raw) in luma_f32_out[..width].iter_mut().zip(plane[..width].iter()) {
    *out = if BE {
      f32::from_bits(u32::from_be(raw.to_bits()))
    } else {
      f32::from_bits(u32::from_le(raw.to_bits()))
    };
  }
}

/// Grayf32 → HSV u8. Gray fast-path: H=0, S=0, V = clamp(Y, 0, 1) × 255.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
/// Gray sources are achromatic (saturation = 0 identically). H is fixed to 0
/// to match OpenCV's `cv2.COLOR_GRAY2HSV` convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf32_to_hsv_row<const BE: bool>(
  plane: &[f32],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(h_out.len() >= width, "h_out too short");
  debug_assert!(s_out.len() >= width, "s_out too short");
  debug_assert!(v_out.len() >= width, "v_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let y = if BE {
      f32::from_bits(u32::from_be(raw.to_bits()))
    } else {
      f32::from_bits(u32::from_le(raw.to_bits()))
    };
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = f32_to_u8(y);
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
