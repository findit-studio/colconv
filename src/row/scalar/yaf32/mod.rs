//! Scalar Yaf32 → {RGB, RGBA, RGB-u16, RGBA-u16, RGB-f32, luma, luma-u16,
//! luma-f32, HSV} kernels.
//!
//! Source is a `&[f32]` packed plane in `[Y0, A0, Y1, A1, ...]` order. Each
//! pixel occupies 2 f32 elements: Y at offset `2*x`, A at offset `2*x + 1`.
//! The single-precision gray+alpha twin of [`super::grayf32`] — Y is the luma
//! element handled exactly as Grayf32, A is real source alpha (also nominal
//! `[0.0, 1.0]`, clamp/scale/round identically at output).
//!
//! Nominal range `[0.0, 1.0]`; HDR > 1.0 is permitted — all integer-output
//! kernels clamp via `.clamp(0.0, 1.0)` before scaling, using the same
//! MXCSR-independent pattern as the Grayf32 scalar.
//!
//! # Rounding (float → integer)
//!
//! `(v.clamp(0.0, 1.0) * scale + 0.5) as T`
//!
//! Adding 0.5 before truncation gives round-to-nearest (ties round up) without
//! depending on the floating-point rounding mode register (MXCSR on x86). This
//! matches the Grayf32 scalar pattern exactly, applied to both Y and A.
//!
//! # Lossless paths (float → float)
//!
//! `yaf32_to_rgb_f32_row` and `yaf32_to_luma_f32_row` perform no clamping and
//! no rounding — the f32 Y value is forwarded as-is (memcpy-equivalent for the
//! host-native byte order).
//!
//! # HSV gray fast-path
//!
//! Gray sources are achromatic (S = 0 identically). H is fixed to 0 to match
//! OpenCV's `cv2.COLOR_GRAY2HSV` convention. V is the clamped Y in u8. α is
//! dropped for HSV output.

// ---- shared helpers ---------------------------------------------------------

/// Decode one packed f32 element from `BE` byte order to host-native.
#[inline(always)]
fn load_f32<const BE: bool>(raw: f32) -> f32 {
  let bits = raw.to_bits();
  f32::from_bits(if BE {
    u32::from_be(bits)
  } else {
    u32::from_le(bits)
  })
}

/// Round-to-nearest f32 → u8, MXCSR-independent.
/// Clamps `v` to `[0.0, 1.0]`, multiplies by 255, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u8(v: f32) -> u8 {
  (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Round-to-nearest f32 → u16, MXCSR-independent.
/// Clamps `v` to `[0.0, 1.0]`, multiplies by 65535, adds 0.5, truncates.
#[inline(always)]
fn f32_to_u16(v: f32) -> u16 {
  (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16
}

// ---- kernel implementations -------------------------------------------------

/// Yaf32 → packed u8 RGB. Clamp Y `[0,1] x 255` → u8, broadcast R=G=B=Y; α
/// dropped.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgb_row<const BE: bool>(packed: &[f32], rgb_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let v = f32_to_u8(load_f32::<BE>(packed[x * 2]));
    let i = x * 3;
    rgb_out[i] = v;
    rgb_out[i + 1] = v;
    rgb_out[i + 2] = v;
  }
}

/// Yaf32 → packed u8 RGBA. Clamp Y broadcast R=G=B; α = clamp(A) `x 255` from
/// source slot 1.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgba_row<const BE: bool>(packed: &[f32], rgba_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let y = f32_to_u8(load_f32::<BE>(packed[x * 2]));
    let a = f32_to_u8(load_f32::<BE>(packed[x * 2 + 1]));
    let i = x * 4;
    rgba_out[i] = y;
    rgba_out[i + 1] = y;
    rgba_out[i + 2] = y;
    rgba_out[i + 3] = a;
  }
}

/// Yaf32 → packed u16 RGB. Clamp Y `[0,1] x 65535` → u16, broadcast R=G=B=Y; α
/// dropped.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgb_u16_row<const BE: bool>(
  packed: &[f32],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let v = f32_to_u16(load_f32::<BE>(packed[x * 2]));
    let i = x * 3;
    rgb_u16_out[i] = v;
    rgb_u16_out[i + 1] = v;
    rgb_u16_out[i + 2] = v;
  }
}

/// Yaf32 → packed u16 RGBA. Clamp Y broadcast; α = clamp(A) `x 65535` from
/// source slot 1.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgba_u16_row<const BE: bool>(
  packed: &[f32],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let y = f32_to_u16(load_f32::<BE>(packed[x * 2]));
    let a = f32_to_u16(load_f32::<BE>(packed[x * 2 + 1]));
    let i = x * 4;
    rgba_u16_out[i] = y;
    rgba_u16_out[i + 1] = y;
    rgba_u16_out[i + 2] = y;
    rgba_u16_out[i + 3] = a;
  }
}

/// Yaf32 → packed f32 RGB. Lossless: replicate Y → R=G=B (no clamp, no round);
/// α dropped.
///
/// When `BE = true`, each f32 element is byte-swapped (treats stored bits as
/// BE-encoded IEEE 754 and converts to host-native before replication).
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_rgb_f32_row<const BE: bool>(
  packed: &[f32],
  rgb_f32_out: &mut [f32],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(rgb_f32_out.len() >= width * 3, "rgb_f32_out too short");
  for x in 0..width {
    let y = load_f32::<BE>(packed[x * 2]);
    let i = x * 3;
    rgb_f32_out[i] = y;
    rgb_f32_out[i + 1] = y;
    rgb_f32_out[i + 2] = y;
  }
}

/// Yaf32 → luma u8. Clamp Y `[0,1] x 255` → u8.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_luma_row<const BE: bool>(packed: &[f32], luma_out: &mut [u8], width: usize) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(luma_out.len() >= width, "luma_out too short");
  for x in 0..width {
    luma_out[x] = f32_to_u8(load_f32::<BE>(packed[x * 2]));
  }
}

/// Yaf32 → luma u16. Clamp Y `[0,1] x 65535` → u16.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_luma_u16_row<const BE: bool>(
  packed: &[f32],
  luma_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(luma_u16_out.len() >= width, "luma_u16_out too short");
  for x in 0..width {
    luma_u16_out[x] = f32_to_u16(load_f32::<BE>(packed[x * 2]));
  }
}

/// Yaf32 → luma f32. Lossless host-native pass-through of Y (or byte-swap copy
/// when the encoded byte order differs from the host).
///
/// `BE` selects the **encoded byte order** of the input buffer: `false` =
/// LE-encoded, `true` = BE-encoded. A swap happens only when the encoded order
/// differs from the host CPU's native order.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_luma_f32_row<const BE: bool>(
  packed: &[f32],
  luma_f32_out: &mut [f32],
  width: usize,
) {
  debug_assert!(packed.len() >= width * 2, "packed too short");
  debug_assert!(luma_f32_out.len() >= width, "luma_f32_out too short");
  for x in 0..width {
    luma_f32_out[x] = load_f32::<BE>(packed[x * 2]);
  }
}

/// Yaf32 → HSV u8. Gray fast-path: H=0, S=0, V = clamp(Y, 0, 1) `x 255`. α
/// dropped.
///
/// When `BE = true`, each f32 element is loaded via byte-swapped u32 bits.
/// Gray sources are achromatic (saturation = 0 identically). H is fixed to 0
/// to match OpenCV's `cv2.COLOR_GRAY2HSV` convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yaf32_to_hsv_row<const BE: bool>(
  packed: &[f32],
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
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = f32_to_u8(load_f32::<BE>(packed[x * 2]));
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
