//! Scalar Grayf16 → {RGB, RGBA, RGB-u16, RGBA-u16, RGB-f32, luma, luma-u16,
//! luma-f32, HSV} kernels.
//!
//! Source is a `&[half::f16]` luma plane. Nominal range `[0.0, 1.0]`; HDR > 1.0
//! is permitted — all integer-output kernels widen each `f16` to `f32` and then
//! clamp via `.clamp(0.0, 1.0)` before scaling, using the same MXCSR-independent
//! pattern as the Grayf32 scalar. The half-float twin of [`super::grayf32`].
//!
//! # f16 reading
//!
//! Each `half::f16` element is decoded from `BE` byte order to host-native via
//! `half::f16::from_bits(u16::from_{be,le}(bits))` and widened with
//! [`half::f16::to_f32`] — the exact load + widen the Rgbf16 scalar uses. The
//! widen is lossless (every `f16` is representable in `f32`).
//!
//! # Rounding (float → integer)
//!
//! `(y.clamp(0.0, 1.0) * scale + 0.5) as T`
//!
//! Adding 0.5 before truncation gives round-to-nearest (ties round up) without
//! depending on the floating-point rounding mode register. This matches the
//! Grayf32 scalar pattern exactly (so the integer-output math is identical to
//! Grayf32 once the `f16` is widened to `f32`).
//!
//! # Lossless paths (f16 → f32)
//!
//! `grayf16_to_rgb_f32_row` and `grayf16_to_luma_f32_row` widen each `f16` to
//! `f32` with no clamping and no rounding — the widened value (HDR > 1.0 and
//! negatives included) is forwarded as-is.
//!
//! # HSV gray fast-path
//!
//! Gray sources are achromatic (S = 0 identically). H is fixed to 0 to match
//! OpenCV's `cv2.COLOR_GRAY2HSV` convention. V is the clamped Y in u8.

// ---- shared helpers ---------------------------------------------------------

/// Read one `half::f16` element from `raw`, decoding the bit pattern from `BE`
/// byte order to host-native, and widen to `f32`. Mirrors the Rgbf16 scalar
/// `load_f16` + `to_f32`.
#[inline(always)]
fn load_f16<const BE: bool>(raw: half::f16) -> f32 {
  let bits = raw.to_bits();
  half::f16::from_bits(if BE {
    u16::from_be(bits)
  } else {
    u16::from_le(bits)
  })
  .to_f32()
}

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

/// Grayf16 → packed u8 RGB. Widen f16 → f32, clamp [0,1] x 255 → u8, broadcast
/// R=G=B=Y.
///
/// When `BE = true`, each f16 element is loaded via byte-swapped u16 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgb_row<const BE: bool>(
  plane: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let v = f32_to_u8(load_f16::<BE>(raw));
    let i = x * 3;
    rgb_out[i] = v;
    rgb_out[i + 1] = v;
    rgb_out[i + 2] = v;
  }
}

/// Grayf16 → packed u8 RGBA. Same broadcast as rgb; α = 0xFF.
///
/// When `BE = true`, each f16 element is loaded via byte-swapped u16 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgba_row<const BE: bool>(
  plane: &[half::f16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let v = f32_to_u8(load_f16::<BE>(raw));
    let i = x * 4;
    rgba_out[i] = v;
    rgba_out[i + 1] = v;
    rgba_out[i + 2] = v;
    rgba_out[i + 3] = 0xFF;
  }
}

/// Grayf16 → packed u16 RGB. Widen f16 → f32, clamp [0,1] x 65535 → u16,
/// broadcast R=G=B=Y.
///
/// When `BE = true`, each f16 element is loaded via byte-swapped u16 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgb_u16_row<const BE: bool>(
  plane: &[half::f16],
  rgb_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let v = f32_to_u16(load_f16::<BE>(raw));
    let i = x * 3;
    rgb_u16_out[i] = v;
    rgb_u16_out[i + 1] = v;
    rgb_u16_out[i + 2] = v;
  }
}

/// Grayf16 → packed u16 RGBA. Same broadcast; α = 0xFFFF.
///
/// When `BE = true`, each f16 element is loaded via byte-swapped u16 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgba_u16_row<const BE: bool>(
  plane: &[half::f16],
  rgba_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let v = f32_to_u16(load_f16::<BE>(raw));
    let i = x * 4;
    rgba_u16_out[i] = v;
    rgba_u16_out[i + 1] = v;
    rgba_u16_out[i + 2] = v;
    rgba_u16_out[i + 3] = 0xFFFF;
  }
}

/// Grayf16 → packed f32 RGB. Lossless: widen f16 → f32 then replicate Y → R=G=B
/// (no clamp, no round). HDR values > 1.0 and negatives are preserved.
///
/// When `BE = true`, each f16 element is byte-swapped before widening.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_rgb_f32_row<const BE: bool>(
  plane: &[half::f16],
  rgb_f32_out: &mut [f32],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(rgb_f32_out.len() >= width * 3, "rgb_f32_out too short");
  for (x, &raw) in plane[..width].iter().enumerate() {
    let y = load_f16::<BE>(raw);
    let i = x * 3;
    rgb_f32_out[i] = y;
    rgb_f32_out[i + 1] = y;
    rgb_f32_out[i + 2] = y;
  }
}

/// Grayf16 → luma u8. Widen f16 → f32, clamp [0,1] x 255 → u8.
///
/// When `BE = true`, each f16 element is loaded via byte-swapped u16 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_luma_row<const BE: bool>(
  plane: &[half::f16],
  luma_out: &mut [u8],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(luma_out.len() >= width, "luma_out too short");
  for (out, &raw) in luma_out[..width].iter_mut().zip(plane[..width].iter()) {
    *out = f32_to_u8(load_f16::<BE>(raw));
  }
}

/// Grayf16 → luma u16. Widen f16 → f32, clamp [0,1] x 65535 → u16.
///
/// When `BE = true`, each f16 element is loaded via byte-swapped u16 bits.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_luma_u16_row<const BE: bool>(
  plane: &[half::f16],
  luma_u16_out: &mut [u16],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(luma_u16_out.len() >= width, "luma_u16_out too short");
  for (out, &raw) in luma_u16_out[..width].iter_mut().zip(plane[..width].iter()) {
    *out = f32_to_u16(load_f16::<BE>(raw));
  }
}

/// Grayf16 → luma f32. Lossless widen f16 → f32 (HDR > 1.0 and negatives
/// preserved). Output is always host-native f32.
///
/// When `BE = true`, each f16 element is byte-swapped before widening.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_luma_f32_row<const BE: bool>(
  plane: &[half::f16],
  luma_f32_out: &mut [f32],
  width: usize,
) {
  debug_assert!(plane.len() >= width, "plane too short");
  debug_assert!(luma_f32_out.len() >= width, "luma_f32_out too short");
  for (out, &raw) in luma_f32_out[..width].iter_mut().zip(plane[..width].iter()) {
    *out = load_f16::<BE>(raw);
  }
}

/// Grayf16 → HSV u8. Gray fast-path: H=0, S=0, V = clamp(Y, 0, 1) x 255.
///
/// When `BE = true`, each f16 element is loaded via byte-swapped u16 bits.
/// Gray sources are achromatic (saturation = 0 identically). H is fixed to 0
/// to match OpenCV's `cv2.COLOR_GRAY2HSV` convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn grayf16_to_hsv_row<const BE: bool>(
  plane: &[half::f16],
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
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = f32_to_u8(load_f16::<BE>(raw));
  }
}

#[cfg(all(test, feature = "std"))]
mod tests;
