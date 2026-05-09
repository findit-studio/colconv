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
mod tests {
  use super::*;

  /// Re-encode a host-native f32 slice as LE-encoded byte storage (each element
  /// stored with LE u32-bit byte layout). Kernels called with `BE = false`
  /// recover the intended host-native value via `u32::from_le` on both LE
  /// (no-op) and BE (byte-swap) hosts.
  fn as_le_f32(host: &[f32]) -> std::vec::Vec<f32> {
    host
      .iter()
      .map(|v| f32::from_bits(u32::from_ne_bytes(v.to_bits().to_le_bytes())))
      .collect()
  }

  // ---- grayf32_to_rgb_row --------------------------------------------------

  #[test]
  fn grayf32_to_rgb_zero() {
    let plane = [0.0f32];
    let mut out = [0xFFu8; 3];
    grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [0, 0, 0]);
  }

  #[test]
  fn grayf32_to_rgb_max() {
    let plane = as_le_f32(&[1.0f32]);
    let mut out = [0u8; 3];
    grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [255, 255, 255]);
  }

  #[test]
  fn grayf32_to_rgb_mid() {
    // Mid-gray Y=0.5 with round-half-up:
    //   0.5 * 255      = 127.5  (pure truncation would give 127)
    //   127.5 + 0.5    = 128.0  (round-half-up adds 0.5 first)
    //   trunc(128.0)   = 128
    // See module-level "Rounding (float → integer)" doc — `+ 0.5 then
    // truncate` is the contract this crate uses across scalar + SIMD.
    let plane = as_le_f32(&[0.5f32]);
    let mut out = [0u8; 3];
    grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [128, 128, 128]);
  }

  #[test]
  fn grayf32_to_rgb_saturates_high() {
    let plane = as_le_f32(&[1.5f32]);
    let mut out = [0u8; 3];
    grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [255, 255, 255]);
  }

  #[test]
  fn grayf32_to_rgb_saturates_low() {
    let plane = [-0.1f32];
    let mut out = [0xFFu8; 3];
    grayf32_to_rgb_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [0, 0, 0]);
  }

  // ---- grayf32_to_rgba_row -------------------------------------------------

  #[test]
  fn grayf32_to_rgba_zero_alpha_opaque() {
    let plane = [0.0f32];
    let mut out = [0u8; 4];
    grayf32_to_rgba_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [0, 0, 0, 0xFF]);
  }

  #[test]
  fn grayf32_to_rgba_max_alpha_opaque() {
    let plane = as_le_f32(&[1.0f32]);
    let mut out = [0u8; 4];
    grayf32_to_rgba_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [255, 255, 255, 0xFF]);
  }

  // ---- grayf32_to_rgb_u16_row ----------------------------------------------

  #[test]
  fn grayf32_to_rgb_u16_zero() {
    let plane = [0.0f32];
    let mut out = [0xFFFFu16; 3];
    grayf32_to_rgb_u16_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [0, 0, 0]);
  }

  #[test]
  fn grayf32_to_rgb_u16_max() {
    let plane = as_le_f32(&[1.0f32]);
    let mut out = [0u16; 3];
    grayf32_to_rgb_u16_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [65535, 65535, 65535]);
  }

  #[test]
  fn grayf32_to_rgb_u16_saturates_high() {
    let plane = as_le_f32(&[2.0f32]);
    let mut out = [0u16; 3];
    grayf32_to_rgb_u16_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [65535, 65535, 65535]);
  }

  // ---- grayf32_to_rgba_u16_row ---------------------------------------------

  #[test]
  fn grayf32_to_rgba_u16_opaque() {
    let plane = as_le_f32(&[1.0f32]);
    let mut out = [0u16; 4];
    grayf32_to_rgba_u16_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [65535, 65535, 65535, 0xFFFF]);
  }

  // ---- grayf32_to_rgb_f32_row ----------------------------------------------

  #[test]
  fn grayf32_to_rgb_f32_lossless_replicate() {
    // Non-clamped value preserved exactly. Output is host-native f32.
    let plane = as_le_f32(&[1.5f32]);
    let mut out = [0.0f32; 3];
    grayf32_to_rgb_f32_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [1.5, 1.5, 1.5]);
  }

  #[test]
  fn grayf32_to_rgb_f32_negative_preserved() {
    let plane = as_le_f32(&[-0.5f32]);
    let mut out = [0.0f32; 3];
    grayf32_to_rgb_f32_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [-0.5, -0.5, -0.5]);
  }

  // ---- grayf32_to_luma_row -------------------------------------------------

  #[test]
  fn grayf32_to_luma_zero() {
    let plane = [0.0f32];
    let mut out = [0xFFu8; 1];
    grayf32_to_luma_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [0]);
  }

  #[test]
  fn grayf32_to_luma_max() {
    let plane = as_le_f32(&[1.0f32]);
    let mut out = [0u8; 1];
    grayf32_to_luma_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [255]);
  }

  // ---- grayf32_to_luma_u16_row ---------------------------------------------

  #[test]
  fn grayf32_to_luma_u16_max() {
    let plane = as_le_f32(&[1.0f32]);
    let mut out = [0u16; 1];
    grayf32_to_luma_u16_row::<false>(&plane, &mut out, 1);
    assert_eq!(out, [65535]);
  }

  // ---- grayf32_to_luma_f32_row ---------------------------------------------

  #[test]
  fn grayf32_to_luma_f32_identity() {
    let plane = as_le_f32(&[0.0f32, 0.5, 1.0, 1.5, -0.1]);
    let mut out = [99.0f32; 5];
    grayf32_to_luma_f32_row::<false>(&plane, &mut out, 5);
    // Lossless pass-through — exact bit equality. Output is host-native f32.
    assert_eq!(out, [0.0, 0.5, 1.0, 1.5, -0.1]);
  }

  // ---- grayf32_to_hsv_row --------------------------------------------------

  #[test]
  fn grayf32_to_hsv_zero() {
    let plane = [0.0f32];
    let mut h = [0xFFu8; 1];
    let mut s = [0xFFu8; 1];
    let mut v = [0u8; 1];
    grayf32_to_hsv_row::<false>(&plane, &mut h, &mut s, &mut v, 1);
    assert_eq!(h[0], 0, "H must be 0 for achromatic source");
    assert_eq!(s[0], 0, "S must be 0 for achromatic source");
    assert_eq!(v[0], 0);
  }

  #[test]
  fn grayf32_to_hsv_max() {
    let plane = as_le_f32(&[1.0f32]);
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    grayf32_to_hsv_row::<false>(&plane, &mut h, &mut s, &mut v, 1);
    assert_eq!(h[0], 0);
    assert_eq!(s[0], 0);
    assert_eq!(v[0], 255);
  }

  #[test]
  fn grayf32_to_hsv_mid() {
    // 0.5 → (0.5 * 255 + 0.5) as u8 = 128
    let plane = as_le_f32(&[0.5f32]);
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    grayf32_to_hsv_row::<false>(&plane, &mut h, &mut s, &mut v, 1);
    assert_eq!(h[0], 0);
    assert_eq!(s[0], 0);
    assert_eq!(v[0], 128);
  }

  #[test]
  fn grayf32_to_hsv_clamps_hdr() {
    // HDR value > 1.0 saturates to V=255.
    let plane = as_le_f32(&[2.0f32]);
    let mut h = [0u8; 1];
    let mut s = [0u8; 1];
    let mut v = [0u8; 1];
    grayf32_to_hsv_row::<false>(&plane, &mut h, &mut s, &mut v, 1);
    assert_eq!(v[0], 255);
  }

  #[test]
  fn grayf32_to_rgb_multi_pixel() {
    let plane = as_le_f32(&[0.0f32, 1.0, 0.5]);
    let mut out = [0u8; 9];
    grayf32_to_rgb_row::<false>(&plane, &mut out, 3);
    assert_eq!(&out[0..3], &[0, 0, 0]);
    assert_eq!(&out[3..6], &[255, 255, 255]);
    assert_eq!(&out[6..9], &[128, 128, 128]); // 0.5 → 128
  }

  // ---- BE parity tests: grayf32 ---------------------------------------------
  // Pattern: construct LE f32 input, reinterpret bytes as BE-encoded f32
  // (i.e. byte-swap the u32 bits), call BE kernel, assert output matches LE run.

  /// Helper: produce a BE-encoded copy of an f32 slice (swap u32 bits of each element).
  fn f32_to_be_bytes(src: &[f32]) -> std::vec::Vec<f32> {
    src
      .iter()
      .map(|&v| f32::from_bits(v.to_bits().swap_bytes()))
      .collect()
  }

  #[test]
  fn grayf32_be_parity_rgb() {
    let le = [0.5f32];
    let be = f32_to_be_bytes(&le);
    let mut out_le = [0u8; 3];
    let mut out_be = [0u8; 3];
    grayf32_to_rgb_row::<false>(&le, &mut out_le, 1);
    grayf32_to_rgb_row::<true>(&be, &mut out_be, 1);
    assert_eq!(out_le, out_be, "BE and LE grayf32 rgb outputs must match");
  }

  #[test]
  fn grayf32_be_parity_luma() {
    let le = [0.25f32];
    let be = f32_to_be_bytes(&le);
    let mut out_le = [0u8; 1];
    let mut out_be = [0u8; 1];
    grayf32_to_luma_row::<false>(&le, &mut out_le, 1);
    grayf32_to_luma_row::<true>(&be, &mut out_be, 1);
    assert_eq!(out_le, out_be, "BE and LE grayf32 luma outputs must match");
  }

  /// Closes Copilot review PR #85 finding 3: the existing `grayf32_be_parity_*`
  /// suite covers the integer-output paths but never exercises the lossless
  /// `grayf32_to_luma_f32_row::<true>` (BE-encoded f32 → host-native f32)
  /// fast/slow paths. This test constructs an LE input + a BE-encoded mirror
  /// (built by swapping the u32 bits of each intended value, matching the
  /// existing `f32_to_be_bytes` helper convention used by the suite above),
  /// runs both kernels, and asserts bitwise equality of their outputs
  /// (NaN-safe via `f32::to_bits`).
  ///
  /// Path coverage by host:
  ///   LE host: LE kernel = memcpy fast path; BE kernel = slow swap path.
  ///   BE host: LE kernel = slow swap path; BE kernel = memcpy fast path.
  /// Either way both outputs must agree bit-for-bit, exercising the
  /// `BE == HOST_NATIVE_BE` gate from both directions.
  #[test]
  fn grayf32_to_luma_f32_row_be_le_parity_lossless() {
    // Mix of normal, HDR, negative, subnormal, and exact-zero values to
    // ensure non-symmetric byte layouts in every position.
    let le: std::vec::Vec<f32> = std::vec![0.25f32, 1.5, -0.5, 1e-5, 0.0, 65504.0];
    let width = le.len();
    // BE-encoded mirror: bit-swap each f32's u32 representation (the same
    // construction used by the existing `f32_to_be_bytes` helper).
    let be: std::vec::Vec<f32> = le
      .iter()
      .map(|&v| f32::from_bits(v.to_bits().swap_bytes()))
      .collect();
    let mut out_le = std::vec![0.0f32; width];
    let mut out_be = std::vec![0.0f32; width];
    grayf32_to_luma_f32_row::<false>(&le, &mut out_le, width);
    grayf32_to_luma_f32_row::<true>(&be, &mut out_be, width);
    // Bitwise equality is NaN-safe and confirms no rounding/clamping was
    // applied along either path.
    let bits_le: std::vec::Vec<u32> = out_le.iter().map(|v| v.to_bits()).collect();
    let bits_be: std::vec::Vec<u32> = out_be.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
      bits_le, bits_be,
      "BE and LE grayf32 luma_f32 outputs must match bit-for-bit"
    );
  }
}
