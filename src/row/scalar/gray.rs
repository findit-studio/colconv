//! Scalar gray → {RGB, RGBA, HSV, luma, luma_u16} kernels.
//!
//! # Gray8 (u8 source)
//!
//! `gray8_to_rgb_row`  — broadcast Y to each R/G/B channel.
//! `gray8_to_rgba_row` — broadcast Y + force α = 0xFF.
//! `gray8_to_hsv_row`  — H=0, S=0, V=Y (S=0 convention: H is fixed to 0).
//! `gray8_to_luma_u16_row` — zero-extend Y to u16 (`out[x] = y[x] as u16`).
//!   (luma u8 identity copy is handled by the sinker directly via
//!   `copy_from_slice`, no dedicated kernel needed.)
//!
//! # GrayN / Gray16 (u16 source)
//!
//! `gray_n_to_rgb_row<BITS>`       — mask → downshift to u8, broadcast.
//! `gray_n_to_rgba_row<BITS>`      — same + α = 0xFF.
//! `gray_n_to_rgb_u16_row<BITS>`   — mask, keep native depth, broadcast.
//! `gray_n_to_rgba_u16_row<BITS>`  — same + α = full-range max for BITS.
//! `gray_n_to_luma_row<BITS>`      — mask → downshift to u8.
//! `gray_n_to_luma_u16_row<BITS>`  — mask → identity (samples already u16).
//! `gray_n_to_hsv_row<BITS>`       — mask → downshift to u8 → H=0 S=0 V=Y8.
//!
//! `gray16_to_rgb_row`    — `>> 8` to u8, broadcast.
//! `gray16_to_rgba_row`   — same + α = 0xFF.
//! `gray16_to_rgb_u16_row`  — native u16, broadcast.
//! `gray16_to_rgba_u16_row` — native u16 + α = 0xFFFF.
//! `gray16_to_luma_row`   — `>> 8` to u8.
//! `gray16_to_luma_u16_row` — identity copy to u16.
//! `gray16_to_hsv_row`    — `>> 8` to u8 → H=0 S=0 V=Y8.
//!
//! # HSV S=0 convention
//!
//! When S=0 (which is always for gray sources — delta = 0), H is set to 0.
//! This matches OpenCV `cv2.COLOR_GRAY2HSV` behavior.
//!
//! # `full_range` parameter (RGB/RGBA/HSV outputs only)
//!
//! - `full_range = true`: raw Y is used directly (broadcast / downshift).
//!   This is the standard full-range path.
//! - `full_range = false`: limited-range Y is rescaled to full range before
//!   broadcast. For 8-bit limited range: black=16, white=235. The rescaled
//!   value is `clamp_u8(((y - black) * 255 + range/2) / range)`.
//!
//! **Luma outputs** (`with_luma`, `with_luma_u16`) always pass Y through
//! without rescaling — the caller is explicitly requesting the source luma
//! plane as-is, regardless of `full_range`.

use super::bits_mask;

// ---- helpers ----------------------------------------------------------------

/// Broadcasts a `u8` gray value to packed RGB (3 bytes: R=G=B=y).
#[inline(always)]
fn broadcast_u8_to_rgb(y: u8, out: &mut [u8], x: usize) {
  let i = x * 3;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
}

/// Broadcasts a `u8` gray value to packed RGBA (4 bytes: R=G=B=y, A=0xFF).
#[inline(always)]
fn broadcast_u8_to_rgba(y: u8, out: &mut [u8], x: usize) {
  let i = x * 4;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
  out[i + 3] = 0xFF;
}

/// Broadcasts a `u16` gray value to packed u16 RGB (3 u16: R=G=B=y).
#[inline(always)]
fn broadcast_u16_to_rgb(y: u16, out: &mut [u16], x: usize) {
  let i = x * 3;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
}

/// Broadcasts a `u16` gray value to packed u16 RGBA (4 u16: R=G=B=y, A=alpha).
#[inline(always)]
fn broadcast_u16_to_rgba(y: u16, alpha: u16, out: &mut [u16], x: usize) {
  let i = x * 4;
  out[i] = y;
  out[i + 1] = y;
  out[i + 2] = y;
  out[i + 3] = alpha;
}

/// Rescales a limited-range 8-bit luma value to full-range u8.
///
/// Limited-range 8-bit: black=16, white=235 (range=219).
/// Formula: `clamp_u8(((y - 16) * 255 + 109) / 219)` (109 = 219/2, rounding).
#[inline(always)]
fn limited_to_full_u8(y: u8) -> u8 {
  let y = y as i32;
  let rescaled = (y - 16) * 255 + 109; // 109 = 219/2 for rounding
  let result = rescaled / 219;
  result.clamp(0, 255) as u8
}

/// Rescales a limited-range N-bit luma value to full-range u8.
///
/// Limited-range: black = 16 << (BITS-8), range = 219 << (BITS-8).
#[inline(always)]
fn limited_n_to_full_u8<const BITS: u32>(y: u16) -> u8 {
  let black = 16i32 << (BITS - 8);
  let range = 219i32 << (BITS - 8);
  let y = y as i32;
  let rescaled = (y - black) * 255 + range / 2;
  let result = rescaled / range;
  result.clamp(0, 255) as u8
}

/// Rescales a limited-range N-bit luma value to full-range u16 (native depth).
///
/// Limited-range: black = 16 << (BITS-8), range = 219 << (BITS-8).
/// Output is clamped to [0, max_native] where max_native = (1 << BITS) - 1.
///
/// Math runs in `i64` to keep the `(y - black) * max_native` product safe at
/// `BITS = 16`: limited-range white `60160` minus black `4096` times
/// `max_native = 65535` is `~3.67 × 10^9`, which overflows `i32`. Lower bit
/// depths fit in `i32` but using `i64` uniformly keeps one signature and
/// avoids per-BITS branches.
#[inline(always)]
fn limited_n_to_full_u16<const BITS: u32>(y: u16) -> u16 {
  let black = 16i64 << (BITS - 8);
  let range = 219i64 << (BITS - 8);
  let max_native = ((1u64 << BITS) - 1) as i64;
  let y = y as i64;
  let rescaled = (y - black) * max_native + range / 2;
  let result = rescaled / range;
  result.clamp(0, max_native) as u16
}

/// Rescales a limited-range 16-bit luma value to full-range u8.
///
/// Limited-range 16-bit: black = 16 << 8 = 4096, range = 219 << 8 = 56064.
#[inline(always)]
fn limited_16_to_full_u8(y: u16) -> u8 {
  limited_n_to_full_u8::<16>(y)
}

/// Rescales a limited-range 16-bit luma value to full-range u16.
///
/// Limited-range 16-bit: black = 4096, range = 56064, max_native = 65535.
#[inline(always)]
fn limited_16_to_full_u16(y: u16) -> u16 {
  limited_n_to_full_u16::<16>(y)
}

// ---- Gray8 ------------------------------------------------------------------

/// Broadcasts each `u8` gray sample to packed RGB (`R = G = B = Y`).
///
/// When `full_range = false`, limited-range Y (black=16, white=235) is
/// rescaled to full-range [0, 255] before broadcast.
/// When `full_range = true`, raw Y is used directly.
///
/// Luma outputs always pass Y through without rescaling.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_rgb_row(y_plane: &[u8], out: &mut [u8], width: usize, full_range: bool) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &y) in y_plane[..width].iter().enumerate() {
    let y_out = if full_range { y } else { limited_to_full_u8(y) };
    broadcast_u8_to_rgb(y_out, out, x);
  }
}

/// Broadcasts each `u8` gray sample to packed RGBA (`R = G = B = Y`, `A = 0xFF`).
///
/// When `full_range = false`, limited-range Y is rescaled to full range.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_rgba_row(y_plane: &[u8], out: &mut [u8], width: usize, full_range: bool) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &y) in y_plane[..width].iter().enumerate() {
    let y_out = if full_range { y } else { limited_to_full_u8(y) };
    broadcast_u8_to_rgba(y_out, out, x);
  }
}

/// Gray8 → HSV row. Convention: S=0, H=0, V=Y (or rescaled Y if limited range).
///
/// Gray sources are achromatic (saturation = 0). When S=0, H is
/// undefined in the continuous HSV model; this crate fixes H=0 to
/// match OpenCV's `cv2.COLOR_GRAY2HSV` convention and avoid
/// non-deterministic hue output.
///
/// When `full_range = false`, the V channel uses the rescaled luma value.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray8_to_hsv_row(
  y_plane: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  for (x, &y) in y_plane[..width].iter().enumerate() {
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = if full_range { y } else { limited_to_full_u8(y) };
  }
}

// ---- GrayN (u16 low-bit-packed, BITS in {9,10,12,14}) ----------------------

/// GrayN → packed RGB u8. Masks to BITS bits, downshifts `BITS - 8` to u8,
/// broadcasts.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 255]
/// before broadcast. Luma outputs always pass Y through without rescaling.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgb_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    let y8 = if full_range {
      (masked >> shift) as u8
    } else {
      limited_n_to_full_u8::<BITS>(masked)
    };
    broadcast_u8_to_rgb(y8, out, x);
  }
}

/// GrayN → packed RGBA u8. Masks to BITS bits, downshifts to u8, broadcasts,
/// α = 0xFF.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 255].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgba_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    let y8 = if full_range {
      (masked >> shift) as u8
    } else {
      limited_n_to_full_u8::<BITS>(masked)
    };
    broadcast_u8_to_rgba(y8, out, x);
  }
}

/// GrayN → packed u16 RGB. Masks to BITS bits, broadcasts at native depth.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to full native range
/// [0, (1<<BITS)-1] before broadcast.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgb_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  let mask = bits_mask::<BITS>();
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    let y_out = if full_range {
      masked
    } else {
      limited_n_to_full_u16::<BITS>(masked)
    };
    broadcast_u16_to_rgb(y_out, out, x);
  }
}

/// GrayN → packed u16 RGBA. Masks to BITS bits, broadcasts, α = `(1 << BITS) - 1`.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to full native range.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_rgba_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  let mask = bits_mask::<BITS>();
  let alpha = mask; // full-range max for BITS
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    let y_out = if full_range {
      masked
    } else {
      limited_n_to_full_u16::<BITS>(masked)
    };
    broadcast_u16_to_rgba(y_out, alpha, out, x);
  }
}

/// GrayN → luma u8. Masks to BITS bits, downshifts `BITS - 8`.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_luma_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (out_byte, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    *out_byte = ((raw & mask) >> shift) as u8;
  }
}

/// GrayN → luma u16. Masks to BITS bits, identity copy.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_luma_u16_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  let mask = bits_mask::<BITS>();
  for (out_el, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    *out_el = raw & mask;
  }
}

/// GrayN → HSV u8. Masks to BITS bits, downshifts to u8, H=0 S=0 V=Y8.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, the V channel uses the rescaled luma value.
/// See [`gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray_n_to_hsv_row<const BITS: u32, const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  let mask = bits_mask::<BITS>();
  let shift = BITS - 8;
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let masked = raw & mask;
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = if full_range {
      (masked >> shift) as u8
    } else {
      limited_n_to_full_u8::<BITS>(masked)
    };
  }
}

// ---- Gray16 (u16, all 16 bits active) ----------------------------------------

/// Gray16 → packed RGB u8. Downshifts `>> 8` to u8, broadcasts.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y (black=4096, white=56064+4096)
/// is rescaled to [0, 255] before broadcast.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgb_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let y8 = if full_range {
      (raw >> 8) as u8
    } else {
      limited_16_to_full_u8(raw)
    };
    broadcast_u8_to_rgb(y8, out, x);
  }
}

/// Gray16 → packed RGBA u8. Downshifts `>> 8`, broadcasts, α = 0xFF.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 255].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgba_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let y8 = if full_range {
      (raw >> 8) as u8
    } else {
      limited_16_to_full_u8(raw)
    };
    broadcast_u8_to_rgba(y8, out, x);
  }
}

/// Gray16 → packed u16 RGB. Identity broadcast, native 16-bit depth.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 65535].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgb_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 3, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let y_out = if full_range {
      raw
    } else {
      limited_16_to_full_u16(raw)
    };
    broadcast_u16_to_rgb(y_out, out, x);
  }
}

/// Gray16 → packed u16 RGBA. Identity broadcast, α = 0xFFFF.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, limited-range Y is rescaled to [0, 65535].
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_rgba_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width * 4, "out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    let y_out = if full_range {
      raw
    } else {
      limited_16_to_full_u16(raw)
    };
    broadcast_u16_to_rgba(y_out, 0xFFFF, out, x);
  }
}

/// Gray16 → luma u8. Downshifts `>> 8`.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_luma_row<const BE: bool>(y_plane: &[u16], out: &mut [u8], width: usize) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  for (out_byte, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    *out_byte = (raw >> 8) as u8;
  }
}

/// Gray16 → luma u16. Identity copy (or byte-swap copy for BE).
///
/// When `BE = true`, each u16 sample is byte-swapped before output.
/// Always passes raw Y through without `full_range` rescaling —
/// the caller is explicitly requesting the source luma plane as-is.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_luma_u16_row<const BE: bool>(
  y_plane: &[u16],
  out: &mut [u16],
  width: usize,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(out.len() >= width, "out too short");
  for (o, &raw) in out[..width].iter_mut().zip(y_plane[..width].iter()) {
    *o = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
  }
}

/// Gray16 → HSV u8. `>> 8` to u8, H=0 S=0 V=Y8.
///
/// When `BE = true`, each u16 sample is byte-swapped before processing.
/// When `full_range = false`, the V channel uses the rescaled luma value.
/// See [`gray8_to_hsv_row`] for the S=0 convention.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gray16_to_hsv_row<const BE: bool>(
  y_plane: &[u16],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  full_range: bool,
) {
  debug_assert!(y_plane.len() >= width, "y_plane too short");
  debug_assert!(h_out.len() >= width, "H out too short");
  debug_assert!(s_out.len() >= width, "S out too short");
  debug_assert!(v_out.len() >= width, "V out too short");
  for (x, &raw) in y_plane[..width].iter().enumerate() {
    let raw = if BE {
      u16::from_be(raw)
    } else {
      u16::from_le(raw)
    };
    h_out[x] = 0;
    s_out[x] = 0;
    v_out[x] = if full_range {
      (raw >> 8) as u8
    } else {
      limited_16_to_full_u8(raw)
    };
  }
}

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  #[test]
  fn gray8_to_rgb_broadcasts() {
    let y = [0u8, 128, 255];
    let mut out = [0u8; 9];
    gray8_to_rgb_row(&y, &mut out, 3, true);
    assert_eq!(&out[0..3], &[0, 0, 0]);
    assert_eq!(&out[3..6], &[128, 128, 128]);
    assert_eq!(&out[6..9], &[255, 255, 255]);
  }

  #[test]
  fn gray8_to_rgba_broadcasts_opaque() {
    let y = [100u8, 200];
    let mut out = [0u8; 8];
    gray8_to_rgba_row(&y, &mut out, 2, true);
    assert_eq!(&out[0..4], &[100, 100, 100, 0xFF]);
    assert_eq!(&out[4..8], &[200, 200, 200, 0xFF]);
  }

  #[test]
  fn gray8_to_hsv_h0_s0_v_y() {
    let y = [0u8, 128, 255];
    let mut h = [0xFFu8; 3];
    let mut s = [0xFFu8; 3];
    let mut v = [0u8; 3];
    gray8_to_hsv_row(&y, &mut h, &mut s, &mut v, 3, true);
    assert_eq!(h, [0, 0, 0]);
    assert_eq!(s, [0, 0, 0]);
    assert_eq!(v, [0, 128, 255]);
  }

  // ---- LE-host gating rationale (BE-tier11 follow-up) -----------------------
  //
  // The Gray9/10/12/14/16 limited-range / full-range / mask / luma / HSV /
  // identity / opaque tests below construct fixtures as host-native
  // `Vec<u16>` literals (e.g. `std::vec![512u16]`) and call the kernels with
  // `::<false>`, which means "input is LE-encoded — decode to host-native by
  // applying `u16::from_le`".
  //
  // On a little-endian host, host-native u16 bits and LE-encoded bits are the
  // same byte sequence, so `u16::from_le` is a no-op and the assertions hold.
  //
  // On a big-endian host (powerpc64 / s390x / aarch64-be / mips), host-native
  // u16 bits do NOT lay out little-endian, so the kernel's `from_le`
  // byte-swap correctly reinterprets the host-native fixture as if it were
  // an LE-encoded payload — producing a different (corrupted) value than the
  // test expects.  The kernel itself is correct; this is purely a
  // fixture-vs-kernel byte-order mismatch on BE hosts (same class as the
  // PR #82 alpha_extract / planar_gbr_high_bit gates in `8f2e329` and the
  // PR #83 Rgbf16 gates in `56342c0`).
  //
  // Kernel BE-host correctness is locked down separately by the dedicated
  // `gray*_be_parity_*` tests further down in this module, which build the
  // BE-encoded input via `swap_bytes()` and assert that BE+`<true>` matches
  // LE+`<false>`. Those tests are intentionally NOT gated.
  //
  // Byte-symmetric value tests (`0x0000`, `0xFFFF`, `u16::MAX`) are also NOT
  // gated — their bytes lay out the same in either order, so the kernel's
  // `from_le` swap is a true no-op on every host.

  // ---- limited-range tests: Gray8 ----

  #[test]
  fn gray8_limited_range_black() {
    // Y=16 → full-range black → RGB(0,0,0)
    let y = [16u8];
    let mut out = [0u8; 3];
    gray8_to_rgb_row(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[0, 0, 0]);
  }

  #[test]
  fn gray8_limited_range_white() {
    // Y=235 → full-range white → RGB(255,255,255)
    let y = [235u8];
    let mut out = [0u8; 3];
    gray8_to_rgb_row(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[255, 255, 255]);
  }

  #[test]
  fn gray8_limited_range_midpoint() {
    // Y=125 (8-bit ref mid of 16..235 span) → approx 127
    let y = [125u8]; // (125-16)*255/219 ≈ 127
    let mut out = [0u8; 3];
    gray8_to_rgb_row(&y, &mut out, 1, false);
    // Allow ±1 for rounding
    assert!(
      out[0] >= 126 && out[0] <= 128,
      "expected ~127 got {}",
      out[0]
    );
    assert_eq!(out[0], out[1]);
    assert_eq!(out[0], out[2]);
  }

  #[test]
  fn gray8_limited_range_rgba() {
    let y = [16u8, 235];
    let mut out = [0u8; 8];
    gray8_to_rgba_row(&y, &mut out, 2, false);
    assert_eq!(&out[0..4], &[0, 0, 0, 0xFF]);
    assert_eq!(&out[4..8], &[255, 255, 255, 0xFF]);
  }

  #[test]
  fn gray8_limited_range_hsv() {
    let y = [16u8, 235];
    let mut h = [0xFFu8; 2];
    let mut s = [0xFFu8; 2];
    let mut v = [0u8; 2];
    gray8_to_hsv_row(&y, &mut h, &mut s, &mut v, 2, false);
    assert_eq!(h, [0, 0]);
    assert_eq!(s, [0, 0]);
    assert_eq!(v, [0, 255]);
  }

  // ---- Gray10 limited-range tests ----

  #[test]
  #[cfg(target_endian = "little")]
  fn gray10_limited_range_black() {
    // 10-bit black = 16 << 2 = 64
    let y: std::vec::Vec<u16> = std::vec![64u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<10, false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[0, 0, 0]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray10_limited_range_white() {
    // 10-bit white = 235 << 2 = 940
    let y: std::vec::Vec<u16> = std::vec![940u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<10, false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[255, 255, 255]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray10_limited_range_midpoint() {
    // 10-bit mid: 125 << 2 = 500 → approx 127
    let y: std::vec::Vec<u16> = std::vec![500u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<10, false>(&y, &mut out, 1, false);
    assert!(
      out[0] >= 126 && out[0] <= 128,
      "expected ~127 got {}",
      out[0]
    );
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray10_full_range_pass_through() {
    // 10-bit full range: value 512 >> 2 = 128
    let y: std::vec::Vec<u16> = std::vec![512u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<10, false>(&y, &mut out, 1, true);
    assert_eq!(&out[0..3], &[128, 128, 128]);
  }

  // ---- Gray12 limited-range tests ----

  #[test]
  #[cfg(target_endian = "little")]
  fn gray12_limited_range_black() {
    // 12-bit black = 16 << 4 = 256
    let y: std::vec::Vec<u16> = std::vec![256u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<12, false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[0, 0, 0]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray12_limited_range_white() {
    // 12-bit white = 235 << 4 = 3760
    let y: std::vec::Vec<u16> = std::vec![3760u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<12, false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[255, 255, 255]);
  }

  // ---- Gray14 limited-range tests ----

  #[test]
  #[cfg(target_endian = "little")]
  fn gray14_limited_range_black() {
    // 14-bit black = 16 << 6 = 1024
    let y: std::vec::Vec<u16> = std::vec![1024u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<14, false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[0, 0, 0]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray14_limited_range_white() {
    // 14-bit white = 235 << 6 = 15040
    let y: std::vec::Vec<u16> = std::vec![15040u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<14, false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[255, 255, 255]);
  }

  // ---- Gray16 limited-range tests ----

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_limited_range_black() {
    // 16-bit black = 16 << 8 = 4096
    let y: std::vec::Vec<u16> = std::vec![4096u16];
    let mut out = std::vec![0u8; 3];
    gray16_to_rgb_row::<false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[0, 0, 0]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_limited_range_white() {
    // 16-bit white = 235 << 8 = 60160
    let y: std::vec::Vec<u16> = std::vec![60160u16];
    let mut out = std::vec![0u8; 3];
    gray16_to_rgb_row::<false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[255, 255, 255]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_limited_range_midpoint() {
    // 16-bit mid: 125 << 8 = 32000 → approx 127
    let y: std::vec::Vec<u16> = std::vec![32000u16];
    let mut out = std::vec![0u8; 3];
    gray16_to_rgb_row::<false>(&y, &mut out, 1, false);
    assert!(
      out[0] >= 126 && out[0] <= 128,
      "expected ~127 got {}",
      out[0]
    );
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_full_range_pass_through() {
    // 16-bit full range: 0x8000 >> 8 = 128
    let y: std::vec::Vec<u16> = std::vec![0x8000u16];
    let mut out = std::vec![0u8; 3];
    gray16_to_rgb_row::<false>(&y, &mut out, 1, true);
    assert_eq!(&out[0..3], &[128, 128, 128]);
  }

  // ---- Gray16 u16-output limited-range tests (i32 overflow regression) ----
  //
  // The native-u16 limited-range rescale `(y - black) * max_native` overflows
  // i32 at BITS=16: `(60160 - 4096) * 65535 ≈ 3.67e9` > `i32::MAX`. Math runs
  // in i64 to keep the product safe. These tests exercise the boundary
  // values (black, white, over-white) end-to-end.

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_to_rgb_u16_limited_range_black() {
    let y: std::vec::Vec<u16> = std::vec![4096u16]; // limited-range black
    let mut out = std::vec![0u16; 3];
    gray16_to_rgb_u16_row::<false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[0, 0, 0]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_to_rgb_u16_limited_range_white() {
    let y: std::vec::Vec<u16> = std::vec![60160u16]; // limited-range white
    let mut out = std::vec![0u16; 3];
    gray16_to_rgb_u16_row::<false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[65535, 65535, 65535]);
  }

  #[test]
  fn gray16_to_rgb_u16_limited_range_over_white_clamps() {
    // Over-white (Y > 60160) is clamped to max_native=65535.
    let y: std::vec::Vec<u16> = std::vec![65535u16];
    let mut out = std::vec![0u16; 3];
    gray16_to_rgb_u16_row::<false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[65535, 65535, 65535]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_to_rgba_u16_limited_range_black_and_white() {
    let y: std::vec::Vec<u16> = std::vec![4096u16, 60160u16];
    let mut out = std::vec![0u16; 8];
    gray16_to_rgba_u16_row::<false>(&y, &mut out, 2, false);
    assert_eq!(&out[0..3], &[0, 0, 0]);
    assert_eq!(out[3], 0xFFFF);
    assert_eq!(&out[4..7], &[65535, 65535, 65535]);
    assert_eq!(out[7], 0xFFFF);
  }

  // ---- Original tests (now with full_range=true) ----

  #[test]
  #[cfg(target_endian = "little")]
  fn gray_n_to_rgb_10bit_downshifts() {
    // 10-bit: 1023 >> 2 = 255; 0 >> 2 = 0; 512 >> 2 = 128
    let y: std::vec::Vec<u16> = std::vec![0, 512, 1023];
    let mut out = std::vec![0u8; 9];
    gray_n_to_rgb_row::<10, false>(&y, &mut out, 3, true);
    assert_eq!(&out[0..3], &[0, 0, 0]);
    assert_eq!(&out[3..6], &[128, 128, 128]);
    assert_eq!(&out[6..9], &[255, 255, 255]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray_n_to_rgb_u16_10bit_masks() {
    // Upper bits should be masked out: 0xFFFF & 0x03FF = 0x03FF = 1023
    let y: std::vec::Vec<u16> = std::vec![0xFFFF, 512, 0];
    let mut out = std::vec![0u16; 9];
    gray_n_to_rgb_u16_row::<10, false>(&y, &mut out, 3, true);
    assert_eq!(&out[0..3], &[1023, 1023, 1023]);
    assert_eq!(&out[3..6], &[512, 512, 512]);
    assert_eq!(&out[6..9], &[0, 0, 0]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray_n_to_hsv_h0_s0() {
    let y: std::vec::Vec<u16> = std::vec![512u16]; // 512 >> 2 = 128
    let mut h = std::vec![0xFFu8; 1];
    let mut s = std::vec![0xFFu8; 1];
    let mut v = std::vec![0u8; 1];
    gray_n_to_hsv_row::<10, false>(&y, &mut h, &mut s, &mut v, 1, true);
    assert_eq!(h[0], 0);
    assert_eq!(s[0], 0);
    assert_eq!(v[0], 128);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_to_rgb_downshifts_8() {
    let y: std::vec::Vec<u16> = std::vec![0, 0x8000, 0xFFFF];
    let mut out = std::vec![0u8; 9];
    gray16_to_rgb_row::<false>(&y, &mut out, 3, true);
    assert_eq!(&out[0..3], &[0, 0, 0]);
    assert_eq!(&out[3..6], &[0x80, 0x80, 0x80]);
    assert_eq!(&out[6..9], &[0xFF, 0xFF, 0xFF]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_to_luma_u16_identity() {
    let y: std::vec::Vec<u16> = std::vec![0, 1000, 65535];
    let mut out = std::vec![0u16; 3];
    gray16_to_luma_u16_row::<false>(&y, &mut out, 3);
    assert_eq!(out.as_slice(), &[0, 1000, 65535]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray16_to_rgba_u16_opaque() {
    let y: std::vec::Vec<u16> = std::vec![12345u16];
    let mut out = std::vec![0u16; 4];
    gray16_to_rgba_u16_row::<false>(&y, &mut out, 1, true);
    assert_eq!(&out[0..4], &[12345, 12345, 12345, 0xFFFF]);
  }

  #[test]
  fn gray_n_to_luma_u16_10bit_masks() {
    let y: std::vec::Vec<u16> = std::vec![0xFFFF]; // should mask to 1023
    let mut out = std::vec![0u16; 1];
    gray_n_to_luma_u16_row::<10, false>(&y, &mut out, 1);
    assert_eq!(out[0], 1023);
  }

  // ---- Gray9 limited-range tests ----

  #[test]
  #[cfg(target_endian = "little")]
  fn gray9_limited_range_black() {
    // 9-bit black = 16 << 1 = 32
    let y: std::vec::Vec<u16> = std::vec![32u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<9, false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[0, 0, 0]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray9_limited_range_white() {
    // 9-bit white = 235 << 1 = 470
    let y: std::vec::Vec<u16> = std::vec![470u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<9, false>(&y, &mut out, 1, false);
    assert_eq!(&out[0..3], &[255, 255, 255]);
  }

  #[test]
  #[cfg(target_endian = "little")]
  fn gray9_full_range_pass_through() {
    // 9-bit full range: value 256 >> 1 = 128
    let y: std::vec::Vec<u16> = std::vec![256u16];
    let mut out = std::vec![0u8; 3];
    gray_n_to_rgb_row::<9, false>(&y, &mut out, 1, true);
    assert_eq!(&out[0..3], &[128, 128, 128]);
  }

  // ---- BE parity tests: gray_n (Gray9-14) -----------------------------------
  // Pattern: construct LE input, byte-swap to produce BE input, call with
  // BE=true, assert output equals LE-input run output.

  #[test]
  fn gray10_be_parity_rgb() {
    // LE value 512 >> 2 = 128. BE encoding: 512 = 0x0200, BE bytes = [0x02, 0x00].
    let le: std::vec::Vec<u16> = std::vec![512u16];
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut out_le = std::vec![0u8; 3];
    let mut out_be = std::vec![0u8; 3];
    gray_n_to_rgb_row::<10, false>(&le, &mut out_le, 1, true);
    gray_n_to_rgb_row::<10, true>(&be, &mut out_be, 1, true);
    assert_eq!(out_le, out_be, "BE and LE gray10 rgb outputs must match");
  }

  #[test]
  fn gray10_be_parity_rgba() {
    let le: std::vec::Vec<u16> = std::vec![768u16]; // 768 >> 2 = 192
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut out_le = std::vec![0u8; 4];
    let mut out_be = std::vec![0u8; 4];
    gray_n_to_rgba_row::<10, false>(&le, &mut out_le, 1, true);
    gray_n_to_rgba_row::<10, true>(&be, &mut out_be, 1, true);
    assert_eq!(out_le, out_be, "BE and LE gray10 rgba outputs must match");
  }

  #[test]
  fn gray10_be_parity_luma() {
    let le: std::vec::Vec<u16> = std::vec![256u16]; // 256 >> 2 = 64
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut out_le = std::vec![0u8; 1];
    let mut out_be = std::vec![0u8; 1];
    gray_n_to_luma_row::<10, false>(&le, &mut out_le, 1);
    gray_n_to_luma_row::<10, true>(&be, &mut out_be, 1);
    assert_eq!(out_le, out_be, "BE and LE gray10 luma outputs must match");
  }

  #[test]
  fn gray10_be_parity_luma_u16() {
    let le: std::vec::Vec<u16> = std::vec![512u16];
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut out_le = std::vec![0u16; 1];
    let mut out_be = std::vec![0u16; 1];
    gray_n_to_luma_u16_row::<10, false>(&le, &mut out_le, 1);
    gray_n_to_luma_u16_row::<10, true>(&be, &mut out_be, 1);
    assert_eq!(
      out_le, out_be,
      "BE and LE gray10 luma_u16 outputs must match"
    );
  }

  // ---- BE parity tests: gray16 -----------------------------------------------

  #[test]
  fn gray16_be_parity_rgb() {
    // LE value 0x8000 >> 8 = 128.
    let le: std::vec::Vec<u16> = std::vec![0x8000u16];
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut out_le = std::vec![0u8; 3];
    let mut out_be = std::vec![0u8; 3];
    gray16_to_rgb_row::<false>(&le, &mut out_le, 1, true);
    gray16_to_rgb_row::<true>(&be, &mut out_be, 1, true);
    assert_eq!(out_le, out_be, "BE and LE gray16 rgb outputs must match");
  }

  #[test]
  fn gray16_be_parity_rgba() {
    let le: std::vec::Vec<u16> = std::vec![0xC000u16]; // 0xC0 = 192
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut out_le = std::vec![0u8; 4];
    let mut out_be = std::vec![0u8; 4];
    gray16_to_rgba_row::<false>(&le, &mut out_le, 1, true);
    gray16_to_rgba_row::<true>(&be, &mut out_be, 1, true);
    assert_eq!(out_le, out_be, "BE and LE gray16 rgba outputs must match");
  }

  #[test]
  fn gray16_be_parity_luma() {
    let le: std::vec::Vec<u16> = std::vec![0x4000u16]; // 0x40 = 64
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut out_le = std::vec![0u8; 1];
    let mut out_be = std::vec![0u8; 1];
    gray16_to_luma_row::<false>(&le, &mut out_le, 1);
    gray16_to_luma_row::<true>(&be, &mut out_be, 1);
    assert_eq!(out_le, out_be, "BE and LE gray16 luma outputs must match");
  }

  #[test]
  fn gray16_be_parity_luma_u16() {
    // For gray16_to_luma_u16_row with BE=true, swap_bytes is applied.
    // LE: 0x1234. BE encoding of that value: swap bytes → 0x3412.
    // After BE kernel processes 0x3412 with swap_bytes → 0x1234. Output = 0x1234.
    let le_val: u16 = 0x1234;
    let le: std::vec::Vec<u16> = std::vec![le_val];
    let be: std::vec::Vec<u16> = le.iter().map(|v| v.swap_bytes()).collect();
    let mut out_le = std::vec![0u16; 1];
    let mut out_be = std::vec![0u16; 1];
    gray16_to_luma_u16_row::<false>(&le, &mut out_le, 1);
    gray16_to_luma_u16_row::<true>(&be, &mut out_be, 1);
    assert_eq!(
      out_le, out_be,
      "BE and LE gray16 luma_u16 outputs must match"
    );
  }
}
