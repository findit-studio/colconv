//! NEON kernels for 16-bit packed RGB/BGR/RGBA/BGRA sources (Tier 8 finish).
//!
// Kernels are wired into the dispatcher in the SIMD dispatch task; suppress
// dead_code until then.
#![allow(dead_code)]
//!
//! ## Format layouts
//!
//! | Format | Elements per pixel | Channel order in memory |
//! |--------|--------------------|------------------------|
//! | Rgb48  | 3 u16              | R, G, B                |
//! | Bgr48  | 3 u16              | B, G, R                |
//! | Rgba64 | 4 u16              | R, G, B, A             |
//! | Bgra64 | 4 u16              | B, G, R, A             |
//!
//! ## Per-format SIMD strategy (8 pixels per SIMD iteration)
//!
//! - **Rgb48 / Bgr48:** `vld3q_u16(src_ptr)` → `uint16x8x3_t(ch0, ch1, ch2)`
//!   deinterleaves 8 pixels (24 u16 elements). For Bgr48, the channel fields
//!   are reordered on store (`.0` ↔ `.2`) so output is always R-first.
//! - **Rgba64 / Bgra64:** `vld4q_u16(src_ptr)` → `uint16x8x4_t(ch0, ch1, ch2, ch3)`.
//!   For Bgra64, `ch0` = B and `ch2` = R (swapped on store).
//!
//! ## Big-endian support
//!
//! Every public kernel accepts `<const BE: bool>`. When `BE = true`, each
//! per-channel `uint16x8_t` vector produced by `vld3q_u16`/`vld4q_u16` is
//! byte-swapped via `byteswap_u16x8::<BE>` before any channel math. On LE
//! targets (all current AArch64 hardware) the helper is a no-op and emits
//! zero extra instructions.
//!
//! ## Depth conversion
//!
//! - **u16 → u8:** `vshrn_n_u16::<8>(v)` — high-byte extraction, matching
//!   `scalar`'s `(v >> 8) as u8`.
//! - **u16 → u16:** native passthrough (`vst3q_u16` / `vst4q_u16` directly).
//!
//! ## Scalar tail
//!
//! All kernels handle `width % 8` remaining pixels via the scalar reference.

use core::arch::aarch64::*;

use crate::row::scalar;

// ---- endian byte-swap helper ------------------------------------------------

/// Byte-swap every u16 lane in `v` when `BE = true`; no-op otherwise.
///
/// Implemented as `vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v)))`,
/// the same transform used inside `load_be_u16x8` in the NEON endian module.
///
/// # Safety
///
/// Caller must have NEON enabled.
#[inline(always)]
unsafe fn byteswap_u16x8<const BE: bool>(v: uint16x8_t) -> uint16x8_t {
  if BE {
    unsafe { vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v))) }
  } else {
    v
  }
}

// =============================================================================
// Rgb48 (R, G, B — 3 u16 elements per pixel)
// =============================================================================

/// NEON Rgb48 → packed u8 RGB. 8 pixels per SIMD iteration.
///
/// `vld3q_u16` deinterleaves into `(R, G, B)` u16x8; `vshrn_n_u16::<8>`
/// narrows each channel; `vst3_u8` interleaves back.
///
/// When `BE = true` each channel vector is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgb48_to_rgb_row<const BE: bool>(
  rgb48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x3_t = vld3q_u16(rgb48.as_ptr().add(x * 3));
      let r8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.0));
      let g8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.1));
      let b8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.2));
      vst3_u8(rgb_out.as_mut_ptr().add(x * 3), uint8x8x3_t(r8, g8, b8));
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgb_row::<BE>(&rgb48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Rgb48 → packed u8 RGBA. 8 pixels per SIMD iteration. Alpha forced to 0xFF.
///
/// When `BE = true` each channel vector is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgb48_to_rgba_row<const BE: bool>(
  rgb48: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdup_n_u8(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x3_t = vld3q_u16(rgb48.as_ptr().add(x * 3));
      let r8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.0));
      let g8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.1));
      let b8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.2));
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r8, g8, b8, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgba_row::<BE>(&rgb48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON Rgb48 → native-depth u16 RGB. 8 pixels per SIMD iteration.
///
/// `vld3q_u16` deinterleaves, `vst3q_u16` reinterleaves.
/// When `BE = true` each channel is byte-swapped to host-native order before storing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgb48_to_rgb_u16_row<const BE: bool>(
  rgb48: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x3_t = vld3q_u16(rgb48.as_ptr().add(x * 3));
      vst3q_u16(
        rgb_out.as_mut_ptr().add(x * 3),
        uint16x8x3_t(
          byteswap_u16x8::<BE>(px.0),
          byteswap_u16x8::<BE>(px.1),
          byteswap_u16x8::<BE>(px.2),
        ),
      );
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgb_u16_row::<BE>(&rgb48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Rgb48 → native-depth u16 RGBA. 8 pixels per SIMD iteration. Alpha forced to 0xFFFF.
///
/// When `BE = true` each channel is byte-swapped to host-native order before storing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgb48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgb48_to_rgba_u16_row<const BE: bool>(
  rgb48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb48.len() >= width * 3, "rgb48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdupq_n_u16(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x3_t = vld3q_u16(rgb48.as_ptr().add(x * 3));
      vst4q_u16(
        rgba_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(
          byteswap_u16x8::<BE>(px.0),
          byteswap_u16x8::<BE>(px.1),
          byteswap_u16x8::<BE>(px.2),
          alpha,
        ),
      );
      x += 8;
    }
    if x < width {
      scalar::rgb48_to_rgba_u16_row::<BE>(&rgb48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Bgr48 (B, G, R — 3 u16 elements per pixel)
// =============================================================================

/// NEON Bgr48 → packed u8 RGB. 8 pixels per SIMD iteration.
///
/// `vld3q_u16` deinterleaves into `(B, G, R)` u16x8; channels are swapped
/// (`px.2` = R, `px.0` = B) in the `vst3_u8` call to produce R-first output.
/// When `BE = true` each channel is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_bgr48_to_rgb_row<const BE: bool>(
  bgr48: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      // px.0 = B, px.1 = G, px.2 = R (source BGR order)
      let px: uint16x8x3_t = vld3q_u16(bgr48.as_ptr().add(x * 3));
      let r8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.2)); // R (was at position 2)
      let g8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.1)); // G (unchanged)
      let b8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.0)); // B (was at position 0)
      vst3_u8(rgb_out.as_mut_ptr().add(x * 3), uint8x8x3_t(r8, g8, b8));
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgb_row::<BE>(&bgr48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Bgr48 → packed u8 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap on output; alpha forced to 0xFF.
/// When `BE = true` each channel is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_bgr48_to_rgba_row<const BE: bool>(
  bgr48: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdup_n_u8(0xFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x3_t = vld3q_u16(bgr48.as_ptr().add(x * 3));
      let r8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.2));
      let g8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.1));
      let b8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.0));
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r8, g8, b8, alpha),
      );
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgba_row::<BE>(&bgr48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON Bgr48 → native-depth u16 RGB. 8 pixels per SIMD iteration.
/// B↔R swap: `px.2` → position 0 (R), `px.0` → position 2 (B).
/// When `BE = true` each channel is byte-swapped to host-native order.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_bgr48_to_rgb_u16_row<const BE: bool>(
  bgr48: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x3_t = vld3q_u16(bgr48.as_ptr().add(x * 3));
      // Swap B↔R: store (R=px.2, G=px.1, B=px.0)
      vst3q_u16(
        rgb_out.as_mut_ptr().add(x * 3),
        uint16x8x3_t(
          byteswap_u16x8::<BE>(px.2),
          byteswap_u16x8::<BE>(px.1),
          byteswap_u16x8::<BE>(px.0),
        ),
      );
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgb_u16_row::<BE>(&bgr48[x * 3..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Bgr48 → native-depth u16 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap; alpha forced to 0xFFFF.
/// When `BE = true` each channel is byte-swapped to host-native order.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgr48.len() >= width * 3`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_bgr48_to_rgba_u16_row<const BE: bool>(
  bgr48: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgr48.len() >= width * 3, "bgr48 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let alpha = vdupq_n_u16(0xFFFF);
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x3_t = vld3q_u16(bgr48.as_ptr().add(x * 3));
      // Store (R=px.2, G=px.1, B=px.0, A=0xFFFF)
      vst4q_u16(
        rgba_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(
          byteswap_u16x8::<BE>(px.2),
          byteswap_u16x8::<BE>(px.1),
          byteswap_u16x8::<BE>(px.0),
          alpha,
        ),
      );
      x += 8;
    }
    if x < width {
      scalar::bgr48_to_rgba_u16_row::<BE>(&bgr48[x * 3..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Rgba64 (R, G, B, A — 4 u16 elements per pixel)
// =============================================================================

/// NEON Rgba64 → packed u8 RGB. 8 pixels per SIMD iteration. Alpha discarded.
///
/// `vld4q_u16` deinterleaves into `(R, G, B, A)` u16x8; R/G/B narrowed;
/// `vst3_u8` writes only 3 channels.
/// When `BE = true` each channel is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgba64_to_rgb_row<const BE: bool>(
  rgba64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x4_t = vld4q_u16(rgba64.as_ptr().add(x * 4));
      let r8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.0));
      let g8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.1));
      let b8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.2));
      // Alpha (px.3) discarded.
      vst3_u8(rgb_out.as_mut_ptr().add(x * 3), uint8x8x3_t(r8, g8, b8));
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgb_row::<BE>(&rgba64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Rgba64 → packed u8 RGBA. 8 pixels per SIMD iteration. Source alpha passes through.
///
/// All 4 channels narrowed via `vshrn_n_u16::<8>`.
/// When `BE = true` each channel is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgba64_to_rgba_row<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x4_t = vld4q_u16(rgba64.as_ptr().add(x * 4));
      let r8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.0));
      let g8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.1));
      let b8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.2));
      let a8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.3)); // source alpha depth-converted
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r8, g8, b8, a8),
      );
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgba_row::<BE>(&rgba64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON Rgba64 → native-depth u16 RGB. 8 pixels per SIMD iteration. Alpha discarded.
///
/// `vld4q_u16` deinterleaves; `vst3q_u16` writes R, G, B channels only.
/// When `BE = true` each channel is byte-swapped to host-native order.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgba64_to_rgb_u16_row<const BE: bool>(
  rgba64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x4_t = vld4q_u16(rgba64.as_ptr().add(x * 4));
      // Alpha (px.3) discarded.
      vst3q_u16(
        rgb_out.as_mut_ptr().add(x * 3),
        uint16x8x3_t(
          byteswap_u16x8::<BE>(px.0),
          byteswap_u16x8::<BE>(px.1),
          byteswap_u16x8::<BE>(px.2),
        ),
      );
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgb_u16_row::<BE>(&rgba64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Rgba64 → native-depth u16 RGBA. 8 pixels per SIMD iteration.
///
/// `vld4q_u16` deinterleaves; `vst4q_u16` reinterleaves — source alpha preserved.
/// When `BE = true` each channel is byte-swapped to host-native order.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgba64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_rgba64_to_rgba_u16_row<const BE: bool>(
  rgba64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgba64.len() >= width * 4, "rgba64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x4_t = vld4q_u16(rgba64.as_ptr().add(x * 4));
      vst4q_u16(
        rgba_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(
          byteswap_u16x8::<BE>(px.0),
          byteswap_u16x8::<BE>(px.1),
          byteswap_u16x8::<BE>(px.2),
          byteswap_u16x8::<BE>(px.3),
        ),
      );
      x += 8;
    }
    if x < width {
      scalar::rgba64_to_rgba_u16_row::<BE>(&rgba64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

// =============================================================================
// Bgra64 (B, G, R, A — 4 u16 elements per pixel)
// =============================================================================

/// NEON Bgra64 → packed u8 RGB. 8 pixels per SIMD iteration.
/// B↔R swap; alpha discarded.
///
/// `vld4q_u16` gives `(B, G, R, A)` → store `(R=px.2, G=px.1, B=px.0)`.
/// When `BE = true` each channel is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_bgra64_to_rgb_row<const BE: bool>(
  bgra64: &[u16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      // px.0 = B, px.1 = G, px.2 = R, px.3 = A
      let px: uint16x8x4_t = vld4q_u16(bgra64.as_ptr().add(x * 4));
      let r8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.2)); // R (from position 2)
      let g8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.1)); // G (unchanged)
      let b8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.0)); // B (from position 0)
      // Alpha (px.3) discarded.
      vst3_u8(rgb_out.as_mut_ptr().add(x * 3), uint8x8x3_t(r8, g8, b8));
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgb_row::<BE>(&bgra64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Bgra64 → packed u8 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap; source alpha passes through (narrowed via `>> 8`).
/// When `BE = true` each channel is byte-swapped before narrowing.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_bgra64_to_rgba_row<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x4_t = vld4q_u16(bgra64.as_ptr().add(x * 4));
      let r8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.2));
      let g8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.1));
      let b8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.0));
      let a8 = vshrn_n_u16::<8>(byteswap_u16x8::<BE>(px.3)); // source alpha depth-converted
      vst4_u8(
        rgba_out.as_mut_ptr().add(x * 4),
        uint8x8x4_t(r8, g8, b8, a8),
      );
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgba_row::<BE>(&bgra64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}

/// NEON Bgra64 → native-depth u16 RGB. 8 pixels per SIMD iteration.
/// B↔R swap; alpha discarded. `vld4q_u16` → `vst3q_u16(R, G, B)`.
/// When `BE = true` each channel is byte-swapped to host-native order.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgb_out.len() >= width * 3`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_bgra64_to_rgb_u16_row<const BE: bool>(
  bgra64: &[u16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x4_t = vld4q_u16(bgra64.as_ptr().add(x * 4));
      // Swap B↔R, drop alpha: store (R=px.2, G=px.1, B=px.0)
      vst3q_u16(
        rgb_out.as_mut_ptr().add(x * 3),
        uint16x8x3_t(
          byteswap_u16x8::<BE>(px.2),
          byteswap_u16x8::<BE>(px.1),
          byteswap_u16x8::<BE>(px.0),
        ),
      );
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgb_u16_row::<BE>(&bgra64[x * 4..], &mut rgb_out[x * 3..], width - x);
    }
  }
}

/// NEON Bgra64 → native-depth u16 RGBA. 8 pixels per SIMD iteration.
/// B↔R swap; source alpha preserved at position 3.
///
/// `vld4q_u16` gives `(B, G, R, A)` → `vst4q_u16(R=px.2, G=px.1, B=px.0, A=px.3)`.
/// When `BE = true` each channel is byte-swapped to host-native order.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `bgra64.len() >= width * 4`.
/// 3. `rgba_out.len() >= width * 4`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_bgra64_to_rgba_u16_row<const BE: bool>(
  bgra64: &[u16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(bgra64.len() >= width * 4, "bgra64 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let mut x = 0usize;
    while x + 8 <= width {
      let px: uint16x8x4_t = vld4q_u16(bgra64.as_ptr().add(x * 4));
      // Swap B↔R, preserve A: store (R=px.2, G=px.1, B=px.0, A=px.3)
      vst4q_u16(
        rgba_out.as_mut_ptr().add(x * 4),
        uint16x8x4_t(
          byteswap_u16x8::<BE>(px.2),
          byteswap_u16x8::<BE>(px.1),
          byteswap_u16x8::<BE>(px.0),
          byteswap_u16x8::<BE>(px.3),
        ),
      );
      x += 8;
    }
    if x < width {
      scalar::bgra64_to_rgba_u16_row::<BE>(&bgra64[x * 4..], &mut rgba_out[x * 4..], width - x);
    }
  }
}
