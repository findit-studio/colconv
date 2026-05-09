//! NEON kernels for the Tier 5.25 packed YUV 4:1:1 source (UYYVYY411).
//!
//! P3 legacy DV format — the per-block decode (one chroma pair shared
//! by 4 luma samples) doesn't map cleanly onto the existing
//! `vld2q_u8` / `vuzp_u8` packed-4:2:2 NEON shape, and the format is
//! seen rarely enough in practice that the engineering effort to
//! write a hand-tuned 4:1:1 NEON pipeline isn't justified at this
//! tier. The dispatcher routes here under
//! `#[target_feature(enable = "neon")]` and the kernel forwards to
//! the byte-identical scalar reference, keeping the API shape
//! consistent with every other format and preserving the
//! NEON-availability gate at the call site.

use crate::{ColorMatrix, row::scalar};

/// NEON UYYVYY411 → packed RGB. Semantics match
/// [`scalar::uyyvyy411_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 3 == 0` (4:1:1 chroma group).
/// 3. `packed.len() >= width * 3 / 2`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyyvyy411_to_rgb_row(
  packed: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  scalar::uyyvyy411_to_rgb_row(packed, rgb_out, width, matrix, full_range);
}

/// NEON UYYVYY411 → packed RGBA (alpha = 0xFF).
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_rgb_row`] with `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyyvyy411_to_rgba_row(
  packed: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  scalar::uyyvyy411_to_rgba_row(packed, rgba_out, width, matrix, full_range);
}

/// NEON UYYVYY411 luma extraction — Y bytes at offsets 1, 2, 4, 5 of
/// each 6-byte block.
///
/// # Safety
///
/// 1. **NEON must be available on the current CPU.**
/// 2. `width & 3 == 0`.
/// 3. `packed.len() >= width * 3 / 2`, `luma_out.len() >= width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyyvyy411_to_luma_row(packed: &[u8], luma_out: &mut [u8], width: usize) {
  scalar::uyyvyy411_to_luma_row(packed, luma_out, width);
}

/// NEON UYYVYY411 luma extraction → u16 (zero-extended).
///
/// # Safety
///
/// Same contract as [`uyyvyy411_to_luma_row`] with the output as
/// `&mut [u16]` of `width` elements.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn uyyvyy411_to_luma_u16_row(packed: &[u8], out: &mut [u16], width: usize) {
  scalar::uyyvyy411_to_luma_u16_row(packed, out, width);
}
