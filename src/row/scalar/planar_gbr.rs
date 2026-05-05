//! Scalar reference kernels for planar GBR sources (Tier 10).
//!
//! Three flavours, all `width`-keyed:
//!
//! - [`gbr_to_rgb_row`] — interleave G/B/R planes into packed `R, G, B`
//!   bytes. Reorders G/B/R → R/G/B per pixel (no chroma matrix).
//! - [`gbra_to_rgba_row`] — same plus alpha plane → `R, G, B, A`.
//! - [`gbr_to_rgba_opaque_row`] — same interleave into `R, G, B, A` with α
//!   forced to `0xFF` (for `Gbrp` sources with no alpha plane).
//!
//! HSV reuses the existing `rgb_to_hsv_row` kernel after a staged RGB
//! pass via `gbr_to_rgb_row`.

/// Interleaves three planar G/B/R rows into packed `R, G, B` bytes
/// (output order is **R, G, B** per FFmpeg packed-RGB convention).
///
/// # Panics (debug builds)
///
/// - `g.len() >= width`
/// - `b.len() >= width`
/// - `r.len() >= width`
/// - `rgb_out.len() >= 3 * width`
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgb_row(g: &[u8], b: &[u8], r: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");
  for x in 0..width {
    let dst = x * 3;
    rgb_out[dst] = r[x];
    rgb_out[dst + 1] = g[x];
    rgb_out[dst + 2] = b[x];
  }
}

/// Interleaves four planar G/B/R/A rows into packed `R, G, B, A`
/// bytes. Alpha is sourced from the `a` plane (real per-pixel α).
///
/// # Panics (debug builds)
///
/// - `g.len() >= width`
/// - `b.len() >= width`
/// - `r.len() >= width`
/// - `a.len() >= width`
/// - `rgba_out.len() >= 4 * width`
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbra_to_rgba_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  a: &[u8],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(a.len() >= width, "a row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = r[x];
    rgba_out[dst + 1] = g[x];
    rgba_out[dst + 2] = b[x];
    rgba_out[dst + 3] = a[x];
  }
}

/// Interleaves three planar G/B/R rows into packed `R, G, B` bytes
/// **with α appended** as a constant `0xFF`. Used for `Gbrp` sources
/// (no alpha plane) when `with_rgba` is requested.
///
/// # Panics (debug builds)
///
/// - `g.len() >= width`
/// - `b.len() >= width`
/// - `r.len() >= width`
/// - `rgba_out.len() >= 4 * width`
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn gbr_to_rgba_opaque_row(
  g: &[u8],
  b: &[u8],
  r: &[u8],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(g.len() >= width, "g row too short");
  debug_assert!(b.len() >= width, "b row too short");
  debug_assert!(r.len() >= width, "r row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");
  for x in 0..width {
    let dst = x * 4;
    rgba_out[dst] = r[x];
    rgba_out[dst + 1] = g[x];
    rgba_out[dst + 2] = b[x];
    rgba_out[dst + 3] = 0xFF;
  }
}

// NOTE: `gbr_to_luma_row` / `gbr_to_luma_u16_row` are not provided as
// dedicated planar kernels — the sinker stages packed RGB into the
// existing `rgb_scratch` and dispatches to the shared
// `rgb_to_luma_row` for both u8 and u16 luma paths (the u16 path
// zero-extends the u8 result inline). This avoids an additional
// kernel family while reusing the already-SIMD-accelerated
// `rgb_to_luma_row` infrastructure across packed-RGB and planar-GBR
// sources.
