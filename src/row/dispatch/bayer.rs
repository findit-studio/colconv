//! Bayer dispatchers (`bayer_to_rgb_row`, `bayer16_to_rgb_row`,
//! `bayer16_to_rgb_u16_row`) extracted from `row::mod` for organization.
//!
//! `use_simd` is currently a no-op for all Bayer paths — they route to
//! scalar regardless. Per-arch SIMD backends ship in a follow-up; the
//! parameter is wired through so callers don't have to touch their
//! call sites when SIMD lands.

use crate::row::{assert_color_transform_well_formed, rgb_row_bytes, rgb_row_elems, scalar};

/// Converts one row of an 8-bit Bayer plane to packed RGB.
///
/// Dispatches to the best available backend for the current target.
/// See [`scalar::bayer_to_rgb_row`] for the full semantic specification
/// (bilinear demosaic geometry, edge handling, output layout).
///
/// `above` / `mid` / `below` are row-aligned slices into the source
/// Bayer plane via the **mirror-by-2** boundary contract: at the
/// top edge the caller supplies `above = mid_row(1)`, at the bottom
/// edge `below = mid_row(h - 2)`; replicate fallback only when
/// `height < 2`. See [`crate::raw::BayerRow::above`] for the full
/// rationale (CFA-parity preservation across boundaries).
/// `above` / `mid` / `below` must all be the same length — that
/// length is the row's pixel width.
///
/// `m` is the precomputed `CCM · diag(wb)` 3×3 transform. Every
/// element must be finite (not NaN, not ±∞); the dispatcher
/// asserts this at the boundary so future unsafe SIMD kernels can
/// trust the contract.
///
/// `rgb_out` must have at least `3 * mid.len()` bytes.
///
/// **`use_simd` is currently a no-op.** All Bayer paths run the
/// scalar reference today; per-arch SIMD backends (NEON / SSE4.1 /
/// AVX2 / AVX-512 / wasm simd128) ship in a follow-up. The
/// parameter is wired through `MixedSinker` and the public
/// dispatchers now so callers don't have to touch their call sites
/// when SIMD lands.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bayer_to_rgb_row(
  above: &[u8],
  mid: &[u8],
  below: &[u8],
  row_parity: u32,
  pattern: crate::raw::BayerPattern,
  demosaic: crate::raw::BayerDemosaic,
  m: &[[f32; 3]; 3],
  rgb_out: &mut [u8],
  _use_simd: bool,
) {
  // Release-mode preflight: future unsafe SIMD backends will rely on
  // these invariants for bounds-free pointer arithmetic, so we
  // validate here rather than only via `debug_assert!` inside the
  // scalar kernel. Same pattern as `yuv_420_to_rgb_row`.
  let width = mid.len();
  assert_eq!(above.len(), width, "above row length must match mid");
  assert_eq!(below.len(), width, "below row length must match mid");
  let rgb_min = rgb_row_bytes(width);
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");
  assert_color_transform_well_formed(m);

  scalar::bayer_to_rgb_row(above, mid, below, row_parity, pattern, demosaic, m, rgb_out);
}

/// Converts one row of a 10/12/14/16-bit **low-packed** Bayer
/// plane to packed `u8` RGB, reading the source `u16` samples in the
/// wire byte order selected by `BE`.
///
/// `BITS` ∈ {10, 12, 14, 16}; samples are low-packed `u16` (active
/// values in the low `BITS` bits, range `[0, (1 << BITS) - 1]`) in
/// the wire byte order selected by `BE` (`false` = little-endian;
/// `true` = big-endian). Direct row-API callers are responsible for
/// upholding the low-packed contract; samples whose *logical* value
/// (after byte-order normalization) exceeds `(1 << BITS) - 1` produce
/// defined-but-saturated output (no panic, no UB). The walker
/// [`crate::frame::bayer16_to_endian`] never sees out-of-range input
/// because [`crate::frame::BayerFrame16::try_new`] validates every
/// active sample at frame-construction time.
///
/// `m` is the unscaled `CCM · diag(wb)` — the kernel bakes the
/// input→u8 rescale (`255 / ((1 << BITS) - 1)`) at output time.
/// `above` / `mid` / `below` must all be the same length;
/// `rgb_out` must have at least `3 * mid.len()` bytes.
///
/// **`use_simd` is currently a no-op** (see
/// [`bayer_to_rgb_row`] for the deferred-SIMD note).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bayer16_to_rgb_row_endian<const BITS: u32, const BE: bool>(
  above: &[u16],
  mid: &[u16],
  below: &[u16],
  row_parity: u32,
  pattern: crate::raw::BayerPattern,
  demosaic: crate::raw::BayerDemosaic,
  m: &[[f32; 3]; 3],
  rgb_out: &mut [u8],
  _use_simd: bool,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12 || BITS == 14 || BITS == 16,
      "bayer16_to_rgb_row_endian: BITS must be 10, 12, 14, or 16"
    )
  };
  let width = mid.len();
  assert_eq!(above.len(), width, "above row length must match mid");
  assert_eq!(below.len(), width, "below row length must match mid");
  let rgb_min = rgb_row_bytes(width);
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");
  assert_color_transform_well_formed(m);

  scalar::bayer16_to_rgb_row::<BITS, BE>(
    above, mid, below, row_parity, pattern, demosaic, m, rgb_out,
  );
}

/// LE-only wrapper around [`bayer16_to_rgb_row_endian`]; preserves the
/// pre-endian-aware single-const public signature so existing
/// little-endian callers (`bayer16_to_rgb_row::<BITS>(...)`) compile
/// unchanged. Equivalent to `bayer16_to_rgb_row_endian::<BITS, false>(...)`.
///
/// Rust forbids defaults on function-position const-generic parameters,
/// so an explicit-turbofish caller written before the endian extension
/// (`bayer16_to_rgb_row::<12>(...)`) would otherwise fail to compile if
/// `BE` were appended in place. This thin LE wrapper preserves source
/// compatibility for those call sites; BE-aware callers use
/// [`bayer16_to_rgb_row_endian`] directly. Mirrors the
/// `bayer16_to` → `bayer16_to_endian` pairing in `mediaframe`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bayer16_to_rgb_row<const BITS: u32>(
  above: &[u16],
  mid: &[u16],
  below: &[u16],
  row_parity: u32,
  pattern: crate::raw::BayerPattern,
  demosaic: crate::raw::BayerDemosaic,
  m: &[[f32; 3]; 3],
  rgb_out: &mut [u8],
  use_simd: bool,
) {
  bayer16_to_rgb_row_endian::<BITS, false>(
    above, mid, below, row_parity, pattern, demosaic, m, rgb_out, use_simd,
  );
}

/// Converts one row of a 10/12/14/16-bit **low-packed** Bayer
/// plane to packed `u16` RGB (also low-packed at `BITS`), reading the
/// source `u16` samples in the wire byte order selected by `BE`.
///
/// `BITS` ∈ {10, 12, 14, 16}. Input and output share the same
/// low-packed range `[0, (1 << BITS) - 1]` per channel — no
/// rescale, just clamp. The `u16` **input** is read in the wire byte
/// order selected by `BE` (`false` = little-endian; `true` =
/// big-endian); the `u16` **output** is always host-native.
/// `above` / `mid` / `below` must all be the same length; `rgb_out`
/// must have at least `3 * mid.len()` `u16` elements.
///
/// Direct row-API callers are responsible for upholding the
/// low-packed contract — see [`bayer16_to_rgb_row`] for the
/// full rationale on the safe path
/// ([`crate::frame::BayerFrame16::try_new`] + [`crate::frame::bayer16_to_endian`])
/// vs. the direct row API.
///
/// **`use_simd` is currently a no-op** (see
/// [`bayer_to_rgb_row`] for the deferred-SIMD note).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bayer16_to_rgb_u16_row_endian<const BITS: u32, const BE: bool>(
  above: &[u16],
  mid: &[u16],
  below: &[u16],
  row_parity: u32,
  pattern: crate::raw::BayerPattern,
  demosaic: crate::raw::BayerDemosaic,
  m: &[[f32; 3]; 3],
  rgb_out: &mut [u16],
  _use_simd: bool,
) {
  const {
    assert!(
      BITS == 10 || BITS == 12 || BITS == 14 || BITS == 16,
      "bayer16_to_rgb_u16_row_endian: BITS must be 10, 12, 14, or 16"
    )
  };
  let width = mid.len();
  assert_eq!(above.len(), width, "above row length must match mid");
  assert_eq!(below.len(), width, "below row length must match mid");
  let rgb_min = rgb_row_elems(width);
  assert!(rgb_out.len() >= rgb_min, "rgb_out row too short");
  assert_color_transform_well_formed(m);

  scalar::bayer16_to_rgb_u16_row::<BITS, BE>(
    above, mid, below, row_parity, pattern, demosaic, m, rgb_out,
  );
}

/// LE-only wrapper around [`bayer16_to_rgb_u16_row_endian`]; preserves
/// the pre-endian-aware single-const public signature so existing
/// little-endian callers (`bayer16_to_rgb_u16_row::<BITS>(...)`)
/// compile unchanged. Equivalent to
/// `bayer16_to_rgb_u16_row_endian::<BITS, false>(...)`.
///
/// See [`bayer16_to_rgb_row`] for the source-compatibility rationale
/// (function-position const generics cannot carry a default, so the
/// LE entry point is kept as a thin single-const wrapper).
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub fn bayer16_to_rgb_u16_row<const BITS: u32>(
  above: &[u16],
  mid: &[u16],
  below: &[u16],
  row_parity: u32,
  pattern: crate::raw::BayerPattern,
  demosaic: crate::raw::BayerDemosaic,
  m: &[[f32; 3]; 3],
  rgb_out: &mut [u16],
  use_simd: bool,
) {
  bayer16_to_rgb_u16_row_endian::<BITS, false>(
    above, mid, below, row_parity, pattern, demosaic, m, rgb_out, use_simd,
  );
}
