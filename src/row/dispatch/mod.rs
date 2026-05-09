//! Public row-dispatcher submodules. The dispatchers were extracted
//! from `row::mod` here so the parent module stays focused on
//! shared helpers, runtime CPU feature detection, and crate-private
//! `arch` / `scalar` glue.
//!
//! Submodules are `pub(super)` — they are re-exported via `pub use` in
//! `row::mod`, so the public API still appears at `crate::row::*`
//! (e.g. `crate::row::yuv_420_to_rgb_row`). Sinker code that needs
//! dispatcher functions reaches them through those same re-exports.
//! Callers see no API change from the split.

pub(crate) mod alpha_extract;
pub(super) mod ayuv64;
pub(super) mod bayer;
pub(super) mod gray;
pub(super) mod grayf32;
pub(super) mod legacy_rgb;
pub(super) mod mono1bit;
pub(super) mod nv;
pub(super) mod packed_rgb_16bit;
pub(super) mod packed_yuv411;
pub(super) mod packed_yuv422;
pub(super) mod pal8;
pub(super) mod planar_gbr;
pub(super) mod planar_gbr_float;
pub(super) mod planar_gbr_high_bit;
pub(super) mod pn;
pub(super) mod rgb_f16_ops;
pub(super) mod rgb_float_ops;
pub(super) mod rgb_ops;
pub(super) mod v210;
pub(super) mod v30x;
pub(super) mod v410;
pub(super) mod vuya;
pub(super) mod vuyx;
pub(super) mod xv36;
#[cfg(any(feature = "std", feature = "alloc"))]
pub(super) mod xyz12;
pub(super) mod y210;
pub(super) mod y212;
pub(super) mod y216;
pub(crate) mod y_plane_to_luma_u16;
pub(super) mod ya16;
pub(super) mod ya8;
pub(super) mod yuv411p;
pub(super) mod yuv420;
pub(super) mod yuv444;
pub(super) mod yuva;

// Dispatch-level BE/LE parity tests for the high-bit YUV planar and
// P-format families (codex round-3 follow-up on `feat/be-yuv-hb`). The
// per-format dispatchers in `yuv420::*`, `yuv444::*`, and `pn::*` each
// gained `_endian` entry points; this module asserts that the BE path
// is reachable and produces byte-identical output to the LE path on
// matching fixtures.
#[cfg(all(test, feature = "std"))]
mod be_yuv_hb_parity_tests;
