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
pub(super) mod y210;
pub(super) mod y212;
pub(super) mod y216;
pub(crate) mod y_plane_to_luma_u16;
pub(super) mod ya16;
pub(super) mod ya8;
pub(super) mod yuv420;
pub(super) mod yuv444;
pub(super) mod yuva;
