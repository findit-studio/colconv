//! Validated source-frame types.
//!
//! Each pixel family has its own frame struct carrying the backing
//! plane slice(s), pixel dimensions, and byte strides. Construction
//! validates strides vs. widths and that each plane covers its
//! declared area.

// Per-format-family submodules. Each file groups one logical pixel
// family so no single source ends up over ~1.5k lines, mirroring the
// split already done for `yuva/`, `packed_rgb_8bit`, etc.
mod bayer;
mod gray;
mod legacy_rgb;
mod mono1bit;
mod packed_rgb_10bit;
mod packed_rgb_8bit;
mod packed_rgb_f16;
mod packed_rgb_float;
mod packed_yuv_4_4_4;
mod packed_yuv_8bit;
mod pal8;
mod planar_8bit;
mod planar_gbr_8bit;
mod planar_gbr_float;
mod planar_gbr_high_bit;
mod semi_planar_8bit;
mod subsampled_high_bit_planar;
mod subsampled_high_bit_pn;
mod v210;
mod y2xx;
mod yuva;

pub use bayer::*;
pub use gray::*;
pub use legacy_rgb::{
  Bgr444Frame, Bgr555Frame, Bgr565Frame, LegacyRgbFrameError, Rgb444Frame, Rgb555Frame, Rgb565Frame,
};
pub use mono1bit::*;
pub use packed_rgb_8bit::*;
pub use packed_rgb_10bit::*;
pub use packed_rgb_f16::*;
pub use packed_rgb_float::*;
pub use packed_yuv_4_4_4::*;
pub use packed_yuv_8bit::*;
pub use pal8::*;
pub use planar_8bit::*;
pub use planar_gbr_8bit::*;
pub use planar_gbr_float::*;
pub use planar_gbr_high_bit::*;
pub use semi_planar_8bit::*;
pub use subsampled_high_bit_planar::*;
pub use subsampled_high_bit_pn::*;
pub use v210::*;
pub use y2xx::*;
pub use yuva::*;

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests;
