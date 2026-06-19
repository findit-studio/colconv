//! High-bit-depth 4:2:2 / 4:4:0 `MixedSinker` impls, split per
//! sub-format so no single source exceeds ~1.5 KLoC: Yuv422p9/10/12/14/16
//! + Yuv440p10/12 (`yuv-planar`) + P210/P212/P216 (`yuv-semi-planar`).

#[cfg(feature = "yuv-semi-planar")]
mod p2xx;
#[cfg(feature = "yuv-planar")]
mod yuv422p;
#[cfg(feature = "yuv-planar")]
mod yuv440p;
