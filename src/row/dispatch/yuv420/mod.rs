//! YUV 4:2:0 dispatchers, split per source format for readability.
//!
//! - `yuv_410` — 8-bit YUV 4:1:0 → RGB / RGBA (Cinepak / Sorenson
//!   legacy). Co-located here because 4:1:0 shares the vertical-
//!   subsampling walker shape with 4:2:0.
//! - `yuv_420` — 8-bit YUV 4:2:0 → RGB / RGBA.
//! - `yuv420p9` / `yuv420p10` / `yuv420p12` / `yuv420p14` /
//!   `yuv420p16` — high-bit planar 4:2:0 (4 variants per format:
//!   RGB, RGB-u16, RGBA, RGBA-u16).
//! - `p010` / `p012` / `p016` — high-bit semi-planar 4:2:0
//!   (4 variants per format).
//!
//! Public functions re-exported up to `crate::row::*` via parent
//! `dispatch/mod.rs`.

#[cfg(feature = "yuv-semi-planar")]
pub(super) mod p010;
#[cfg(feature = "yuv-semi-planar")]
pub(super) mod p012;
#[cfg(feature = "yuv-semi-planar")]
pub(super) mod p016;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p10;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p12;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p14;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p16;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv420p9;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv_410;
#[cfg(feature = "yuv-planar")]
pub(super) mod yuv_420;

#[cfg(feature = "yuv-semi-planar")]
pub use p010::*;
#[cfg(feature = "yuv-semi-planar")]
pub use p012::*;
#[cfg(feature = "yuv-semi-planar")]
pub use p016::*;
#[cfg(feature = "yuv-planar")]
pub use yuv_410::*;
#[cfg(feature = "yuv-planar")]
pub use yuv_420::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p9::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p10::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p12::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p14::*;
#[cfg(feature = "yuv-planar")]
pub use yuv420p16::*;
