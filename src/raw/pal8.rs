//! 8-bit indexed-color (`AV_PIX_FMT_PAL8`) — single-plane mosaic with a
//! 256-entry BGRA palette.
//!
//! The full quartet (marker, [`Pal8Row`], [`Pal8Sink`], [`pal8_to`]) now
//! lives in `videoframe::source` (under the `mono` feature). This module
//! is a thin re-export shim so downstream code that uses the
//! `colconv::raw::Pal8*` paths keeps compiling unchanged.

// Re-export everything so downstream code that uses `colconv::raw::Pal8`,
// `colconv::raw::Pal8Row`, `colconv::raw::Pal8Sink`, and
// `colconv::raw::pal8_to` keeps compiling unchanged.
pub use videoframe::source::{Pal8, Pal8Row, Pal8Sink, pal8_to};
