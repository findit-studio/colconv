//! 10 / 12 / 14 / 16-bit Bayer — single-plane mosaic source
//! carrying **low-packed** `u16` samples.
//!
//! Shape mirrors [`super::bayer`] for the 8-bit case but with a
//! `u16` plane and a `BITS` const generic. Sinks consume
//! [`BayerRow16<'_, BITS>`] (different row type from the 8-bit
//! [`super::BayerRow`] so the type system pins the input bit depth
//! at the sink boundary).
//!
//! Sample convention is **low-packed**: active samples occupy the
//! low `BITS` bits of each `u16`, valid range
//! `[0, (1 << BITS) - 1]`. This matches the planar
//! [`Yuv420pFrame16`](crate::frame::Yuv420pFrame16) family in
//! packing (low bits) but not validation cost: Bayer16's
//! [`crate::frame::BayerFrame16::try_new`] validates every active
//! sample's range as part of construction, so the
//! [`bayer16_to`] walker is fully fallible — no data-dependent
//! panic surface. **Note:** this is the opposite of
//! [`PnFrame`](crate::frame::PnFrame) (high-bit-packed semi-planar
//! `u16`); if your upstream provides high-bit-packed Bayer,
//! right-shift by `(16 - BITS)` before constructing
//! [`BayerFrame16`](crate::frame::BayerFrame16).

// The Bayer16 marker family now lives in mediaframe::source. Re-export
// everything so downstream code that uses `colconv::raw::Bayer16<BITS>`,
// `colconv::raw::Bayer10`, etc. keeps compiling unchanged.
pub use mediaframe::{
  frame::{
    Bayer10Frame, Bayer12Frame, Bayer14Frame, Bayer16Frame, BayerFrame16Error, BayerRow16,
    BayerSink16, bayer16_to,
  },
  source::{Bayer10, Bayer12, Bayer14, Bayer16, Bayer16Bit},
};

#[cfg(all(test, feature = "std"))]
#[cfg(any(feature = "std", feature = "alloc"))]
mod tests;
