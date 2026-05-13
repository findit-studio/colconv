// Tests reference `videoframe::frame::{WhiteBalance, ColorCorrectionMatrix,
// fuse_wb_ccm, ...}` which only exist when videoframe's `bayer` feature is
// enabled (colconv's `bayer` feature passes through to it). Without the gate,
// `cargo test --no-default-features` would try to import names that don't
// exist in the videoframe build it's linking against.
#[cfg(all(test, feature = "bayer"))]
mod tests;
