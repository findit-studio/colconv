//! High-bit-depth 4:2:0 `MixedSinker` impls, split per sub-format
//! so no single source exceeds ~1.5 KLoC: Yuv420p9/10/12/14/16
//! (`yuv-planar`) + P010/P012/P016 (`yuv-semi-planar`). The `native`
//! fast-tier join is `yuv-planar` (the P-format native tier reuses it,
//! so it is `all(yuv-semi-planar, yuv-planar)` inside `p0xx`); the
//! semi-planar P-format sinks fall back to the shared row-stage tail
//! when `yuv-planar` is absent.

#[cfg(feature = "yuv-planar")]
mod native;
#[cfg(feature = "yuv-semi-planar")]
mod p0xx;
#[cfg(feature = "yuv-planar")]
mod yuv420p;

#[cfg(all(test, feature = "std", feature = "yuv-planar"))]
pub(crate) use native::arm_native_u16_alloc_failure;
#[cfg(feature = "yuv-planar")]
pub(crate) use native::{NativeYuv420U16, yuv420p16_process_native};
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
pub(crate) use p0xx::arm_p0xx_alloc_failure;
