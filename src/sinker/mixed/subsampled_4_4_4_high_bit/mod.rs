//! High-bit-depth 4:4:4 `MixedSinker` impls, split per sub-format
//! so no single source exceeds ~1.5 KLoC: Yuv444p9/10/12/14/16
//! (`yuv-planar`) + P410/P412/P416 (`yuv-semi-planar`).

#[cfg(feature = "yuv-semi-planar")]
mod p4xx;
#[cfg(feature = "yuv-planar")]
mod yuv444p;
#[cfg(feature = "yuv-planar")]
mod yuv444p_msb;

#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
pub(crate) use p4xx::arm_p4xx_alloc_failure;
