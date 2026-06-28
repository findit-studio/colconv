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
// Shared by the 4:2:2 high-bit centered-siting path (#302): the same u16
// half-width->full-width phase-0.5 INTERLEAVED chroma staging (4:2:0 and 4:2:2
// subsample chroma 2:1 horizontally identically), exported under format-neutral
// `_pn` names so `p2xx` reuses the `p0xx` wrappers verbatim rather than
// duplicating them. `reserve_pn_chroma_full_u16` is the semi-planar interleaved
// twin of the planar `yuv420p::reserve_420_chroma_full_u16` re-exported below;
// the alias also avoids a name collision between the two.
#[cfg(all(
  test,
  feature = "std",
  feature = "yuv-semi-planar",
  feature = "yuv-planar"
))]
pub(crate) use p0xx::arm_p0xx_alloc_failure;
#[cfg(all(feature = "yuv-semi-planar", feature = "yuv-planar"))]
pub(crate) use p0xx::{
  reserve_420_chroma_full_u16 as reserve_pn_chroma_full_u16,
  upsample_420_chroma_center_h_p0xx as upsample_pn_chroma_center_h,
};
#[cfg(feature = "yuv-planar")]
pub(crate) use yuv420p::{reserve_420_chroma_full_u16, upsample_420_chroma_center_h_u16};
