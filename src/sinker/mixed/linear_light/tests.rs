//! Unit tests for the linear-light frame allocator's *error typing*.
//!
//! [`LinearLightFrame::new`] has two distinct failure modes that must surface
//! as distinct typed errors (the alloc-vs-overflow split this module owns):
//!
//! - a **size overflow** — the `src_w * src_h * 3` element count cannot be
//!   represented in `usize` — is a [`MixedSinkerError::GeometryOverflow`], and
//! - an allocator **refusal** of a representable size is a
//!   [`ResampleError::AllocationFailed`] (mirroring the row-stage / native
//!   frame tails), NOT a geometry overflow.
//!
//! The overflow path is exercised directly here because it short-circuits on a
//! `checked_mul` *before* any allocation, so it is deterministic and never
//! OOMs (the products are rejected, not allocated). The allocator-refusal path
//! is not OOM-forceable from a unit test, so its *typing* is pinned by the
//! `arm_linear_tail_alloc_failure` failpoint in the integration suite
//! (`tests::resample_linear_domain::…_alloc_failure_leaves_frame_retryable`),
//! which now asserts `AllocationFailed`.

use super::*;
use crate::resample::{PlanGeometry, ResampleError};

/// A representable, allocatable frame builds: the `checked_mul` guard does not
/// false-positive on in-range dims, and with memory available the
/// `AllocationFailed` path stays dormant (the happy path returns `Ok`). Covers
/// both `want_luma` arms — the Y-plane buffer is allocated only when asked.
#[test]
fn new_builds_for_representable_dims() {
  for want_luma in [false, true] {
    let frame = LinearLightFrame::new(
      8,
      8,
      4,
      4,
      want_luma,
      TransferFunction::Srgb,
      LinearMode::SceneReferred,
    )
    .expect("8x8 frame must allocate");
    assert_eq!(frame.linear_rgb.len(), 8 * 8 * 3);
    assert_eq!(frame.y_plane.len(), if want_luma { 8 * 8 } else { 0 });
    assert_eq!(frame.next_y, 0);
    assert_eq!(frame.src_w, 8);
    assert_eq!(frame.src_h, 8);
    assert_eq!(frame.frozen_transfer, TransferFunction::Srgb);
    assert_eq!(frame.frozen_linear_mode, LinearMode::SceneReferred);
  }
}

/// `src_w * src_h` overflowing `usize` is a [`GeometryOverflow`] — a geometry
/// error, NOT an [`AllocationFailed`]. The `checked_mul` rejects before any
/// `try_zeroed`, so this allocates nothing (no OOM).
#[test]
fn new_overflowing_luma_product_is_geometry_overflow() {
  // `usize::MAX * 2` overflows the very first `checked_mul` (the `luma`
  // product), before the `* 3` step or any allocation.
  let err = LinearLightFrame::new(
    usize::MAX,
    2,
    4,
    4,
    false,
    TransferFunction::Srgb,
    LinearMode::DisplayReferred,
  )
  .expect_err("usize::MAX x 2 must overflow the luma product");
  assert!(
    matches!(err, MixedSinkerError::GeometryOverflow(_)),
    "overflowing luma product must be GeometryOverflow (a size overflow is not \
     an allocation failure), got {err:?}",
  );
}

/// `src_w * src_h * 3` overflowing `usize` (while `src_w * src_h` itself fits)
/// is likewise a [`GeometryOverflow`] from the second `checked_mul`.
#[test]
fn new_overflowing_rgb_product_is_geometry_overflow() {
  // Choose dims whose product fits usize but whose `* 3` does not:
  // `(usize::MAX / 2) * 1` fits, and `* 3` overflows.
  let luma = usize::MAX / 2;
  let err = LinearLightFrame::new(
    luma,
    1,
    4,
    4,
    false,
    TransferFunction::Srgb,
    LinearMode::DisplayReferred,
  )
  .expect_err("(usize::MAX / 2) x 3 must overflow the RGB product");
  assert!(
    matches!(err, MixedSinkerError::GeometryOverflow(_)),
    "overflowing RGB product must be GeometryOverflow, got {err:?}",
  );
}

/// The overflow guard fires the same way (a [`GeometryOverflow`], not an
/// allocation failure) on the `want_luma` path — the Y-plane allocation is
/// never reached because the `checked_mul` short-circuits first.
#[test]
fn new_overflow_with_luma_requested_is_geometry_overflow() {
  let err = LinearLightFrame::new(
    usize::MAX,
    2,
    4,
    4,
    true,
    TransferFunction::Srgb,
    LinearMode::DisplayReferred,
  )
  .expect_err("usize::MAX x 2 must overflow even with want_luma");
  assert!(
    matches!(err, MixedSinkerError::GeometryOverflow(_)),
    "overflow with want_luma must still be GeometryOverflow, got {err:?}",
  );
}

/// Sanity: the [`AllocationFailed`] payload an allocator refusal *would* carry
/// reports the full plan geometry (src + out), the same shape the row-stage /
/// native tails use — proving the alloc-failure variant is distinct from the
/// overflow variant in both type and payload. (We construct the error value
/// the `new` allocator-refusal arm builds; an actual refusal is not
/// OOM-forceable from a unit test, but its *typing* is what matters here.)
#[test]
fn allocation_failed_variant_carries_plan_geometry() {
  let err = MixedSinkerError::Resample(ResampleError::AllocationFailed(PlanGeometry::new(
    8, 8, 4, 4,
  )));
  match err {
    MixedSinkerError::Resample(ResampleError::AllocationFailed(g)) => {
      assert_eq!((g.src_w(), g.src_h(), g.out_w(), g.out_h()), (8, 8, 4, 4));
    }
    other => panic!("expected AllocationFailed, got {other:?}"),
  }
}

/// The final-row bin tail's RGB element count (`out_w * out_h * 3`) is computed
/// with checked arithmetic: an overflow is a typed [`GeometryOverflow`], never
/// a wrap into an undersized `binned` allocation. This guards the one-channel-
/// only Linear runs whose plan validates only the one-channel output size, so
/// `ow * oh * 3` is otherwise unbounded on a 32-bit target. Exercised directly
/// (the helper) so the check is provable on a 64-bit host without a giant plan.
#[test]
fn linear_tail_rgb_len_overflow_is_geometry_overflow() {
  // `out_w * out_h` overflows.
  let err = linear_tail_rgb_len(usize::MAX, 2).expect_err("usize::MAX x 2 must overflow");
  assert!(
    matches!(err, MixedSinkerError::GeometryOverflow(_)),
    "out_w * out_h overflow must be GeometryOverflow, got {err:?}",
  );
  // `out_w * out_h` fits but the `* 3` overflows.
  let err = linear_tail_rgb_len(usize::MAX / 2, 1).expect_err("(usize::MAX / 2) x 3 must overflow");
  assert!(
    matches!(err, MixedSinkerError::GeometryOverflow(_)),
    "the x3 overflow must be GeometryOverflow, got {err:?}",
  );
  // A representable geometry yields the exact element count.
  assert_eq!(
    linear_tail_rgb_len(4, 4).expect("4x4x3 is representable"),
    48,
  );
}
