//! Chroma-siting-aware 4:2:2 upsampling for `Yuv422p` (#302).
//!
//! 4:2:2 subsamples chroma 2:1 horizontally ONLY — one chroma row per luma row,
//! no vertical subsampling — so the centered siting reduces to the SAME
//! phase-0.5 horizontal upsample 4:2:0 uses
//! ([`chroma_upsample_2to1_center_h`](crate::row::scalar::chroma_upsample_2to1_center_h)):
//! there is no vertical blend / `Bottom` path. Covers: the centered RGB / RGBA /
//! HSV decodes matching an independent "upsample-then-4:4:4" reference; the
//! default / co-sited path staying byte-identical to the pre-#302
//! nearest-neighbor decode (the regression guard, plus its negative control);
//! SIMD-vs-scalar parity; the repo-wide no-output invariant (no scratch grows,
//! no overflow on absurd geometry); the preflight-ordering atomicity failpoint;
//! and `ChromaDerivedNcl` staying internally consistent across the two paths.

use super::*;
use crate::ChromaLocation;

const W: u32 = 16;
const H: u32 = 8;

/// A `Yuv422p` frame: flat luma plus a per-column chroma ramp on a half-width,
/// **full-height** chroma plane (one chroma row per luma row). The `+ r` term
/// keeps chroma rows distinct, so a (hypothetical) vertical mistake would show.
fn ramp_yuv422p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let y = std::vec![128u8; w * h];
  let mut u = std::vec![0u8; cw * h];
  let mut v = std::vec![0u8; cw * h];
  for r in 0..h {
    for c in 0..cw {
      u[r * cw + c] = (16 + c * 24 + r * 3).min(240) as u8;
      v[r * cw + c] = (240 - c * 24).max(16) as u8;
    }
  }
  (y, u, v)
}

/// Independent reference for the centered-siting horizontal upsample — the
/// MPEG-1 / JPEG phase-0.5 `1/4`–`3/4` weights with edge clamp. Written
/// separately from the production kernel so it is a real oracle.
fn ref_upsample_center_h(c_half: &[u8], width: usize) -> Vec<u8> {
  let half = width / 2;
  let mut out = std::vec![0u8; width];
  for j in 0..half {
    let l = c_half[j.saturating_sub(1)] as i32;
    let m = c_half[j] as i32;
    let r = c_half[if j + 1 < half { j + 1 } else { j }] as i32;
    out[2 * j] = ((l + 3 * m + 2) >> 2) as u8;
    out[2 * j + 1] = ((3 * m + r + 2) >> 2) as u8;
  }
  out
}

/// Builds the full-resolution U / V a centered-siting `Yuv422p` decode should
/// reconstruct: each luma row `r` takes chroma row `r` (1:1 — NO vertical
/// subsampling, unlike 4:2:0's `r / 2`) horizontally upsampled with the centered
/// weights. Feeding these to a `Yuv444p` conversion is the end-to-end oracle.
fn ref_full_chroma(u422: &[u8], v422: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u8; w * h];
  let mut v444 = std::vec![0u8; w * h];
  for r in 0..h {
    let urow = ref_upsample_center_h(&u422[r * cw..r * cw + cw], w);
    let vrow = ref_upsample_center_h(&v422[r * cw..r * cw + cw], w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
  let (yp, up, vp) = ramp_yuv422p();
  let src = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(loc)
    .with_simd(simd);
  yuv422p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  rgb
}

// ---- kernel oracle (the shared 2:1 horizontal upsample) --------------------

#[test]
fn center_upsample_kernel_matches_hand_computed() {
  // Same shared kernel 4:2:0 uses; re-asserted here to document the 4:2:2 reuse.
  // c = [0, 0, 100, 100] (half = 4, width = 8) ramps the step right of the
  // co-sited boundary with the 1/4–3/4 weights and edge clamp.
  let c_half = [0u8, 0, 100, 100];
  let mut out = [0u8; 8];
  crate::row::scalar::chroma_upsample_2to1_center_h(&c_half, &mut out, 8);
  assert_eq!(out, [0, 0, 0, 25, 75, 100, 100, 100]);
}

// ---- default / co-sited path is byte-identical (regression guard) ----------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn default_and_cosited_sitings_are_byte_identical() {
  let baseline = convert_rgb(ChromaLocation::Unspecified, true);
  for loc in [
    ChromaLocation::Unspecified,
    ChromaLocation::Unknown(99),
    ChromaLocation::Left,
    ChromaLocation::TopLeft,
    ChromaLocation::BottomLeft,
  ] {
    assert_eq!(
      convert_rgb(loc, true),
      baseline,
      "siting {loc:?} must keep the byte-identical default decode"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn default_path_does_not_allocate_chroma_scratch() {
  let (yp, up, vp) = ramp_yuv422p();
  let src = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Left);
  yuv422p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  let chroma_len = sink.chroma_full.len();
  drop(sink);
  assert_eq!(
    chroma_len, 0,
    "co-sited path must not grow the chroma scratch"
  );
}

// ---- centered path correctness ---------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_grows_chroma_scratch_to_full_width() {
  let (yp, up, vp) = ramp_yuv422p();
  let src = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Center);
  yuv422p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  let chroma_len = sink.chroma_full.len();
  drop(sink);
  assert_eq!(
    chroma_len,
    2 * W as usize,
    "centered path stages U+V at full width"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_rgb_matches_upsample_then_444_reference() {
  let (yp, up, vp) = ramp_yuv422p();

  // Reference: horizontally upsample chroma (centered weights, 1:1 rows) to full
  // resolution, then run the ordinary 4:4:4 decode.
  let (u444, v444) = ref_full_chroma(&up, &vp);
  let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
  let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
  let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb_ref)
    .unwrap();
  yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();

  assert_eq!(
    convert_rgb(ChromaLocation::Center, true),
    rgb_ref,
    "centered 4:2:2 RGB must equal upsample-then-4:4:4"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn top_center_bottom_horizontally_equal() {
  // 4:2:2 has no vertical subsampling, so every horizontally-centered siting
  // (Center / Top / Bottom) reduces to the SAME horizontal phase — all equal.
  let center = convert_rgb(ChromaLocation::Center, true);
  assert_eq!(
    convert_rgb(ChromaLocation::Top, true),
    center,
    "Top shares Center's horizontal phase (4:2:2 has no vertical axis)"
  );
  assert_eq!(
    convert_rgb(ChromaLocation::Bottom, true),
    center,
    "Bottom shares Center's horizontal phase (4:2:2 has no vertical axis)"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_phase_differs_from_default() {
  // Negative control: on a chroma ramp the centered phase must move chroma
  // relative to the co-sited / nearest-neighbor default.
  assert_ne!(
    convert_rgb(ChromaLocation::Center, true),
    convert_rgb(ChromaLocation::Left, true),
    "centered siting must shift chroma vs the co-sited default"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_path_simd_matches_scalar() {
  assert_eq!(
    convert_rgb(ChromaLocation::Center, true),
    convert_rgb(ChromaLocation::Center, false),
    "centered path must be bit-identical across the SIMD and scalar tiers"
  );
}

// ---- RGBA / HSV identity outputs also honor siting -------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_rgba_and_hsv_match_444_reference() {
  let (yp, up, vp) = ramp_yuv422p();
  let (u444, v444) = ref_full_chroma(&up, &vp);

  // RGBA-only path.
  {
    let src = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuv422p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba_ref)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      rgba, rgba_ref,
      "centered RGBA must equal upsample-then-4:4:4"
    );
  }

  // HSV-direct path (no RGB / RGBA attached).
  {
    let src = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let (mut h, mut s, mut v) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut sink = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuv422p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

    let ref_src = Yuv444pFrame::new(&yp, &u444, &v444, W, H, W, W, W);
    let (mut hr, mut sr, mut vr) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut ref_sink = MixedSinker::<Yuv444p>::new(W as usize, H as usize)
      .with_hsv(&mut hr, &mut sr, &mut vr)
      .unwrap();
    yuv444p_to(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
    assert_eq!(
      (h, s, v),
      (hr, sr, vr),
      "centered HSV must equal upsample-then-4:4:4"
    );
  }
}

// ---- preflight-ordering atomicity (#302, cf. #180) -------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  // luma PLUS a centered RGB decode whose chroma-scratch allocation fails must
  // leave EVERY output buffer — luma included — untouched: the centered scratch
  // is reserved (fallibly) BEFORE any output row is written.
  let (yp, up, vp) = ramp_yuv422p();
  let src = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut luma = std::vec![0xABu8; (W * H) as usize];
  let mut rgb = std::vec![0xCDu8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Center);

  super::super::arm_chroma_full_alloc_failure();
  let err = yuv422p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "centered chroma-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the centered alloc-failure path"
  );
  assert!(
    rgb.iter().all(|&b| b == 0xCD),
    "rgb must be untouched on the centered alloc-failure path"
  );
}

// ---- no-output invariant ----------------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn no_output_row_does_not_allocate_chroma_scratch() {
  // A centered-siting sink with NO outputs attached must honour the repo-wide
  // no-output invariant: every `process` call returns before the preflight, so
  // the centered chroma scratch is NEVER reserved.
  let (yp, up, vp) = ramp_yuv422p();
  let src = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut sink = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
    .with_chroma_location(ChromaLocation::Center);
  yuv422p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  let chroma_len = sink.chroma_full.len();
  drop(sink);
  assert_eq!(
    chroma_len, 0,
    "a no-output centered row must not reserve the chroma scratch"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "constructs an absurd geometry; the no-op contract is the point, not Miri"
)]
fn no_output_row_large_geometry_does_not_overflow() {
  // The no-output guard must run BEFORE the `idx * w` single-plane offset
  // arithmetic. A no-output `process` call never ran an attach-time `w x h x 1`
  // validation, so on a 32-bit target (`usize == u32`) an absurd geometry where
  // `idx * w` exceeds `u32::MAX` would overflow that offset and panic under
  // overflow checks. With no outputs attached, `process` must return `Ok(())`
  // having done NO row math and NO allocation.
  //
  // w = 4, idx = 2^30 → idx * w = 2^32 = u32::MAX + 1 (overflows u32).
  let w: usize = 4;
  let idx: usize = 1 << 30;
  let h: usize = idx + 1; // idx < height so the row-index check passes
  assert!(
    (idx as u64) * (w as u64) > u32::MAX as u64,
    "test geometry must exceed u32::MAX to exercise the 32-bit offset overflow"
  );

  let y = std::vec![128u8; w];
  let c = std::vec![128u8; w / 2];
  let mut sink = MixedSinker::<Yuv422p>::new(w, h).with_chroma_location(ChromaLocation::Center);
  let row = Yuv422pRow::new(&y, &c, &c, idx, ColorMatrix::Bt601, false);
  crate::PixelSink::process(&mut sink, row).unwrap();
  let chroma_len = sink.chroma_full.len();
  drop(sink);
  assert_eq!(
    chroma_len, 0,
    "a no-output large-geometry row must allocate nothing"
  );
}

// ---- centered siting + ChromaDerivedNcl consistency (#302 / #303 seam) ------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_chroma_derived_ncl_consistent_with_default() {
  // Yuv422p is NOT ChromaDerivedNcl-primaries-wired (only 8-bit Yuv420p got
  // #316). BOTH paths — the default fused 4:2:2 kernel AND the centered 4:4:4
  // kernel — resolve ChromaDerivedNcl via the shared BT.709 matrix-tag fallback
  // (`Coefficients::for_matrix`), IGNORING the ColorSpec primaries, so default
  // and centered stay internally consistent (the centered phase shift is the
  // ONLY difference). Full primaries-derived support is a documented follow-up.
  use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

  let (yp, up, vp) = ramp_yuv422p();
  let spec = |loc: ChromaLocation| {
    ColorSpec::from_info(
      PixelFormat::Yuv422p,
      ColorInfo::new(
        Primaries::Bt2020,
        Transfer::Bt709,
        ColorMatrix::ChromaDerivedNcl,
        DynamicRange::Limited,
        loc,
      ),
    )
  };
  let decode_cdn = |loc: ChromaLocation| -> Vec<u8> {
    let src = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgb = std::vec![0u8; (W * H * 3) as usize];
    let mut sink = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(spec(loc));
    yuv422p_to(&src, false, ColorMatrix::ChromaDerivedNcl, &mut sink).unwrap();
    rgb
  };
  let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
    let src = Yuv422pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgb = std::vec![0u8; (W * H * 3) as usize];
    let mut sink = MixedSinker::<Yuv422p>::new(W as usize, H as usize)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_chroma_location(loc);
    yuv422p_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
    rgb
  };

  assert_eq!(
    decode_cdn(ChromaLocation::Center),
    decode_bt709(ChromaLocation::Center),
    "centered ChromaDerivedNcl must resolve via the BT.709 matrix-tag fallback"
  );
  assert_eq!(
    decode_cdn(ChromaLocation::Left),
    decode_bt709(ChromaLocation::Left),
    "default ChromaDerivedNcl must resolve via the same BT.709 fallback"
  );
  // Guard: the centered phase actually changed the decode (so the equalities
  // above are not vacuous on a flat output).
  assert_ne!(
    decode_cdn(ChromaLocation::Center),
    decode_cdn(ChromaLocation::Left),
    "centered ChromaDerivedNcl must still shift chroma vs the co-sited default"
  );
}
