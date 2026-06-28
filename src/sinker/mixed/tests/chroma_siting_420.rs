//! Chroma-siting-aware 4:2:0 upsampling for `Yuv420p` (#302).
//!
//! Covers: the centered-siting horizontal upsample kernel against a
//! hand-computed oracle; the default / co-sited path staying byte-identical
//! to the pre-#302 nearest-neighbor decode (the regression guard); the
//! centered RGB / RGBA / HSV identity decodes matching an independent
//! "upsample-then-4:4:4" reference; SIMD-vs-scalar parity of the centered
//! path; and that the centered phase actually shifts chroma horizontally.

use super::*;
use crate::ChromaLocation;

const W: u32 = 16;
const H: u32 = 8;

/// A `Yuv420p` frame with flat luma and a per-column chroma ramp, so the
/// horizontal chroma phase is observable (a solid chroma frame would make
/// every siting identical).
fn ramp_yuv420p() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h / 2;
  let y = std::vec![128u8; w * h];
  let mut u = std::vec![0u8; cw * ch];
  let mut v = std::vec![0u8; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      // Distinct per-column ramps for U and V; the `+ r` keeps chroma rows
      // from being identical so a (hypothetical) vertical mistake would show.
      u[r * cw + c] = (16 + c * 24 + r * 3).min(240) as u8;
      v[r * cw + c] = (240 - c * 24).max(16) as u8;
    }
  }
  (y, u, v)
}

/// Independent reference for the centered-siting horizontal upsample — the
/// MPEG-1 / JPEG phase-0.5 `1/4`–`3/4` weights with edge clamp. Mirrors the
/// documented formula but is written separately from the production kernel so
/// it is a real oracle.
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

/// Builds the full-resolution U / V planes a centered-siting `Yuv420p`
/// decode should reconstruct: each luma row `r` takes chroma row `r / 2`
/// (the walker's vertical replication, unchanged by #302) horizontally
/// upsampled with the centered weights. Feeding these to a `Yuv444p`
/// conversion is the end-to-end oracle for the centered path.
fn ref_full_chroma(u420: &[u8], v420: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u8; w * h];
  let mut v444 = std::vec![0u8; w * h];
  for r in 0..h {
    let cr = r / 2;
    let urow = ref_upsample_center_h(&u420[cr * cw..cr * cw + cw], w);
    let vrow = ref_upsample_center_h(&v420[cr * cw..cr * cw + cw], w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(loc)
    .with_simd(simd);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
  rgb
}

// ---- kernel oracle ---------------------------------------------------------

#[test]
fn center_upsample_kernel_matches_hand_computed() {
  // c = [0, 0, 100, 100] (half = 4, width = 8). Centered reconstruction
  // ramps the step and shifts it right of the co-sited boundary:
  //   even 2j = (c[j-1] + 3·c[j] + 2) >> 2, odd 2j+1 = (3·c[j] + c[j+1] + 2) >> 2.
  let c_half = [0u8, 0, 100, 100];
  let mut out = [0u8; 8];
  crate::row::scalar::chroma_upsample_420_center_h(&c_half, &mut out, 8);
  assert_eq!(out, [0, 0, 0, 25, 75, 100, 100, 100]);
}

#[test]
fn center_upsample_kernel_clamps_edges() {
  // Width 4: left edge even = c[0] exactly, right edge odd = c[last] exactly.
  let c_half = [10u8, 20];
  let mut out = [0u8; 4];
  crate::row::scalar::chroma_upsample_420_center_h(&c_half, &mut out, 4);
  assert_eq!(out, [10, 13, 18, 20]);
  assert_eq!(out[0], c_half[0], "left edge even column is co-sited");
  assert_eq!(out[3], c_half[1], "right edge odd column is co-sited");
}

// ---- default / co-sited path is byte-identical (regression guard) ----------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn default_and_cosited_sitings_are_byte_identical() {
  // The pre-#302 baseline: a sink that never sets a chroma location.
  let baseline = convert_rgb(ChromaLocation::Unspecified, true);

  // Unspecified, Unknown, and every horizontally co-sited value keep the
  // exact nearest-neighbor decode — bit-for-bit equal to the baseline even
  // though the chroma plane is a non-trivial ramp.
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
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Left);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
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
fn center_rgb_matches_upsample_then_444_reference() {
  let (yp, up, vp) = ramp_yuv420p();

  // Reference: horizontally upsample chroma (centered weights) to full
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
    "centered 4:2:0 RGB must equal upsample-then-4:4:4"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_grows_chroma_scratch_to_full_width() {
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Center);
  yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
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
fn top_and_bottom_route_like_center_horizontally() {
  // Top / Bottom share Center's horizontal (centered) phase; the vertical
  // phase is not yet consumed (#302 horizontal-only), so all three produce
  // the same RGB here.
  let center = convert_rgb(ChromaLocation::Center, true);
  assert_eq!(convert_rgb(ChromaLocation::Top, true), center);
  assert_eq!(convert_rgb(ChromaLocation::Bottom, true), center);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_phase_differs_from_default() {
  // The whole point: on a chroma ramp the centered phase must move chroma
  // relative to the co-sited / nearest-neighbor default.
  assert_ne!(
    convert_rgb(ChromaLocation::Center, true),
    convert_rgb(ChromaLocation::Left, true),
    "centered siting must shift chroma vs the co-sited default"
  );
}

// ---- SIMD vs scalar parity -------------------------------------------------

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
  let (yp, up, vp) = ramp_yuv420p();
  let (u444, v444) = ref_full_chroma(&up, &vp);

  // RGBA-only path.
  {
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgba = std::vec![0u8; (W * H * 4) as usize];
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

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
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let (mut h, mut s, mut v) = (
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
      std::vec![0u8; (W * H) as usize],
    );
    let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap()
      .with_chroma_location(ChromaLocation::Center);
    yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

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

// ---- end-to-end ColorSpec flow (no manual with_chroma_location) ------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn color_spec_center_drives_decode_without_manual_chroma_call() {
  use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer, YuvOptions};

  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);

  // Drive the decode from a ColorSpec carrying ChromaLocation::Center via the
  // NORMAL path: YuvOptions::from_color_spec(spec) for the walk (matrix +
  // range) and the sink's ColorSpec entry point for the siting — with NO
  // manual `with_chroma_location` call.
  let info = ColorInfo::new(
    Primaries::Bt709,
    Transfer::Bt709,
    ColorMatrix::Bt601,
    DynamicRange::Limited,
    ChromaLocation::Center,
  );
  let spec = ColorSpec::from_info(PixelFormat::Yuv420p, info);
  let opts = YuvOptions::from_color_spec(spec);
  let mut rgb = std::vec![0u8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_color_spec(spec);
  yuv420p_to(&src, opts.full_range(), opts.matrix(), &mut sink).unwrap();
  drop(sink);

  // `opts` mirrors `convert_rgb`'s limited-range Bt601 walk, so the
  // spec-driven output must (a) differ from the default (no siting) and
  // (b) match the explicit centered path bit-for-bit.
  assert_ne!(
    rgb,
    convert_rgb(ChromaLocation::Unspecified, true),
    "ColorSpec ChromaLocation::Center must change the decode via the options path"
  );
  assert_eq!(
    rgb,
    convert_rgb(ChromaLocation::Center, true),
    "ColorSpec-driven centered decode must equal the explicit centered path"
  );
}

// ---- centered siting + ChromaDerivedNcl (#303 cross-feature seam) -----------

/// The centered chroma-siting decode must honour
/// [`ColorMatrix::ChromaDerivedNcl`] just like the non-centered path: its
/// Kr/Kb are derived from the ColorSpec's primaries, NOT the BT.709 fallback.
/// Before the fix the centered branch called the 4:4:4 kernels with only the
/// matrix tag, so a non-BT709 ChromaDerivedNcl decoded with BT.709
/// coefficients (wrong colors) and could even run on SIMD — this test fails
/// against that bug. Covers RGB and RGBA, and the scalar-routing invariant.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn center_chroma_derived_ncl_uses_primaries_not_bt709() {
  use crate::{
    ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer,
    row::{
      yuv_444_to_rgb_row, yuv_444_to_rgb_row_primaries, yuv_444_to_rgba_row,
      yuv_444_to_rgba_row_primaries,
    },
  };

  let w = W as usize;
  let h = H as usize;
  let (yp, up, vp) = ramp_yuv420p();
  // The centered full-width chroma the decode reconstructs per row.
  let (u444, v444) = ref_full_chroma(&up, &vp);
  // A non-BT709 primary set: ChromaDerivedNcl here must diverge from BT.709.
  let prim = Primaries::Bt2020;
  let info = ColorInfo::new(
    prim,
    Transfer::Bt709,
    ColorMatrix::ChromaDerivedNcl,
    DynamicRange::Limited,
    ChromaLocation::Center,
  );
  let spec = ColorSpec::from_info(PixelFormat::Yuv420p, info);

  // Drive the centered Yuv420p decode via the ColorSpec (limited-range).
  let decode_rgb = |simd: bool| {
    let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
    let mut rgb = std::vec![0u8; w * h * 3];
    let mut sink = MixedSinker::<Yuv420p>::new(w, h)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_color_spec(spec)
      .with_simd(simd);
    yuv420p_to(&src, false, ColorMatrix::ChromaDerivedNcl, &mut sink).unwrap();
    rgb
  };

  // Exact reference: same centered-upsampled chroma, decoded 4:4:4 with the
  // primaries-derived coefficients (vs the BT.709 fallback the bug produced).
  let mut rgb_derived = std::vec![0u8; w * h * 3];
  let mut rgb_bt709 = std::vec![0u8; w * h * 3];
  for r in 0..h {
    let (ys, us, vs) = (
      &yp[r * w..r * w + w],
      &u444[r * w..r * w + w],
      &v444[r * w..r * w + w],
    );
    yuv_444_to_rgb_row_primaries(
      ys,
      us,
      vs,
      &mut rgb_derived[r * w * 3..(r + 1) * w * 3],
      w,
      ColorMatrix::ChromaDerivedNcl,
      prim,
      false,
      true,
    );
    yuv_444_to_rgb_row(
      ys,
      us,
      vs,
      &mut rgb_bt709[r * w * 3..(r + 1) * w * 3],
      w,
      ColorMatrix::Bt709,
      false,
      true,
    );
  }
  let rgb = decode_rgb(true);
  assert_eq!(
    rgb, rgb_derived,
    "centered ChromaDerivedNcl RGB must use the primaries-derived coefficients"
  );
  assert_ne!(
    rgb, rgb_bt709,
    "centered ChromaDerivedNcl RGB must NOT decode as the BT.709 fallback (the bug)"
  );
  assert_ne!(
    rgb_derived, rgb_bt709,
    "sanity: Bt2020-derived and BT.709 coefficients differ on the chroma ramp"
  );
  // Scalar-routing invariant: ChromaDerivedNcl is deterministic across tiers
  // (it is forced onto the scalar kernel), so SIMD == scalar.
  assert_eq!(
    rgb,
    decode_rgb(false),
    "centered ChromaDerivedNcl must be scalar-routed (no SIMD/scalar split)"
  );

  // ---- RGBA twin ----
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut rgba = std::vec![0u8; w * h * 4];
  let mut sink = MixedSinker::<Yuv420p>::new(w, h)
    .with_rgba(&mut rgba)
    .unwrap()
    .with_color_spec(spec);
  yuv420p_to(&src, false, ColorMatrix::ChromaDerivedNcl, &mut sink).unwrap();
  drop(sink);

  let mut rgba_derived = std::vec![0u8; w * h * 4];
  let mut rgba_bt709 = std::vec![0u8; w * h * 4];
  for r in 0..h {
    let (ys, us, vs) = (
      &yp[r * w..r * w + w],
      &u444[r * w..r * w + w],
      &v444[r * w..r * w + w],
    );
    yuv_444_to_rgba_row_primaries(
      ys,
      us,
      vs,
      &mut rgba_derived[r * w * 4..(r + 1) * w * 4],
      w,
      ColorMatrix::ChromaDerivedNcl,
      prim,
      false,
      true,
    );
    yuv_444_to_rgba_row(
      ys,
      us,
      vs,
      &mut rgba_bt709[r * w * 4..(r + 1) * w * 4],
      w,
      ColorMatrix::Bt709,
      false,
      true,
    );
  }
  assert_eq!(
    rgba, rgba_derived,
    "centered ChromaDerivedNcl RGBA must use the primaries-derived coefficients"
  );
  assert_ne!(
    rgba, rgba_bt709,
    "centered ChromaDerivedNcl RGBA must NOT decode as the BT.709 fallback (the bug)"
  );
}

// ---- preflight-ordering atomicity (#302, cf. #180) -------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn centered_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  // A sink requesting luma PLUS a centered RGB decode whose chroma-scratch
  // allocation fails must leave EVERY output buffer — luma included —
  // untouched on the error path: the centered scratch is reserved (fallibly)
  // BEFORE any output row is written, so a refusal can't half-update the
  // frame.
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut luma = std::vec![0xABu8; (W * H) as usize];
  let mut rgb = std::vec![0xCDu8; (W * H * 3) as usize];
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap()
    .with_chroma_location(ChromaLocation::Center);

  super::super::arm_chroma_full_alloc_failure();
  let err = yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
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

// Gated on `yuva`: reuses the crate's existing RGB-scratch allocation
// failpoint (`arm_rgb_scratch_alloc_failure`, itself `yuva`-gated). Under
// `--all-features` both `yuv-planar` and `yuva` are on, so it runs.
#[cfg(feature = "yuva")]
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgb_scratch_alloc_failure_leaves_outputs_untouched() {
  use crate::resample::ResampleError;

  // Codex R3's exact case: luma + RGBA + HSV with NO rgb output. That is
  // `want_hsv && want_rgba && !want_rgb` → `need_rgb_kernel` with no caller
  // RGB buffer, so the decode grows the RGB row scratch
  // (`rgb_row_buf_or_scratch`'s scratch arm). With that allocation armed to
  // fail, the up-front preflight returns AllocationFailed BEFORE any output
  // row — luma included — is written.
  let (yp, up, vp) = ramp_yuv420p();
  let src = Yuv420pFrame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
  let mut luma = std::vec![0xABu8; (W * H) as usize];
  let mut rgba = std::vec![0xCDu8; (W * H * 4) as usize];
  let (mut hh, mut ss, mut vv) = (
    std::vec![0xCDu8; (W * H) as usize],
    std::vec![0xCDu8; (W * H) as usize],
    std::vec![0xCDu8; (W * H) as usize],
  );
  let mut sink = MixedSinker::<Yuv420p>::new(W as usize, H as usize)
    .with_luma(&mut luma)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap()
    .with_hsv(&mut hh, &mut ss, &mut vv)
    .unwrap();

  super::super::arm_rgb_scratch_alloc_failure();
  let err = yuv420p_to(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
  drop(sink);

  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::AllocationFailed(_))
    ),
    "RGB-scratch refusal must surface as a recoverable AllocationFailed, got {err:?}"
  );
  assert!(
    luma.iter().all(|&b| b == 0xAB),
    "luma must be untouched on the rgb-scratch alloc-failure path"
  );
}
