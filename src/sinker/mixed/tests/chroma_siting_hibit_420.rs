//! Chroma-siting-aware **high-bit** 4:2:0 upsampling for `Yuv420p9` …
//! `Yuv420p16` (#302).
//!
//! Covers, per bit depth (9 / 10 / 12 / 14 / 16, via the macro below): the
//! default / co-sited path staying byte-identical to the pre-#302
//! nearest-neighbor decode (the regression guard, plus its negative control
//! that the centered phase actually moves chroma); the centered RGB / RGBA /
//! HSV decodes — and their `u16` twins — matching an independent
//! "upsample-then-4:4:4" reference; SIMD-vs-scalar parity of the centered
//! path; the preflight-ordering atomicity (a centered chroma-scratch alloc
//! failure leaves luma AND colour untouched); and the `ChromaDerivedNcl`
//! consistency invariant (the high-bit formats are NOT primaries-wired, so
//! BOTH the default and centered paths resolve it via the BT.709 matrix-tag
//! fallback). The bit-exact `u16` upsample kernel is also checked directly
//! against a hand-computed oracle, including the big-endian wire path.
//!
//! The macro instantiates each bit depth with its **little-endian** marker, so
//! a sample's wire `u16` equals its logical value on the (little-endian) test
//! host; the references compute in that logical domain. The endianness
//! re-encode is exercised host-independently by the kernel-level BE oracle.

use super::*;
use crate::ChromaLocation;

const W: u32 = 16;
const H: u32 = 8;

/// Builds a high-bit 4:2:0 frame's logical planes: flat mid-gray luma plus a
/// per-column chroma ramp (distinct adjacent columns so the horizontal phase
/// is observable; the small `+ r` term keeps chroma rows from being identical
/// so a vertical mistake would surface). Values are clamped to `maxv =
/// (1 << BITS) - 1`.
fn ramp_planes_n(maxv: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let ch = h / 2;
  let step = (maxv / 16).max(1);
  let y = std::vec![(maxv / 2) as u16; w * h];
  let mut u = std::vec![0u16; cw * ch];
  let mut v = std::vec![0u16; cw * ch];
  for r in 0..ch {
    for c in 0..cw {
      u[r * cw + c] = (step * c as u32 + step + r as u32 * 5).min(maxv) as u16;
      v[r * cw + c] = maxv.saturating_sub(step * c as u32).max(step) as u16;
    }
  }
  (y, u, v)
}

/// Independent reference for the centered-siting horizontal upsample — the
/// MPEG-1 / JPEG phase-0.5 `1/4`–`3/4` weights with edge clamp, on logical
/// `u16`. Written separately from the production kernel so it is a real oracle.
fn ref_upsample_center_h_u16(c_half: &[u16], width: usize) -> Vec<u16> {
  let half = width / 2;
  let mut out = std::vec![0u16; width];
  for j in 0..half {
    let l = c_half[j.saturating_sub(1)] as u32;
    let m = c_half[j] as u32;
    let r = c_half[if j + 1 < half { j + 1 } else { j }] as u32;
    out[2 * j] = ((l + 3 * m + 2) >> 2) as u16;
    out[2 * j + 1] = ((3 * m + r + 2) >> 2) as u16;
  }
  out
}

/// Builds the full-resolution U / V a centered-siting high-bit 4:2:0 decode
/// reconstructs: each luma row `r` takes chroma row `r / 2` (the walker's
/// vertical replication, unchanged by #302) horizontally upsampled with the
/// centered weights. Feeding these to the matching `Yuv444pN` conversion is the
/// end-to-end oracle for the centered path.
fn ref_full_chroma_u16(u420: &[u16], v420: &[u16]) -> (Vec<u16>, Vec<u16>) {
  let w = W as usize;
  let h = H as usize;
  let cw = w / 2;
  let mut u444 = std::vec![0u16; w * h];
  let mut v444 = std::vec![0u16; w * h];
  for r in 0..h {
    let cr = r / 2;
    let urow = ref_upsample_center_h_u16(&u420[cr * cw..cr * cw + cw], w);
    let vrow = ref_upsample_center_h_u16(&v420[cr * cw..cr * cw + cw], w);
    u444[r * w..r * w + w].copy_from_slice(&urow);
    v444[r * w..r * w + w].copy_from_slice(&vrow);
  }
  (u444, v444)
}

// ---- u16 kernel oracle (endianness-explicit) -------------------------------

#[test]
fn center_upsample_u16_kernel_matches_hand_computed() {
  // c = [0, 0, 400, 400] (half = 4, width = 8), little-endian wire.
  //   even 2j   = (c[j-1] + 3·c[j] + 2) >> 2
  //   odd  2j+1 = (3·c[j] + c[j+1] + 2) >> 2
  // Values < 512 fit every depth, so the `BITS` mask is a no-op here; the
  // dirty-upper-bit masking is exercised by
  // `center_upsample_u16_kernel_masks_dirty_upper_bits`.
  let c_half = [0u16, 0, 400, 400];
  let mut out = [0u16; 8];
  crate::row::scalar::chroma_upsample_420_center_h_u16::<10>(&c_half, &mut out, 8, false);
  assert_eq!(out, [0, 0, 0, 100, 300, 400, 400, 400]);
}

#[test]
fn center_upsample_u16_kernel_clamps_edges() {
  // Width 4: left edge even = c[0] exactly, right edge odd = c[last] exactly.
  let c_half = [1000u16, 2000];
  let mut out = [0u16; 4];
  crate::row::scalar::chroma_upsample_420_center_h_u16::<12>(&c_half, &mut out, 4, false);
  assert_eq!(out, [1000, 1250, 1750, 2000]);
  assert_eq!(out[0], c_half[0], "left edge even column is co-sited");
  assert_eq!(out[3], c_half[1], "right edge odd column is co-sited");
}

#[test]
fn center_upsample_u16_kernel_big_endian_matches_le_logical() {
  // Same LOGICAL input, wire-encoded big-endian: the kernel interpolates in the
  // logical domain and re-encodes to the same wire order, so decoding the BE
  // output back yields the SAME logical result as the LE path. Host-independent.
  let logical = [0u16, 0, 400, 400];
  let le: Vec<u16> = logical.iter().map(|&x| x.to_le()).collect();
  let be: Vec<u16> = logical.iter().map(|&x| x.to_be()).collect();
  let mut out_le = [0u16; 8];
  let mut out_be = [0u16; 8];
  crate::row::scalar::chroma_upsample_420_center_h_u16::<10>(&le, &mut out_le, 8, false);
  crate::row::scalar::chroma_upsample_420_center_h_u16::<10>(&be, &mut out_be, 8, true);
  let dec_le: Vec<u16> = out_le.iter().map(|&x| u16::from_le(x)).collect();
  let dec_be: Vec<u16> = out_be.iter().map(|&x| u16::from_be(x)).collect();
  assert_eq!(
    dec_be, dec_le,
    "BE wire path must equal the LE logical interpolation"
  );
  assert_eq!(dec_be, std::vec![0u16, 0, 0, 100, 300, 400, 400, 400]);
}

#[test]
fn center_upsample_u16_kernel_masks_dirty_upper_bits() {
  // The fused high-bit decode kernels mask low-packed samples to BITS
  // (`& bits_mask::<BITS>()`) before use, sanitizing dirty upper bits in a
  // malformed-but-accepted frame. The centered upsample must do the SAME BEFORE
  // its 1/4-3/4 blend — otherwise a dirty sample's high bits leak into a
  // neighbour's low bits. For every sub-16-bit depth and both wire endians, a
  // frame with ALL bits above BITS set must blend identically to the masked
  // (clean) frame, and stay within `[0, (1 << BITS) - 1]`.
  fn check<const BITS: u32>() {
    let mask = ((1u32 << BITS) - 1) as u16;
    let upper = !mask; // every bit above BITS
    let clean = [0u16, 0, mask, mask]; // half = 4, width = 8; a non-constant ramp
    let dirty = [
      clean[0] | upper,
      clean[1] | upper,
      clean[2] | upper,
      clean[3] | upper,
    ];
    for &be in &[false, true] {
      let enc = |v: u16| if be { v.to_be() } else { v.to_le() };
      let dec = |v: u16| if be { u16::from_be(v) } else { u16::from_le(v) };
      let dirty_wire: Vec<u16> = dirty.iter().map(|&v| enc(v)).collect();
      let clean_wire: Vec<u16> = clean.iter().map(|&v| enc(v)).collect();
      let mut out_dirty = [0u16; 8];
      let mut out_clean = [0u16; 8];
      crate::row::scalar::chroma_upsample_420_center_h_u16::<BITS>(
        &dirty_wire,
        &mut out_dirty,
        8,
        be,
      );
      crate::row::scalar::chroma_upsample_420_center_h_u16::<BITS>(
        &clean_wire,
        &mut out_clean,
        8,
        be,
      );
      let dec_dirty: Vec<u16> = out_dirty.iter().map(|&v| dec(v)).collect();
      let dec_clean: Vec<u16> = out_clean.iter().map(|&v| dec(v)).collect();
      assert_eq!(
        dec_dirty, dec_clean,
        "BITS={BITS} be={be}: dirty upper bits must be masked before the blend"
      );
      assert!(
        dec_dirty.iter().all(|&v| v <= mask),
        "BITS={BITS} be={be}: blended output must stay within the bit depth"
      );
    }
  }
  check::<9>();
  check::<10>();
  check::<12>();
  check::<14>();

  // 16-bit has no spare bits: the mask is `u16::MAX` (a no-op), so a
  // top-of-range sample is preserved through the blend.
  let mut out = [0u16; 8];
  crate::row::scalar::chroma_upsample_420_center_h_u16::<16>(
    &[0u16, 0, 65535, 65535],
    &mut out,
    8,
    false,
  );
  assert_eq!(out, [0, 0, 0, 16384, 49151, 65535, 65535, 65535]);
}

// ---- per-bit-depth suite ---------------------------------------------------

// The suite is identical bar the bit depth, format marker, frame type, and
// walker, so generate it once per depth. Each lands in its own `mod` so the
// names don't collide.
macro_rules! hibit_420_chroma_tests {
  ($mod:ident, $bits:expr, $Marker:ident, $Frame:ident, $walker:ident, $Ref:ident, $RefFrame:ident, $ref_walker:ident, $MarkerBe:ty, $FrameBe:ident, $walker_be:ident) => {
    mod $mod {
      use super::*;

      const MAXV: u32 = (1u32 << $bits) - 1;

      /// Centered/default identity-decode RGB for a siting + SIMD toggle.
      fn convert_rgb(loc: ChromaLocation, simd: bool) -> Vec<u8> {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(loc)
          .with_simd(simd);
        $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
        rgb
      }

      // ---- default / co-sited path is byte-identical (regression guard) ----

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
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Left);
        $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
        drop(sink);
        assert_eq!(chroma_len, 0, "co-sited path must not grow the u16 chroma scratch");
      }

      // ---- centered path correctness ---------------------------------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_grows_chroma_scratch_to_full_width() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb = std::vec![0u8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
        let chroma_len = sink.chroma_full_u16.len();
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
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_u16(&up, &vp);
        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb_ref = std::vec![0u8; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb(&mut rgb_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
        assert_eq!(
          convert_rgb(ChromaLocation::Center, true),
          rgb_ref,
          "centered high-bit 4:2:0 RGB must equal upsample-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgb_u16_matches_upsample_then_444_reference() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_u16(&up, &vp);

        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut rgb16 = std::vec![0u16; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);
        $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

        let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
        let mut rgb16_ref = std::vec![0u16; (W * H * 3) as usize];
        let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
          .with_rgb_u16(&mut rgb16_ref)
          .unwrap();
        $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();

        assert_eq!(
          rgb16, rgb16_ref,
          "centered high-bit 4:2:0 RGB(u16) must equal upsample-then-4:4:4"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn center_rgba_rgba_u16_and_hsv_match_444_reference() {
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let (u444, v444) = ref_full_chroma_u16(&up, &vp);

        // RGBA (u8).
        {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgba = std::vec![0u8; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba(&mut rgba)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let mut rgba_ref = std::vec![0u8; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba(&mut rgba_ref)
            .unwrap();
          $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(rgba, rgba_ref, "centered RGBA must equal upsample-then-4:4:4");
        }

        // RGBA (u16).
        {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgba16 = std::vec![0u16; (W * H * 4) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba16)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let mut rgba16_ref = std::vec![0u16; (W * H * 4) as usize];
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_rgba_u16(&mut rgba16_ref)
            .unwrap();
          $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(
            rgba16, rgba16_ref,
            "centered RGBA(u16) must equal upsample-then-4:4:4"
          );
        }

        // HSV-direct (no RGB / RGBA attached).
        {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let (mut h, mut s, mut v) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_hsv(&mut h, &mut s, &mut v)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();

          let ref_src = $RefFrame::new(&yp, &u444, &v444, W, H, W, W, W);
          let (mut hr, mut sr, mut vr) = (
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
            std::vec![0u8; (W * H) as usize],
          );
          let mut ref_sink = MixedSinker::<$Ref>::new(W as usize, H as usize)
            .with_hsv(&mut hr, &mut sr, &mut vr)
            .unwrap();
          $ref_walker(&ref_src, false, ColorMatrix::Bt601, &mut ref_sink).unwrap();
          assert_eq!(
            (h, s, v),
            (hr, sr, vr),
            "centered HSV must equal upsample-then-4:4:4"
          );
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn top_and_bottom_route_like_center_horizontally() {
        // Top / Bottom share Center's horizontal (centered) phase; the vertical
        // phase is not yet consumed (#302 horizontal-only), so all three agree.
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
        // Negative control: on a chroma ramp the centered phase must move chroma
        // relative to the co-sited / nearest-neighbor default — otherwise the
        // byte-identity assertions above would be vacuous.
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

      // ---- dirty-upper-bit sanitization (mask before the blend) ------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_upper_bits_le() {
        // A malformed-but-accepted low-packed frame with bits set ABOVE BITS must
        // decode (centered) identically to the masked clean frame: the centered
        // upsample masks each sample to BITS BEFORE the 1/4-3/4 blend, exactly as
        // the fused decode kernels do, so a dirty sample's high bits never leak
        // into a neighbour's low bits. (At BITS = 16 `upper` is 0, so this is the
        // clean == clean identity — 16-bit has no spare bits.)
        let upper = !(MAXV as u16);
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let decode = |u: &[u16], v: &[u16]| -> Vec<u8> {
          let src = $Frame::new(&yp, u, v, W, H, W, W / 2, W / 2);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
          rgb
        };
        let up_dirty: Vec<u16> = up.iter().map(|&x| x | upper).collect();
        let vp_dirty: Vec<u16> = vp.iter().map(|&x| x | upper).collect();
        assert_eq!(
          decode(&up_dirty, &vp_dirty),
          decode(&up, &vp),
          "centered LE decode must sanitize dirty upper bits (mask before blend)"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_sanitizes_dirty_upper_bits_be() {
        // Same invariant on the big-endian wire path: the mask is applied in the
        // logical domain (after the endian load), so dirty bits are stripped for
        // BE inputs too. Planes are BE-encoded and decoded via the BE marker /
        // frame / walker.
        let upper = !(MAXV as u16);
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let y_be: Vec<u16> = yp.iter().map(|&x| x.to_be()).collect();
        let decode = |u_logical: &[u16], v_logical: &[u16]| -> Vec<u8> {
          let u_be: Vec<u16> = u_logical.iter().map(|&x| x.to_be()).collect();
          let v_be: Vec<u16> = v_logical.iter().map(|&x| x.to_be()).collect();
          let src = $FrameBe::try_new(&y_be, &u_be, &v_be, W, H, W, W / 2, W / 2).unwrap();
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$MarkerBe>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(ChromaLocation::Center);
          $walker_be(&src, false, ColorMatrix::Bt601, &mut sink).unwrap();
          rgb
        };
        let up_dirty: Vec<u16> = up.iter().map(|&x| x | upper).collect();
        let vp_dirty: Vec<u16> = vp.iter().map(|&x| x | upper).collect();
        assert_eq!(
          decode(&up_dirty, &vp_dirty),
          decode(&up, &vp),
          "centered BE decode must sanitize dirty upper bits (mask before blend)"
        );
      }

      // ---- preflight-ordering atomicity (#302 / #314, cf. #180) ------------

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_alloc_failure_leaves_outputs_untouched() {
        use crate::resample::ResampleError;

        // luma PLUS a centered RGB decode whose u16 chroma-scratch allocation
        // fails must leave EVERY output buffer — luma included — untouched: the
        // centered scratch is reserved (fallibly) BEFORE any output row is
        // written, so a refusal can't half-update the frame.
        let (yp, up, vp) = ramp_planes_n(MAXV);
        let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
        let mut luma = std::vec![0xABu8; (W * H) as usize];
        let mut rgb = std::vec![0xCDu8; (W * H * 3) as usize];
        let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
          .with_luma(&mut luma)
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_chroma_location(ChromaLocation::Center);

        super::super::super::arm_chroma_full_alloc_failure();
        let err = $walker(&src, false, ColorMatrix::Bt601, &mut sink).unwrap_err();
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

      // ---- ChromaDerivedNcl consistency (#302 / #303 cross-feature seam) ----

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn centered_chroma_derived_ncl_uses_matrix_tag_fallback() {
        // The high-bit Yuv420p formats are NOT ChromaDerivedNcl-primaries-wired
        // (only 8-bit Yuv420p got #316). BOTH paths — the default fused 4:2:0
        // kernel AND the centered 4:4:4 kernel — resolve ChromaDerivedNcl via the
        // shared BT.709 matrix-tag fallback (`Coefficients::for_matrix`), IGNORING
        // the ColorSpec primaries, so default and centered stay internally
        // consistent (the centered phase shift is the ONLY difference between
        // them). Full primaries-derived support is a documented Yuv420p-8bit-only
        // follow-up. This guards that consistency AND that the centered path did
        // not accidentally half-adopt primaries on one tier.
        use crate::{ColorInfo, ColorSpec, DynamicRange, PixelFormat, Primaries, Transfer};

        let (yp, up, vp) = ramp_planes_n(MAXV);
        // ChromaDerivedNcl + Bt2020 primaries: were the decode to honour the
        // primaries (it must NOT here), it would diverge from BT.709. The
        // PixelFormat in the spec is cosmetic — the sink consumes only
        // chroma_location + primaries.
        let spec = |loc: ChromaLocation| {
          ColorSpec::from_info(
            PixelFormat::Yuv420p,
            ColorInfo::new(
              Primaries::Bt2020,
              Transfer::Bt709,
              ColorMatrix::ChromaDerivedNcl,
              DynamicRange::Limited,
              loc,
            ),
          )
        };
        // ChromaDerivedNcl(Bt2020) decode via the ColorSpec path.
        let decode_cdn = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_color_spec(spec(loc));
          $walker(&src, false, ColorMatrix::ChromaDerivedNcl, &mut sink).unwrap();
          rgb
        };
        // The BT.709 reference the matrix-tag fallback must equal.
        let decode_bt709 = |loc: ChromaLocation| -> Vec<u8> {
          let src = $Frame::new(&yp, &up, &vp, W, H, W, W / 2, W / 2);
          let mut rgb = std::vec![0u8; (W * H * 3) as usize];
          let mut sink = MixedSinker::<$Marker>::new(W as usize, H as usize)
            .with_rgb(&mut rgb)
            .unwrap()
            .with_chroma_location(loc);
          $walker(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
          rgb
        };

        // Centered: ChromaDerivedNcl(Bt2020) resolves to the BT.709 fallback, NOT
        // the Bt2020-primaries-derived coefficients.
        assert_eq!(
          decode_cdn(ChromaLocation::Center),
          decode_bt709(ChromaLocation::Center),
          "centered high-bit ChromaDerivedNcl must resolve via the BT.709 matrix-tag fallback"
        );
        // Default (co-sited): same fallback → default and centered agree on the
        // coefficient path (neither half-adopts primaries).
        assert_eq!(
          decode_cdn(ChromaLocation::Left),
          decode_bt709(ChromaLocation::Left),
          "default high-bit ChromaDerivedNcl must resolve via the same BT.709 fallback"
        );
      }
    }
  };
}

hibit_420_chroma_tests!(
  p9,
  9,
  Yuv420p9,
  Yuv420p9Frame,
  yuv420p9_to,
  Yuv444p9,
  Yuv444p9Frame,
  yuv444p9_to,
  Yuv420p9<true>,
  Yuv420p9BeFrame,
  yuv420p9_to_endian
);
hibit_420_chroma_tests!(
  p10,
  10,
  Yuv420p10,
  Yuv420p10Frame,
  yuv420p10_to,
  Yuv444p10,
  Yuv444p10Frame,
  yuv444p10_to,
  Yuv420p10<true>,
  Yuv420p10BeFrame,
  yuv420p10_to_endian
);
hibit_420_chroma_tests!(
  p12,
  12,
  Yuv420p12,
  Yuv420p12Frame,
  yuv420p12_to,
  Yuv444p12,
  Yuv444p12Frame,
  yuv444p12_to,
  Yuv420p12<true>,
  Yuv420p12BeFrame,
  yuv420p12_to_endian
);
hibit_420_chroma_tests!(
  p14,
  14,
  Yuv420p14,
  Yuv420p14Frame,
  yuv420p14_to,
  Yuv444p14,
  Yuv444p14Frame,
  yuv444p14_to,
  Yuv420p14<true>,
  Yuv420p14BeFrame,
  yuv420p14_to_endian
);
hibit_420_chroma_tests!(
  p16,
  16,
  Yuv420p16,
  Yuv420p16Frame,
  yuv420p16_to,
  Yuv444p16,
  Yuv444p16Frame,
  yuv444p16_to,
  Yuv420p16<true>,
  Yuv420p16BeFrame,
  yuv420p16_to_endian
);
