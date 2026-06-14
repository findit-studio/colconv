//! Fused-downscale coverage for the high-bit planar GBR family
//! (`Gbrp9` / `Gbrp10` / `Gbrp12` / `Gbrp14` / `Gbrp16`).
//!
//! Each `GbrpN` scatters its native-depth G/B/R planes into a source-width
//! packed `u16` RGB row and feeds the shared high-bit packed-RGB resample
//! tail (the same one `Rgb48` / `Bgr48` use, parameterized by the source
//! depth `BITS`). Binning runs at native depth, so:
//! - `rgb_u16` is the exact native 2x2 block mean,
//! - every output (rgb / rgba / rgb_u16 / rgba_u16 / luma / luma_u16 / hsv)
//!   matches a **direct** full-resolution `GbrpN` conversion of the
//!   pre-binned frame — `luma_u16` at native precision, full parity.
//!
//! The out-of-sequence / mid-frame contract is exercised by the shared
//! tail's `resample_rgb48` suite against the exact same stream/preflight
//! functions; `GbrpNRow::new` is `pub(crate)` in `mediaframe`, so a high-bit
//! GBR row can only reach `process` through the in-order walker and a direct
//! out-of-order `process` call cannot be constructed here (mirrors the 8-bit
//! `resample_gbrp` suite).

use crate::{ColorMatrix, sinker::MixedSinker};

const SRC: usize = 8;
const OUT: usize = 4;
const MATRIX: ColorMatrix = ColorMatrix::Bt709;

/// Native-depth `(r, g, b)` ramp for source pixel `i`, masked to `BITS` so
/// every sample is a legal native code; interior values so the derived luma
/// / HSV kernels see real math and the wide accumulator carries bits a u8
/// path would drop.
fn rgb_px<const BITS: u32>(i: usize) -> [u16; 3] {
  let mask = (1u32 << BITS) - 1;
  let r = (40u32 * BITS + (i as u32) * 173) & mask;
  let g = mask.wrapping_sub((i as u32) * 211) & mask;
  let b = (1000u32 + (i as u32 % 8) * 4099) & mask;
  [r as u16, g as u16, b as u16]
}

/// Source-width packed native-u16 RGB ramp (`SRC * SRC * 3` elements).
fn rgb_ramp<const BITS: u32>() -> Vec<u16> {
  let mut buf = std::vec![0u16; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px.copy_from_slice(&rgb_px::<BITS>(i));
  }
  buf
}

/// Scatter a packed-RGB u16 buffer into `(g, b, r)` planes — the inverse of
/// `gbr_to_rgb_u16_high_bit_row`. Each plane has `width * height` elements.
fn planes_from_packed_rgb(rgb: &[u16], n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let (mut g, mut b, mut r) = (std::vec![0u16; n], std::vec![0u16; n], std::vec![0u16; n]);
  for i in 0..n {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }
  (g, b, r)
}

/// Exact 2x2 block mean with round-half-up over native u16 values — the
/// integer-area-mean contract for a 2:1 downscale at native depth.
fn expected_block_mean(rgb: &[u16], ox: usize, oy: usize, c: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += rgb[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u64;
    }
  }
  ((acc + 2) / 4) as u16
}

macro_rules! gbr_high_bit_resample_tests {
  ($mod:ident, $marker:ident, $walker:ident, $bits:literal) => {
    mod $mod {
      use super::*;

      fn frame<'a>(
        g: &'a [u16],
        b: &'a [u16],
        r: &'a [u16],
        w: usize,
        h: usize,
      ) -> crate::frame::GbrpHighBitFrame<'a, $bits> {
        crate::frame::GbrpHighBitFrame::try_new(g, b, r, w as u32, h as u32, w as u32, w as u32, w as u32)
          .unwrap()
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn downscale_rgb_u16_is_exact_native_area_mean() {
        let rgb = rgb_ramp::<$bits>();
        let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
        let src = frame(&g, &b, &r, SRC, SRC);

        let mut out = std::vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
            SRC,
            SRC,
            crate::resample::AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb_u16(&mut out)
          .unwrap();
          crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        for oy in 0..OUT {
          for ox in 0..OUT {
            for c in 0..3 {
              assert_eq!(
                out[(oy * OUT + ox) * 3 + c],
                expected_block_mean(&rgb, ox, oy, c),
                "({ox},{oy}) c{c}"
              );
            }
          }
        }
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn all_outputs_match_direct_conversion_of_prebinned_frame() {
        // Resample SRC->OUT with every output attached, then compare against
        // a full-resolution direct GbrpN conversion of the pre-binned
        // (native block-mean) frame — the parity oracle. Every output,
        // luma_u16 included at native precision, matches the direct path.
        let rgb = rgb_ramp::<$bits>();
        let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
        let src = frame(&g, &b, &r, SRC, SRC);

        let mut rgb_o = std::vec![0u8; OUT * OUT * 3];
        let mut rgb_u16_o = std::vec![0u16; OUT * OUT * 3];
        let mut rgba_o = std::vec![0u8; OUT * OUT * 4];
        let mut rgba_u16_o = std::vec![0u16; OUT * OUT * 4];
        let mut luma_o = std::vec![0u8; OUT * OUT];
        let mut lu16_o = std::vec![0u16; OUT * OUT];
        let mut h_o = std::vec![0u8; OUT * OUT];
        let mut s_o = std::vec![0u8; OUT * OUT];
        let mut v_o = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
            SRC,
            SRC,
            crate::resample::AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb(&mut rgb_o)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16_o)
          .unwrap()
          .with_rgba(&mut rgba_o)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16_o)
          .unwrap()
          .with_luma(&mut luma_o)
          .unwrap()
          .with_luma_u16(&mut lu16_o)
          .unwrap()
          .with_hsv(&mut h_o, &mut s_o, &mut v_o)
          .unwrap();
          crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
        }

        // The resampled rgb_u16 IS the exact native block mean; assert that
        // link explicitly, then drive the oracle from the same samples.
        let mut binned = std::vec![0u16; OUT * OUT * 3];
        for oy in 0..OUT {
          for ox in 0..OUT {
            for c in 0..3 {
              binned[(oy * OUT + ox) * 3 + c] = expected_block_mean(&rgb, ox, oy, c);
            }
          }
        }
        assert_eq!(rgb_u16_o, binned, "resample rgb_u16 == exact native block-mean");

        let (bg, bb, br) = planes_from_packed_rgb(&binned, OUT * OUT);
        let binned_src = frame(&bg, &bb, &br, OUT, OUT);
        let mut rgb_ref = std::vec![0u8; OUT * OUT * 3];
        let mut rgba_ref = std::vec![0u8; OUT * OUT * 4];
        let mut rgba_u16_ref = std::vec![0u16; OUT * OUT * 4];
        let mut luma_ref = std::vec![0u8; OUT * OUT];
        let mut lu16_ref = std::vec![0u16; OUT * OUT];
        let mut h_ref = std::vec![0u8; OUT * OUT];
        let mut s_ref = std::vec![0u8; OUT * OUT];
        let mut v_ref = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<crate::source::$marker>::new(OUT, OUT)
            .with_rgb(&mut rgb_ref)
            .unwrap()
            .with_rgba(&mut rgba_ref)
            .unwrap()
            .with_rgba_u16(&mut rgba_u16_ref)
            .unwrap()
            .with_luma(&mut luma_ref)
            .unwrap()
            .with_luma_u16(&mut lu16_ref)
            .unwrap()
            .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
            .unwrap();
          crate::source::$walker(&binned_src, true, MATRIX, &mut sink).unwrap();
        }

        assert_eq!(rgb_o, rgb_ref, "rgb (narrowed)");
        assert_eq!(rgba_o, rgba_ref, "rgba (narrowed, alpha forced max)");
        assert_eq!(rgba_u16_o, rgba_u16_ref, "rgba_u16 (native, alpha forced max)");
        assert_eq!(luma_o, luma_ref, "luma (narrowed)");
        assert_eq!(h_o, h_ref, "hsv H");
        assert_eq!(s_o, s_ref, "hsv S");
        assert_eq!(v_o, v_ref, "hsv V");
        // luma_u16 on the fused path is native-precision — byte-identical
        // to the direct GbrpN with_luma_u16 of the binned frame.
        assert_eq!(lu16_o, lu16_ref, "luma_u16 (native, full parity)");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_new_sink() {
        let rgb = rgb_ramp::<$bits>();
        let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
        let src = frame(&g, &b, &r, SRC, SRC);

        let mut direct = std::vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<crate::source::$marker>::new(SRC, SRC)
            .with_rgb_u16(&mut direct)
            .unwrap();
          crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        let mut via_area = std::vec![0u16; SRC * SRC * 3];
        {
          let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
            SRC,
            SRC,
            crate::resample::AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgb_u16(&mut via_area)
          .unwrap();
          crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(direct, via_area, "identity-plan resample == direct sink");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resample_no_outputs_is_a_no_op() {
        // A resampling sink with no attached outputs is the documented legal
        // no-op: it walks every row and returns Ok without touching any
        // caller buffer (there is none to touch).
        let rgb = rgb_ramp::<$bits>();
        let (g, b, r) = planes_from_packed_rgb(&rgb, SRC * SRC);
        let src = frame(&g, &b, &r, SRC, SRC);

        let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
          SRC,
          SRC,
          crate::resample::AreaResampler::to(OUT, OUT),
        )
        .unwrap();
        crate::source::$walker(&src, true, MATRIX, &mut sink).unwrap();
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn resample_reuses_stream_across_frames() {
        // begin_frame resets the u16 area stream + frozen output set, so
        // frame 2's row 0 is accepted (not rejected as out-of-sequence) and
        // the output reflects frame 2's input — without the reset it would
        // still show frame 1. Both frames share one output buffer; only the
        // input data changes.
        let mask = ((1u32 << $bits) - 1) as u16;
        let rgb1 = rgb_ramp::<$bits>();
        let rgb2: Vec<u16> = rgb1.iter().map(|&p| mask - p).collect();
        let (g1, b1, r1) = planes_from_packed_rgb(&rgb1, SRC * SRC);
        let (g2, b2, r2) = planes_from_packed_rgb(&rgb2, SRC * SRC);

        let mut out = std::vec![0u16; OUT * OUT * 3];
        {
          let mut sink = MixedSinker::<crate::source::$marker, crate::resample::AreaResampler>::with_resampler(
            SRC,
            SRC,
            crate::resample::AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb_u16(&mut out)
          .unwrap();
          crate::source::$walker(&frame(&g1, &b1, &r1, SRC, SRC), true, MATRIX, &mut sink).unwrap();
          crate::source::$walker(&frame(&g2, &b2, &r2, SRC, SRC), true, MATRIX, &mut sink).unwrap();
        }

        let mut expected = std::vec![0u16; OUT * OUT * 3];
        for oy in 0..OUT {
          for ox in 0..OUT {
            for c in 0..3 {
              expected[(oy * OUT + ox) * 3 + c] = expected_block_mean(&rgb2, ox, oy, c);
            }
          }
        }
        assert_eq!(out, expected, "frame 2 output must area-downscale frame 2");
      }
    }
  };
}

gbr_high_bit_resample_tests!(gbrp9, Gbrp9, gbrp9_to, 9);
gbr_high_bit_resample_tests!(gbrp10, Gbrp10, gbrp10_to, 10);
gbr_high_bit_resample_tests!(gbrp12, Gbrp12, gbrp12_to, 12);
gbr_high_bit_resample_tests!(gbrp14, Gbrp14, gbrp14_to, 14);
gbr_high_bit_resample_tests!(gbrp16, Gbrp16, gbrp16_to, 16);
