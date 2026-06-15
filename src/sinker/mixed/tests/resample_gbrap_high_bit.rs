//! Alpha-aware fused-downscale coverage for the high-bit planar GBR+alpha
//! family (`Gbrap10` / `Gbrap12` / `Gbrap14` / `Gbrap16`).
//!
//! Each `GbrapN` de-interleaves its native-depth G/B/R/A planes into the
//! canonical host-native `R, G, B, A` u16 row the high-bit packed-RGBA
//! sources stage (`gbra_to_rgba_u16_high_bit_row`) and feeds the **same**
//! 4-channel high-bit packed-RGBA resample tail
//! (`packed_rgba_u16_resample::<BITS>`) — so this suite is the planar twin
//! of `resample_packed_rgba_16bit`, asserting the identical alpha contract
//! at the source's native depth:
//! - native `rgba_u16` is the exact 2x2 native block mean (alpha averaged,
//!   not forced opaque);
//! - the u8 / narrowed outputs derive from a single `>> (BITS - 8)`
//!   narrowing of the straight color;
//! - premultiplied bins premultiplied color and un-premultiplies against
//!   the **native** max `(1 << BITS) - 1` (transparent pixels never bleed);
//! - LE and BE wire encodings produce byte-identical output.
//!
//! `GbrapNRow::new` is `pub(crate)` in `mediaframe`, so (as with the
//! `resample_gbrp` / `resample_gbr_high_bit` suites) a high-bit GBR+alpha
//! row can only reach `process` through the in-order walker; the mid-frame
//! alpha-mode-freeze / out-of-sequence rejections are covered by the
//! shared-tail `resample_packed_rgba_16bit` suite against the exact same
//! `check_frozen_alpha_mode` / `packed_rgba_u16_resample` functions.

use super::*;
use crate::{ColorMatrix, resample::AreaResampler, sinker::AlphaMode, sinker::MixedSinker};

const SRC: usize = 8;
const OUT: usize = 4;
const MATRIX: ColorMatrix = ColorMatrix::Bt709;

/// Pseudo-random canonical host-native `R, G, B, A` plane, every sample
/// masked to `BITS` so it is a legal native code; alpha varies.
fn canonical_frame<const BITS: u32>(seed: u32) -> Vec<u16> {
  let mut buf = std::vec![0u16; SRC * SRC * 4];
  pseudo_random_u16_low_n_bits(&mut buf, seed, BITS);
  buf
}

/// Scatter a canonical interleaved `R, G, B, A` u16 plane into the four
/// `(g, b, r, a)` planes a `GbrapHighBitFrame` carries.
fn planes_from_canonical(rgba: &[u16], n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
  let (mut g, mut b, mut r, mut a) = (
    std::vec![0u16; n],
    std::vec![0u16; n],
    std::vec![0u16; n],
    std::vec![0u16; n],
  );
  for i in 0..n {
    r[i] = rgba[i * 4];
    g[i] = rgba[i * 4 + 1];
    b[i] = rgba[i * 4 + 2];
    a[i] = rgba[i * 4 + 3];
  }
  (g, b, r, a)
}

/// Re-encode host-native u16 to wire byte storage per endianness, so a
/// fixture reads back identically on LE/BE hosts.
fn as_wire(host: &[u16], be: bool) -> Vec<u16> {
  host
    .iter()
    .map(|&v| if be { as_be_u16(v) } else { as_le_u16(v) })
    .collect()
}

/// Round-half-up 2x2 block mean over native u16 canonical RGBA, all four
/// channels (alpha included).
fn block_mean_rgba(src: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; OUT * OUT * 4];
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        let mut acc = 0u64;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += src[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c] as u64;
          }
        }
        out[(oy * OUT + ox) * 4 + c] = ((acc + 2) / 4) as u16;
      }
    }
  }
  out
}

/// Premultiply one canonical RGBA plane in place against `max`.
fn premultiply(plane: &mut [u16], max: u32) {
  for px in plane.chunks_exact_mut(4) {
    let a = px[3] as u32;
    for c in &mut px[..3] {
      *c = ((*c as u32 * a + max / 2) / max) as u16;
    }
  }
}

/// Un-premultiply one binned canonical RGBA plane against `max`.
fn unpremultiply(plane: &[u16], max: u32) -> Vec<u16> {
  let mut out = std::vec![0u16; plane.len()];
  for (o, i) in out.chunks_exact_mut(4).zip(plane.chunks_exact(4)) {
    let a = i[3] as u32;
    for c in 0..3 {
      o[c] = (i[c] as u32 * max + a / 2)
        .checked_div(a)
        .map_or(0, |q| q.min(max)) as u16;
    }
    o[3] = i[3];
  }
  out
}

/// Drop alpha from a canonical RGBA u16 plane → packed RGB u16.
fn drop_alpha_u16(rgba: &[u16]) -> Vec<u16> {
  let mut out = std::vec![0u16; rgba.len() / 4 * 3];
  for (o, i) in out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)) {
    o.copy_from_slice(&i[..3]);
  }
  out
}

/// Narrow a canonical RGBA u16 plane to u8 via `>> shift`.
fn narrow_rgba_u8(rgba: &[u16], shift: u32) -> Vec<u8> {
  rgba.iter().map(|&v| (v >> shift) as u8).collect()
}

/// Narrow an RGB u16 plane to u8 via `>> shift`.
fn narrow_rgb_u8(rgb_u16: &[u16], shift: u32) -> Vec<u8> {
  rgb_u16.iter().map(|&v| (v >> shift) as u8).collect()
}

macro_rules! gbrap_high_bit_resample_suite {
  (
    $modname:ident,
    $marker:ident,
    $le_frame:ident,
    $be_frame:ident,
    $walk:ident,
    $walk_endian:ident,
    $gbrp_marker:ident,
    $gbrp_walk:ident,
    $bits:literal
  ) => {
    mod $modname {
      use super::*;

      const BITS: u32 = $bits;
      const MAX: u32 = (1u32 << BITS) - 1;
      const SHIFT: u32 = BITS - 8;

      fn le_frame<'a>(
        g: &'a [u16],
        b: &'a [u16],
        r: &'a [u16],
        a: &'a [u16],
        w: usize,
        h: usize,
      ) -> crate::frame::$le_frame<'a> {
        crate::frame::$le_frame::try_new(
          g, b, r, a, w as u32, h as u32, w as u32, w as u32, w as u32, w as u32,
        )
        .unwrap()
      }

      /// Native-u16 canonical RGBA of the source — a direct (identity)
      /// `GbrapN` conversion over an LE-wire frame. The oracles consume it.
      fn direct_rgba_u16(host: &[u16]) -> Vec<u16> {
        let (g, b, r, a) = planes_from_canonical(host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);
        let mut rgba_u16 = std::vec![0u16; SRC * SRC * 4];
        let mut sink = MixedSinker::<crate::source::$marker<false>>::new(SRC, SRC)
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
        crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        rgba_u16
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_rgba_u16_is_block_mean_of_direct() {
        let host = canonical_frame::<BITS>(0x51A1);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&host)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_alpha_is_averaged_not_forced_opaque() {
        let mut host = canonical_frame::<BITS>(0x9E37);
        for (i, px) in host.chunks_exact_mut(4).enumerate() {
          px[3] = ((i as u32 * 37) & MAX) as u16;
        }
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&host)), "block mean");
        assert!(
          rgba_u16.chunks_exact(4).any(|px| px[3] != MAX as u16),
          "resampled alpha was forced opaque — area-mean alpha lost"
        );
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn straight_all_outputs_derive_from_binned_color() {
        // Every output flavour at once: native u16 copies the binned color,
        // the u8 / narrowed outputs derive from `>> (BITS - 8)`, and luma /
        // hsv match a direct GbrpN conversion of the binned RGB.
        let host = canonical_frame::<BITS>(0xBEEF);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgb = std::vec![0u8; OUT * OUT * 3];
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut luma = std::vec![0u8; OUT * OUT];
        let mut lu16 = std::vec![0u16; OUT * OUT];
        let mut h = std::vec![0u8; OUT * OUT];
        let mut s = std::vec![0u8; OUT * OUT];
        let mut v = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgb(&mut rgb)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_luma(&mut luma)
          .unwrap()
          .with_luma_u16(&mut lu16)
          .unwrap()
          .with_hsv(&mut h, &mut s, &mut v)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }

        let binned = block_mean_rgba(&direct_rgba_u16(&host));
        assert_eq!(rgba_u16, binned, "rgba_u16 == native block mean");
        assert_eq!(rgb_u16, drop_alpha_u16(&binned), "rgb_u16 == drop-alpha(binned)");
        assert_eq!(rgba, narrow_rgba_u8(&binned, SHIFT), "rgba == narrowed binned");
        assert_eq!(
          rgb,
          narrow_rgb_u8(&drop_alpha_u16(&binned), SHIFT),
          "rgb == narrowed drop-alpha(binned)"
        );

        // luma / luma_u16 / hsv from a direct GbrpN conversion of the binned
        // RGB (GBR->luma == RGB->luma; luma_u16 at native precision).
        let binned_rgb = drop_alpha_u16(&binned);
        let (bg, bb, br) = planes_from_packed_rgb_u16(&binned_rgb, OUT * OUT);
        let (bgw, bbw, brw) = (as_wire(&bg, false), as_wire(&bb, false), as_wire(&br, false));
        let binned_src = crate::frame::GbrpHighBitFrame::<BITS>::try_new(
          &bgw,
          &bbw,
          &brw,
          OUT as u32,
          OUT as u32,
          OUT as u32,
          OUT as u32,
          OUT as u32,
        )
        .unwrap();
        let mut luma_ref = std::vec![0u8; OUT * OUT];
        let mut lu16_ref = std::vec![0u16; OUT * OUT];
        let mut h_ref = std::vec![0u8; OUT * OUT];
        let mut s_ref = std::vec![0u8; OUT * OUT];
        let mut v_ref = std::vec![0u8; OUT * OUT];
        {
          let mut sink = MixedSinker::<crate::source::$gbrp_marker<false>>::new(OUT, OUT)
            .with_luma(&mut luma_ref)
            .unwrap()
            .with_luma_u16(&mut lu16_ref)
            .unwrap()
            .with_hsv(&mut h_ref, &mut s_ref, &mut v_ref)
            .unwrap();
          crate::source::$gbrp_walk(&binned_src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(luma, luma_ref, "luma (narrowed)");
        assert_eq!(lu16, lu16_ref, "luma_u16 (native, full parity)");
        assert_eq!(h, h_ref, "hsv H");
        assert_eq!(s, s_ref, "hsv S");
        assert_eq!(v, v_ref, "hsv V");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_matches_premult_bin_unpremult_oracle() {
        let host = canonical_frame::<BITS>(0x1234);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        let mut rgb_u16 = std::vec![0u16; OUT * OUT * 3];
        let mut rgba = std::vec![0u8; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgb_u16(&mut rgb_u16)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }

        let mut pm = direct_rgba_u16(&host);
        premultiply(&mut pm, MAX);
        let binned = block_mean_rgba(&pm);
        let oracle = unpremultiply(&binned, MAX);
        assert_eq!(rgba_u16, oracle, "premult rgba_u16");
        assert_eq!(rgb_u16, drop_alpha_u16(&oracle), "premult rgb_u16");
        assert_eq!(rgba, narrow_rgba_u8(&oracle, SHIFT), "premult rgba (narrowed)");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn premultiplied_transparent_block_does_not_bleed() {
        let mut host = canonical_frame::<BITS>(0xABCD);
        let hi = MAX as u16;
        for off in [(0, 0), (1, 0), (0, 1), (1, 1)] {
          let i = off.1 * SRC + off.0;
          host[i * 4] = hi - 100;
          host[i * 4 + 1] = hi - 200;
          host[i * 4 + 2] = hi - 300;
          host[i * 4 + 3] = 0;
        }
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_alpha_mode(AlphaMode::Premultiplied)
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(&rgba_u16[..4], &[0, 0, 0, 0], "transparent block bled color");

        let mut pm = direct_rgba_u16(&host);
        premultiply(&mut pm, MAX);
        let oracle = unpremultiply(&block_mean_rgba(&pm), MAX);
        assert_eq!(rgba_u16, oracle, "premult output != oracle");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn le_be_parity() {
        // The same host values through LE and BE wire frames must produce
        // byte-identical resampled output.
        let host = canonical_frame::<BITS>(0xC0DE);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);

        let render_le = || {
          let (gw, bw, rw, aw) = (
            as_wire(&g, false),
            as_wire(&b, false),
            as_wire(&r, false),
            as_wire(&a, false),
          );
          let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);
          let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
          let mut rgba = std::vec![0u8; OUT * OUT * 4];
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
          (rgba_u16, rgba)
        };
        let render_be = || {
          let (gw, bw, rw, aw) = (
            as_wire(&g, true),
            as_wire(&b, true),
            as_wire(&r, true),
            as_wire(&a, true),
          );
          let src = crate::frame::$be_frame::try_new(
            &gw, &bw, &rw, &aw, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
            SRC as u32,
          )
          .unwrap();
          let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
          let mut rgba = std::vec![0u8; OUT * OUT * 4];
          let mut sink = MixedSinker::<crate::source::$marker<true>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap()
          .with_rgba(&mut rgba)
          .unwrap();
          crate::source::$walk_endian(&src, true, MATRIX, &mut sink).unwrap();
          (rgba_u16, rgba)
        };
        assert_eq!(render_le(), render_be(), "LE/BE outputs diverge");
      }

      #[test]
      fn default_alpha_mode_is_straight() {
        let sink = MixedSinker::<crate::source::$marker<false>>::new(SRC, SRC);
        assert_eq!(sink.alpha_mode(), AlphaMode::Straight);
        assert!(sink.alpha_mode().is_straight());
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn identity_plan_matches_direct() {
        let host = canonical_frame::<BITS>(0x0F0F);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; SRC * SRC * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(SRC, SRC),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, direct_rgba_u16(&host), "identity plan == direct");
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn cross_frame_reset_reuses_streams() {
        let host = canonical_frame::<BITS>(0x5151);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
        }
        assert_eq!(rgba_u16, block_mean_rgba(&direct_rgba_u16(&host)));
      }

      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn accepts_alpha_mode_change_across_frames() {
        // begin_frame (re-run by the walker each frame) re-arms the frozen
        // mode, so frame 2 may flip to Premultiplied without a false
        // ResampleOutputsChanged.
        let host = canonical_frame::<BITS>(0xB2B2);
        let (g, b, r, a) = planes_from_canonical(&host, SRC * SRC);
        let (gw, bw, rw, aw) = (
          as_wire(&g, false),
          as_wire(&b, false),
          as_wire(&r, false),
          as_wire(&a, false),
        );
        let src = le_frame(&gw, &bw, &rw, &aw, SRC, SRC);

        let mut rgba_u16 = std::vec![0u16; OUT * OUT * 4];
        {
          let mut sink = MixedSinker::<crate::source::$marker<false>, AreaResampler>::with_resampler(
            SRC,
            SRC,
            AreaResampler::to(OUT, OUT),
          )
          .unwrap()
          .with_rgba_u16(&mut rgba_u16)
          .unwrap();
          crate::source::$walk(&src, true, MATRIX, &mut sink).unwrap();
          sink.set_alpha_mode(AlphaMode::Premultiplied);
          crate::source::$walk(&src, true, MATRIX, &mut sink)
            .expect("a fresh frame must accept a different alpha mode");
        }
        let mut pm = direct_rgba_u16(&host);
        premultiply(&mut pm, MAX);
        let oracle = unpremultiply(&block_mean_rgba(&pm), MAX);
        assert_eq!(rgba_u16, oracle, "premult frame 2 output");
      }
    }
  };
}

/// Scatter a packed-RGB u16 buffer into `(g, b, r)` planes — shared by the
/// per-format suites' luma/hsv reference (binned RGB → GbrpN).
fn planes_from_packed_rgb_u16(rgb: &[u16], n: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
  let (mut g, mut b, mut r) = (std::vec![0u16; n], std::vec![0u16; n], std::vec![0u16; n]);
  for i in 0..n {
    r[i] = rgb[i * 3];
    g[i] = rgb[i * 3 + 1];
    b[i] = rgb[i * 3 + 2];
  }
  (g, b, r)
}

gbrap_high_bit_resample_suite!(
  gbrap10,
  Gbrap10,
  Gbrap10LeFrame,
  Gbrap10BeFrame,
  gbrap10_to,
  gbrap10_to_endian,
  Gbrp10,
  gbrp10_to,
  10
);
gbrap_high_bit_resample_suite!(
  gbrap12,
  Gbrap12,
  Gbrap12LeFrame,
  Gbrap12BeFrame,
  gbrap12_to,
  gbrap12_to_endian,
  Gbrp12,
  gbrp12_to,
  12
);
gbrap_high_bit_resample_suite!(
  gbrap14,
  Gbrap14,
  Gbrap14LeFrame,
  Gbrap14BeFrame,
  gbrap14_to,
  gbrap14_to_endian,
  Gbrp14,
  gbrp14_to,
  14
);
gbrap_high_bit_resample_suite!(
  gbrap16,
  Gbrap16,
  Gbrap16LeFrame,
  Gbrap16BeFrame,
  gbrap16_to,
  gbrap16_to_endian,
  Gbrp16,
  gbrp16_to,
  16
);
