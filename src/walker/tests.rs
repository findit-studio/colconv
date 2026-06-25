use super::*;

// ---- Structural round-trip: Options builders / getters ----------------

#[test]
fn xyz12_options_default_is_dcip3() {
  assert_eq!(Xyz12Options::default(), Xyz12Options::new());
  assert_eq!(Xyz12Options::new().target_gamut(), DcpTargetGamut::DciP3);
  // The honest default delegates to mediaframe's own gamut default.
  assert_eq!(
    Xyz12Options::default().target_gamut(),
    DcpTargetGamut::default()
  );
}

#[test]
fn xyz12_options_with_target_gamut_round_trips() {
  for g in [
    DcpTargetGamut::DciP3,
    DcpTargetGamut::Rec709,
    DcpTargetGamut::Rec2020,
  ] {
    assert_eq!(Xyz12Options::new().with_target_gamut(g).target_gamut(), g);
  }
}

#[test]
fn yuv_options_default_is_limited_bt709() {
  assert_eq!(YuvOptions::default(), YuvOptions::new());
  assert!(!YuvOptions::new().full_range());
  assert_eq!(YuvOptions::new().matrix(), ColorMatrix::Bt709);
}

#[test]
fn yuv_options_builders_and_mutators_round_trip() {
  // with_matrix
  assert_eq!(
    YuvOptions::new().with_matrix(ColorMatrix::Bt601).matrix(),
    ColorMatrix::Bt601
  );

  // bool consuming builders
  assert!(YuvOptions::new().with_full_range().full_range());
  assert!(YuvOptions::new().maybe_full_range(true).full_range());
  assert!(!YuvOptions::new().maybe_full_range(false).full_range());

  // bool in-place mutators (chainable via &mut Self)
  let mut o = YuvOptions::new();
  o.set_full_range();
  assert!(o.full_range());
  o.clear_full_range();
  assert!(!o.full_range());
  o.update_full_range(true);
  assert!(o.full_range());
}

#[cfg(feature = "bayer")]
mod bayer_options {
  use super::*;
  use crate::raw::{BayerDemosaic, BayerPattern, ColorCorrectionMatrix, WhiteBalance};

  #[test]
  fn new_defaults_demosaic_wb_ccm() {
    let o = BayerOptions::new(BayerPattern::Rggb);
    assert_eq!(o.pattern(), BayerPattern::Rggb);
    assert_eq!(o.demosaic(), BayerDemosaic::Bilinear);
    assert_eq!(o.wb(), WhiteBalance::neutral());
    assert_eq!(o.ccm(), ColorCorrectionMatrix::identity());
  }

  #[test]
  fn builders_round_trip() {
    let wb = WhiteBalance::new(1.5, 1.0, 2.0);
    let ccm = ColorCorrectionMatrix::new([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
    let o = BayerOptions::new(BayerPattern::Bggr)
      .with_demosaic(BayerDemosaic::Bilinear)
      .with_wb(wb)
      .with_ccm(ccm);
    assert_eq!(o.pattern(), BayerPattern::Bggr);
    assert_eq!(o.demosaic(), BayerDemosaic::Bilinear);
    assert_eq!(o.wb(), wb);
    assert_eq!(o.ccm(), ccm);
  }
}

// ---- Parity: Walker::walk byte-identical to xyz12_to directly ---------

#[cfg(feature = "xyz")]
mod xyz12_parity {
  use super::*;
  use crate::{
    frame::{Xyz12BeFrame, Xyz12LeFrame},
    sinker::MixedSinker,
    source::{Xyz12Be, Xyz12Le, xyz12_to},
  };

  /// Packs a 12-bit code into the high-bit-packed LE wire `u16`
  /// (active bits in `[15:4]`), host-independent.
  fn pack12_le(code: u16) -> u16 {
    u16::from_ne_bytes((code << 4).to_le_bytes())
  }

  /// BE-wire variant of [`pack12_le`].
  fn pack12_be(code: u16) -> u16 {
    u16::from_ne_bytes((code << 4).to_be_bytes())
  }

  /// A small ramp frame so different rows / columns carry different
  /// XYZ triples (exercises a non-degenerate conversion).
  fn ramp_frame<F>(width: u32, height: u32, pack: F) -> std::vec::Vec<u16>
  where
    F: Fn(u16) -> u16,
  {
    let w = width as usize;
    let h = height as usize;
    let mut buf = std::vec![0u16; w * h * 3];
    for (i, px) in buf.chunks_mut(3).enumerate() {
      let base = ((i * 37) % 4096) as u16;
      px[0] = pack(base);
      px[1] = pack((base + 411) % 4096);
      px[2] = pack((base + 822) % 4096);
    }
    buf
  }

  const W: u32 = 8;
  const H: u32 = 4;

  /// Asserts the LE u8-RGB output of `Walker::walk` equals `xyz12_to`.
  fn assert_parity_rgb_u8_le(gamut: DcpTargetGamut) {
    let pix = ramp_frame(W, H, pack12_le);
    let src = Xyz12LeFrame::try_new(&pix, W, H, W * 3).unwrap();
    let opts = Xyz12Options::new().with_target_gamut(gamut);

    let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
    let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

    let mut sink_w = MixedSinker::<Xyz12Le>::new(W as usize, H as usize)
      .with_rgb(&mut via_walker)
      .unwrap();
    <Xyz12<false> as Walker<_>>::walk(&src, &opts, &mut sink_w).unwrap();

    let mut sink_d = MixedSinker::<Xyz12Le>::new(W as usize, H as usize)
      .with_rgb(&mut via_direct)
      .unwrap();
    xyz12_to(&src, gamut, &mut sink_d).unwrap();

    assert_eq!(via_walker, via_direct, "rgb u8 LE parity (gamut {gamut:?})");
  }

  /// Asserts the BE u16-RGB output of `Walker::walk` equals `xyz12_to`.
  fn assert_parity_rgb_u16_be(gamut: DcpTargetGamut) {
    let pix = ramp_frame(W, H, pack12_be);
    let src = Xyz12BeFrame::try_new(&pix, W, H, W * 3).unwrap();
    let opts = Xyz12Options::new().with_target_gamut(gamut);

    let mut via_walker = std::vec![0u16; (W * H * 3) as usize];
    let mut via_direct = std::vec![0u16; (W * H * 3) as usize];

    let mut sink_w = MixedSinker::<Xyz12Be>::new(W as usize, H as usize)
      .with_rgb_u16(&mut via_walker)
      .unwrap();
    <Xyz12<true> as Walker<_>>::walk(&src, &opts, &mut sink_w).unwrap();

    let mut sink_d = MixedSinker::<Xyz12Be>::new(W as usize, H as usize)
      .with_rgb_u16(&mut via_direct)
      .unwrap();
    xyz12_to(&src, gamut, &mut sink_d).unwrap();

    assert_eq!(
      via_walker, via_direct,
      "rgb u16 BE parity (gamut {gamut:?})"
    );
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb_u8_le_matches_direct() {
    // >=2 gamuts x LE + rgb u8.
    assert_parity_rgb_u8_le(DcpTargetGamut::DciP3);
    assert_parity_rgb_u8_le(DcpTargetGamut::Rec709);
    assert_parity_rgb_u8_le(DcpTargetGamut::Rec2020);
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_rgb_u16_be_matches_direct() {
    // >=2 gamuts x BE + rgb u16.
    assert_parity_rgb_u16_be(DcpTargetGamut::DciP3);
    assert_parity_rgb_u16_be(DcpTargetGamut::Rec709);
    assert_parity_rgb_u16_be(DcpTargetGamut::Rec2020);
  }

  /// Cross-check: the LE u16 + BE u8 corners too, so the full
  /// 2-gamut x {LE,BE} x {u8,u16} matrix is covered.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_le_u16_and_be_u8_match_direct() {
    for gamut in [DcpTargetGamut::DciP3, DcpTargetGamut::Rec709] {
      // LE, u16.
      {
        let pix = ramp_frame(W, H, pack12_le);
        let src = Xyz12LeFrame::try_new(&pix, W, H, W * 3).unwrap();
        let opts = Xyz12Options::new().with_target_gamut(gamut);
        let mut via_walker = std::vec![0u16; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u16; (W * H * 3) as usize];
        let mut sw = MixedSinker::<Xyz12Le>::new(W as usize, H as usize)
          .with_rgb_u16(&mut via_walker)
          .unwrap();
        <Xyz12<false> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Xyz12Le>::new(W as usize, H as usize)
          .with_rgb_u16(&mut via_direct)
          .unwrap();
        xyz12_to(&src, gamut, &mut sd).unwrap();
        assert_eq!(
          via_walker, via_direct,
          "rgb u16 LE parity (gamut {gamut:?})"
        );
      }
      // BE, u8.
      {
        let pix = ramp_frame(W, H, pack12_be);
        let src = Xyz12BeFrame::try_new(&pix, W, H, W * 3).unwrap();
        let opts = Xyz12Options::new().with_target_gamut(gamut);
        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];
        let mut sw = MixedSinker::<Xyz12Be>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Xyz12<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();
        let mut sd = MixedSinker::<Xyz12Be>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        xyz12_to(&src, gamut, &mut sd).unwrap();
        assert_eq!(via_walker, via_direct, "rgb u8 BE parity (gamut {gamut:?})");
      }
    }
  }
}

// ---- Parity: Bayer 8-bit + Bayer16 ------------------------------------

#[cfg(feature = "bayer")]
mod bayer_parity {
  use super::*;
  use crate::{
    frame::{
      Bayer10Frame, Bayer12Frame, Bayer14Frame, Bayer16Frame, BayerDemosaic, BayerFrame,
      BayerPattern, ColorCorrectionMatrix, WhiteBalance, bayer_to, bayer16_to,
    },
    sinker::MixedSinker,
    source::{Bayer, Bayer10, Bayer12, Bayer14, Bayer16Bit},
  };

  const W: u32 = 8;
  const H: u32 = 6;

  /// A column/row-varying `u8` Bayer plane (non-degenerate mosaic).
  fn ramp8() -> std::vec::Vec<u8> {
    let mut data = std::vec![0u8; (W * H) as usize];
    for (i, p) in data.iter_mut().enumerate() {
      *p = ((i * 17 + 3) % 251) as u8;
    }
    data
  }

  /// A column/row-varying low-packed `u16` Bayer plane for `BITS`.
  fn ramp16(bits: u32) -> std::vec::Vec<u16> {
    let max = (1u32 << bits) - 1;
    let mut data = std::vec![0u16; (W * H) as usize];
    for (i, p) in data.iter_mut().enumerate() {
      *p = (((i as u32) * 1103 + 7) % (max + 1)) as u16;
    }
    data
  }

  /// A non-neutral white balance + non-identity CCM, so the fused 3×3
  /// is exercised (a neutral/identity pair would hide a mis-forwarded
  /// param).
  fn nontrivial_opts(pattern: BayerPattern) -> BayerOptions {
    BayerOptions::new(pattern)
      .with_demosaic(BayerDemosaic::Bilinear)
      .with_wb(WhiteBalance::new(1.5, 1.0, 1.75))
      .with_ccm(ColorCorrectionMatrix::new([
        [1.2, -0.1, -0.1],
        [-0.2, 1.3, -0.1],
        [-0.1, -0.2, 1.4],
      ]))
  }

  const PATTERNS: [BayerPattern; 4] = [
    BayerPattern::Rggb,
    BayerPattern::Bggr,
    BayerPattern::Grbg,
    BayerPattern::Gbrg,
  ];

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_bayer8_matches_direct() {
    let plane = ramp8();
    for pattern in PATTERNS {
      let opts = nontrivial_opts(pattern);
      let src = BayerFrame::try_new(&plane, W, H, W).unwrap();

      let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
      let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

      let mut sw = MixedSinker::<Bayer>::new(W as usize, H as usize)
        .with_rgb(&mut via_walker)
        .unwrap();
      <Bayer as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

      let mut sd = MixedSinker::<Bayer>::new(W as usize, H as usize)
        .with_rgb(&mut via_direct)
        .unwrap();
      bayer_to(
        &src,
        opts.pattern(),
        opts.demosaic(),
        opts.wb(),
        opts.ccm(),
        &mut sd,
      )
      .unwrap();

      assert_eq!(
        via_walker, via_direct,
        "bayer8 parity (pattern {pattern:?})"
      );
    }
  }

  /// Drives the Bayer16 parity for one concrete `BITS` marker `$marker`
  /// + its `$frame` alias, across all four patterns.
  macro_rules! bayer16_case {
    ($name:ident, $marker:ty, $frame:ident, $bits:expr) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        let plane = ramp16($bits);
        for pattern in PATTERNS {
          let opts = nontrivial_opts(pattern);
          let src = $frame::try_new(&plane, W, H, W).unwrap();

          let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
          let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

          let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
            .with_rgb(&mut via_walker)
            .unwrap();
          <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

          let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
            .with_rgb(&mut via_direct)
            .unwrap();
          bayer16_to::<$bits, _>(
            &src,
            opts.pattern(),
            opts.demosaic(),
            opts.wb(),
            opts.ccm(),
            &mut sd,
          )
          .unwrap();

          assert_eq!(
            via_walker, via_direct,
            "bayer16<{}> parity (pattern {pattern:?})",
            $bits
          );
        }
      }
    };
  }

  bayer16_case!(walk_bayer10_matches_direct, Bayer10, Bayer10Frame, 10);
  bayer16_case!(walk_bayer12_matches_direct, Bayer12, Bayer12Frame, 12);
  bayer16_case!(walk_bayer14_matches_direct, Bayer14, Bayer14Frame, 14);
  bayer16_case!(walk_bayer16_matches_direct, Bayer16Bit, Bayer16Frame, 16);
}

// ---- Parity: Pal8 (palette is frame-intrinsic; Options = ()) ----------

#[cfg(feature = "mono")]
mod pal8_parity {
  use super::*;
  use crate::{
    frame::Pal8Frame,
    sinker::MixedSinker,
    source::{Pal8, pal8_to},
  };

  const W: u32 = 8;
  const H: u32 = 4;

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_pal8_matches_direct() {
    // A non-degenerate BGRA palette + an index ramp that hits a spread
    // of entries.
    let mut palette = [[0u8; 4]; 256];
    for (i, e) in palette.iter_mut().enumerate() {
      let i = i as u8;
      *e = [i, i.wrapping_mul(3), i.wrapping_mul(7), 255];
    }
    let mut data = std::vec![0u8; (W * H) as usize];
    for (i, p) in data.iter_mut().enumerate() {
      *p = ((i * 29 + 5) % 256) as u8;
    }
    let src = Pal8Frame::try_new(&data, &palette, W, H, W).unwrap();

    let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
    let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

    let mut sw = MixedSinker::<Pal8>::new(W as usize, H as usize)
      .with_rgb(&mut via_walker)
      .unwrap();
    // `()` Options — the palette rides on the frame, not the knobs.
    <Pal8 as Walker<_>>::walk(&src, &(), &mut sw).unwrap();

    let mut sd = MixedSinker::<Pal8>::new(W as usize, H as usize)
      .with_rgb(&mut via_direct)
      .unwrap();
    pal8_to(&src, &mut sd).unwrap();

    assert_eq!(via_walker, via_direct, "pal8 rgb parity");
  }
}

// ---- Parity: Monoblack + Monowhite (reuse YuvOptions) -----------------

#[cfg(feature = "mono")]
mod mono_parity {
  use super::*;
  use crate::{
    PixelSink,
    frame::{MonoblackFrame, MonowhiteFrame},
    sinker::MixedSinker,
    source::{
      Monoblack, MonoblackRow, MonoblackSink, Monowhite, MonowhiteRow, MonowhiteSink, monoblack_to,
      monowhite_to,
    },
  };

  const W: u32 = 13; // not a byte multiple → exercises the tail bits
  const H: u32 = 5;

  /// MSB-first 1-bit packed plane, `div_ceil(8)` bytes per row.
  fn packed_1bpp() -> (std::vec::Vec<u8>, u32) {
    let stride = W.div_ceil(8);
    let mut data = std::vec![0u8; (stride * H) as usize];
    for (i, b) in data.iter_mut().enumerate() {
      *b = ((i * 53 + 9) % 256) as u8;
    }
    (data, stride)
  }

  const MATRICES: [ColorMatrix; 2] = [ColorMatrix::Bt709, ColorMatrix::Bt601];

  // An instrumented sink recording each row's forwarded `(full_range, matrix)`.
  // The mono luma path expands bits to 0/255 and ignores that metadata, so a
  // byte-parity test cannot see a dropped forward — this can.
  macro_rules! metadata_probe {
    ($probe:ident, $row:ident, $sink:ident) => {
      #[derive(Default)]
      struct $probe {
        seen: std::vec::Vec<(bool, ColorMatrix)>,
      }
      impl PixelSink for $probe {
        type Input<'r> = $row<'r>;
        type Error = core::convert::Infallible;
        fn begin_frame(&mut self, _w: u32, _h: u32) -> Result<(), Self::Error> {
          Ok(())
        }
        fn process(&mut self, row: $row<'_>) -> Result<(), Self::Error> {
          self.seen.push((row.full_range(), row.matrix()));
          Ok(())
        }
      }
      impl $sink for $probe {}
    };
  }
  metadata_probe!(MonoblackProbe, MonoblackRow, MonoblackSink);
  metadata_probe!(MonowhiteProbe, MonowhiteRow, MonowhiteSink);

  /// The luma path discards `full_range`/`matrix`, so byte parity can't prove
  /// the Walker forwards them; instrument the sink and assert every emitted row
  /// carries exactly the supplied `YuvOptions` values.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_mono_forwards_full_range_and_matrix() {
    let (data, stride) = packed_1bpp();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);

        let mut bp = MonoblackProbe::default();
        let src = MonoblackFrame::try_new(&data, W, H, stride).unwrap();
        <Monoblack as Walker<_>>::walk(&src, &opts, &mut bp).unwrap();
        assert!(!bp.seen.is_empty(), "monoblack walked at least one row");
        for &(fr, m) in &bp.seen {
          assert_eq!(
            (fr, m),
            (full_range, matrix),
            "monoblack forwards full_range/matrix into the row"
          );
        }

        let mut wp = MonowhiteProbe::default();
        let src = MonowhiteFrame::try_new(&data, W, H, stride).unwrap();
        <Monowhite as Walker<_>>::walk(&src, &opts, &mut wp).unwrap();
        assert!(!wp.seen.is_empty(), "monowhite walked at least one row");
        for &(fr, m) in &wp.seen {
          assert_eq!(
            (fr, m),
            (full_range, matrix),
            "monowhite forwards full_range/matrix into the row"
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
  fn walk_monoblack_matches_direct() {
    let (data, stride) = packed_1bpp();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = MonoblackFrame::try_new(&data, W, H, stride).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Monoblack>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Monoblack as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Monoblack>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        monoblack_to(&src, opts.full_range(), opts.matrix(), &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "monoblack parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_monowhite_matches_direct() {
    let (data, stride) = packed_1bpp();
    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = MonowhiteFrame::try_new(&data, W, H, stride).unwrap();

        let mut via_walker = std::vec![0u8; (W * H) as usize];
        let mut via_direct = std::vec![0u8; (W * H) as usize];

        let mut sw = MixedSinker::<Monowhite>::new(W as usize, H as usize)
          .with_luma(&mut via_walker)
          .unwrap();
        <Monowhite as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Monowhite>::new(W as usize, H as usize)
          .with_luma(&mut via_direct)
          .unwrap();
        monowhite_to(&src, opts.full_range(), opts.matrix(), &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "monowhite parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

// ---- Parity: uniform YUV families (reuse YuvOptions) ------------------
//
// Each test asserts `<Marker as Walker<_>>::walk` is byte-identical to a
// direct walker call — same `MixedSinker::with_rgb` sink, same walker fn
// on both sides — across full/limited × Bt709/Bt601. Coverage spans every
// family plus each const-generic axis: an 8-bit and a high-bit case for
// planar / semi-planar / YUVA, a packed case, and a Y2xx case.
//
// The 8-bit families have no byte-order axis — their walk delegates to
// the plain `{fmt}_to` and the "direct" side calls the same fn, so
// byte-identity is structural. The high-bit families are endian-generic:
// the [`Walker`] impl delegates to the const-generic
// `{fmt}_to_endian::<_, BE>` (the LE `{fmt}_to` is just its `BE = false`
// shim), so **both** ends of the matrix are proven — the LE cases below
// drive the impl at the marker's `<const BE = false>` default against the
// LE frame alias, and the dedicated BE cases drive `Marker<true>` against
// the `{Fmt}BeFrame` alias, each compared to a direct
// `{fmt}_to_endian::<_, BE>` call. The non-degenerate ramps still guard
// against a mis-forwarded `full_range` / `matrix` (a swapped pair changes
// the output).

#[cfg(feature = "yuv-planar")]
mod yuv_planar_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const MATRICES: [ColorMatrix; 2] = [ColorMatrix::Bt709, ColorMatrix::Bt601];

  /// A deterministic, column/row-varying `u8` plane of `n` samples.
  fn ramp8(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 17 + 3) % 251) as u8).collect()
  }

  /// A deterministic low-packed `u16` plane of `n` samples bounded to
  /// `bits` (active bits in the low end, matching the LE wire layout on
  /// the test host).
  fn ramp16(n: usize, bits: u32) -> std::vec::Vec<u16> {
    let max = (1u32 << bits) - 1;
    (0..n)
      .map(|i| (((i as u32) * 1103 + 7) & max) as u16)
      .collect()
  }

  /// Drives a 3-plane planar YUV family (8-bit or high-bit-LE). `$ramp`
  /// builds a plane of the right element type; `$try_new` is the frame
  /// ctor (8-arg `y,u,v,w,h,ys,us,vs`); `$walker` the direct walker fn.
  macro_rules! planar3_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker:path,
      ramp = $ramp:expr, cw_div = $cw_div:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let cw = (W as usize) / $cw_div;
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let u = make(cw * ch);
        let v = make(cw * ch);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &u, &v, W, H, W, cw as u32, cw as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // 8-bit: 4:2:0 (half/half chroma).
  planar3_case!(
    walk_yuv420p_matches_direct,
    crate::source::Yuv420p,
    crate::frame::Yuv420pFrame::try_new,
    crate::source::yuv420p_to,
    ramp = ramp8,
    cw_div = 2,
    ch_div = 2,
  );
  // High-bit-LE: 4:2:2 10-bit (half-width, full-height chroma).
  planar3_case!(
    walk_yuv422p10_matches_direct,
    crate::source::Yuv422p10,
    crate::frame::Yuv422p10LeFrame::try_new,
    crate::source::yuv422p10_to,
    ramp = |n| ramp16(n, 10),
    cw_div = 2,
    ch_div = 1,
  );
  // High-bit-LE: 4:4:4 16-bit (full chroma; the i64 chroma_sum kernel).
  planar3_case!(
    walk_yuv444p16_matches_direct,
    crate::source::Yuv444p16,
    crate::frame::Yuv444p16LeFrame::try_new,
    crate::source::yuv444p16_to,
    ramp = |n| ramp16(n, 16),
    cw_div = 1,
    ch_div = 1,
  );

  /// BE sibling of [`planar3_case`]: drives the `@const_bits` impl at
  /// `Marker<true>` against a `{Fmt}BeFrame` and compares to a direct
  /// `{fmt}_to_endian::<_, true>` call. `$marker` is the BE-pinned marker
  /// (`Yuv422p10<true>`); `$walker_endian` the const-generic walker.
  macro_rules! planar3_be_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker_endian:path,
      ramp = $ramp:expr, cw_div = $cw_div:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let cw = (W as usize) / $cw_div;
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let u = make(cw * ch);
        let v = make(cw * ch);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &u, &v, W, H, W, cw as u32, cw as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker_endian(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} BE parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // High-bit-BE: 4:2:2 10-bit (half-width, full-height chroma).
  planar3_be_case!(
    walk_yuv422p10_be_matches_direct,
    crate::source::Yuv422p10<true>,
    crate::frame::Yuv422p10BeFrame::try_new,
    crate::source::yuv422p10_to_endian::<_, true>,
    ramp = |n| ramp16(n, 10),
    cw_div = 2,
    ch_div = 1,
  );
}

#[cfg(feature = "yuv-semi-planar")]
mod yuv_semi_planar_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const MATRICES: [ColorMatrix; 2] = [ColorMatrix::Bt709, ColorMatrix::Bt601];

  fn ramp8(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 23 + 5) % 251) as u8).collect()
  }
  fn ramp16(n: usize, bits: u32) -> std::vec::Vec<u16> {
    let max = (1u32 << bits) - 1;
    (0..n)
      .map(|i| (((i as u32) * 1399 + 11) & max) as u16)
      .collect()
  }

  /// Drives a semi-planar (Y + interleaved chroma) family. `$try_new`
  /// is the 6-arg ctor (`y,uv,w,h,ys,uvs`); `chroma_w_factor` is the UV
  /// row length in elements as a multiple of width; `ch_div` the chroma
  /// vertical divisor.
  macro_rules! semi_planar_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker:path,
      ramp = $ramp:expr, chroma_w_factor = $cwf:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let uv_row = ($cwf as usize) * (W as usize);
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let uv = make(uv_row * ch);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &uv, W, H, W, uv_row as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // 8-bit: Nv12 (4:2:0, UV interleaved half-width/half-height).
  semi_planar_case!(
    walk_nv12_matches_direct,
    crate::source::Nv12,
    crate::frame::Nv12Frame::try_new,
    crate::source::nv12_to,
    ramp = ramp8,
    chroma_w_factor = 1,
    ch_div = 2,
  );
  // High-bit-LE: P010 (4:2:0 10-bit packed u16, MSB-justified).
  semi_planar_case!(
    walk_p010_matches_direct,
    crate::source::P010,
    crate::frame::P010LeFrame::try_new,
    crate::source::p010_to,
    ramp = |n| ramp16(n, 16),
    chroma_w_factor = 1,
    ch_div = 2,
  );

  /// BE sibling of [`semi_planar_case`]: drives the `@const_bits` impl at
  /// `Marker<true>` against a `{Fmt}BeFrame` and compares to a direct
  /// `{fmt}_to_endian::<_, true>` call.
  macro_rules! semi_planar_be_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker_endian:path,
      ramp = $ramp:expr, chroma_w_factor = $cwf:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let uv_row = ($cwf as usize) * (W as usize);
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let uv = make(uv_row * ch);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &uv, W, H, W, uv_row as u32).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker_endian(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} BE parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // High-bit-BE: P010 (4:2:0 10-bit packed u16, MSB-justified).
  semi_planar_be_case!(
    walk_p010_be_matches_direct,
    crate::source::P010<true>,
    crate::frame::P010BeFrame::try_new,
    crate::source::p010_to_endian::<_, true>,
    ramp = |n| ramp16(n, 16),
    chroma_w_factor = 1,
    ch_div = 2,
  );
}

#[cfg(feature = "yuv-packed")]
mod yuv_packed_parity {
  use super::*;
  use crate::{
    frame::Yuyv422Frame,
    sinker::MixedSinker,
    source::{Yuyv422, yuyv422_to},
  };

  const MATRICES: [ColorMatrix; 2] = [ColorMatrix::Bt709, ColorMatrix::Bt601];

  /// Packed YUYV 4:2:2 — single buffer, `width * 2` u8 per row.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_yuyv422_matches_direct() {
    const W: u32 = 16;
    const H: u32 = 4;
    let buf: std::vec::Vec<u8> = (0..(W * H * 2) as usize)
      .map(|i| ((i * 19 + 7) % 251) as u8)
      .collect();

    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Yuyv422Frame::try_new(&buf, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Yuyv422>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Yuyv422 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Yuyv422>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        yuyv422_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "yuyv422 parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

#[cfg(feature = "y2xx")]
mod y2xx_parity {
  use super::*;
  use crate::{
    frame::{Y210BeFrame, Y210LeFrame},
    sinker::MixedSinker,
    source::{Y210, y210_to, y210_to_endian},
  };

  const MATRICES: [ColorMatrix; 2] = [ColorMatrix::Bt709, ColorMatrix::Bt601];

  /// MSB-justified 10-bit Y210 wire samples (low 6 bits zero).
  fn y210_ramp() -> std::vec::Vec<u16> {
    const W: u32 = 16;
    const H: u32 = 4;
    (0..(W * H * 2) as usize)
      .map(|i| (((i as u32 * 1103 + 7) & 0x03FF) << 6) as u16)
      .collect()
  }

  /// Y210 LE — packed 4:2:2 10-bit, single `u16` buffer, `width * 2` per
  /// row. Drives the `@const_bits` impl at the marker's `<const BE =
  /// false>` default against the LE `y210_to` wrapper.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_y210_matches_direct() {
    const W: u32 = 16;
    const H: u32 = 4;
    let buf = y210_ramp();

    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Y210LeFrame::try_new(&buf, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Y210>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Y210 as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Y210>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        y210_to(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "y210 parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }

  /// Y210 BE — drives the `@const_bits` impl at `Y210<true>` against the
  /// `Y210BeFrame` alias, compared to a direct `y210_to_endian::<_, true>`.
  #[test]
  #[cfg_attr(
    miri,
    ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
  )]
  fn walk_y210_be_matches_direct() {
    const W: u32 = 16;
    const H: u32 = 4;
    let buf = y210_ramp();

    for full_range in [false, true] {
      for matrix in MATRICES {
        let opts = YuvOptions::new()
          .maybe_full_range(full_range)
          .with_matrix(matrix);
        let src = Y210BeFrame::try_new(&buf, W, H, W * 2).unwrap();

        let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
        let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

        let mut sw = MixedSinker::<Y210<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_walker)
          .unwrap();
        <Y210<true> as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

        let mut sd = MixedSinker::<Y210<true>>::new(W as usize, H as usize)
          .with_rgb(&mut via_direct)
          .unwrap();
        y210_to_endian::<_, true>(&src, full_range, matrix, &mut sd).unwrap();

        assert_eq!(
          via_walker, via_direct,
          "y210 BE parity (full_range={full_range}, matrix={matrix:?})"
        );
      }
    }
  }
}

#[cfg(feature = "yuva")]
mod yuva_parity {
  use super::*;
  use crate::sinker::MixedSinker;

  const MATRICES: [ColorMatrix; 2] = [ColorMatrix::Bt709, ColorMatrix::Bt601];

  fn ramp8(n: usize) -> std::vec::Vec<u8> {
    (0..n).map(|i| ((i * 31 + 9) % 251) as u8).collect()
  }
  fn ramp16(n: usize, bits: u32) -> std::vec::Vec<u16> {
    let max = (1u32 << bits) - 1;
    (0..n)
      .map(|i| (((i as u32) * 911 + 13) & max) as u16)
      .collect()
  }

  /// Drives a 4-plane planar YUVA family. The alpha plane is read inside
  /// `{fmt}_to`, never an `Options` knob — so a [`YuvOptions`] is all the
  /// walk needs, and byte-identity to the direct call proves the alpha
  /// path is forwarded too. `$try_new` is the 10-arg ctor
  /// (`y,u,v,a,w,h,ys,us,vs,as`).
  macro_rules! planar4_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker:path,
      ramp = $ramp:expr, cw_div = $cw_div:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let cw = (W as usize) / $cw_div;
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let u = make(cw * ch);
        let v = make(cw * ch);
        let a = make((W * H) as usize);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &u, &v, &a, W, H, W, cw as u32, cw as u32, W).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // 8-bit: Yuva420p (half/half chroma + full-res alpha).
  planar4_case!(
    walk_yuva420p_matches_direct,
    crate::source::Yuva420p,
    crate::frame::Yuva420pFrame::try_new,
    crate::source::yuva420p_to,
    ramp = ramp8,
    cw_div = 2,
    ch_div = 2,
  );
  // High-bit-LE: Yuva444p12 (full chroma + full-res alpha).
  planar4_case!(
    walk_yuva444p12_matches_direct,
    crate::source::Yuva444p12,
    crate::frame::Yuva444p12LeFrame::try_new,
    crate::source::yuva444p12_to,
    ramp = |n| ramp16(n, 12),
    cw_div = 1,
    ch_div = 1,
  );

  /// BE sibling of [`planar4_case`]: drives the `@const_bits` impl at
  /// `Marker<true>` against a `{Fmt}BeFrame`, compared to a direct
  /// `{fmt}_to_endian::<_, true>` call (alpha read inside the walker).
  macro_rules! planar4_be_case {
    (
      $name:ident, $marker:ty, $try_new:path, $walker_endian:path,
      ramp = $ramp:expr, cw_div = $cw_div:expr, ch_div = $ch_div:expr,
    ) => {
      #[test]
      #[cfg_attr(
        miri,
        ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
      )]
      fn $name() {
        const W: u32 = 16;
        const H: u32 = 4;
        let cw = (W as usize) / $cw_div;
        let ch = (H as usize).div_ceil($ch_div);
        let make = $ramp;
        let y = make((W * H) as usize);
        let u = make(cw * ch);
        let v = make(cw * ch);
        let a = make((W * H) as usize);

        for full_range in [false, true] {
          for matrix in MATRICES {
            let opts = YuvOptions::new().maybe_full_range(full_range).with_matrix(matrix);
            let src = $try_new(&y, &u, &v, &a, W, H, W, cw as u32, cw as u32, W).unwrap();

            let mut via_walker = std::vec![0u8; (W * H * 3) as usize];
            let mut via_direct = std::vec![0u8; (W * H * 3) as usize];

            let mut sw = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_walker)
              .unwrap();
            <$marker as Walker<_>>::walk(&src, &opts, &mut sw).unwrap();

            let mut sd = MixedSinker::<$marker>::new(W as usize, H as usize)
              .with_rgb(&mut via_direct)
              .unwrap();
            $walker_endian(&src, full_range, matrix, &mut sd).unwrap();

            assert_eq!(
              via_walker, via_direct,
              "{} BE parity (full_range={full_range}, matrix={matrix:?})",
              stringify!($name)
            );
          }
        }
      }
    };
  }

  // High-bit-BE: Yuva444p12 (full chroma + full-res alpha).
  planar4_be_case!(
    walk_yuva444p12_be_matches_direct,
    crate::source::Yuva444p12<true>,
    crate::frame::Yuva444p12BeFrame::try_new,
    crate::source::yuva444p12_to_endian::<_, true>,
    ramp = |n| ramp16(n, 12),
    cw_div = 1,
    ch_div = 1,
  );
}
