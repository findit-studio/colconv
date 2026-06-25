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
