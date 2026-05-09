//! `MixedSinker<Yuv411p>` integration tests — covers the full output
//! set (rgb / rgba / luma / luma_u16 / hsv) plus Strategy A
//! (RGB + RGBA both attached → run RGB kernel once, fan out via
//! `expand_rgb_to_rgba_row`).
//!
//! 4:1:1 is DV-NTSC legacy with quarter-width chroma. Frame width
//! must be a multiple of 4; the sinker rejects other widths through
//! `MixedSinkerError::OddWidth` (variant reused — see the impl-side
//! comment in `planar_8bit.rs`).

use super::*;

/// Build a solid 4:1:1 frame with the given Y / U / V byte values.
/// Uses contiguous strides (`y_stride = w`, `u_stride = v_stride = w / 4`).
fn solid_yuv411p_frame(
  width: u32,
  height: u32,
  y: u8,
  u: u8,
  v: u8,
) -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let w = width as usize;
  let h = height as usize;
  let cw = w / 4;
  (
    std::vec![y; w * h],
    std::vec![u; cw * h],
    std::vec![v; cw * h],
  )
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_luma_only_copies_y_plane() {
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 42, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut luma = std::vec![0u8; 16 * 8];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_luma(&mut luma)
    .unwrap();
  yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(luma.iter().all(|&y| y == 42), "luma should be solid 42");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_luma_u16_zero_extends_y_plane() {
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 200, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut luma_u16 = std::vec![0u16; 16 * 8];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  // 8-bit Y zero-extends into u16.
  assert!(luma_u16.iter().all(|&y| y == 200));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_rgb_only_converts_gray_to_gray() {
  // Neutral chroma → gray RGB; solid Y=128 → ~128 in every RGB byte.
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 128, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut rgb = std::vec![0u8; 16 * 8 * 3];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_rgb(&mut rgb)
    .unwrap();
  yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgb.chunks(3) {
    assert!(px[0].abs_diff(128) <= 1);
    assert_eq!(px[0], px[1]);
    assert_eq!(px[1], px[2]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_rgba_only_with_opaque_alpha() {
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 200, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut rgba = std::vec![0u8; 16 * 8 * 4];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_rgba(&mut rgba)
    .unwrap();
  yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  for px in rgba.chunks(4) {
    assert!(px[0].abs_diff(200) <= 1);
    assert_eq!(px[3], 0xFF, "alpha must be opaque");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_strategy_a_rgb_and_rgba_match_byte_for_byte() {
  // Strategy A: when both RGB and RGBA are attached, the RGB kernel
  // runs once and `expand_rgb_to_rgba_row` fans out into RGBA.
  // The first three bytes per pixel must match the dedicated
  // RGB-only output, with alpha = 0xFF.
  let w: u32 = 16;
  let h: u32 = 4;
  let (yp, up, vp) = solid_yuv411p_frame(w, h, 180, 60, 200);
  let src = Yuv411pFrame::new(&yp, &up, &vp, w, h, w, w / 4, w / 4);

  let ws = w as usize;
  let hs = h as usize;

  let mut rgb_only = std::vec![0u8; ws * hs * 3];
  let mut sink_rgb = MixedSinker::<Yuv411p>::new(ws, hs)
    .with_rgb(&mut rgb_only)
    .unwrap();
  yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink_rgb).unwrap();

  let mut rgb_combo = std::vec![0u8; ws * hs * 3];
  let mut rgba_combo = std::vec![0u8; ws * hs * 4];
  let mut sink_combo = MixedSinker::<Yuv411p>::new(ws, hs)
    .with_rgb(&mut rgb_combo)
    .unwrap()
    .with_rgba(&mut rgba_combo)
    .unwrap();
  yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink_combo).unwrap();

  assert_eq!(rgb_only, rgb_combo, "RGB-only and combo RGB must match");
  for px in 0..(ws * hs) {
    assert_eq!(rgba_combo[px * 4], rgb_only[px * 3]);
    assert_eq!(rgba_combo[px * 4 + 1], rgb_only[px * 3 + 1]);
    assert_eq!(rgba_combo[px * 4 + 2], rgb_only[px * 3 + 2]);
    assert_eq!(rgba_combo[px * 4 + 3], 0xFF);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_hsv_only_allocates_scratch_and_produces_gray_hsv() {
  // Neutral gray → H=0, S=0, V=~128. No RGB buffer provided.
  let (yp, up, vp) = solid_yuv411p_frame(16, 8, 128, 128, 128);
  let src = Yuv411pFrame::new(&yp, &up, &vp, 16, 8, 16, 4, 4);

  let mut h = std::vec![0xFFu8; 16 * 8];
  let mut s = std::vec![0xFFu8; 16 * 8];
  let mut v = std::vec![0xFFu8; 16 * 8];
  let mut sink = MixedSinker::<Yuv411p>::new(16, 8)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  yuv411p_to(&src, true, ColorMatrix::Bt601, &mut sink).unwrap();

  assert!(h.iter().all(|&b| b == 0));
  assert!(s.iter().all(|&b| b == 0));
  assert!(v.iter().all(|&b| b.abs_diff(128) <= 1));
}

#[test]
fn yuv411p_rejects_width_not_multiple_of_four() {
  // Width=15 isn't a multiple of 4 → frame construction fails.
  let yp = std::vec![0u8; 15 * 4];
  let up = std::vec![128u8; 4 * 4];
  let vp = std::vec![128u8; 4 * 4];
  let err = Yuv411pFrame::try_new(&yp, &up, &vp, 15, 4, 15, 4, 4).unwrap_err();
  assert!(matches!(
    err,
    Yuv411pFrameError::WidthNotMultipleOfFour { width: 15 }
  ));

  // The sinker also rejects via begin_frame (defense in depth — direct
  // process callers may have skipped frame validation).
  let mut sink: MixedSinker<'_, Yuv411p> = MixedSinker::new(15usize, 4usize);
  let err =
    <MixedSinker<'_, Yuv411p> as crate::PixelSink>::begin_frame(&mut sink, 15, 4).unwrap_err();
  assert!(matches!(err, MixedSinkerError::OddWidth { width: 15 }));
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn yuv411p_simd_matches_scalar_with_random_yuv() {
  // End-to-end SIMD vs scalar parity for the sinker pipeline.
  let w: u32 = 64;
  let h: u32 = 8;
  let ws = w as usize;
  let hs = h as usize;

  let mut yp = std::vec![0u8; ws * hs];
  let mut up = std::vec![0u8; (ws / 4) * hs];
  let mut vp = std::vec![0u8; (ws / 4) * hs];
  pseudo_random_u8(&mut yp, 0x1111);
  pseudo_random_u8(&mut up, 0x2222);
  pseudo_random_u8(&mut vp, 0x3333);
  let src = Yuv411pFrame::new(&yp, &up, &vp, w, h, w, w / 4, w / 4);

  for &matrix in &[ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::YCgCo] {
    for full_range in [true, false] {
      let mut rgb_simd = std::vec![0u8; ws * hs * 3];
      let mut rgb_scalar = std::vec![0u8; ws * hs * 3];

      let mut s_simd = MixedSinker::<Yuv411p>::new(ws, hs)
        .with_rgb(&mut rgb_simd)
        .unwrap();
      yuv411p_to(&src, full_range, matrix, &mut s_simd).unwrap();

      let mut s_scalar = MixedSinker::<Yuv411p>::new(ws, hs)
        .with_rgb(&mut rgb_scalar)
        .unwrap();
      s_scalar.set_simd(false);
      yuv411p_to(&src, full_range, matrix, &mut s_scalar).unwrap();

      assert_eq!(
        rgb_simd, rgb_scalar,
        "SIMD vs scalar diverges at matrix={matrix:?} full_range={full_range}"
      );
    }
  }
}

#[test]
fn yuv411p_luma_u16_buffer_too_short_returns_err() {
  // Regression: validation must measure `buf.len()` in u16 elements,
  // not bytes. A buffer one element short of `width × height` u16s
  // must be rejected — this would have slipped through if the check
  // had compared `buf.len()` (u16 count) against a byte count.
  let mut buf = std::vec![0u16; 16 * 8 - 1];
  let result = MixedSinker::<Yuv411p>::new(16, 8).with_luma_u16(&mut buf);
  assert!(matches!(
    result,
    Err(MixedSinkerError::LumaU16BufferTooShort {
      expected: 128,
      actual: 127,
    })
  ));
}

#[test]
fn yuv411p_luma_u16_buffer_exactly_sized_accepts() {
  // Companion to the negative test above: an exactly-sized buffer
  // (`width × height` u16 elements) must be accepted. Pins down the
  // boundary condition so the negative test can't pass for the wrong
  // reason (e.g. an off-by-one in the opposite direction).
  let mut buf = std::vec![0u16; 16 * 8];
  let result = MixedSinker::<Yuv411p>::new(16, 8).with_luma_u16(&mut buf);
  assert!(result.is_ok());
}
