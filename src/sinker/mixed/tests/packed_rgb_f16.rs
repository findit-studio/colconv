use super::*;

// ---- Tier 9 — Rgbf16 packed-half-float-RGB source family ----------------

/// Builds a tightly-packed Rgbf16 row buffer (`width * height * 3` `f16`
/// elements, no row stride padding) filled with a constant `(R, G, B)` triple.
fn solid_rgbf16_frame(
  width: u32,
  height: u32,
  r: half::f16,
  g: half::f16,
  b: half::f16,
) -> std::vec::Vec<half::f16> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![half::f16::ZERO; w * h * 3];
  for px in buf.chunks_mut(3) {
    px[0] = r;
    px[1] = g;
    px[2] = b;
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_clamps_to_u8() {
  // 1.0 → 255, 2.0 → 255 (HDR clamp), -0.5 → 0 (negative clamp).
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(2.0),
    half::f16::from_f32(-0.5),
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_u16_clamps_to_u16() {
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(0.5),
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.5),
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_u16(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // 0.5 * 65535 ≈ 32767 or 32768 (half-precision rounds 0.5 to exact 0.5,
  // so downstream is the same as Rgbf32); 1.0 → 65535; 1.5 → 65535 (clamp).
  for px in rgb_out.chunks(3) {
    assert!(
      px[0] >= 32767 && px[0] <= 32768,
      "unexpected mid: {}",
      px[0]
    );
    assert_eq!(px[1], 65535);
    assert_eq!(px[2], 65535);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_f16_is_lossless() {
  // Include HDR, negatives, and in-range values to confirm bit-exact
  // pass-through.
  let vals_f32 = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0];
  let n_pixels = 16 * 4;
  let pix: std::vec::Vec<half::f16> = (0..n_pixels * 3)
    .map(|i| half::f16::from_f32(vals_f32[i % vals_f32.len()]))
    .collect();
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![half::f16::ZERO; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_f16(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // Bit-exact equality (no rounding, no clamping in the f16 path).
  assert_eq!(rgb_out, pix);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_f32_widens_losslessly() {
  // Includes HDR (> 1.0), negatives, and exact values.
  let vals_f32 = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0];
  let n_pixels = 16 * 4;
  let pix: std::vec::Vec<half::f16> = (0..n_pixels * 3)
    .map(|i| half::f16::from_f32(vals_f32[i % vals_f32.len()]))
    .collect();
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_f32(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // Each widened f32 must equal the f16 widened via to_f32.
  let expected: std::vec::Vec<f32> = pix.iter().map(|h| h.to_f32()).collect();
  assert_eq!(rgb_out, expected, "rgb_f32 widen is not lossless");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_luma_u8() {
  // Constant white → BT.709 full-range luma 255.
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_luma(&mut luma_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_luma_u16() {
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u16; 16 * 4];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_luma_u16(&mut luma_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // u8 luma 255 → u16 255 (zero-extended).
  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_hsv() {
  // Pure red → H=0, S=255, V=255 in the OpenCV 8-bit HSV encoding.
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::ZERO,
    half::f16::ZERO,
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let n = 16 * 4;
  let mut h_out = std::vec![0u8; n];
  let mut s_out = std::vec![0u8; n];
  let mut v_out = std::vec![0u8; n];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_hsv(&mut h_out, &mut s_out, &mut v_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for i in 0..n {
    assert_eq!(h_out[i], 0);
    assert_eq!(s_out[i], 255);
    assert_eq!(v_out[i], 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both SIMD main loop and scalar tail across
  // every backend block size.
  let w = 1921usize;
  let h = 4usize;
  let n_lanes = w * h * 3;
  let mut pix = std::vec![half::f16::ZERO; n_lanes];
  let mut state: u32 = 0xDEAD_BEEF;
  for (i, v) in pix.iter_mut().enumerate() {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let f = match (state >> 28) & 0b11 {
      0 => ((state >> 8) & 0xFF) as f32 / 255.0,
      1 => (((i as u32 & 0x7F) as f32) + 0.5) / 255.0,
      2 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.25,
      _ => -(((state >> 4) & 0xFF) as f32) / 255.0,
    };
    *v = half::f16::from_f32(f);
  }
  let src = Rgbf16Frame::try_new(&pix, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];
  let mut rgba_u16_simd = std::vec![0u16; w * h * 4];
  let mut rgba_u16_scalar = std::vec![0u16; w * h * 4];
  let mut rgb_f16_simd = std::vec![half::f16::ZERO; w * h * 3];
  let mut rgb_f16_scalar = std::vec![half::f16::ZERO; w * h * 3];
  let mut rgb_f32_simd = std::vec![0.0f32; w * h * 3];
  let mut rgb_f32_scalar = std::vec![0.0f32; w * h * 3];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];
  let mut luma_u16_simd = std::vec![0u16; w * h];
  let mut luma_u16_scalar = std::vec![0u16; w * h];

  let mut s_simd = MixedSinker::<Rgbf16>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_simd)
    .unwrap()
    .with_rgb_f16(&mut rgb_f16_simd)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap()
    .with_luma_u16(&mut luma_u16_simd)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Rgbf16>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_scalar)
    .unwrap()
    .with_rgb_f16(&mut rgb_f16_scalar)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap()
    .with_luma_u16(&mut luma_u16_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(rgb_u16_simd, rgb_u16_scalar, "RGB u16 output diverges");
  assert_eq!(rgba_u16_simd, rgba_u16_scalar, "RGBA u16 output diverges");
  assert_eq!(rgb_f16_simd, rgb_f16_scalar, "RGB f16 output diverges");
  assert_eq!(rgb_f32_simd, rgb_f32_scalar, "RGB f32 output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
  assert_eq!(luma_u16_simd, luma_u16_scalar, "Luma u16 output diverges");
  assert_eq!(rgb_f16_simd, pix, "RGB f16 output is not lossless");
}
