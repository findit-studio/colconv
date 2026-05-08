use super::*;
use crate::DcpTargetGamut;

// ---- Tier 12 — Xyz12 packed-CIE-XYZ-12-bit source family --------------

/// Tolerance for f32 comparisons. The OETF + matmul + f32 narrow chain
/// adds about 4e-7 of noise; 4e-6 is comfortably above any platform
/// drift. Matches the spec's `EPSILON_F32 = 4e-6` constant.
const EPSILON_F32: f32 = 4e-6;

/// Builds a row-padded Xyz12LE frame with a constant `(X, Y, Z)` triple.
fn solid_xyz12_frame_le(width: u32, height: u32, x: u16, y: u16, z: u16) -> Vec<u16> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u16; w * h * 3];
  for px in buf.chunks_mut(3) {
    px[0] = x;
    px[1] = y;
    px[2] = z;
  }
  buf
}

/// Builds the BE-byte-swapped variant of `solid_xyz12_frame_le`.
fn solid_xyz12_frame_be(width: u32, height: u32, x: u16, y: u16, z: u16) -> Vec<u16> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0u16; w * h * 3];
  for px in buf.chunks_mut(3) {
    px[0] = x.swap_bytes();
    px[1] = y.swap_bytes();
    px[2] = z.swap_bytes();
  }
  buf
}

#[test]
fn xyz12_with_rgb_zero_input_zero_output() {
  let pix = solid_xyz12_frame_le(8, 4, 0, 0, 0);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 8 * 4 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [0, 0, 0]);
  }
}

#[test]
fn xyz12_with_rgb_max_input_clamps_to_255() {
  // (4095, 4095, 4095) under DCI-P3 → linear ~(1.265, 1.044, 1.0) →
  // after clamp [0,1] + OETF + ×255 → all channels saturate to 255.
  let pix = solid_xyz12_frame_le(8, 4, 0xFFF, 0xFFF, 0xFFF);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 8 * 4 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 255]);
  }
}

#[test]
fn xyz12_with_rgba_fills_alpha_max_u8() {
  let pix = solid_xyz12_frame_le(8, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgba_out = std::vec![0u8; 8 * 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgba(&mut rgba_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFF, "alpha");
  }
}

#[test]
fn xyz12_with_rgb_u16_full_range_at_max() {
  let pix = solid_xyz12_frame_le(8, 4, 0xFFF, 0xFFF, 0xFFF);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgb_out = std::vec![0u16; 8 * 4 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgb_u16(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [65535, 65535, 65535]);
  }
}

#[test]
fn xyz12_with_rgba_u16_fills_alpha_max() {
  let pix = solid_xyz12_frame_le(8, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgba_out = std::vec![0u16; 8 * 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgba_u16(&mut rgba_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::Rec709, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px[3], 0xFFFF);
  }
}

#[test]
fn xyz12_with_rgb_f32_preserves_negatives() {
  // y_only_max under Rec.709 → R = -1.677, G = +2.05, B = -0.222.
  // f32 path skips clamp, so negatives + > 1 values are preserved.
  let pix = solid_xyz12_frame_le(4, 2, 0, 0xFFF, 0);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgb_out = std::vec![0.0f32; 4 * 2 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_rgb_f32(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::Rec709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert!(px[0] < 0.0, "R should be negative, got {}", px[0]);
    assert!(px[1] > 1.0, "G should be > 1, got {}", px[1]);
    assert!(px[2] < 0.0, "B should be negative, got {}", px[2]);
  }
}

#[test]
fn xyz12_with_xyz_f32_lossless_passthrough() {
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut xyz_out = std::vec![0.0f32; 4 * 2 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_xyz_f32(&mut xyz_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();

  // Linear XYZ pass-through: only step-1 inverse OETF applied. All
  // three channels should have identical values for an input of
  // (0x800, 0x800, 0x800).
  for px in xyz_out.chunks(3) {
    assert!((px[0] - px[1]).abs() < EPSILON_F32);
    assert!((px[1] - px[2]).abs() < EPSILON_F32);
    // Should be > 0 and bounded by 1 / 0.91653 ≈ 1.091.
    assert!(px[0] > 0.0);
    assert!(px[0] < 1.1);
  }
}

#[test]
fn xyz12_with_rgb_f16_rec709_mid_gray() {
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgb_out = std::vec![half::f16::ZERO; 4 * 2 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_rgb_f16(&mut rgb_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::Rec709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    // Each channel should be a positive value in (0, 1) since OETF +
    // clamp [0, 1] applies.
    for c in px {
      let v = c.to_f32();
      assert!((0.0..=1.0).contains(&v), "f16 channel out of [0,1]: {v}");
    }
  }
}

#[test]
fn xyz12_with_rgba_f16_alpha_one() {
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgba_out = std::vec![half::f16::ZERO; 4 * 2 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_rgba_f16(&mut rgba_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();

  for px in rgba_out.chunks(4) {
    assert_eq!(px[3].to_f32(), 1.0);
  }
}

#[test]
fn xyz12_with_luma_via_staging() {
  let pix = solid_xyz12_frame_le(4, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 4, 4 * 3).unwrap();

  let mut luma_out = std::vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 4)
    .with_luma(&mut luma_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::Rec709, &mut sink).unwrap();

  // Mid-gray under Rec.709 → grayish output (around 113-128 u8).
  for &y in &luma_out {
    assert!(y > 0, "expected non-zero luma");
    assert!(y < 255);
  }
}

#[test]
fn xyz12_with_luma_u16_zero_extends() {
  let pix = solid_xyz12_frame_le(4, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 4, 4 * 3).unwrap();

  let mut luma_out = std::vec![0u16; 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 4)
    .with_luma_u16(&mut luma_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::Rec709, &mut sink).unwrap();

  for &y in &luma_out {
    // Same value as `with_luma` (u8) but zero-extended into u16.
    assert!(y > 0);
  }
}

#[test]
fn xyz12_with_hsv_via_staging() {
  let pix = solid_xyz12_frame_le(4, 4, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 4, 4 * 3).unwrap();

  let mut h_out = std::vec![0u8; 4 * 4];
  let mut s_out = std::vec![0u8; 4 * 4];
  let mut v_out = std::vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 4)
    .with_hsv(&mut h_out, &mut s_out, &mut v_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();

  // Mid-gray → low saturation. Hue undefined for grayscale but should
  // not panic.
  for &v in &v_out {
    assert!(v > 0);
  }
}

#[test]
fn xyz12_target_gamut_changes_output() {
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut out_p3 = std::vec![0u8; 4 * 2 * 3];
  let mut out_709 = std::vec![0u8; 4 * 2 * 3];
  let mut out_2020 = std::vec![0u8; 4 * 2 * 3];
  {
    let mut sink_p3 = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_p3)
      .unwrap();
    xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink_p3).unwrap();
  }
  {
    let mut sink_709 = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_709)
      .unwrap();
    xyz12_to(&src, DcpTargetGamut::Rec709, &mut sink_709).unwrap();
  }
  {
    let mut sink_2020 = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_2020)
      .unwrap();
    xyz12_to(&src, DcpTargetGamut::Rec2020, &mut sink_2020).unwrap();
  }

  // The three gamut matrices differ at the second-decimal level, so
  // the resulting u8 outputs must differ on at least one channel.
  assert_ne!(out_p3, out_709, "DCI-P3 vs Rec.709");
  assert_ne!(out_p3, out_2020, "DCI-P3 vs Rec.2020");
}

#[test]
fn xyz12_be_byte_swap_matches_le() {
  // BE input that is the byte-swapped LE encoding of the same
  // logical sample value should produce identical output.
  let pix_le = solid_xyz12_frame_le(4, 2, 0x800, 0x800, 0x800);
  let pix_be = solid_xyz12_frame_be(4, 2, 0x800, 0x800, 0x800);
  let src_le = Xyz12LeFrame::try_new(&pix_le, 4, 2, 4 * 3).unwrap();
  let src_be = Xyz12BeFrame::try_new(&pix_be, 4, 2, 4 * 3).unwrap();

  let mut out_le = std::vec![0u8; 4 * 2 * 3];
  let mut out_be = std::vec![0u8; 4 * 2 * 3];
  {
    let mut sink_le = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_le)
      .unwrap();
    xyz12_to(&src_le, DcpTargetGamut::DciP3, &mut sink_le).unwrap();
  }
  {
    let mut sink_be = MixedSinker::<Xyz12Be>::new(4, 2)
      .with_rgb(&mut out_be)
      .unwrap();
    xyz12_to(&src_be, DcpTargetGamut::DciP3, &mut sink_be).unwrap();
  }
  assert_eq!(out_le, out_be);
}

#[test]
fn xyz12_combined_rgb_rgba_consistent() {
  // Combined `with_rgb + with_rgba` must produce the same RGB on both
  // outputs (alpha = 0xFF on the rgba path).
  let pix = solid_xyz12_frame_le(4, 2, 0x800, 0x900, 0x700);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 4 * 2 * 3];
  let mut rgba_out = std::vec![0u8; 4 * 2 * 4];
  let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
    .with_rgb(&mut rgb_out)
    .unwrap()
    .with_rgba(&mut rgba_out)
    .unwrap();
  xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink).unwrap();

  for x in 0..(4 * 2) {
    let rgb_idx = x * 3;
    let rgba_idx = x * 4;
    assert_eq!(rgb_out[rgb_idx], rgba_out[rgba_idx]);
    assert_eq!(rgb_out[rgb_idx + 1], rgba_out[rgba_idx + 1]);
    assert_eq!(rgb_out[rgb_idx + 2], rgba_out[rgba_idx + 2]);
    assert_eq!(rgba_out[rgba_idx + 3], 0xFF);
  }
}

#[test]
fn xyz12_dirty_high_bits_masked_in_kernel() {
  // Input with bit 13 set — kernel masks to 12 bits, output should
  // match clean 12-bit input.
  let clean = solid_xyz12_frame_le(4, 2, 0x0800, 0x0800, 0x0800);
  let dirty = solid_xyz12_frame_le(4, 2, 0xF800, 0xA800, 0x2800);
  let src_clean = Xyz12LeFrame::try_new(&clean, 4, 2, 4 * 3).unwrap();
  let src_dirty = Xyz12LeFrame::try_new(&dirty, 4, 2, 4 * 3).unwrap();

  let mut out_clean = std::vec![0u8; 4 * 2 * 3];
  let mut out_dirty = std::vec![0u8; 4 * 2 * 3];
  {
    let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_clean)
      .unwrap();
    xyz12_to(&src_clean, DcpTargetGamut::DciP3, &mut sink).unwrap();
  }
  {
    let mut sink = MixedSinker::<Xyz12Le>::new(4, 2)
      .with_rgb(&mut out_dirty)
      .unwrap();
    xyz12_to(&src_dirty, DcpTargetGamut::DciP3, &mut sink).unwrap();
  }
  assert_eq!(out_clean, out_dirty);
}

#[test]
fn xyz12_short_buffer_rejected_at_attach() {
  let pix = solid_xyz12_frame_le(8, 4, 0, 0, 0);
  let _src = Xyz12LeFrame::try_new(&pix, 8, 4, 8 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 8 * 4 * 3 - 1]; // one byte short
  let res = MixedSinker::<Xyz12Le>::new(8, 4).with_rgb(&mut rgb_out);
  assert!(matches!(
    res,
    Err(MixedSinkerError::RgbBufferTooShort { .. })
  ));
}

#[test]
fn xyz12_short_xyz_f32_buffer_rejected_at_attach() {
  let mut xyz_out = std::vec![0.0f32; 8 * 4 * 3 - 1];
  let res = MixedSinker::<Xyz12Le>::new(8, 4).with_xyz_f32(&mut xyz_out);
  assert!(matches!(
    res,
    Err(MixedSinkerError::XyzF32BufferTooShort { .. })
  ));
}

#[test]
fn xyz12_dimension_mismatch_rejected_at_begin_frame() {
  let pix = solid_xyz12_frame_le(4, 2, 0, 0, 0);
  let src = Xyz12LeFrame::try_new(&pix, 4, 2, 4 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 8 * 4 * 3];
  let mut sink = MixedSinker::<Xyz12Le>::new(8, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  let res = xyz12_to(&src, DcpTargetGamut::DciP3, &mut sink);
  assert!(matches!(
    res,
    Err(MixedSinkerError::DimensionMismatch { .. })
  ));
}
