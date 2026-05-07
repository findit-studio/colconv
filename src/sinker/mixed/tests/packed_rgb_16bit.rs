use super::*;
use crate::{
  frame::{Bgr48Frame, Bgra64Frame, Rgb48Frame, Rgba64Frame},
  sinker::mixed::MixedSinker,
  yuv::{Bgr48, Bgra64, Rgb48, Rgba64, bgr48_to, bgra64_to, rgb48_to, rgba64_to},
};

// ---- Rgb48 -----------------------------------------------------------------

#[test]
fn rgb48_with_rgb_u16_identity() {
  // Native passthrough: each channel copied verbatim (no shift).
  let src: Vec<u16> = vec![0x1234, 0x5678, 0x9ABC];
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u16; 3];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1)
    .with_rgb_u16(&mut out)
    .unwrap();
  rgb48_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out, vec![0x1234u16, 0x5678, 0x9ABC]);
}

#[test]
fn rgb48_with_rgb_narrows_correctly() {
  // Each 16-bit channel narrowed >> 8: 0xFF00 → 0xFF, 0x8000 → 0x80, 0x0100 → 0x01.
  let src: Vec<u16> = vec![0xFF00, 0x8000, 0x0100];
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u8; 3];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1).with_rgb(&mut out).unwrap();
  rgb48_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out, vec![0xFFu8, 0x80, 0x01]);
}

#[test]
fn rgb48_with_rgba_forces_0xff() {
  let src: Vec<u16> = vec![0xFFFF, 0x8000, 0x0000];
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1).with_rgba(&mut out).unwrap();
  rgb48_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xFF); // alpha forced to opaque
}

#[test]
fn rgb48_with_rgba_u16_forces_0xffff() {
  let src: Vec<u16> = vec![0x1000, 0x2000, 0x3000];
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  rgb48_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xFFFF);
}

#[test]
fn rgb48_with_rgb_u16_and_rgba_u16_both_correct() {
  // Two pixels: verify identity copy for RGB and forced alpha for RGBA.
  let src: Vec<u16> = vec![0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x1357, 0x2468];
  let frame = Rgb48Frame::new(&src, 2, 1, 6);
  let mut rgb_u16 = vec![0u16; 6];
  let mut rgba_u16 = vec![0u16; 8];
  let mut sinker = MixedSinker::<Rgb48>::new(2, 1)
    .with_rgb_u16(&mut rgb_u16)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  rgb48_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  // RGB u16: identity copy of source
  assert_eq!(rgb_u16[..6], src[..6]);
  // RGBA u16: alpha slots forced to 0xFFFF
  assert_eq!(rgba_u16[3], 0xFFFF);
  assert_eq!(rgba_u16[7], 0xFFFF);
}

#[test]
fn rgb48_with_luma_non_zero_for_nonzero_input() {
  // Any non-zero equal R/G/B → non-zero luma.
  let src: Vec<u16> = vec![0x8000, 0x8000, 0x8000];
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut luma = vec![0u8; 1];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1)
    .with_luma(&mut luma)
    .unwrap();
  rgb48_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_ne!(luma[0], 0);
}

#[test]
fn rgb48_with_luma_u16_non_zero_for_nonzero_input() {
  let src: Vec<u16> = vec![0x8000, 0x8000, 0x8000];
  let frame = Rgb48Frame::new(&src, 1, 1, 3);
  let mut luma_u16 = vec![0u16; 1];
  let mut sinker = MixedSinker::<Rgb48>::new(1, 1)
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  rgb48_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_ne!(luma_u16[0], 0);
}

#[test]
fn rgb48_simd_matches_scalar() {
  // width=8: exercises SIMD path (if available) vs forced scalar path.
  let w: usize = 8;
  let src: Vec<u16> = (0..w * 3).map(|i| (i * 0x1001) as u16).collect();
  let frame = Rgb48Frame::new(&src, w as u32, 1, w as u32 * 3);
  let mut rgb_simd = vec![0u8; w * 3];
  let mut rgb_scalar = vec![0u8; w * 3];
  let mut s1 = MixedSinker::<Rgb48>::new(w, 1)
    .with_rgb(&mut rgb_simd)
    .unwrap();
  rgb48_to(&frame, true, ColorMatrix::Bt709, &mut s1).unwrap();
  let mut s2 = MixedSinker::<Rgb48>::new(w, 1)
    .with_rgb(&mut rgb_scalar)
    .unwrap();
  s2.set_simd(false);
  rgb48_to(&frame, true, ColorMatrix::Bt709, &mut s2).unwrap();
  assert_eq!(rgb_simd, rgb_scalar);
}

// ---- Bgr48 -----------------------------------------------------------------

#[test]
fn bgr48_channel_order_swapped_vs_rgb48() {
  // B=0x1000, G=0x2000, R=0x3000 stored in BGR input order.
  // After sinker's B↔R swap on output: R=0x30, G=0x20, B=0x10 in u8 RGB.
  let src: Vec<u16> = vec![0x1000, 0x2000, 0x3000];
  let rgb_frame = Rgb48Frame::new(&src, 1, 1, 3);
  let bgr_frame = Bgr48Frame::new(&src, 1, 1, 3);
  let mut rgb_from_rgb48 = vec![0u8; 3];
  let mut rgb_from_bgr48 = vec![0u8; 3];
  let mut s1 = MixedSinker::<Rgb48>::new(1, 1)
    .with_rgb(&mut rgb_from_rgb48)
    .unwrap();
  rgb48_to(&rgb_frame, false, ColorMatrix::Bt709, &mut s1).unwrap();
  let mut s2 = MixedSinker::<Bgr48>::new(1, 1)
    .with_rgb(&mut rgb_from_bgr48)
    .unwrap();
  bgr48_to(&bgr_frame, false, ColorMatrix::Bt709, &mut s2).unwrap();
  // Bgr48 with_rgb: R=0x3000>>8=0x30, G=0x2000>>8=0x20, B=0x1000>>8=0x10.
  assert_eq!(rgb_from_bgr48[0], 0x30); // R
  assert_eq!(rgb_from_bgr48[1], 0x20); // G
  assert_eq!(rgb_from_bgr48[2], 0x10); // B
  // Rgb48 on same bytes treats first element as R, which differs.
  assert_ne!(rgb_from_rgb48[0], rgb_from_bgr48[0]);
}

#[test]
fn bgr48_with_rgba_forces_0xff() {
  let src: Vec<u16> = vec![0xAAAA, 0xBBBB, 0xCCCC];
  let frame = Bgr48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Bgr48>::new(1, 1).with_rgba(&mut out).unwrap();
  bgr48_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xFF);
}

#[test]
fn bgr48_with_rgba_u16_forces_0xffff() {
  let src: Vec<u16> = vec![0x1234, 0x5678, 0x9ABC];
  let frame = Bgr48Frame::new(&src, 1, 1, 3);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Bgr48>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  bgr48_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xFFFF);
}

// ---- Rgba64 ----------------------------------------------------------------

#[test]
fn rgba64_with_rgba_passes_source_alpha_u8() {
  // R=0xFFFF, G=0x8000, B=0x0000, A=0xABFF → alpha byte = 0xABFF >> 8 = 0xAB.
  let src: Vec<u16> = vec![0xFFFF, 0x8000, 0x0000, 0xABFF];
  let frame = Rgba64Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Rgba64>::new(1, 1)
    .with_rgba(&mut out)
    .unwrap();
  rgba64_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xAB); // 0xABFF >> 8 = 0xAB
}

#[test]
fn rgba64_with_rgba_u16_passes_source_alpha_native() {
  // Source α = 0xABCD must be preserved verbatim (no shift).
  let src: Vec<u16> = vec![0xFFFF, 0x8000, 0x0000, 0xABCD];
  let frame = Rgba64Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Rgba64>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  rgba64_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xABCD);
}

#[test]
fn rgba64_strategy_a_plus_rgb_and_rgba_byte_identical() {
  // Strategy A+: running with_rgb + with_rgba together must produce the
  // same RGBA bytes as running with_rgba alone (chroma kernel once).
  let w: usize = 4;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1111) as u16).collect();
  let frame = Rgba64Frame::new(&src, w as u32, 1, w as u32 * 4);

  let mut rgb_combo = vec![0u8; w * 3];
  let mut rgba_combo = vec![0u8; w * 4];
  let mut sinker = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgb(&mut rgb_combo)
    .unwrap()
    .with_rgba(&mut rgba_combo)
    .unwrap();
  rgba64_to(&frame, false, ColorMatrix::Bt709, &mut sinker).unwrap();

  let mut rgba_only = vec![0u8; w * 4];
  let mut sinker2 = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgba(&mut rgba_only)
    .unwrap();
  rgba64_to(&frame, false, ColorMatrix::Bt709, &mut sinker2).unwrap();

  assert_eq!(rgba_combo, rgba_only);
}

#[test]
fn rgba64_strategy_a_plus_u16_path_byte_identical() {
  // Strategy A+ on u16 path: with_rgb_u16 + with_rgba_u16 combined must
  // match with_rgba_u16 alone (single deinterleave pass + alpha scatter).
  let w: usize = 4;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1000) as u16).collect();
  let frame = Rgba64Frame::new(&src, w as u32, 1, w as u32 * 4);

  let mut rgb_u16_combo = vec![0u16; w * 3];
  let mut rgba_u16_combo = vec![0u16; w * 4];
  let mut sinker = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgb_u16(&mut rgb_u16_combo)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_combo)
    .unwrap();
  rgba64_to(&frame, false, ColorMatrix::Bt709, &mut sinker).unwrap();

  let mut rgba_u16_only = vec![0u16; w * 4];
  let mut sinker2 = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgba_u16(&mut rgba_u16_only)
    .unwrap();
  rgba64_to(&frame, false, ColorMatrix::Bt709, &mut sinker2).unwrap();

  assert_eq!(rgba_u16_combo, rgba_u16_only);
}

#[test]
fn rgba64_simd_matches_scalar() {
  // SIMD path (if available) must match the forced scalar path byte-for-byte.
  let w: usize = 8;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1001) as u16).collect();
  let frame = Rgba64Frame::new(&src, w as u32, 1, w as u32 * 4);
  let mut rgba_simd = vec![0u8; w * 4];
  let mut rgba_scalar = vec![0u8; w * 4];
  let mut s1 = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgba(&mut rgba_simd)
    .unwrap();
  rgba64_to(&frame, true, ColorMatrix::Bt709, &mut s1).unwrap();
  let mut s2 = MixedSinker::<Rgba64>::new(w, 1)
    .with_rgba(&mut rgba_scalar)
    .unwrap();
  s2.set_simd(false);
  rgba64_to(&frame, true, ColorMatrix::Bt709, &mut s2).unwrap();
  assert_eq!(rgba_simd, rgba_scalar);
}

#[test]
fn rgba64_with_rgb_u16_drops_alpha() {
  // with_rgb_u16 must output exactly 3 elements per pixel (alpha slot dropped).
  let src: Vec<u16> = vec![0x1111, 0x2222, 0x3333, 0xAAAA];
  let frame = Rgba64Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u16; 3];
  let mut sinker = MixedSinker::<Rgba64>::new(1, 1)
    .with_rgb_u16(&mut out)
    .unwrap();
  rgba64_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[0], 0x1111);
  assert_eq!(out[1], 0x2222);
  assert_eq!(out[2], 0x3333);
}

// ---- Bgra64 ----------------------------------------------------------------

#[test]
fn bgra64_channel_order_and_alpha_preserved() {
  // B=0x1000, G=0x2000, R=0x3000, A=0xAAAA in BGRA source order.
  // After B↔R swap on output: R=0x30, G=0x20, B=0x10. Alpha 0xAAAA>>8=0xAA.
  let src: Vec<u16> = vec![0x1000, 0x2000, 0x3000, 0xAAAA];
  let frame = Bgra64Frame::new(&src, 1, 1, 4);
  let mut rgb_out = vec![0u8; 3];
  let mut rgba_out = vec![0u8; 4];
  let mut sinker = MixedSinker::<Bgra64>::new(1, 1)
    .with_rgb(&mut rgb_out)
    .unwrap()
    .with_rgba(&mut rgba_out)
    .unwrap();
  bgra64_to(&frame, false, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(rgb_out[0], 0x30); // R
  assert_eq!(rgb_out[1], 0x20); // G
  assert_eq!(rgb_out[2], 0x10); // B
  assert_eq!(rgba_out[3], 0xAA); // A = 0xAAAA >> 8
}

#[test]
fn bgra64_strategy_a_plus_rgb_and_rgba_byte_identical() {
  // Strategy A+ must produce same RGBA as standalone with_rgba path.
  let w: usize = 4;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1111) as u16).collect();
  let frame = Bgra64Frame::new(&src, w as u32, 1, w as u32 * 4);

  let mut rgb_combo = vec![0u8; w * 3];
  let mut rgba_combo = vec![0u8; w * 4];
  let mut sinker = MixedSinker::<Bgra64>::new(w, 1)
    .with_rgb(&mut rgb_combo)
    .unwrap()
    .with_rgba(&mut rgba_combo)
    .unwrap();
  bgra64_to(&frame, false, ColorMatrix::Bt709, &mut sinker).unwrap();

  let mut rgba_only = vec![0u8; w * 4];
  let mut sinker2 = MixedSinker::<Bgra64>::new(w, 1)
    .with_rgba(&mut rgba_only)
    .unwrap();
  bgra64_to(&frame, false, ColorMatrix::Bt709, &mut sinker2).unwrap();

  assert_eq!(rgba_combo, rgba_only);
}

#[test]
fn bgra64_strategy_a_plus_u16_path_byte_identical() {
  let w: usize = 4;
  let src: Vec<u16> = (0..w * 4).map(|i| (i * 0x1000) as u16).collect();
  let frame = Bgra64Frame::new(&src, w as u32, 1, w as u32 * 4);

  let mut rgb_u16_combo = vec![0u16; w * 3];
  let mut rgba_u16_combo = vec![0u16; w * 4];
  let mut sinker = MixedSinker::<Bgra64>::new(w, 1)
    .with_rgb_u16(&mut rgb_u16_combo)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_combo)
    .unwrap();
  bgra64_to(&frame, false, ColorMatrix::Bt709, &mut sinker).unwrap();

  let mut rgba_u16_only = vec![0u16; w * 4];
  let mut sinker2 = MixedSinker::<Bgra64>::new(w, 1)
    .with_rgba_u16(&mut rgba_u16_only)
    .unwrap();
  bgra64_to(&frame, false, ColorMatrix::Bt709, &mut sinker2).unwrap();

  assert_eq!(rgba_u16_combo, rgba_u16_only);
}

#[test]
fn bgra64_with_rgba_u16_passes_source_alpha_native() {
  let src: Vec<u16> = vec![0x1000, 0x2000, 0x3000, 0xBEEF];
  let frame = Bgra64Frame::new(&src, 1, 1, 4);
  let mut out = vec![0u16; 4];
  let mut sinker = MixedSinker::<Bgra64>::new(1, 1)
    .with_rgba_u16(&mut out)
    .unwrap();
  bgra64_to(&frame, true, ColorMatrix::Bt709, &mut sinker).unwrap();
  assert_eq!(out[3], 0xBEEF); // native u16 alpha verbatim
}

// ---- Edge cases / error paths ----------------------------------------------

#[test]
fn rgb48_row_shape_mismatch_returns_error() {
  use crate::{PixelSink, sinker::mixed::MixedSinkerError, yuv::Rgb48Row};
  let mut out = vec![0u8; 6];
  let mut sinker = MixedSinker::<Rgb48>::new(2, 1).with_rgb(&mut out).unwrap();
  sinker.begin_frame(2, 1).unwrap();
  // width=2 expects 6 u16 elements; give 3 — triggers RowShapeMismatch.
  let data = vec![0u16; 3];
  let row = Rgb48Row::new(&data, 0, ColorMatrix::Bt709, true);
  let err = sinker.process(row).unwrap_err();
  assert!(matches!(err, MixedSinkerError::RowShapeMismatch { .. }));
}

#[test]
fn rgba64_row_shape_mismatch_returns_error() {
  use crate::{PixelSink, sinker::mixed::MixedSinkerError, yuv::Rgba64Row};
  let mut out = vec![0u8; 8];
  let mut sinker = MixedSinker::<Rgba64>::new(2, 1)
    .with_rgba(&mut out)
    .unwrap();
  sinker.begin_frame(2, 1).unwrap();
  // width=2 expects 8 u16 elements; give 4 — triggers RowShapeMismatch.
  let data = vec![0u16; 4];
  let row = Rgba64Row::new(&data, 0, ColorMatrix::Bt709, true);
  let err = sinker.process(row).unwrap_err();
  assert!(matches!(err, MixedSinkerError::RowShapeMismatch { .. }));
}

#[test]
fn rgb48_multi_row_frame() {
  // 2×2 frame: verify correct row-by-row accumulation.
  let src: Vec<u16> = vec![
    0xFF00, 0x0000, 0x0000, // row 0, px 0: R=0xFF, G=0x00, B=0x00
    0x0000, 0xFF00, 0x0000, // row 0, px 1: R=0x00, G=0xFF, B=0x00
    0x0000, 0x0000, 0xFF00, // row 1, px 0: R=0x00, G=0x00, B=0xFF
    0xFF00, 0xFF00, 0xFF00, // row 1, px 1: R=0xFF, G=0xFF, B=0xFF
  ];
  let frame = Rgb48Frame::new(&src, 2, 2, 6);
  let mut out = vec![0u8; 12];
  let mut sinker = MixedSinker::<Rgb48>::new(2, 2).with_rgb(&mut out).unwrap();
  rgb48_to(&frame, false, ColorMatrix::Bt709, &mut sinker).unwrap();
  // Row 0, px 0: only R set
  assert_eq!(out[0], 0xFF); // R
  assert_eq!(out[1], 0x00); // G
  assert_eq!(out[2], 0x00); // B
  // Row 1, px 1: all channels max
  assert_eq!(out[9], 0xFF);
  assert_eq!(out[10], 0xFF);
  assert_eq!(out[11], 0xFF);
}
