use crate::{
  ColorMatrix,
  frame::{Gray8Frame, Gray16Frame, Grayf32Frame, Ya8Frame, Ya16Frame},
  sinker::MixedSinker,
  yuv::{gray8_to, gray16_to, grayf32_to, ya8_to, ya16_to},
};
// GrayN<BITS> frame + non-byte-aligned dispatchers are consumed only by
// tests gated on LE hosts (their `Vec<u16>` fixtures travel through
// `from_le` on the sink path and would be byte-swapped on a BE host).
#[cfg(target_endian = "little")]
use crate::{
  frame::GrayNFrame,
  yuv::{gray9_to, gray10_to, gray12_to, gray14_to},
};

// Gray formats are luma-only; full_range and matrix are unused by the kernels
// but are required by the walker signature. Use full_range=true, Bt709.
const FR: bool = true;
const M: ColorMatrix = ColorMatrix::Bt709;

fn make_gray8_frame(data: &[u8], w: u32, h: u32) -> Gray8Frame<'_> {
  Gray8Frame::new(data, w, h, w)
}
// Only used by `gray10_with_*` tests, all of which are gated to LE hosts
// because their `Vec<u16>` plane fixtures travel through `from_le` on the
// sink path and would be byte-swapped on a BE host.
#[cfg(target_endian = "little")]
fn make_gray10_frame(data: &[u16], w: u32, h: u32) -> GrayNFrame<'_, 10> {
  GrayNFrame::new(data, w, h, w)
}
fn make_gray16_frame(data: &[u16], w: u32, h: u32) -> Gray16Frame<'_> {
  Gray16Frame::new(data, w, h, w)
}

#[test]
fn gray8_with_rgb_broadcasts_to_packed() {
  // 4×1 frame: [0, 64, 128, 255]
  let plane = [0u8, 64, 128, 255];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 4 * 3];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  // Each pixel should be [Y, Y, Y]
  assert_eq!(rgb[0..3], [0, 0, 0]);
  assert_eq!(rgb[3..6], [64, 64, 64]);
  assert_eq!(rgb[6..9], [128, 128, 128]);
  assert_eq!(rgb[9..12], [255, 255, 255]);
}

#[test]
fn gray8_with_rgba_alpha_is_0xff() {
  let plane = [100u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgba = std::vec![0u8; 4 * 4];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  // Alpha byte (index 3, 7, 11, 15) should be 0xFF.
  for i in 0..4 {
    assert_eq!(rgba[i * 4 + 3], 0xFF, "pixel {i} alpha");
    assert_eq!(rgba[i * 4], 100, "pixel {i} R");
  }
}

#[test]
fn gray8_with_luma_copies_plane() {
  let plane: Vec<u8> = (0..16u8).collect();
  let frame = make_gray8_frame(&plane, 4, 4);
  let mut luma = std::vec![0u8; 16];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 4)
    .with_luma(&mut luma)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(luma, plane);
}

#[test]
fn gray8_with_luma_u16_zero_extends() {
  let plane = [0u8, 64, 128, 255];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut lu16 = std::vec![0u16; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_luma_u16(&mut lu16)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(lu16, [0, 64, 128, 255]);
}

#[test]
fn gray8_with_hsv_h_s_zero_v_equals_y() {
  let plane = [50u8, 100, 200, 0];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut h = std::vec![0xFFu8; 4];
  let mut s = std::vec![0xFFu8; 4];
  let mut v = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  gray8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0, 0, 0], "H must be 0");
  assert_eq!(s, [0, 0, 0, 0], "S must be 0");
  assert_eq!(v, plane.as_slice(), "V must equal Y");
}

#[cfg(target_endian = "little")]
#[test]
fn gray10_with_rgb_masks_and_shifts() {
  // 10-bit sample: value 512 = 0b10_0000_0000, masked = 512, >> 2 = 128
  let plane = [512u16; 4];
  let frame = make_gray10_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<crate::yuv::Gray10>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray10_to(&frame, FR, M, &mut sink).unwrap();
  // 512 & 0x3FF = 512, >> 2 = 128. All channels should be 128.
  assert_eq!(rgb[0..3], [128, 128, 128]);
  assert_eq!(rgb[3..6], [128, 128, 128]);
}

#[cfg(target_endian = "little")]
#[test]
fn gray10_with_luma_u16_masks_only() {
  // 10-bit, over-range sample: 0x0800 (bit 11 set) masked → 0.
  let plane = [0x0800u16, 0x03FFu16, 0x0200u16, 0x0001u16];
  let frame = make_gray10_frame(&plane, 4, 1);
  let mut lu16 = std::vec![0u16; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray10>::new(4, 1)
    .with_luma_u16(&mut lu16)
    .unwrap();
  gray10_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(lu16, [0x0000, 0x03FF, 0x0200, 0x0001]);
}

#[cfg(target_endian = "little")]
#[test]
fn gray16_with_rgb_shifts_to_u8() {
  // Gray16 sample 0x8000 → >> 8 = 0x80 = 128.
  let plane = [0x8000u16, 0xFFFFu16, 0x0000u16, 0x0100u16];
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray16_to(&frame, FR, M, &mut sink).unwrap();
  // Each pixel [Y>>8, Y>>8, Y>>8]
  assert_eq!(rgb[0..3], [0x80, 0x80, 0x80]);
  assert_eq!(rgb[3..6], [0xFF, 0xFF, 0xFF]);
  assert_eq!(rgb[6..9], [0x00, 0x00, 0x00]);
  assert_eq!(rgb[9..12], [0x01, 0x01, 0x01]);
}

#[cfg(target_endian = "little")]
#[test]
fn gray16_with_luma_u16_copies_plane() {
  let plane: Vec<u16> = (0u16..16).map(|x| x * 4096).collect();
  let frame = make_gray16_frame(&plane, 4, 4);
  let mut lu16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 4)
    .with_luma_u16(&mut lu16)
    .unwrap();
  gray16_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(lu16, plane);
}

#[cfg(target_endian = "little")]
#[test]
fn gray16_with_rgba_u16_alpha_is_0xffff() {
  let plane = [0x1234u16; 4];
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgba_u16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  gray16_to(&frame, FR, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF, "pixel {i} alpha");
    assert_eq!(rgba_u16[i * 4], 0x1234, "pixel {i} R");
  }
}

#[cfg(target_endian = "little")]
#[test]
fn gray9_walker_smoke_test() {
  use crate::frame::GrayNFrame;
  let plane = [100u16; 4];
  let frame: GrayNFrame<'_, 9> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut luma = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray9>::new(4, 1)
    .with_luma(&mut luma)
    .unwrap();
  gray9_to(&frame, FR, M, &mut sink).unwrap();
  // 100 & 0x1FF = 100, >> 1 = 50.
  assert_eq!(luma, [50, 50, 50, 50]);
}

#[cfg(target_endian = "little")]
#[test]
fn gray12_walker_smoke_test() {
  use crate::frame::GrayNFrame;
  let plane = [0x0FFFu16; 4];
  let frame: GrayNFrame<'_, 12> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut luma = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray12>::new(4, 1)
    .with_luma(&mut luma)
    .unwrap();
  gray12_to(&frame, FR, M, &mut sink).unwrap();
  // 0x0FFF & 0x0FFF = 0x0FFF = 4095. >> 4 = 255.
  assert_eq!(luma, [255, 255, 255, 255]);
}

#[cfg(target_endian = "little")]
#[test]
fn gray14_walker_smoke_test() {
  use crate::frame::GrayNFrame;
  let plane = [0x3FFFu16; 4];
  let frame: GrayNFrame<'_, 14> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut luma = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray14>::new(4, 1)
    .with_luma(&mut luma)
    .unwrap();
  gray14_to(&frame, FR, M, &mut sink).unwrap();
  // 0x3FFF & 0x3FFF = 0x3FFF = 16383. >> 6 = 255.
  assert_eq!(luma, [255, 255, 255, 255]);
}

// ---- Limited-range integration tests ----------------------------------------
//
// For 8-bit limited-range: black=16, white=235, range=219.
//   rescale(y) = clamp_u8(((y - 16) * 255 + 109) / 219)
// For N-bit limited-range: black = 16 << (N-8), range = 219 << (N-8).
//   rescale(y) = clamp_u8(((y - black) * 255 + range/2) / range)
// Luma outputs always pass raw Y through (no rescaling regardless of
// full_range).

#[test]
fn gray8_limited_range_black_maps_to_zero() {
  // Y=16 (limited-range black) → RGB(0, 0, 0).
  let plane = [16u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgb = std::vec![0xFFu8; 12];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgb[i * 3..i * 3 + 3], [0, 0, 0], "pixel {i}");
  }
}

#[test]
fn gray8_limited_range_white_maps_to_255() {
  // Y=235 (limited-range white) → RGB(255, 255, 255).
  let plane = [235u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgb[i * 3..i * 3 + 3], [255, 255, 255], "pixel {i}");
  }
}

#[test]
fn gray8_limited_range_midpoint() {
  // Y=125 → ((125-16)*255+109)/219 = 27904/219 = 127.
  let plane = [125u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgb = std::vec![0u8; 12];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgb[i * 3], 127, "pixel {i} R");
  }
}

#[test]
fn gray8_limited_range_luma_passthrough_unchanged() {
  // Luma output must pass raw Y through even for limited-range; no rescaling.
  let plane = [16u8, 235u8, 125u8, 0u8];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut luma = std::vec![0xAAu8; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_luma(&mut luma)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(luma, [16, 235, 125, 0]);
}

#[test]
fn gray8_limited_range_rgba_alpha_is_0xff() {
  // Verify limited-range RGBA: alpha=0xFF, channels rescaled.
  let plane = [235u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut rgba = std::vec![0u8; 16];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgba[i * 4], 255, "pixel {i} R");
    assert_eq!(rgba[i * 4 + 3], 0xFF, "pixel {i} alpha");
  }
}

#[test]
fn gray8_limited_range_hsv_v_is_rescaled() {
  // HSV V channel must use rescaled Y in limited-range mode.
  let plane = [235u8; 4];
  let frame = make_gray8_frame(&plane, 4, 1);
  let mut h = std::vec![0xFFu8; 4];
  let mut s = std::vec![0xFFu8; 4];
  let mut v = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray8>::new(4, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  gray8_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0, 0, 0], "H must be 0");
  assert_eq!(s, [0, 0, 0, 0], "S must be 0");
  assert_eq!(v, [255, 255, 255, 255], "V must be 255 for white");
}

#[cfg(target_endian = "little")]
#[test]
fn gray10_limited_range_black_and_white() {
  use crate::frame::GrayNFrame;
  // 10-bit: black=64, white=940, range=876.
  let plane = [64u16, 940, 64, 940];
  let frame: GrayNFrame<'_, 10> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut rgb = std::vec![0x80u8; 12];
  let mut sink = MixedSinker::<crate::yuv::Gray10>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray10_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0, 0, 0], "Y=64 → black");
  assert_eq!(rgb[3..6], [255, 255, 255], "Y=940 → white");
  assert_eq!(rgb[6..9], [0, 0, 0], "Y=64 → black");
  assert_eq!(rgb[9..12], [255, 255, 255], "Y=940 → white");
}

#[cfg(target_endian = "little")]
#[test]
fn gray12_limited_range_black_and_white() {
  use crate::frame::GrayNFrame;
  // 12-bit: black=256, white=3760, range=3504.
  let plane = [256u16, 3760, 256, 3760];
  let frame: GrayNFrame<'_, 12> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut rgb = std::vec![0x80u8; 12];
  let mut sink = MixedSinker::<crate::yuv::Gray12>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray12_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0, 0, 0], "Y=256 → black");
  assert_eq!(rgb[3..6], [255, 255, 255], "Y=3760 → white");
  assert_eq!(rgb[6..9], [0, 0, 0], "Y=256 → black");
  assert_eq!(rgb[9..12], [255, 255, 255], "Y=3760 → white");
}

#[cfg(target_endian = "little")]
#[test]
fn gray14_limited_range_black_and_white() {
  use crate::frame::GrayNFrame;
  // 14-bit: black=1024, white=15040, range=14016.
  let plane = [1024u16, 15040, 1024, 15040];
  let frame: GrayNFrame<'_, 14> = GrayNFrame::new(&plane, 4, 1, 4);
  let mut rgb = std::vec![0x80u8; 12];
  let mut sink = MixedSinker::<crate::yuv::Gray14>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray14_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0, 0, 0], "Y=1024 → black");
  assert_eq!(rgb[3..6], [255, 255, 255], "Y=15040 → white");
  assert_eq!(rgb[6..9], [0, 0, 0], "Y=1024 → black");
  assert_eq!(rgb[9..12], [255, 255, 255], "Y=15040 → white");
}

#[cfg(target_endian = "little")]
#[test]
fn gray16_limited_range_black_and_white() {
  // 16-bit: black=4096, white=60160, range=56064.
  let plane = [4096u16, 60160, 4096, 60160];
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgb = std::vec![0x80u8; 12];
  let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(rgb[0..3], [0, 0, 0], "Y=4096 → black");
  assert_eq!(rgb[3..6], [255, 255, 255], "Y=60160 → white");
  assert_eq!(rgb[6..9], [0, 0, 0], "Y=4096 → black");
  assert_eq!(rgb[9..12], [255, 255, 255], "Y=60160 → white");
}

#[cfg(target_endian = "little")]
#[test]
fn gray16_limited_range_luma_passthrough_unchanged() {
  // Luma u16 must copy raw Y regardless of full_range.
  let plane = [4096u16, 60160, 32768, 0];
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut lu16 = std::vec![0u16; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
    .with_luma_u16(&mut lu16)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(lu16, [4096, 60160, 32768, 0]);
}

#[test]
fn gray16_limited_range_rgba_u16_alpha_is_0xffff() {
  // RGBA u16 — alpha=0xFFFF; channels hold the native Y broadcast.
  // In limited-range the u16 RGB path passes native Y through (no >>8).
  let plane = [4096u16; 4];
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgba_u16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  for i in 0..4 {
    assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF, "pixel {i} alpha");
  }
}

#[cfg(target_endian = "little")]
#[test]
fn gray16_limited_range_rgba_u16_channels_rescale_at_boundaries() {
  // Regression for the i32-overflow bug at BITS=16: limited-range white
  // 60160 × max_native 65535 ≈ 3.67e9 overflows i32. Math runs in i64;
  // assert that RGB channels reach black=0 and white=65535 at the
  // limited-range boundaries (codex finding requested
  // u16-channel-value asserts, not only alpha).
  let plane = [4096u16, 60160u16, 65535u16, 0u16];
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut rgba_u16 = std::vec![0u16; 16];
  let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  // pixel 0: limited black 4096 → 0
  assert_eq!(&rgba_u16[0..3], &[0, 0, 0]);
  // pixel 1: limited white 60160 → 65535 (over-i32 path)
  assert_eq!(&rgba_u16[4..7], &[65535, 65535, 65535]);
  // pixel 2: over-white 65535 → clamped to 65535
  assert_eq!(&rgba_u16[8..11], &[65535, 65535, 65535]);
  // pixel 3: below-black 0 → clamped to 0
  assert_eq!(&rgba_u16[12..15], &[0, 0, 0]);
  // alpha unchanged
  for i in 0..4 {
    assert_eq!(rgba_u16[i * 4 + 3], 0xFFFF);
  }
}

#[cfg(target_endian = "little")]
#[test]
fn gray16_limited_range_rgb_u16_channels_rescale_at_boundaries() {
  // Same i32-overflow regression on the with_rgb_u16 path.
  let plane = [4096u16, 60160u16];
  let frame = make_gray16_frame(&plane, 2, 1);
  let mut rgb_u16 = std::vec![0u16; 6];
  let mut sink = MixedSinker::<crate::yuv::Gray16>::new(2, 1)
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(&rgb_u16[0..3], &[0, 0, 0]);
  assert_eq!(&rgb_u16[3..6], &[65535, 65535, 65535]);
}

#[cfg(target_endian = "little")]
#[test]
fn gray16_limited_range_hsv_v_is_rescaled() {
  // HSV V must reflect limited-range rescaling.
  let plane = [60160u16; 4]; // white
  let frame = make_gray16_frame(&plane, 4, 1);
  let mut h = std::vec![0xFFu8; 4];
  let mut s = std::vec![0xFFu8; 4];
  let mut v = std::vec![0u8; 4];
  let mut sink = MixedSinker::<crate::yuv::Gray16>::new(4, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  gray16_to(&frame, false, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0, 0, 0], "H must be 0");
  assert_eq!(s, [0, 0, 0, 0], "S must be 0");
  assert_eq!(v, [255, 255, 255, 255], "V must be 255 for white");
}

// ---- Grayf32 integration tests ----------------------------------------------

#[cfg(target_endian = "little")]
#[test]
fn grayf32_with_luma_f32_passthrough() {
  // NaN, out-of-range, and normal values all pass through unchanged.
  let data: std::vec::Vec<f32> = std::vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5, f32::NAN];
  let frame = Grayf32Frame::new(&data, 8, 1, 8);
  let mut out = std::vec![0.0f32; 8];
  let mut sink = MixedSinker::<crate::yuv::Grayf32>::new(8, 1)
    .with_luma_f32(&mut out)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  for (i, (&a, &b)) in data.iter().zip(out.iter()).enumerate() {
    if a.is_nan() {
      assert!(b.is_nan(), "pixel {i}: expected NaN");
    } else {
      assert_eq!(a, b, "pixel {i}");
    }
  }
}

#[cfg(target_endian = "little")]
#[test]
fn grayf32_with_rgb_f32_replicates_losslessly() {
  let data: std::vec::Vec<f32> = std::vec![0.25, 0.75, 1.5, -0.5];
  let frame = Grayf32Frame::new(&data, 4, 1, 4);
  let mut out = std::vec![0.0f32; 4 * 3];
  let mut sink = MixedSinker::<crate::yuv::Grayf32>::new(4, 1)
    .with_rgb_f32(&mut out)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  for (x, &y) in data.iter().enumerate() {
    assert_eq!(out[x * 3], y, "pixel {x} R");
    assert_eq!(out[x * 3 + 1], y, "pixel {x} G");
    assert_eq!(out[x * 3 + 2], y, "pixel {x} B");
  }
}

#[cfg(target_endian = "little")]
#[test]
fn grayf32_with_rgb_saturates() {
  // -0.5 → 0, 0.5 → 128, 1.0 → 255, 1.5 → 255
  let data: std::vec::Vec<f32> = std::vec![-0.5, 0.0, 0.5, 1.0, 1.5];
  let frame = Grayf32Frame::new(&data, 5, 1, 5);
  let mut rgb = std::vec![0u8; 5 * 3];
  let mut sink = MixedSinker::<crate::yuv::Grayf32>::new(5, 1)
    .with_rgb(&mut rgb)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(&rgb[0..3], &[0, 0, 0]); // -0.5 clamps to 0
  assert_eq!(&rgb[3..6], &[0, 0, 0]); // 0.0
  assert_eq!(&rgb[6..9], &[128, 128, 128]); // 0.5 × 255 + 0.5 = 128
  assert_eq!(&rgb[9..12], &[255, 255, 255]); // 1.0
  assert_eq!(&rgb[12..15], &[255, 255, 255]); // 1.5 clamps to 255
}

#[cfg(target_endian = "little")]
#[test]
fn grayf32_with_hsv_h0_s0_v_saturated() {
  let data: std::vec::Vec<f32> = std::vec![0.0, 0.5, 1.0];
  let frame = Grayf32Frame::new(&data, 3, 1, 3);
  let mut h = std::vec![0xFFu8; 3];
  let mut s = std::vec![0xFFu8; 3];
  let mut v = std::vec![0u8; 3];
  let mut sink = MixedSinker::<crate::yuv::Grayf32>::new(3, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0, 0]);
  assert_eq!(s, [0, 0, 0]);
  assert_eq!(v, [0, 128, 255]);
}

#[cfg(target_endian = "little")]
#[test]
fn grayf32_with_luma_u16_and_rgb_u16() {
  // 1×1 frame: Y = 0.5 → luma_u16 ≈ 32768, rgb_u16 ≈ [32768, 32768, 32768]
  let data = std::vec![0.5f32];
  let frame = Grayf32Frame::new(&data, 1, 1, 1);
  let mut lu16 = std::vec![0u16; 1];
  let mut rgb_u16 = std::vec![0u16; 3];
  let mut sink = MixedSinker::<crate::yuv::Grayf32>::new(1, 1)
    .with_luma_u16(&mut lu16)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16)
    .unwrap();
  grayf32_to(&frame, FR, M, &mut sink).unwrap();
  // (0.5 * 65535 + 0.5) as u16 = 32768
  assert_eq!(lu16[0], 32768);
  assert_eq!(rgb_u16, [32768, 32768, 32768]);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_width_128_and_130_smoke() {
  for &w in &[128usize, 130usize] {
    let data: std::vec::Vec<f32> = (0..w).map(|i| i as f32 / w as f32).collect();
    let frame = Grayf32Frame::new(&data, w as u32, 1, w as u32);
    let mut rgb = std::vec![0u8; w * 3];
    let mut luma_f32 = std::vec![0.0f32; w];
    let mut sink = MixedSinker::<crate::yuv::Grayf32>::new(w, 1)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_luma_f32(&mut luma_f32)
      .unwrap();
    grayf32_to(&frame, FR, M, &mut sink).unwrap();
    // Verify first and last pixel.
    assert_eq!(rgb[0], 0, "w={w} first R");
    assert_eq!(luma_f32[0], 0.0, "w={w} first luma_f32");
    assert!(luma_f32[w - 1] > 0.9, "w={w} last luma_f32");
  }
}

/// Sinker-layer Frame-contract regression for codex 3rd-pass review of
/// PR #85.
///
/// [`Grayf32Frame`] documents its `&[f32]` plane as **FFmpeg `grayf32le`**
/// (see `src/frame/gray.rs`): the byte layout is little-endian-encoded f32,
/// produced by FFmpeg and reinterpreted as `&[f32]` via
/// `bytemuck::cast_slice`. This is **not** host-native f32 on a BE host —
/// the bytes are byte-swapped from the intended values until the loader
/// applies `u32::from_le`.
///
/// The `Grayf32` sinker therefore correctly hardcodes `::<false>` (i.e.
/// "input is LE-encoded") on every host:
///
///   • LE host: `from_le` is a no-op → LE bytes read as LE-interpreted f32
///     → correct host-native value.
///   • BE host: `from_le` is a byte-swap → restores LE-encoded bytes to
///     host-native f32 → correct host-native value.
///
/// This test constructs an explicitly LE-encoded f32 fixture (mirroring
/// `bytemuck::cast_slice` over `f32::to_le_bytes` output) and feeds it
/// through the sinker. On a LE host the assertion is vacuous (LE bytes
/// already are host-native), but it pins the contract; on a BE host it
/// catches any regression that drops the `::<false>` routing.
///
/// Replaces the two earlier (incorrectly-typed) regressions that assumed
/// `Grayf32Frame` was host-native f32; the codex 3rd-pass review of
/// commit `1bd851a` caught the contract conflict.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn grayf32_sinker_le_encoded_frame_decodes_correctly() {
  use crate::yuv::Grayf32;

  // Host-native intended values (mix in-range, HDR, negative).
  let w = 16usize;
  let h = 4usize;
  let mut intended = std::vec![0.0f32; w * h];
  for (i, v) in intended.iter_mut().enumerate() {
    *v = match i % 4 {
      0 => 0.5,
      1 => 1.5,
      2 => -0.25,
      _ => 100.0,
    };
  }

  // Build an `&[f32]` whose bit pattern, when read as raw bytes, is
  // little-endian-encoded — i.e. the layout an FFmpeg `grayf32le` plane
  // hands to `bytemuck::cast_slice`. We do this without a `&[u8]` →
  // `&[f32]` cast (which would need 4-byte alignment) by storing the
  // LE-encoded `u32` bit pattern directly into an aligned `Vec<f32>`:
  //
  //   `f32::from_bits(intended.to_bits().to_le())`
  //
  //   • LE host: `to_le` is a no-op → element bits = intended bits → the
  //     in-memory bytes are LE-encoded (which on a LE host is also the
  //     host-native f32 = intended).
  //   • BE host: `to_le` byte-swaps → element bits = byte-swapped intended
  //     → the in-memory bytes are LE-encoded; reinterpreted as host-native
  //     (BE) f32 they are *not* the intended value. The sinker's
  //     `from_le` swap then restores the intended bits.
  let le_plane: std::vec::Vec<f32> = intended
    .iter()
    .map(|&v| f32::from_bits(v.to_bits().to_le()))
    .collect();
  let frame = Grayf32Frame::new(&le_plane, w as u32, h as u32, w as u32);

  // luma_f32 pass-through must restore host-native intended values.
  let mut luma_f32_out = std::vec![0.0f32; w * h];
  {
    let mut sink = MixedSinker::<Grayf32>::new(w, h)
      .with_luma_f32(&mut luma_f32_out)
      .unwrap();
    grayf32_to(&frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma_f32_out, intended,
    "Grayf32 sinker failed to decode LE-encoded plane to host-native"
  );

  // rgb_f32 lossless replicate (R = G = B = host-native Y, bit-exact).
  let mut rgb_f32_out = std::vec![0.0f32; w * h * 3];
  {
    let mut sink = MixedSinker::<Grayf32>::new(w, h)
      .with_rgb_f32(&mut rgb_f32_out)
      .unwrap();
    grayf32_to(&frame, FR, M, &mut sink).unwrap();
  }
  for (x, &y) in intended.iter().enumerate() {
    assert_eq!(rgb_f32_out[x * 3], y, "pixel {x} R diverges");
    assert_eq!(rgb_f32_out[x * 3 + 1], y, "pixel {x} G diverges");
    assert_eq!(rgb_f32_out[x * 3 + 2], y, "pixel {x} B diverges");
  }
}

// ---- Ya8 integration tests --------------------------------------------------

#[test]
fn ya8_with_rgb_and_rgba_strategy_a_plus() {
  // 2-pixel Ya8: [Y=100, A=200], [Y=50, A=150]
  let packed: std::vec::Vec<u8> = std::vec![100, 200, 50, 150];
  let frame = Ya8Frame::new(&packed, 2, 1, 4);
  let mut rgb = std::vec![0u8; 6];
  let mut rgba = std::vec![0u8; 8];
  let mut sink = MixedSinker::<crate::yuv::Ya8>::new(2, 1)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  ya8_to(&frame, FR, M, &mut sink).unwrap();
  // RGB: Y broadcast, α dropped.
  assert_eq!(&rgb[0..3], &[100, 100, 100]);
  assert_eq!(&rgb[3..6], &[50, 50, 50]);
  // RGBA: Y broadcast, α from source.
  assert_eq!(&rgba[0..4], &[100, 100, 100, 200]);
  assert_eq!(&rgba[4..8], &[50, 50, 50, 150]);
}

#[test]
fn ya8_standalone_rgba_source_alpha() {
  // Standalone RGBA path (no RGB requested).
  let packed: std::vec::Vec<u8> = std::vec![77, 11, 88, 22];
  let frame = Ya8Frame::new(&packed, 2, 1, 4);
  let mut rgba = std::vec![0u8; 8];
  let mut sink = MixedSinker::<crate::yuv::Ya8>::new(2, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  ya8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(&rgba[0..4], &[77, 77, 77, 11]);
  assert_eq!(&rgba[4..8], &[88, 88, 88, 22]);
}

#[test]
fn ya8_with_luma_u16_zero_extends() {
  let packed: std::vec::Vec<u8> = std::vec![200, 50, 100, 25];
  let frame = Ya8Frame::new(&packed, 2, 1, 4);
  let mut lu16 = std::vec![0u16; 2];
  let mut sink = MixedSinker::<crate::yuv::Ya8>::new(2, 1)
    .with_luma_u16(&mut lu16)
    .unwrap();
  ya8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(lu16, [200, 100]);
}

#[test]
fn ya8_with_hsv_h0_s0_v_y() {
  let packed: std::vec::Vec<u8> = std::vec![200, 50, 100, 25];
  let frame = Ya8Frame::new(&packed, 2, 1, 4);
  let mut h = std::vec![0xFFu8; 2];
  let mut s = std::vec![0xFFu8; 2];
  let mut v = std::vec![0u8; 2];
  let mut sink = MixedSinker::<crate::yuv::Ya8>::new(2, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  ya8_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0]);
  assert_eq!(s, [0, 0]);
  assert_eq!(v, [200, 100]);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ya8_width_128_and_130_smoke() {
  for &w in &[128usize, 130usize] {
    let packed: std::vec::Vec<u8> = (0..w).flat_map(|i| [i as u8, (255 - i as u8)]).collect();
    let frame = Ya8Frame::new(&packed, w as u32, 1, (w * 2) as u32);
    let mut rgb = std::vec![0u8; w * 3];
    let mut rgba = std::vec![0u8; w * 4];
    let mut sink = MixedSinker::<crate::yuv::Ya8>::new(w, 1)
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap();
    ya8_to(&frame, FR, M, &mut sink).unwrap();
    // Spot-check pixel 0: Y=0, A=255
    assert_eq!(&rgb[0..3], &[0, 0, 0], "w={w}");
    assert_eq!(&rgba[0..4], &[0, 0, 0, 255], "w={w}");
  }
}

// ---- Ya16 integration tests -------------------------------------------------

#[cfg(target_endian = "little")]
#[test]
fn ya16_with_rgba_u16_source_alpha() {
  // 1-pixel: Y=0x8000, A=0x4000
  let packed: std::vec::Vec<u16> = std::vec![0x8000, 0x4000];
  let frame = Ya16Frame::new(&packed, 1, 1, 2);
  let mut rgba_u16 = std::vec![0u16; 4];
  let mut luma_u16 = std::vec![0u16; 1];
  let mut sink = MixedSinker::<crate::yuv::Ya16>::new(1, 1)
    .with_rgba_u16(&mut rgba_u16)
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(&rgba_u16, &[0x8000, 0x8000, 0x8000, 0x4000]);
  assert_eq!(luma_u16[0], 0x8000);
}

#[cfg(target_endian = "little")]
#[test]
fn ya16_with_rgba_u8_source_alpha_shifted() {
  // 2-pixel: [Y=0x8000, A=0x4000], [Y=0xFFFF, A=0x8000]
  let packed: std::vec::Vec<u16> = std::vec![0x8000, 0x4000, 0xFFFF, 0x8000];
  let frame = Ya16Frame::new(&packed, 2, 1, 4);
  let mut rgba = std::vec![0u8; 8];
  let mut sink = MixedSinker::<crate::yuv::Ya16>::new(2, 1)
    .with_rgba(&mut rgba)
    .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
  // Y=0x8000>>8=0x80=128, A=0x4000>>8=0x40=64
  assert_eq!(&rgba[0..4], &[0x80, 0x80, 0x80, 0x40]);
  // Y=0xFFFF>>8=0xFF=255, A=0x8000>>8=0x80=128
  assert_eq!(&rgba[4..8], &[0xFF, 0xFF, 0xFF, 0x80]);
}

#[cfg(target_endian = "little")]
#[test]
fn ya16_with_rgb_and_rgba_strategy_a_plus() {
  let packed: std::vec::Vec<u16> = std::vec![0x8000, 0x4000, 0x2000, 0xC000];
  let frame = Ya16Frame::new(&packed, 2, 1, 4);
  let mut rgb = std::vec![0u8; 6];
  let mut rgba = std::vec![0u8; 8];
  let mut sink = MixedSinker::<crate::yuv::Ya16>::new(2, 1)
    .with_rgb(&mut rgb)
    .unwrap()
    .with_rgba(&mut rgba)
    .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
  // Y=0x8000>>8=0x80, A dropped for rgb.
  assert_eq!(&rgb[0..3], &[0x80, 0x80, 0x80]);
  // RGBA: Y broadcast, A=0x4000>>8=0x40
  assert_eq!(&rgba[0..4], &[0x80, 0x80, 0x80, 0x40]);
  // Pixel 1: Y=0x2000>>8=0x20, A=0xC000>>8=0xC0
  assert_eq!(&rgb[3..6], &[0x20, 0x20, 0x20]);
  assert_eq!(&rgba[4..8], &[0x20, 0x20, 0x20, 0xC0]);
}

#[cfg(target_endian = "little")]
#[test]
fn ya16_with_hsv_h0_s0_v_shifted() {
  let packed: std::vec::Vec<u16> = std::vec![0x8000, 0x4000, 0xFFFF, 0x0000];
  let frame = Ya16Frame::new(&packed, 2, 1, 4);
  let mut h = std::vec![0xFFu8; 2];
  let mut s = std::vec![0xFFu8; 2];
  let mut v = std::vec![0u8; 2];
  let mut sink = MixedSinker::<crate::yuv::Ya16>::new(2, 1)
    .with_hsv(&mut h, &mut s, &mut v)
    .unwrap();
  ya16_to(&frame, FR, M, &mut sink).unwrap();
  assert_eq!(h, [0, 0]);
  assert_eq!(s, [0, 0]);
  assert_eq!(v, [0x80, 0xFF]);
}

/// Strategy A+ (combined `with_rgb` + `with_rgba`) must produce α bytes
/// byte-identical to the standalone `with_rgba` path. Locks down the
/// codex-flagged corruption where a BE host processing the LE-encoded
/// `Ya16Frame` would otherwise diverge between the two paths: standalone
/// uses the endian-aware `ya16_to_rgba_row::<false>` kernel; combined
/// expanded RGB → RGBA then patched α via `copy_alpha_ya_u16_to_u8` which
/// previously read raw `packed[n*2+1]` host-native and so emitted a
/// byte-reversed α byte on BE. After the fix, `copy_alpha_ya_u16_to_u8`
/// is target-endian-aware (`<false>` for the LE Frame contract) and the
/// two paths agree on every host.
///
/// To exercise the LE-encoded byte contract on every host we build the
/// `&[u16]` plane by bit-casting LE bytes — `u16::from_le_bytes` per
/// sample. On LE hosts that's a no-op; on BE hosts it byte-swaps so the
/// in-memory bytes match the FFmpeg `AV_PIX_FMT_YA16LE` layout.
#[test]
fn ya16_combined_rgb_and_rgba_alpha_matches_standalone_le_encoded() {
  let w: u32 = 8;
  let h: u32 = 1;
  // Logical samples (Y, A) per pixel.
  let samples: [(u16, u16); 8] = [
    (0x0000, 0xFFFF),
    (0x8000, 0x4000),
    (0xFFFF, 0x0000),
    (0x1234, 0xABCD),
    (0x00FF, 0xFF00),
    (0x5A5A, 0xA5A5),
    (0x7FFF, 0x8000),
    (0xC000, 0x3FFF),
  ];
  // Build the `&[u16]` plane such that its in-memory bytes match the
  // FFmpeg `AV_PIX_FMT_YA16LE` byte layout on every host. We want a
  // host-native u16 whose underlying bytes spell `[low, high]` (LE):
  // `u16::from_ne_bytes(x.to_le_bytes())` is `x` on LE and `x.swap_bytes()`
  // on BE — the right value to store in either case.
  let le_encoded = |x: u16| -> u16 { u16::from_ne_bytes(x.to_le_bytes()) };
  let packed: std::vec::Vec<u16> = samples
    .iter()
    .flat_map(|&(y, a)| [le_encoded(y), le_encoded(a)])
    .collect();
  let frame = Ya16Frame::new(&packed, w, h, w * 2);

  // Run combined (with_rgb + with_rgba) — exercises Strategy A+ with the
  // newly endian-aware `copy_alpha_ya_u16_to_u8::<false>`. Forces
  // `with_simd(false)` so the test runs purely scalar — no SIMD intrinsics
  // — which lets it execute under `cargo miri test`. BE CI is driven by
  // miri on s390x / powerpc64; gating it out of miri would skip exactly
  // the host where BE corruption would surface.
  let mut rgb_combined = std::vec![0u8; (w * h * 3) as usize];
  let mut rgba_combined = std::vec![0u8; (w * h * 4) as usize];
  {
    let mut sink = MixedSinker::<crate::yuv::Ya16>::new(w as usize, h as usize)
      .with_simd(false)
      .with_rgb(&mut rgb_combined)
      .unwrap()
      .with_rgba(&mut rgba_combined)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  // Run standalone (with_rgba only) — exercises the endian-aware
  // `ya16_to_rgba_row::<false>` kernel. Same scalar-only rationale.
  let mut rgba_standalone = std::vec![0u8; (w * h * 4) as usize];
  {
    let mut sink = MixedSinker::<crate::yuv::Ya16>::new(w as usize, h as usize)
      .with_simd(false)
      .with_rgba(&mut rgba_standalone)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combined, rgba_standalone,
    "combined (with_rgb+with_rgba) RGBA must equal standalone with_rgba"
  );
}

/// u16 RGBA variant of the combined-vs-standalone parity check. Locks
/// down `copy_alpha_ya_u16::<false>` (the u16 alpha-patch helper for
/// 16-bit RGBA outputs).
#[test]
fn ya16_combined_rgb_u16_and_rgba_u16_alpha_matches_standalone_le_encoded() {
  let w: u32 = 8;
  let h: u32 = 1;
  let samples: [(u16, u16); 8] = [
    (0x0000, 0xFFFF),
    (0x8000, 0x4000),
    (0xFFFF, 0x0000),
    (0x1234, 0xABCD),
    (0x00FF, 0xFF00),
    (0x5A5A, 0xA5A5),
    (0x7FFF, 0x8000),
    (0xC000, 0x3FFF),
  ];
  // See sibling test for the `le_encoded` rationale.
  let le_encoded = |x: u16| -> u16 { u16::from_ne_bytes(x.to_le_bytes()) };
  let packed: std::vec::Vec<u16> = samples
    .iter()
    .flat_map(|&(y, a)| [le_encoded(y), le_encoded(a)])
    .collect();
  let frame = Ya16Frame::new(&packed, w, h, w * 2);

  // Forces `with_simd(false)` so this test runs purely scalar — no SIMD
  // intrinsics — which lets it execute under `cargo miri test`. BE CI is
  // driven by miri on s390x / powerpc64; gating it out of miri would skip
  // exactly the host where BE corruption would surface.
  let mut rgb_combined = std::vec![0u16; (w * h * 3) as usize];
  let mut rgba_combined = std::vec![0u16; (w * h * 4) as usize];
  {
    let mut sink = MixedSinker::<crate::yuv::Ya16>::new(w as usize, h as usize)
      .with_simd(false)
      .with_rgb_u16(&mut rgb_combined)
      .unwrap()
      .with_rgba_u16(&mut rgba_combined)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  let mut rgba_standalone = std::vec![0u16; (w * h * 4) as usize];
  {
    let mut sink = MixedSinker::<crate::yuv::Ya16>::new(w as usize, h as usize)
      .with_simd(false)
      .with_rgba_u16(&mut rgba_standalone)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
  }

  assert_eq!(
    rgba_combined, rgba_standalone,
    "combined (with_rgb_u16+with_rgba_u16) RGBA u16 must equal standalone"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn ya16_width_128_and_130_smoke() {
  for &w in &[128usize, 130usize] {
    let packed: std::vec::Vec<u16> = (0..w)
      .flat_map(|i| [(i as u16) << 8, (255u16 - i as u16) << 8])
      .collect();
    let frame = Ya16Frame::new(&packed, w as u32, 1, (w * 2) as u32);
    let mut rgba = std::vec![0u8; w * 4];
    let mut luma_u16 = std::vec![0u16; w];
    let mut sink = MixedSinker::<crate::yuv::Ya16>::new(w, 1)
      .with_rgba(&mut rgba)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
    ya16_to(&frame, FR, M, &mut sink).unwrap();
    // Pixel 0: Y=0, A=255<<8=0xFF00 → a8=0xFF
    assert_eq!(&rgba[0..4], &[0, 0, 0, 0xFF], "w={w} px0");
  }
}
