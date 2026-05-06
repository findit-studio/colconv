//! Frame validation and kernel correctness tests for Monoblack / Monowhite.

#[cfg(test)]
mod tests {
  use crate::{
    frame::{MonoFrameError, MonoblackFrame, MonowhiteFrame},
    row::scalar::mono1bit as scalar,
  };

  #[test]
  fn monoblack_frame_construction_ok() {
    let data = [0b10101010u8; 16]; // 2 rows of 8 bytes each
    let frame = MonoblackFrame::try_new(&data, 64, 2, 8).expect("valid");
    assert_eq!(frame.width(), 64);
    assert_eq!(frame.height(), 2);
    assert_eq!(frame.stride(), 8);
  }

  #[test]
  fn monowhite_frame_construction_ok() {
    let data = [0b01010101u8; 16];
    let frame = MonowhiteFrame::try_new(&data, 64, 2, 8).expect("valid");
    assert_eq!(frame.width(), 64);
    assert_eq!(frame.height(), 2);
    assert_eq!(frame.stride(), 8);
  }

  #[test]
  fn monoblack_frame_zero_width() {
    let data = [0u8; 16];
    let err = MonoblackFrame::try_new(&data, 0, 2, 1).unwrap_err();
    assert!(matches!(err, MonoFrameError::ZeroDimension { .. }));
  }

  #[test]
  fn monoblack_frame_zero_height() {
    let data = [0u8; 16];
    let err = MonoblackFrame::try_new(&data, 64, 0, 8).unwrap_err();
    assert!(matches!(err, MonoFrameError::ZeroDimension { .. }));
  }

  #[test]
  fn monoblack_frame_stride_too_small() {
    let data = [0u8; 16];
    // width=64 requires at least 8 bytes stride; 7 is too small
    let err = MonoblackFrame::try_new(&data, 64, 2, 7).unwrap_err();
    assert!(matches!(err, MonoFrameError::StrideTooSmall { .. }));
  }

  #[test]
  fn monoblack_frame_data_too_short() {
    let data = [0u8; 15]; // stride=8, height=2 needs 16 bytes
    let err = MonoblackFrame::try_new(&data, 64, 2, 8).unwrap_err();
    assert!(matches!(err, MonoFrameError::DataPlaneTooShort { .. }));
  }

  #[test]
  fn monowhite_frame_stride_too_small() {
    let data = [0u8; 16];
    let err = MonowhiteFrame::try_new(&data, 33, 2, 4).unwrap_err();
    // width=33 requires ceil(33/8) = 5 bytes; 4 is too small
    assert!(matches!(err, MonoFrameError::StrideTooSmall { .. }));
  }

  #[test]
  fn monoblack_unpack_alternating() {
    let data = [0b10101010u8]; // alternating: bit=1,0,1,0,1,0,1,0
    let mut out = vec![0u8; 8];
    scalar::monoblack_to_luma_row(&data, &mut out, 8);
    // MSB first: pixel 0 is bit 7 = 1, pixel 1 is bit 6 = 0, etc.
    assert_eq!(out, vec![255, 0, 255, 0, 255, 0, 255, 0]);
  }

  #[test]
  fn monoblack_all_ones() {
    let data = [0xFFu8];
    let mut out = vec![0u8; 8];
    scalar::monoblack_to_luma_row(&data, &mut out, 8);
    assert_eq!(out, vec![255; 8]);
  }

  #[test]
  fn monoblack_all_zeros() {
    let data = [0x00u8];
    let mut out = vec![0u8; 8];
    scalar::monoblack_to_luma_row(&data, &mut out, 8);
    assert_eq!(out, vec![0; 8]);
  }

  #[test]
  fn monoblack_partial_row() {
    let data = [0b11100000u8]; // 3 ones, 5 zeros; width=5 takes first 5 bits
    let mut out = vec![0u8; 5];
    scalar::monoblack_to_luma_row(&data, &mut out, 5);
    // first 5 bits (MSB-first): 1,1,1,0,0 → 255,255,255,0,0
    assert_eq!(out, vec![255, 255, 255, 0, 0]);
  }

  #[test]
  fn monoblack_partial_byte_edge() {
    let data = [0b11110000u8, 0b10000000u8]; // crosses byte boundary
    let mut out = vec![0u8; 10];
    scalar::monoblack_to_luma_row(&data, &mut out, 10);
    // first byte: 1,1,1,1,0,0,0,0
    // second byte (first 2 bits): 1,0
    assert_eq!(out, vec![255, 255, 255, 255, 0, 0, 0, 0, 255, 0]);
  }

  #[test]
  fn monowhite_unpack_alternating() {
    let data = [0b10101010u8];
    let mut out = vec![0u8; 8];
    scalar::monowhite_to_luma_row(&data, &mut out, 8);
    // Inverted: pixel 0 (bit 1) → 0, pixel 1 (bit 0) → 255, etc.
    assert_eq!(out, vec![0, 255, 0, 255, 0, 255, 0, 255]);
  }

  #[test]
  fn monowhite_all_ones() {
    let data = [0xFFu8];
    let mut out = vec![0u8; 8];
    scalar::monowhite_to_luma_row(&data, &mut out, 8);
    assert_eq!(out, vec![0; 8]); // inverted: all ones → all zeros
  }

  #[test]
  fn monowhite_all_zeros() {
    let data = [0x00u8];
    let mut out = vec![0u8; 8];
    scalar::monowhite_to_luma_row(&data, &mut out, 8);
    assert_eq!(out, vec![255; 8]); // inverted: all zeros → all ones
  }

  #[test]
  fn monowhite_partial_row() {
    let data = [0b11100000u8];
    let mut out = vec![0u8; 5];
    scalar::monowhite_to_luma_row(&data, &mut out, 5);
    // Inverted: 1,1,1,0,0 → 0,0,0,255,255
    assert_eq!(out, vec![0, 0, 0, 255, 255]);
  }

  #[test]
  fn monoblack_to_rgb_broadcast() {
    let data = [0b11110000u8];
    let mut out = vec![0u8; 8 * 3];
    scalar::monoblack_to_rgb_row(&data, &mut out, 8);
    // First 4 pixels: 255 (broadcast to R, G, B)
    // Next 4 pixels: 0
    for i in 0..4 {
      assert_eq!(out[i * 3], 255);
      assert_eq!(out[i * 3 + 1], 255);
      assert_eq!(out[i * 3 + 2], 255);
    }
    for i in 4..8 {
      assert_eq!(out[i * 3], 0);
      assert_eq!(out[i * 3 + 1], 0);
      assert_eq!(out[i * 3 + 2], 0);
    }
  }

  #[test]
  fn monoblack_to_rgba_broadcast() {
    let data = [0xFFu8];
    let mut out = vec![0u8; 8 * 4];
    scalar::monoblack_to_rgba_row(&data, &mut out, 8);
    for i in 0..8 {
      assert_eq!(out[i * 4], 255); // R
      assert_eq!(out[i * 4 + 1], 255); // G
      assert_eq!(out[i * 4 + 2], 255); // B
      assert_eq!(out[i * 4 + 3], 0xFF); // A
    }
  }

  #[test]
  fn monowhite_to_rgb_broadcast() {
    let data = [0b11110000u8];
    let mut out = vec![0u8; 8 * 3];
    scalar::monowhite_to_rgb_row(&data, &mut out, 8);
    // Inverted: 1,1,1,1,0,0,0,0 → 0,0,0,0,255,255,255,255
    for i in 0..4 {
      assert_eq!(out[i * 3], 0);
      assert_eq!(out[i * 3 + 1], 0);
      assert_eq!(out[i * 3 + 2], 0);
    }
    for i in 4..8 {
      assert_eq!(out[i * 3], 255);
      assert_eq!(out[i * 3 + 1], 255);
      assert_eq!(out[i * 3 + 2], 255);
    }
  }

  #[test]
  fn monoblack_to_luma_u16() {
    let data = [0b11110000u8];
    let mut out = vec![0u16; 8];
    scalar::monoblack_to_luma_u16_row(&data, &mut out, 8);
    for i in 0..4 {
      assert_eq!(out[i], 0xFFFF); // white
    }
    for i in 4..8 {
      assert_eq!(out[i], 0x0000); // black
    }
  }

  #[test]
  fn monowhite_to_rgba_u16() {
    let data = [0x00u8];
    let mut out = vec![0u16; 8 * 4];
    scalar::monowhite_to_rgba_u16_row(&data, &mut out, 8);
    for i in 0..8 {
      assert_eq!(out[i * 4], 0xFFFF); // R (inverted: 0 → 255)
      assert_eq!(out[i * 4 + 1], 0xFFFF); // G
      assert_eq!(out[i * 4 + 2], 0xFFFF); // B
      assert_eq!(out[i * 4 + 3], 0xFFFF); // A
    }
  }

  #[test]
  fn monoblack_to_hsv() {
    let data = [0b11110000u8];
    let mut h = vec![0u8; 8];
    let mut s = vec![0u8; 8];
    let mut v = vec![0u8; 8];
    scalar::monoblack_to_hsv_row(&data, &mut h, &mut s, &mut v, 8);
    assert_eq!(h, vec![0; 8]); // all H=0
    assert_eq!(s, vec![0; 8]); // all S=0
    assert_eq!(v, vec![255, 255, 255, 255, 0, 0, 0, 0]); // V follows luma
  }

  #[test]
  fn monowhite_to_hsv() {
    let data = [0xFFu8];
    let mut h = vec![0u8; 8];
    let mut s = vec![0u8; 8];
    let mut v = vec![0u8; 8];
    scalar::monowhite_to_hsv_row(&data, &mut h, &mut s, &mut v, 8);
    assert_eq!(h, vec![0; 8]); // all H=0
    assert_eq!(s, vec![0; 8]); // all S=0
    assert_eq!(v, vec![0; 8]); // inverted: all ones → all zeros
  }

  #[test]
  fn monoblack_to_rgb_u16() {
    let data = [0x80u8]; // 1,0,0,0,0,0,0,0
    let mut out = vec![0u16; 8 * 3];
    scalar::monoblack_to_rgb_u16_row(&data, &mut out, 8);
    // First pixel: 255 (0xFFFF)
    assert_eq!(out[0], 0xFFFF);
    assert_eq!(out[1], 0xFFFF);
    assert_eq!(out[2], 0xFFFF);
    // Remaining pixels: 0
    for i in 1..8 {
      assert_eq!(out[i * 3], 0);
      assert_eq!(out[i * 3 + 1], 0);
      assert_eq!(out[i * 3 + 2], 0);
    }
  }

  #[test]
  fn monoblack_frame_const_new() {
    // Test const constructor
    const DATA: &[u8] = &[0xABu8; 8];
    let frame = MonoblackFrame::new(DATA, 32, 1, 4);
    assert_eq!(frame.width(), 32);
    assert_eq!(frame.height(), 1);
    assert_eq!(frame.stride(), 4);
  }

  #[test]
  #[should_panic(expected = "invalid MonoFrame")]
  fn monoblack_frame_new_panics_on_invalid() {
    const DATA: &[u8] = &[0u8; 4];
    // Too small: needs 64 width → 8 bytes, but data is only 4
    let _ = MonoblackFrame::new(DATA, 64, 1, 8);
  }
}
