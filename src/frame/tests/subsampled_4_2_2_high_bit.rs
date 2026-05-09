use super::*;

// ---- Yuv422pFrame16::try_new_checked ---------------------------------

fn p422_planes_10bit() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // Width 16, height 8 — 4:2:2 chroma is half-width, FULL-height.
  let y = std::vec![64u16; 16 * 8];
  let u = std::vec![512u16; 8 * 8];
  let v = std::vec![512u16; 8 * 8];
  (y, u, v)
}

#[test]
fn yuv422p10_try_new_checked_accepts_in_range_samples() {
  let (y, u, v) = p422_planes_10bit();
  let f = Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid 10-bit");
  assert_eq!(f.width(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv422p10_try_new_checked_accepts_max_valid_value() {
  // Exactly `(1 << 10) - 1 = 1023` must pass.
  let y = std::vec![1023u16; 16 * 8];
  let u = std::vec![1023u16; 8 * 8];
  let v = std::vec![1023u16; 8 * 8];
  Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).expect("max valid passes");
}

#[test]
fn yuv422p10_try_new_checked_rejects_y_high_bit_set() {
  let mut y = std::vec![0u16; 16 * 8];
  y[3 * 16 + 5] = 0x8000;
  let u = std::vec![512u16; 8 * 8];
  let v = std::vec![512u16; 8 * 8];
  let e = Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  match e {
    Yuv420pFrame16Error::SampleOutOfRange {
      plane,
      value,
      max_valid,
      ..
    } => {
      assert_eq!(plane, Yuv420pFrame16Plane::Y);
      assert_eq!(value, 0x8000);
      assert_eq!(max_valid, 1023);
    }
    other => panic!("expected SampleOutOfRange, got {other:?}"),
  }
}

#[test]
fn yuv422p10_try_new_checked_rejects_u_plane_sample_in_full_height_chroma() {
  // Crucial 4:2:2-specific test: the offending sample is on the
  // last chroma row (row 7), which only exists because 4:2:2
  // chroma is full-height (8 rows). The 4:2:0 scan would stop at
  // row 3.
  let y = std::vec![0u16; 16 * 8];
  let mut u = std::vec![512u16; 8 * 8];
  u[7 * 8 + 3] = 1024; // last chroma row, just above max
  let v = std::vec![512u16; 8 * 8];
  let e = Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::U,
      value: 1024,
      max_valid: 1023,
      ..
    }
  ));
}

#[test]
fn yuv422p10_try_new_checked_rejects_v_plane_sample() {
  let y = std::vec![0u16; 16 * 8];
  let u = std::vec![512u16; 8 * 8];
  let mut v = std::vec![512u16; 8 * 8];
  v[5 * 8 + 6] = 0xFFFF;
  let e = Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::V,
      ..
    }
  ));
}

#[test]
fn yuv422p12_try_new_checked_rejects_above_4095() {
  let mut y = std::vec![2048u16; 16 * 8];
  y[0] = 4096; // just above 12-bit max
  let u = std::vec![2048u16; 8 * 8];
  let v = std::vec![2048u16; 8 * 8];
  let e = Yuv422p12Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      value: 4096,
      max_valid: 4095,
      ..
    }
  ));
}

#[test]
fn yuv422p16_try_new_checked_accepts_full_u16_range() {
  // At 16 bits the full u16 range is valid — no scan needed.
  let y = std::vec![65535u16; 16 * 8];
  let u = std::vec![32768u16; 8 * 8];
  let v = std::vec![32768u16; 8 * 8];
  Yuv422p16Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8)
    .expect("every u16 value is in range at 16 bits");
}

// ---- Host-independent BE-host regressions (codex round-2) -----------
//
// Build planes from LE-encoded bytes via `to_le_bytes` and read back
// via `from_ne_bytes`. On LE host the buffer matches the literal; on
// BE host every `u16` is byte-swapped. The validator must `from_le`-
// normalize before the range check on both hosts. See the comment at
// the bottom of `subsampled_4_2_0_high_bit.rs` for the full rationale.

fn le_encoded_u16_buf(intended: &[u16]) -> std::vec::Vec<u16> {
  let bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect()
}

#[test]
fn yuv422p10_try_new_checked_accepts_le_encoded_buffer_on_any_host() {
  // 4:2:2: chroma is half-width × full-height.
  let intended_y = std::vec![1023u16; 16 * 8];
  let intended_uv = std::vec![512u16; 8 * 8];
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_uv);
  let v = le_encoded_u16_buf(&intended_uv);
  Yuv422p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8)
    .expect("LE-encoded valid yuv422p10le must be accepted on both LE and BE hosts");
}

#[test]
fn yuv422p12_try_new_checked_rejects_le_encoded_out_of_range_on_any_host() {
  // After `from_le` normalization, the offending sample is 4096
  // (just above 12-bit max 4095).
  let mut intended_y = std::vec![2048u16; 16 * 8];
  intended_y[5] = 4096;
  let intended_uv = std::vec![2048u16; 8 * 8];
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_uv);
  let v = le_encoded_u16_buf(&intended_uv);
  let e = Yuv422p12Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::Y,
      value: 4096,
      max_valid: 4095,
      ..
    }
  ));
}

#[test]
fn p210_try_new_checked_accepts_le_encoded_buffer_on_any_host() {
  // P210 white = 1023 << 6 = 0xFFC0; LE bytes [0xC0, 0xFF].
  // 4:2:2 PnFrame422: chroma is half-width pairs × full-height ⇒
  // each UV row holds `width` u16 elements (= width/2 pairs × 2).
  let intended_y = std::vec![0xFFC0u16; 16 * 8];
  let intended_uv = std::vec![0x8000u16; 16 * 8];
  let y = le_encoded_u16_buf(&intended_y);
  let uv = le_encoded_u16_buf(&intended_uv);
  P210Frame::try_new_checked(&y, &uv, 16, 8, 16, 16)
    .expect("LE-encoded valid P210 must be accepted on both LE and BE hosts");
}

#[test]
fn p210_try_new_checked_rejects_le_encoded_low_bits_on_any_host() {
  // After `from_le` normalization, the logical 0x03FF has all six
  // low bits set — `yuv422p10le`-style data wrongly handed to P210.
  let mut intended_y = std::vec![0xFFC0u16; 16 * 8];
  intended_y[2 * 16 + 7] = 0x03FF;
  let intended_uv = std::vec![0x8000u16; 16 * 8];
  let y = le_encoded_u16_buf(&intended_y);
  let uv = le_encoded_u16_buf(&intended_uv);
  let e = P210Frame::try_new_checked(&y, &uv, 16, 8, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    PnFrameError::SampleLowBitsSet {
      plane: PnFramePlane::Y,
      value: 0x03FF,
      low_bits: 6,
      ..
    }
  ));
}
