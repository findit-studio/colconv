use super::*;

// ---- Yuv444pFrame16::try_new_checked ---------------------------------

fn p444_planes_10bit() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // 4:4:4: chroma is FULL-width, full-height (1:1 with Y).
  let y = le_encoded_u16_buf(&std::vec![64u16; 16 * 8]);
  let u = le_encoded_u16_buf(&std::vec![512u16; 16 * 8]);
  let v = le_encoded_u16_buf(&std::vec![512u16; 16 * 8]);
  (y, u, v)
}

#[test]
fn yuv444p10_try_new_checked_accepts_in_range_samples() {
  let (y, u, v) = p444_planes_10bit();
  let f = Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).expect("valid 10-bit");
  assert_eq!(f.width(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv444p10_try_new_checked_accepts_max_valid_value() {
  let y = le_encoded_u16_buf(&std::vec![1023u16; 16 * 8]);
  let u = le_encoded_u16_buf(&std::vec![1023u16; 16 * 8]);
  let v = le_encoded_u16_buf(&std::vec![1023u16; 16 * 8]);
  Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).expect("max valid passes");
}

#[test]
fn yuv444p10_try_new_checked_rejects_y_high_bit_set() {
  let mut intended_y = std::vec![0u16; 16 * 8];
  intended_y[2 * 16 + 9] = 0x8000;
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&std::vec![512u16; 16 * 8]);
  let v = le_encoded_u16_buf(&std::vec![512u16; 16 * 8]);
  let e = Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::Y,
      value: 0x8000,
      max_valid: 1023,
      ..
    }
  ));
}

#[test]
fn yuv444p10_try_new_checked_rejects_u_plane_sample_in_full_width_chroma() {
  // 4:4:4-specific: the offending sample is in the FULL-WIDTH
  // chroma plane, at column 13 (which doesn't exist in 4:2:0/4:2:2
  // half-width chroma). Forces the scan to extend across the full
  // chroma width.
  let mut intended_u = std::vec![512u16; 16 * 8];
  intended_u[3 * 16 + 13] = 1024;
  let y = le_encoded_u16_buf(&std::vec![0u16; 16 * 8]);
  let u = le_encoded_u16_buf(&intended_u);
  let v = le_encoded_u16_buf(&std::vec![512u16; 16 * 8]);
  let e = Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
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
fn yuv444p10_try_new_checked_rejects_v_plane_sample() {
  // LE-encoded byte storage so the test asserts the same logical values
  // on every host (see 4:2:0 sibling for the BE-host failure mode).
  let intended_y = std::vec![0u16; 16 * 8];
  let intended_u = std::vec![512u16; 16 * 8];
  let mut intended_v = std::vec![512u16; 16 * 8];
  intended_v[7 * 16 + 15] = 0xFFFF; // last chroma sample
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_u);
  let v = le_encoded_u16_buf(&intended_v);
  let e = Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::V,
      ..
    }
  ));
}

#[test]
fn yuv444p14_try_new_checked_rejects_above_16383() {
  let mut intended_y = std::vec![8192u16; 16 * 8];
  intended_y[42] = 16384; // just above 14-bit max
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&std::vec![8192u16; 16 * 8]);
  let v = le_encoded_u16_buf(&std::vec![8192u16; 16 * 8]);
  let e = Yuv444p14Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      value: 16384,
      max_valid: 16383,
      ..
    }
  ));
}

#[test]
fn yuv444p16_try_new_checked_accepts_full_u16_range() {
  let y = std::vec![65535u16; 16 * 8];
  let u = std::vec![32768u16; 16 * 8];
  let v = std::vec![32768u16; 16 * 8];
  Yuv444p16Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16)
    .expect("every u16 value is in range at 16 bits");
}

// ---- Yuv440p10/12 checked-constructor tests ---------------------------

fn p440_planes_10bit() -> (std::vec::Vec<u16>, std::vec::Vec<u16>, std::vec::Vec<u16>) {
  // 4:4:0: chroma is FULL-width × HALF-height (8 / 2 = 4 chroma rows).
  let y = le_encoded_u16_buf(&std::vec![64u16; 16 * 8]);
  let u = le_encoded_u16_buf(&std::vec![512u16; 16 * 4]);
  let v = le_encoded_u16_buf(&std::vec![512u16; 16 * 4]);
  (y, u, v)
}

#[test]
fn yuv440p10_try_new_checked_accepts_in_range_samples() {
  let (y, u, v) = p440_planes_10bit();
  let f = Yuv440p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).expect("valid 10-bit");
  assert_eq!(f.width(), 16);
  assert_eq!(f.bits(), 10);
}

#[test]
fn yuv440p10_try_new_checked_rejects_y_high_bit_set() {
  let mut intended_y = std::vec![0u16; 16 * 8];
  intended_y[2 * 16 + 9] = 0x8000;
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&std::vec![512u16; 16 * 4]);
  let v = le_encoded_u16_buf(&std::vec![512u16; 16 * 4]);
  let e = Yuv440p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::Y,
      value: 0x8000,
      max_valid: 1023,
      ..
    }
  ));
}

#[test]
fn yuv440p10_try_new_checked_rejects_u_plane_sample_in_full_width_chroma() {
  // 4:4:0-specific: chroma is full-width × half-height. Plant the
  // bad sample at column 13 (would be out of range for half-width
  // 4:2:0/4:2:2 chroma) on the last chroma row (index 3 for height
  // 8 ⇒ 4 chroma rows).
  let mut intended_u = std::vec![512u16; 16 * 4];
  intended_u[3 * 16 + 13] = 1024;
  let y = le_encoded_u16_buf(&std::vec![0u16; 16 * 8]);
  let u = le_encoded_u16_buf(&intended_u);
  let v = le_encoded_u16_buf(&std::vec![512u16; 16 * 4]);
  let e = Yuv440p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
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
fn yuv440p10_try_new_checked_rejects_v_plane_sample() {
  // LE-encoded byte storage so the test asserts the same logical values
  // on every host (see 4:2:0 sibling for the BE-host failure mode).
  let intended_y = std::vec![0u16; 16 * 8];
  let intended_u = std::vec![512u16; 16 * 4];
  let mut intended_v = std::vec![512u16; 16 * 4];
  intended_v[3 * 16 + 15] = 0xFFFF; // last chroma sample of the last chroma row
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_u);
  let v = le_encoded_u16_buf(&intended_v);
  let e = Yuv440p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::V,
      ..
    }
  ));
}

#[test]
fn yuv440p12_try_new_checked_rejects_above_4095() {
  let mut intended_y = std::vec![2048u16; 16 * 8];
  intended_y[42] = 4096; // just above 12-bit max
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&std::vec![2048u16; 16 * 4]);
  let v = le_encoded_u16_buf(&std::vec![2048u16; 16 * 4]);
  let e = Yuv440p12Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      value: 4096,
      max_valid: 4095,
      ..
    }
  ));
}

// ---- Host-independent BE-host regressions (codex round-2) -----------
//
// See `subsampled_4_2_0_high_bit.rs` for the full rationale: build
// planes from LE-encoded bytes so on BE hosts the validator's
// `u16::from_le` normalization is exercised end-to-end.

fn le_encoded_u16_buf(intended: &[u16]) -> std::vec::Vec<u16> {
  let bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect()
}

#[test]
fn yuv444p10_try_new_checked_accepts_le_encoded_buffer_on_any_host() {
  // 4:4:4: chroma is full-width × full-height.
  let intended_y = std::vec![1023u16; 16 * 8];
  let intended_uv = std::vec![512u16; 16 * 8];
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_uv);
  let v = le_encoded_u16_buf(&intended_uv);
  Yuv444p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16)
    .expect("LE-encoded valid yuv444p10le must be accepted on both LE and BE hosts");
}

#[test]
fn yuv444p14_try_new_checked_rejects_le_encoded_out_of_range_on_any_host() {
  // After `from_le` normalization, the offending sample is 16384
  // (just above 14-bit max 16383).
  let intended_y = std::vec![8192u16; 16 * 8];
  let intended_u = std::vec![8192u16; 16 * 8];
  let mut intended_v = std::vec![8192u16; 16 * 8];
  intended_v[3 * 16 + 11] = 16384;
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_u);
  let v = le_encoded_u16_buf(&intended_v);
  let e = Yuv444p14Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::V,
      value: 16384,
      max_valid: 16383,
      ..
    }
  ));
}

#[test]
fn yuv440p10_try_new_checked_accepts_le_encoded_buffer_on_any_host() {
  // 4:4:0: chroma is full-width × half-height.
  let intended_y = std::vec![1023u16; 16 * 8];
  let intended_uv = std::vec![512u16; 16 * 4];
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_uv);
  let v = le_encoded_u16_buf(&intended_uv);
  Yuv440p10Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16)
    .expect("LE-encoded valid yuv440p10le must be accepted on both LE and BE hosts");
}

#[test]
fn yuv440p12_try_new_checked_rejects_le_encoded_out_of_range_on_any_host() {
  let intended_y = std::vec![2048u16; 16 * 8];
  let mut intended_u = std::vec![2048u16; 16 * 4];
  intended_u[2 * 16 + 5] = 4096;
  let intended_v = std::vec![2048u16; 16 * 4];
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_u);
  let v = le_encoded_u16_buf(&intended_v);
  let e = Yuv440p12Frame::try_new_checked(&y, &u, &v, 16, 8, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuv420pFrame16Error::SampleOutOfRange {
      plane: Yuv420pFrame16Plane::U,
      value: 4096,
      max_valid: 4095,
      ..
    }
  ));
}

#[test]
fn p410_try_new_checked_accepts_le_encoded_buffer_on_any_host() {
  // 4:4:4 PnFrame444: chroma is full-width × full-height with 2 u16
  // per pair ⇒ each UV row holds 2 * width u16 elements.
  let intended_y = std::vec![0xFFC0u16; 16 * 8];
  let intended_uv = std::vec![0x8000u16; 32 * 8];
  let y = le_encoded_u16_buf(&intended_y);
  let uv = le_encoded_u16_buf(&intended_uv);
  P410Frame::try_new_checked(&y, &uv, 16, 8, 16, 32)
    .expect("LE-encoded valid P410 must be accepted on both LE and BE hosts");
}

#[test]
fn p410_try_new_checked_rejects_le_encoded_low_bits_on_any_host() {
  // Logical 0x03FF (low 6 bits all set) on the UV plane.
  let intended_y = std::vec![0xFFC0u16; 16 * 8];
  let mut intended_uv = std::vec![0x8000u16; 32 * 8];
  intended_uv[4 * 32 + 17] = 0x03FF;
  let y = le_encoded_u16_buf(&intended_y);
  let uv = le_encoded_u16_buf(&intended_uv);
  let e = P410Frame::try_new_checked(&y, &uv, 16, 8, 16, 32).unwrap_err();
  assert!(matches!(
    e,
    PnFrameError::SampleLowBitsSet {
      plane: PnFramePlane::Uv,
      value: 0x03FF,
      low_bits: 6,
      ..
    }
  ));
}
