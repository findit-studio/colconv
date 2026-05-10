use super::*;

// ---- YUVA high-bit BE-host regression tests ---------------------------
//
// `Yuva420pFrame16` / `Yuva422pFrame16` / `Yuva444pFrame16` document a
// **LE-encoded byte-layout** contract on their `&[u16]` planes (the
// FFmpeg `*LE` byte buffer reinterpreted as `u16`). The
// `try_new_checked` validators must therefore normalize each sample
// with `u16::from_le` before comparing against `max_valid`; otherwise
// a valid LE-encoded plane on a BE host has every `u16` byte-swapped
// relative to the intended logical value and the validator falsely
// rejects every row.
//
// These tests build the planes explicitly from LE-encoded bytes via
// `to_le_bytes` and read back as `&[u16]` via `from_ne_bytes`. On an
// LE host the resulting `u16` values are identical to the intended
// literals; on a BE host every `u16` is byte-swapped relative to the
// intent, exercising the `u16::from_le` normalization inside the
// validators. Without that normalization the validators would falsely
// reject every valid LE-encoded plane on a BE host.
//
// Each family covers (1) a positive case — a logical LE buffer of
// valid samples (including the alpha plane) that must be accepted on
// both LE and BE hosts — and (2) a negative case where a sample is
// invalid even after `from_le` normalization, ensuring the validator
// still surfaces real errors. Negative cases place the bad sample on
// the alpha plane to give that plane its own dedicated coverage.

/// Build a `Vec<u16>` representing the LE-encoded byte layout of
/// `intended` (i.e., what FFmpeg would emit on the wire). On an LE
/// host the result equals `intended` element-wise; on a BE host every
/// element is byte-swapped relative to `intended`.
fn le_encoded_u16_buf(intended: &[u16]) -> std::vec::Vec<u16> {
  let bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect()
}

// ---- Yuva420p10 -------------------------------------------------------

#[test]
fn yuva420p10_try_new_checked_accepts_le_encoded_buffer_on_any_host() {
  // 10-bit-low-packed white = 1023 (LE bytes [0xFF, 0x03]). The alpha
  // plane is full-width × full-height; chroma is half × half.
  let intended_y = std::vec![1023u16; 16 * 8];
  let intended_uv = std::vec![512u16; 8 * 4];
  let intended_a = std::vec![1023u16; 16 * 8];
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_uv);
  let v = le_encoded_u16_buf(&intended_uv);
  let a = le_encoded_u16_buf(&intended_a);
  Yuva420p10Frame::try_new_checked(&y, &u, &v, &a, 16, 8, 16, 8, 8, 16)
    .expect("LE-encoded valid yuva420p10le must be accepted on both LE and BE hosts");
}

#[test]
fn yuva420p10_try_new_checked_rejects_le_encoded_alpha_out_of_range_on_any_host() {
  // After `u16::from_le` normalization the offending alpha sample is
  // 1024 (just above the 10-bit max of 1023). On both LE and BE hosts
  // the validator must catch this — the LE-encoded byte buffer carries
  // the logical value 1024 in `a[3 * 16 + 5]`.
  let intended_y = std::vec![0u16; 16 * 8];
  let intended_uv = std::vec![512u16; 8 * 4];
  let mut intended_a = std::vec![1023u16; 16 * 8];
  intended_a[3 * 16 + 5] = 1024;
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_uv);
  let v = le_encoded_u16_buf(&intended_uv);
  let a = le_encoded_u16_buf(&intended_a);
  let e = Yuva420p10Frame::try_new_checked(&y, &u, &v, &a, 16, 8, 16, 8, 8, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuva420pFrame16Error::SampleOutOfRange {
      plane: Yuva420pFrame16Plane::A,
      value: 1024,
      max_valid: 1023,
      ..
    }
  ));
}

// ---- Yuva422p10 -------------------------------------------------------

#[test]
fn yuva422p10_try_new_checked_accepts_le_encoded_buffer_on_any_host() {
  // 4:2:2 geometry: Y/A are full-width × full-height; U/V are
  // half-width × full-height. 10-bit white = 1023 (LE bytes
  // [0xFF, 0x03]).
  let intended_y = std::vec![1023u16; 16 * 8];
  let intended_uv = std::vec![512u16; 8 * 8];
  let intended_a = std::vec![1023u16; 16 * 8];
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_uv);
  let v = le_encoded_u16_buf(&intended_uv);
  let a = le_encoded_u16_buf(&intended_a);
  Yuva422p10Frame::try_new_checked(&y, &u, &v, &a, 16, 8, 16, 8, 8, 16)
    .expect("LE-encoded valid yuva422p10le must be accepted on both LE and BE hosts");
}

#[test]
fn yuva422p10_try_new_checked_rejects_le_encoded_alpha_out_of_range_on_any_host() {
  // Plant an out-of-range logical alpha sample (1024 > 10-bit max
  // 1023) in the LE byte buffer. The validator must surface the
  // normalized logical value on both LE and BE hosts.
  let intended_y = std::vec![0u16; 16 * 8];
  let intended_uv = std::vec![512u16; 8 * 8];
  let mut intended_a = std::vec![1023u16; 16 * 8];
  intended_a[2 * 16 + 7] = 1024;
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_uv);
  let v = le_encoded_u16_buf(&intended_uv);
  let a = le_encoded_u16_buf(&intended_a);
  let e = Yuva422p10Frame::try_new_checked(&y, &u, &v, &a, 16, 8, 16, 8, 8, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuva422pFrame16Error::SampleOutOfRange {
      plane: Yuva422pFrame16Plane::A,
      value: 1024,
      max_valid: 1023,
      ..
    }
  ));
}

// ---- Yuva444p10 -------------------------------------------------------

#[test]
fn yuva444p10_try_new_checked_accepts_le_encoded_buffer_on_any_host() {
  // 4:4:4 geometry: every plane (Y, U, V, A) is full-width ×
  // full-height. 10-bit white = 1023 (LE bytes [0xFF, 0x03]).
  let intended_full = std::vec![1023u16; 16 * 8];
  let intended_chroma = std::vec![512u16; 16 * 8];
  let y = le_encoded_u16_buf(&intended_full);
  let u = le_encoded_u16_buf(&intended_chroma);
  let v = le_encoded_u16_buf(&intended_chroma);
  let a = le_encoded_u16_buf(&intended_full);
  Yuva444p10Frame::try_new_checked(&y, &u, &v, &a, 16, 8, 16, 16, 16, 16)
    .expect("LE-encoded valid yuva444p10le must be accepted on both LE and BE hosts");
}

#[test]
fn yuva444p10_try_new_checked_rejects_le_encoded_alpha_out_of_range_on_any_host() {
  // Plant an out-of-range logical alpha sample (1024 > 10-bit max
  // 1023). The validator must catch this regardless of host endianness.
  let intended_y = std::vec![0u16; 16 * 8];
  let intended_chroma = std::vec![512u16; 16 * 8];
  let mut intended_a = std::vec![1023u16; 16 * 8];
  intended_a[4 * 16 + 9] = 1024;
  let y = le_encoded_u16_buf(&intended_y);
  let u = le_encoded_u16_buf(&intended_chroma);
  let v = le_encoded_u16_buf(&intended_chroma);
  let a = le_encoded_u16_buf(&intended_a);
  let e = Yuva444p10Frame::try_new_checked(&y, &u, &v, &a, 16, 8, 16, 16, 16, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuva444pFrame16Error::SampleOutOfRange {
      plane: Yuva444pFrame16Plane::A,
      value: 1024,
      max_valid: 1023,
      ..
    }
  ));
}

// ---- BE checked-constructor regressions -------------------------------
//
// `try_new_checked` on `*BeFrame` (i.e. `<const BE = true>`) MUST
// normalize via `u16::from_be` before the range check. See
// `subsampled_4_2_0_high_bit::p010_be_try_new_checked_*` tests for
// the full rationale.

/// Build a `Vec<u16>` representing the BE-encoded byte layout of
/// `intended` (i.e. what FFmpeg would emit on the wire for `*BE`).
fn be_encoded_u16_buf(intended: &[u16]) -> std::vec::Vec<u16> {
  let bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
  bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect()
}

#[test]
fn yuva420p10_be_try_new_checked_accepts_be_encoded_buffer_on_any_host() {
  let intended_y = std::vec![1023u16; 16 * 8];
  let intended_uv = std::vec![512u16; 8 * 4];
  let intended_a = std::vec![1023u16; 16 * 8];
  let y = be_encoded_u16_buf(&intended_y);
  let u = be_encoded_u16_buf(&intended_uv);
  let v = be_encoded_u16_buf(&intended_uv);
  let a = be_encoded_u16_buf(&intended_a);
  Yuva420p10BeFrame::try_new_checked(&y, &u, &v, &a, 16, 8, 16, 8, 8, 16)
    .expect("BE-encoded valid yuva420p10be must be accepted on both LE and BE hosts");
}

#[test]
fn yuva420p10_be_try_new_checked_rejects_be_encoded_alpha_out_of_range() {
  // Logical 1024 (just above 10-bit max) on the alpha plane —
  // BE-encoded — must be rejected on every host.
  let intended_y = std::vec![1023u16; 16 * 8];
  let intended_uv = std::vec![512u16; 8 * 4];
  let mut intended_a = std::vec![1023u16; 16 * 8];
  intended_a[3 * 16 + 5] = 1024;
  let y = be_encoded_u16_buf(&intended_y);
  let u = be_encoded_u16_buf(&intended_uv);
  let v = be_encoded_u16_buf(&intended_uv);
  let a = be_encoded_u16_buf(&intended_a);
  let e = Yuva420p10BeFrame::try_new_checked(&y, &u, &v, &a, 16, 8, 16, 8, 8, 16).unwrap_err();
  assert!(matches!(
    e,
    Yuva420pFrame16Error::SampleOutOfRange {
      plane: Yuva420pFrame16Plane::A,
      value: 1024,
      max_valid: 1023,
      ..
    }
  ));
}
