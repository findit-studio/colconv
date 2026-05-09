use super::super::{Y2xxFrame, Y2xxFrameError, Y210Frame, Y212Frame, Y216Frame};

// Build a `Vec<u16>` whose in-memory byte layout is the LE-encoded byte
// representation of `intended` (i.e., what FFmpeg would emit on the wire).
// On an LE host the result equals `intended` element-wise; on a BE host
// every element is byte-swapped relative to `intended`. Matches the
// `le_encoded_u16_buf` helpers in the high-bit subsampled-frame test files.
fn le_encoded_u16_buf(intended: &[u16]) -> std::vec::Vec<u16> {
  let bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect()
}

#[test]
fn y210_frame_try_new_accepts_valid_tight() {
  // Width 4, height 2, stride = 4 * 2 = 8 u16 elements per row.
  let buf = std::vec![0u16; 8 * 2];
  let frame = Y210Frame::try_new(&buf, 4, 2, 8).unwrap();
  assert_eq!(frame.width(), 4);
  assert_eq!(frame.height(), 2);
  assert_eq!(frame.stride(), 8);
}

#[test]
fn y210_frame_try_new_accepts_oversized_stride() {
  // Padded-row case: caller may supply a larger stride.
  let buf = std::vec![0u16; 16 * 4];
  Y210Frame::try_new(&buf, 4, 4, 16).unwrap();
}

#[test]
fn y210_frame_try_new_rejects_zero_dimension() {
  let buf: [u16; 0] = [];
  // Frame structs don't derive `PartialEq` (matching V210Frame), so
  // we extract the error before comparing.
  let err = Y210Frame::try_new(&buf, 0, 1, 0).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::ZeroDimension {
      width: 0,
      height: 1
    }
  );
  let err = Y210Frame::try_new(&buf, 4, 0, 8).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::ZeroDimension {
      width: 4,
      height: 0
    }
  );
}

#[test]
fn y210_frame_try_new_rejects_odd_width() {
  let buf = std::vec![0u16; 64];
  for w in [1u32, 3, 5, 7, 9, 11, 13] {
    let stride = (w as usize) * 2;
    let err = Y210Frame::try_new(&buf, w, 1, stride as u32).unwrap_err();
    assert_eq!(err, Y2xxFrameError::OddWidth { width: w });
  }
  // 2, 4, 6, 8 must succeed.
  for w in [2u32, 4, 6, 8] {
    let stride = w * 2;
    let buf = std::vec![0u16; stride as usize];
    Y210Frame::try_new(&buf, w, 1, stride).unwrap();
  }
}

#[test]
fn y210_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u16; 16];
  // For width=4, min_stride = 8 u16 elements.
  let err = Y210Frame::try_new(&buf, 4, 1, 7).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::StrideTooSmall {
      min_stride: 8,
      stride: 7
    }
  );
}

#[test]
fn y210_frame_try_new_rejects_short_plane() {
  let buf = std::vec![0u16; 7]; // need 8 for width=4 height=1
  let err = Y210Frame::try_new(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::PlaneTooShort {
      expected: 8,
      actual: 7
    }
  );
}

#[test]
fn y210_frame_accessors_round_trip() {
  let buf = std::vec![0u16; 16 * 4];
  let frame = Y210Frame::try_new(&buf, 8, 4, 16).unwrap();
  assert_eq!(frame.packed().len(), 16 * 4);
  assert_eq!(frame.width(), 8);
  assert_eq!(frame.height(), 4);
  assert_eq!(frame.stride(), 16);
}

#[test]
fn y2xx_frame_try_new_rejects_unsupported_bits() {
  // BITS must be {10, 12, 16}. The compile-time-asserted dimensions
  // 8 are valid but BITS=11 is not.
  let buf = std::vec![0u16; 16];
  let err = Y2xxFrame::<11>::try_new(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::UnsupportedBits { bits: 11 });
  let err = Y2xxFrame::<8>::try_new(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::UnsupportedBits { bits: 8 });
  // 14 is NOT in the supported set for Y2xx (no FFmpeg y214 format exists).
  let err = Y2xxFrame::<14>::try_new(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::UnsupportedBits { bits: 14 });
}

#[test]
fn y210_frame_try_new_checked_rejects_low_bit_violations() {
  // Y210 = MSB-aligned 10-bit; low 6 bits must be zero.
  //
  // `try_new_checked` applies `u16::from_le` before the low-bits check,
  // so feed LE-encoded byte storage. Without the wrap the test still
  // rejects on a BE host but for the wrong reason — the host-native
  // `0xFFC0` decodes to `0xC0FF` (low 6 bits = `0x3F`) and triggers
  // rejection on `buf[0]`, which the test claims is valid.
  let mut intended = std::vec![0u16; 8]; // width=4, height=1
  intended[0] = 0xFFC0; // valid: 10-bit value 0x3FF in high 10
  intended[1] = 0xFFC1; // INVALID: low 6 bits = 0x01 (non-zero)
  let buf = le_encoded_u16_buf(&intended);
  let err = Y210Frame::try_new_checked(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::SampleLowBitsSet);
}

/// Host-independent: builds the plane explicitly from LE-encoded bytes via
/// `to_le_bytes` then reinterprets as `&[u16]` via `from_ne_bytes`. The
/// validator's `from_le` normalization recovers the intended MSB-aligned
/// values on both LE and BE hosts.
#[test]
fn y210_frame_try_new_checked_accepts_valid_msb_aligned_data() {
  // All samples have low 6 bits == 0.
  let intended: std::vec::Vec<u16> = (0..8).map(|i| ((i as u16) << 6) & 0xFFC0).collect();
  let buf: std::vec::Vec<u16> = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect();
  Y210Frame::try_new_checked(&buf, 4, 1, 8).unwrap();
}

/// Host-independent regression for [`Y2xxFrame::try_new_checked`]'s LE-encoded
/// byte contract. Builds the plane explicitly from LE-encoded bytes via
/// `to_le_bytes` then reinterprets as `&[u16]` via `from_ne_bytes`. The
/// validator must accept this on both LE and BE hosts: on LE host
/// `from_le` is a no-op (host-native already matches); on BE host
/// `from_le` byte-swaps each sample back into host-native form before
/// the bit-check, recovering the intended MSB-aligned value.
///
/// Without the LE-aware bit-check, this test would reject every sample
/// on a BE host (the byte-swapped storage has the active bits in the
/// low byte, which fails the low-`(16 - BITS)`-bits-zero check).
#[test]
fn y210_frame_try_new_checked_accepts_le_encoded_buffer() {
  // Intended values: 10-bit MSB-aligned `(i << 6) & 0xFFC0` for i in 0..8.
  let intended: std::vec::Vec<u16> = (0..8u16).map(|i| (i << 6) & 0xFFC0).collect();
  let le_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  let buf: std::vec::Vec<u16> = le_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  Y210Frame::try_new_checked(&buf, 4, 1, 8).unwrap();
}

/// Host-independent BE-host regression: a *BE-encoded* buffer of valid
/// MSB-aligned values must be rejected when fed to a Y2xx frame
/// (which assumes the LE-encoded byte contract). Pick a logical value
/// whose BE-byte form, when re-interpreted as LE, has non-zero low bits.
///
/// Logical value `0xFFC0` BE-encoded = `[0xFF, 0xC0]`. Re-interpreted
/// via `from_le_bytes([0xFF, 0xC0])` = `0xC0FF`, whose low 6 bits =
/// `0x3F` (non-zero) → `SampleLowBitsSet`.
#[test]
fn y210_frame_try_new_checked_rejects_be_encoded_buffer_with_low_bits() {
  // Use the same `0xFFC0` value across the row so we get a
  // deterministic rejection regardless of host.
  let intended: std::vec::Vec<u16> = std::vec![0xFFC0u16; 8];
  let be_bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
  let buf: std::vec::Vec<u16> = be_bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect();
  let err = Y210Frame::try_new_checked(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::SampleLowBitsSet);
}

#[test]
#[should_panic(expected = "invalid Y2xxFrame:")]
fn y210_frame_new_panics_on_invalid() {
  let buf: [u16; 0] = [];
  let _ = Y210Frame::new(&buf, 0, 0, 0);
}

/// Host-independent: declared-payload samples (low 6 bits == 0) are LE-encoded
/// so the validator's `from_le` recovers them on both hosts; padding bytes
/// stay outside the declared payload window so they're never scanned.
#[test]
fn y210_frame_try_new_checked_ignores_stride_padding_bytes() {
  // Width=4 → row_elems = 8 u16; stride = 12 u16 (4 u16 padding per row).
  // All declared-payload samples have low 6 bits == 0 (valid 10-bit MSB-aligned).
  // Padding samples have arbitrary low bits set — must not trigger
  // SampleLowBitsSet (matches PnFrame::try_new_checked behavior).
  let mut intended = std::vec![0u16; 12 * 2]; // height=2
  for row in 0..2 {
    // Declared payload (first 8 u16 of each row) — clean MSB-aligned.
    for i in 0..8 {
      intended[row * 12 + i] = ((i as u16) << 6) & 0xFFC0;
    }
    // Stride padding (last 4 u16 of each row) — arbitrary low bits.
    for i in 8..12 {
      intended[row * 12 + i] = 0xFFFF; // every bit set, including low 6
    }
  }
  let buf: std::vec::Vec<u16> = intended
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect();
  // try_new_checked must accept this — it scans only the declared payload.
  Y210Frame::try_new_checked(&buf, 4, 2, 12).unwrap();
}

#[test]
fn y212_frame_try_new_accepts_valid_tight() {
  let buf = std::vec![0u16; 8 * 2];
  let frame = Y212Frame::try_new(&buf, 4, 2, 8).unwrap();
  assert_eq!(frame.width(), 4);
  assert_eq!(frame.height(), 2);
}

#[test]
fn y212_frame_try_new_checked_rejects_low_bit_violations() {
  // Y212 = MSB-aligned 12-bit; low 4 bits must be zero.
  //
  // LE-encoded byte storage so the test asserts the same logical values
  // on every host (see `y210_frame_try_new_checked_rejects_low_bit_violations`
  // for the BE-host failure mode).
  let mut intended = std::vec![0u16; 8]; // width=4, height=1
  intended[0] = 0xFFF0; // valid: 12-bit value 0xFFF in high 12, low 4 = 0
  intended[1] = 0xFFF1; // INVALID: low 4 bits = 0x1
  let buf = le_encoded_u16_buf(&intended);
  let err = Y212Frame::try_new_checked(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(err, Y2xxFrameError::SampleLowBitsSet);
}

// ── Y216 tests ────────────────────────────────────────────────────────────────

#[test]
fn y216_frame_try_new_accepts_valid_tight() {
  // Width 4, height 2, stride = 4 * 2 = 8 u16 elements per row.
  let buf = std::vec![0xFFFFu16; 8 * 2];
  let frame = Y216Frame::try_new(&buf, 4, 2, 8).unwrap();
  assert_eq!(frame.width(), 4);
  assert_eq!(frame.height(), 2);
  assert_eq!(frame.stride(), 8);
}

#[test]
fn y216_frame_try_new_accepts_oversized_stride() {
  // Padded-row case: caller may supply a larger stride.
  let buf = std::vec![0u16; 16 * 4];
  Y216Frame::try_new(&buf, 4, 4, 16).unwrap();
}

#[test]
fn y216_frame_try_new_rejects_zero_dimension() {
  let buf: [u16; 0] = [];
  let err = Y216Frame::try_new(&buf, 0, 1, 0).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::ZeroDimension {
      width: 0,
      height: 1
    }
  );
  let err = Y216Frame::try_new(&buf, 4, 0, 8).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::ZeroDimension {
      width: 4,
      height: 0
    }
  );
}

#[test]
fn y216_frame_try_new_rejects_odd_width() {
  let buf = std::vec![0u16; 64];
  for w in [1u32, 3, 5, 7, 9, 11, 13] {
    let stride = (w as usize) * 2;
    let err = Y216Frame::try_new(&buf, w, 1, stride as u32).unwrap_err();
    assert_eq!(err, Y2xxFrameError::OddWidth { width: w });
  }
  // Even widths must succeed.
  for w in [2u32, 4, 6, 8] {
    let stride = w * 2;
    let buf = std::vec![0u16; stride as usize];
    Y216Frame::try_new(&buf, w, 1, stride).unwrap();
  }
}

#[test]
fn y216_frame_try_new_rejects_stride_too_small() {
  let buf = std::vec![0u16; 16];
  // For width=4, min_stride = 8 u16 elements.
  let err = Y216Frame::try_new(&buf, 4, 1, 7).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::StrideTooSmall {
      min_stride: 8,
      stride: 7
    }
  );
}

#[test]
fn y216_frame_try_new_rejects_short_plane() {
  let buf = std::vec![0u16; 7]; // need 8 for width=4, height=1
  let err = Y216Frame::try_new(&buf, 4, 1, 8).unwrap_err();
  assert_eq!(
    err,
    Y2xxFrameError::PlaneTooShort {
      expected: 8,
      actual: 7
    }
  );
}

#[test]
fn y216_frame_accessors_round_trip() {
  let buf = std::vec![0xFFFFu16; 16 * 4];
  let frame = Y216Frame::try_new(&buf, 8, 4, 16).unwrap();
  assert_eq!(frame.packed().len(), 16 * 4);
  assert_eq!(frame.width(), 8);
  assert_eq!(frame.height(), 4);
  assert_eq!(frame.stride(), 16);
}

#[test]
fn y216_frame_try_new_checked_accepts_arbitrary_low_bits() {
  // Y216 = full 16-bit range; all bits are active, so any sample value
  // is valid. try_new_checked must succeed even when every bit is set.
  let buf = std::vec![0xFFFFu16; 8]; // width=4, height=1, stride=8
  Y216Frame::try_new_checked(&buf, 4, 1, 8).unwrap();
  // Also verify with alternating patterns to rule out accidental masking.
  let buf: std::vec::Vec<u16> = (0..8u16).map(|i| 0x0001 + i).collect();
  Y216Frame::try_new_checked(&buf, 4, 1, 8).unwrap();
}

#[test]
fn y216_frame_try_new_checked_accepts_valid_tight() {
  // try_new_checked at BITS=16 is identical to try_new — no low-bit scan.
  let buf = std::vec![0u16; 8 * 2];
  let frame = Y216Frame::try_new_checked(&buf, 4, 2, 8).unwrap();
  assert_eq!(frame.width(), 4);
  assert_eq!(frame.height(), 2);
}

#[test]
#[should_panic(expected = "invalid Y2xxFrame:")]
fn y216_frame_new_panics_on_invalid() {
  let buf: [u16; 0] = [];
  let _ = Y216Frame::new(&buf, 0, 0, 0);
}
