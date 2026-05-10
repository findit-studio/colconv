//! Shared test helpers for high-bit frame validators.
//!
//! These helpers build `Vec<u16>` planes whose in-memory byte layout
//! matches the wire-encoded representation FFmpeg would emit. On an LE
//! host the LE-encoded buffer equals `intended` element-wise; on a BE
//! host every `u16` is byte-swapped relative to `intended`. The BE
//! variant is the mirror image. Frame validators must `from_le` /
//! `from_be`-normalize before any range check on both hosts — see the
//! comment at the bottom of `subsampled_4_2_0_high_bit.rs` for the
//! full rationale.

/// Build a `Vec<u16>` representing the LE-encoded byte layout of
/// `intended` (i.e., what FFmpeg would emit on the wire for `*LE`
/// formats).
pub(super) fn le_encoded_u16_buf(intended: &[u16]) -> std::vec::Vec<u16> {
  let bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_le_bytes()).collect();
  bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect()
}

/// Build a `Vec<u16>` representing the BE-encoded byte layout of
/// `intended` (i.e., what FFmpeg would emit on the wire for `*BE`
/// formats). Mirror of [`le_encoded_u16_buf`].
pub(super) fn be_encoded_u16_buf(intended: &[u16]) -> std::vec::Vec<u16> {
  let bytes: std::vec::Vec<u8> = intended.iter().flat_map(|v| v.to_be_bytes()).collect();
  bytes
    .chunks_exact(2)
    .map(|b| u16::from_ne_bytes([b[0], b[1]]))
    .collect()
}
