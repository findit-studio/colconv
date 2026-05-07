// Dispatcher (Task 3) and SIMD backends (Tasks 4-8) will consume these via the
// `crate::row::scalar::legacy_rgb` module path; suppress dead-code lint until then.
#![allow(dead_code)]
//! Scalar reference kernels for legacy 16-bit packed-RGB source formats
//! (Tier 7 — `AV_PIX_FMT_RGB565LE`, `AV_PIX_FMT_RGB555LE`,
//! `AV_PIX_FMT_RGB444LE` and their BGR companions).
//!
//! # Bit extraction
//!
//! Each source pixel is a little-endian `u16` word. The kernel reads two
//! consecutive bytes as `u16::from_le_bytes([src[2*x], src[2*x+1]])` and
//! extracts sub-fields with bit-shift + AND-mask. Mask constants are defined
//! once per format family at the top of each function group.
//!
//! # Channel expansion to u8
//!
//! | Bits | Formula               | Maps 0→0, max→255       |
//! |------|-----------------------|-------------------------|
//! | 5    | `(c << 3) \| (c >> 2)` | 0→0, 31→255            |
//! | 6    | `(c << 2) \| (c >> 4)` | 0→0, 63→255            |
//! | 4    | `(c << 4) \| c`        | 0→0, 15→255            |
//!
//! # u16 output
//!
//! `*_to_rgb_u16_row` / `*_to_rgba_u16_row` return channels low-bit aligned
//! in `u16` at native bit width — no expansion applied. Max values: R5=31,
//! G6=63, B5=31 (RGB565); R5=G5=B5=31 (RGB555); R4=G4=B4=15 (RGB444).
//!
//! # BGR variants
//!
//! BGR sources swap R↔C0 and B↔C2 in the extraction step; output byte order
//! is always R, G, B regardless of source order.

// ---- RGB565 (R5 G6 B5, bits [15:11] [10:5] [4:0]) ----------------------

/// Converts one row of packed RGB565 pixels to packed `R, G, B` bytes.
///
/// Channels are expanded to full u8 via bit-replication:
/// R5 → `(r5 << 3) | (r5 >> 2)`, G6 → `(g6 << 2) | (g6 >> 4)`,
/// B5 → `(b5 << 3) | (b5 >> 2)`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r5 = ((px >> 11) & 0x1F) as u8;
    let g6 = ((px >> 5) & 0x3F) as u8;
    let b5 = (px & 0x1F) as u8;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g6 << 2) | (g6 >> 4);
    let b = (b5 << 3) | (b5 >> 2);
    let dst = x * 3;
    rgb_out[dst] = r;
    rgb_out[dst + 1] = g;
    rgb_out[dst + 2] = b;
  }
}

/// Converts one row of packed RGB565 pixels to packed `R, G, B, A` bytes.
///
/// Channels are expanded to full u8 (same as `rgb565_to_rgb_row`); alpha is
/// forced to `0xFF` (no source alpha).
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r5 = ((px >> 11) & 0x1F) as u8;
    let g6 = ((px >> 5) & 0x3F) as u8;
    let b5 = (px & 0x1F) as u8;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g6 << 2) | (g6 >> 4);
    let b = (b5 << 3) | (b5 >> 2);
    let dst = x * 4;
    rgba_out[dst] = r;
    rgba_out[dst + 1] = g;
    rgba_out[dst + 2] = b;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Converts one row of packed RGB565 pixels to packed `R, G, B` u16 samples.
///
/// Channels are returned at native bit width, low-bit aligned — no expansion
/// applied. Output ranges: R ∈ [0, 31], G ∈ [0, 63], B ∈ [0, 31].
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_u16_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r = (px >> 11) & 0x1F;
    let g = (px >> 5) & 0x3F;
    let b = px & 0x1F;
    let dst = x * 3;
    rgb_u16_out[dst] = r;
    rgb_u16_out[dst + 1] = g;
    rgb_u16_out[dst + 2] = b;
  }
}

/// Converts one row of packed RGB565 pixels to packed `R, G, B, A` u16 samples.
///
/// Channels are returned at native bit width, low-bit aligned — no expansion
/// applied. Alpha is forced to `0xFFFF`. Output ranges: R ∈ [0, 31],
/// G ∈ [0, 63], B ∈ [0, 31], A = 65535.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_u16_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r = (px >> 11) & 0x1F;
    let g = (px >> 5) & 0x3F;
    let b = px & 0x1F;
    let dst = x * 4;
    rgba_u16_out[dst] = r;
    rgba_u16_out[dst + 1] = g;
    rgba_u16_out[dst + 2] = b;
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- BGR565 (B5 G6 R5, bits [15:11]=B [10:5]=G [4:0]=R) ----------------

/// Converts one row of packed BGR565 pixels to packed `R, G, B` bytes.
///
/// BGR565 stores B in bits [15:11], G in [10:5], R in [4:0]. Output byte
/// order is always R, G, B. Channels are expanded to full u8 via
/// bit-replication: R5/B5 → `(c << 3) | (c >> 2)`, G6 → `(c << 2) | (c >> 4)`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr565_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b5 = ((px >> 11) & 0x1F) as u8;
    let g6 = ((px >> 5) & 0x3F) as u8;
    let r5 = (px & 0x1F) as u8;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g6 << 2) | (g6 >> 4);
    let b = (b5 << 3) | (b5 >> 2);
    let dst = x * 3;
    rgb_out[dst] = r;
    rgb_out[dst + 1] = g;
    rgb_out[dst + 2] = b;
  }
}

/// Converts one row of packed BGR565 pixels to packed `R, G, B, A` bytes.
///
/// Same as `bgr565_to_rgb_row`; alpha is forced to `0xFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr565_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b5 = ((px >> 11) & 0x1F) as u8;
    let g6 = ((px >> 5) & 0x3F) as u8;
    let r5 = (px & 0x1F) as u8;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g6 << 2) | (g6 >> 4);
    let b = (b5 << 3) | (b5 >> 2);
    let dst = x * 4;
    rgba_out[dst] = r;
    rgba_out[dst + 1] = g;
    rgba_out[dst + 2] = b;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Converts one row of packed BGR565 pixels to packed `R, G, B` u16 samples.
///
/// Channels are returned at native bit width, low-bit aligned, in R, G, B
/// output order. Output ranges: R ∈ [0, 31], G ∈ [0, 63], B ∈ [0, 31].
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_u16_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr565_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b = (px >> 11) & 0x1F;
    let g = (px >> 5) & 0x3F;
    let r = px & 0x1F;
    let dst = x * 3;
    rgb_u16_out[dst] = r;
    rgb_u16_out[dst + 1] = g;
    rgb_u16_out[dst + 2] = b;
  }
}

/// Converts one row of packed BGR565 pixels to packed `R, G, B, A` u16 samples.
///
/// Channels at native bit width in R, G, B output order. Alpha forced to `0xFFFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_u16_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr565_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b = (px >> 11) & 0x1F;
    let g = (px >> 5) & 0x3F;
    let r = px & 0x1F;
    let dst = x * 4;
    rgba_u16_out[dst] = r;
    rgba_u16_out[dst + 1] = g;
    rgba_u16_out[dst + 2] = b;
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- RGB555 (1X R5 G5 B5, bits [14:10] [9:5] [4:0]) --------------------

/// Converts one row of packed RGB555 pixels to packed `R, G, B` bytes.
///
/// Bit 15 is unused padding (ignored). Channels are expanded to full u8
/// via bit-replication: R5/G5/B5 → `(c << 3) | (c >> 2)`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r5 = ((px >> 10) & 0x1F) as u8;
    let g5 = ((px >> 5) & 0x1F) as u8;
    let b5 = (px & 0x1F) as u8;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g5 << 3) | (g5 >> 2);
    let b = (b5 << 3) | (b5 >> 2);
    let dst = x * 3;
    rgb_out[dst] = r;
    rgb_out[dst + 1] = g;
    rgb_out[dst + 2] = b;
  }
}

/// Converts one row of packed RGB555 pixels to packed `R, G, B, A` bytes.
///
/// Same as `rgb555_to_rgb_row`; alpha is forced to `0xFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r5 = ((px >> 10) & 0x1F) as u8;
    let g5 = ((px >> 5) & 0x1F) as u8;
    let b5 = (px & 0x1F) as u8;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g5 << 3) | (g5 >> 2);
    let b = (b5 << 3) | (b5 >> 2);
    let dst = x * 4;
    rgba_out[dst] = r;
    rgba_out[dst + 1] = g;
    rgba_out[dst + 2] = b;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Converts one row of packed RGB555 pixels to packed `R, G, B` u16 samples.
///
/// Channels at native bit width, low-bit aligned — no expansion applied.
/// Output ranges: R ∈ [0, 31], G ∈ [0, 31], B ∈ [0, 31].
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_u16_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r = (px >> 10) & 0x1F;
    let g = (px >> 5) & 0x1F;
    let b = px & 0x1F;
    let dst = x * 3;
    rgb_u16_out[dst] = r;
    rgb_u16_out[dst + 1] = g;
    rgb_u16_out[dst + 2] = b;
  }
}

/// Converts one row of packed RGB555 pixels to packed `R, G, B, A` u16 samples.
///
/// Channels at native bit width, low-bit aligned. Alpha forced to `0xFFFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_u16_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r = (px >> 10) & 0x1F;
    let g = (px >> 5) & 0x1F;
    let b = px & 0x1F;
    let dst = x * 4;
    rgba_u16_out[dst] = r;
    rgba_u16_out[dst + 1] = g;
    rgba_u16_out[dst + 2] = b;
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- BGR555 (1X B5 G5 R5, bits [14:10]=B [9:5]=G [4:0]=R) --------------

/// Converts one row of packed BGR555 pixels to packed `R, G, B` bytes.
///
/// BGR555 stores B in bits [14:10], G in [9:5], R in [4:0] (bit 15 padding).
/// Output byte order is always R, G, B. All channels are 5-bit, expanded via
/// `(c << 3) | (c >> 2)`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr555_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b5 = ((px >> 10) & 0x1F) as u8;
    let g5 = ((px >> 5) & 0x1F) as u8;
    let r5 = (px & 0x1F) as u8;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g5 << 3) | (g5 >> 2);
    let b = (b5 << 3) | (b5 >> 2);
    let dst = x * 3;
    rgb_out[dst] = r;
    rgb_out[dst + 1] = g;
    rgb_out[dst + 2] = b;
  }
}

/// Converts one row of packed BGR555 pixels to packed `R, G, B, A` bytes.
///
/// Same as `bgr555_to_rgb_row`; alpha is forced to `0xFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr555_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b5 = ((px >> 10) & 0x1F) as u8;
    let g5 = ((px >> 5) & 0x1F) as u8;
    let r5 = (px & 0x1F) as u8;
    let r = (r5 << 3) | (r5 >> 2);
    let g = (g5 << 3) | (g5 >> 2);
    let b = (b5 << 3) | (b5 >> 2);
    let dst = x * 4;
    rgba_out[dst] = r;
    rgba_out[dst + 1] = g;
    rgba_out[dst + 2] = b;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Converts one row of packed BGR555 pixels to packed `R, G, B` u16 samples.
///
/// Channels at native bit width in R, G, B output order — no expansion applied.
/// Output ranges: R ∈ [0, 31], G ∈ [0, 31], B ∈ [0, 31].
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_u16_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr555_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b = (px >> 10) & 0x1F;
    let g = (px >> 5) & 0x1F;
    let r = px & 0x1F;
    let dst = x * 3;
    rgb_u16_out[dst] = r;
    rgb_u16_out[dst + 1] = g;
    rgb_u16_out[dst + 2] = b;
  }
}

/// Converts one row of packed BGR555 pixels to packed `R, G, B, A` u16 samples.
///
/// Channels at native bit width in R, G, B output order. Alpha forced to `0xFFFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_u16_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr555_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b = (px >> 10) & 0x1F;
    let g = (px >> 5) & 0x1F;
    let r = px & 0x1F;
    let dst = x * 4;
    rgba_u16_out[dst] = r;
    rgba_u16_out[dst + 1] = g;
    rgba_u16_out[dst + 2] = b;
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- RGB444 (4X R4 G4 B4, bits [11:8] [7:4] [3:0]) ---------------------

/// Converts one row of packed RGB444 pixels to packed `R, G, B` bytes.
///
/// Bits [15:12] are unused padding (ignored). Channels are expanded to full
/// u8 via nibble-replication: R4/G4/B4 → `(c << 4) | c` (equivalent to
/// `c * 17`).
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r4 = ((px >> 8) & 0x0F) as u8;
    let g4 = ((px >> 4) & 0x0F) as u8;
    let b4 = (px & 0x0F) as u8;
    let r = (r4 << 4) | r4;
    let g = (g4 << 4) | g4;
    let b = (b4 << 4) | b4;
    let dst = x * 3;
    rgb_out[dst] = r;
    rgb_out[dst + 1] = g;
    rgb_out[dst + 2] = b;
  }
}

/// Converts one row of packed RGB444 pixels to packed `R, G, B, A` bytes.
///
/// Same as `rgb444_to_rgb_row`; alpha is forced to `0xFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r4 = ((px >> 8) & 0x0F) as u8;
    let g4 = ((px >> 4) & 0x0F) as u8;
    let b4 = (px & 0x0F) as u8;
    let r = (r4 << 4) | r4;
    let g = (g4 << 4) | g4;
    let b = (b4 << 4) | b4;
    let dst = x * 4;
    rgba_out[dst] = r;
    rgba_out[dst + 1] = g;
    rgba_out[dst + 2] = b;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Converts one row of packed RGB444 pixels to packed `R, G, B` u16 samples.
///
/// Channels at native 4-bit width, low-bit aligned — no expansion applied.
/// Output ranges: R ∈ [0, 15], G ∈ [0, 15], B ∈ [0, 15].
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_u16_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r = (px >> 8) & 0x0F;
    let g = (px >> 4) & 0x0F;
    let b = px & 0x0F;
    let dst = x * 3;
    rgb_u16_out[dst] = r;
    rgb_u16_out[dst + 1] = g;
    rgb_u16_out[dst + 2] = b;
  }
}

/// Converts one row of packed RGB444 pixels to packed `R, G, B, A` u16 samples.
///
/// Channels at native 4-bit width, low-bit aligned. Alpha forced to `0xFFFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_u16_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn rgb444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let r = (px >> 8) & 0x0F;
    let g = (px >> 4) & 0x0F;
    let b = px & 0x0F;
    let dst = x * 4;
    rgba_u16_out[dst] = r;
    rgba_u16_out[dst + 1] = g;
    rgba_u16_out[dst + 2] = b;
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- BGR444 (4X B4 G4 R4, bits [11:8]=B [7:4]=G [3:0]=R) ---------------

/// Converts one row of packed BGR444 pixels to packed `R, G, B` bytes.
///
/// BGR444 stores B in bits [11:8], G in [7:4], R in [3:0] (bits [15:12] padding).
/// Output byte order is always R, G, B. Each 4-bit channel is expanded via
/// nibble-replication: `(c << 4) | c`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr444_to_rgb_row(src: &[u8], rgb_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b4 = ((px >> 8) & 0x0F) as u8;
    let g4 = ((px >> 4) & 0x0F) as u8;
    let r4 = (px & 0x0F) as u8;
    let r = (r4 << 4) | r4;
    let g = (g4 << 4) | g4;
    let b = (b4 << 4) | b4;
    let dst = x * 3;
    rgb_out[dst] = r;
    rgb_out[dst + 1] = g;
    rgb_out[dst + 2] = b;
  }
}

/// Converts one row of packed BGR444 pixels to packed `R, G, B, A` bytes.
///
/// Same as `bgr444_to_rgb_row`; alpha is forced to `0xFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr444_to_rgba_row(src: &[u8], rgba_out: &mut [u8], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b4 = ((px >> 8) & 0x0F) as u8;
    let g4 = ((px >> 4) & 0x0F) as u8;
    let r4 = (px & 0x0F) as u8;
    let r = (r4 << 4) | r4;
    let g = (g4 << 4) | g4;
    let b = (b4 << 4) | b4;
    let dst = x * 4;
    rgba_out[dst] = r;
    rgba_out[dst + 1] = g;
    rgba_out[dst + 2] = b;
    rgba_out[dst + 3] = 0xFF;
  }
}

/// Converts one row of packed BGR444 pixels to packed `R, G, B` u16 samples.
///
/// Channels at native 4-bit width in R, G, B output order — no expansion applied.
/// Output ranges: R ∈ [0, 15], G ∈ [0, 15], B ∈ [0, 15].
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgb_u16_out.len() >= width * 3`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr444_to_rgb_u16_row(src: &[u8], rgb_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgb_u16_out.len() >= width * 3, "rgb_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b = (px >> 8) & 0x0F;
    let g = (px >> 4) & 0x0F;
    let r = px & 0x0F;
    let dst = x * 3;
    rgb_u16_out[dst] = r;
    rgb_u16_out[dst + 1] = g;
    rgb_u16_out[dst + 2] = b;
  }
}

/// Converts one row of packed BGR444 pixels to packed `R, G, B, A` u16 samples.
///
/// Channels at native 4-bit width in R, G, B output order. Alpha forced to `0xFFFF`.
///
/// # Panics (debug builds)
///
/// Asserts `src.len() >= width * 2` and `rgba_u16_out.len() >= width * 4`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn bgr444_to_rgba_u16_row(src: &[u8], rgba_u16_out: &mut [u16], width: usize) {
  debug_assert!(src.len() >= width * 2, "src row too short");
  debug_assert!(rgba_u16_out.len() >= width * 4, "rgba_u16_out too short");
  for x in 0..width {
    let px = u16::from_le_bytes([src[x * 2], src[x * 2 + 1]]);
    let b = (px >> 8) & 0x0F;
    let g = (px >> 4) & 0x0F;
    let r = px & 0x0F;
    let dst = x * 4;
    rgba_u16_out[dst] = r;
    rgba_u16_out[dst + 1] = g;
    rgba_u16_out[dst + 2] = b;
    rgba_u16_out[dst + 3] = 0xFFFF;
  }
}

// ---- Unit tests ---------------------------------------------------------

#[cfg(all(test, feature = "std"))]
mod tests {
  use super::*;

  // ---- RGB565 -------------------------------------------------------------

  #[test]
  fn rgb565_known_values() {
    // 0x07E0: G=63 (bits [10:5]=0b111111), R=0, B=0 → R=0, G=255, B=0
    let px: u16 = 0x07E0;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    rgb565_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb[0], 0, "R should be 0");
    assert_eq!(rgb[1], 255, "G should be 255");
    assert_eq!(rgb[2], 0, "B should be 0");
  }

  #[test]
  fn rgb565_all_ones() {
    // 0xFFFF: R5=31, G6=63, B5=31 → all channels 255 after expansion
    let px: u16 = 0xFFFF;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    rgb565_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [255, 255, 255]);
  }

  #[test]
  fn rgb565_all_zeros() {
    let src = [0u8; 2];
    let mut rgb = [0xFFu8; 3];
    rgb565_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [0, 0, 0]);
  }

  #[test]
  fn rgb565_r5_expansion_boundary() {
    // All 32 values of a 5-bit channel must map monotonically, 0→0, 31→255.
    let mut prev = 0u8;
    for c in 0u8..=31 {
      let expanded = (c << 3) | (c >> 2);
      if c > 0 {
        assert!(expanded >= prev, "5-bit expansion not monotone at c={c}");
      }
      prev = expanded;
    }
    assert_eq!(
      (31u8 << 3) | (31u8 >> 2),
      255,
      "5-bit max should expand to 255"
    );
    let zero5: u8 = 0;
    assert_eq!(
      (zero5 << 3) | (zero5 >> 2),
      0,
      "5-bit zero should expand to 0"
    );
  }

  #[test]
  fn rgb565_g6_expansion_boundary() {
    // All 64 values of a 6-bit channel must map monotonically, 0→0, 63→255.
    let mut prev = 0u8;
    for c in 0u8..=63 {
      let expanded = (c << 2) | (c >> 4);
      if c > 0 {
        assert!(expanded >= prev, "6-bit expansion not monotone at c={c}");
      }
      prev = expanded;
    }
    assert_eq!(
      (63u8 << 2) | (63u8 >> 4),
      255,
      "6-bit max should expand to 255"
    );
    let zero6: u8 = 0;
    assert_eq!(
      (zero6 << 2) | (zero6 >> 4),
      0,
      "6-bit zero should expand to 0"
    );
  }

  #[test]
  fn rgb565_u16_native_precision() {
    // Max pixel 0xFFFF: R=31, G=63, B=31 (native bit widths, no expansion)
    let px: u16 = 0xFFFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 3];
    rgb565_to_rgb_u16_row(&src, &mut out, 1);
    assert_eq!(out, [31, 63, 31]);
  }

  // ---- BGR565 -------------------------------------------------------------

  #[test]
  fn bgr565_channel_order() {
    // BGR565: bits [4:0]=R, so R5=31 means B=0, G=0, R=31 in source.
    // Pixel: bits [4:0] = 0x1F (R=31, G=0, B=0 in BGR565 encoding)
    let px: u16 = 0x001F;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    bgr565_to_rgb_row(&src, &mut rgb, 1);
    // Output must be R, G, B order: R=255 (expanded from 31), G=0, B=0
    assert_eq!(rgb[0], 255, "R (first byte of output) should be 255");
    assert_eq!(rgb[1], 0, "G should be 0");
    assert_eq!(rgb[2], 0, "B should be 0");
  }

  #[test]
  fn bgr565_all_ones() {
    let px: u16 = 0xFFFF;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    bgr565_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [255, 255, 255]);
  }

  #[test]
  fn bgr565_all_zeros() {
    let src = [0u8; 2];
    let mut rgb = [0xFFu8; 3];
    bgr565_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [0, 0, 0]);
  }

  #[test]
  fn bgr565_u16_native_precision() {
    // BGR565 max pixel: B5=31, G6=63, R5=31 → output R=31, G=63, B=31
    let px: u16 = 0xFFFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 3];
    bgr565_to_rgb_u16_row(&src, &mut out, 1);
    assert_eq!(out, [31, 63, 31]);
  }

  // ---- RGB555 -------------------------------------------------------------

  #[test]
  fn rgb555_known_values() {
    // RGB555: bits [14:10]=R5, [9:5]=G5, [4:0]=B5. Bit 15 = padding.
    // 0x7C00: R=31 (bits [14:10]=0x1F), G=0, B=0 → R=255, G=0, B=0
    let px: u16 = 0x7C00;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    rgb555_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb[0], 255, "R should be 255");
    assert_eq!(rgb[1], 0, "G should be 0");
    assert_eq!(rgb[2], 0, "B should be 0");
  }

  #[test]
  fn rgb555_all_ones() {
    // 0x7FFF (bit 15=0 for standard RGB555, rest all 1s): all channels 255
    let px: u16 = 0x7FFF;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    rgb555_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [255, 255, 255]);
  }

  #[test]
  fn rgb555_all_zeros() {
    let src = [0u8; 2];
    let mut rgb = [0xFFu8; 3];
    rgb555_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [0, 0, 0]);
  }

  #[test]
  fn rgb555_u16_native_precision() {
    // Max: R=31, G=31, B=31 (no expansion on u16 path)
    let px: u16 = 0x7FFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 3];
    rgb555_to_rgb_u16_row(&src, &mut out, 1);
    assert_eq!(out, [31, 31, 31]);
  }

  // ---- BGR555 -------------------------------------------------------------

  #[test]
  fn bgr555_channel_order() {
    // BGR555: bits [4:0]=R, [9:5]=G, [14:10]=B. Pixel 0x001F: R=31, G=0, B=0.
    let px: u16 = 0x001F;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    bgr555_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb[0], 255, "R (first byte of output) should be 255");
    assert_eq!(rgb[1], 0, "G should be 0");
    assert_eq!(rgb[2], 0, "B should be 0");
  }

  #[test]
  fn bgr555_all_ones() {
    let px: u16 = 0x7FFF;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    bgr555_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [255, 255, 255]);
  }

  #[test]
  fn bgr555_all_zeros() {
    let src = [0u8; 2];
    let mut rgb = [0xFFu8; 3];
    bgr555_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [0, 0, 0]);
  }

  #[test]
  fn bgr555_u16_native_precision() {
    // BGR555 max pixel (0x7FFF): B=31, G=31, R=31 → output R=31, G=31, B=31
    let px: u16 = 0x7FFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 3];
    bgr555_to_rgb_u16_row(&src, &mut out, 1);
    assert_eq!(out, [31, 31, 31]);
  }

  // ---- RGB444 -------------------------------------------------------------

  #[test]
  fn rgb444_known_values() {
    // RGB444: bits [11:8]=R4, [7:4]=G4, [3:0]=B4. Bits [15:12] padding.
    // 0x0F00: R=15 (bits [11:8]=0xF), G=0, B=0 → R=255, G=0, B=0
    let px: u16 = 0x0F00;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    rgb444_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb[0], 255, "R should be 255");
    assert_eq!(rgb[1], 0, "G should be 0");
    assert_eq!(rgb[2], 0, "B should be 0");
  }

  #[test]
  fn rgb444_all_ones() {
    // 0x0FFF: R=15, G=15, B=15 → all channels 255 after nibble-replication
    let px: u16 = 0x0FFF;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    rgb444_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [255, 255, 255]);
  }

  #[test]
  fn rgb444_all_zeros() {
    let src = [0u8; 2];
    let mut rgb = [0xFFu8; 3];
    rgb444_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [0, 0, 0]);
  }

  #[test]
  fn rgb444_4bit_expansion_boundary() {
    // All 16 values of a 4-bit channel: 0→0, 15→255, monotone.
    let mut prev = 0u8;
    for c in 0u8..=15 {
      let expanded = (c << 4) | c;
      if c > 0 {
        assert!(expanded >= prev, "4-bit expansion not monotone at c={c}");
      }
      prev = expanded;
    }
    assert_eq!((15u8 << 4) | 15u8, 255, "4-bit max should expand to 255");
    let zero4: u8 = 0;
    assert_eq!((zero4 << 4) | zero4, 0, "4-bit zero should expand to 0");
  }

  #[test]
  fn rgb444_u16_native_precision() {
    // Max: R=15, G=15, B=15 (no expansion on u16 path)
    let px: u16 = 0x0FFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 3];
    rgb444_to_rgb_u16_row(&src, &mut out, 1);
    assert_eq!(out, [15, 15, 15]);
  }

  // ---- BGR444 -------------------------------------------------------------

  #[test]
  fn bgr444_channel_order() {
    // BGR444: bits [3:0]=R, [7:4]=G, [11:8]=B. Pixel 0x000F: R=15, G=0, B=0.
    let px: u16 = 0x000F;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    bgr444_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb[0], 255, "R (first byte of output) should be 255");
    assert_eq!(rgb[1], 0, "G should be 0");
    assert_eq!(rgb[2], 0, "B should be 0");
  }

  #[test]
  fn bgr444_all_ones() {
    let px: u16 = 0x0FFF;
    let src = px.to_le_bytes();
    let mut rgb = [0u8; 3];
    bgr444_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [255, 255, 255]);
  }

  #[test]
  fn bgr444_all_zeros() {
    let src = [0u8; 2];
    let mut rgb = [0xFFu8; 3];
    bgr444_to_rgb_row(&src, &mut rgb, 1);
    assert_eq!(rgb, [0, 0, 0]);
  }

  #[test]
  fn bgr444_u16_native_precision() {
    // BGR444 max pixel (0x0FFF): B=15, G=15, R=15 → output R=15, G=15, B=15
    let px: u16 = 0x0FFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 3];
    bgr444_to_rgb_u16_row(&src, &mut out, 1);
    assert_eq!(out, [15, 15, 15]);
  }

  // ---- Multi-pixel correctness tests --------------------------------------

  #[test]
  fn rgb565_rgba_alpha_forced() {
    // RGBA output must have alpha=0xFF for every pixel regardless of source data.
    let pixels: &[u16] = &[0x0000, 0xFFFF, 0x07E0, 0xF800, 0x001F];
    let src: std::vec::Vec<u8> = pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
    let mut rgba = std::vec![0u8; pixels.len() * 4];
    rgb565_to_rgba_row(&src, &mut rgba, pixels.len());
    for i in 0..pixels.len() {
      assert_eq!(rgba[i * 4 + 3], 0xFF, "alpha at pixel {i} must be 0xFF");
    }
  }

  #[test]
  fn rgb555_rgba_alpha_forced() {
    let pixels: &[u16] = &[0x0000, 0x7FFF, 0x03E0];
    let src: std::vec::Vec<u8> = pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
    let mut rgba = std::vec![0u8; pixels.len() * 4];
    rgb555_to_rgba_row(&src, &mut rgba, pixels.len());
    for i in 0..pixels.len() {
      assert_eq!(rgba[i * 4 + 3], 0xFF, "alpha at pixel {i} must be 0xFF");
    }
  }

  #[test]
  fn rgb444_rgba_alpha_forced() {
    let pixels: &[u16] = &[0x0000, 0x0FFF, 0x0F00];
    let src: std::vec::Vec<u8> = pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
    let mut rgba = std::vec![0u8; pixels.len() * 4];
    rgb444_to_rgba_row(&src, &mut rgba, pixels.len());
    for i in 0..pixels.len() {
      assert_eq!(rgba[i * 4 + 3], 0xFF, "alpha at pixel {i} must be 0xFF");
    }
  }

  #[test]
  fn rgb565_rgba_u16_alpha_forced() {
    let px: u16 = 0xF800; // R=31, G=0, B=0
    let src = px.to_le_bytes();
    let mut out = [0u16; 4];
    rgb565_to_rgba_u16_row(&src, &mut out, 1);
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }

  #[test]
  fn bgr565_rgba_u16_alpha_forced() {
    let px: u16 = 0x001F; // R=31 in BGR565 position
    let src = px.to_le_bytes();
    let mut out = [0u16; 4];
    bgr565_to_rgba_u16_row(&src, &mut out, 1);
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }

  #[test]
  fn rgb555_rgba_u16_alpha_forced() {
    let px: u16 = 0x7FFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 4];
    rgb555_to_rgba_u16_row(&src, &mut out, 1);
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }

  #[test]
  fn bgr555_rgba_u16_alpha_forced() {
    let px: u16 = 0x7FFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 4];
    bgr555_to_rgba_u16_row(&src, &mut out, 1);
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }

  #[test]
  fn rgb444_rgba_u16_alpha_forced() {
    let px: u16 = 0x0FFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 4];
    rgb444_to_rgba_u16_row(&src, &mut out, 1);
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }

  #[test]
  fn bgr444_rgba_u16_alpha_forced() {
    let px: u16 = 0x0FFF;
    let src = px.to_le_bytes();
    let mut out = [0u16; 4];
    bgr444_to_rgba_u16_row(&src, &mut out, 1);
    assert_eq!(out[3], 0xFFFF, "alpha must be 0xFFFF");
  }
}
