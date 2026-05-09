mod ayuv64;
mod be_parity;
mod endian;
mod gray;
mod high_bit_4_2_0;
mod high_bit_4_4_4_and_pn;
mod legacy_rgb;
mod mono1bit;
mod packed_rgb_16bit;
mod packed_rgb_float;
mod packed_yuv_4_1_1;
mod packed_yuv_8bit;
mod planar_8bit_and_nv;
mod planar_gbr;
mod planar_gbr_float;
mod planar_gbr_high_bit;
mod v210;
mod v30x;
mod v410;
mod vuya;
mod xv36;
#[cfg(any(feature = "std", feature = "alloc"))]
mod xyz12;
mod y216;
mod y2xx;
mod yuva;

// ---- Shared test helpers (used across submodule tests) -------------

pub(super) fn p010_uv_interleave(u: &[u16], v: &[u16]) -> std::vec::Vec<u16> {
  let pairs = u.len();
  debug_assert_eq!(u.len(), v.len());
  let mut out = std::vec::Vec::with_capacity(pairs * 2);
  for i in 0..pairs {
    out.push(u[i]);
    out.push(v[i]);
  }
  out
}

pub(super) fn planar_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  (0..n)
    .map(|i| ((i * seed + seed * 3) as u32 & mask) as u16)
    .collect()
}

pub(super) fn p_n_packed_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i * seed + seed * 3) as u32 & mask) as u16) << shift)
    .collect()
}

pub(super) fn p16_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

pub(super) fn high_bit_plane_sse41<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask) << shift)
    .collect()
}

pub(super) fn interleave_uv_sse41(u_full: &[u16], v_full: &[u16]) -> std::vec::Vec<u16> {
  debug_assert_eq!(u_full.len(), v_full.len());
  let mut out = std::vec::Vec::with_capacity(u_full.len() * 2);
  for i in 0..u_full.len() {
    out.push(u_full[i]);
    out.push(v_full[i]);
  }
  out
}

/// Deterministic packed UYYVYY411 buffer: `width * 3 / 2` bytes per
/// row, hash-like seed per byte position. Shared across the
/// packed‑4:1:1 SIMD parity tests.
pub(super) fn packed_yuv411_buffer(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 3 / 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFF) as u8)
    .collect()
}
