#[cfg(feature = "yuv-444-packed")]
mod ayuv64;
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod be_parity;
mod endian;
#[cfg(feature = "gray")]
mod gray;
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod high_bit_4_2_0;
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod high_bit_4_4_4_and_pn;
#[cfg(feature = "rgb-legacy")]
mod legacy_rgb;
#[cfg(feature = "mono")]
mod mono1bit;
#[cfg(feature = "rgb")]
mod packed_rgb_16bit;
#[cfg(feature = "rgb-float")]
mod packed_rgb_float;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_4_1_1;
#[cfg(feature = "yuv-packed")]
mod packed_yuv_8bit;
#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
mod planar_8bit_and_nv;
#[cfg(feature = "gbr")]
mod planar_gbr;
#[cfg(feature = "gbr")]
mod planar_gbr_float;
#[cfg(feature = "gbr")]
mod planar_gbr_high_bit;
#[cfg(feature = "v210")]
mod v210;
#[cfg(feature = "yuv-444-packed")]
mod v30x;
#[cfg(feature = "yuv-444-packed")]
mod v410;
#[cfg(feature = "yuv-444-packed")]
mod vuya;
#[cfg(feature = "yuv-444-packed")]
mod xv36;
#[cfg(all(feature = "xyz", any(feature = "std", feature = "alloc")))]
mod xyz12;
#[cfg(feature = "y2xx")]
mod y216;
#[cfg(feature = "y2xx")]
mod y2xx;
#[cfg(feature = "yuva")]
mod yuva;

// ---- Shared test helpers (used across submodule tests) -------------

#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
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

#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar", feature = "yuva",))]
pub(super) fn planar_n_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  (0..n)
    .map(|i| ((i * seed + seed * 3) as u32 & mask) as u16)
    .collect()
}

#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar"))]
pub(super) fn p_n_packed_plane<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = (1u32 << BITS) - 1;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i * seed + seed * 3) as u32 & mask) as u16) << shift)
    .collect()
}

#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar", feature = "yuva",))]
pub(super) fn p16_plane(n: usize, seed: usize) -> std::vec::Vec<u16> {
  (0..n)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFFFF) as u16)
    .collect()
}

#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar", feature = "yuva",))]
pub(super) fn high_bit_plane_sse41<const BITS: u32>(n: usize, seed: usize) -> std::vec::Vec<u16> {
  let mask = ((1u32 << BITS) - 1) as u16;
  let shift = 16 - BITS;
  (0..n)
    .map(|i| (((i.wrapping_mul(seed).wrapping_add(seed * 3)) as u16) & mask) << shift)
    .collect()
}

#[cfg(any(feature = "yuv-planar", feature = "yuv-semi-planar", feature = "yuva",))]
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
#[cfg(feature = "yuv-packed")]
pub(super) fn packed_yuv411_buffer(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 3 / 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFF) as u8)
    .collect()
}
