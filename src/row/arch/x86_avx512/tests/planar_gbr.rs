//! AVX-512 parity tests for the planar-GBR kernels (Tier 10).

use super::super::*;

fn pseudo_random_plane(width: usize, seed: u32) -> std::vec::Vec<u8> {
  let mut state = seed;
  (0..width)
    .map(|_| {
      state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
      (state >> 8) as u8
    })
    .collect()
}

#[test]
fn avx512_gbr_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [
    1usize, 7, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 1920, 1921,
  ] {
    let g = pseudo_random_plane(w, 0x6CCD_5C7B);
    let b = pseudo_random_plane(w, 0x12AB_34CD);
    let r = pseudo_random_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_avx = std::vec![0u8; w * 3];
    scalar::gbr_to_rgb_row(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgb_row(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 gbr_to_rgb diverges (width={w})"
    );
  }
}

#[test]
fn avx512_gbra_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [
    1usize, 7, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 1920, 1921,
  ] {
    let g = pseudo_random_plane(w, 0x6CCD_5C7B);
    let b = pseudo_random_plane(w, 0x12AB_34CD);
    let r = pseudo_random_plane(w, 0xDEAD_BEEF);
    let a = pseudo_random_plane(w, 0xCAFE_F00D);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::gbra_to_rgba_row(&g, &b, &r, &a, &mut out_scalar, w);
    unsafe {
      gbra_to_rgba_row(&g, &b, &r, &a, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 gbra_to_rgba diverges (width={w})"
    );
  }
}

#[test]
fn avx512_gbr_to_rgba_opaque_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  for w in [
    1usize, 7, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 1920, 1921,
  ] {
    let g = pseudo_random_plane(w, 0x6CCD_5C7B);
    let b = pseudo_random_plane(w, 0x12AB_34CD);
    let r = pseudo_random_plane(w, 0xDEAD_BEEF);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_avx = std::vec![0u8; w * 4];
    scalar::gbr_to_rgba_opaque_row(&g, &b, &r, &mut out_scalar, w);
    unsafe {
      gbr_to_rgba_opaque_row(&g, &b, &r, &mut out_avx, w);
    }
    assert_eq!(
      out_scalar, out_avx,
      "AVX-512 gbr_to_rgba_opaque diverges (width={w})"
    );
  }
}
