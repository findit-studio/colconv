//! BE parity tests for AVX2 high-bit YUV / P-format kernels.
//!
//! Each test takes a randomized LE input buffer, byte-swaps every u16
//! element to produce a BE-encoded buffer, then asserts that
//! `kernel::<BE = true>(swapped_input)` produces byte-identical output
//! to `kernel::<BE = false>(original_input)`.

use super::{
  super::*, high_bit_plane_avx2, interleave_uv_avx2, p_n_packed_plane, p010_uv_interleave,
  p16_plane_avx2, planar_n_plane,
};

fn byteswap_u16_buf(buf: &[u16]) -> std::vec::Vec<u16> {
  buf.iter().map(|x| x.swap_bytes()).collect()
}

#[test]
fn avx2_yuv_420p10_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let width = 64;
  let y = planar_n_plane::<10>(width, 13);
  let u = planar_n_plane::<10>(width / 2, 17);
  let v = planar_n_plane::<10>(width / 2, 19);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    yuv_420p_n_to_rgb_row::<10, false>(&y, &u, &v, &mut out_le, width, ColorMatrix::Bt709, true);
    yuv_420p_n_to_rgb_row::<10, true>(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      width,
      ColorMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

#[test]
fn avx2_yuv_420p10_be_parity_u16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let width = 64;
  let y = planar_n_plane::<10>(width, 23);
  let u = planar_n_plane::<10>(width / 2, 29);
  let v = planar_n_plane::<10>(width / 2, 31);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    yuv_420p_n_to_rgb_u16_row::<10, false>(
      &y,
      &u,
      &v,
      &mut out_le,
      width,
      ColorMatrix::Bt709,
      true,
    );
    yuv_420p_n_to_rgb_u16_row::<10, true>(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      width,
      ColorMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

#[test]
fn avx2_yuv_444p12_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let width = 64;
  let y = planar_n_plane::<12>(width, 41);
  let u = planar_n_plane::<12>(width, 43);
  let v = planar_n_plane::<12>(width, 47);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    yuv_444p_n_to_rgb_row::<12, false>(&y, &u, &v, &mut out_le, width, ColorMatrix::Bt709, true);
    yuv_444p_n_to_rgb_row::<12, true>(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      width,
      ColorMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

#[test]
fn avx2_yuv_420p16_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let width = 64;
  let y = p16_plane_avx2(width, 53);
  let u = p16_plane_avx2(width / 2, 59);
  let v = p16_plane_avx2(width / 2, 61);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    yuv_420p16_to_rgb_row::<false>(&y, &u, &v, &mut out_le, width, ColorMatrix::Bt709, true);
    yuv_420p16_to_rgb_row::<true>(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      width,
      ColorMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

#[test]
fn avx2_yuv_444p16_be_parity_u16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let width = 64;
  let y = p16_plane_avx2(width, 67);
  let u = p16_plane_avx2(width, 71);
  let v = p16_plane_avx2(width, 73);
  let y_be = byteswap_u16_buf(&y);
  let u_be = byteswap_u16_buf(&u);
  let v_be = byteswap_u16_buf(&v);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    yuv_444p16_to_rgb_u16_row::<false>(&y, &u, &v, &mut out_le, width, ColorMatrix::Bt709, true);
    yuv_444p16_to_rgb_u16_row::<true>(
      &y_be,
      &u_be,
      &v_be,
      &mut out_be,
      width,
      ColorMatrix::Bt709,
      true,
    );
  }
  assert_eq!(out_le, out_be);
}

#[test]
fn avx2_p010_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let width = 64;
  let y = p_n_packed_plane::<10>(width, 79);
  let u_half = p_n_packed_plane::<10>(width / 2, 83);
  let v_half = p_n_packed_plane::<10>(width / 2, 89);
  let uv_half = p010_uv_interleave(&u_half, &v_half);
  let y_be = byteswap_u16_buf(&y);
  let uv_be = byteswap_u16_buf(&uv_half);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    p_n_to_rgb_row::<10, false>(&y, &uv_half, &mut out_le, width, ColorMatrix::Bt709, true);
    p_n_to_rgb_row::<10, true>(&y_be, &uv_be, &mut out_be, width, ColorMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[test]
fn avx2_p410_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let width = 64;
  let y = p_n_packed_plane::<10>(width, 97);
  let u_full = high_bit_plane_avx2::<10>(width, 101);
  let v_full = high_bit_plane_avx2::<10>(width, 103);
  let uv_full = interleave_uv_avx2(&u_full, &v_full);
  let y_be = byteswap_u16_buf(&y);
  let uv_be = byteswap_u16_buf(&uv_full);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    p_n_444_to_rgb_row::<10, false>(&y, &uv_full, &mut out_le, width, ColorMatrix::Bt709, true);
    p_n_444_to_rgb_row::<10, true>(&y_be, &uv_be, &mut out_be, width, ColorMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[test]
fn avx2_p016_be_parity_u8() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let width = 64;
  let y = p16_plane_avx2(width, 107);
  let u_half = p16_plane_avx2(width / 2, 109);
  let v_half = p16_plane_avx2(width / 2, 113);
  let uv_half = p010_uv_interleave(&u_half, &v_half);
  let y_be = byteswap_u16_buf(&y);
  let uv_be = byteswap_u16_buf(&uv_half);

  let mut out_le = std::vec![0u8; width * 3];
  let mut out_be = std::vec![0u8; width * 3];
  unsafe {
    p16_to_rgb_row::<false>(&y, &uv_half, &mut out_le, width, ColorMatrix::Bt709, true);
    p16_to_rgb_row::<true>(&y_be, &uv_be, &mut out_be, width, ColorMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}

#[test]
fn avx2_p416_be_parity_u16() {
  if !std::arch::is_x86_feature_detected!("avx2") {
    return;
  }
  let width = 32;
  let y = p16_plane_avx2(width, 127);
  let u_full = p16_plane_avx2(width, 131);
  let v_full = p16_plane_avx2(width, 137);
  let uv_full = interleave_uv_avx2(&u_full, &v_full);
  let y_be = byteswap_u16_buf(&y);
  let uv_be = byteswap_u16_buf(&uv_full);

  let mut out_le = std::vec![0u16; width * 3];
  let mut out_be = std::vec![0u16; width * 3];
  unsafe {
    p_n_444_16_to_rgb_u16_row::<false>(&y, &uv_full, &mut out_le, width, ColorMatrix::Bt709, true);
    p_n_444_16_to_rgb_u16_row::<true>(&y_be, &uv_be, &mut out_be, width, ColorMatrix::Bt709, true);
  }
  assert_eq!(out_le, out_be);
}
