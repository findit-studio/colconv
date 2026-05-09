use super::super::*;

// ---- Tier 5.25 packed YUV 4:1:1 simd128 scalar-equivalence ---------

fn packed_yuv411_buffer(width: usize, seed: usize) -> std::vec::Vec<u8> {
  (0..width * 3 / 2)
    .map(|i| ((i.wrapping_mul(seed).wrapping_add(seed * 3)) & 0xFF) as u8)
    .collect()
}

fn check_uyyvyy411_rgb(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = packed_yuv411_buffer(width, 37);
  let mut s = std::vec![0u8; width * 3];
  let mut k = std::vec![0u8; width * 3];
  scalar::uyyvyy411_to_rgb_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyyvyy411_to_rgb_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "simd128 uyyvyy411→RGB diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

fn check_uyyvyy411_rgba(width: usize, matrix: ColorMatrix, full_range: bool) {
  let p = packed_yuv411_buffer(width, 37);
  let mut s = std::vec![0u8; width * 4];
  let mut k = std::vec![0u8; width * 4];
  scalar::uyyvyy411_to_rgba_row(&p, &mut s, width, matrix, full_range);
  unsafe {
    uyyvyy411_to_rgba_row(&p, &mut k, width, matrix, full_range);
  }
  assert_eq!(
    s, k,
    "simd128 uyyvyy411→RGBA diverges (width={width}, matrix={matrix:?}, full_range={full_range})"
  );
}

#[test]
#[cfg_attr(miri, ignore = "simd128 SIMD intrinsics unsupported by Miri")]
fn simd128_uyyvyy411_rgb_matches_scalar_all_matrices() {
  for m in [
    ColorMatrix::Bt601,
    ColorMatrix::Bt709,
    ColorMatrix::Bt2020Ncl,
    ColorMatrix::Smpte240m,
    ColorMatrix::Fcc,
    ColorMatrix::YCgCo,
  ] {
    for full in [true, false] {
      check_uyyvyy411_rgb(16, m, full);
      check_uyyvyy411_rgba(16, m, full);
    }
  }
}

#[test]
#[cfg_attr(miri, ignore = "simd128 SIMD intrinsics unsupported by Miri")]
fn simd128_uyyvyy411_matches_scalar_widths() {
  for w in [4usize, 8, 12, 16, 20, 28, 32, 48, 64, 128, 1920] {
    check_uyyvyy411_rgb(w, ColorMatrix::Bt709, false);
    check_uyyvyy411_rgba(w, ColorMatrix::Bt709, true);
    check_uyyvyy411_rgb(w, ColorMatrix::Bt2020Ncl, true);
    check_uyyvyy411_rgba(w, ColorMatrix::Bt2020Ncl, false);
  }
}

#[test]
#[cfg_attr(miri, ignore = "simd128 SIMD intrinsics unsupported by Miri")]
fn simd128_uyyvyy411_luma_matches_scalar() {
  for w in [4usize, 8, 12, 16, 20, 28, 32, 64, 128, 1920] {
    let p = packed_yuv411_buffer(w, 53);
    let mut s = std::vec![0u8; w];
    let mut k = std::vec![0u8; w];
    scalar::uyyvyy411_to_luma_row(&p, &mut s, w);
    unsafe {
      uyyvyy411_to_luma_row(&p, &mut k, w);
    }
    assert_eq!(s, k, "simd128 uyyvyy411→luma diverges (width={w})");
  }
}

#[test]
#[cfg_attr(miri, ignore = "simd128 SIMD intrinsics unsupported by Miri")]
fn simd128_uyyvyy411_luma_u16_matches_scalar() {
  for w in [4usize, 8, 12, 16, 20, 28, 32, 64, 128, 1920] {
    let p = packed_yuv411_buffer(w, 71);
    let mut s = std::vec![0u16; w];
    let mut k = std::vec![0u16; w];
    scalar::uyyvyy411_to_luma_u16_row(&p, &mut s, w);
    unsafe {
      uyyvyy411_to_luma_u16_row(&p, &mut k, w);
    }
    assert_eq!(s, k, "simd128 uyyvyy411→luma_u16 diverges (width={w})");
  }
}
