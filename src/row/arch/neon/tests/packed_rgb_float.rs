use super::*;

// ---- Tier 9 Rgbf32 SIMD-vs-scalar parity tests --------------------------

/// Generates a row of pseudo-random `f32` RGB samples. Mix of in-range
/// `[0, 1]` values, exact `0.5` (round-half-even tie), and HDR > 1.0
/// values to exercise clamping and rounding rules.
fn pseudo_random_rgbf32(width: usize) -> std::vec::Vec<f32> {
  let n = width * 3;
  let mut out = std::vec::Vec::with_capacity(n);
  let mut state: u32 = 0xA5A5_3C3C;
  for i in 0..n {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let kind = (state >> 28) & 0b11;
    let v = match kind {
      // 25% — in-range [0, 1] from low 8 bits / 255.
      0 => ((state >> 8) & 0xFF) as f32 / 255.0,
      // 25% — half-LSB ties (drives round-to-even branch).
      1 => (((i as u32 & 0x7F) as f32) + 0.5) / 255.0,
      // 25% — HDR > 1.0 (clamps to 1.0).
      2 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.25,
      // 25% — negatives (clamp to 0.0).
      _ => -(((state >> 4) & 0xFF) as f32) / 255.0,
    };
    out.push(v);
  }
  out
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn rgbf32_to_rgb_neon_matches_scalar_widths() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::rgbf32_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn rgbf32_to_rgba_neon_matches_scalar_widths() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::rgbf32_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgba_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn rgbf32_to_rgb_u16_neon_matches_scalar_widths() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::rgbf32_to_rgb_u16_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_u16_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn rgbf32_to_rgba_u16_neon_matches_scalar_widths() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::rgbf32_to_rgba_u16_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgba_u16_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn rgbf32_to_rgb_f32_neon_matches_scalar_widths() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0.0f32; w * 3];
    let mut out_neon = std::vec![0.0f32; w * 3];
    scalar::rgbf32_to_rgb_f32_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_f32_row(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Lossless: output should equal input bit-exact.
    assert_eq!(out_neon, input[..w * 3], "lossless width {w}");
  }
}
