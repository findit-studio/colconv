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
    scalar::rgbf32_to_rgb_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_row::<false>(&input, &mut out_neon, w);
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
    scalar::rgbf32_to_rgba_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgba_row::<false>(&input, &mut out_neon, w);
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
    scalar::rgbf32_to_rgb_u16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_u16_row::<false>(&input, &mut out_neon, w);
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
    scalar::rgbf32_to_rgba_u16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgba_u16_row::<false>(&input, &mut out_neon, w);
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
    scalar::rgbf32_to_rgb_f32_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_f32_row::<false>(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Lossless: output should equal input bit-exact.
    assert_eq!(out_neon, input[..w * 3], "lossless width {w}");
  }
}

// ---- Tier 9 Rgbf16 NEON parity tests ----------------------------------------
//
// Each test converts the same pseudo-random f16 input through both the scalar
// and NEON paths and asserts bit-exact equality.  Because half::f16 widening
// is lossless for values that originated as f32 (bit patterns round-trip),
// any divergence would be a logic bug in the NEON widening or downstream SIMD.

fn pseudo_random_rgbf16(width: usize) -> std::vec::Vec<half::f16> {
  pseudo_random_rgbf32(width)
    .iter()
    .map(|&v| half::f16::from_f32(v))
    .collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgb_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_neon = std::vec![0u8; w * 3];
    scalar::rgbf16_to_rgb_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgb_row::<false>(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgba_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_neon = std::vec![0u8; w * 4];
    scalar::rgbf16_to_rgba_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgba_row::<false>(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgb_u16_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_neon = std::vec![0u16; w * 3];
    scalar::rgbf16_to_rgb_u16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgb_u16_row::<false>(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgba_u16_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_neon = std::vec![0u16; w * 4];
    scalar::rgbf16_to_rgba_u16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgba_u16_row::<false>(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgb_f32_matches_scalar() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0.0f32; w * 3];
    let mut out_neon = std::vec![0.0f32; w * 3];
    scalar::rgbf16_to_rgb_f32_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgb_f32_row::<false>(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgb_f16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![half::f16::ZERO; w * 3];
    let mut out_neon = std::vec![half::f16::ZERO; w * 3];
    scalar::rgbf16_to_rgb_f16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgb_f16_row::<false>(&input, &mut out_neon, w);
    }
    assert_eq!(out_scalar, out_neon, "width {w}");
    // Lossless: output should equal input bit-exact.
    assert_eq!(out_neon, input[..w * 3], "lossless width {w}");
  }
}

// ---- BE parity tests — Rgbf32 -----------------------------------------------
//
// For each kernel: byte-swap the LE f32 inputs into a BE buffer, call the
// kernel with `BE=true`, and assert the output matches the LE run (`BE=false`).

/// Build a BE-encoded f32 slice by byte-swapping every 32-bit element.
fn be_rgbf32(le: &[f32]) -> std::vec::Vec<f32> {
  le.iter()
    .map(|v| f32::from_bits(v.to_bits().swap_bytes()))
    .collect()
}

/// Build a BE-encoded f16 slice by byte-swapping every 16-bit element.
fn be_rgbf16(le: &[half::f16]) -> std::vec::Vec<half::f16> {
  le.iter()
    .map(|v| half::f16::from_bits(v.to_bits().swap_bytes()))
    .collect()
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgbf32_to_rgb_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf32(w);
    let be_in = be_rgbf32(&le_in);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      rgbf32_to_rgb_row::<false>(&le_in, &mut out_le, w);
      rgbf32_to_rgb_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "NEON rgbf32_to_rgb BE parity width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgbf32_to_rgba_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf32(w);
    let be_in = be_rgbf32(&le_in);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      rgbf32_to_rgba_row::<false>(&le_in, &mut out_le, w);
      rgbf32_to_rgba_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "NEON rgbf32_to_rgba BE parity width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgbf32_to_rgb_u16_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf32(w);
    let be_in = be_rgbf32(&le_in);
    let mut out_le = std::vec![0u16; w * 3];
    let mut out_be = std::vec![0u16; w * 3];
    unsafe {
      rgbf32_to_rgb_u16_row::<false>(&le_in, &mut out_le, w);
      rgbf32_to_rgb_u16_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "NEON rgbf32_to_rgb_u16 BE parity width {w}");
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgbf32_to_rgba_u16_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf32(w);
    let be_in = be_rgbf32(&le_in);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      rgbf32_to_rgba_u16_row::<false>(&le_in, &mut out_le, w);
      rgbf32_to_rgba_u16_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "NEON rgbf32_to_rgba_u16 BE parity width {w}"
    );
  }
}

#[test]
#[cfg_attr(miri, ignore = "NEON SIMD intrinsics unsupported by Miri")]
fn neon_rgbf32_to_rgb_f32_be_is_byteswap() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf32(w);
    let be_in = be_rgbf32(&le_in);
    let mut out_le = std::vec![0.0f32; w * 3];
    let mut out_be = std::vec![0.0f32; w * 3];
    unsafe {
      rgbf32_to_rgb_f32_row::<false>(&le_in, &mut out_le, w);
      rgbf32_to_rgb_f32_row::<true>(&be_in, &mut out_be, w);
    }
    // BE path byte-swaps each f32, producing host-native = same as LE.
    assert_eq!(out_le, out_be, "NEON rgbf32_to_rgb_f32 BE parity width {w}");
  }
}

// ---- BE parity tests — Rgbf16 -----------------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgb_be_matches_le() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      rgbf16_to_rgb_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgb_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "NEON rgbf16_to_rgb BE parity width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgba_be_matches_le() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      rgbf16_to_rgba_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgba_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "NEON rgbf16_to_rgba BE parity width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgb_u16_be_matches_le() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![0u16; w * 3];
    let mut out_be = std::vec![0u16; w * 3];
    unsafe {
      rgbf16_to_rgb_u16_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgb_u16_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "NEON rgbf16_to_rgb_u16 BE parity width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgba_u16_be_matches_le() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![0u16; w * 4];
    let mut out_be = std::vec![0u16; w * 4];
    unsafe {
      rgbf16_to_rgba_u16_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgba_u16_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(
      out_le, out_be,
      "NEON rgbf16_to_rgba_u16 BE parity width {w}"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgb_f32_be_matches_le() {
  if !std::arch::is_aarch64_feature_detected!("fp16") {
    return;
  }
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![0.0f32; w * 3];
    let mut out_be = std::vec![0.0f32; w * 3];
    unsafe {
      rgbf16_to_rgb_f32_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgb_f32_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "NEON rgbf16_to_rgb_f32 BE parity width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn neon_rgbf16_to_rgb_f16_be_is_byteswap() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![half::f16::ZERO; w * 3];
    let mut out_be = std::vec![half::f16::ZERO; w * 3];
    unsafe {
      rgbf16_to_rgb_f16_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgb_f16_row::<true>(&be_in, &mut out_be, w);
    }
    // BE byte-swap should reconstruct original LE output bit-exact.
    assert_eq!(out_le, out_be, "NEON rgbf16_to_rgb_f16 BE parity width {w}");
  }
}
