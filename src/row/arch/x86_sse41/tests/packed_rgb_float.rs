use super::super::*;

// ---- Tier 9 Rgbf32 SIMD-vs-scalar parity tests --------------------------

#[test]
#[cfg(target_arch = "x86_64")]
#[cfg_attr(miri, ignore = "MXCSR + SIMD intrinsics unsupported by Miri")]
fn rgbf32_to_rgb_row_simd_matches_scalar_under_truncate_mxcsr() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  // Save ambient MXCSR and set round-toward-zero (bits 13-14 = 0b11 = 0x6000).
  let saved = unsafe { core::arch::x86_64::_mm_getcsr() };
  let mxcsr_rz = (saved & !0x6000) | 0x6000;
  unsafe { core::arch::x86_64::_mm_setcsr(mxcsr_rz) };

  // Input: every channel is exactly 0.5 → after ×255 = 127.5, a half-boundary
  // value. Scalar (round-ties-even) → 128. SIMD without the fix → 127 under
  // truncate MXCSR. Use 16 pixels so the SIMD loop body executes at least once.
  let width = 16usize;
  let rgb = std::vec![0.5_f32; width * 3];
  let mut simd_out = std::vec![0u8; width * 3];
  let mut scalar_out = std::vec![0u8; width * 3];

  unsafe { rgbf32_to_rgb_row(&rgb, &mut simd_out, width) };
  scalar::rgbf32_to_rgb_row(&rgb, &mut scalar_out, width);

  // Restore MXCSR before any assertion so panic formatting doesn't misfire.
  unsafe { core::arch::x86_64::_mm_setcsr(saved) };

  assert_eq!(
    simd_out, scalar_out,
    "SSE4.1 SIMD diverged from scalar under truncate MXCSR (Codex #69)"
  );
}

fn pseudo_random_rgbf32(width: usize) -> std::vec::Vec<f32> {
  let n = width * 3;
  let mut out = std::vec::Vec::with_capacity(n);
  let mut state: u32 = 0xA5A5_3C3C;
  for i in 0..n {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let kind = (state >> 28) & 0b11;
    let v = match kind {
      0 => ((state >> 8) & 0xFF) as f32 / 255.0,
      1 => (((i as u32 & 0x7F) as f32) + 0.5) / 255.0,
      2 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.25,
      _ => -(((state >> 4) & 0xFF) as f32) / 255.0,
    };
    out.push(v);
  }
  out
}

#[test]
fn sse41_rgbf32_to_rgb_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_simd = std::vec![0u8; w * 3];
    scalar::rgbf32_to_rgb_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_row(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "SSE4.1 rgbf32_to_rgb width {w}");
  }
}

#[test]
fn sse41_rgbf32_to_rgba_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_simd = std::vec![0u8; w * 4];
    scalar::rgbf32_to_rgba_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgba_row(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "SSE4.1 rgbf32_to_rgba width {w}");
  }
}

#[test]
fn sse41_rgbf32_to_rgb_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_simd = std::vec![0u16; w * 3];
    scalar::rgbf32_to_rgb_u16_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_u16_row(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "SSE4.1 rgbf32_to_rgb_u16 width {w}");
  }
}

#[test]
fn sse41_rgbf32_to_rgba_u16_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_simd = std::vec![0u16; w * 4];
    scalar::rgbf32_to_rgba_u16_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgba_u16_row(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "SSE4.1 rgbf32_to_rgba_u16 width {w}");
  }
}

#[test]
fn sse41_rgbf32_to_rgb_f32_matches_scalar() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0.0f32; w * 3];
    let mut out_simd = std::vec![0.0f32; w * 3];
    scalar::rgbf32_to_rgb_f32_row(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_f32_row(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "SSE4.1 rgbf32_to_rgb_f32 width {w}");
    assert_eq!(out_simd, input[..w * 3], "lossless width {w}");
  }
}
