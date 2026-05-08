use super::*;

// ---- Tier 9 Rgbf32 SIMD-vs-scalar parity tests --------------------------

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
fn wasm_rgbf32_to_rgb_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_simd = std::vec![0u8; w * 3];
    scalar::rgbf32_to_rgb_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf32_to_rgb width {w}");
  }
}

#[test]
fn wasm_rgbf32_to_rgba_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_simd = std::vec![0u8; w * 4];
    scalar::rgbf32_to_rgba_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgba_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf32_to_rgba width {w}");
  }
}

#[test]
fn wasm_rgbf32_to_rgb_u16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_simd = std::vec![0u16; w * 3];
    scalar::rgbf32_to_rgb_u16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_u16_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf32_to_rgb_u16 width {w}");
  }
}

#[test]
fn wasm_rgbf32_to_rgba_u16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_simd = std::vec![0u16; w * 4];
    scalar::rgbf32_to_rgba_u16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgba_u16_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf32_to_rgba_u16 width {w}");
  }
}

#[test]
fn wasm_rgbf32_to_rgb_f32_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf32(w);
    let mut out_scalar = std::vec![0.0f32; w * 3];
    let mut out_simd = std::vec![0.0f32; w * 3];
    scalar::rgbf32_to_rgb_f32_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf32_to_rgb_f32_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf32_to_rgb_f32 width {w}");
    assert_eq!(out_simd, input[..w * 3], "lossless width {w}");
  }
}

// ---- Tier 9 Rgbf16 wasm-simd128 parity tests ---------------------------------

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
fn wasm_rgbf16_to_rgb_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0u8; w * 3];
    let mut out_simd = std::vec![0u8; w * 3];
    scalar::rgbf16_to_rgb_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgb_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf16_to_rgb width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgba_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0u8; w * 4];
    let mut out_simd = std::vec![0u8; w * 4];
    scalar::rgbf16_to_rgba_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgba_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf16_to_rgba width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgb_u16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0u16; w * 3];
    let mut out_simd = std::vec![0u16; w * 3];
    scalar::rgbf16_to_rgb_u16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgb_u16_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf16_to_rgb_u16 width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgba_u16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0u16; w * 4];
    let mut out_simd = std::vec![0u16; w * 4];
    scalar::rgbf16_to_rgba_u16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgba_u16_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf16_to_rgba_u16 width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgb_f32_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![0.0f32; w * 3];
    let mut out_simd = std::vec![0.0f32; w * 3];
    scalar::rgbf16_to_rgb_f32_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgb_f32_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf16_to_rgb_f32 width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgb_f16_matches_scalar() {
  for w in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 1920, 1921] {
    let input = pseudo_random_rgbf16(w);
    let mut out_scalar = std::vec![half::f16::ZERO; w * 3];
    let mut out_simd = std::vec![half::f16::ZERO; w * 3];
    scalar::rgbf16_to_rgb_f16_row::<false>(&input, &mut out_scalar, w);
    unsafe {
      rgbf16_to_rgb_f16_row::<false>(&input, &mut out_simd, w);
    }
    assert_eq!(out_scalar, out_simd, "wasm rgbf16_to_rgb_f16 width {w}");
    assert_eq!(out_simd, input[..w * 3], "lossless width {w}");
  }
}

// ---- BE parity tests — wasm-simd128 Rgbf32 -----------------------------------

fn be_rgbf32(le: &[f32]) -> std::vec::Vec<f32> {
  le.iter()
    .map(|v| f32::from_bits(v.to_bits().swap_bytes()))
    .collect()
}

fn be_rgbf16(le: &[half::f16]) -> std::vec::Vec<half::f16> {
  le.iter()
    .map(|v| half::f16::from_bits(v.to_bits().swap_bytes()))
    .collect()
}

#[test]
fn wasm_rgbf32_to_rgb_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf32(w);
    let be_in = be_rgbf32(&le_in);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      rgbf32_to_rgb_row::<false>(&le_in, &mut out_le, w);
      rgbf32_to_rgb_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "wasm rgbf32_to_rgb BE parity width {w}");
  }
}

#[test]
fn wasm_rgbf32_to_rgba_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf32(w);
    let be_in = be_rgbf32(&le_in);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      rgbf32_to_rgba_row::<false>(&le_in, &mut out_le, w);
      rgbf32_to_rgba_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "wasm rgbf32_to_rgba BE parity width {w}");
  }
}

#[test]
fn wasm_rgbf32_to_rgb_u16_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf32(w);
    let be_in = be_rgbf32(&le_in);
    let mut out_le = std::vec![0u16; w * 3];
    let mut out_be = std::vec![0u16; w * 3];
    unsafe {
      rgbf32_to_rgb_u16_row::<false>(&le_in, &mut out_le, w);
      rgbf32_to_rgb_u16_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "wasm rgbf32_to_rgb_u16 BE parity width {w}");
  }
}

#[test]
fn wasm_rgbf32_to_rgba_u16_be_matches_le() {
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
      "wasm rgbf32_to_rgba_u16 BE parity width {w}"
    );
  }
}

#[test]
fn wasm_rgbf32_to_rgb_f32_be_is_byteswap() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf32(w);
    let be_in = be_rgbf32(&le_in);
    let mut out_le = std::vec![0.0f32; w * 3];
    let mut out_be = std::vec![0.0f32; w * 3];
    unsafe {
      rgbf32_to_rgb_f32_row::<false>(&le_in, &mut out_le, w);
      rgbf32_to_rgb_f32_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "wasm rgbf32_to_rgb_f32 BE parity width {w}");
  }
}

/// Feeds an explicitly LE-encoded fixture through `rgbf32_to_rgb_f32_row::<false>`
/// and asserts it decodes to the host-native expected values.
///
/// On LE hosts this is a vacuous sanity check (LE-encoded == host-native), but
/// on BE hosts it guards against the historical bug where the kernel used a raw
/// `v128_load`/`v128_store` copy in the `BE = false` branch, which preserved
/// the LE byte order on store and produced corrupted (byte-swapped) host f32s.
/// The current kernel falls through to the endian-aware `load_f32x4::<false>`
/// slow path on BE hosts (`HOST_NATIVE_BE != BE`) so this test passes on both.
/// (`wasm32-*` is LE today, but the routing is endian-agnostic for any future
/// BE wasm target.)
#[test]
fn wasm_rgbf32_to_rgb_f32_row_le_input_decodes_correctly_on_any_host() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let expected = pseudo_random_rgbf32(w); // host-native f32 values
    // Build LE-encoded input: each lane's bits, written as if LE on disk, then
    // reinterpreted as host-native f32. On LE hosts this is identical to
    // `expected`; on BE hosts each lane is byte-swapped.
    let le_in: std::vec::Vec<f32> = expected
      .iter()
      .map(|v| f32::from_bits(u32::from_le(v.to_bits())))
      .collect();
    let mut out = std::vec![0.0f32; w * 3];
    unsafe {
      rgbf32_to_rgb_f32_row::<false>(&le_in, &mut out, w);
    }
    assert_eq!(
      out, expected,
      "wasm rgbf32_to_rgb_f32_row::<false> must decode LE input to host-native (width {w})"
    );
  }
}

// ---- BE parity tests — wasm-simd128 Rgbf16 -----------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgb_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![0u8; w * 3];
    let mut out_be = std::vec![0u8; w * 3];
    unsafe {
      rgbf16_to_rgb_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgb_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "wasm rgbf16_to_rgb BE parity width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgba_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![0u8; w * 4];
    let mut out_be = std::vec![0u8; w * 4];
    unsafe {
      rgbf16_to_rgba_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgba_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "wasm rgbf16_to_rgba BE parity width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgb_u16_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![0u16; w * 3];
    let mut out_be = std::vec![0u16; w * 3];
    unsafe {
      rgbf16_to_rgb_u16_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgb_u16_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "wasm rgbf16_to_rgb_u16 BE parity width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgba_u16_be_matches_le() {
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
      "wasm rgbf16_to_rgba_u16 BE parity width {w}"
    );
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgb_f32_be_matches_le() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![0.0f32; w * 3];
    let mut out_be = std::vec![0.0f32; w * 3];
    unsafe {
      rgbf16_to_rgb_f32_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgb_f32_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "wasm rgbf16_to_rgb_f32 BE parity width {w}");
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn wasm_rgbf16_to_rgb_f16_be_is_byteswap() {
  for w in [1usize, 4, 7, 16, 33, 1920, 1921] {
    let le_in = pseudo_random_rgbf16(w);
    let be_in = be_rgbf16(&le_in);
    let mut out_le = std::vec![half::f16::ZERO; w * 3];
    let mut out_be = std::vec![half::f16::ZERO; w * 3];
    unsafe {
      rgbf16_to_rgb_f16_row::<false>(&le_in, &mut out_le, w);
      rgbf16_to_rgb_f16_row::<true>(&be_in, &mut out_be, w);
    }
    assert_eq!(out_le, out_be, "wasm rgbf16_to_rgb_f16 BE parity width {w}");
  }
}
