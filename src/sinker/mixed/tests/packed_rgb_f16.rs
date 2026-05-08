use super::*;

// ---- Tier 9 — Rgbf16 packed-half-float-RGB source family ----------------

/// Builds a tightly-packed Rgbf16 row buffer (`width * height * 3` `f16`
/// elements, no row stride padding) filled with a constant `(R, G, B)` triple.
fn solid_rgbf16_frame(
  width: u32,
  height: u32,
  r: half::f16,
  g: half::f16,
  b: half::f16,
) -> std::vec::Vec<half::f16> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![half::f16::ZERO; w * h * 3];
  for px in buf.chunks_mut(3) {
    px[0] = r;
    px[1] = g;
    px[2] = b;
  }
  buf
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_clamps_to_u8() {
  // 1.0 → 255, 2.0 → 255 (HDR clamp), -0.5 → 0 (negative clamp).
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(2.0),
    half::f16::from_f32(-0.5),
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_u16_clamps_to_u16() {
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(0.5),
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.5),
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_u16(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // 0.5 * 65535 ≈ 32767 or 32768 (half-precision rounds 0.5 to exact 0.5,
  // so downstream is the same as Rgbf32); 1.0 → 65535; 1.5 → 65535 (clamp).
  for px in rgb_out.chunks(3) {
    assert!(
      px[0] >= 32767 && px[0] <= 32768,
      "unexpected mid: {}",
      px[0]
    );
    assert_eq!(px[1], 65535);
    assert_eq!(px[2], 65535);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_f16_is_lossless() {
  // Include HDR, negatives, and in-range values to confirm bit-exact
  // pass-through.
  let vals_f32 = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0];
  let n_pixels = 16 * 4;
  let pix: std::vec::Vec<half::f16> = (0..n_pixels * 3)
    .map(|i| half::f16::from_f32(vals_f32[i % vals_f32.len()]))
    .collect();
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![half::f16::ZERO; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_f16(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // Bit-exact equality (no rounding, no clamping in the f16 path).
  assert_eq!(rgb_out, pix);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_rgb_f32_widens_losslessly() {
  // Includes HDR (> 1.0), negatives, and exact values.
  let vals_f32 = [0.0f32, 1.0, -0.25, 1.5, 0.5, 100.0];
  let n_pixels = 16 * 4;
  let pix: std::vec::Vec<half::f16> = (0..n_pixels * 3)
    .map(|i| half::f16::from_f32(vals_f32[i % vals_f32.len()]))
    .collect();
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_f32(&mut rgb_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // Each widened f32 must equal the f16 widened via to_f32.
  let expected: std::vec::Vec<f32> = pix.iter().map(|h| h.to_f32()).collect();
  assert_eq!(rgb_out, expected, "rgb_f32 widen is not lossless");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_luma_u8() {
  // Constant white → BT.709 full-range luma 255.
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_luma(&mut luma_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_luma_u16() {
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
    half::f16::from_f32(1.0),
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u16; 16 * 4];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_luma_u16(&mut luma_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // u8 luma 255 → u16 255 (zero-extended).
  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_with_hsv() {
  // Pure red → H=0, S=255, V=255 in the OpenCV 8-bit HSV encoding.
  let pix = solid_rgbf16_frame(
    16,
    4,
    half::f16::from_f32(1.0),
    half::f16::ZERO,
    half::f16::ZERO,
  );
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let n = 16 * 4;
  let mut h_out = std::vec![0u8; n];
  let mut s_out = std::vec![0u8; n];
  let mut v_out = std::vec![0u8; n];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_hsv(&mut h_out, &mut s_out, &mut v_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for i in 0..n {
    assert_eq!(h_out[i], 0);
    assert_eq!(s_out[i], 255);
    assert_eq!(v_out[i], 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both SIMD main loop and scalar tail across
  // every backend block size.
  let w = 1921usize;
  let h = 4usize;
  let n_lanes = w * h * 3;
  let mut pix = std::vec![half::f16::ZERO; n_lanes];
  let mut state: u32 = 0xDEAD_BEEF;
  for (i, v) in pix.iter_mut().enumerate() {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let f = match (state >> 28) & 0b11 {
      0 => ((state >> 8) & 0xFF) as f32 / 255.0,
      1 => (((i as u32 & 0x7F) as f32) + 0.5) / 255.0,
      2 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.25,
      _ => -(((state >> 4) & 0xFF) as f32) / 255.0,
    };
    *v = half::f16::from_f32(f);
  }
  let src = Rgbf16Frame::try_new(&pix, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];
  let mut rgba_u16_simd = std::vec![0u16; w * h * 4];
  let mut rgba_u16_scalar = std::vec![0u16; w * h * 4];
  let mut rgb_f16_simd = std::vec![half::f16::ZERO; w * h * 3];
  let mut rgb_f16_scalar = std::vec![half::f16::ZERO; w * h * 3];
  let mut rgb_f32_simd = std::vec![0.0f32; w * h * 3];
  let mut rgb_f32_scalar = std::vec![0.0f32; w * h * 3];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];
  let mut luma_u16_simd = std::vec![0u16; w * h];
  let mut luma_u16_scalar = std::vec![0u16; w * h];

  let mut s_simd = MixedSinker::<Rgbf16>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_simd)
    .unwrap()
    .with_rgb_f16(&mut rgb_f16_simd)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap()
    .with_luma_u16(&mut luma_u16_simd)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Rgbf16>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_scalar)
    .unwrap()
    .with_rgb_f16(&mut rgb_f16_scalar)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap()
    .with_luma_u16(&mut luma_u16_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(rgb_u16_simd, rgb_u16_scalar, "RGB u16 output diverges");
  assert_eq!(rgba_u16_simd, rgba_u16_scalar, "RGBA u16 output diverges");
  assert_eq!(rgb_f16_simd, rgb_f16_scalar, "RGB f16 output diverges");
  assert_eq!(rgb_f32_simd, rgb_f32_scalar, "RGB f32 output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
  assert_eq!(luma_u16_simd, luma_u16_scalar, "Luma u16 output diverges");
  assert_eq!(rgb_f16_simd, pix, "RGB f16 output is not lossless");
}

/// Sinker-layer host-native-`f16` regression for the bug fixed alongside
/// `c3a6478` (PR #83 codex 2nd-pass review): the [`Rgbf16`] sinker used to
/// hardcode `::<false>` when calling the row dispatchers, telling them to
/// "decode LE-encoded input". Because [`Rgbf16Frame`] hands us a host-native
/// `&[half::f16]` row, that routing was a no-op on LE hosts but corrupted
/// every output path on BE hosts (the `u16` loaders would byte-swap an
/// already-decoded f16 bit-pattern). The fix replaces those `::<false>` with
/// `::<HOST_NATIVE_BE>`, which is `false` on LE and `true` on BE — a no-op
/// byte-swap on either host.
///
/// On a LE host (the only target Apple-Silicon and x86_64 macOS can run),
/// `HOST_NATIVE_BE = false` and `::<HOST_NATIVE_BE>` is byte-for-byte
/// identical to `::<false>`, so this test cannot distinguish the broken vs
/// fixed code on LE. It instead documents the equivalence at the **kernel
/// dispatch** layer — calling each `rgbf16_to_*` dispatcher with both
/// `BE = false` and `BE = HOST_NATIVE_BE` (= `cfg!(target_endian = "big")`)
/// must produce identical output on the active host. On a hypothetical BE
/// host (full QEMU s390x coverage is Phase 3), the same equivalence holds
/// for the **fixed** sinker but would fail for the broken one.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_kernel_host_native_be_matches_false_on_le_host() {
  use crate::row::{
    rgbf16_to_rgb_f16_row, rgbf16_to_rgb_f32_row, rgbf16_to_rgb_row, rgbf16_to_rgb_u16_row,
    rgbf16_to_rgba_row, rgbf16_to_rgba_u16_row,
  };

  // The sinker layer's `HOST_NATIVE_BE` mirrors `cfg!(target_endian = "big")`.
  // Compute it locally so the test asserts the same condition without taking
  // a dependency on a private const.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

  // Width 33 covers SIMD main loop + scalar tail across every backend.
  let w = 33usize;
  let f32_inputs = [0.0f32, 0.5, 1.0, 1.75, -0.25];
  let pix: std::vec::Vec<half::f16> = (0..w * 3)
    .map(|i| half::f16::from_f32(f32_inputs[i % f32_inputs.len()]))
    .collect();

  // u8 RGB.
  let mut rgb_false = std::vec![0u8; w * 3];
  let mut rgb_host = std::vec![0u8; w * 3];
  rgbf16_to_rgb_row::<false>(&pix, &mut rgb_false, w, true);
  rgbf16_to_rgb_row::<HOST_NATIVE_BE>(&pix, &mut rgb_host, w, true);
  assert_eq!(rgb_false, rgb_host, "u8 RGB diverges");

  // u8 RGBA.
  let mut rgba_false = std::vec![0u8; w * 4];
  let mut rgba_host = std::vec![0u8; w * 4];
  rgbf16_to_rgba_row::<false>(&pix, &mut rgba_false, w, true);
  rgbf16_to_rgba_row::<HOST_NATIVE_BE>(&pix, &mut rgba_host, w, true);
  assert_eq!(rgba_false, rgba_host, "u8 RGBA diverges");

  // u16 RGB.
  let mut rgb_u16_false = std::vec![0u16; w * 3];
  let mut rgb_u16_host = std::vec![0u16; w * 3];
  rgbf16_to_rgb_u16_row::<false>(&pix, &mut rgb_u16_false, w, true);
  rgbf16_to_rgb_u16_row::<HOST_NATIVE_BE>(&pix, &mut rgb_u16_host, w, true);
  assert_eq!(rgb_u16_false, rgb_u16_host, "u16 RGB diverges");

  // u16 RGBA.
  let mut rgba_u16_false = std::vec![0u16; w * 4];
  let mut rgba_u16_host = std::vec![0u16; w * 4];
  rgbf16_to_rgba_u16_row::<false>(&pix, &mut rgba_u16_false, w, true);
  rgbf16_to_rgba_u16_row::<HOST_NATIVE_BE>(&pix, &mut rgba_u16_host, w, true);
  assert_eq!(rgba_u16_false, rgba_u16_host, "u16 RGBA diverges");

  // f16 lossless pass-through.
  let mut f16_false = std::vec![half::f16::ZERO; w * 3];
  let mut f16_host = std::vec![half::f16::ZERO; w * 3];
  rgbf16_to_rgb_f16_row::<false>(&pix, &mut f16_false, w, true);
  rgbf16_to_rgb_f16_row::<HOST_NATIVE_BE>(&pix, &mut f16_host, w, true);
  assert_eq!(f16_false, f16_host, "f16 RGB diverges");
  if !HOST_NATIVE_BE {
    assert_eq!(
      f16_host, pix,
      "f16 lossless pass-through corrupted on LE host"
    );
  }

  // f32 lossless widen.
  let mut f32_false = std::vec![0.0f32; w * 3];
  let mut f32_host = std::vec![0.0f32; w * 3];
  rgbf16_to_rgb_f32_row::<false>(&pix, &mut f32_false, w, true);
  rgbf16_to_rgb_f32_row::<HOST_NATIVE_BE>(&pix, &mut f32_host, w, true);
  assert_eq!(f32_false, f32_host, "f32 widen diverges");
}

/// End-to-end sinker contract test: feeding host-native `half::f16` through
/// [`MixedSinker<Rgbf16>`] must round-trip the f16 input bit-exact via
/// `with_rgb_f16` on every host. Documents the public-API contract that the
/// [`HOST_NATIVE_BE`] routing fix preserves. Pairs with the kernel-level
/// test above; together they cover both the dispatch boundary and the public
/// sinker boundary.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf16_sinker_host_native_contract_lossless_passthrough() {
  let vals_f32 = [0.5f32, 1.5, -0.25, 100.0];
  let pix: std::vec::Vec<half::f16> = (0..16 * 4 * 3)
    .map(|i| half::f16::from_f32(vals_f32[i % vals_f32.len()]))
    .collect();
  let src = Rgbf16Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_f16_out = std::vec![half::f16::ZERO; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf16>::new(16, 4)
    .with_rgb_f16(&mut rgb_f16_out)
    .unwrap();
  rgbf16_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // Bit-exact pass-through on every host — broken `::<false>` routing
  // would byte-swap on a BE host; the fixed routing keeps the f16 intact.
  assert_eq!(rgb_f16_out, pix, "Rgbf16 sinker f16 pass-through corrupted");
}
