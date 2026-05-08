use super::*;

// ---- Tier 9 — Rgbf32 packed-float-RGB source family ---------------------

/// Builds a row-padded Rgbf32 frame with a constant `(R, G, B)` triple.
fn solid_rgbf32_frame(width: u32, height: u32, r: f32, g: f32, b: f32) -> Vec<f32> {
  let w = width as usize;
  let h = height as usize;
  let mut buf = std::vec![0.0f32; w * h * 3];
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
fn rgbf32_with_rgb_clamps_to_u8() {
  // 0.5 → 128 (round-half-even at 127.5), 1.0 → 255, 2.0 → 255 (HDR
  // clamp), -0.5 → 0 (negative clamp).
  let pix = solid_rgbf32_frame(16, 4, 1.0, 2.0, -0.5);
  let src = Rgbf32Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u8; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_rgb(&mut rgb_out)
    .unwrap();
  rgbf32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  for px in rgb_out.chunks(3) {
    assert_eq!(px, [255, 255, 0]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_rgb_u16_clamps_to_u16() {
  let pix = solid_rgbf32_frame(16, 4, 0.5, 1.0, 1.5);
  let src = Rgbf32Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0u16; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_rgb_u16(&mut rgb_out)
    .unwrap();
  rgbf32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // 0.5 * 65535 = 32767.5 → 32768 (round-half-even); 1.0 → 65535;
  // 1.5 → 65535 (clamp).
  for px in rgb_out.chunks(3) {
    assert_eq!(px, [32768, 65535, 65535]);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_rgb_f32_passes_through_lossless() {
  // Includes HDR (> 1.0), negatives, and exact integer values to
  // confirm bit-exact pass-through.
  let mut pix = std::vec![0.0f32; 16 * 4 * 3];
  for (i, v) in pix.iter_mut().enumerate() {
    // Build a deterministic mix: HDR, negative, in-range.
    *v = match i % 6 {
      0 => 0.0,
      1 => 1.0,
      2 => -0.25,
      3 => 1.5,
      4 => 0.5,
      _ => 100.0,
    };
  }
  let src = Rgbf32Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_out = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_rgb_f32(&mut rgb_out)
    .unwrap();
  rgbf32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // Bit-exact equality (no rounding, no clamping in the f32 path).
  assert_eq!(rgb_out, pix);
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_luma_u8() {
  // Constant white → BT.709 luma 235 (limited range) or 255 (full range).
  let pix = solid_rgbf32_frame(16, 4, 1.0, 1.0, 1.0);
  let src = Rgbf32Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u8; 16 * 4];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_luma(&mut luma_out)
    .unwrap();
  rgbf32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // Full-range BT.709: white maps to 255.
  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_luma_u16() {
  let pix = solid_rgbf32_frame(16, 4, 1.0, 1.0, 1.0);
  let src = Rgbf32Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut luma_out = std::vec![0u16; 16 * 4];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_luma_u16(&mut luma_out)
    .unwrap();
  rgbf32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // u8 luma 255 → u16 255 (zero-extended, matching the packed-YUV
  // luma_u16 convention).
  for &y in &luma_out {
    assert_eq!(y, 255);
  }
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_with_hsv() {
  // Pure red → H=0, S=255, V=255 in the OpenCV 8-bit HSV encoding.
  let pix = solid_rgbf32_frame(16, 4, 1.0, 0.0, 0.0);
  let src = Rgbf32Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let n = 16 * 4;
  let mut h_out = std::vec![0u8; n];
  let mut s_out = std::vec![0u8; n];
  let mut v_out = std::vec![0u8; n];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_hsv(&mut h_out, &mut s_out, &mut v_out)
    .unwrap();
  rgbf32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

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
fn rgbf32_simd_matches_scalar_with_random_input() {
  // Width 1921 forces both SIMD main loop and scalar tail across
  // every backend block size.
  let w = 1921usize;
  let h = 4usize;
  let n_lanes = w * h * 3;
  // Mix of in-range, HDR, and negative values to exercise all branches.
  let mut pix = std::vec![0.0f32; n_lanes];
  let mut state: u32 = 0xDEAD_BEEF;
  for (i, v) in pix.iter_mut().enumerate() {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *v = match (state >> 28) & 0b11 {
      0 => ((state >> 8) & 0xFF) as f32 / 255.0,
      1 => (((i as u32 & 0x7F) as f32) + 0.5) / 255.0,
      2 => 1.0 + ((state >> 16) & 0xF) as f32 * 0.25,
      _ => -(((state >> 4) & 0xFF) as f32) / 255.0,
    };
  }
  let src = Rgbf32Frame::try_new(&pix, w as u32, h as u32, (w * 3) as u32).unwrap();

  let mut rgb_simd = std::vec![0u8; w * h * 3];
  let mut rgb_scalar = std::vec![0u8; w * h * 3];
  let mut rgba_simd = std::vec![0u8; w * h * 4];
  let mut rgba_scalar = std::vec![0u8; w * h * 4];
  let mut rgb_u16_simd = std::vec![0u16; w * h * 3];
  let mut rgb_u16_scalar = std::vec![0u16; w * h * 3];
  let mut rgba_u16_simd = std::vec![0u16; w * h * 4];
  let mut rgba_u16_scalar = std::vec![0u16; w * h * 4];
  let mut rgb_f32_simd = std::vec![0.0f32; w * h * 3];
  let mut rgb_f32_scalar = std::vec![0.0f32; w * h * 3];
  let mut luma_simd = std::vec![0u8; w * h];
  let mut luma_scalar = std::vec![0u8; w * h];
  let mut luma_u16_simd = std::vec![0u16; w * h];
  let mut luma_u16_scalar = std::vec![0u16; w * h];

  let mut s_simd = MixedSinker::<Rgbf32>::new(w, h)
    .with_rgb(&mut rgb_simd)
    .unwrap()
    .with_rgba(&mut rgba_simd)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_simd)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_simd)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_simd)
    .unwrap()
    .with_luma(&mut luma_simd)
    .unwrap()
    .with_luma_u16(&mut luma_u16_simd)
    .unwrap();
  rgbf32_to(&src, true, ColorMatrix::Bt709, &mut s_simd).unwrap();

  let mut s_scalar = MixedSinker::<Rgbf32>::new(w, h)
    .with_rgb(&mut rgb_scalar)
    .unwrap()
    .with_rgba(&mut rgba_scalar)
    .unwrap()
    .with_rgb_u16(&mut rgb_u16_scalar)
    .unwrap()
    .with_rgba_u16(&mut rgba_u16_scalar)
    .unwrap()
    .with_rgb_f32(&mut rgb_f32_scalar)
    .unwrap()
    .with_luma(&mut luma_scalar)
    .unwrap()
    .with_luma_u16(&mut luma_u16_scalar)
    .unwrap();
  s_scalar.set_simd(false);
  rgbf32_to(&src, true, ColorMatrix::Bt709, &mut s_scalar).unwrap();

  assert_eq!(rgb_simd, rgb_scalar, "RGB output diverges");
  assert_eq!(rgba_simd, rgba_scalar, "RGBA output diverges");
  assert_eq!(rgb_u16_simd, rgb_u16_scalar, "RGB u16 output diverges");
  assert_eq!(rgba_u16_simd, rgba_u16_scalar, "RGBA u16 output diverges");
  assert_eq!(rgb_f32_simd, rgb_f32_scalar, "RGB f32 output diverges");
  assert_eq!(luma_simd, luma_scalar, "Luma output diverges");
  assert_eq!(luma_u16_simd, luma_u16_scalar, "Luma u16 output diverges");
  assert_eq!(rgb_f32_simd, pix, "RGB f32 output is not lossless");
}

/// Sinker-layer host-native-`f32` regression for the bug fixed alongside
/// `c3a6478` (PR #83 codex 2nd-pass review): the [`Rgbf32`] sinker used to
/// hardcode `::<false>` when calling the row dispatchers, telling them to
/// "decode LE-encoded input". Because [`Rgbf32Frame`] hands us a host-native
/// `&[f32]` row, that routing was a no-op on LE hosts but corrupted every
/// output path on BE hosts (the loaders would byte-swap an already-decoded
/// f32). The fix replaces those `::<false>` with `::<HOST_NATIVE_BE>`, which
/// is `false` on LE and `true` on BE — a no-op byte-swap on either host.
///
/// On a LE host (the only target Apple-Silicon and x86_64 macOS can run),
/// `HOST_NATIVE_BE = false` and `::<HOST_NATIVE_BE>` is byte-for-byte
/// identical to `::<false>`, so this test cannot distinguish the broken vs
/// fixed code on LE. It instead documents the equivalence at the **kernel
/// dispatch** layer — calling each `rgbf32_to_*` dispatcher with both
/// `BE = false` and `BE = HOST_NATIVE_BE` (= `cfg!(target_endian = "big")`)
/// must produce identical output on the active host. On a hypothetical BE
/// host (full QEMU s390x coverage is Phase 3), the same equivalence holds
/// for the **fixed** sinker but would fail for the broken one — making this
/// the natural regression test for the routing change.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_kernel_host_native_be_matches_false_on_le_host() {
  use crate::row::{
    rgbf32_to_rgb_f32_row, rgbf32_to_rgb_row, rgbf32_to_rgb_u16_row, rgbf32_to_rgba_row,
    rgbf32_to_rgba_u16_row,
  };

  // The sinker layer's `HOST_NATIVE_BE` mirrors `cfg!(target_endian = "big")`.
  // Compute it locally so the test asserts the same condition without taking
  // a dependency on a private const.
  const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

  // Width 33 covers SIMD main loop + scalar tail across every backend.
  let w = 33usize;
  let mut pix = std::vec![0.0f32; w * 3];
  for (i, v) in pix.iter_mut().enumerate() {
    // Mix in-range, HDR, and negative values to exercise every clamp branch.
    *v = match i % 5 {
      0 => 0.0,
      1 => 0.5,
      2 => 1.0,
      3 => 1.75,
      _ => -0.25,
    };
  }

  // u8 RGB.
  let mut rgb_false = std::vec![0u8; w * 3];
  let mut rgb_host = std::vec![0u8; w * 3];
  rgbf32_to_rgb_row::<false>(&pix, &mut rgb_false, w, true);
  rgbf32_to_rgb_row::<HOST_NATIVE_BE>(&pix, &mut rgb_host, w, true);
  assert_eq!(rgb_false, rgb_host, "u8 RGB diverges");

  // u8 RGBA.
  let mut rgba_false = std::vec![0u8; w * 4];
  let mut rgba_host = std::vec![0u8; w * 4];
  rgbf32_to_rgba_row::<false>(&pix, &mut rgba_false, w, true);
  rgbf32_to_rgba_row::<HOST_NATIVE_BE>(&pix, &mut rgba_host, w, true);
  assert_eq!(rgba_false, rgba_host, "u8 RGBA diverges");

  // u16 RGB.
  let mut rgb_u16_false = std::vec![0u16; w * 3];
  let mut rgb_u16_host = std::vec![0u16; w * 3];
  rgbf32_to_rgb_u16_row::<false>(&pix, &mut rgb_u16_false, w, true);
  rgbf32_to_rgb_u16_row::<HOST_NATIVE_BE>(&pix, &mut rgb_u16_host, w, true);
  assert_eq!(rgb_u16_false, rgb_u16_host, "u16 RGB diverges");

  // u16 RGBA.
  let mut rgba_u16_false = std::vec![0u16; w * 4];
  let mut rgba_u16_host = std::vec![0u16; w * 4];
  rgbf32_to_rgba_u16_row::<false>(&pix, &mut rgba_u16_false, w, true);
  rgbf32_to_rgba_u16_row::<HOST_NATIVE_BE>(&pix, &mut rgba_u16_host, w, true);
  assert_eq!(rgba_u16_false, rgba_u16_host, "u16 RGBA diverges");

  // f32 lossless pass-through.
  let mut f32_false = std::vec![0.0f32; w * 3];
  let mut f32_host = std::vec![0.0f32; w * 3];
  rgbf32_to_rgb_f32_row::<false>(&pix, &mut f32_false, w, true);
  rgbf32_to_rgb_f32_row::<HOST_NATIVE_BE>(&pix, &mut f32_host, w, true);
  assert_eq!(f32_false, f32_host, "f32 RGB diverges");
  // And on the host (LE on every CI runner) both must equal `pix` bit-exact.
  if !HOST_NATIVE_BE {
    assert_eq!(
      f32_host, pix,
      "f32 lossless pass-through corrupted on LE host"
    );
  }
}

/// End-to-end sinker contract test: feeding host-native `f32` through
/// [`MixedSinker<Rgbf32>`] must produce the same output every other sinker
/// would expect from a host-native source — specifically, `with_rgb_f32`
/// must be bit-exact identical to the input on every host. Documents the
/// public-API contract that the [`HOST_NATIVE_BE`] routing fix preserves.
/// Pairs with the kernel-level test above; together they cover both the
/// dispatch boundary and the public sinker boundary.
#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn rgbf32_sinker_host_native_contract_lossless_passthrough() {
  // Mix HDR, in-range, and negative values — the f32 lossless path must
  // round-trip them bit-exact on every host.
  let mut pix = std::vec![0.0f32; 16 * 4 * 3];
  for (i, v) in pix.iter_mut().enumerate() {
    *v = match i % 4 {
      0 => 0.5,
      1 => 1.5,
      2 => -0.25,
      _ => 100.0,
    };
  }
  let src = Rgbf32Frame::try_new(&pix, 16, 4, 16 * 3).unwrap();

  let mut rgb_f32_out = std::vec![0.0f32; 16 * 4 * 3];
  let mut sink = MixedSinker::<Rgbf32>::new(16, 4)
    .with_rgb_f32(&mut rgb_f32_out)
    .unwrap();
  rgbf32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();

  // Bit-exact pass-through on every host. On the buggy `::<false>` routing
  // a BE host would see byte-swapped output here; on the fixed routing the
  // assertion holds on both LE and BE.
  assert_eq!(rgb_f32_out, pix, "Rgbf32 sinker f32 pass-through corrupted");
}
