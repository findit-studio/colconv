//! Fused-downscale coverage for `Gray32` — the full-bit (u32) integer twin of
//! `Gray16`. The wire row converts to a host-native u16 luma plane first via
//! the same `>> 16` narrow the direct `luma_u16` path uses (u16 is the widest
//! depth colconv emits), then a single 1-channel `AreaStream<u16>` /
//! `FilterStream<u16>` resamples that plane at u16 precision and every attached
//! output derives from each finalized u16 luma row using the **Gray16** derive
//! kernels — once narrowed to u16 a Gray32 sample behaves exactly like a
//! Gray16 one. So every resampled output equals the direct Gray32 sink run over
//! a frame that already holds the `(>> 16)`-narrowed-then-resampled luma
//! (widened back to `u32` via `<< 16`).

use crate::{
  ColorMatrix, PixelSink,
  frame::Gray32Frame,
  resample::{AreaResampler, ResampleError},
  sinker::{MixedSinker, MixedSinkerError},
  source::{Gray32, Gray32Row, gray32_to, gray32_to_endian},
};

const SRC: usize = 8;
const OUT: usize = 4;
const FR: bool = true;
const M: ColorMatrix = ColorMatrix::Bt709;

/// Re-encode a host-native u32 slice as LE-encoded byte storage (the
/// `gray32le` plane contract), recovered via `u32::from_le`.
fn as_le_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u32 slice as BE-encoded byte storage (the
/// `gray32be` plane contract), recovered via `u32::from_be`.
fn as_be_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// A 32-bit luma ramp with a non-trivial low half so the `>> 16` narrow and the
/// u16-precision area mean both see real per-2x2-block variation.
fn ramp() -> Vec<u32> {
  let mut y = vec![0u32; SRC * SRC];
  for (i, p) in y.iter_mut().enumerate() {
    *p = ((i as u32).wrapping_mul(50_529_027)) ^ 0x0F0F_0F0F;
  }
  y
}

/// The host-native u16 luma plane the Gray32 resample bins: each sample is the
/// source `u32 >> 16` narrow (the direct `luma_u16` semantics).
fn narrowed_u16(plane: &[u32]) -> Vec<u16> {
  plane.iter().map(|&v| (v >> 16) as u16).collect()
}

/// Exact 2x2-block area mean (round-half-up) of the narrowed u16 plane.
fn block_mean_2x2(narrowed: &[u16]) -> Vec<u16> {
  let mut out = vec![0u16; OUT * OUT];
  for oy in 0..OUT {
    for ox in 0..OUT {
      let mut s = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          s += narrowed[(oy * 2 + dy) * SRC + ox * 2 + dx] as u32;
        }
      }
      out[oy * OUT + ox] = ((s + 2) / 4) as u16;
    }
  }
  out
}

/// Widen a u16 luma plane back to a `u32` Gray32 plane (`<< 16`) so the direct
/// Gray32 sink over it reproduces the resample path's per-output derivation:
/// `luma_u16 = v`, `luma = v >> 8`, native broadcast `= v`, u8 `= v >> 8`.
fn widen_to_gray32(binned: &[u16]) -> Vec<u32> {
  binned.iter().map(|&v| (v as u32) << 16).collect()
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_downscale_luma_u16_is_exact_area_mean() {
  let plane = ramp();
  let pix = as_le_u32(&plane);
  let src = Gray32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut luma_u16 = vec![0u16; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap();
    gray32_to(&src, FR, M, &mut sink).unwrap();
  }
  assert_eq!(
    luma_u16,
    block_mean_2x2(&narrowed_u16(&plane)),
    "luma_u16 must be the exact 2x2 block mean of the (>> 16)-narrowed luma"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_all_outputs_match_direct_over_binned_luma() {
  // Every attached output must equal the direct Gray32 sink over the binned
  // luma, widened back to a `u32` frame (`<< 16`) so the direct path's `>> 16`
  // narrow recovers the binned u16 — the resample contract.
  let plane = ramp();
  let pix = as_le_u32(&plane);
  let src = Gray32Frame::new(&pix, SRC as u32, SRC as u32, SRC as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Gray32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb(&mut rgb)
        .unwrap()
        .with_rgba(&mut rgba)
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap()
        .with_luma(&mut luma)
        .unwrap()
        .with_luma_u16(&mut luma_u16)
        .unwrap()
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    gray32_to(&src, FR, M, &mut sink).unwrap();
  }

  let binned = block_mean_2x2(&narrowed_u16(&plane));
  let binned_pix = as_le_u32(&widen_to_gray32(&binned));
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned_frame = Gray32Frame::new(&binned_pix, OUT as u32, OUT as u32, OUT as u32);
    let mut sink = MixedSinker::<Gray32>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgb_u16(&mut ref_rgb_u16)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    gray32_to(&binned_frame, FR, M, &mut sink).unwrap();
  }
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgb_u16, ref_rgb_u16, "rgb_u16");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_le_be_resample_outputs_identical() {
  // The binned luma is host-native, so the derive kernels must run with
  // `HOST_NATIVE_BE`. The LE-vs-BE parity catches a wrong const on either host:
  // LE and BE wire encodings of the same logical plane must resample identically.
  let plane = ramp();
  let pix_le = as_le_u32(&plane);
  let pix_be = as_be_u32(&plane);

  let mut le_luma_u16 = vec![0u16; OUT * OUT];
  let mut le_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut le_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Gray32Frame::new(&pix_le, SRC as u32, SRC as u32, SRC as u32);
    let mut sink =
      MixedSinker::<Gray32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_luma_u16(&mut le_luma_u16)
        .unwrap()
        .with_rgb_u16(&mut le_rgb_u16)
        .unwrap()
        .with_rgba(&mut le_rgba)
        .unwrap();
    gray32_to(&frame, FR, M, &mut sink).unwrap();
  }

  let mut be_luma_u16 = vec![0u16; OUT * OUT];
  let mut be_rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut be_rgba = vec![0u8; OUT * OUT * 4];
  {
    let frame = Gray32Frame::<true>::new(&pix_be, SRC as u32, SRC as u32, SRC as u32);
    let mut sink = MixedSinker::<Gray32<true>, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_luma_u16(&mut be_luma_u16)
    .unwrap()
    .with_rgb_u16(&mut be_rgb_u16)
    .unwrap()
    .with_rgba(&mut be_rgba)
    .unwrap();
    gray32_to_endian::<_, true>(&frame, FR, M, &mut sink).unwrap();
  }

  assert_eq!(le_luma_u16, be_luma_u16, "luma_u16 LE/BE diverge");
  assert_eq!(le_rgb_u16, be_rgb_u16, "rgb_u16 LE/BE diverge");
  assert_eq!(le_rgba, be_rgba, "rgba LE/BE diverge");
  assert_eq!(
    le_luma_u16,
    block_mean_2x2(&narrowed_u16(&plane)),
    "luma_u16 not area mean"
  );
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_out_of_sequence_first_row_rejected_before_allocation() {
  // Feeding row 1 before row 0 is rejected with OutOfSequenceRow, before any
  // stream allocation — the atomic-preflight contract shared with Gray16.
  let pix = as_le_u32(&ramp());
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut sink =
    MixedSinker::<Gray32, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap();
  sink.begin_frame(SRC as u32, SRC as u32).unwrap();
  let g = &pix[SRC..2 * SRC];
  let err = sink.process(Gray32Row::new(g, 1, M, FR)).unwrap_err();
  assert!(
    matches!(
      err,
      MixedSinkerError::Resample(ResampleError::OutOfSequenceRow(_))
    ),
    "expected OutOfSequenceRow, got {err:?}"
  );
}

// ---- Filter-plan routing ----------------------------------------------------
//
// The filter path narrows the wire row to a host-native u16 luma plane (the
// same `>> 16` narrow the area path uses) and resamples it through the
// signed-coefficient single-channel `FilterStream<u16>` (the filter twin of the
// area bin). Gray32 carries the full native u16 range after the narrow, so the
// `FilterStream`'s `0..=65535` clamp *is* the native clamp — no extra clamp. So
// the filter `luma_u16` must equal a single-channel `FilterStream<u16>`
// resample of the narrowed source plane value for value.

use crate::resample::{
  CatmullRom, FilterKernel, FilterStream, FilteredResampler, Lanczos3, Resampler, Triangle,
};

const FW: usize = 8;
const FH: usize = 8;
const FOUT_DOWN: usize = 4;
const FUP: usize = 7;

/// A u32 ramp with a hard mid-column edge plus two textured interior rows so
/// filter windows straddling the edge produce real intermediate values,
/// exercising the low half the `>> 16` narrow and the u16 stream must carry.
fn filter_ramp() -> Vec<u32> {
  let mut y = vec![0u32; FW * FH];
  for row in 0..FH {
    for col in 0..FW {
      y[row * FW + col] = if col < FW / 2 {
        0x1234_5678
      } else {
        0xC0DE_4321
      };
    }
  }
  for col in 0..FW {
    y[4 * FW + col] = (col as u32 * 8000) << 16;
    y[5 * FW + col] = (60000 - col as u32 * 8000) << 16;
  }
  y
}

/// Single-channel filter resample of the narrowed host-native u16 luma plane —
/// the Gray32 luma oracle (full native u16 range, so the engine clamp is native).
fn native_luma_filter<K: FilterKernel>(
  kernel: K,
  luma_plane: &[u16],
  sw: usize,
  sh: usize,
  ow: usize,
  oh: usize,
) -> Vec<u16> {
  let plan = FilteredResampler::new(ow, oh, kernel)
    .plan(sw, sh)
    .expect("valid filter plan")
    .expect("non-identity");
  let fh = plan.filter_h().expect("h windows");
  let fv = plan.filter_v().expect("v windows");
  let mut stream = FilterStream::<u16>::new(fh, fv, sw, sh, 1).expect("geometry");
  let mut out = vec![0u16; ow * oh];
  for row in 0..sh {
    stream
      .feed_row(
        row,
        &luma_plane[row * sw..(row + 1) * sw],
        true,
        |oy, fin| {
          out[oy * ow..(oy + 1) * ow].copy_from_slice(fin);
        },
      )
      .expect("rows in order");
  }
  out
}

fn gray32_filter_luma_u16<K: FilterKernel + Copy>(ow: usize, oh: usize, kernel: K) -> Vec<u16> {
  let plane = filter_ramp();
  let pix = as_le_u32(&plane);
  let src = Gray32Frame::new(&pix, FW as u32, FH as u32, FW as u32);
  let mut luma_u16 = vec![0u16; ow * oh];
  let mut rgb = vec![0u8; ow * oh * 3];
  let mut luma = vec![0u8; ow * oh];
  {
    let mut sink = MixedSinker::<Gray32, FilteredResampler<K>>::with_resampler(
      FW,
      FH,
      FilteredResampler::new(ow, oh, kernel),
    )
    .unwrap()
    .with_luma_u16(&mut luma_u16)
    .unwrap()
    .with_luma(&mut luma)
    .unwrap()
    .with_rgb(&mut rgb)
    .unwrap();
    gray32_to(&src, FR, M, &mut sink).unwrap();
  }
  // Derived outputs follow from the resampled luma exactly as the Gray16
  // derive kernels do: luma = luma_u16 >> 8, rgb broadcasts that byte.
  for (i, &lv) in luma_u16.iter().enumerate() {
    assert_eq!(
      luma[i],
      (lv >> 8) as u8,
      "luma must be luma_u16 >> 8 at {i}"
    );
    assert_eq!(rgb[i * 3], (lv >> 8) as u8, "rgb broadcasts luma_u16 >> 8");
  }
  luma_u16
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_filter_luma_u16_matches_native_oracle() {
  let narrowed = narrowed_u16(&filter_ramp());
  for (ow, oh) in [(FOUT_DOWN, FOUT_DOWN), (FUP, FUP)] {
    let tri = gray32_filter_luma_u16(ow, oh, Triangle);
    assert_eq!(
      tri,
      native_luma_filter(Triangle, &narrowed, FW, FH, ow, oh),
      "Triangle {ow}x{oh}"
    );
    let cat = gray32_filter_luma_u16(ow, oh, CatmullRom);
    assert_eq!(
      cat,
      native_luma_filter(CatmullRom, &narrowed, FW, FH, ow, oh),
      "CatmullRom {ow}x{oh}"
    );
    let lan = gray32_filter_luma_u16(ow, oh, Lanczos3);
    assert_eq!(
      lan,
      native_luma_filter(Lanczos3, &narrowed, FW, FH, ow, oh),
      "Lanczos3 {ow}x{oh}"
    );
  }
}

// ---- Limited-range (full_range=false) narrow-first behavior pin -------------
//
// This pins the accepted within-1-LSB narrow-first behavior of the Gray32
// *limited-range* resample per issue #289 — NOT a parity-vs-direct oracle.

/// Run a Gray32 `full_range = false` resample of a constant source through
/// `resampler` (BE flag `BE`), attaching every output, and assert the pinned
/// narrow-first values. A constant source makes binning trivial — the area
/// mean and a normalized filter of a constant are the constant — so area and
/// filter, LE and BE all yield identical, pinnable outputs.
fn pin_narrow_first<R: Resampler, const BE: bool>(plane: &[u32], resampler: R, label: &str) {
  const PSRC: usize = 8;
  const POUT: usize = 4;
  let frame = Gray32Frame::<BE>::new(plane, PSRC as u32, PSRC as u32, PSRC as u32);
  let mut rgb = vec![0u8; POUT * POUT * 3];
  let mut rgb_u16 = vec![0u16; POUT * POUT * 3];
  let mut rgba = vec![0u8; POUT * POUT * 4];
  let mut rgba_u16 = vec![0u16; POUT * POUT * 4];
  let mut luma = vec![0u8; POUT * POUT];
  let mut luma_u16 = vec![0u16; POUT * POUT];
  let mut h = vec![0u8; POUT * POUT];
  let mut s = vec![0u8; POUT * POUT];
  let mut v = vec![0u8; POUT * POUT];
  {
    let mut sink = MixedSinker::<Gray32<BE>, R>::with_resampler(PSRC, PSRC, resampler)
      .unwrap()
      .with_rgb(&mut rgb)
      .unwrap()
      .with_rgb_u16(&mut rgb_u16)
      .unwrap()
      .with_rgba(&mut rgba)
      .unwrap()
      .with_rgba_u16(&mut rgba_u16)
      .unwrap()
      .with_luma(&mut luma)
      .unwrap()
      .with_luma_u16(&mut luma_u16)
      .unwrap()
      .with_hsv(&mut h, &mut s, &mut v)
      .unwrap();
    // full_range = false — the limited-range narrow-first path under test.
    gray32_to_endian::<_, BE>(&frame, false, M, &mut sink).unwrap();
  }
  // Source 0x106dedee: narrow `>> 16` = 0x106d = 4205. The narrow-first
  // limited path yields rgb_u16 = limited16(4205) = 127, rgb_u8 = 0; the
  // 0-ULP direct Gray32 limited path would yield 129 / 1 (the <=1 LSB gap
  // accepted in #289). Luma is unrescaled: luma_u16 = 4205, luma_u8 = 16.
  assert!(rgb.iter().all(|&x| x == 0), "{label} rgb");
  assert!(rgb_u16.iter().all(|&x| x == 127), "{label} rgb_u16");
  for (i, &x) in rgba.iter().enumerate() {
    assert_eq!(x, if i % 4 == 3 { 0xFF } else { 0 }, "{label} rgba[{i}]");
  }
  for (i, &x) in rgba_u16.iter().enumerate() {
    assert_eq!(
      x,
      if i % 4 == 3 { 0xFFFF } else { 127 },
      "{label} rgba_u16[{i}]"
    );
  }
  assert!(luma.iter().all(|&x| x == 16), "{label} luma");
  assert!(luma_u16.iter().all(|&x| x == 4205), "{label} luma_u16");
  assert!(h.iter().all(|&x| x == 0), "{label} h");
  assert!(s.iter().all(|&x| x == 0), "{label} s");
  assert!(v.iter().all(|&x| x == 0), "{label} v");
}

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn gray32_limited_resample_pins_narrow_first_behavior() {
  // Constant source with nonzero low 16 bits near a rounding threshold.
  const V: u32 = 0x106d_edee;
  let le = as_le_u32(&vec![V; 8 * 8]);
  let be = as_be_u32(&vec![V; 8 * 8]);
  // Area engine, LE + BE.
  pin_narrow_first::<_, false>(&le, AreaResampler::to(4, 4), "area LE");
  pin_narrow_first::<_, true>(&be, AreaResampler::to(4, 4), "area BE");
  // Filter engine (Triangle), LE + BE.
  pin_narrow_first::<_, false>(&le, FilteredResampler::new(4, 4, Triangle), "filter LE");
  pin_narrow_first::<_, true>(&be, FilteredResampler::new(4, 4, Triangle), "filter BE");
}
