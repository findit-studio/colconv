//! Fused-downscale coverage for the 32-bit packed RGB family (`Rgb96`): the
//! wire u32 row converts to source-width host u16 RGB (the `>> 16` narrow),
//! binning runs at native 16-bit depth, the native-depth `rgb_u16` /
//! `rgba_u16` outputs are exact area means of the narrowed source, and the
//! u8 / `luma_u16` outputs derive from a single further `>> 8` narrow — the
//! same source-of-truth ordering the direct path uses (net `>> 24` for u8).

use crate::{
  ColorMatrix,
  resample::AreaResampler,
  sinker::MixedSinker,
  source::{Rgb48, Rgb96, Rgba128, rgb48_to, rgb96_to, rgba128_to},
};
use mediaframe::frame::{Rgb48Frame, Rgb96Frame, Rgba128Frame};

const SRC: usize = 8;
const OUT: usize = 4;

/// Re-encode a host-native u32 slice as LE-wire byte storage.
fn as_le_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Re-encode a host-native u16 slice as LE-wire byte storage.
fn as_le_u16(host: &[u16]) -> Vec<u16> {
  host
    .iter()
    .map(|v| u16::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

/// Per-channel full-range u32 ramps; the low 16 bits vary too so the `>> 16`
/// staging is genuinely lossy (matching the format contract).
fn packed_frame_u32() -> Vec<u32> {
  let mut buf = vec![0u32; SRC * SRC * 3];
  for (i, px) in buf.chunks_exact_mut(3).enumerate() {
    px[0] = 0x2000_0000 + (i as u32) * 0x0123_4567;
    px[1] = 0xC000_0000u32.wrapping_sub((i as u32) * 0x0098_7654);
    px[2] = 0x1000_0000 + ((i % 8) as u32) * 0x0555_5555;
  }
  buf
}

/// The host-native u16 RGB the sinker stages: each u32 narrowed `>> 16`.
fn staged_u16(host: &[u32]) -> Vec<u16> {
  host.iter().map(|&v| (v >> 16) as u16).collect()
}

/// Exact 2x2 block mean with round-half-up over the staged u16 values.
fn expected_block_mean_u16(staged: &[u16], ox: usize, oy: usize, c: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += staged[((oy * 2 + dy) * SRC + ox * 2 + dx) * 3 + c] as u64;
    }
  }
  ((acc + 2) / 4) as u16
}

#[test]
fn rgb96_downscale_rgb_u16_is_exact_area_mean() {
  let host = packed_frame_u32();
  let staged = staged_u16(&host);
  let wire = as_le_u32(&host);
  let src = Rgb96Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  {
    let mut sink =
      MixedSinker::<Rgb96, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgb_u16(&mut rgb_u16)
        .unwrap();
    rgb96_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..3 {
        assert_eq!(
          rgb_u16[(oy * OUT + ox) * 3 + c],
          expected_block_mean_u16(&staged, ox, oy, c),
          "({ox},{oy}) c{c}"
        );
      }
    }
  }
}

#[test]
fn rgb96_derived_outputs_come_from_binned_rgb() {
  // Every attached output — native-depth u16 and narrowed u8 — must equal what
  // a direct full-res Rgb48 sink produces over the (exact) binned u16 RGB: once
  // staged to u16, Rgb96 shares the identical binning + derivation engine, so
  // the binned u16 is the single source of truth for the narrowed outputs.
  let host = packed_frame_u32();
  let wire = as_le_u32(&host);
  let src = Rgb96Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut rgb = vec![0u8; OUT * OUT * 3];
  let mut rgb_u16 = vec![0u16; OUT * OUT * 3];
  let mut rgba = vec![0u8; OUT * OUT * 4];
  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut luma = vec![0u8; OUT * OUT];
  let mut luma_u16 = vec![0u16; OUT * OUT];
  let mut h = vec![0u8; OUT * OUT];
  let mut s_ = vec![0u8; OUT * OUT];
  let mut v_ = vec![0u8; OUT * OUT];
  {
    let mut sink =
      MixedSinker::<Rgb96, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
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
        .with_hsv(&mut h, &mut s_, &mut v_)
        .unwrap();
    rgb96_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }

  // Reference: the full-res Rgb48 sink over the (exact) binned u16 RGB.
  let binned_wire = as_le_u16(&rgb_u16);
  let mut ref_rgb = vec![0u8; OUT * OUT * 3];
  let mut ref_rgba = vec![0u8; OUT * OUT * 4];
  let mut ref_rgba_u16 = vec![0u16; OUT * OUT * 4];
  let mut ref_luma = vec![0u8; OUT * OUT];
  let mut ref_luma_u16 = vec![0u16; OUT * OUT];
  let mut ref_h = vec![0u8; OUT * OUT];
  let mut ref_s = vec![0u8; OUT * OUT];
  let mut ref_v = vec![0u8; OUT * OUT];
  {
    let binned = Rgb48Frame::new(&binned_wire, OUT as u32, OUT as u32, (OUT * 3) as u32);
    let mut sink = MixedSinker::<Rgb48>::new(OUT, OUT)
      .with_rgb(&mut ref_rgb)
      .unwrap()
      .with_rgba(&mut ref_rgba)
      .unwrap()
      .with_rgba_u16(&mut ref_rgba_u16)
      .unwrap()
      .with_luma(&mut ref_luma)
      .unwrap()
      .with_luma_u16(&mut ref_luma_u16)
      .unwrap()
      .with_hsv(&mut ref_h, &mut ref_s, &mut ref_v)
      .unwrap();
    rgb48_to(&binned, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(rgb, ref_rgb, "rgb");
  assert_eq!(rgba, ref_rgba, "rgba");
  assert_eq!(rgba_u16, ref_rgba_u16, "rgba_u16");
  assert_eq!(luma, ref_luma, "luma");
  assert_eq!(luma_u16, ref_luma_u16, "luma_u16");
  assert_eq!(h, ref_h, "h");
  assert_eq!(s_, ref_s, "s");
  assert_eq!(v_, ref_v, "v");
}

#[test]
fn rgb96_identity_plan_matches_new_sink() {
  let host = packed_frame_u32();
  let wire = as_le_u32(&host);
  let src = Rgb96Frame::new(&wire, SRC as u32, SRC as u32, (SRC * 3) as u32);

  let mut direct = vec![0u16; SRC * SRC * 3];
  {
    let mut sink = MixedSinker::<Rgb96>::new(SRC, SRC)
      .with_rgb_u16(&mut direct)
      .unwrap();
    rgb96_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 3];
  {
    let mut sink =
      MixedSinker::<Rgb96, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgb_u16(&mut via_area)
        .unwrap();
    rgb96_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

// ---- Rgba128 (alpha-aware area resample) ------------------------------------

/// Full-range u32 RGBA ramps with real per-pixel alpha.
fn packed_rgba_frame_u32() -> Vec<u32> {
  let mut buf = vec![0u32; SRC * SRC * 4];
  for (i, px) in buf.chunks_exact_mut(4).enumerate() {
    px[0] = 0x2000_0000 + (i as u32) * 0x0123_4567;
    px[1] = 0xC000_0000u32.wrapping_sub((i as u32) * 0x0098_7654);
    px[2] = 0x1000_0000 + ((i % 8) as u32) * 0x0555_5555;
    px[3] = 0x3000_0000 + (i as u32) * 0x0222_2222;
  }
  buf
}

/// The host-native u16 RGBA the sinker stages: each u32 narrowed `>> 16`.
fn staged_rgba_u16(host: &[u32]) -> Vec<u16> {
  host.iter().map(|&v| (v >> 16) as u16).collect()
}

/// Exact 2x2 block mean over the staged u16 RGBA (4 channels per pixel).
fn expected_block_mean_rgba_u16(staged: &[u16], ox: usize, oy: usize, c: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += staged[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c] as u64;
    }
  }
  ((acc + 2) / 4) as u16
}

#[test]
fn rgba128_downscale_rgba_u16_is_exact_area_mean_incl_alpha() {
  let host = packed_rgba_frame_u32();
  let staged = staged_rgba_u16(&host);
  let src = Rgba128Frame::new(&host, SRC as u32, SRC as u32, (SRC * 4) as u32);

  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  {
    let mut sink =
      MixedSinker::<Rgba128, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(OUT, OUT))
        .unwrap()
        .with_rgba_u16(&mut rgba_u16)
        .unwrap();
    rgba128_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        assert_eq!(
          rgba_u16[(oy * OUT + ox) * 4 + c],
          expected_block_mean_rgba_u16(&staged, ox, oy, c),
          "({ox},{oy}) c{c} (alpha is c3)"
        );
      }
    }
  }
}

#[test]
fn rgba128_identity_plan_matches_new_sink() {
  let host = packed_rgba_frame_u32();
  let src = Rgba128Frame::new(&host, SRC as u32, SRC as u32, (SRC * 4) as u32);

  let mut direct = vec![0u16; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<Rgba128>::new(SRC, SRC)
      .with_rgba_u16(&mut direct)
      .unwrap();
    rgba128_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 4];
  {
    let mut sink =
      MixedSinker::<Rgba128, AreaResampler>::with_resampler(SRC, SRC, AreaResampler::to(SRC, SRC))
        .unwrap()
        .with_rgba_u16(&mut via_area)
        .unwrap();
    rgba128_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

// ---- full_range = false resample behavior pins (issue #289) -----------------
// These PIN the current narrow-first (u32 `>> 16` before binning) resample
// output — within 1 LSB of an exact u32-domain mean, NOT parity vs a direct
// u32-domain oracle. A single host-native fixture is re-encoded LE / BE so both
// endian arms decode the same logical values and must produce the identical
// pinned output (area: mean-of-narrowed; filter: a captured golden).

use crate::resample::{FilteredResampler, Triangle};
use mediaframe::frame::{Rgb96BeFrame, Rgba128BeFrame};

const FRP: usize = 4; // source side
const FRO: usize = 2; // output side

fn as_be_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// Nonzero low-16-bit u32 RGB ramp so the `>> 16` staging narrow is lossy.
fn frp_rgb96() -> Vec<u32> {
  (0..FRP * FRP * 3)
    .map(|i| (i as u32).wrapping_mul(0x0123_4567).wrapping_add(0xABCD))
    .collect()
}
/// Nonzero low-16-bit u32 RGBA ramp.
fn frp_rgba128() -> Vec<u32> {
  (0..FRP * FRP * 4)
    .map(|i| (i as u32).wrapping_mul(0x0123_4567).wrapping_add(0xBEEF))
    .collect()
}

fn frp_block_mean(staged: &[u16], ox: usize, oy: usize, c: usize, ch: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += staged[((oy * 2 + dy) * FRP + ox * 2 + dx) * ch + c] as u64;
    }
  }
  ((acc + 2) / 4) as u16
}

#[test]
fn rgb96_fr_false_area_rgb_u16_pins_mean_of_narrowed() {
  let intended = frp_rgb96();
  let staged: Vec<u16> = intended.iter().map(|&v| (v >> 16) as u16).collect();
  let mut expected = vec![0u16; FRO * FRO * 3];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..3 {
        expected[(oy * FRO + ox) * 3 + c] = frp_block_mean(&staged, ox, oy, c, 3);
      }
    }
  }
  // LE arm
  let le = as_le_u32(&intended);
  let mut out_le = vec![0u16; FRO * FRO * 3];
  {
    let src = Rgb96Frame::new(&le, FRP as u32, FRP as u32, (FRP * 3) as u32);
    let mut sink =
      MixedSinker::<Rgb96, AreaResampler>::with_resampler(FRP, FRP, AreaResampler::to(FRO, FRO))
        .unwrap()
        .with_rgb_u16(&mut out_le)
        .unwrap();
    rgb96_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  // BE arm
  let be = as_be_u32(&intended);
  let mut out_be = vec![0u16; FRO * FRO * 3];
  {
    let src = Rgb96BeFrame::new(&be, FRP as u32, FRP as u32, (FRP * 3) as u32);
    let mut sink = MixedSinker::<Rgb96<true>, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgb_u16(&mut out_be)
    .unwrap();
    crate::source::rgb96_to_endian::<_, true>(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(out_le, expected, "Rgb96 FR=false area rgb_u16 LE");
  assert_eq!(out_be, expected, "Rgb96 FR=false area rgb_u16 BE");
}

#[test]
fn rgba128_fr_false_area_rgba_u16_pins_mean_of_narrowed() {
  let intended = frp_rgba128();
  let staged: Vec<u16> = intended.iter().map(|&v| (v >> 16) as u16).collect();
  let mut expected = vec![0u16; FRO * FRO * 4];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..4 {
        expected[(oy * FRO + ox) * 4 + c] = frp_block_mean(&staged, ox, oy, c, 4);
      }
    }
  }
  let le = as_le_u32(&intended);
  let mut out_le = vec![0u16; FRO * FRO * 4];
  {
    let src = Rgba128Frame::new(&le, FRP as u32, FRP as u32, (FRP * 4) as u32);
    let mut sink =
      MixedSinker::<Rgba128, AreaResampler>::with_resampler(FRP, FRP, AreaResampler::to(FRO, FRO))
        .unwrap()
        .with_rgba_u16(&mut out_le)
        .unwrap();
    rgba128_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let be = as_be_u32(&intended);
  let mut out_be = vec![0u16; FRO * FRO * 4];
  {
    let src = Rgba128BeFrame::new(&be, FRP as u32, FRP as u32, (FRP * 4) as u32);
    let mut sink = MixedSinker::<Rgba128<true>, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_be)
    .unwrap();
    crate::source::rgba128_to_endian::<_, true>(&src, false, ColorMatrix::Bt709, &mut sink)
      .unwrap();
  }
  assert_eq!(out_le, expected, "Rgba128 FR=false area rgba_u16 LE");
  assert_eq!(out_be, expected, "Rgba128 FR=false area rgba_u16 BE");
}

#[test]
fn rgb96_fr_false_filter_rgb_u16_pins_current_output() {
  // Golden captured from the current narrow-first Triangle filter — pins the
  // ≤1-LSB behavior, not a u32-domain oracle (issue #289).
  let golden: [u16; FRO * FRO * 3] = [
    3121, 3412, 3703, 4494, 4785, 5077, 8613, 8905, 9196, 9987, 10278, 10569,
  ];
  let intended = frp_rgb96();
  let le = as_le_u32(&intended);
  let mut out_le = vec![0u16; FRO * FRO * 3];
  {
    let src = Rgb96Frame::new(&le, FRP as u32, FRP as u32, (FRP * 3) as u32);
    let mut sink = MixedSinker::<Rgb96, FilteredResampler<Triangle>>::with_resampler(
      FRP,
      FRP,
      FilteredResampler::new(FRO, FRO, Triangle),
    )
    .unwrap()
    .with_rgb_u16(&mut out_le)
    .unwrap();
    rgb96_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let be = as_be_u32(&intended);
  let mut out_be = vec![0u16; FRO * FRO * 3];
  {
    let src = Rgb96BeFrame::new(&be, FRP as u32, FRP as u32, (FRP * 3) as u32);
    let mut sink = MixedSinker::<Rgb96<true>, FilteredResampler<Triangle>>::with_resampler(
      FRP,
      FRP,
      FilteredResampler::new(FRO, FRO, Triangle),
    )
    .unwrap()
    .with_rgb_u16(&mut out_be)
    .unwrap();
    crate::source::rgb96_to_endian::<_, true>(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(out_le, golden, "Rgb96 FR=false filter rgb_u16 LE");
  assert_eq!(out_be, golden, "Rgb96 FR=false filter rgb_u16 BE");
}

#[test]
fn rgba128_fr_false_filter_rgba_u16_pins_current_output() {
  // Golden captured from the current narrow-first Triangle filter (issue #289).
  let golden: [u16; FRO * FRO * 4] = [
    4161, 4453, 4744, 5035, 5992, 6284, 6575, 6866, 11484, 11776, 12067, 12358, 13315, 13607,
    13898, 14189,
  ];
  let intended = frp_rgba128();
  let le = as_le_u32(&intended);
  let mut out_le = vec![0u16; FRO * FRO * 4];
  {
    let src = Rgba128Frame::new(&le, FRP as u32, FRP as u32, (FRP * 4) as u32);
    let mut sink = MixedSinker::<Rgba128, FilteredResampler<Triangle>>::with_resampler(
      FRP,
      FRP,
      FilteredResampler::new(FRO, FRO, Triangle),
    )
    .unwrap()
    .with_rgba_u16(&mut out_le)
    .unwrap();
    rgba128_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let be = as_be_u32(&intended);
  let mut out_be = vec![0u16; FRO * FRO * 4];
  {
    let src = Rgba128BeFrame::new(&be, FRP as u32, FRP as u32, (FRP * 4) as u32);
    let mut sink = MixedSinker::<Rgba128<true>, FilteredResampler<Triangle>>::with_resampler(
      FRP,
      FRP,
      FilteredResampler::new(FRO, FRO, Triangle),
    )
    .unwrap()
    .with_rgba_u16(&mut out_be)
    .unwrap();
    crate::source::rgba128_to_endian::<_, true>(&src, false, ColorMatrix::Bt709, &mut sink)
      .unwrap();
  }
  assert_eq!(out_le, golden, "Rgba128 FR=false filter rgba_u16 LE");
  assert_eq!(out_be, golden, "Rgba128 FR=false filter rgba_u16 BE");
}
