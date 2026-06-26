//! Fused-downscale coverage for the 32-bit planar GBR + alpha (`Gbrap32`)
//! family: the four `u32` planes are de-interleaved to a source-width host u16
//! RGBA row (the `>> 16` narrow), binning runs at native 16-bit depth, and the
//! native-depth `rgba_u16` output is an exact area mean of the narrowed source
//! (including alpha). This mirrors `Rgba128`'s resample coverage at a planar
//! source.

use crate::{
  ColorMatrix,
  resample::{AreaResampler, FilteredResampler, Triangle},
  sinker::MixedSinker,
};

const SRC: usize = 8;
const OUT: usize = 4;

fn as_le_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_le_bytes()))
    .collect()
}

fn as_be_u32(host: &[u32]) -> Vec<u32> {
  host
    .iter()
    .map(|v| u32::from_ne_bytes(v.to_be_bytes()))
    .collect()
}

/// One `u32` plane ramp with nonzero low-16 bits so the `>> 16` staging narrow
/// is genuinely lossy (matching the format contract).
fn plane_u32(seed: u32) -> Vec<u32> {
  (0..SRC * SRC)
    .map(|i| (i as u32).wrapping_mul(seed).wrapping_add(0xABCD))
    .collect()
}

/// Exact 2x2 block mean (round-half-up) over the staged u16 RGBA, channel `c`.
fn block_mean(staged: &[u16], ox: usize, oy: usize, c: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += staged[((oy * 2 + dy) * SRC + ox * 2 + dx) * 4 + c] as u64;
    }
  }
  ((acc + 2) / 4) as u16
}

/// Stage the four planes to the canonical host-native RGBA u16 row (each
/// `>> 16`, channel order R, G, B, A).
fn staged_rgba_u16(g: &[u32], b: &[u32], r: &[u32], a: &[u32]) -> Vec<u16> {
  let mut out = vec![0u16; SRC * SRC * 4];
  for i in 0..SRC * SRC {
    out[i * 4] = (r[i] >> 16) as u16;
    out[i * 4 + 1] = (g[i] >> 16) as u16;
    out[i * 4 + 2] = (b[i] >> 16) as u16;
    out[i * 4 + 3] = (a[i] >> 16) as u16;
  }
  out
}

#[test]
fn gbrap32_downscale_rgba_u16_is_exact_area_mean_incl_alpha() {
  let g = plane_u32(0x0011_0007);
  let b = plane_u32(0x0033_0009);
  let r = plane_u32(0x0055_000B);
  let a = plane_u32(0x0077_000D);
  let staged = staged_rgba_u16(&g, &b, &r, &a);
  let src = crate::frame::Gbrap32LeFrame::try_new(
    &g, &b, &r, &a, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  )
  .unwrap();

  let mut rgba_u16 = vec![0u16; OUT * OUT * 4];
  {
    let mut sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(OUT, OUT),
    )
    .unwrap()
    .with_rgba_u16(&mut rgba_u16)
    .unwrap();
    crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  for oy in 0..OUT {
    for ox in 0..OUT {
      for c in 0..4 {
        assert_eq!(
          rgba_u16[(oy * OUT + ox) * 4 + c],
          block_mean(&staged, ox, oy, c),
          "({ox},{oy}) c{c} (alpha is c3)"
        );
      }
    }
  }
}

#[test]
fn gbrap32_identity_plan_matches_new_sink() {
  let g = plane_u32(0x0011_0007);
  let b = plane_u32(0x0033_0009);
  let r = plane_u32(0x0055_000B);
  let a = plane_u32(0x0077_000D);
  let src = crate::frame::Gbrap32LeFrame::try_new(
    &g, &b, &r, &a, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32, SRC as u32,
  )
  .unwrap();

  let mut direct = vec![0u16; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<crate::source::Gbrap32>::new(SRC, SRC)
      .with_rgba_u16(&mut direct)
      .unwrap();
    crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let mut via_area = vec![0u16; SRC * SRC * 4];
  {
    let mut sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
      SRC,
      SRC,
      AreaResampler::to(SRC, SRC),
    )
    .unwrap()
    .with_rgba_u16(&mut via_area)
    .unwrap();
    crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(direct, via_area);
}

// ---- resample behavior pins (issue #289) ------------------------------------
// These PIN the current narrow-first (u32 `>> 16` before binning) resample
// output — within 1 LSB of the exact u32-domain mean, NOT parity vs a direct
// u32-domain oracle. The narrow-first gap applies to BOTH `full_range = true`
// and `full_range = false` (the limited-range case only amplifies it via the
// luma rescale); ONLY the direct identity-plan conversion is exact /
// byte-identical. A single host-native fixture is re-encoded LE / BE so both
// endian arms decode the same logical values and must produce the identical
// pinned output (area: mean-of-narrowed; filter: a captured golden). Accepted —
// 0-ULP fix (u128 area tier + u32 filter tier) tracked in issue #289.
// The `gbrap32_fr_true_*` pins below additionally assert (via a u32-domain
// oracle) that the full-range resample is genuinely narrow-first — it differs
// from the exact u32-domain mean on a crafted averaging boundary.

const FRP: usize = 4; // source side
const FRO: usize = 2; // output side

fn frp_plane(seed: u32) -> Vec<u32> {
  (0..FRP * FRP)
    .map(|i| (i as u32).wrapping_mul(seed).wrapping_add(0xBEEF))
    .collect()
}

fn frp_block_mean(staged: &[u16], ox: usize, oy: usize, c: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += staged[((oy * 2 + dy) * FRP + ox * 2 + dx) * 4 + c] as u64;
    }
  }
  ((acc + 2) / 4) as u16
}

#[test]
fn gbrap32_fr_false_area_rgba_u16_pins_mean_of_narrowed() {
  let g = frp_plane(0x0123_4567);
  let b = frp_plane(0x0246_8ACE);
  let r = frp_plane(0x0FED_CBA9);
  let a = frp_plane(0x0777_1333);
  // staged R,G,B,A = each plane >> 16.
  let mut staged = vec![0u16; FRP * FRP * 4];
  for i in 0..FRP * FRP {
    staged[i * 4] = (r[i] >> 16) as u16;
    staged[i * 4 + 1] = (g[i] >> 16) as u16;
    staged[i * 4 + 2] = (b[i] >> 16) as u16;
    staged[i * 4 + 3] = (a[i] >> 16) as u16;
  }
  let mut expected = vec![0u16; FRO * FRO * 4];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..4 {
        expected[(oy * FRO + ox) * 4 + c] = frp_block_mean(&staged, ox, oy, c);
      }
    }
  }

  // LE arm
  let (gl, bl, rl, al) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
  let mut out_le = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32LeFrame::try_new(
      &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_le)
    .unwrap();
    crate::source::gbrap32_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  // BE arm
  let (gb, bb, rb, ab) = (as_be_u32(&g), as_be_u32(&b), as_be_u32(&r), as_be_u32(&a));
  let mut out_be = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32BeFrame::try_new(
      &gb, &bb, &rb, &ab, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink = MixedSinker::<crate::source::Gbrap32<true>, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_be)
    .unwrap();
    crate::source::gbrap32_to_endian::<_, true>(&src, false, ColorMatrix::Bt709, &mut sink)
      .unwrap();
  }
  assert_eq!(out_le, expected, "Gbrap32 FR=false area rgba_u16 LE");
  assert_eq!(out_be, expected, "Gbrap32 FR=false area rgba_u16 BE");
}

#[test]
fn gbrap32_fr_false_filter_rgba_u16_pins_current_output() {
  // Golden captured from the current narrow-first Triangle filter — pins the
  // ≤1-LSB behavior, not a u32-domain oracle (issue #289).
  let golden: [u16; FRO * FRO * 4] = [
    14564, 1040, 2080, 6826, 20972, 1498, 2996, 9829, 40196, 2871, 5743, 18838, 46603, 3329, 6658,
    21841,
  ];
  let g = frp_plane(0x0123_4567);
  let b = frp_plane(0x0246_8ACE);
  let r = frp_plane(0x0FED_CBA9);
  let a = frp_plane(0x0777_1333);

  let (gl, bl, rl, al) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
  let mut out_le = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32LeFrame::try_new(
      &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink =
      MixedSinker::<crate::source::Gbrap32, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgba_u16(&mut out_le)
      .unwrap();
    crate::source::gbrap32_to(&src, false, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let (gb, bb, rb, ab) = (as_be_u32(&g), as_be_u32(&b), as_be_u32(&r), as_be_u32(&a));
  let mut out_be = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32BeFrame::try_new(
      &gb, &bb, &rb, &ab, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink =
      MixedSinker::<crate::source::Gbrap32<true>, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgba_u16(&mut out_be)
      .unwrap();
    crate::source::gbrap32_to_endian::<_, true>(&src, false, ColorMatrix::Bt709, &mut sink)
      .unwrap();
  }
  assert_eq!(out_le, golden, "Gbrap32 FR=false filter rgba_u16 LE");
  assert_eq!(out_be, golden, "Gbrap32 FR=false filter rgba_u16 BE");
}

/// A `4x4` R plane whose top-left `2x2` averaging block `{(0,0),(1,0),(0,1),
/// (1,1)}` is crafted so the mean of the **narrowed** samples differs from the
/// exact u32-domain mean by 1 LSB: values `[0x0001_FFFF, 0x0000_FFFF,
/// 0x0000_FFFF, 0x0000_FFFF]` narrow (`>> 16`) to `[1, 0, 0, 0]` → mean `0`,
/// while `((Σ u32 + 2) / 4) >> 16 = 81919 >> 16 = 1`. The remaining cells carry
/// nonzero low-16 bits. This is the fixture that makes the full-range
/// narrow-first behaviour (issue #289) observable.
fn gap_r_plane() -> Vec<u32> {
  let mut p: Vec<u32> = (0..FRP * FRP)
    .map(|i| (i as u32).wrapping_mul(0x00AB_1357).wrapping_add(0xABCD))
    .collect();
  p[0] = 0x0001_FFFF;
  p[1] = 0x0000_FFFF;
  p[FRP] = 0x0000_FFFF; // (0,1)
  p[FRP + 1] = 0x0000_FFFF; // (1,1)
  p
}

/// Exact u32-domain `2x2` block mean (round-half-up) then narrow `>> 16` — the
/// oracle the narrow-first path deviates from by ≤1 LSB.
fn u32_oracle_narrowed(plane: &[u32], ox: usize, oy: usize) -> u16 {
  let mut acc = 0u64;
  for dy in 0..2 {
    for dx in 0..2 {
      acc += plane[(oy * 2 + dy) * FRP + ox * 2 + dx] as u64;
    }
  }
  (((acc + 2) / 4) >> 16) as u16
}

#[test]
fn gbrap32_fr_true_area_rgba_u16_pins_mean_of_narrowed() {
  // Full-range resample is ALSO narrow-first (issue #289): it bins the `>> 16`
  // narrowed samples, NOT the u32-domain samples. Pin the current output as the
  // mean-of-narrowed AND prove it deviates from the exact u32-domain oracle.
  let r = gap_r_plane();
  let g = frp_plane(0x0246_8ACE);
  let b = frp_plane(0x0FED_CBA9);
  let a = frp_plane(0x0777_1333);
  let mut staged = vec![0u16; FRP * FRP * 4];
  for i in 0..FRP * FRP {
    staged[i * 4] = (r[i] >> 16) as u16;
    staged[i * 4 + 1] = (g[i] >> 16) as u16;
    staged[i * 4 + 2] = (b[i] >> 16) as u16;
    staged[i * 4 + 3] = (a[i] >> 16) as u16;
  }
  let mut expected = vec![0u16; FRO * FRO * 4];
  for oy in 0..FRO {
    for ox in 0..FRO {
      for c in 0..4 {
        expected[(oy * FRO + ox) * 4 + c] = frp_block_mean(&staged, ox, oy, c);
      }
    }
  }
  // Narrow-first proof: the R channel of output pixel (0,0) is the mean of the
  // narrowed block (0) and NOT the exact u32-domain mean (1).
  assert_eq!(expected[0], 0, "mean-of-narrowed R(0,0)");
  assert_eq!(u32_oracle_narrowed(&r, 0, 0), 1, "u32-domain oracle R(0,0)");
  assert_ne!(
    expected[0],
    u32_oracle_narrowed(&r, 0, 0),
    "fixture must expose the #289 narrow-first gap"
  );

  let (gl, bl, rl, al) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
  let mut out_le = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32LeFrame::try_new(
      &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink = MixedSinker::<crate::source::Gbrap32, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_le)
    .unwrap();
    crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let (gb, bb, rb, ab) = (as_be_u32(&g), as_be_u32(&b), as_be_u32(&r), as_be_u32(&a));
  let mut out_be = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32BeFrame::try_new(
      &gb, &bb, &rb, &ab, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink = MixedSinker::<crate::source::Gbrap32<true>, AreaResampler>::with_resampler(
      FRP,
      FRP,
      AreaResampler::to(FRO, FRO),
    )
    .unwrap()
    .with_rgba_u16(&mut out_be)
    .unwrap();
    crate::source::gbrap32_to_endian::<_, true>(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(
    out_le, expected,
    "Gbrap32 FR=true area rgba_u16 LE (narrow-first)"
  );
  assert_eq!(
    out_be, expected,
    "Gbrap32 FR=true area rgba_u16 BE (narrow-first)"
  );
}

/// PIL `Triangle` (BILINEAR) per-axis taps for an exact `4 -> 2` downscale
/// (`scale = 2`, `support = 2`): output 0 reads source `[0,1,2]` with weights
/// `[3/7, 3/7, 1/7]`; output 1 reads `[1,2,3]` with `[1/7, 3/7, 3/7]`. These
/// are PIL `precompute_coeffs` evaluated by hand (verified against
/// `FilterAxis::build`) for this geometry.
fn tri_taps_4to2(o: usize) -> ([f64; 3], [usize; 3]) {
  if o == 0 {
    ([3.0 / 7.0, 3.0 / 7.0, 1.0 / 7.0], [0, 1, 2])
  } else {
    ([1.0 / 7.0, 3.0 / 7.0, 3.0 / 7.0], [1, 2, 3])
  }
}

/// EXACT u32-domain Triangle-filter oracle for output `(ox, oy)` of a `4x4`
/// plane: separably filter the **raw u32** samples (the 0-ULP path a future
/// `u32` filter tier would take), then narrow `>> 16` only AFTER filtering.
/// The current engine instead narrows `>> 16` BEFORE filtering (issue #289),
/// so its output differs from this oracle by ≤1 LSB on a crafted boundary.
fn triangle_u32_oracle_4to2(plane: &[u32], ox: usize, oy: usize) -> u16 {
  let (wh, xs) = tri_taps_4to2(ox);
  let (wv, ys) = tri_taps_4to2(oy);
  let mut acc = 0.0f64;
  for (iy, &y) in ys.iter().enumerate() {
    for (ix, &x) in xs.iter().enumerate() {
      acc += wv[iy] * wh[ix] * plane[y * FRP + x] as f64;
    }
  }
  // Filter at u32 precision (round to the nearest u32), then narrow `>> 16`.
  ((acc.round() as u64) >> 16) as u16
}

#[test]
fn gbrap32_fr_true_filter_rgba_u16_pins_current_output() {
  // Full-range FILTER pin with the area pin's rigor (issue #289): a crafted
  // averaging-boundary fixture + an EXACT u32-domain Triangle oracle, asserting
  // the engine's narrow-first golden DIFFERS from that oracle on at least one
  // channel — so a future u32-domain-exact filter would FAIL this test.
  //
  // Crafted R plane: output (0,0)'s `3x3` Triangle window (rows/cols {0,1,2})
  // has only the two heaviest taps (0,0)+(0,1) carrying high16 = 1 and every
  // sample carrying low16 = 0xFFFF. Narrow-first (engine) filters the narrowed
  // window {1,1,0,…} → 18/49 ≈ 0.367 → rounds to 0. The u32-domain oracle
  // filters the raw samples → ≈ 89609 → `>> 16` = 1. The 0xFFFF low bits, kept
  // until after filtering, cross the narrow boundary: oracle 1 vs golden 0.
  let mut r = std::vec![0x0000_FFFFu32; FRP * FRP];
  r[0] = 0x0001_FFFF; // (0,0) — tap weight 9/49
  r[1] = 0x0001_FFFF; // (0,1) — tap weight 9/49
  let g = frp_plane(0x0123_4567);
  let b = frp_plane(0x0246_8ACE);
  let a = frp_plane(0x0777_1333);

  // Captured from the current narrow-first full-range Triangle filter (the R
  // channel is 0 at every output — the crafted plane's narrowed window is
  // mostly 0; the oracle gap lives at R(0,0)).
  let golden: [u16; FRO * FRO * 4] = [
    0, 1040, 2080, 6826, 0, 1498, 2996, 9829, 0, 2871, 5743, 18838, 0, 3329, 6658, 21841,
  ];

  // Narrow-first proof for the R channel of output (0,0) (golden index 0): the
  // engine emits the mean of the narrowed window (0) and NOT the exact
  // u32-domain Triangle mean (1).
  let oracle_r00 = triangle_u32_oracle_4to2(&r, 0, 0);
  assert_eq!(oracle_r00, 1, "u32-domain Triangle oracle R(0,0)");
  assert_eq!(golden[0], 0, "narrow-first golden R(0,0)");
  assert_ne!(
    golden[0], oracle_r00,
    "full-range FILTER must be narrow-first: golden must differ from the exact \
     u32-domain Triangle oracle (issue #289)"
  );

  let (gl, bl, rl, al) = (as_le_u32(&g), as_le_u32(&b), as_le_u32(&r), as_le_u32(&a));
  let mut out_le = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32LeFrame::try_new(
      &gl, &bl, &rl, &al, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink =
      MixedSinker::<crate::source::Gbrap32, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgba_u16(&mut out_le)
      .unwrap();
    crate::source::gbrap32_to(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  let (gb, bb, rb, ab) = (as_be_u32(&g), as_be_u32(&b), as_be_u32(&r), as_be_u32(&a));
  let mut out_be = vec![0u16; FRO * FRO * 4];
  {
    let src = crate::frame::Gbrap32BeFrame::try_new(
      &gb, &bb, &rb, &ab, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32, FRP as u32,
    )
    .unwrap();
    let mut sink =
      MixedSinker::<crate::source::Gbrap32<true>, FilteredResampler<Triangle>>::with_resampler(
        FRP,
        FRP,
        FilteredResampler::new(FRO, FRO, Triangle),
      )
      .unwrap()
      .with_rgba_u16(&mut out_be)
      .unwrap();
    crate::source::gbrap32_to_endian::<_, true>(&src, true, ColorMatrix::Bt709, &mut sink).unwrap();
  }
  assert_eq!(out_le, golden, "Gbrap32 FR=true filter rgba_u16 LE");
  assert_eq!(out_be, golden, "Gbrap32 FR=true filter rgba_u16 BE");
}
