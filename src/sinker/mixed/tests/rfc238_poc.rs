//! RFC #238 staged-resampling-pipeline vertical-slice PoC tests —
//! `Yuva420p` across the Encoded / Linear / Premultiplied averaging
//! domains.
//!
//! The four assertions that make this a *validation* of the
//! architecture, not just a smoke test:
//!
//! 1. [`premultiplied_domain_byte_identical_to_current`] — the anchor:
//!    `with_averaging_domain(Premultiplied)` on a premultiplied sink is
//!    **bit-exact** to the same sink without the override (the current
//!    default), across several geometries.
//! 2. [`encoded_domain_equals_independent_straight_native_oracle`] —
//!    Encoded equals a from-scratch bin-the-YUV-codes-then-convert
//!    oracle.
//! 3. [`linear_domain_equals_independent_linear_light_oracle`] — Linear
//!    equals a from-scratch decode → linearise → bin → re-encode oracle.
//! 4. [`encoded_and_linear_domains_differ`] — the two domains produce
//!    materially different RGB (the affine convert makes the choice
//!    observable).

use crate::{
  ColorMatrix,
  frame::Yuva420pFrame,
  resample::AreaResampler,
  sinker::{AlphaMode, AveragingDomain, MixedSinker, TransferFunction},
  source::{Yuva420p, yuva420p_to},
};

const M: ColorMatrix = ColorMatrix::Bt709;
const FR: bool = true;

/// Pseudo-random Y / U / V / A planes for an `s x s` frame (chroma
/// half-resolution); alpha varies (not all-opaque).
fn planes(s: usize, seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
  let cw = s / 2;
  let mut y = std::vec![0u8; s * s];
  let mut u = std::vec![0u8; cw * cw];
  let mut v = std::vec![0u8; cw * cw];
  let mut a = std::vec![0u8; s * s];
  super::pseudo_random_u8(&mut y, seed);
  super::pseudo_random_u8(&mut u, seed ^ 0x1111_1111);
  super::pseudo_random_u8(&mut v, seed ^ 0x2222_2222);
  super::pseudo_random_u8(&mut a, seed ^ 0x3333_3333);
  (y, u, v, a)
}

fn frame<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], a: &'a [u8], s: usize) -> Yuva420pFrame<'a> {
  let cw = (s / 2) as u32;
  Yuva420pFrame::try_new(y, u, v, a, s as u32, s as u32, s as u32, cw, cw, s as u32).unwrap()
}

/// Full-resolution canonical straight RGBA of the source — a direct
/// (identity) `Yuva420p` conversion through the real kernel.
fn direct_rgba(y: &[u8], u: &[u8], v: &[u8], a: &[u8], s: usize) -> Vec<u8> {
  let mut rgba = std::vec![0u8; s * s * 4];
  {
    let mut sink = MixedSinker::<Yuva420p>::new(s, s)
      .with_rgba(&mut rgba)
      .unwrap();
    yuva420p_to(&frame(y, u, v, a, s), FR, M, &mut sink).unwrap();
  }
  rgba
}

/// Round-half-up 2x2 block mean of one single-channel plane (`src_w` →
/// `src_w / 2`). The factor is exactly 2 each way (the PoC geometries are
/// all `s -> s / 2`).
fn block_mean_plane(src: &[u8], src_w: usize, src_h: usize) -> Vec<u8> {
  let ow = src_w / 2;
  let oh = src_h / 2;
  let mut out = std::vec![0u8; ow * oh];
  for oy in 0..oh {
    for ox in 0..ow {
      let mut acc = 0u32;
      for dy in 0..2 {
        for dx in 0..2 {
          acc += src[(oy * 2 + dy) * src_w + ox * 2 + dx] as u32;
        }
      }
      out[oy * ow + ox] = ((acc + 2) / 4) as u8;
    }
  }
  out
}

/// Builds a sink over an `s x s` `Yuva420p` source resampled to
/// `s/2 x s/2`, optionally with an averaging domain and alpha mode, and
/// returns the output straight RGBA.
fn resample_rgba(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a: &[u8],
  s: usize,
  alpha: AlphaMode,
  domain: Option<AveragingDomain>,
) -> Vec<u8> {
  let o = s / 2;
  let mut rgba = std::vec![0u8; o * o * 4];
  {
    let mut sink =
      MixedSinker::<Yuva420p, AreaResampler>::with_resampler(s, s, AreaResampler::to(o, o))
        .unwrap()
        .with_alpha_mode(alpha);
    if let Some(d) = domain {
      sink = sink.with_averaging_domain(d);
    }
    let mut sink = sink.with_rgba(&mut rgba).unwrap();
    yuva420p_to(&frame(y, u, v, a, s), FR, M, &mut sink).unwrap();
  }
  rgba
}

// ---- 1. The byte-identity anchor ---------------------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn premultiplied_domain_byte_identical_to_current() {
  // Across several geometries: a Premultiplied-alpha sink WITH the
  // Premultiplied averaging domain must be bit-exact to the same sink
  // WITHOUT the override (the historical default premult area path). This
  // proves the domain dispatch routes to the existing tail with zero
  // behaviour change.
  for &s in &[4usize, 8, 16, 32] {
    let (y, u, v, a) = planes(s, 0xA5A5 ^ s as u32);
    let current = resample_rgba(&y, &u, &v, &a, s, AlphaMode::Premultiplied, None);
    let via_domain = resample_rgba(
      &y,
      &u,
      &v,
      &a,
      s,
      AlphaMode::Premultiplied,
      Some(AveragingDomain::Premultiplied),
    );
    assert_eq!(
      via_domain, current,
      "Premultiplied domain not byte-identical to the current default at {s}x{s}"
    );
    // The domain alone drives premultiplied semantics: selecting it on a
    // default (Straight) sink yields the same bytes — the domain, not the
    // sink's alpha mode, is what routes to the premult tail.
    let domain_on_straight = resample_rgba(
      &y,
      &u,
      &v,
      &a,
      s,
      AlphaMode::Straight,
      Some(AveragingDomain::Premultiplied),
    );
    assert_eq!(
      domain_on_straight, current,
      "Premultiplied domain on a Straight sink diverged from the premult default at {s}x{s}"
    );
  }
}

// ---- 2. Encoded domain vs independent native oracle --------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn encoded_domain_equals_independent_straight_native_oracle() {
  let s = 8usize;
  let o = s / 2;
  let cw = s / 2;
  let (y, u, v, a) = planes(s, 0x1357);

  let got = resample_rgba(
    &y,
    &u,
    &v,
    &a,
    s,
    AlphaMode::Straight,
    Some(AveragingDomain::Encoded),
  );

  // Independent oracle: block-mean each native plane FROM SCRATCH, then
  // convert the downscaled YUVA frame through a direct (identity)
  // `Yuva420p` sink — a convert path independent of the resample
  // dispatch under test.
  let y_b = block_mean_plane(&y, s, s);
  let a_b = block_mean_plane(&a, s, s);
  let u_b = block_mean_plane(&u, cw, cw);
  let v_b = block_mean_plane(&v, cw, cw);
  let oracle = direct_rgba(&y_b, &u_b, &v_b, &a_b, o);

  // Luma (R/G/B carry it) and alpha are bit-identical; the whole RGBA is
  // in fact bit-identical because both feed the same binned planes to the
  // same convert kernel. Assert the strongest property that holds.
  for (px_got, px_oracle) in got.chunks_exact(4).zip(oracle.chunks_exact(4)) {
    assert_eq!(px_got[3], px_oracle[3], "alpha must be bit-identical");
    for c in 0..3 {
      let d = (px_got[c] as i32 - px_oracle[c] as i32).abs();
      assert!(d <= 1, "RGB channel {c} off by {d} (> tol)");
    }
  }
  assert_eq!(got, oracle, "Encoded output != independent native oracle");
  // Alpha was a real area mean, not forced opaque.
  assert!(
    got.chunks_exact(4).any(|px| px[3] != 0xFF),
    "resampled alpha forced opaque"
  );
}

// ---- 3. Linear domain vs independent linear-light oracle ---------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn linear_domain_equals_independent_linear_light_oracle() {
  let s = 8usize;
  let o = s / 2;
  let (y, u, v, a) = planes(s, 0x2468);

  let got = resample_rgba(
    &y,
    &u,
    &v,
    &a,
    s,
    AlphaMode::Straight,
    Some(AveragingDomain::Linear),
  );

  // Independent oracle, computed from scratch: decode full-res straight
  // RGBA, linearise R/G/B via the sRGB EOTF, 2x2 block-mean the linear
  // RGB and the straight alpha, then re-encode via the sRGB OETF.
  let tf = TransferFunction::SRGB;
  let full = direct_rgba(&y, &u, &v, &a, s);
  // Per-source-pixel linear RGB + straight alpha.
  let mut lin = std::vec![0f32; s * s * 3];
  let mut alpha = std::vec![0u8; s * s];
  for (px, (l, av)) in full
    .chunks_exact(4)
    .zip(lin.chunks_exact_mut(3).zip(alpha.iter_mut()))
  {
    for c in 0..3 {
      l[c] = tf.eotf(px[c] as f32 / 255.0);
    }
    *av = px[3];
  }
  // 2x2 block-mean the linear RGB (f32) and re-encode; block-mean alpha.
  let a_b = block_mean_plane(&alpha, s, s);
  let mut oracle = std::vec![0u8; o * o * 4];
  for oy in 0..o {
    for ox in 0..o {
      for c in 0..3 {
        let mut acc = 0f32;
        for dy in 0..2 {
          for dx in 0..2 {
            acc += lin[((oy * 2 + dy) * s + ox * 2 + dx) * 3 + c];
          }
        }
        let enc = tf.oetf(acc / 4.0) * 255.0 + 0.5;
        oracle[(oy * o + ox) * 4 + c] = enc.clamp(0.0, 255.0) as u8;
      }
      oracle[(oy * o + ox) * 4 + 3] = a_b[oy * o + ox];
    }
  }

  assert_eq!(
    got, oracle,
    "Linear output != independent linear-light oracle"
  );
}

// ---- 4. The domains are materially different ---------------------------

#[test]
#[cfg_attr(
  miri,
  ignore = "SIMD-dispatched row kernels use intrinsics unsupported by Miri"
)]
fn encoded_and_linear_domains_differ() {
  // The affine YUV→RGB convert means averaging YUV codes (Encoded) is not
  // the same as averaging in linear light (Linear). Prove the RGB diverges
  // materially on real (non-flat) content, validating that the domain
  // choice is meaningful.
  let s = 16usize;
  let (y, u, v, a) = planes(s, 0x9E37);
  let encoded = resample_rgba(
    &y,
    &u,
    &v,
    &a,
    s,
    AlphaMode::Straight,
    Some(AveragingDomain::Encoded),
  );
  let linear = resample_rgba(
    &y,
    &u,
    &v,
    &a,
    s,
    AlphaMode::Straight,
    Some(AveragingDomain::Linear),
  );

  let max_rgb_diff = encoded
    .chunks_exact(4)
    .zip(linear.chunks_exact(4))
    .flat_map(|(e, l)| (0..3).map(move |c| (e[c] as i32 - l[c] as i32).abs()))
    .max()
    .unwrap();
  assert!(
    max_rgb_diff > 4,
    "Encoded and Linear RGB barely differ (max {max_rgb_diff}) — domain choice not meaningful"
  );
  // Alpha is straight in both, so it must match exactly.
  for (e, l) in encoded.chunks_exact(4).zip(linear.chunks_exact(4)) {
    assert_eq!(e[3], l[3], "straight alpha must match across domains");
  }
}

// ---- Transfer-function round-trip sanity (PoC stand-in) ----------------

#[test]
fn transfer_function_srgb_roundtrips() {
  let tf = TransferFunction::SRGB;
  // EOTF then OETF is the identity (within f32 tolerance) across [0, 1].
  for i in 0..=255u32 {
    let c = i as f32 / 255.0;
    let round = tf.oetf(tf.eotf(c));
    assert!(
      (round - c).abs() < 1e-4,
      "sRGB round-trip drift at {c}: got {round}"
    );
  }
  // Known anchors.
  assert!(tf.eotf(0.0).abs() < 1e-7, "EOTF(0) != 0");
  assert!((tf.oetf(1.0) - 1.0).abs() < 1e-4, "OETF(1) != 1");
}
