use super::*;

/// Largest absolute deviation between a kernel's `weight` and a reference
/// closure across a dense sample grid covering `[-span, span]`.
fn max_weight_dev(k: &dyn FilterKernel, span: f64, reference: impl Fn(f64) -> f64) -> f64 {
  let mut worst = 0.0f64;
  let steps = 4000;
  for i in 0..=steps {
    let x = -span + 2.0 * span * (i as f64) / (steps as f64);
    let d = (k.weight(x) - reference(x)).abs();
    if d > worst {
      worst = d;
    }
  }
  worst
}

#[test]
fn triangle_profile() {
  let k = Triangle;
  assert_eq!(k.support(), 1.0);
  assert_eq!(k.weight(0.0), 1.0);
  // Symmetric tent, zero at and past the support.
  assert!((k.weight(0.5) - 0.5).abs() < 1e-12);
  assert!((k.weight(-0.5) - 0.5).abs() < 1e-12);
  assert_eq!(k.weight(1.0), 0.0);
  assert_eq!(k.weight(1.5), 0.0);
  assert_eq!(k.weight(-2.0), 0.0);
  // Symmetry.
  for &x in &[0.1, 0.37, 0.9] {
    assert_eq!(k.weight(x), k.weight(-x));
  }
}

#[test]
fn catmull_rom_profile() {
  let k = CatmullRom;
  assert_eq!(k.support(), 2.0);
  assert_eq!(k.weight(0.0), 1.0);
  // Interpolating cubic: zero at every nonzero integer node.
  assert!(k.weight(1.0).abs() < 1e-12);
  assert!(k.weight(-1.0).abs() < 1e-12);
  assert_eq!(k.weight(2.0), 0.0);
  assert_eq!(k.weight(2.5), 0.0);
  // Negative outer lobe on (1, 2): the Keys a=-0.5 cubic at |x| = 1.5 has
  // the known closed form -0.0625, and at |x| = 0.5 the inner cubic is
  // 0.5625.
  let w15 = k.weight(1.5);
  assert!(w15 < 0.0, "outer lobe must be negative, got {w15}");
  assert!((w15 - (-0.0625)).abs() < 1e-12, "got {w15}");
  assert!((k.weight(0.5) - 0.5625).abs() < 1e-12);
  assert_eq!(k.weight(1.5), k.weight(-1.5));
  // Symmetry across the dense grid.
  assert!(max_weight_dev(&k, 2.5, |x| k.weight(-x)) < 1e-15);
}

#[test]
fn lanczos3_profile() {
  let k = Lanczos3;
  assert_eq!(k.support(), 3.0);
  assert_eq!(k.weight(0.0), 1.0);
  // sinc zeros at nonzero integers within support.
  for n in 1..3 {
    assert!(k.weight(n as f64).abs() < 1e-12, "zero at {n}");
    assert!(k.weight(-(n as f64)).abs() < 1e-12, "zero at -{n}");
  }
  assert_eq!(k.weight(3.0), 0.0);
  assert_eq!(k.weight(3.5), 0.0);
  // Reference windowed-sinc; symmetric and matching a direct evaluation.
  let reference = |x: f64| {
    let s = |t: f64| {
      if t == 0.0 {
        1.0
      } else {
        (core::f64::consts::PI * t).sin() / (core::f64::consts::PI * t)
      }
    };
    if x > -3.0 && x < 3.0 {
      s(x) * s(x / 3.0)
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 3.5, reference) < 1e-12);
}

#[test]
fn mitchell_profile() {
  let k = Mitchell;
  assert_eq!(k.support(), 2.0);
  // Mitchell is non-interpolating: nonzero at the unit sample (8/9 at the
  // center, 1/18 at |x| = 1), unlike the interpolating Catmull-Rom cubic.
  assert!((k.weight(0.0) - 8.0 / 9.0).abs() < 1e-12);
  assert!((k.weight(1.0) - 1.0 / 18.0).abs() < 1e-12);
  assert!((k.weight(-1.0) - 1.0 / 18.0).abs() < 1e-12);
  assert_eq!(k.weight(2.0), 0.0);
  assert_eq!(k.weight(2.5), 0.0);
  // Known interior value and the negative outer (ring) lobe.
  assert!((k.weight(0.5) - 77.0 / 144.0).abs() < 1e-12);
  let w15 = k.weight(1.5);
  assert!(w15 < 0.0, "outer lobe must be negative, got {w15}");
  assert!((w15 - (-5.0 / 144.0)).abs() < 1e-12, "got {w15}");
  // Symmetric.
  assert_eq!(k.weight(1.5), k.weight(-1.5));
  assert!(max_weight_dev(&k, 2.5, |x| k.weight(-x)) < 1e-15);
  // Parity with the closed-form Mitchell-Netravali weights (B = C = 1/3),
  // evaluated as an explicit (non-Horner) polynomial so a factoring slip
  // in the kernel would show up here.
  let reference = |x: f64| {
    let t = x.abs();
    let (b, c) = (1.0 / 3.0_f64, 1.0 / 3.0_f64);
    if t < 1.0 {
      ((12.0 - 9.0 * b - 6.0 * c) * t.powi(3)
        + (-18.0 + 12.0 * b + 6.0 * c) * t.powi(2)
        + (6.0 - 2.0 * b))
        / 6.0
    } else if t < 2.0 {
      ((-b - 6.0 * c) * t.powi(3)
        + (6.0 * b + 30.0 * c) * t.powi(2)
        + (-12.0 * b - 48.0 * c) * t
        + (8.0 * b + 24.0 * c))
        / 6.0
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 2.5, reference) < 1e-12);
}

#[test]
fn opencv_cubic_profile() {
  let k = OpenCvCubic;
  assert_eq!(k.support(), 2.0);
  // Interpolating Keys cubic: 1 at the center, 0 at the unit sample.
  assert_eq!(k.weight(0.0), 1.0);
  assert!(k.weight(1.0).abs() < 1e-12);
  assert!(k.weight(-1.0).abs() < 1e-12);
  assert_eq!(k.weight(2.0), 0.0);
  assert_eq!(k.weight(2.5), 0.0);
  // Deeper negative outer lobe than Catmull-Rom (a = -0.75 vs -0.5):
  // at |x| = 1.5 the Keys cubic gives a * 0.125 = -0.09375.
  let w15 = k.weight(1.5);
  assert!(w15 < 0.0, "outer lobe must be negative, got {w15}");
  assert!((w15 - (-0.09375)).abs() < 1e-12, "got {w15}");
  assert_eq!(k.weight(1.5), k.weight(-1.5));
  assert!(max_weight_dev(&k, 2.5, |x| k.weight(-x)) < 1e-15);
  // Parity with the closed-form Keys cubic at a = -0.75 (explicit powi
  // form, so a Horner slip in the kernel would show up here).
  let reference = |x: f64| {
    let a = -0.75_f64;
    let t = x.abs();
    if t < 1.0 {
      (a + 2.0) * t.powi(3) - (a + 3.0) * t.powi(2) + 1.0
    } else if t < 2.0 {
      a * t.powi(3) - 5.0 * a * t.powi(2) + 8.0 * a * t - 4.0 * a
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 2.5, reference) < 1e-12);
}

#[test]
fn lanczos4_profile() {
  let k = Lanczos4;
  assert_eq!(k.support(), 4.0);
  assert_eq!(k.weight(0.0), 1.0);
  // sinc zeros at nonzero integers within support.
  for n in 1..4 {
    assert!(k.weight(n as f64).abs() < 1e-12, "zero at {n}");
    assert!(k.weight(-(n as f64)).abs() < 1e-12, "zero at -{n}");
  }
  assert_eq!(k.weight(4.0), 0.0);
  assert_eq!(k.weight(4.5), 0.0);
  // Reference windowed sinc (a = 4), symmetric and matching a direct eval.
  let reference = |x: f64| {
    let s = |t: f64| {
      if t == 0.0 {
        1.0
      } else {
        (core::f64::consts::PI * t).sin() / (core::f64::consts::PI * t)
      }
    };
    if x > -4.0 && x < 4.0 {
      s(x) * s(x / 4.0)
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 4.5, reference) < 1e-12);
}

#[test]
fn swscale_bicubic_profile() {
  let k = SwscaleBicubic;
  assert_eq!(k.support(), 2.0);
  // Interpolating Keys cubic: 1 at the center, 0 at the unit sample and the
  // support boundary.
  assert_eq!(k.weight(0.0), 1.0);
  assert!(k.weight(1.0).abs() < 1e-12);
  assert!(k.weight(-1.0).abs() < 1e-12);
  assert_eq!(k.weight(2.0), 0.0);
  assert_eq!(k.weight(2.5), 0.0);
  // Negative outer lobe on (1, 2): the Keys a = -0.6 cubic at |x| = 1.5 is
  // a * 0.125 = -0.075 (shallower than cv2's a = -0.75 -> -0.09375, deeper
  // than PIL's a = -0.5 -> -0.0625).
  let w15 = k.weight(1.5);
  assert!(w15 < 0.0, "outer lobe must be negative, got {w15}");
  assert!((w15 - (-0.075)).abs() < 1e-12, "got {w15}");
  // Inner segment at |x| = 0.5: 1.4*0.125 - 2.4*0.25 + 1 = 0.575.
  assert!((k.weight(0.5) - 0.575).abs() < 1e-12);
  assert_eq!(k.weight(1.5), k.weight(-1.5));
  assert!(max_weight_dev(&k, 2.5, |x| k.weight(-x)) < 1e-15);
  // Parity with the closed-form Keys cubic at a = -0.6 (explicit powi form,
  // so a Horner slip in the kernel would show up here).
  let reference = |x: f64| {
    let a = -0.6_f64;
    let t = x.abs();
    if t < 1.0 {
      (a + 2.0) * t.powi(3) - (a + 3.0) * t.powi(2) + 1.0
    } else if t < 2.0 {
      a * t.powi(3) - 5.0 * a * t.powi(2) + 8.0 * a * t - 4.0 * a
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 2.5, reference) < 1e-12);
}

#[test]
fn gaussian_profile() {
  let k = Gaussian;
  assert_eq!(k.support(), 3.0);
  // Peak is 2^0 = 1 at the center.
  assert_eq!(k.weight(0.0), 1.0);
  // Truncated to the half-open [-3, 3): zero at and past the support.
  assert_eq!(k.weight(3.0), 0.0);
  assert_eq!(k.weight(-3.0), 0.0);
  assert_eq!(k.weight(3.5), 0.0);
  // The Gaussian is NOT interpolating and NOT a partition of unity (the
  // engine renormalizes each window — `axis_windows_normalize_to_one`
  // covers that), so it is nonzero at the nonzero integers.
  assert!(k.weight(1.0) > 0.0);
  assert!(k.weight(2.0) > 0.0);
  // Symmetric across the dense grid.
  assert!(max_weight_dev(&k, 3.5, |x| k.weight(-x)) < 1e-15);
  // Strictly monotonically decreasing on [0, 3) — a Gaussian shape, no
  // lobes or ripples.
  let mut prev = f64::INFINITY;
  for i in 0..300 {
    let x = i as f64 / 100.0;
    let v = k.weight(x);
    assert!(v < prev, "must decrease at x = {x}: {v} !< {prev}");
    prev = v;
  }
  // Parity with the closed-form swscale Gauss shape 2^(-3 x^2), evaluated
  // with `exp` (a different transcendental than the kernel's `exp2`), so a
  // wrong base or parameter in the kernel would show up here. The tolerance
  // absorbs f64 rounding across the two evaluation paths.
  let reference = |x: f64| {
    if x.abs() < 3.0 {
      (-3.0 * x * x * core::f64::consts::LN_2).exp()
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 3.5, reference) < 1e-12);
}

#[test]
fn blackman_sinc_profile() {
  let k = BlackmanSinc;
  assert_eq!(k.support(), 3.0);
  // Window at the center is 0.42 + 0.5 + 0.08 = 1, so weight(0) = sinc(0) = 1.
  assert!((k.weight(0.0) - 1.0).abs() < 1e-12);
  // Interpolating at the integer nodes within support (the sinc factor is
  // zero at every nonzero integer), like the Lanczos kernels.
  for n in 1..3 {
    assert!(k.weight(n as f64).abs() < 1e-12, "zero at {n}");
    assert!(k.weight(-(n as f64)).abs() < 1e-12, "zero at -{n}");
  }
  // Half-open [-3, 3): zero at and past the support boundary (and the
  // window itself vanishes there: 0.42 - 0.5 + 0.08 = 0).
  assert_eq!(k.weight(3.0), 0.0);
  assert_eq!(k.weight(-3.0), 0.0);
  assert_eq!(k.weight(3.5), 0.0);
  // Negative outer lobe, shallower than Lanczos3's: the Blackman window
  // suppresses the side lobes more than Lanczos's sinc window.
  let w15 = k.weight(1.5);
  assert!(w15 < 0.0, "outer lobe must be negative, got {w15}");
  assert!(
    w15 > Lanczos3.weight(1.5),
    "Blackman lobe is shallower than Lanczos3's"
  );
  // Symmetric across the dense grid.
  assert!(max_weight_dev(&k, 3.5, |x| k.weight(-x)) < 1e-15);
  // Reference windowed sinc (conventional 0.42 / 0.5 / 0.08 Blackman window,
  // a = 3), evaluated directly (not via the kernel's factored helpers) so a
  // coefficient or argument slip would show up here.
  let reference = |x: f64| {
    let s = |t: f64| {
      if t == 0.0 {
        1.0
      } else {
        (core::f64::consts::PI * t).sin() / (core::f64::consts::PI * t)
      }
    };
    if x > -3.0 && x < 3.0 {
      let u = core::f64::consts::PI * x / 3.0;
      let win = 0.42 + 0.5 * u.cos() + 0.08 * (2.0 * u).cos();
      s(x) * win
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 3.5, reference) < 1e-12);
}

#[test]
fn cubic_b_spline_profile() {
  let k = CubicBSpline;
  assert_eq!(k.support(), 2.0);
  // The smoothing cubic B-spline is NON-interpolating: 2/3 at the center,
  // 1/6 at the unit sample (unlike the interpolating Catmull-Rom cubic,
  // which is 1 and 0). It blurs rather than passes samples through.
  assert!((k.weight(0.0) - 2.0 / 3.0).abs() < 1e-12);
  assert!((k.weight(1.0) - 1.0 / 6.0).abs() < 1e-12);
  assert!((k.weight(-1.0) - 1.0 / 6.0).abs() < 1e-12);
  assert_eq!(k.weight(2.0), 0.0);
  assert_eq!(k.weight(2.5), 0.0);
  // Non-negative everywhere (no ringing / no negative lobes), the
  // signature of the smoothing B-spline.
  for i in 0..400 {
    let x = i as f64 / 100.0;
    assert!(k.weight(x) >= 0.0, "no negative lobe at x = {x}");
  }
  // Known interior values: B3(1/2) = 23/48, B3(3/2) = 1/48.
  assert!((k.weight(0.5) - 23.0 / 48.0).abs() < 1e-12);
  assert!((k.weight(1.5) - 1.0 / 48.0).abs() < 1e-12);
  // C2-continuous at the internal knot |x| = 1 (both segments agree at 1/6).
  assert!((k.weight(0.999_999) - k.weight(1.000_001)).abs() < 1e-4);
  // Symmetric.
  assert_eq!(k.weight(1.5), k.weight(-1.5));
  assert!(max_weight_dev(&k, 2.5, |x| k.weight(-x)) < 1e-15);
  // Partition of unity (preserves DC) at several fractional offsets — even
  // though it is non-interpolating, its taps sum to one.
  for &x in &[0.0, 0.25, 0.5, 0.5_f64.sqrt(), 0.9] {
    assert!((tap_sum(&k, x) - 1.0).abs() < 1e-12, "PoU at {x}");
  }
  // Parity with the closed-form B3 (explicit, non-Horner form), so a
  // factoring slip in the kernel would show up here.
  let reference = |x: f64| {
    let t = x.abs();
    if t < 1.0 {
      (4.0 - 6.0 * t.powi(2) + 3.0 * t.powi(3)) / 6.0
    } else if t < 2.0 {
      (2.0 - t).powi(3) / 6.0
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 2.5, reference) < 1e-12);
}

/// Sum of a kernel's taps at unit-spaced offsets around `x` — the partition
/// of unity an interpolating kernel must satisfy (`== 1`) to preserve DC.
fn tap_sum(k: &dyn FilterKernel, x: f64) -> f64 {
  let r = k.support().ceil() as i64;
  let mut s = 0.0;
  for n in -r..=r {
    s += k.weight(x - n as f64);
  }
  s
}

#[test]
fn spline16_profile() {
  let k = Spline16;
  assert_eq!(k.support(), 2.0);
  // Interpolating: 1 at the center, 0 at every other integer + the boundary.
  assert_eq!(k.weight(0.0), 1.0);
  assert!(k.weight(1.0).abs() < 1e-12);
  assert!(k.weight(2.0).abs() < 1e-12);
  assert_eq!(k.weight(2.5), 0.0);
  // Symmetric.
  assert!(max_weight_dev(&k, 2.5, |x| k.weight(-x)) < 1e-15);
  // C0-continuous at the internal knot (both segments agree at |x| = 1 = 0).
  assert!((k.weight(0.999_999) - k.weight(1.000_001)).abs() < 1e-4);
  // Partition of unity (preserves DC) at several fractional offsets.
  for &x in &[0.0, 0.25, 0.5, 0.5_f64.sqrt(), 0.9] {
    assert!((tap_sum(&k, x) - 1.0).abs() < 1e-12, "PoU at {x}");
  }
  // Closed-form zimg reference (poly3 = c0 + t(c1 + t(c2 + t c3))).
  let reference = |x: f64| {
    let p = |t: f64, c: [f64; 4]| c[0] + t * (c[1] + t * (c[2] + t * c[3]));
    let t = x.abs();
    if t < 1.0 {
      p(t, [1.0, -1.0 / 5.0, -9.0 / 5.0, 1.0])
    } else if t < 2.0 {
      p(t - 1.0, [0.0, -7.0 / 15.0, 4.0 / 5.0, -1.0 / 3.0])
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 2.5, reference) < 1e-15);
}

#[test]
fn spline36_profile() {
  let k = Spline36;
  assert_eq!(k.support(), 3.0);
  assert_eq!(k.weight(0.0), 1.0);
  for n in 1..=3 {
    assert!(k.weight(n as f64).abs() < 1e-12, "zero at {n}");
  }
  assert_eq!(k.weight(3.5), 0.0);
  assert!(max_weight_dev(&k, 3.5, |x| k.weight(-x)) < 1e-15);
  for &x in &[0.0, 0.25, 0.5, 0.7, 0.9] {
    assert!((tap_sum(&k, x) - 1.0).abs() < 1e-12, "PoU at {x}");
  }
  let reference = |x: f64| {
    let p = |t: f64, c: [f64; 4]| c[0] + t * (c[1] + t * (c[2] + t * c[3]));
    let t = x.abs();
    if t < 1.0 {
      p(t, [1.0, -3.0 / 209.0, -453.0 / 209.0, 13.0 / 11.0])
    } else if t < 2.0 {
      p(t - 1.0, [0.0, -156.0 / 209.0, 270.0 / 209.0, -6.0 / 11.0])
    } else if t < 3.0 {
      p(t - 2.0, [0.0, 26.0 / 209.0, -45.0 / 209.0, 1.0 / 11.0])
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 3.5, reference) < 1e-15);
}

#[test]
fn spline64_profile() {
  let k = Spline64;
  assert_eq!(k.support(), 4.0);
  assert_eq!(k.weight(0.0), 1.0);
  for n in 1..=4 {
    assert!(k.weight(n as f64).abs() < 1e-12, "zero at {n}");
  }
  assert_eq!(k.weight(4.5), 0.0);
  assert!(max_weight_dev(&k, 4.5, |x| k.weight(-x)) < 1e-15);
  for &x in &[0.0, 0.25, 0.5, 0.7, 0.9] {
    assert!((tap_sum(&k, x) - 1.0).abs() < 1e-12, "PoU at {x}");
  }
  let reference = |x: f64| {
    let p = |t: f64, c: [f64; 4]| c[0] + t * (c[1] + t * (c[2] + t * c[3]));
    let t = x.abs();
    if t < 1.0 {
      p(t, [1.0, -3.0 / 2911.0, -6387.0 / 2911.0, 49.0 / 41.0])
    } else if t < 2.0 {
      p(
        t - 1.0,
        [0.0, -2328.0 / 2911.0, 4032.0 / 2911.0, -24.0 / 41.0],
      )
    } else if t < 3.0 {
      p(t - 2.0, [0.0, 582.0 / 2911.0, -1008.0 / 2911.0, 6.0 / 41.0])
    } else if t < 4.0 {
      p(t - 3.0, [0.0, -97.0 / 2911.0, 168.0 / 2911.0, -1.0 / 41.0])
    } else {
      0.0
    }
  };
  assert!(max_weight_dev(&k, 4.5, reference) < 1e-15);
}

#[test]
fn spline_matches_zimg_golden_fixtures() {
  // Independent exactness oracle for the Spline coefficients. These
  // (x, weight) pairs were computed OUTSIDE this crate by exact rational
  // arithmetic over zimg's published `Spline{16,36,64}Filter` segment
  // coefficients (Python `fractions.Fraction`, then cast to `f64`), at
  // non-knot points in every segment. Because they are literal numbers from
  // a different tool and evaluation order — not the production Horner form —
  // a transcription error in a kernel coefficient (which the closed-form
  // reference in the per-kernel profile tests would copy and miss) shows up
  // here as a mismatch. The tolerance only absorbs f64 rounding across the
  // two evaluation orders.
  let s16: &[(f64, f64)] = &[
    (0.25, 0.853125),
    (0.5, 0.575),
    (0.75, 0.259375),
    (1.25, -0.071875),
    (1.5, -0.075),
    (1.75, -0.040625),
  ];
  let s36: &[(f64, f64)] = &[
    (0.25, 0.8794108851674641),
    (0.5, 0.5986842105263158),
    (0.75, 0.26861543062200954),
    (1.25, -0.11438397129186603),
    (1.5, -0.11842105263157894),
    (2.25, 0.019063995215311005),
    (2.5, 0.019736842105263157),
    (2.75, 0.010541267942583732),
  ];
  let s64: &[(f64, f64)] = &[
    (0.25, 0.8812854259704569),
    (0.5, 0.6003521126760564),
    (1.4, -0.1357389213328753),
    (2.25, 0.03062736173136379),
    (2.5, 0.03169014084507042),
    (3.25, -0.005104560288560632),
    (3.5, -0.00528169014084507),
    (3.75, -0.0028179749227069738),
  ];
  for &(x, want) in s16 {
    assert!((Spline16.weight(x) - want).abs() < 1e-12, "Spline16({x})");
    assert!((Spline16.weight(-x) - want).abs() < 1e-12, "Spline16(-{x})");
  }
  for &(x, want) in s36 {
    assert!((Spline36.weight(x) - want).abs() < 1e-12, "Spline36({x})");
    assert!((Spline36.weight(-x) - want).abs() < 1e-12, "Spline36(-{x})");
  }
  for &(x, want) in s64 {
    assert!((Spline64.weight(x) - want).abs() < 1e-12, "Spline64({x})");
    assert!((Spline64.weight(-x) - want).abs() < 1e-12, "Spline64(-{x})");
  }
}

#[test]
fn ffmpeg_kernels_match_golden_fixtures() {
  // Independent exactness oracle for the two FFmpeg/swscale kernels. These
  // (x, weight) pairs were computed OUTSIDE the production Horner / `exp2`
  // code paths — by exact rational arithmetic over the Keys a = -0.6 cubic
  // (Python `fractions.Fraction`, cast to f64) for `SwscaleBicubic`, and by
  // the defining identity `2^(-3 x^2)` at points where the exponent is exact
  // for `Gaussian`. A transcription slip in a kernel coefficient or the
  // Gauss parameter (which the closed-form references in the per-kernel
  // profile tests evaluate, but in the same family of expression) shows up
  // here against literal numbers. The tolerance only absorbs f64 rounding.

  // SwscaleBicubic, Keys a = -0.6. Inner segment (|x| < 1):
  //   1.4|x|^3 - 2.4|x|^2 + 1; outer (1 <= |x| < 2): -0.6|x|^3 + 3|x|^2
  //   - 4.8|x| + 2.4. Exact rationals: e.g. weight(1/4) = 279/320,
  //   weight(1/2) = 23/40, weight(3/4) = 77/320, weight(5/4) = -27/320,
  //   weight(3/2) = -3/40, weight(7/4) = -9/320.
  let bicubic: &[(f64, f64)] = &[
    (0.25, 0.871875),
    (0.5, 0.575),
    (0.75, 0.240625),
    (1.25, -0.084375),
    (1.5, -0.075),
    (1.75, -0.028125),
  ];
  // Gaussian 2^(-3 x^2) at points with an exact (or near-exact) f64
  // exponent: weight(0.5) = 2^(-0.75), weight(1) = 2^(-3) = 0.125,
  // weight(1.5) = 2^(-6.75), weight(2) = 2^(-12) = 1/4096. The 2^(-0.75)
  // and 2^(-6.75) literals are the correctly-rounded f64 values of those
  // powers (verified against an independent base-2 exponential).
  let gaussian: &[(f64, f64)] = &[
    (0.5, 0.5946035575013605),
    (1.0, 0.125),
    (1.5, 0.009290680585958758),
    (2.0, 0.000244140625),
  ];
  for &(x, want) in bicubic {
    assert!(
      (SwscaleBicubic.weight(x) - want).abs() < 1e-12,
      "SwscaleBicubic({x})"
    );
    assert!(
      (SwscaleBicubic.weight(-x) - want).abs() < 1e-12,
      "SwscaleBicubic(-{x})"
    );
  }
  for &(x, want) in gaussian {
    assert!((Gaussian.weight(x) - want).abs() < 1e-12, "Gaussian({x})");
    assert!((Gaussian.weight(-x) - want).abs() < 1e-12, "Gaussian(-{x})");
  }
}

#[test]
fn phase4_kernels_match_golden_fixtures() {
  // Independent exactness oracle for the two RFC #238 Phase 4 kernels. These
  // (x, weight) literals were computed OUTSIDE the production code paths —
  // the `CubicBSpline` values by exact rational arithmetic over the B3
  // segment polynomials (Python `fractions.Fraction`, cast to f64), the
  // `BlackmanSinc` values by a direct (un-factored) evaluation of
  // `sinc(x) * (0.42 + 0.5 cos(pi x/3) + 0.08 cos(2 pi x/3))`. A
  // transcription slip in a kernel coefficient (which the closed-form
  // references in the per-kernel profile tests evaluate, but in the same
  // family of expression) shows up here against literal numbers; the
  // tolerance only absorbs f64 rounding across the two evaluation orders.

  // CubicBSpline B3, exact rationals: weight(1/4) = 235/384,
  // weight(1/2) = 23/48, weight(3/4) = 121/384, weight(5/4) = 9/128,
  // weight(3/2) = 1/48, weight(7/4) = 1/384.
  let bspline: &[(f64, f64)] = &[
    (0.25, 0.6119791666666666),
    (0.5, 0.4791666666666667),
    (0.75, 0.3151041666666667),
    (1.25, 0.0703125),
    (1.5, 0.020833333333333332),
    (1.75, 0.0026041666666666665),
  ];
  // BlackmanSinc (a = 3) at non-node fractional points (the integer nodes
  // are sinc zeros, covered by the profile test): direct evaluation of the
  // windowed sinc.
  let blackman: &[(f64, f64)] = &[
    (0.5, 0.5685095429999835),
    (0.75, 0.23214757965659177),
    (1.5, -0.0721502408683259),
    (2.5, 0.003436129515638274),
    (2.75, 0.0005172005528988453),
  ];
  for &(x, want) in bspline {
    assert!(
      (CubicBSpline.weight(x) - want).abs() < 1e-12,
      "CubicBSpline({x})"
    );
    assert!(
      (CubicBSpline.weight(-x) - want).abs() < 1e-12,
      "CubicBSpline(-{x})"
    );
  }
  for &(x, want) in blackman {
    assert!(
      (BlackmanSinc.weight(x) - want).abs() < 1e-12,
      "BlackmanSinc({x})"
    );
    assert!(
      (BlackmanSinc.weight(-x) - want).abs() < 1e-12,
      "BlackmanSinc(-{x})"
    );
  }
}

#[test]
fn gaussian_softens_a_scale_one_axis() {
  // A non-interpolating kernel does NOT pass a scale-1 axis through: under the
  // PIL half-pixel center convention a 1:1 axis evaluates the kernel at
  // integer offsets, so an interior output window is `weight(0)` at the
  // center and `weight(±1)` at the neighbours. For the Gaussian those
  // neighbours are nonzero (it softens the axis); for an interpolating kernel
  // they are zero (a bit-exact passthrough). This pins the documented
  // behaviour — a Gaussian resize of one axis also filters the other — and
  // proves it is accepted, not a bug.
  let n = 8;
  let g = FilterAxis::build(n, n, &Gaussian).expect("1:1 Gaussian axis");
  let c = FilterAxis::build(n, n, &CatmullRom).expect("1:1 CatmullRom axis");
  // Interior output (away from the clamped edges) for each.
  let j = 4;
  let (g_start, g_win) = g.span(j);
  let (c_start, c_win) = c.span(j);
  // CatmullRom (interpolating): one tap ~1, the rest ~0 — passthrough.
  let c_peak = c_win.iter().map(|&w| f64::from(w)).fold(0.0_f64, f64::max);
  assert!(
    c_peak > 0.99,
    "CatmullRom 1:1 axis is ~passthrough (peak {c_peak})"
  );
  // Gaussian: the peak tap is well below 1 and at least one neighbour carries
  // real weight — the axis is softened, not passed through.
  let g_peak = g_win.iter().map(|&w| f64::from(w)).fold(0.0_f64, f64::max);
  assert!(
    g_peak < 0.95,
    "Gaussian 1:1 axis is softened (peak {g_peak})"
  );
  let g_neighbours: f64 = g_win.iter().map(|&w| f64::from(w)).sum::<f64>() - g_peak;
  assert!(
    g_neighbours > 0.05,
    "Gaussian 1:1 axis spreads weight to neighbours ({g_neighbours})"
  );
  // Both windows still renormalize to 1 (brightness preserved).
  for (label, win) in [("gaussian", g_win), ("catmullrom", c_win)] {
    let sum: f64 = win.iter().map(|&w| f64::from(w)).sum();
    assert!(
      (sum - 1.0).abs() < 1e-4,
      "{label} 1:1 window sums to 1 ({sum})"
    );
  }
  let _ = (g_start, c_start);
}

#[test]
fn axis_windows_normalize_to_one() {
  // Every output window must sum to ~1 (PIL renormalizes after clamping),
  // so average brightness is preserved including at the clipped edges.
  for k in [
    &Triangle as &dyn FilterKernel,
    &CatmullRom as &dyn FilterKernel,
    &Lanczos3 as &dyn FilterKernel,
    &Mitchell as &dyn FilterKernel,
    &OpenCvCubic as &dyn FilterKernel,
    &Lanczos4 as &dyn FilterKernel,
    &SwscaleBicubic as &dyn FilterKernel,
    &Gaussian as &dyn FilterKernel,
    &Spline16 as &dyn FilterKernel,
    &Spline36 as &dyn FilterKernel,
    &Spline64 as &dyn FilterKernel,
    &BlackmanSinc as &dyn FilterKernel,
    &CubicBSpline as &dyn FilterKernel,
  ] {
    for &(in_size, out_size) in &[(8usize, 3usize), (64, 17), (1920, 640), (1000, 333)] {
      let axis = FilterAxis::build(in_size, out_size, k).expect("valid downscale");
      assert_eq!(axis.out_len(), out_size);
      for j in 0..out_size {
        let (start, win) = axis.span(j);
        assert!(start + win.len() <= in_size, "window in bounds");
        let sum: f64 = win.iter().map(|&w| f64::from(w)).sum();
        assert!(
          (sum - 1.0).abs() < 1e-4,
          "window {j} sum {sum} (in={in_size} out={out_size})"
        );
      }
    }
  }
}

#[test]
fn axis_center_convention_matches_pil_first_output() {
  // The first output's window start must follow PIL precompute_coeffs:
  // center = 0.5*scale, xmin = max(0, floor(center - support*filterscale)).
  // For 8 -> 4 (scale 2) with Triangle (support 1, filterscale 2):
  // center = 1.0, support = 2.0, xmin = floor(1 - 2) clamped to 0.
  let axis = FilterAxis::build(8, 4, &Triangle).unwrap();
  assert_eq!(axis.span(0).0, 0);
  // The last output center is (3.5)*2 = 7.0; xmax = min(8, ceil(7+2)) = 8,
  // xmin = floor(7-2) = 5 — the window hugs the right edge.
  assert_eq!(axis.span(3).0, 5);
}

#[test]
fn invalid_support_rejected() {
  // A hostile kernel cannot size an unsafe window: only non-finite and
  // non-positive supports are rejected before any allocation. A support
  // wider than the source is NOT rejected (see
  // `support_wider_than_source_builds`).
  struct Bad(f64);
  impl FilterKernel for Bad {
    fn support(&self) -> f64 {
      self.0
    }
    fn weight(&self, _x: f64) -> f64 {
      1.0
    }
  }
  for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
    let err = FilterAxis::build(16, 4, &Bad(bad)).unwrap_err();
    assert!(
      matches!(err, ResampleError::InvalidFilterSupport(_)),
      "support {bad} should reject, got {err:?}"
    );
  }
  // A finite positive support builds fine.
  assert!(FilterAxis::build(16, 4, &Bad(2.0)).is_ok());
}

#[test]
fn support_wider_than_source_builds() {
  // A support wider than the source (Lanczos3's 3 over a 1- or 2-wide axis,
  // CatmullRom's 2 over a 2-wide axis) is the ordinary narrow-source enlarge
  // case: the window clamps to `[0, in_size)` and normalizes over the
  // available samples, exactly as PIL does. It must build, not reject.
  assert!(FilterAxis::build(1, 7, &Lanczos3).is_ok());
  assert!(FilterAxis::build(2, 5, &Lanczos3).is_ok());
  assert!(FilterAxis::build(2, 9, &CatmullRom).is_ok());
  // A huge finite support clamps to the source the same way (at most
  // `in_size` taps), rather than overrunning.
  struct Wide;
  impl FilterKernel for Wide {
    fn support(&self) -> f64 {
      1e9
    }
    fn weight(&self, _x: f64) -> f64 {
      1.0
    }
  }
  assert!(FilterAxis::build(8, 32, &Wide).is_ok());
}

#[test]
fn tiny_positive_support_rejected_not_panic() {
  // A sub-ULP positive support passes the `> 0` / finite / `<= in_size`
  // checks, but for an integral projected center `floor(center - support)`
  // and `ceil(center + support)` round to the same integer (the offset is
  // absorbed below the center's ULP), leaving a zero-tap window. The old
  // build emitted that empty window and the overlap sweep then advanced its
  // lower pointer past `starts`, panicking. It must now reject with
  // `InvalidFilterSupport` instead.
  struct TinySupport;
  impl FilterKernel for TinySupport {
    fn support(&self) -> f64 {
      // Smaller than the ULP of the integral centers a 2:1 downscale
      // produces (center 1.0, 3.0, …), so `center ± support` collapses
      // back onto `center`.
      1e-20
    }
    fn weight(&self, _x: f64) -> f64 {
      1.0
    }
  }
  // scale == 2 makes every projected center `(xx + 0.5) * 2` an odd
  // integer, so the very first window already degenerates to zero taps.
  let err = FilterAxis::build(202, 101, &TinySupport).unwrap_err();
  assert!(
    matches!(err, ResampleError::InvalidFilterSupport(_)),
    "tiny support must reject, got {err:?}"
  );
}

#[test]
fn zero_tap_support_rejected_before_allocation() {
  // Hardening contract: an invalid (zero-tap) support is rejected by the
  // no-allocation dry pass BEFORE any plan table is sized. Arm the
  // first-table-reservation failpoint, then build a sub-ULP-support kernel
  // whose first integral center degenerates to an empty window. If the dry
  // pass runs first the error is `InvalidFilterSupport`; were the
  // reservation reached first it would be the armed `AllocationFailed`.
  struct TinySupport;
  impl FilterKernel for TinySupport {
    fn support(&self) -> f64 {
      1e-20
    }
    fn weight(&self, _x: f64) -> f64 {
      1.0
    }
  }
  arm_filter_axis_alloc_failure();
  let err = FilterAxis::build(202, 101, &TinySupport).unwrap_err();
  assert!(
    matches!(err, ResampleError::InvalidFilterSupport(_)),
    "zero-tap support must be rejected before allocation, got {err:?}"
  );
  // The dry pass returned before the failpoint check, so the flag is still
  // armed; consume it with a valid build (which now trips and reports the
  // armed `AllocationFailed`) so it cannot leak into a later test on this
  // thread — and incidentally re-confirming the failpoint was never reached
  // on the rejected build above.
  let drained = FilterAxis::build(16, 4, &Triangle).unwrap_err();
  assert!(
    matches!(drained, ResampleError::AllocationFailed(_)),
    "the still-armed failpoint must trip on the next valid build, got {drained:?}"
  );
}

#[test]
fn valid_support_hits_armed_alloc_failpoint() {
  // The failpoint is real and reachable for a VALID kernel: a normal
  // Triangle 2:1 downscale passes the dry pass, so it reaches the armed
  // first-table reservation and surfaces the recoverable `AllocationFailed`.
  arm_filter_axis_alloc_failure();
  let err = FilterAxis::build(16, 8, &Triangle).unwrap_err();
  assert!(
    matches!(err, ResampleError::AllocationFailed(_)),
    "a valid kernel must reach the armed table reservation, got {err:?}"
  );
}

#[test]
#[cfg(target_pointer_width = "64")]
fn huge_out_size_fails_fast_without_scan() {
  // Hostile-metadata DoS guard. `build` validates the zero-tap geometry in
  // O(1) (no per-output scan), so every adversarial axis below returns a
  // recoverable error immediately; a regression that reintroduced an
  // `O(out_size)` scan would hang here (the test would never complete) rather
  // than fail an assertion. (The huge values are 64-bit only — on a 32-bit
  // `usize` no axis is large enough for a normal support to fall sub-grid.)

  // (a) Identity axis at `usize::MAX`: the overflow preflight rejects it
  // (`out_size + 1` overflows `usize`) before any allocation.
  assert!(matches!(
    FilterAxis::build(usize::MAX, usize::MAX, &Triangle).unwrap_err(),
    ResampleError::Overflow(_) | ResampleError::AllocationFailed(_)
  ));

  // (b) The case the bounded validation closes: an identity axis whose extent
  // (`1 << 54`) gives the largest center a ULP (4.0) exceeding Triangle's
  // support (1.0). The support is sub-grid at the output extent, so it is
  // rejected as `InvalidFilterSupport` in O(1) — a per-output scan would have
  // ground through ~1.8e16 iterations here.
  assert!(matches!(
    FilterAxis::build(1usize << 54, 1usize << 54, &Triangle).unwrap_err(),
    ResampleError::InvalidFilterSupport(_)
  ));

  // (c) Right-edge clamp: near-2:1 huge dims where the rounded f64 `scale`
  // nudges the last center past `in_size`, so `floor(center - support)` would
  // exceed the clamped `xmax` and invert the window. The endpoint guard rejects
  // it in O(1) before any reservation; an unchecked `xmax - xmin` would
  // underflow (panic in debug, wrap in release) at the last window instead.
  assert!(matches!(
    FilterAxis::build(36028797018962971, 18014398509481485, &CatmullRom).unwrap_err(),
    ResampleError::InvalidFilterSupport(_)
  ));
}

#[test]
fn max_overlap_bounds_the_ring() {
  // The accumulator-ring capacity (max window overlap) must cover every
  // window open at a given source row. Cross-check the stored value
  // against a brute-force count over all source indices.
  for k in [
    &Triangle as &dyn FilterKernel,
    &CatmullRom as &dyn FilterKernel,
    &Lanczos3 as &dyn FilterKernel,
    &Mitchell as &dyn FilterKernel,
    &OpenCvCubic as &dyn FilterKernel,
    &Lanczos4 as &dyn FilterKernel,
    &SwscaleBicubic as &dyn FilterKernel,
    &Gaussian as &dyn FilterKernel,
    &Spline16 as &dyn FilterKernel,
    &Spline36 as &dyn FilterKernel,
    &Spline64 as &dyn FilterKernel,
    &BlackmanSinc as &dyn FilterKernel,
    &CubicBSpline as &dyn FilterKernel,
  ] {
    for &(in_size, out_size) in &[(64usize, 17usize), (200, 41), (1920, 360)] {
      let axis = FilterAxis::build(in_size, out_size, k).unwrap();
      let mut brute = 0usize;
      for y in 0..in_size {
        let mut c = 0usize;
        for j in 0..out_size {
          let (start, win) = axis.span(j);
          if y >= start && y < start + win.len() {
            c += 1;
          }
        }
        brute = brute.max(c);
      }
      assert_eq!(axis.max_overlap(), brute, "in={in_size} out={out_size}");
    }
  }
}
