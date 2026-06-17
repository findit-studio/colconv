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
fn axis_windows_normalize_to_one() {
  // Every output window must sum to ~1 (PIL renormalizes after clamping),
  // so average brightness is preserved including at the clipped edges.
  for k in [
    &Triangle as &dyn FilterKernel,
    &CatmullRom as &dyn FilterKernel,
    &Lanczos3 as &dyn FilterKernel,
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
  // A hostile kernel cannot size an unsafe window: non-finite, zero, and
  // over-extent supports are all rejected before any allocation.
  struct Bad(f64);
  impl FilterKernel for Bad {
    fn support(&self) -> f64 {
      self.0
    }
    fn weight(&self, _x: f64) -> f64 {
      1.0
    }
  }
  for bad in [f64::NAN, f64::INFINITY, 0.0, -1.0, 1e9] {
    let err = FilterAxis::build(16, 4, &Bad(bad)).unwrap_err();
    assert!(
      matches!(err, ResampleError::InvalidFilterSupport(_)),
      "support {bad} should reject, got {err:?}"
    );
  }
  // A finite in-bounds support builds fine.
  assert!(FilterAxis::build(16, 4, &Bad(2.0)).is_ok());
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
