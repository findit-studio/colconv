//! One-shot polynomial-derivation tool for the Tier 12 (Xyz12) sRGB
//! OETF upper segment.
//!
//! Run once at commit time:
//!
//! ```ignore
//! cargo run --example derive_oetf_polynomial --release
//! ```
//!
//! Prints a piecewise-minimax polynomial fit for the upper segment of
//! the sRGB-shape OETF:
//!
//! ```text
//!   f(c) = 1.055 * c^(1/2.4) - 0.055,   c ∈ [0.0031308, 1.0]
//! ```
//!
//! The lower segment (`12.92 * c`, `c < 0.0031308`) is exact under
//! f32 and does not need a polynomial fit; only the upper segment does.
//!
//! # Why Remez (and not least-squares)
//!
//! Least-squares fitting minimises L2 error and concentrates *all* the
//! residual at the segment boundaries; the previous attempt at this
//! ship measured ~11 M ULP error which is unusable. The Remez exchange
//! algorithm minimises the L∞ (uniform / sup) norm — yielding an
//! *equi-oscillating* error curve where the polynomial's peak ULP
//! error is the same at every interior alternation point. That's the
//! optimal shape for a fixed-degree polynomial on a closed interval.
//!
//! # Reference: f64-narrowed OETF
//!
//! The reference is computed in f64 and narrowed to f32 once at the
//! end (B' decision):
//!
//! ```text
//!   ref(c_f32) = (1.055_f64 * (c_f32 as f64).powf(1.0/2.4) - 0.055_f64) as f32
//! ```
//!
//! Rationale: `f32::powf` itself is platform-dependent and ~2 ULP off
//! mathematical truth — chasing it as the reference made convergence
//! impossible (a moving target). The f64-narrowed reference is closer
//! to mathematically-correct sRGB OETF; both the polynomial and the
//! scalar fallback chase the same reference, so scalar-vs-SIMD parity
//! becomes 0 ULP by construction. Outputs may diverge from `f32::powf`
//! by up to ~4 ULP, but `f32::powf` itself is ~2 ULP off truth, so this
//! is strictly closer to correct.
//!
//! All math runs in f64; the final f32 narrow happens once at the
//! end. Doing the post-process `* 1.055 - 0.055` after a partial
//! narrow to f32 re-introduces wobble (the codex Phase A wall
//! finding) — keeping every operation in f64 until the last narrow
//! avoids that catastrophic-cancellation pathway.
//!
//! # Segmenting strategy
//!
//! `c^(1/2.4)` has highest curvature at `c → 0`, so a single uniform
//! polynomial across `[0.0031308, 1.0]` would need an absurd degree
//! (~25+) to achieve ≤ 2 ULP at the lower end where the output
//! magnitude is ~0.04 (so 2 ULP ≈ 6e-9 absolute). We split into
//! geometrically-spaced segments so the curvature-per-segment is
//! roughly uniform, then run Remez per-segment until each segment's
//! max ULP ≤ 2 over a 65 536-point dense sweep.
//!
//! # Output
//!
//! Prints the segment table (boundaries + per-segment Horner
//! coefficients) ready to paste into
//! `src/row/scalar/xyz12_constants.rs`.
//!
//! Each segment carries one polynomial of fixed degree `DEG`. The
//! kernel evaluates the polynomial inline via Horner; segment lookup
//! is a small `if/else` chain (3–5 segments, branchless via
//! comparison-select for SIMD).

#![allow(clippy::needless_range_loop)]

// ---------------------------------------------------------------------------
// Reference function: target OETF upper segment in f64.
// ---------------------------------------------------------------------------

const SEGMENT_LO_BOUND: f64 = 0.003_130_8_f64;
const SEGMENT_HI_BOUND: f64 = 1.0_f64;

/// Reference: f64 evaluation of the OETF upper segment.
///
/// Used by Remez — minimax in f64 against the *true* function, then
/// f32-narrowed coefficients. The polynomial's *output* ULP target is
/// measured against [`reference_f64_narrowed_f32`] below (which is
/// what the scalar kernel will run in production).
fn f_ref(c: f64) -> f64 {
  1.055_f64 * c.powf(1.0_f64 / 2.4_f64) - 0.055_f64
}

/// Reference oracle: f64 evaluation narrowed to f32 once at the end.
///
/// This is what both the scalar fallback and the polynomial must match
/// within ≤ 2 ULP. Pure f32 (`f32::powf`) is ~2 ULP off mathematical
/// truth and platform-dependent; this f64-narrowed form is essentially
/// correctly-rounded f32 OETF (the f64 powf is ≤ 1 ULP off truth in
/// f64, and the final f32 narrow loses ≤ 0.5 f32 ULP), so chasing it
/// is strictly closer to correct than chasing `f32::powf`.
///
/// All arithmetic done in f64; narrow once at the very end so we never
/// lose precision in the post-process `* 1.055 - 0.055` step (doing
/// that step in f32 after a partial-narrow re-introduces wobble — the
/// codex Phase A wall finding).
fn reference_f64_narrowed_f32(c_f32: f32) -> f32 {
  let c64 = c_f32 as f64;
  (1.055_f64 * c64.powf(1.0_f64 / 2.4_f64) - 0.055_f64) as f32
}

// ---------------------------------------------------------------------------
// Generic Remez exchange algorithm.
// ---------------------------------------------------------------------------

/// Remez exchange algorithm, second algorithm (Pólya).
///
/// Given a function `f` continuous on `[a, b]`, finds the polynomial
/// of degree `n` that minimises the maximum absolute error.
///
/// Iterates by:
/// 1. Solving a linear system at the current `n+2` reference points
///    for `(c_0, ..., c_n, E)` such that `Σ c_k x_i^k - f(x_i) =
///    (-1)^i · E` at every reference.
/// 2. Locating the new reference points: each new x_i is the location
///    where the error function `p(x) - f(x)` reaches a local extremum
///    of alternating sign on the corresponding sub-interval. Found
///    via golden-section search on each sub-interval.
/// 3. Repeats until reference-point movement is below tolerance.
///
/// Returns the polynomial coefficients in ascending order
/// (`c[0] + c[1]·(x - center) + ... + c[n]·(x - center)^n`), the
/// achieved L∞ error, and the center used.
///
/// The centered representation (`x - center` instead of `x`) keeps
/// coefficient magnitudes balanced when the segment is far from the
/// origin, dramatically reducing f32-evaluation roundoff in the
/// Horner step versus the textbook `c_0 + c_1*x + ...` form. Center
/// is chosen as the segment midpoint.
fn remez<F: Fn(f64) -> f64>(
  f: F,
  a: f64,
  b: f64,
  degree: usize,
  iters: usize,
) -> (Vec<f64>, f64, f64) {
  let n = degree + 2;
  let center = 0.5 * (a + b);

  // Initial reference points: Chebyshev nodes mapped to [a, b]. Robust
  // starting point — the alternation pattern of a Chebyshev interpolant
  // already approximates Remez's optimal alternation.
  let mut refs = vec![0.0_f64; n];
  for k in 0..n {
    let theta = std::f64::consts::PI * ((k as f64) + 0.5) / (n as f64);
    let cheb_x = -theta.cos(); // in [-1, 1]
    refs[k] = 0.5 * (a + b) + 0.5 * (b - a) * cheb_x;
  }
  refs.sort_by(|x, y| x.partial_cmp(y).unwrap());
  // Pin the endpoints to keep the system non-degenerate.
  refs[0] = a;
  refs[n - 1] = b;

  let mut coeffs = vec![0.0_f64; degree + 1];
  let mut error_e = 0.0_f64;

  for _iter in 0..iters {
    // 1. Solve the (n × n) linear system.
    //
    //    [1 (x_i - c) (x_i - c)^2 ... (-1)^i] · [c_0 c_1 ... c_d E]^T = f(x_i)
    let mut mat = vec![vec![0.0_f64; n]; n];
    let mut rhs = vec![0.0_f64; n];
    for i in 0..n {
      let dx = refs[i] - center;
      for j in 0..=degree {
        mat[i][j] = dx.powi(j as i32);
      }
      mat[i][degree + 1] = if i % 2 == 0 { 1.0 } else { -1.0 };
      rhs[i] = f(refs[i]);
    }
    let sol = match gaussian_elim(mat, rhs) {
      Some(s) => s,
      None => break,
    };
    coeffs[..(degree + 1)].copy_from_slice(&sol[..(degree + 1)]);
    error_e = sol[degree + 1].abs();

    // 2. Find new reference points: extrema of `p - f` on each
    // sub-interval [refs[i], refs[i+1]] plus the endpoints.
    let mut new_refs = Vec::with_capacity(n);
    new_refs.push(a);
    for i in 0..(n - 1) {
      let lo = refs[i];
      let hi = refs[i + 1];
      // Search for the extremum of `|p(x) - f(x)|` on (lo, hi).
      let extremum_x = find_extremum(&coeffs, &f, center, lo, hi);
      // Skip endpoints — already added in the bracket, and reusing
      // them yields a degenerate system.
      if extremum_x > a + 1e-15 && extremum_x < b - 1e-15 {
        new_refs.push(extremum_x);
      }
    }
    new_refs.push(b);

    // Trim/pad to exactly `n` points (extrema-search rarely returns
    // more than `n - 2` interior points for well-behaved smooth
    // targets).
    if new_refs.len() < n {
      // Insufficient interior extrema — fall back to current refs
      // (will converge slowly but won't loop forever).
      break;
    }
    new_refs.truncate(n);
    new_refs.sort_by(|x, y| x.partial_cmp(y).unwrap());

    // 3. Convergence: stop when reference points have stabilised.
    let max_move = refs
      .iter()
      .zip(new_refs.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f64, f64::max);
    refs.copy_from_slice(&new_refs);
    if max_move < 1e-15 {
      break;
    }
  }

  (coeffs, error_e, center)
}

/// Gaussian elimination with partial pivoting. Returns `None` if the
/// system is singular (zero pivot).
fn gaussian_elim(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
  let n = b.len();
  for k in 0..n {
    let mut pivot = k;
    let mut pivot_abs = a[k][k].abs();
    for i in (k + 1)..n {
      if a[i][k].abs() > pivot_abs {
        pivot = i;
        pivot_abs = a[i][k].abs();
      }
    }
    if pivot_abs < 1e-30 {
      return None;
    }
    if pivot != k {
      a.swap(k, pivot);
      b.swap(k, pivot);
    }
    for i in (k + 1)..n {
      let f = a[i][k] / a[k][k];
      for j in k..n {
        a[i][j] -= f * a[k][j];
      }
      b[i] -= f * b[k];
    }
  }
  let mut x = vec![0.0_f64; n];
  for i in (0..n).rev() {
    let mut s = b[i];
    for j in (i + 1)..n {
      s -= a[i][j] * x[j];
    }
    x[i] = s / a[i][i];
  }
  Some(x)
}

/// Evaluate centered polynomial `c[0] + c[1]·(x - center) + ... +
/// c[d]·(x - center)^d` via Horner (right-to-left).
fn poly_eval(coeffs: &[f64], center: f64, x: f64) -> f64 {
  let dx = x - center;
  let mut acc = 0.0_f64;
  for &c in coeffs.iter().rev() {
    acc = acc * dx + c;
  }
  acc
}

/// Find the location of the maximum of `|p(x) - f(x)|` on the closed
/// interval `[lo, hi]` via golden-section search on the squared error
/// (which is smooth and unimodal on each sub-interval after Remez has
/// converged toward equi-oscillation).
fn find_extremum<F: Fn(f64) -> f64>(coeffs: &[f64], f: &F, center: f64, lo: f64, hi: f64) -> f64 {
  // Switch to dense scan + parabolic refinement: golden-section
  // sometimes converges to the wrong root when the error has multiple
  // local extrema. A coarse sweep finds the right basin first.
  const SCAN: usize = 256;
  let mut best_x = lo;
  let mut best_err = (poly_eval(coeffs, center, lo) - f(lo)).abs();
  for k in 1..SCAN {
    let t = (k as f64) / (SCAN as f64);
    let x = lo + (hi - lo) * t;
    let err = (poly_eval(coeffs, center, x) - f(x)).abs();
    if err > best_err {
      best_err = err;
      best_x = x;
    }
  }
  // Refine via golden-section on a small bracket around best_x.
  let bracket = (hi - lo) / (SCAN as f64);
  let mut a = (best_x - bracket).max(lo);
  let mut d = (best_x + bracket).min(hi);
  let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
  let invphi = 1.0 / phi;
  let mut b = d - (d - a) * invphi;
  let mut c = a + (d - a) * invphi;
  for _ in 0..100 {
    let fb = -(poly_eval(coeffs, center, b) - f(b)).abs();
    let fc = -(poly_eval(coeffs, center, c) - f(c)).abs();
    if fb < fc {
      d = c;
      c = b;
      b = d - (d - a) * invphi;
    } else {
      a = b;
      b = c;
      c = a + (d - a) * invphi;
    }
    if (d - a).abs() < 1e-15 {
      break;
    }
  }
  0.5 * (a + d)
}

// ---------------------------------------------------------------------------
// f32 ULP-distance measurement.
// ---------------------------------------------------------------------------

/// Distance in f32 ULPs between two finite f32 values.
/// Encodes the IEEE-754 monotone bijection between f32 and i32 (sign-
/// magnitude → biased) so that subtraction in i32 == ULP distance.
fn f32_ulps(a: f32, b: f32) -> u64 {
  let a_b = f32_to_sortable(a);
  let b_b = f32_to_sortable(b);
  a_b.abs_diff(b_b)
}

fn f32_to_sortable(x: f32) -> i64 {
  let bits = x.to_bits() as i32;
  if bits >= 0 {
    bits as i64
  } else {
    (i32::MIN as i64) - (bits as i64)
  }
}

// ---------------------------------------------------------------------------
// Polynomial-evaluation modes (mirror what the kernel does in f32).
// ---------------------------------------------------------------------------

/// f32 Horner evaluation matching what the SIMD/scalar kernel will
/// run. Coefficients are passed as `&[f32]` (already narrowed). FMA
/// not assumed — pure mul+add to match the most conservative target
/// path.
///
/// Centered representation: evaluates
/// `c[0] + c[1]·(x - center) + ... + c[d]·(x - center)^d`. The kernel
/// does the same `(x - center)` step before Horner.
fn poly_eval_f32(coeffs_f32: &[f32], center_f32: f32, x: f32) -> f32 {
  let dx = x - center_f32;
  let mut acc = 0.0_f32;
  for &c in coeffs_f32.iter().rev() {
    acc = acc * dx + c;
  }
  acc
}

// ---------------------------------------------------------------------------
// Segment fitting + ULP measurement.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Segment {
  lo: f64,
  hi: f64,
  center_f32: f32,
  coeffs_f32: Vec<f32>,
  l_inf_err_f64: f64,
  max_ulp_f32: u64,
  max_ulp_x_f32: f32,
}

fn fit_segment(lo: f64, hi: f64, degree: usize) -> Segment {
  let (coeffs_f64, l_inf_err_f64, center_f64) = remez(f_ref, lo, hi, degree, 200);
  let coeffs_f32: Vec<f32> = coeffs_f64.iter().map(|&c| c as f32).collect();
  let center_f32 = center_f64 as f32;

  // Measure ULP error vs f64-narrowed reference at 65 536 evenly-
  // spaced sample points across [lo, hi]. The reference is what the
  // scalar path runs in production (B' decision — see module docs);
  // this is what the polynomial must match within ≤ 2 ULP.
  const SAMPLES: usize = 65_536;
  let mut max_ulp = 0_u64;
  let mut max_ulp_x = 0.0_f32;
  for i in 0..SAMPLES {
    let t = (i as f64) / ((SAMPLES - 1) as f64);
    let x_f64 = lo + (hi - lo) * t;
    let x_f32 = x_f64 as f32;
    let poly = poly_eval_f32(&coeffs_f32, center_f32, x_f32);
    let reference = reference_f64_narrowed_f32(x_f32);
    let ulps = f32_ulps(poly, reference);
    if ulps > max_ulp {
      max_ulp = ulps;
      max_ulp_x = x_f32;
    }
  }
  Segment {
    lo,
    hi,
    center_f32,
    coeffs_f32,
    l_inf_err_f64,
    max_ulp_f32: max_ulp,
    max_ulp_x_f32: max_ulp_x,
  }
}

// ---------------------------------------------------------------------------
// Boundary search: binary-search for the smallest piecewise schedule
// that gets every segment to ≤ 2 ULP at a fixed degree.
// ---------------------------------------------------------------------------

/// Geometric segmentation of `[lo, hi]` into `n` segments via
/// log-space midpoints — produces near-uniform relative-error
/// distribution for monotone power-law-like targets.
fn geometric_segments(lo: f64, hi: f64, n: usize) -> Vec<(f64, f64)> {
  let log_lo = lo.ln();
  let log_hi = hi.ln();
  let step = (log_hi - log_lo) / (n as f64);
  let mut bounds = Vec::with_capacity(n + 1);
  for i in 0..=n {
    bounds.push((log_lo + step * (i as f64)).exp());
  }
  bounds[0] = lo;
  bounds[n] = hi;
  let mut segs = Vec::with_capacity(n);
  for i in 0..n {
    segs.push((bounds[i], bounds[i + 1]));
  }
  segs
}

// ---------------------------------------------------------------------------
// Main: try increasing degree / segment count until ≤ 2 ULP achieved.
// ---------------------------------------------------------------------------

fn main() {
  println!("# sRGB-shape OETF upper segment polynomial fit");
  println!(
    "# f(c) = 1.055 * c^(1/2.4) - 0.055,  c ∈ [{}, {}]",
    SEGMENT_LO_BOUND, SEGMENT_HI_BOUND
  );
  println!("# Target: max ULP ≤ 2 vs f64-narrowed reference at 65 536 sample points\n");

  // Strategy: scan (segments, degree) combinations from cheap to
  // expensive until the maximum ULP across all segments is ≤ 2.
  let candidates: &[(usize, usize)] = &[
    // (n_segments, degree)
    (1, 6),
    (1, 8),
    (1, 10),
    (1, 12),
    (2, 6),
    (2, 8),
    (3, 5),
    (3, 6),
    (3, 7),
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 4),
    (5, 5),
    (6, 4),
    (8, 3),
    (8, 4),
    (12, 3),
    (12, 4),
    (16, 3),
    (16, 4),
    (24, 3),
    (32, 3),
    (48, 3),
    (64, 3),
    (96, 3),
    (128, 3),
    (24, 4),
    (32, 4),
    (48, 4),
    (64, 4),
    (96, 4),
    (24, 5),
    (32, 5),
    (48, 5),
    (24, 6),
    (32, 6),
    (48, 6),
    (24, 7),
    (32, 7),
    (24, 8),
    (32, 8),
    (16, 8),
    (16, 10),
    (16, 12),
    (192, 3),
    (256, 3),
    (384, 3),
    (512, 3),
    (768, 3),
    (1024, 3),
    (192, 2),
    (256, 2),
    (512, 2),
    (1024, 2),
  ];
  let target_ulp = 2_u64;
  let mut chosen: Option<(usize, usize, Vec<Segment>)> = None;
  for &(n_segs, deg) in candidates {
    let segs = geometric_segments(SEGMENT_LO_BOUND, SEGMENT_HI_BOUND, n_segs);
    let fitted: Vec<Segment> = segs
      .into_iter()
      .map(|(lo, hi)| fit_segment(lo, hi, deg))
      .collect();
    let max_ulp = fitted.iter().map(|s| s.max_ulp_f32).max().unwrap();
    let coef_count = fitted.iter().map(|s| s.coeffs_f32.len()).sum::<usize>();
    println!(
      "  candidate: {} segs × deg {}  →  max ULP = {}  ({} f32 coeffs total)",
      n_segs, deg, max_ulp, coef_count
    );
    if max_ulp <= target_ulp && chosen.is_none() {
      chosen = Some((n_segs, deg, fitted));
      break;
    }
  }

  let Some((n_segs, deg, segs)) = chosen else {
    println!(
      "\n# NO COMBINATION CONVERGED to ≤ {} ULP across the candidate sweep.",
      target_ulp
    );
    println!(
      "# Surface this finding and propose alternative (option 2: scalar `powf` per SIMD lane)."
    );
    std::process::exit(2);
  };

  println!("\n# Chosen schedule: {} segments × degree {}", n_segs, deg);
  println!("#");
  println!("# Per-segment summary:");
  let max_ulp_overall = segs.iter().map(|s| s.max_ulp_f32).max().unwrap();
  for (i, s) in segs.iter().enumerate() {
    println!(
      "#   seg {}: [{:.10}, {:.10}]  L∞(f64)={:.3e}  max_ulp(f32)={}  at x={:.6e}",
      i, s.lo, s.hi, s.l_inf_err_f64, s.max_ulp_f32, s.max_ulp_x_f32
    );
  }
  println!("# Max ULP overall: {}", max_ulp_overall);

  // -----------------------------------------------------------------
  // Print constants ready to paste into xyz12_constants.rs.
  // -----------------------------------------------------------------
  println!("\n// ---- BEGIN COPY-PASTE INTO xyz12_constants.rs ----");
  println!("/// Number of segments in the piecewise-minimax sRGB OETF polynomial.");
  println!(
    "pub(crate) const OETF_POLY_SEGMENTS: usize = {};",
    segs.len()
  );
  println!("/// Polynomial degree per segment.");
  println!("pub(crate) const OETF_POLY_DEGREE: usize = {};", deg);

  println!(
    "\n/// Segment boundaries in the polynomial OETF (low edge of each\n\
     /// segment, in ascending order). The upper bound of the final\n\
     /// segment is implicit at `1.0`. A sample `c >= bound[i]` falls\n\
     /// into segment `i` (or higher); the dispatch loop walks the\n\
     /// table from highest to lowest."
  );
  println!("pub(crate) const OETF_POLY_SEG_BOUNDS: [f32; OETF_POLY_SEGMENTS] = [",);
  for s in &segs {
    println!("  {}_f32,", format_f32_lit(s.lo as f32));
  }
  println!("];");

  println!(
    "\n/// Per-segment Horner-evaluation centers. The polynomial is\n\
     /// evaluated as `c[0] + c[1]·(x - center) + ... + c[d]·(x -\n\
     /// center)^d`. Centering keeps coefficient magnitudes balanced,\n\
     /// dramatically reducing f32 roundoff vs the naive `c[0] + c[1]·x +\n\
     /// ...` form. Center is the segment midpoint."
  );
  println!("pub(crate) const OETF_POLY_SEG_CENTERS: [f32; OETF_POLY_SEGMENTS] = [",);
  for s in &segs {
    println!("  {}_f32,", format_f32_lit(s.center_f32));
  }
  println!("];");

  println!(
    "\n/// Per-segment ascending-order Horner coefficients (constant\n\
     /// term first, leading term last). Stored as a flat `[f32; N×(D+1)]`\n\
     /// so both scalar and SIMD evaluators can index by\n\
     /// `seg * (DEG + 1) + i` without an extra indirection. Coefficients\n\
     /// are for the *centered* polynomial (see `OETF_POLY_SEG_CENTERS`)."
  );
  let total = segs.len() * (deg + 1);
  println!("pub(crate) const OETF_POLY_COEFFS: [f32; {}] = [", total);
  for (si, s) in segs.iter().enumerate() {
    println!("  // segment {}: x ∈ [{:.10}, {:.10}]", si, s.lo, s.hi);
    for c in &s.coeffs_f32 {
      println!("  {}_f32,", format_f32_lit(*c));
    }
  }
  println!("];");
  println!("// ---- END COPY-PASTE ----");

  // -----------------------------------------------------------------
  // Final verification print: 65 536 sample sweep across full upper
  // segment range, max ULP across the whole pipeline.
  // -----------------------------------------------------------------
  println!(
    "\n# Whole-range verification (65 536 samples across [{}, 1]):",
    SEGMENT_LO_BOUND
  );
  let mut overall_max_ulp = 0_u64;
  let mut overall_x = 0.0_f32;
  for i in 0..65_536 {
    let t = (i as f64) / 65_535.0_f64;
    let c_f64 = SEGMENT_LO_BOUND + (SEGMENT_HI_BOUND - SEGMENT_LO_BOUND) * t;
    let c_f32 = c_f64 as f32;
    // Pick the segment.
    let mut seg_idx = 0;
    for (idx, s) in segs.iter().enumerate() {
      if (c_f32 as f64) >= s.lo - 1e-12 {
        seg_idx = idx;
      }
    }
    let s = &segs[seg_idx];
    let poly = poly_eval_f32(&s.coeffs_f32, s.center_f32, c_f32);
    let reference = reference_f64_narrowed_f32(c_f32);
    let ulps = f32_ulps(poly, reference);
    if ulps > overall_max_ulp {
      overall_max_ulp = ulps;
      overall_x = c_f32;
    }
  }
  println!(
    "# Max ULP across full sweep: {} at x = {:.6e}",
    overall_max_ulp, overall_x
  );
  if overall_max_ulp <= target_ulp {
    println!("# OK — polynomial meets ≤ {} ULP target.", target_ulp);
  } else {
    println!(
      "# FAIL — polynomial exceeds {} ULP target. Re-run with a finer schedule.",
      target_ulp
    );
    std::process::exit(2);
  }
}

/// Format a f32 as a Rust literal that round-trips to the same f32
/// value when re-parsed. Uses `{:.10e}` to ensure precision; underscores
/// are not added (the exponent form is concise enough).
fn format_f32_lit(x: f32) -> String {
  if x.is_nan() {
    return "f32::NAN".to_owned();
  }
  if x.is_infinite() {
    return if x > 0.0 {
      "f32::INFINITY".to_owned()
    } else {
      "f32::NEG_INFINITY".to_owned()
    };
  }
  // `{:e}` in Rust uses `e` rather than `E` and emits the minimal
  // representation that round-trips for f32.
  format!("{:e}", x)
}
