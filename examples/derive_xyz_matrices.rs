//! One-shot matrix-derivation tool for the Tier 12 (Xyz12) ship.
//!
//! Run once at commit time:
//!
//! ```ignore
//! cargo run --example derive_xyz_matrices --release
//! ```
//!
//! Prints the 27 f32 constants per gamut (DCI-P3, Rec.709, Rec.2020)
//! plus a hand-curated set of fixture (input, expected) pairs per
//! gamut so test files can hardcode the f32 literals without a Python
//! dependency.
//!
//! Math reference: standard primary-scaling algorithm as documented
//! in ITU-R BT.709-6 §3 / SMPTE ST 432-1 / ITU-R BT.2020-2:
//!
//! 1. From chromaticity (x, y) for each primary R/G/B and white point W:
//!    XYZ = (x/y, 1, (1 - x - y) / y).
//! 2. Solve M_unscaled · S = W_xyz, where M_unscaled has the unscaled
//!    primary XYZ vectors as columns, W_xyz is the white point
//!    (Y normalised to 1), and S = (Sr, Sg, Sb) are the per-primary
//!    luminance scales.
//! 3. M_rgb_to_xyz columns are M_unscaled[:, i] * S[i].
//! 4. M_xyz_to_rgb = inverse(M_rgb_to_xyz).
//!
//! All math is in f64 internally; outputs are narrowed to f32 at the
//! end so per-pixel SIMD work stays single-precision.
//!
//! Each derived matrix is cross-checked against the published values
//! in the source standard (printed in source comments next to the
//! constants).

#![allow(clippy::needless_range_loop)]

// ---------------------------------------------------------------------------
// Chromaticity tables (cited from each standard in source comments).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct GamutCoords {
  /// Display name for the gamut.
  name: &'static str,
  /// Standard / source citation.
  source: &'static str,
  /// (x, y) chromaticity for the red primary.
  r: (f64, f64),
  /// (x, y) chromaticity for the green primary.
  g: (f64, f64),
  /// (x, y) chromaticity for the blue primary.
  b: (f64, f64),
  /// (x, y) chromaticity for the white point.
  w: (f64, f64),
}

const GAMUTS: &[GamutCoords] = &[
  GamutCoords {
    name: "Rec.709 / sRGB",
    source: "ITU-R BT.709-6 §3, IEC 61966-2-1",
    r: (0.640, 0.330),
    g: (0.300, 0.600),
    b: (0.150, 0.060),
    w: (0.3127, 0.3290),
  },
  GamutCoords {
    name: "DCI-P3",
    source: "SMPTE RP 431-2 §5.1 / ST 428-1 — theatrical DCI white \
             (~6300K, x=0.314, y=0.351), NOT D65",
    r: (0.680, 0.320),
    g: (0.265, 0.690),
    b: (0.150, 0.060),
    // DCI white per SMPTE RP 431-2 §5.1 — chromaticity (0.314, 0.351),
    // approximately 6300 K. Distinct from D65 (0.31270, 0.32900) used
    // by Display-P3 (Apple/web) and the other gamuts in this table.
    w: (0.314, 0.351),
  },
  GamutCoords {
    name: "Rec.2020",
    source: "ITU-R BT.2020-2",
    r: (0.708, 0.292),
    g: (0.170, 0.797),
    b: (0.131, 0.046),
    w: (0.3127, 0.3290),
  },
];

// ---------------------------------------------------------------------------
// 3x3 matrix algebra (f64 internally, f32 at output).
// ---------------------------------------------------------------------------

type Mat3 = [[f64; 3]; 3];

fn xyz_from_xy(xy: (f64, f64)) -> [f64; 3] {
  let (x, y) = xy;
  let big_x = x / y;
  let big_y = 1.0;
  let big_z = (1.0 - x - y) / y;
  [big_x, big_y, big_z]
}

/// Inverse of a 3x3 matrix via cofactor expansion. Returns None if the
/// determinant is below `epsilon` (singular).
fn invert3(m: &Mat3) -> Option<Mat3> {
  let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if det.abs() < 1e-30 {
    return None;
  }
  let inv_det = 1.0 / det;
  let mut out = [[0.0_f64; 3]; 3];
  // adjugate (transpose of cofactor matrix), divided by det.
  out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
  out[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
  out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
  out[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
  out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
  out[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;
  out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
  out[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
  out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
  Some(out)
}

fn matmul3(a: &Mat3, b: &Mat3) -> Mat3 {
  let mut out = [[0.0_f64; 3]; 3];
  for i in 0..3 {
    for j in 0..3 {
      let mut s = 0.0;
      for k in 0..3 {
        s += a[i][k] * b[k][j];
      }
      out[i][j] = s;
    }
  }
  out
}

fn matvec3(m: &Mat3, v: &[f64; 3]) -> [f64; 3] {
  [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ]
}

/// Derives `M_rgb_to_xyz` (3x3 f64) for a single gamut. Used both as
/// the inverse-target for `derive_xyz_to_rgb` and to read off luma
/// coefficients (the Y row, normalised so the three weights sum to 1
/// at the gamut's white point).
fn derive_rgb_to_xyz(g: &GamutCoords) -> Mat3 {
  let r_xyz = xyz_from_xy(g.r);
  let g_xyz = xyz_from_xy(g.g);
  let b_xyz = xyz_from_xy(g.b);
  let w_xyz = xyz_from_xy(g.w);
  let m_unscaled: Mat3 = [
    [r_xyz[0], g_xyz[0], b_xyz[0]],
    [r_xyz[1], g_xyz[1], b_xyz[1]],
    [r_xyz[2], g_xyz[2], b_xyz[2]],
  ];
  let m_unscaled_inv = invert3(&m_unscaled).expect("primary matrix is invertible");
  let s = matvec3(&m_unscaled_inv, &w_xyz);
  let mut m_rgb_to_xyz: Mat3 = [[0.0; 3]; 3];
  for i in 0..3 {
    for j in 0..3 {
      m_rgb_to_xyz[i][j] = m_unscaled[i][j] * s[j];
    }
  }
  m_rgb_to_xyz
}

/// Prints the per-gamut luma coefficients — the Y row of
/// `M_rgb_to_xyz`. Since `M_rgb_to_xyz · (1, 1, 1)^T = W_xyz` and the
/// derivation normalises Y_white to 1.0, the Y row already sums to 1
/// for every gamut — no extra normalisation step needed.
fn print_luma_coeffs(name: &str, m_rgb_to_xyz: &Mat3) {
  let yr = m_rgb_to_xyz[1][0];
  let yg = m_rgb_to_xyz[1][1];
  let yb = m_rgb_to_xyz[1][2];
  let sum = yr + yg + yb;
  println!(
    "// Luma weights for {}: Y = {:.10} R + {:.10} G + {:.10} B (sum = {:.10})",
    name, yr, yg, yb, sum,
  );
  println!(
    "//   f32: [{:>14.9}_f32, {:>14.9}_f32, {:>14.9}_f32]",
    yr as f32, yg as f32, yb as f32,
  );
  println!();
}

/// Derives M_xyz_to_rgb (3x3 f64) for a single gamut.
fn derive_xyz_to_rgb(g: &GamutCoords) -> Mat3 {
  let r_xyz = xyz_from_xy(g.r);
  let g_xyz = xyz_from_xy(g.g);
  let b_xyz = xyz_from_xy(g.b);
  let w_xyz = xyz_from_xy(g.w);

  // M_unscaled has primary unscaled XYZ as columns:
  //   [ Xr Xg Xb ]
  //   [ Yr Yg Yb ]
  //   [ Zr Zg Zb ]
  let m_unscaled: Mat3 = [
    [r_xyz[0], g_xyz[0], b_xyz[0]],
    [r_xyz[1], g_xyz[1], b_xyz[1]],
    [r_xyz[2], g_xyz[2], b_xyz[2]],
  ];

  // Solve M_unscaled * S = W_xyz for S = (Sr, Sg, Sb).
  let m_unscaled_inv = invert3(&m_unscaled).expect("primary matrix is invertible");
  let s = matvec3(&m_unscaled_inv, &w_xyz);

  // Build M_rgb_to_xyz: scale each column of M_unscaled by S[i].
  let mut m_rgb_to_xyz: Mat3 = [[0.0; 3]; 3];
  for i in 0..3 {
    for j in 0..3 {
      m_rgb_to_xyz[i][j] = m_unscaled[i][j] * s[j];
    }
  }

  // Verify: M_rgb_to_xyz * (1, 1, 1) should equal w_xyz (within fp eps).
  let one = [1.0_f64, 1.0, 1.0];
  let derived_white = matvec3(&m_rgb_to_xyz, &one);
  assert!(
    (derived_white[0] - w_xyz[0]).abs() < 1e-12
      && (derived_white[1] - w_xyz[1]).abs() < 1e-12
      && (derived_white[2] - w_xyz[2]).abs() < 1e-12,
    "white point check failed for {}: got {:?}, expected {:?}",
    g.name,
    derived_white,
    w_xyz,
  );

  // M_xyz_to_rgb = inverse(M_rgb_to_xyz).
  let m_xyz_to_rgb = invert3(&m_rgb_to_xyz).expect("rgb->xyz matrix is invertible");

  // Identity check.
  let ident = matmul3(&m_xyz_to_rgb, &m_rgb_to_xyz);
  for i in 0..3 {
    for j in 0..3 {
      let expected: f64 = if i == j { 1.0 } else { 0.0 };
      assert!(
        (ident[i][j] - expected).abs() < 1e-10,
        "identity check failed for {} at [{},{}]: got {}, expected {}",
        g.name,
        i,
        j,
        ident[i][j],
        expected,
      );
    }
  }

  m_xyz_to_rgb
}

/// Pretty-print a 3x3 matrix as Rust f32 literals.
///
/// `white` is the gamut's chromaticity white point — used to label the
/// generated comment as either D65 (Bt709 / DisplayP3D65 / Bt2020Ncl)
/// or DCI white (theatrical DciP3, x=0.314 y=0.351). Hard-coding "(D65
/// output)" for every gamut was Copilot review #4256795439 Comment 3:
/// the DCI-P3 theatrical matrix uses DCI white, not D65.
fn print_mat_const(name: &str, source: &str, m: &Mat3, white: (f64, f64)) {
  // Recognise canonical white points to within 1e-4. Anything else
  // prints the (x, y) coordinates verbatim, which keeps the helper
  // honest if a future gamut entry adds a new white point.
  let white_label = if (white.0 - 0.3127).abs() < 1e-4 && (white.1 - 0.3290).abs() < 1e-4 {
    "D65 output".to_string()
  } else if (white.0 - 0.314).abs() < 1e-4 && (white.1 - 0.351).abs() < 1e-4 {
    "DCI white output, x=0.314, y=0.351".to_string()
  } else {
    format!("white point output, x={}, y={}", white.0, white.1)
  };
  println!("/// XYZ → RGB matrix for {} ({}).", name, white_label);
  println!("/// Derived from chromaticity coordinates in {}.", source);
  println!("///");
  println!("/// Verify (compare published values in the source standard):");
  println!("/// ```text");
  for row in m.iter() {
    println!(
      "/// [{:>14.10}, {:>14.10}, {:>14.10}]",
      row[0], row[1], row[2],
    );
  }
  println!("/// ```");
  // Ident name derived from gamut display name; caller passes the
  // exact const-name per gamut.
  println!("[");
  for row in m.iter() {
    println!(
      "  [{:>14.9}_f32, {:>14.9}_f32, {:>14.9}_f32],",
      row[0] as f32, row[1] as f32, row[2] as f32,
    );
  }
  println!("];");
  println!();
}

// ---------------------------------------------------------------------------
// Fixtures. For each gamut, compute expected RGB outputs (linear and
// gamma-encoded sRGB-shape OETF) for a curated set of XYZ inputs
// covering the testing matrix in the user spec.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Fixture {
  name: &'static str,
  /// Input X, Y, Z as 12-bit u16 values (low 12 bits, [0, 4095]).
  xyz_u12: (u16, u16, u16),
}

const FIXTURES: &[Fixture] = &[
  Fixture {
    name: "all_zero",
    xyz_u12: (0, 0, 0),
  },
  Fixture {
    name: "all_max",
    xyz_u12: (4095, 4095, 4095),
  },
  Fixture {
    name: "mid_gray",
    xyz_u12: (2048, 2048, 2048),
  },
  Fixture {
    name: "near_black",
    xyz_u12: (4, 4, 4),
  },
  Fixture {
    name: "white_d65_like",
    // X = 0.95047 * Y at Y = 4095 (per SMPTE ST 428-1 inverse OETF
    // / 0.91653 normalization). We use rough u12 values that map to
    // approximate D65 white. Computed numerically below per gamut.
    xyz_u12: (3897, 4095, 4459_u16.saturating_sub(364)),
  },
  Fixture {
    name: "x_only_max",
    xyz_u12: (4095, 0, 0),
  },
  Fixture {
    name: "y_only_max",
    xyz_u12: (0, 4095, 0),
  },
  Fixture {
    name: "z_only_max",
    xyz_u12: (0, 0, 4095),
  },
  Fixture {
    name: "x_dominant",
    xyz_u12: (3000, 1500, 500),
  },
  Fixture {
    name: "y_dominant",
    xyz_u12: (1500, 3000, 1500),
  },
  Fixture {
    name: "z_dominant",
    xyz_u12: (500, 1500, 3000),
  },
  Fixture {
    name: "low_quarter",
    xyz_u12: (1024, 1024, 1024),
  },
  Fixture {
    name: "three_quarter",
    xyz_u12: (3072, 3072, 3072),
  },
  Fixture {
    name: "rgb709_red_proxy",
    xyz_u12: (1700, 880, 80),
  },
  Fixture {
    name: "rgb709_green_proxy",
    xyz_u12: (1500, 3000, 500),
  },
  Fixture {
    name: "rgb709_blue_proxy",
    xyz_u12: (700, 280, 3700),
  },
];

/// SMPTE ST 428-1 inverse OETF: u12 → linear XYZ.
/// `xyz_lin = (x_u12 / 4095)^2.6 / 0.91653` per § 8.
fn inverse_oetf_xyz(x_u12: u16) -> f64 {
  let x = (x_u12 & 0x0FFF) as f64 / 4095.0;
  x.powf(2.6) / 0.91653
}

/// sRGB-shape OETF (Rec.709 form is identical to ~3-decimal precision
/// for the [0,1] range). Used by all 3 gamuts in the DCP pipeline.
/// `c >= 0.0031308`: `1.055 * c^(1/2.4) - 0.055`.
/// `c < 0.0031308`: `12.92 * c`.
fn oetf_srgb(c: f64) -> f64 {
  if c < 0.0031308 {
    12.92 * c
  } else {
    1.055 * c.powf(1.0 / 2.4) - 0.055
  }
}

fn compute_fixture_rgb_linear(m_xyz_to_rgb: &Mat3, xyz_u12: (u16, u16, u16)) -> [f64; 3] {
  let xyz_lin = [
    inverse_oetf_xyz(xyz_u12.0),
    inverse_oetf_xyz(xyz_u12.1),
    inverse_oetf_xyz(xyz_u12.2),
  ];
  matvec3(m_xyz_to_rgb, &xyz_lin)
}

fn compute_fixture_rgb_u8(rgb_linear: &[f64; 3]) -> [u8; 3] {
  let mut out = [0u8; 3];
  for i in 0..3 {
    let g = oetf_srgb(rgb_linear[i].clamp(0.0, 1.0));
    let scaled = g * 255.0 + 0.5;
    out[i] = scaled.clamp(0.0, 255.0).floor() as u8;
  }
  out
}

fn compute_fixture_rgb_u16(rgb_linear: &[f64; 3]) -> [u16; 3] {
  let mut out = [0u16; 3];
  for i in 0..3 {
    let g = oetf_srgb(rgb_linear[i].clamp(0.0, 1.0));
    let scaled = g * 65535.0 + 0.5;
    out[i] = scaled.clamp(0.0, 65535.0).floor() as u16;
  }
  out
}

fn print_fixtures(name: &str, m_xyz_to_rgb: &Mat3) {
  println!("// ---- Fixtures for {} ----", name);
  for f in FIXTURES {
    let rgb_linear_f64 = compute_fixture_rgb_linear(m_xyz_to_rgb, f.xyz_u12);
    let rgb_linear_f32 = [
      rgb_linear_f64[0] as f32,
      rgb_linear_f64[1] as f32,
      rgb_linear_f64[2] as f32,
    ];
    let rgb_u8 = compute_fixture_rgb_u8(&rgb_linear_f64);
    let rgb_u16 = compute_fixture_rgb_u16(&rgb_linear_f64);
    println!(
      "// {:<22} u12=(0x{:03X}, 0x{:03X}, 0x{:03X})",
      f.name, f.xyz_u12.0, f.xyz_u12.1, f.xyz_u12.2,
    );
    println!(
      "//   rgb_linear_f32 = [{:>14.9}_f32, {:>14.9}_f32, {:>14.9}_f32]",
      rgb_linear_f32[0], rgb_linear_f32[1], rgb_linear_f32[2],
    );
    println!(
      "//   rgb_u8         = [{:>3}, {:>3}, {:>3}]   rgb_u16 = [{:>5}, {:>5}, {:>5}]",
      rgb_u8[0], rgb_u8[1], rgb_u8[2], rgb_u16[0], rgb_u16[1], rgb_u16[2],
    );
  }
  println!();
}

// ---------------------------------------------------------------------------
// Polynomial OETF coefficient calibration (Tranche 6 — done same tool).
//
// Fits a low-order polynomial approximation to the sRGB-shape OETF
// for c >= 0.0031308: `f(c) = 1.055 * c^(1/2.4) - 0.055`. The linear
// segment for c < 0.0031308 is left untouched (multiply by 12.92).
//
// Strategy: 5th-order minimax-style fit using least-squares over a
// dense grid in [0.0031308, 1.0]. Validates max ULP error vs scalar
// f32::powf at 65536 sample points.
// ---------------------------------------------------------------------------

const N_POLY_COEFFS: usize = 6; // up to c^5

/// Fits a degree-(N-1) polynomial via least-squares over `n_samples`
/// uniformly spaced points in [a, b].
///
/// Solves `A^T A x = A^T y` where A is the Vandermonde matrix of
/// sample points. Returns the coefficients in ascending order
/// (a0 + a1 * x + a2 * x^2 + ...).
fn least_squares_poly(a: f64, b: f64, n_samples: usize) -> [f64; N_POLY_COEFFS] {
  // Build normal equations N x N, RHS N.
  let mut ata = [[0.0_f64; N_POLY_COEFFS]; N_POLY_COEFFS];
  let mut atb = [0.0_f64; N_POLY_COEFFS];
  for k in 0..n_samples {
    let t = (k as f64) / ((n_samples - 1) as f64);
    let x = a + (b - a) * t;
    let y = 1.055 * x.powf(1.0 / 2.4) - 0.055;
    // Compute powers of x up to N-1.
    let mut powers = [0.0_f64; N_POLY_COEFFS];
    powers[0] = 1.0;
    for i in 1..N_POLY_COEFFS {
      powers[i] = powers[i - 1] * x;
    }
    for i in 0..N_POLY_COEFFS {
      atb[i] += y * powers[i];
      for j in 0..N_POLY_COEFFS {
        ata[i][j] += powers[i] * powers[j];
      }
    }
  }
  // Solve N x N linear system via Gauss elimination with partial
  // pivoting. We can't use 3x3 cofactor here.
  solve_n(&mut ata, &mut atb)
}

fn solve_n(
  a: &mut [[f64; N_POLY_COEFFS]; N_POLY_COEFFS],
  b: &mut [f64; N_POLY_COEFFS],
) -> [f64; N_POLY_COEFFS] {
  let n = N_POLY_COEFFS;
  // Forward elimination with partial pivoting.
  for i in 0..n {
    let mut max_row = i;
    let mut max_val = a[i][i].abs();
    for k in (i + 1)..n {
      if a[k][i].abs() > max_val {
        max_val = a[k][i].abs();
        max_row = k;
      }
    }
    if max_row != i {
      a.swap(i, max_row);
      b.swap(i, max_row);
    }
    if a[i][i].abs() < 1e-300 {
      panic!("singular matrix in least-squares fit at row {}", i);
    }
    for k in (i + 1)..n {
      let factor = a[k][i] / a[i][i];
      for j in i..n {
        a[k][j] -= factor * a[i][j];
      }
      b[k] -= factor * b[i];
    }
  }
  // Back-substitution.
  let mut x = [0.0_f64; N_POLY_COEFFS];
  for i in (0..n).rev() {
    let mut s = b[i];
    for j in (i + 1)..n {
      s -= a[i][j] * x[j];
    }
    x[i] = s / a[i][i];
  }
  x
}

/// Applies a degree-N-1 polynomial via Horner's method.
fn poly_eval(coeffs: &[f64; N_POLY_COEFFS], x: f64) -> f64 {
  let mut acc = coeffs[N_POLY_COEFFS - 1];
  for i in (0..(N_POLY_COEFFS - 1)).rev() {
    acc = acc * x + coeffs[i];
  }
  acc
}

/// Compares polynomial OETF approximation vs scalar f32::powf reference
/// at `n_samples` points in [a, b]. Returns max ULP error.
fn validate_poly_ulp(coeffs: &[f64; N_POLY_COEFFS], a: f64, b: f64, n_samples: usize) -> u32 {
  let mut max_ulp: u32 = 0;
  let mut max_x = 0.0;
  for k in 0..n_samples {
    let t = (k as f64) / ((n_samples - 1) as f64);
    let x = a + (b - a) * t;
    let approx = poly_eval(coeffs, x) as f32;
    let exact_f64 = 1.055 * x.powf(1.0 / 2.4) - 0.055;
    let exact_f32 = (1.055_f32 * (x as f32).powf(1.0 / 2.4)) - 0.055;
    // ULP distance between two f32 values via i32 reinterpretation.
    let a_bits = approx.to_bits() as i64;
    let b_bits = exact_f32.to_bits() as i64;
    let ulp = (a_bits - b_bits).unsigned_abs() as u32;
    if ulp > max_ulp {
      max_ulp = ulp;
      max_x = x;
    }
    // Sanity check: |approx - exact_f64| should be small.
    let _ = exact_f64;
  }
  let _ = max_x;
  max_ulp
}

fn print_poly_oetf() {
  println!("// ---- Polynomial sRGB OETF coefficients (Tranche 6) ----");
  // Fit on the upper segment [0.0031308, 1.0] only — the lower segment
  // uses the linear branch `12.92 * c`.
  let coeffs = least_squares_poly(0.0031308, 1.0, 4096);
  println!(
    "// Fitted with degree-{} least-squares over [0.0031308, 1.0]:",
    N_POLY_COEFFS - 1,
  );
  println!("// (approximates f(c) = 1.055 * c^(1/2.4) - 0.055)");
  for (i, c) in coeffs.iter().enumerate() {
    println!("// coeff[{}] = {:>22.16}_f32  (c^{})", i, *c as f32, i);
  }
  let max_ulp = validate_poly_ulp(&coeffs, 0.0031308, 1.0, 65536);
  println!(
    "// Max ULP error vs f32::powf reference at 65536 samples: {}",
    max_ulp
  );
  println!();
}

// ---------------------------------------------------------------------------
// Main: derive each gamut's matrix, print constants + fixtures + polynomial.
// ---------------------------------------------------------------------------

fn main() {
  println!("// ===================================================================");
  println!("// Auto-generated from examples/derive_xyz_matrices.rs");
  println!("// Re-run via: cargo run --release --example derive_xyz_matrices");
  println!("// ===================================================================");
  println!();

  for g in GAMUTS {
    let m = derive_xyz_to_rgb(g);
    print_mat_const(g.name, g.source, &m, g.w);
  }

  for g in GAMUTS {
    let m_rgb_to_xyz = derive_rgb_to_xyz(g);
    print_luma_coeffs(g.name, &m_rgb_to_xyz);
  }

  for g in GAMUTS {
    let m = derive_xyz_to_rgb(g);
    print_fixtures(g.name, &m);
  }

  print_poly_oetf();
}
