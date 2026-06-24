use super::*;

// ---- YUV 4:2:0 → RGB (fused: upsample + convert) ----------------------

/// Converts one row of 4:2:0 YUV — Y at full width, U/V at half-width —
/// directly to packed RGB. Chroma is nearest-neighbor upsampled **in
/// registers** inside the kernel; no intermediate memory traffic.
///
/// `full_range = true` interprets Y in `[0, 255]` and chroma in
/// `[0, 255]` (JPEG / `yuvjNNNp` convention). `full_range = false`
/// interprets Y in `[16, 235]` and chroma in `[16, 240]` (broadcast /
/// limited-range convention).
///
/// Output is packed `R, G, B` triples: `rgb_out[3*x] = R`,
/// `rgb_out[3*x + 1] = G`, `rgb_out[3*x + 2] = B`.
///
/// # Panics (debug builds)
///
/// - `width` must be even (4:2:0 pairs pixel columns).
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420_to_rgb_or_rgba_row::<false, false>(
    y, u_half, v_half, None, rgb_out, width, matrix, full_range,
  );
}

/// Same as [`yuv_420_to_rgb_row`] but writes packed `R, G, B, A`
/// quadruplets, with `A = 0xFF` (opaque) for every pixel. The first
/// three bytes per pixel are byte-identical to what
/// [`yuv_420_to_rgb_row`] would write — only the per-pixel stride
/// (4 vs 3) and the alpha byte differ. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420_to_rgb_or_rgba_row::<true, false>(
    y, u_half, v_half, None, rgba_out, width, matrix, full_range,
  );
}

/// YUVA 4:2:0 8‑bit → packed **8‑bit** **RGBA**. Same numerical
/// contract as [`yuv_420_to_rgba_row`] for R/G/B; the per-pixel alpha
/// byte is sourced from `a_src` (one byte per pixel, full-width)
/// instead of being constant `0xFF`. Used by the YUVA source family
/// ([`crate::source::Yuva420p`] in tranche 8b‑2a).
///
/// Thin wrapper over [`yuv_420_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `a_src.len() >= width`,
///   `rgba_out.len() >= 4 * width`.
// Reachable only via the yuva dispatcher in `dispatch::yuva` (gated by
// `feature = "yuva"`). The arch-side `yuv_420_to_rgb_or_rgba_row<…,
// ALPHA_SRC = true>` tail-calls into here for widths not divisible by
// the SIMD block. Const evaluation prunes that branch when the public
// wrapper is monomorphized with `ALPHA_SRC = false`, so under
// `yuv-planar` alone Rust sees the helper as dead. A symbol cfg gate
// can't help — `scalar::yuv_420_to_rgba_with_alpha_src_row` must
// resolve at name lookup, before const eval. `#[allow(dead_code)]`
// covers this single helper without re-enabling the workaround
// crate-wide.
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420_to_rgba_with_alpha_src_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_420_to_rgb_or_rgba_row::<true, true>(
    y,
    u_half,
    v_half,
    Some(a_src),
    rgba_out,
    width,
    matrix,
    full_range,
  );
}

/// Shared scalar kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bytes / pixel), [`yuv_420_to_rgba_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, 4 bytes / pixel — 4th is opaque
/// `0xFF`) and [`yuv_420_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bytes / pixel with source-derived alpha). The
/// math is identical; only the per-pixel store differs. The const
/// generics drive compile-time monomorphization — each public wrapper
/// is inlined with the branches eliminated.
///
/// `a_src` is `None` for both `ALPHA_SRC = false` flavors — reading
/// it is a const-disabled branch in those monomorphizations, so
/// callers pay zero overhead for the strategy parameter. The 8-bit
/// alpha path stores `a_src[x]` directly: there's no `bits_mask` step
/// like the high-bit-depth siblings since `u8` already fits the
/// output.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
/// - When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///   `a_src.unwrap().len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);

  // Process two pixels per iteration — they share one chroma sample.
  // Round-to-nearest on every Q15 shift by adding 1 << 14 before the
  // `>> 15`, so 219 * (255/219 in Q15) cleanly produces 255 at the top
  // of limited-range without a 254-truncation bias.
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = ((u_half[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_half[c_idx] as i32 - 128) * c_scale + RND) >> 15;

    // Single-round per channel keeps the math faithful to a 1x2 3x3
    // matrix multiply. All six coefficients are used; standard
    // matrices (BT.601 / 709 / 2020) have `r_u = b_v = 0` so those
    // terms vanish. YCgCo uses all six.
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    // Pixel x.
    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    let r0 = clamp_u8(y0 + r_chroma);
    let g0 = clamp_u8(y0 + g_chroma);
    let b0 = clamp_u8(y0 + b_chroma);
    out[x * bpp] = r0;
    out[x * bpp + 1] = g0;
    out[x * bpp + 2] = b0;
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies the wrapper
      // passed Some(_), validated above by debug_assert. 8-bit input
      // means u8 fits the u8 output directly — no `bits_mask` /
      // depth-conversion shift like the high-bit-depth siblings.
      out[x * bpp + 3] = a_src.as_ref().unwrap()[x];
    } else if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }

    // Pixel x+1 shares chroma.
    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    let r1 = clamp_u8(y1 + r_chroma);
    let g1 = clamp_u8(y1 + g_chroma);
    let b1 = clamp_u8(y1 + b_chroma);
    out[(x + 1) * bpp] = r1;
    out[(x + 1) * bpp + 1] = g1;
    out[(x + 1) * bpp + 2] = b1;
    if ALPHA_SRC {
      out[(x + 1) * bpp + 3] = a_src.as_ref().unwrap()[x + 1];
    } else if ALPHA {
      out[(x + 1) * bpp + 3] = 0xFF;
    }

    x += 2;
  }
}
// ---- YUV 4:1:0 → RGB (fused: 4x horizontal upsample + convert) -------

/// Converts one row of 4:1:0 YUV — Y at full width, U/V at
/// **quarter-width** — directly to packed RGB. Each chroma sample
/// is duplicated across four adjacent Y columns; vertical 4:1
/// subsampling is the walker's job (the same chroma row is fed to
/// four consecutive Y rows).
///
/// `full_range = true` interprets Y in `[0, 255]` and chroma in
/// `[0, 255]` (JPEG / `yuvjNNNp` convention). `full_range = false`
/// interprets Y in `[16, 235]` and chroma in `[16, 240]`.
///
/// Output is packed `R, G, B` triples.
///
/// # Panics (debug builds)
///
/// - `width` must be a multiple of 4 (4:1:0 pairs four columns).
/// - `y.len() >= width`, `u_quarter.len() >= width / 4`,
///   `v_quarter.len() >= width / 4`, `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_410_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_410_to_rgb_or_rgba_row::<false>(y, u_quarter, v_quarter, rgb_out, width, matrix, full_range);
}

/// Same as [`yuv_410_to_rgb_row`] but writes packed `R, G, B, A`
/// quadruplets, with `A = 0xFF` (opaque) for every pixel. The first
/// three bytes per pixel are byte-identical to what
/// [`yuv_410_to_rgb_row`] would write — only the per-pixel stride
/// (4 vs 3) and the alpha byte differ.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_410_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_410_to_rgb_or_rgba_row::<true>(y, u_quarter, v_quarter, rgba_out, width, matrix, full_range);
}

/// Shared scalar kernel for [`yuv_410_to_rgb_row`] (`ALPHA = false`,
/// 3 bpp) and [`yuv_410_to_rgba_row`] (`ALPHA = true`, 4 bpp + opaque
/// `0xFF` alpha). The math is identical to the 4:2:0 sibling; only
/// the chroma-fanout shape differs (one chroma sample per four Y
/// columns instead of two), and there's no source-alpha variant
/// because no YUVA 4:1:0 format ships in the crate's tier list.
///
/// # Panics (debug builds)
///
/// - `width` must be a multiple of 4.
/// - `y.len() >= width`, `u_quarter.len() >= width / 4`,
///   `v_quarter.len() >= width / 4`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_410_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  debug_assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // Process four pixels per iteration — they share one chroma sample.
  let mut x = 0;
  while x < width {
    let c_idx = x / 4;
    let u_d = ((u_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;

    // Same matrix-multiply as 4:2:0: standard matrices have
    // r_u = b_v = 0; YCgCo uses all six.
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    // Each of the 4 pixels in this group: scale Y, add the shared
    // chroma contribution, clamp, store. Unrolling by 4 here matches
    // the 4:2:0 unroll-by-2 pattern and lets the compiler keep the
    // chroma values in registers across the group.
    for k in 0..4 {
      let yk = ((y[x + k] as i32 - y_off) * y_scale + RND) >> 15;
      out[(x + k) * bpp] = clamp_u8(yk + r_chroma);
      out[(x + k) * bpp + 1] = clamp_u8(yk + g_chroma);
      out[(x + k) * bpp + 2] = clamp_u8(yk + b_chroma);
      if ALPHA {
        out[(x + k) * bpp + 3] = 0xFF;
      }
    }

    x += 4;
  }
}

/// YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// One UV pair per Y pixel, U/V from separate planes. Same
/// arithmetic as [`nv24_to_rgb_row`] (4:4:4 semi-planar) but
/// without the deinterleave step — U and V come pre-separated.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444_to_rgb_or_rgba_row::<false, false>(y, u, v, None, rgb_out, width, matrix, full_range);
}

/// YUV 4:4:4 planar → packed `R, G, B, A` quadruplets with constant
/// `A = 0xFF`. First three bytes per pixel are byte-identical to
/// [`yuv_444_to_rgb_row`]. `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444_to_rgb_or_rgba_row::<true, false>(y, u, v, None, rgba_out, width, matrix, full_range);
}

/// YUVA 4:4:4 planar → packed `R, G, B, A` quadruplets with the
/// per-pixel alpha byte sourced from `a_src` instead of constant
/// `0xFF`. R/G/B are byte-identical to [`yuv_444_to_rgb_row`]. Used
/// by the YUVA 4:4:4 source family ([`crate::source::Yuva444p`]).
///
/// Thin wrapper over [`yuv_444_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `a_src.len() >= width`, `rgba_out.len() >= 4 * width`.
// See `yuv_420_to_rgba_with_alpha_src_row` for the per-item
// `#[allow(dead_code)]` rationale.
#[allow(dead_code)]
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444_to_rgba_with_alpha_src_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_444_to_rgb_or_rgba_row::<true, true>(
    y,
    u,
    v,
    Some(a_src),
    rgba_out,
    width,
    matrix,
    full_range,
  );
}

/// Shared scalar kernel for [`yuv_444_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, 3 bpp), [`yuv_444_to_rgba_row`] (`ALPHA = true,
/// ALPHA_SRC = false`, 4 bpp + opaque alpha) and
/// [`yuv_444_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, 4 bpp + source-derived alpha). Math is
/// identical; only the per-pixel store stride and alpha byte differ.
/// `const` generic monomorphizes per call site, so the `if ALPHA` /
/// `if ALPHA_SRC` branches are eliminated.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
/// - When `ALPHA_SRC = true`: `a_src` must be `Some(_)` and
///   `a_src.unwrap().len() >= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output — there is no 3 bpp store with
  // alpha to put it in.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");
  if ALPHA_SRC {
    debug_assert!(
      a_src.as_ref().is_some_and(|s| s.len() >= width),
      "a_src row too short"
    );
  }

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  for x in 0..width {
    // 4:4:4: one UV pair per pixel, no subsampling.
    let u_d = ((u[x] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v[x] as i32 - 128) * c_scale + RND) >> 15;

    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    out[x * bpp] = clamp_u8(y0 + r_chroma);
    out[x * bpp + 1] = clamp_u8(y0 + g_chroma);
    out[x * bpp + 2] = clamp_u8(y0 + b_chroma);
    if ALPHA_SRC {
      // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
      // 8-bit alpha already fits u8 — no shift, no mask.
      out[x * bpp + 3] = a_src.as_ref().unwrap()[x];
    } else if ALPHA {
      out[x * bpp + 3] = 0xFF;
    }
  }
}

// ---- Unclamped real-valued YUV → RGB decode (RFC #238 #244) ----------
//
// The scene-referred [`AveragingDomain::Linear`] decode. These mirror the
// Q15 `yuv_*_to_rgb_row` kernels above EXACTLY — same `Coefficients`, same
// `range_params_n::<8, 8>` offsets/scales — but evaluate the affine matrix
// in real-valued `f32` and DO NOT clamp or round. Each Q15 fixed-point
// coefficient is converted to its real value (`q15 as f32 / 32768.0`), so
// the matrix math is the same as the production decode; the ONLY difference
// is the absent intermediate Q15 rounding and the absent final clamp+round
// to `[0, 255]`. The result is RGB normalized to a `[0, 1]` scale (the
// 8-bit code value divided by 255) that MAY fall below 0 or above 1 where
// the source YUV is out of gamut — exactly the excursions the clamped
// display-referred decode discards. The linear-light tail lifts this through
// the EOTF (whose odd-symmetric extrapolation handles the out-of-`[0, 1]`
// case) and clamps only at the re-encoded output.

/// Q15 → real scale: `1 / 32768`. Multiplying a Q15 fixed-point integer by
/// this yields the real coefficient the production decode approximates, so
/// the unclamped decode below is numerically the same matrix as the Q15
/// kernel, just real-valued and unclamped.
///
/// Gated like its only consumer: the scene-referred linear-light resample
/// tail, which needs the resample / sink path (`std` or `alloc`) and `rgb`
/// output. Without that path (e.g. the direct-convert-only `frame` config),
/// these decoders have no caller, so they must not compile.
#[cfg(all(feature = "rgb", any(feature = "std", feature = "alloc")))]
const Q15_TO_REAL: f32 = 1.0 / 32768.0;

/// Real-valued, unclamped `YUV 4:2:0 → normalized RGB` decode — the
/// scene-referred twin of [`yuv_420_to_rgb_row`]. Chroma is half-width
/// (nearest-neighbor 1→2 upsampled in registers, as the Q15 sibling does);
/// this same kernel also serves 4:2:2 (half-width chroma, full height), the
/// way the Q15 `yuv_420_to_rgb_row` does in the 4:2:2 row stage.
///
/// `out` receives `3 * width` interleaved `R, G, B` `f32` values on a
/// `[0, 1]` scale (the real code value / 255), unclamped — out-of-gamut
/// channels fall outside `[0, 1]`.
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, `out.len() >= 3 * width`.
#[cfg(all(feature = "rgb", any(feature = "std", feature = "alloc")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_420_to_rgb_f32_unclamped_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  out: &mut [f32],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  // The SAME building blocks as the Q15 kernel, lifted to real values.
  let y_off_f = y_off as f32;
  let y_scale_f = y_scale as f32 * Q15_TO_REAL;
  let c_scale_f = c_scale as f32 * Q15_TO_REAL;
  let r_u = coeffs.r_u() as f32 * Q15_TO_REAL;
  let r_v = coeffs.r_v() as f32 * Q15_TO_REAL;
  let g_u = coeffs.g_u() as f32 * Q15_TO_REAL;
  let g_v = coeffs.g_v() as f32 * Q15_TO_REAL;
  let b_u = coeffs.b_u() as f32 * Q15_TO_REAL;
  let b_v = coeffs.b_v() as f32 * Q15_TO_REAL;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    // `u_d` / `v_d` are the real chroma deltas (U - 128) * c_scale, NOT
    // re-quantized — the Q15 path rounds these to integers; the unclamped
    // path keeps them real.
    let u_d = (u_half[c_idx] as f32 - 128.0) * c_scale_f;
    let v_d = (v_half[c_idx] as f32 - 128.0) * c_scale_f;
    let r_chroma = r_u * u_d + r_v * v_d;
    let g_chroma = g_u * u_d + g_v * v_d;
    let b_chroma = b_u * u_d + b_v * v_d;

    // Both pixels share the chroma sample; only Y differs. No clamp, no
    // round — the normalized `/ 255` value may leave `[0, 1]`.
    let y0 = (y[x] as f32 - y_off_f) * y_scale_f;
    out[x * 3] = (y0 + r_chroma) / 255.0;
    out[x * 3 + 1] = (y0 + g_chroma) / 255.0;
    out[x * 3 + 2] = (y0 + b_chroma) / 255.0;

    let y1 = (y[x + 1] as f32 - y_off_f) * y_scale_f;
    out[(x + 1) * 3] = (y1 + r_chroma) / 255.0;
    out[(x + 1) * 3 + 1] = (y1 + g_chroma) / 255.0;
    out[(x + 1) * 3 + 2] = (y1 + b_chroma) / 255.0;

    x += 2;
  }
}

/// Real-valued, unclamped `YUV 4:4:4 → normalized RGB` decode — the
/// scene-referred twin of [`yuv_444_to_rgb_row`]. One UV pair per pixel (no
/// subsampling); also serves 4:4:0 (the way the Q15 `yuv_444_to_rgb_row`
/// does in the 4:4:0 row stage, the chroma row duplicated across two Y
/// rows by the walker).
///
/// `out` receives `3 * width` interleaved `R, G, B` `f32` values on a
/// `[0, 1]` scale (the real code value / 255), unclamped.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///   `out.len() >= 3 * width`.
#[cfg(all(feature = "rgb", any(feature = "std", feature = "alloc")))]
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_444_to_rgb_f32_unclamped_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  out: &mut [f32],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(out.len() >= width * 3, "out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  let y_off_f = y_off as f32;
  let y_scale_f = y_scale as f32 * Q15_TO_REAL;
  let c_scale_f = c_scale as f32 * Q15_TO_REAL;
  let r_u = coeffs.r_u() as f32 * Q15_TO_REAL;
  let r_v = coeffs.r_v() as f32 * Q15_TO_REAL;
  let g_u = coeffs.g_u() as f32 * Q15_TO_REAL;
  let g_v = coeffs.g_v() as f32 * Q15_TO_REAL;
  let b_u = coeffs.b_u() as f32 * Q15_TO_REAL;
  let b_v = coeffs.b_v() as f32 * Q15_TO_REAL;

  for x in 0..width {
    let u_d = (u[x] as f32 - 128.0) * c_scale_f;
    let v_d = (v[x] as f32 - 128.0) * c_scale_f;
    let r_chroma = r_u * u_d + r_v * v_d;
    let g_chroma = g_u * u_d + g_v * v_d;
    let b_chroma = b_u * u_d + b_v * v_d;

    let y0 = (y[x] as f32 - y_off_f) * y_scale_f;
    out[x * 3] = (y0 + r_chroma) / 255.0;
    out[x * 3 + 1] = (y0 + g_chroma) / 255.0;
    out[x * 3 + 2] = (y0 + b_chroma) / 255.0;
  }
}

// ---- YUV 4:1:1 → RGB / RGBA (fused: 1→4 chroma upsample) -------------

/// Converts one row of 4:1:1 YUV — Y at full width, U/V at
/// **quarter-width** — directly to packed RGB. Each chroma sample
/// covers four Y columns; nearest-neighbor 1→4 upsample happens in
/// registers inside the kernel.
///
/// Same range / matrix semantics as [`yuv_420_to_rgb_row`]; only the
/// chroma indexing differs (`x / 4` instead of `x / 2`).
///
/// FFmpeg-compatible widths: arbitrary `width` is accepted. Chroma
/// row size is `width.div_ceil(4)` samples; widths not divisible by
/// 4 leave a partial 1..3-pixel final chroma group, where the last
/// chroma sample covers the trailing 1..3 Y pixels.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u_quarter.len() >= width.div_ceil(4)`,
///   `v_quarter.len() >= width.div_ceil(4)`,
///   `rgb_out.len() >= 3 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_411_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_411_to_rgb_or_rgba_row::<false>(y, u_quarter, v_quarter, rgb_out, width, matrix, full_range);
}

/// Same as [`yuv_411_to_rgb_row`] but writes packed `R, G, B, A`
/// quadruplets, with `A = 0xFF` (opaque) for every pixel. The first
/// three bytes per pixel are byte-identical to what
/// [`yuv_411_to_rgb_row`] would write.
///
/// `rgba_out.len() >= 4 * width`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_411_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  yuv_411_to_rgb_or_rgba_row::<true>(y, u_quarter, v_quarter, rgba_out, width, matrix, full_range);
}

/// Shared scalar kernel for [`yuv_411_to_rgb_row`] (`ALPHA = false`,
/// 3 bpp) and [`yuv_411_to_rgba_row`] (`ALPHA = true`, 4 bpp + opaque
/// alpha). The math is identical; only the per-pixel store differs.
/// `ALPHA` drives compile-time monomorphization — each public wrapper
/// is inlined with the alpha branch eliminated.
///
/// 4:1:1 has no alpha-source variant: there is no `Yuva411p` source
/// format in FFmpeg, so `ALPHA_SRC` is unconditional `false`.
///
/// FFmpeg-compatible widths: chroma row width is
/// `width.div_ceil(4)` samples. The kernel processes full 4-pixel
/// chroma groups, then handles a trailing 1..3-pixel partial group
/// (when `width % 4 != 0`) by reusing the final chroma sample for
/// the remaining Y pixels.
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u_quarter.len() >= width.div_ceil(4)`,
///   `v_quarter.len() >= width.div_ceil(4)`,
///   `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) fn yuv_411_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(
    u_quarter.len() >= width.div_ceil(4),
    "u_quarter row too short"
  );
  debug_assert!(
    v_quarter.len() >= width.div_ceil(4),
    "v_quarter row too short"
  );
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp, "out row too short for {bpp}bpp");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // Aligned body: process full 4-pixel chroma groups. `body_end` is
  // the largest multiple of 4 not exceeding `width`; the trailing
  // 1..3 Y pixels (if any) are handled in the partial-group block
  // below using the final (partial) chroma sample.
  let body_end = width & !3;
  let mut x = 0;
  while x < body_end {
    let c_idx = x / 4;
    let u_d = ((u_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;

    // Single-round per channel keeps the math faithful to a 1x4 3x3
    // matrix multiply. All four pixels in this group share the chroma
    // contributions — only Y differs.
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    // Unrolled fan-out across the four Y pixels that share this chroma
    // sample. The const-generic `ALPHA` decides 3 vs 4 bpp store; the
    // monomorphizer eliminates the branch.
    let mut k = 0;
    while k < 4 {
      let y_k = ((y[x + k] as i32 - y_off) * y_scale + RND) >> 15;
      let pos = (x + k) * bpp;
      out[pos] = clamp_u8(y_k + r_chroma);
      out[pos + 1] = clamp_u8(y_k + g_chroma);
      out[pos + 2] = clamp_u8(y_k + b_chroma);
      if ALPHA {
        out[pos + 3] = 0xFF;
      }
      k += 1;
    }

    x += 4;
  }

  // Trailing 1..3-pixel partial chroma group (FFmpeg ceil-shift
  // chroma). When `width` isn't a multiple of 4, the final chroma
  // sample at index `width.div_ceil(4) - 1` covers the remaining
  // Y pixels at columns `body_end..width`. Width 5 → body_end=4,
  // 1 trailing Y at column 4, paired with chroma[1] (the partial
  // 1-pixel group). Width 641 → body_end=640, 1 trailing Y at
  // column 640, paired with chroma[160]. Same per-pixel math as
  // the body — only the iteration shape changes.
  if x < width {
    let c_idx = x / 4; // == body_end / 4 == width.div_ceil(4) - 1.
    let u_d = ((u_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;
    while x < width {
      let y_k = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
      let pos = x * bpp;
      out[pos] = clamp_u8(y_k + r_chroma);
      out[pos + 1] = clamp_u8(y_k + g_chroma);
      out[pos + 2] = clamp_u8(y_k + b_chroma);
      if ALPHA {
        out[pos + 3] = 0xFF;
      }
      x += 1;
    }
  }
}

// ---- Planar 8-bit YUV → HSV (direct: no RGB scratch) -----------------
//
// The display-referred twins of the `yuv_*_to_rgb_row` kernels above,
// fused with the OpenCV HSV quantizer. Each shares the EXACT per-pixel
// Q15 decode (`Coefficients::for_matrix` + `range_params_n::<8, 8>` +
// the same chroma-upsampling shape) as its `_to_rgb` sibling, then feeds
// the decoded `(r, g, b)` straight into [`rgb_to_hsv_pixel`] and scatters
// to the H/S/V planes — never materializing a packed-RGB row. They are
// therefore byte-identical to `rgb_to_hsv_row(yuv_*_to_rgb_row(...))` but
// allocate no RGB intermediate. Used by the planar 8-bit sink's
// HSV-without-RGB path; the SIMD backends mirror them via a small
// reused-chunk RGB scratch (the chunk filler IS the existing SIMD RGB
// kernel) plus the SIMD `rgb_to_hsv_row` on the chunk.

/// YUV 4:2:0 planar → planar HSV bytes (OpenCV `cv2.COLOR_RGB2HSV`
/// encoding: `H ∈ [0, 179]`, `S, V ∈ [0, 255]`). Chroma is half-width,
/// nearest-neighbor 1→2 upsampled per pixel pair exactly as
/// [`yuv_420_to_rgb_row`]. Also serves 4:2:2 (half-width chroma, full
/// height — the same per-row shape).
///
/// # Panics (debug builds)
///
/// - `width` must be even.
/// - `y.len() >= width`, `u_half.len() >= width / 2`,
///   `v_half.len() >= width / 2`, and each of `h_out` / `s_out` /
///   `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_420_to_hsv_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_half.len() >= width / 2, "u_half row too short");
  debug_assert!(v_half.len() >= width / 2, "v_half row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let c_idx = x / 2;
    let u_d = ((u_half[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_half[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    let (h0, s0, v0) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h0;
    s_out[x] = s0;
    v_out[x] = v0;

    let y1 = ((y[x + 1] as i32 - y_off) * y_scale + RND) >> 15;
    let (h1, s1, v1) = rgb_to_hsv_pixel(
      clamp_u8(y1 + r_chroma) as i32,
      clamp_u8(y1 + g_chroma) as i32,
      clamp_u8(y1 + b_chroma) as i32,
    );
    h_out[x + 1] = h1;
    s_out[x + 1] = s1;
    v_out[x + 1] = v1;

    x += 2;
  }
}

/// YUV 4:4:4 planar → planar HSV bytes. One UV pair per pixel (no
/// subsampling), exactly as [`yuv_444_to_rgb_row`]. Also serves 4:4:0
/// (the chroma row duplicated across two Y rows by the walker — the
/// per-row shape is identical to 4:4:4).
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u.len() >= width`, `v.len() >= width`, and
///   each of `h_out` / `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_444_to_hsv_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u.len() >= width, "u row too short");
  debug_assert!(v.len() >= width, "v row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  for x in 0..width {
    let u_d = ((u[x] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v[x] as i32 - 128) * c_scale + RND) >> 15;
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    let y0 = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    let (h, s, vv) = rgb_to_hsv_pixel(
      clamp_u8(y0 + r_chroma) as i32,
      clamp_u8(y0 + g_chroma) as i32,
      clamp_u8(y0 + b_chroma) as i32,
    );
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = vv;
  }
}

/// YUV 4:1:0 planar → planar HSV bytes. Quarter-width chroma, each
/// sample duplicated across four adjacent Y columns exactly as
/// [`yuv_410_to_rgb_row`].
///
/// # Panics (debug builds)
///
/// - `width` must be a multiple of 4.
/// - `y.len() >= width`, `u_quarter.len() >= width / 4`,
///   `v_quarter.len() >= width / 4`, and each of `h_out` / `s_out` /
///   `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_410_to_hsv_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(u_quarter.len() >= width / 4, "u_quarter row too short");
  debug_assert!(v_quarter.len() >= width / 4, "v_quarter row too short");
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  let mut x = 0;
  while x < width {
    let c_idx = x / 4;
    let u_d = ((u_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;

    for k in 0..4 {
      let yk = ((y[x + k] as i32 - y_off) * y_scale + RND) >> 15;
      let (h, s, vv) = rgb_to_hsv_pixel(
        clamp_u8(yk + r_chroma) as i32,
        clamp_u8(yk + g_chroma) as i32,
        clamp_u8(yk + b_chroma) as i32,
      );
      h_out[x + k] = h;
      s_out[x + k] = s;
      v_out[x + k] = vv;
    }

    x += 4;
  }
}

/// YUV 4:1:1 planar → planar HSV bytes. Quarter-width chroma, each
/// sample covering four Y columns, with FFmpeg-compatible arbitrary
/// widths (a trailing 1..3-pixel partial chroma group reuses the final
/// chroma sample) — the exact shape of [`yuv_411_to_rgb_row`].
///
/// # Panics (debug builds)
///
/// - `y.len() >= width`, `u_quarter.len() >= width.div_ceil(4)`,
///   `v_quarter.len() >= width.div_ceil(4)`, and each of `h_out` /
///   `s_out` / `v_out` `>= width`.
#[cfg_attr(not(tarpaulin), inline(always))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn yuv_411_to_hsv_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  h_out: &mut [u8],
  s_out: &mut [u8],
  v_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width, "y row too short");
  debug_assert!(
    u_quarter.len() >= width.div_ceil(4),
    "u_quarter row too short"
  );
  debug_assert!(
    v_quarter.len() >= width.div_ceil(4),
    "v_quarter row too short"
  );
  debug_assert!(h_out.len() >= width, "h_out row too short");
  debug_assert!(s_out.len() >= width, "s_out row too short");
  debug_assert!(v_out.len() >= width, "v_out row too short");

  let coeffs = Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // One closure for the per-pixel decode + HSV scatter, shared by the
  // aligned body and the trailing partial group below.
  let mut emit = |x: usize, r_chroma: i32, g_chroma: i32, b_chroma: i32| {
    let yk = ((y[x] as i32 - y_off) * y_scale + RND) >> 15;
    let (h, s, vv) = rgb_to_hsv_pixel(
      clamp_u8(yk + r_chroma) as i32,
      clamp_u8(yk + g_chroma) as i32,
      clamp_u8(yk + b_chroma) as i32,
    );
    h_out[x] = h;
    s_out[x] = s;
    v_out[x] = vv;
  };

  let body_end = width & !3;
  let mut x = 0;
  while x < body_end {
    let c_idx = x / 4;
    let u_d = ((u_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;
    let mut k = 0;
    while k < 4 {
      emit(x + k, r_chroma, g_chroma, b_chroma);
      k += 1;
    }
    x += 4;
  }

  // Trailing 1..3-pixel partial chroma group (FFmpeg ceil-shift chroma),
  // reusing the final chroma sample — same shape as the RGB sibling.
  if x < width {
    let c_idx = x / 4;
    let u_d = ((u_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let v_d = ((v_quarter[c_idx] as i32 - 128) * c_scale + RND) >> 15;
    let r_chroma = (coeffs.r_u() * u_d + coeffs.r_v() * v_d + RND) >> 15;
    let g_chroma = (coeffs.g_u() * u_d + coeffs.g_v() * v_d + RND) >> 15;
    let b_chroma = (coeffs.b_u() * u_d + coeffs.b_v() * v_d + RND) >> 15;
    while x < width {
      emit(x, r_chroma, g_chroma, b_chroma);
      x += 1;
    }
  }
}
