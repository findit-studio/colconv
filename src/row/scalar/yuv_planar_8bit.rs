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
