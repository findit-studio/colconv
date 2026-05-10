use core::arch::wasm32::*;

use super::*;

/// WASM simd128 YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **simd128 must be enabled at compile time.** Verified by the
///    dispatcher via `cfg!(target_feature = "simd128")`. WASM has no
///    runtime CPU detection, so the obligation is purely compile‑time:
///    the WASM module was produced with `-C target-feature=+simd128`
///    (or equivalent), and it is being executed in a WASM runtime that
///    supports the SIMD proposal.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`v128_load`, `u16x8_load_extend_u8x8`,
/// `v128_store`).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// WASM simd128 YUV 4:2:0 → packed **RGBA** (8-bit). Same contract
/// as [`yuv_420_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// 1. simd128 must be enabled at compile time.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 availability + slice bounds —
  // see [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// WASM simd128 YUVA 4:2:0 → packed **8-bit RGBA** with the
/// per-pixel alpha byte **sourced from `a_src`** (8-bit YUVA's alpha
/// is already `u8` — no depth conversion needed). Same numerical
/// contract as [`yuv_420_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_420_to_rgba_with_alpha_src_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
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
}

/// Shared WASM simd128 kernel for [`yuv_420_to_rgb_row`]
/// (`ALPHA = false, ALPHA_SRC = false`, [`write_rgb_16`]),
/// [`yuv_420_to_rgba_row`] (`ALPHA = true, ALPHA_SRC = false`,
/// [`write_rgba_16`] with constant `0xFF` alpha) and
/// [`yuv_420_to_rgba_with_alpha_src_row`] (`ALPHA = true,
/// ALPHA_SRC = true`, [`write_rgba_16`] with the alpha lane loaded
/// directly from `a_src`).
///
/// # Safety
///
/// Same as [`yuv_420_to_rgb_row`] / [`yuv_420_to_rgba_row`]; the
/// `out` slice must be `>= width * (if ALPHA { 4 } else { 3 })`
/// bytes long. When `ALPHA_SRC = true`: `a_src` must be `Some(_)`
/// and `a_src.unwrap().len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
unsafe fn yuv_420_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert_eq!(width & 1, 0, "YUV 4:2:0 requires even width");
  debug_assert!(y.len() >= width);
  debug_assert!(u_half.len() >= width / 2);
  debug_assert!(v_half.len() >= width / 2);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: simd128 availability is the caller's compile‑time
  // obligation per the `# Safety` section. All pointer adds below are
  // bounded by the `while x + 16 <= width` loop condition and the
  // caller‑promised slice lengths.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let mid128 = i16x8_splat(128);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = u8x16_splat(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 Y (16 bytes) and 8 U / 8 V (extending each to i16x8).
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      let u_i16_zero = u16x8_load_extend_u8x8(u_half.as_ptr().add(x / 2));
      let v_i16_zero = u16x8_load_extend_u8x8(v_half.as_ptr().add(x / 2));

      // Subtract 128 from chroma (u16 treated as i16).
      let u_i16 = i16x8_sub(u_i16_zero, mid128);
      let v_i16 = i16x8_sub(v_i16_zero, mid128);

      // Split each i16x8 into two i32x4 halves (sign‑extending).
      let u_lo_i32 = i32x4_extend_low_i16x8(u_i16);
      let u_hi_i32 = i32x4_extend_high_i16x8(u_i16);
      let v_lo_i32 = i32x4_extend_low_i16x8(v_i16);
      let v_hi_i32 = i32x4_extend_high_i16x8(v_i16);

      // u_d, v_d = (u * c_scale + RND) >> 15 — bit‑exact to scalar.
      let u_d_lo = q15_shift(i32x4_add(i32x4_mul(u_lo_i32, c_scale_v), rnd_v));
      let u_d_hi = q15_shift(i32x4_add(i32x4_mul(u_hi_i32, c_scale_v), rnd_v));
      let v_d_lo = q15_shift(i32x4_add(i32x4_mul(v_lo_i32, c_scale_v), rnd_v));
      let v_d_hi = q15_shift(i32x4_add(i32x4_mul(v_hi_i32, c_scale_v), rnd_v));

      // Per‑channel chroma → i16x8 (8 chroma values per channel).
      let r_chroma = chroma_i16x8(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x8(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x8(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest‑neighbor upsample: duplicate each of 8 chroma lanes
      // into its pair slot → two i16x8 vectors covering 16 Y lanes.
      // Each i16 value is 2 bytes, so byte‑level shuffle indices
      // `[0,1,0,1, 2,3,2,3, 4,5,4,5, 6,7,6,7]` duplicate the low
      // 4 × i16 lanes; `[8..15 paired]` duplicates the high 4.
      let r_dup_lo = dup_lo(r_chroma);
      let r_dup_hi = dup_hi(r_chroma);
      let g_dup_lo = dup_lo(g_chroma);
      let g_dup_hi = dup_hi(g_chroma);
      let b_dup_lo = dup_lo(b_chroma);
      let b_dup_hi = dup_hi(b_chroma);

      // Y path: widen low / high 8 Y to i16x8, scale.
      let y_low_i16 = u8_low_to_i16x8(y_vec);
      let y_high_i16 = u8_high_to_i16x8(y_vec);
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = i16x8_add_sat(y_scaled_lo, b_dup_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_dup_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_dup_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_dup_hi);
      let r_lo = i16x8_add_sat(y_scaled_lo, r_dup_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_dup_hi);

      // Saturate‑narrow to u8x16 per channel.
      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 8-bit YUVA alpha is already u8; load 16 bytes directly via
          // `v128_load`.
          v128_load(a_src.as_ref().unwrap_unchecked().as_ptr().add(x).cast())
        } else {
          alpha_u8
        };
        // 4‑way interleave → packed RGBA (64 bytes).
        write_rgba_16(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        // 3‑way interleave → packed RGB (48 bytes).
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    // Scalar tail for the 0..14 leftover pixels.
    if x < width {
      let tail_a = if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        Some(&a_src.as_ref().unwrap_unchecked()[x..width])
      } else {
        None
      };
      scalar::yuv_420_to_rgb_or_rgba_row::<ALPHA, ALPHA_SRC>(
        &y[x..width],
        &u_half[x / 2..width / 2],
        &v_half[x / 2..width / 2],
        tail_a,
        &mut out[x * bpp..width * bpp],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}
// ---- YUV 4:1:0 wasm simd128 entries ---------------------------------
//
// 4:1:0: planar YUV with chroma subsampled 4:1 in **both** axes. Each
// (U, V) sample covers a 4×4 luma block; vertical 4× re-use is the
// walker's job. This kernel handles the per-row 4× horizontal
// upsample. Math is byte-identical to scalar.
//
// Block size: 16 Y / 4 chroma per iteration (matches the 4:2:0
// simd128 kernel's 16-Y throughput). The chroma fan-out uses two
// `i8x16_shuffle` invocations with compile-time byte indices that
// duplicate each i16 chroma lane 4×.

/// wasm simd128 YUV 4:1:0 → packed RGB. Semantics match
/// [`scalar::yuv_410_to_rgb_row`] byte-identically.
///
/// # Safety
///
/// 1. **simd128 must be available** (compile-time `target_feature`).
/// 2. `width % 4 == 0` (4:1:0 requires width multiple of 4).
/// 3. `y.len() >= width`, `u_quarter.len() >= width / 4`,
///    `v_quarter.len() >= width / 4`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_410_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 availability + slice bounds — see
  // [`yuv_410_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_410_to_rgb_or_rgba_row::<false>(
      y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 YUV 4:1:0 → packed **RGBA** (8-bit). Same contract
/// as [`yuv_410_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_410_to_rgb_row`] except `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_410_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 availability + slice bounds — see
  // [`yuv_410_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_410_to_rgb_or_rgba_row::<true>(
      y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared wasm simd128 kernel for [`yuv_410_to_rgb_row`]
/// (`ALPHA = false`, [`write_rgb_16`]) and [`yuv_410_to_rgba_row`]
/// (`ALPHA = true`, [`write_rgba_16`] with constant `0xFF` alpha).
/// Math is byte-identical to `scalar::yuv_410_to_rgb_or_rgba_row::<ALPHA>`.
///
/// Pipeline per 16 Y pixels / 4 chroma samples:
/// 1. Load 16 Y (`v128_load`) + 4 U + 4 V (each as a u32 read
///    splatted into a v128).
/// 2. Widen 4 chroma → i16x8 (low 4 lanes meaningful), subtract 128,
///    widen low 4 to i32x4 for Q15 multiplies.
/// 3. `u_d = (u * c_scale + RND) >> 15`, same for `v_d` (i32x4).
/// 4. Per channel: `(C_u*u_d + C_v*v_d + RND) >> 15` (i32x4),
///    saturate-narrow to i16x8 (low 4 lanes carry chroma).
/// 5. 4× fan-out via two `i8x16_shuffle` calls with byte indices
///    duplicating each i16 chroma lane 4×:
///    - low (covers Y[0..8]):
///      `[c0,c0,c0,c0, c1,c1,c1,c1]` → byte indices
///      `[0,1,0,1,0,1,0,1, 2,3,2,3,2,3,2,3]`.
///    - high (covers Y[8..16]):
///      `[c2,c2,c2,c2, c3,c3,c3,c3]` → byte indices
///      `[4,5,4,5,4,5,4,5, 6,7,6,7,6,7,6,7]`.
/// 6. Y path → i16x8 pair via `scale_y`.
/// 7. Saturating add Y + chroma, saturate-narrow to u8x16,
///    interleave via [`write_rgb_16`] / [`write_rgba_16`].
///
/// # Safety
///
/// Same as [`yuv_410_to_rgb_row`] / [`yuv_410_to_rgba_row`].
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn yuv_410_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 3, 0, "YUV 4:1:0 requires width % 4 == 0");
  debug_assert!(y.len() >= width);
  debug_assert!(u_quarter.len() >= width / 4);
  debug_assert!(v_quarter.len() >= width / 4);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: simd128 availability is the caller's compile-time
  // obligation per the `# Safety` section. All pointer adds below are
  // bounded by the `while x + 16 <= width` loop condition and the
  // caller-promised slice lengths.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let mid128 = i16x8_splat(128);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let alpha_u8 = u8x16_splat(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = v128_load(y.as_ptr().add(x).cast());

      // Load 4 chroma bytes per plane via an unaligned u32 read,
      // splatted into a v128 (only the low 4 bytes matter).
      let u_bytes = (u_quarter.as_ptr().add(x / 4) as *const u32).read_unaligned();
      let v_bytes = (v_quarter.as_ptr().add(x / 4) as *const u32).read_unaligned();
      let u_v128 = i32x4_splat(u_bytes as i32);
      let v_v128 = i32x4_splat(v_bytes as i32);

      // Widen low 4 bytes → i16x8. The low 4 i16 lanes hold the 4
      // chroma samples; high 4 i16 lanes are duplicates from the
      // splat (we discard them via i32x4_extend_low). Subtract 128.
      // Use a shuffle to extract just the low 4 bytes interleaved
      // with zeros, similar to `u8_low_to_i16x8` but on a u32-splat.
      let u_widened = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 0, 16, 1, 17, 2, 18, 3, 19>(
        u_v128,
        i16x8_splat(0),
      );
      let v_widened = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 0, 16, 1, 17, 2, 18, 3, 19>(
        v_v128,
        i16x8_splat(0),
      );
      let u_i16 = i16x8_sub(u_widened, mid128);
      let v_i16 = i16x8_sub(v_widened, mid128);

      // Widen low 4 lanes to i32x4 for Q15 multiplies.
      let u_i32 = i32x4_extend_low_i16x8(u_i16);
      let v_i32 = i32x4_extend_low_i16x8(v_i16);

      // u_d, v_d = (u * c_scale + RND) >> 15.
      let u_d = q15_shift(i32x4_add(i32x4_mul(u_i32, c_scale_v), rnd_v));
      let v_d = q15_shift(i32x4_add(i32x4_mul(v_i32, c_scale_v), rnd_v));

      // Per-channel chroma contribution as i32x4.
      let r_i32 = q15_shift(i32x4_add(
        i32x4_add(i32x4_mul(cru, u_d), i32x4_mul(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = q15_shift(i32x4_add(
        i32x4_add(i32x4_mul(cgu, u_d), i32x4_mul(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = q15_shift(i32x4_add(
        i32x4_add(i32x4_mul(cbu, u_d), i32x4_mul(cbv, v_d)),
        rnd_v,
      ));

      // Saturate-narrow i32x4 → i16x8. Pass the same vector twice;
      // we only care about the low 4 i16 lanes ([c0,c1,c2,c3]).
      let r_chroma = i16x8_narrow_i32x4(r_i32, r_i32);
      let g_chroma = i16x8_narrow_i32x4(g_i32, g_i32);
      let b_chroma = i16x8_narrow_i32x4(b_i32, b_i32);

      // 4× fan-out: each chroma lane to 4 adjacent slots.
      // Low half (Y[0..8]): [c0,c0,c0,c0, c1,c1,c1,c1].
      // High half (Y[8..16]): [c2,c2,c2,c2, c3,c3,c3,c3].
      let r_dup_lo =
        i8x16_shuffle::<0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3>(r_chroma, r_chroma);
      let r_dup_hi =
        i8x16_shuffle::<4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7>(r_chroma, r_chroma);
      let g_dup_lo =
        i8x16_shuffle::<0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3>(g_chroma, g_chroma);
      let g_dup_hi =
        i8x16_shuffle::<4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7>(g_chroma, g_chroma);
      let b_dup_lo =
        i8x16_shuffle::<0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3>(b_chroma, b_chroma);
      let b_dup_hi =
        i8x16_shuffle::<4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7>(b_chroma, b_chroma);

      // Y path: widen low/high 8 Y to i16x8, scale.
      let y_low_i16 = u8_low_to_i16x8(y_vec);
      let y_high_i16 = u8_high_to_i16x8(y_vec);
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating add per channel.
      let b_lo = i16x8_add_sat(y_scaled_lo, b_dup_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_dup_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_dup_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_dup_hi);
      let r_lo = i16x8_add_sat(y_scaled_lo, r_dup_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_dup_hi);

      // Saturate-narrow per channel → u8x16.
      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    // Scalar tail. `width % 4 == 0` so `width - x` is a multiple of 4.
    if x < width {
      scalar::yuv_410_to_rgb_or_rgba_row::<ALPHA>(
        &y[x..width],
        &u_quarter[x / 4..width / 4],
        &v_quarter[x / 4..width / 4],
        &mut out[x * bpp..width * bpp],
        width - x,
        matrix,
        full_range,
      );
    }
  }
}

/// wasm simd128 YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **simd128 must be available** (compile-time `target_feature`).
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<false, false>(y, u, v, None, rgb_out, width, matrix, full_range);
  }
}

/// wasm simd128 YUV 4:4:4 planar → packed **RGBA** (8-bit). Same
/// contract as [`yuv_444_to_rgb_row`] but writes 4 bytes per pixel
/// via [`write_rgba_16`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_444_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true, false>(y, u, v, None, rgba_out, width, matrix, full_range);
  }
}

/// wasm simd128 YUVA 4:4:4 → packed **RGBA** with source alpha. R/G/B
/// are byte-identical to [`yuv_444_to_rgb_row`]; the per-pixel alpha
/// byte is sourced from `a_src` (8-bit, no shift needed) instead of
/// being constant `0xFF`. Used by [`crate::yuv::Yuva444p`].
///
/// Thin wrapper over [`yuv_444_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn yuv_444_to_rgba_with_alpha_src_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller obligations forwarded to the shared impl.
  unsafe {
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
}

/// Shared wasm simd128 YUV 4:4:4 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: [`write_rgb_16`].
/// - `ALPHA = true, ALPHA_SRC = false`: [`write_rgba_16`] with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: [`write_rgba_16`] with the
///   alpha lane loaded from `a_src` (8-bit input — no shift needed).
///
/// # Safety
///
/// 1. **simd128 must be enabled at compile time.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
/// 4. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_arguments)]
unsafe fn yuv_444_to_rgb_or_rgba_row<const ALPHA: bool, const ALPHA_SRC: bool>(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  a_src: Option<&[u8]>,
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // Source alpha requires RGBA output.
  const { assert!(!ALPHA_SRC || ALPHA) };
  debug_assert!(y.len() >= width);
  debug_assert!(u.len() >= width);
  debug_assert!(v.len() >= width);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);
  if ALPHA_SRC {
    debug_assert!(a_src.as_ref().is_some_and(|s| s.len() >= width));
  }

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let mid128 = i16x8_splat(128);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let alpha_u8 = u8x16_splat(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      let y_vec = v128_load(y.as_ptr().add(x).cast());
      // 4:4:4: 16 U + 16 V directly (no deinterleave).
      let u_vec = v128_load(u.as_ptr().add(x).cast());
      let v_vec = v128_load(v.as_ptr().add(x).cast());

      // Widen low / high halves of U / V to i16x8 and subtract 128.
      let u_lo_i16 = i16x8_sub(u16x8_extend_low_u8x16(u_vec), mid128);
      let u_hi_i16 = i16x8_sub(u16x8_extend_high_u8x16(u_vec), mid128);
      let v_lo_i16 = i16x8_sub(u16x8_extend_low_u8x16(v_vec), mid128);
      let v_hi_i16 = i16x8_sub(u16x8_extend_high_u8x16(v_vec), mid128);

      let u_lo_a = i32x4_extend_low_i16x8(u_lo_i16);
      let u_lo_b = i32x4_extend_high_i16x8(u_lo_i16);
      let u_hi_a = i32x4_extend_low_i16x8(u_hi_i16);
      let u_hi_b = i32x4_extend_high_i16x8(u_hi_i16);
      let v_lo_a = i32x4_extend_low_i16x8(v_lo_i16);
      let v_lo_b = i32x4_extend_high_i16x8(v_lo_i16);
      let v_hi_a = i32x4_extend_low_i16x8(v_hi_i16);
      let v_hi_b = i32x4_extend_high_i16x8(v_hi_i16);

      let u_d_lo_a = q15_shift(i32x4_add(i32x4_mul(u_lo_a, c_scale_v), rnd_v));
      let u_d_lo_b = q15_shift(i32x4_add(i32x4_mul(u_lo_b, c_scale_v), rnd_v));
      let u_d_hi_a = q15_shift(i32x4_add(i32x4_mul(u_hi_a, c_scale_v), rnd_v));
      let u_d_hi_b = q15_shift(i32x4_add(i32x4_mul(u_hi_b, c_scale_v), rnd_v));
      let v_d_lo_a = q15_shift(i32x4_add(i32x4_mul(v_lo_a, c_scale_v), rnd_v));
      let v_d_lo_b = q15_shift(i32x4_add(i32x4_mul(v_lo_b, c_scale_v), rnd_v));
      let v_d_hi_a = q15_shift(i32x4_add(i32x4_mul(v_hi_a, c_scale_v), rnd_v));
      let v_d_hi_b = q15_shift(i32x4_add(i32x4_mul(v_hi_b, c_scale_v), rnd_v));

      let r_chroma_lo = chroma_i16x8(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x8(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x8(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x8(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x8(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x8(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_low_i16 = u8_low_to_i16x8(y_vec);
      let y_high_i16 = u8_high_to_i16x8(y_vec);
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = i16x8_add_sat(y_scaled_lo, b_chroma_lo);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_chroma_hi);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_chroma_lo);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_chroma_hi);
      let r_lo = i16x8_add_sat(y_scaled_lo, r_chroma_lo);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_chroma_hi);

      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);

      if ALPHA {
        let a_v = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 8-bit alpha — load 16 bytes verbatim.
          v128_load(a_src.as_ref().unwrap_unchecked().as_ptr().add(x).cast())
        } else {
          alpha_u8
        };
        write_rgba_16(r_u8, g_u8, b_u8, a_v, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    if x < width {
      let tail_y = &y[x..width];
      let tail_u = &u[x..width];
      let tail_v = &v[x..width];
      let tail_w = width - x;
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA_SRC {
        // SAFETY (const-checked): ALPHA_SRC = true implies Some(_).
        let tail_a = &a_src.as_ref().unwrap_unchecked()[x..width];
        scalar::yuv_444_to_rgba_with_alpha_src_row(
          tail_y, tail_u, tail_v, tail_a, tail_out, tail_w, matrix, full_range,
        );
      } else if ALPHA {
        scalar::yuv_444_to_rgba_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      } else {
        scalar::yuv_444_to_rgb_row(tail_y, tail_u, tail_v, tail_out, tail_w, matrix, full_range);
      }
    }
  }
}

// ---- YUV 4:1:1 → RGB / RGBA (wasm simd128) -----------------------------

/// wasm simd128 YUV 4:1:1 planar → packed RGB. One chroma sample drives
/// four Y pixels (1→4 nearest-neighbor upsample). Processes 16 Y / 4
/// chroma samples per iteration — matches the wasm 4:2:0 block size
/// with 1/2 the chroma load count.
///
/// Same Q15 arithmetic as the scalar reference; output is byte-identical.
///
/// FFmpeg-compatible widths: arbitrary `width` accepted. Chroma row
/// is `width.div_ceil(4)` samples; the SIMD body strides 16 Y pixels
/// (multiple of 4), and the trailing 1..15 Y pixels — including any
/// partial 1..3-pixel chroma group — fall through to the scalar
/// reference.
///
/// # Safety
///
/// 1. **simd128 must be available** (compile-time `target_feature`).
/// 2. `y.len() >= width`,
///    `u_quarter.len() >= width.div_ceil(4)`,
///    `v_quarter.len() >= width.div_ceil(4)`.
/// 3. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_411_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 + slice bounds — see
  // [`yuv_411_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_411_to_rgb_or_rgba_row::<false>(
      y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
    );
  }
}

/// wasm simd128 YUV 4:1:1 planar → packed **RGBA** (8-bit). Same
/// contract as [`yuv_411_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_16`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_411_to_rgb_row`] except `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn yuv_411_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked simd128 + slice bounds — see
  // [`yuv_411_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_411_to_rgb_or_rgba_row::<true>(
      y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared wasm simd128 YUV 4:1:1 kernel. Processes 16 Y pixels (= 4
/// chroma samples) per iteration; the 1→4 chroma upsample is
/// materialized via two `i8x16_shuffle` masks duplicating each i16
/// chroma lane to 4 i16 output lanes:
///
/// 1. Load 4 chroma bytes via `v128_load32_zero` (32-bit gather, upper
///    96 bits zeroed).
/// 2. Widen low 8 bytes to i16x8; only lanes 0..3 hold real chroma.
/// 3. Compute chroma → R/G/B contribution as i16x8 (only the low 4
///    lanes matter).
/// 4. Stage 1: byte-shuffle pattern
///    `[0,1,0,1, 0,1,0,1, 2,3,2,3, 2,3,2,3]` produces an i16x8 with
///    `[c0,c0,c0,c0, c1,c1,c1,c1]` covering Y[0..8].
/// 5. Stage 2 (high half): byte-shuffle pattern
///    `[4,5,4,5, 4,5,4,5, 6,7,6,7, 6,7,6,7]` produces
///    `[c2,c2,c2,c2, c3,c3,c3,c3]` covering Y[8..16].
///
/// 4:1:1 has no source-alpha variant (no `Yuva411p` exists), so the
/// const-generic surface stays 1-D (`ALPHA` only).
///
/// # Safety
///
/// 1. **simd128 must be available** (compile-time `target_feature`).
/// 2. `y.len() >= width`,
///    `u_quarter.len() >= width.div_ceil(4)`,
///    `v_quarter.len() >= width.div_ceil(4)`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "simd128")]
unsafe fn yuv_411_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert!(y.len() >= width);
  debug_assert!(u_quarter.len() >= width.div_ceil(4));
  debug_assert!(v_quarter.len() >= width.div_ceil(4));
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: simd128 availability is the caller's compile-time
  // obligation per the `# Safety` section. All pointer adds below are
  // bounded by the `while x + 16 <= width` loop condition and the
  // caller-promised slice lengths.
  unsafe {
    let rnd_v = i32x4_splat(RND);
    let y_off_v = i16x8_splat(y_off as i16);
    let y_scale_v = i32x4_splat(y_scale);
    let c_scale_v = i32x4_splat(c_scale);
    let mid128 = i16x8_splat(128);
    let cru = i32x4_splat(coeffs.r_u());
    let crv = i32x4_splat(coeffs.r_v());
    let cgu = i32x4_splat(coeffs.g_u());
    let cgv = i32x4_splat(coeffs.g_v());
    let cbu = i32x4_splat(coeffs.b_u());
    let cbv = i32x4_splat(coeffs.b_v());
    let alpha_u8 = u8x16_splat(0xFF);

    let mut x = 0usize;
    while x + 16 <= width {
      // Load 16 Y bytes.
      let y_vec = v128_load(y.as_ptr().add(x).cast());

      // Load 4 chroma bytes per 16 Y pixels via 32-bit gather; the
      // upper 96 bits are zeroed by `v128_load32_zero`. Only lanes
      // 0..3 of the resulting u8x16 hold real chroma data.
      let u_v128 = v128_load32_zero(u_quarter.as_ptr().add(x / 4).cast());
      let v_v128 = v128_load32_zero(v_quarter.as_ptr().add(x / 4).cast());

      // Widen the low 8 u8 to i16x8 (zero-extended). Lanes 0..3 hold
      // the four chroma samples; lanes 4..7 are zero (which would be
      // -128 after the subtract — but those lanes are never consumed).
      let u_i16_zero = u8_low_to_i16x8(u_v128);
      let v_i16_zero = u8_low_to_i16x8(v_v128);
      let u_i16 = i16x8_sub(u_i16_zero, mid128);
      let v_i16 = i16x8_sub(v_i16_zero, mid128);

      // Sign-extend the low 4 i16 lanes to i32x4 (the only ones that
      // hold real chroma). Lanes 4..7 (which would be -128) are
      // discarded by `i32x4_extend_low_i16x8`.
      let u_i32 = i32x4_extend_low_i16x8(u_i16);
      let v_i32 = i32x4_extend_low_i16x8(v_i16);

      // u_d, v_d as i32x4 (4 chroma values).
      let u_d = q15_shift(i32x4_add(i32x4_mul(u_i32, c_scale_v), rnd_v));
      let v_d = q15_shift(i32x4_add(i32x4_mul(v_i32, c_scale_v), rnd_v));

      // Per-channel chroma → i32x4, narrow to i16 in low 4 lanes via
      // `i16x8_narrow_i32x4(x, x)` (high 4 lanes get a duplicate that
      // we don't consume).
      let r_i32 = q15_shift(i32x4_add(
        i32x4_add(i32x4_mul(cru, u_d), i32x4_mul(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = q15_shift(i32x4_add(
        i32x4_add(i32x4_mul(cgu, u_d), i32x4_mul(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = q15_shift(i32x4_add(
        i32x4_add(i32x4_mul(cbu, u_d), i32x4_mul(cbv, v_d)),
        rnd_v,
      ));

      let r_low = i16x8_narrow_i32x4(r_i32, r_i32);
      let g_low = i16x8_narrow_i32x4(g_i32, g_i32);
      let b_low = i16x8_narrow_i32x4(b_i32, b_i32);

      // 1→4 fan-out via byte-level shuffle. For each i16 chroma lane
      // we want the (low byte, high byte) pair repeated 4 times. The
      // two output vectors cover Y[0..8] (chroma c0..c1) and Y[8..16]
      // (chroma c2..c3) respectively.
      //
      // r_lo16 pattern: [c0,c0,c0,c0, c1,c1,c1,c1]
      //   bytes: [0,1, 0,1, 0,1, 0,1, 2,3, 2,3, 2,3, 2,3]
      // r_hi16 pattern: [c2,c2,c2,c2, c3,c3,c3,c3]
      //   bytes: [4,5, 4,5, 4,5, 4,5, 6,7, 6,7, 6,7, 6,7]
      let r_lo16 = i8x16_shuffle::<0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3>(r_low, r_low);
      let r_hi16 = i8x16_shuffle::<4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7>(r_low, r_low);
      let g_lo16 = i8x16_shuffle::<0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3>(g_low, g_low);
      let g_hi16 = i8x16_shuffle::<4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7>(g_low, g_low);
      let b_lo16 = i8x16_shuffle::<0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3>(b_low, b_low);
      let b_hi16 = i8x16_shuffle::<4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7>(b_low, b_low);

      // Y path: widen low / high 8 Y to i16x8, scale.
      let y_low_i16 = u8_low_to_i16x8(y_vec);
      let y_high_i16 = u8_high_to_i16x8(y_vec);
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = i16x8_add_sat(y_scaled_lo, b_lo16);
      let b_hi = i16x8_add_sat(y_scaled_hi, b_hi16);
      let g_lo = i16x8_add_sat(y_scaled_lo, g_lo16);
      let g_hi = i16x8_add_sat(y_scaled_hi, g_hi16);
      let r_lo = i16x8_add_sat(y_scaled_lo, r_lo16);
      let r_hi = i16x8_add_sat(y_scaled_hi, r_hi16);

      // Saturate-narrow to u8x16 per channel.
      let b_u8 = u8x16_narrow_i16x8(b_lo, b_hi);
      let g_u8 = u8x16_narrow_i16x8(g_lo, g_hi);
      let r_u8 = u8x16_narrow_i16x8(r_lo, r_hi);

      if ALPHA {
        write_rgba_16(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_16(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 16;
    }

    // Scalar tail. The SIMD loop strides 16 Y pixels (multiple of 4),
    // so `x` is a multiple of 4 ≤ width. The remaining 0..15 Y pixels
    // and chroma samples up to `width.div_ceil(4)` (FFmpeg ceil-shift)
    // — which may include a partial 1..3-pixel final chroma group —
    // are handled by the scalar reference.
    if x < width {
      let tail_w = width - x;
      let chroma_end = width.div_ceil(4);
      let tail_u = &u_quarter[x / 4..chroma_end];
      let tail_v = &v_quarter[x / 4..chroma_end];
      let tail_out = &mut out[x * bpp..width * bpp];
      if ALPHA {
        scalar::yuv_411_to_rgba_row(
          &y[x..width],
          tail_u,
          tail_v,
          tail_out,
          tail_w,
          matrix,
          full_range,
        );
      } else {
        scalar::yuv_411_to_rgb_row(
          &y[x..width],
          tail_u,
          tail_v,
          tail_out,
          tail_w,
          matrix,
          full_range,
        );
      }
    }
  }
}
