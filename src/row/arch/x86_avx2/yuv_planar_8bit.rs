use core::arch::x86_64::*;

use super::*;

/// AVX2 YUV 4:2:0 → packed RGB. Semantics match
/// [`scalar::yuv_420_to_rgb_row`] byte‑identically.
///
/// # Safety
///
/// The caller must uphold **all** of the following. Violating any
/// causes undefined behavior:
///
/// 1. **AVX2 must be available on the current CPU.** The dispatcher
///    in [`crate::row`] verifies this with
///    `is_x86_feature_detected!("avx2")` (runtime, std) or
///    `cfg!(target_feature = "avx2")` (compile‑time, no‑std). Calling
///    this kernel on a CPU without AVX2 triggers an illegal‑instruction
///    trap.
/// 2. `width & 1 == 0` (4:2:0 requires even width).
/// 3. `y.len() >= width`.
/// 4. `u_half.len() >= width / 2`.
/// 5. `v_half.len() >= width / 2`.
/// 6. `rgb_out.len() >= 3 * width`.
///
/// Bounds are verified by `debug_assert` in debug builds; release
/// builds trust the caller because the kernel relies on unchecked
/// pointer arithmetic (`_mm256_loadu_si256`, `_mm_loadu_si128`,
/// `_mm_storeu_si128`).
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420_to_rgb_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<false, false>(
      y, u_half, v_half, None, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 YUV 4:2:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_420_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// 1. AVX2 must be available on the current CPU.
/// 2. `width & 1 == 0`.
/// 3. `y.len() >= width`, `u_half.len() >= width / 2`,
///    `v_half.len() >= width / 2`, `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_420_to_rgba_row(
  y: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_420_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_420_to_rgb_or_rgba_row::<true, false>(
      y, u_half, v_half, None, rgba_out, width, matrix, full_range,
    );
  }
}

/// AVX2 YUVA 4:2:0 → packed **8-bit RGBA** with the per-pixel
/// alpha byte **sourced from `a_src`** (8-bit YUVA's alpha is already
/// `u8` — no depth conversion needed). Same numerical contract as
/// [`yuv_420_to_rgba_row`] for R/G/B.
///
/// Thin wrapper over [`yuv_420_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_420_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
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

/// Shared AVX2 kernel for [`yuv_420_to_rgb_row`] (`ALPHA = false,
/// ALPHA_SRC = false`, [`write_rgb_32`]), [`yuv_420_to_rgba_row`]
/// (`ALPHA = true, ALPHA_SRC = false`, [`write_rgba_32`] with constant
/// `0xFF` alpha) and [`yuv_420_to_rgba_with_alpha_src_row`]
/// (`ALPHA = true, ALPHA_SRC = true`, [`write_rgba_32`] with the
/// alpha lane loaded directly from `a_src`).
///
/// # Safety
///
/// Same as [`yuv_420_to_rgb_row`] / [`yuv_420_to_rgba_row`]; the
/// `out` slice must be `>= width * (if ALPHA { 4 } else { 3 })`
/// bytes long. When `ALPHA_SRC = true`: `a_src` must be `Some(_)`
/// and `a_src.unwrap().len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation per the
  // `# Safety` section; the dispatcher in `crate::row` checks it.
  // All pointer adds below are bounded by the `while x + 32 <= width`
  // loop condition and the caller‑promised slice lengths.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let mid128 = _mm256_set1_epi16(128);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    // Constant opaque-alpha vector for the RGBA path; DCE'd when
    // ALPHA = false.
    let alpha_u8 = _mm256_set1_epi8(-1); // 0xFF as i8

    let mut x = 0usize;
    while x + 32 <= width {
      // Load 32 Y, 16 U, 16 V.
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      let u_vec_128 = _mm_loadu_si128(u_half.as_ptr().add(x / 2).cast());
      let v_vec_128 = _mm_loadu_si128(v_half.as_ptr().add(x / 2).cast());

      // Widen U/V to i16x16 and subtract 128.
      let u_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_vec_128), mid128);
      let v_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_vec_128), mid128);

      // Split each i16x16 into two i32x8 halves for the Q15 multiplies
      // (coefficients exceed i16, so i32 precision is required).
      let u_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_i16));
      let u_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_i16));
      let v_lo_i32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16));
      let v_hi_i32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_i16));

      // u_d, v_d = (u * c_scale + RND) >> 15 — bit‑exact to scalar.
      let u_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_i32, c_scale_v),
        rnd_v,
      ));
      let u_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_lo = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_i32, c_scale_v),
        rnd_v,
      ));
      let v_d_hi = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_i32, c_scale_v),
        rnd_v,
      ));

      // Per‑channel chroma → i16x16 (natural order, fixup included).
      let r_chroma = chroma_i16x16(cru, crv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let g_chroma = chroma_i16x16(cgu, cgv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);
      let b_chroma = chroma_i16x16(cbu, cbv, u_d_lo, v_d_lo, u_d_hi, v_d_hi, rnd_v);

      // Nearest‑neighbor upsample: each of the 16 chroma lanes →
      // an adjacent pair, covering 32 Y lanes (split into low‑16 and
      // high‑16 i16x16 vectors).
      let (r_dup_lo, r_dup_hi) = chroma_dup(r_chroma);
      let (g_dup_lo, g_dup_hi) = chroma_dup(g_chroma);
      let (b_dup_lo, b_dup_hi) = chroma_dup(b_chroma);

      // Y path: widen 32 Y to two i16x16 vectors, subtract y_off,
      // apply y_scale in Q15, narrow back to i16.
      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate‑narrow to u8x32 per channel (lane‑fixup included).
      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 8-bit YUVA alpha is already u8; load 32 bytes via 256-bit
          // load.
          _mm256_loadu_si256(a_src.as_ref().unwrap_unchecked().as_ptr().add(x).cast())
        } else {
          alpha_u8
        };
        // 4‑way interleave → packed RGBA (128 bytes = 4 × 32).
        write_rgba_32(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        // 3‑way interleave → packed RGB (96 bytes = 3 × 32).
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail for the 0..30 leftover pixels (always even; 4:2:0
    // requires even width so x/2 and width/2 are well‑defined).
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
// ---- YUV 4:1:0 AVX2 entries -----------------------------------------
//
// 4:1:0: planar YUV with chroma subsampled 4:1 in **both** axes (each
// (U, V) sample covers a 4×4 luma block; vertical 4× re-use is the
// walker's job — chroma row = `y_row / 4`). This kernel handles the
// per-row 4× horizontal upsample. Math is byte-identical to scalar.
//
// Block size: 32 Y / 8 chroma per iteration (matches the 4:2:0 AVX2
// kernel's 32-Y throughput). The chroma upsample is implemented via
// per-128-bit-lane unpack chains (no lane-crossing fixups needed)
// because the 8 chroma samples split cleanly into two groups of 4
// — low 4 cover Y[0..16], high 4 cover Y[16..32], and each group is
// fanned via the same two-pass `_mm_unpack*_epi16` cascade as the
// SSE4.1 4:1:0 kernel.

/// AVX2 YUV 4:1:0 → packed RGB. Semantics match
/// [`scalar::yuv_410_to_rgb_row`] byte-identically.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width % 4 == 0` (4:1:0 requires width multiple of 4).
/// 3. `y.len() >= width`, `u_quarter.len() >= width / 4`,
///    `v_quarter.len() >= width / 4`, `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_410_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_410_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_410_to_rgb_or_rgba_row::<false>(
      y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 YUV 4:1:0 → packed **RGBA** (8-bit). Same contract as
/// [`yuv_410_to_rgb_row`] but writes 4 bytes per pixel (R, G, B,
/// `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_410_to_rgb_row`] except `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_410_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_410_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_410_to_rgb_or_rgba_row::<true>(
      y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX2 kernel for [`yuv_410_to_rgb_row`] (`ALPHA = false`,
/// [`write_rgb_32`]) and [`yuv_410_to_rgba_row`] (`ALPHA = true`,
/// [`write_rgba_32`] with constant `0xFF` alpha). Math is
/// byte-identical to `scalar::yuv_410_to_rgb_or_rgba_row::<ALPHA>`.
///
/// Pipeline per 32 Y pixels / 8 chroma samples:
/// 1. Load 32 Y + 8 U + 8 V (each chroma plane via a single `u64`
///    read since 8 bytes < 16).
/// 2. Widen 8 chroma to i16x8 (in 128-bit register), subtract 128,
///    widen to i32x8 (256-bit) for Q15 multiplies.
/// 3. `u_d = (u * c_scale + RND) >> 15`, same for `v_d` (i32x8).
/// 4. Per channel: `(C_u*u_d + C_v*v_d + RND) >> 15` (i32x8).
/// 5. Saturate-narrow each channel's i32x8 to i16x8 (256→128) via
///    `_mm256_packs_epi32` then permute to fix lane order.
/// 6. Split i16x8 into two i16x4 halves (low 4 chroma cover
///    Y[0..16], high 4 cover Y[16..32]). For each half apply the
///    same two-pass `_mm_unpack*_epi16` cascade as the SSE4.1
///    kernel: produces an i16x8 pair (low/high) covering 16 Y
///    lanes per chroma group; combine the four i16x8 vectors into
///    two i16x16 (one per Y[0..16] / Y[16..32] half).
/// 7. Y path: widen 32 Y → two i16x16, scale.
/// 8. Saturating add Y + chroma per channel (i16x16), saturate-
///    narrow to u8x32, interleave via [`write_rgb_32`] /
///    [`write_rgba_32`].
///
/// # Safety
///
/// Same as [`yuv_410_to_rgb_row`] / [`yuv_410_to_rgba_row`].
#[inline]
#[target_feature(enable = "avx2")]
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

  // SAFETY: AVX2 availability is the caller's obligation per the
  // `# Safety` section. All pointer adds below are bounded by the
  // `while x + 32 <= width` loop condition and caller-promised slice
  // lengths.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let mid128 = _mm_set1_epi16(128);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1);

    let mut x = 0usize;
    while x + 32 <= width {
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());

      // Load 8 chroma bytes per plane via an unaligned u64 read
      // splatted into the low 8 bytes of an XMM vector.
      let u_bytes = (u_quarter.as_ptr().add(x / 4) as *const u64).read_unaligned();
      let v_bytes = (v_quarter.as_ptr().add(x / 4) as *const u64).read_unaligned();
      let u_u64 = _mm_set_epi64x(0, u_bytes as i64);
      let v_u64 = _mm_set_epi64x(0, v_bytes as i64);

      // Widen 8 chroma bytes → i16x8, subtract 128 (in XMM).
      let u_i16x8 = _mm_sub_epi16(_mm_cvtepu8_epi16(u_u64), mid128);
      let v_i16x8 = _mm_sub_epi16(_mm_cvtepu8_epi16(v_u64), mid128);

      // Widen to i32x8 (256-bit) for Q15 multiplies.
      let u_i32x8 = _mm256_cvtepi16_epi32(u_i16x8);
      let v_i32x8 = _mm256_cvtepi16_epi32(v_i16x8);

      // u_d, v_d = (u * c_scale + RND) >> 15.
      let u_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_i32x8, c_scale_v),
        rnd_v,
      ));
      let v_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_i32x8, c_scale_v),
        rnd_v,
      ));

      // Per-channel chroma contribution as i32x8.
      let r_i32 = q15_shift(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(cru, u_d), _mm256_mullo_epi32(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = q15_shift(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(cgu, u_d), _mm256_mullo_epi32(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = q15_shift(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(cbu, u_d), _mm256_mullo_epi32(cbv, v_d)),
        rnd_v,
      ));

      // Saturate-narrow i32x8 → i16x8 (XMM). `_mm256_packs_epi32`
      // is per-128-bit-lane and produces lane-split output:
      // `[c0,c1,c2,c3, _,_,_,_, c4,c5,c6,c7, _,_,_,_]`. Permute via
      // `permute4x64<0xD8>` then take the low XMM gives natural
      // `[c0..c7]` in the low 8 i16 lanes.
      //
      // Equivalent and simpler at this size: pack the i32x8 down via
      // `permute4x64` then extract low 128 = i16x8 of [c0..c7].
      let r_pack = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(r_i32, r_i32));
      let g_pack = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(g_i32, g_i32));
      let b_pack = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(b_i32, b_i32));
      let r_chroma = _mm256_castsi256_si128(r_pack); // i16x8: [c0..c7]
      let g_chroma = _mm256_castsi256_si128(g_pack);
      let b_chroma = _mm256_castsi256_si128(b_pack);

      // Split [c0..c7] i16x8 into [c0,c1,c2,c3] and [c4,c5,c6,c7]
      // halves, then run the SSE4.1-style two-pass unpack cascade on
      // each half to produce four i16x8 vectors covering 32 Y lanes.
      // `_mm_srli_si128::<8>` shifts c4..c7 into the low half.
      let r_lo4 = r_chroma; // [c0,c1,c2,c3, _,_,_,_]
      let r_hi4 = _mm_srli_si128::<8>(r_chroma); // [c4,c5,c6,c7, _,_,_,_]
      let g_lo4 = g_chroma;
      let g_hi4 = _mm_srli_si128::<8>(g_chroma);
      let b_lo4 = b_chroma;
      let b_hi4 = _mm_srli_si128::<8>(b_chroma);

      // First pass: each [c_a,c_b,c_c,c_d] → [c_a,c_a,c_b,c_b,c_c,c_c,c_d,c_d].
      let r_pair_lo = _mm_unpacklo_epi16(r_lo4, r_lo4);
      let r_pair_hi = _mm_unpacklo_epi16(r_hi4, r_hi4);
      let g_pair_lo = _mm_unpacklo_epi16(g_lo4, g_lo4);
      let g_pair_hi = _mm_unpacklo_epi16(g_hi4, g_hi4);
      let b_pair_lo = _mm_unpacklo_epi16(b_lo4, b_lo4);
      let b_pair_hi = _mm_unpacklo_epi16(b_hi4, b_hi4);

      // Second pass on each pair: produces i16x8 [c_a×4, c_b×4]
      // (lo) and [c_c×4, c_d×4] (hi). For each chroma half (lo4 /
      // hi4) this gives two i16x8 vectors covering 16 Y lanes
      // (8 + 8). Combining with `_mm256_set_m128i` builds the
      // i16x16 covering Y[0..16] and Y[16..32] per channel.
      let r_q0 = _mm_unpacklo_epi16(r_pair_lo, r_pair_lo); // c0×4, c1×4 → Y[0..8]
      let r_q1 = _mm_unpackhi_epi16(r_pair_lo, r_pair_lo); // c2×4, c3×4 → Y[8..16]
      let r_q2 = _mm_unpacklo_epi16(r_pair_hi, r_pair_hi); // c4×4, c5×4 → Y[16..24]
      let r_q3 = _mm_unpackhi_epi16(r_pair_hi, r_pair_hi); // c6×4, c7×4 → Y[24..32]
      let g_q0 = _mm_unpacklo_epi16(g_pair_lo, g_pair_lo);
      let g_q1 = _mm_unpackhi_epi16(g_pair_lo, g_pair_lo);
      let g_q2 = _mm_unpacklo_epi16(g_pair_hi, g_pair_hi);
      let g_q3 = _mm_unpackhi_epi16(g_pair_hi, g_pair_hi);
      let b_q0 = _mm_unpacklo_epi16(b_pair_lo, b_pair_lo);
      let b_q1 = _mm_unpackhi_epi16(b_pair_lo, b_pair_lo);
      let b_q2 = _mm_unpacklo_epi16(b_pair_hi, b_pair_hi);
      let b_q3 = _mm_unpackhi_epi16(b_pair_hi, b_pair_hi);

      // Combine into i16x16 vectors. Each `set_m128i(hi, lo)` puts
      // `lo` in lane 0 and `hi` in lane 1 (low/high 128-bit halves).
      let r_dup_lo = _mm256_set_m128i(r_q1, r_q0); // Y[0..16]
      let r_dup_hi = _mm256_set_m128i(r_q3, r_q2); // Y[16..32]
      let g_dup_lo = _mm256_set_m128i(g_q1, g_q0);
      let g_dup_hi = _mm256_set_m128i(g_q3, g_q2);
      let b_dup_lo = _mm256_set_m128i(b_q1, b_q0);
      let b_dup_hi = _mm256_set_m128i(b_q3, b_q2);

      // Y path: widen 32 Y → two i16x16, scale.
      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_dup_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_dup_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_dup_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_dup_hi);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_dup_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_dup_hi);

      // Saturate-narrow per channel → u8x32 (with lane fixup).
      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail. `width % 4 == 0` so `width - x` is also a multiple
    // of 4.
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

/// AVX2 YUV 4:4:4 planar → packed RGB. Thin wrapper over
/// [`yuv_444_to_rgb_or_rgba_row`] with `ALPHA = false`.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`,
///    `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444_to_rgb_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<false, false>(y, u, v, None, rgb_out, width, matrix, full_range);
  }
}

/// AVX2 YUV 4:4:4 planar → packed **RGBA** (8-bit). Same contract
/// as [`yuv_444_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_32`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_444_to_rgb_row`] except the output slice must be
/// `>= 4 * width` bytes.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_444_to_rgba_row(
  y: &[u8],
  u: &[u8],
  v: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_444_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_444_to_rgb_or_rgba_row::<true, false>(y, u, v, None, rgba_out, width, matrix, full_range);
  }
}

/// AVX2 YUVA 4:4:4 → packed **RGBA** with source alpha. R/G/B are
/// byte-identical to [`yuv_444_to_rgb_row`]; the per-pixel alpha byte
/// is sourced from `a_src` (8-bit, no shift needed) instead of being
/// constant `0xFF`. Used by [`crate::yuv::Yuva444p`].
///
/// Thin wrapper over [`yuv_444_to_rgb_or_rgba_row`] with
/// `ALPHA = true, ALPHA_SRC = true`.
///
/// # Safety
///
/// Same as [`yuv_444_to_rgba_row`] plus `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
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

/// Shared AVX2 YUV 4:4:4 kernel.
/// - `ALPHA = false, ALPHA_SRC = false`: [`write_rgb_32`].
/// - `ALPHA = true, ALPHA_SRC = false`: [`write_rgba_32`] with constant
///   `0xFF` alpha.
/// - `ALPHA = true, ALPHA_SRC = true`: [`write_rgba_32`] with the
///   alpha lane loaded from `a_src` (8-bit input — no shift needed).
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `y.len() >= width`, `u.len() >= width`, `v.len() >= width`.
/// 3. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
/// 4. If `ALPHA_SRC = true`, `a_src` is `Some(_)` with
///    `a_src.len() >= width`.
#[inline]
#[target_feature(enable = "avx2")]
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
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let mid128 = _mm256_set1_epi16(128);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1); // 0xFF as i8

    let mut x = 0usize;
    while x + 32 <= width {
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());
      // 4:4:4: 32 U + 32 V directly, no deinterleave.
      let u_vec = _mm256_loadu_si256(u.as_ptr().add(x).cast());
      let v_vec = _mm256_loadu_si256(v.as_ptr().add(x).cast());

      // Widen low / high halves of U / V (16 bytes each) to i16x16.
      let u_lo_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_vec)), mid128);
      let u_hi_i16 = _mm256_sub_epi16(
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(u_vec)),
        mid128,
      );
      let v_lo_i16 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_vec)), mid128);
      let v_hi_i16 = _mm256_sub_epi16(
        _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(v_vec)),
        mid128,
      );

      let u_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_lo_i16));
      let u_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_lo_i16));
      let u_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u_hi_i16));
      let u_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(u_hi_i16));
      let v_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_lo_i16));
      let v_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_lo_i16));
      let v_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_hi_i16));
      let v_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v_hi_i16));

      let u_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_a, c_scale_v),
        rnd_v,
      ));
      let u_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_lo_b, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_a, c_scale_v),
        rnd_v,
      ));
      let u_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_hi_b, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_a, c_scale_v),
        rnd_v,
      ));
      let v_d_lo_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_lo_b, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_a = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_a, c_scale_v),
        rnd_v,
      ));
      let v_d_hi_b = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_hi_b, c_scale_v),
        rnd_v,
      ));

      let r_chroma_lo = chroma_i16x16(cru, crv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let r_chroma_hi = chroma_i16x16(cru, crv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let g_chroma_lo = chroma_i16x16(cgu, cgv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let g_chroma_hi = chroma_i16x16(cgu, cgv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);
      let b_chroma_lo = chroma_i16x16(cbu, cbv, u_d_lo_a, v_d_lo_a, u_d_lo_b, v_d_lo_b, rnd_v);
      let b_chroma_hi = chroma_i16x16(cbu, cbv, u_d_hi_a, v_d_hi_a, u_d_hi_b, v_d_hi_b, rnd_v);

      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_chroma_lo);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_chroma_hi);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_chroma_lo);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_chroma_hi);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_chroma_lo);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_chroma_hi);

      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        let a_u8 = if ALPHA_SRC {
          // SAFETY (const-checked): ALPHA_SRC = true implies the
          // wrapper passed Some(_), validated by debug_assert above.
          // 8-bit alpha — load 32 bytes verbatim.
          _mm256_loadu_si256(a_src.as_ref().unwrap_unchecked().as_ptr().add(x).cast())
        } else {
          alpha_u8
        };
        write_rgba_32(r_u8, g_u8, b_u8, a_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
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

// ---- YUV 4:1:1 → RGB / RGBA (AVX2) -------------------------------------

/// AVX2 YUV 4:1:1 planar → packed RGB. One chroma sample drives four
/// Y pixels (1→4 nearest-neighbor upsample). Processes 32 Y / 8 chroma
/// samples per iteration — matches the AVX2 4:2:0 block size with
/// 1/4 the chroma load count.
///
/// Same Q15 arithmetic as the scalar reference; output is byte-identical.
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width % 4 == 0`.
/// 3. `y.len() >= width`, `u_quarter.len() >= width / 4`,
///    `v_quarter.len() >= width / 4`.
/// 4. `rgb_out.len() >= 3 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_411_to_rgb_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgb_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_411_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_411_to_rgb_or_rgba_row::<false>(
      y, u_quarter, v_quarter, rgb_out, width, matrix, full_range,
    );
  }
}

/// AVX2 YUV 4:1:1 planar → packed **RGBA** (8-bit). Same contract as
/// [`yuv_411_to_rgb_row`] but writes 4 bytes per pixel via
/// [`write_rgba_32`] (R, G, B, `0xFF`).
///
/// # Safety
///
/// Same as [`yuv_411_to_rgb_row`] except `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_411_to_rgba_row(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  rgba_out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  // SAFETY: caller-checked AVX2 availability + slice bounds — see
  // [`yuv_411_to_rgb_or_rgba_row`] safety contract.
  unsafe {
    yuv_411_to_rgb_or_rgba_row::<true>(
      y, u_quarter, v_quarter, rgba_out, width, matrix, full_range,
    );
  }
}

/// Shared AVX2 YUV 4:1:1 kernel. Processes 32 Y pixels (= 8 chroma
/// samples) per iteration; the 1→4 chroma upsample is materialized
/// across two i16x16 vectors covering the 32 Y lanes:
///
/// 1. Compute 8 chroma values per channel as i16x8 in `__m128i` (only
///    the low 4 i16 lanes hold real data initially per i32x4 source;
///    we use both u_d_lo and u_d_hi to fill all 8 lanes — see kernel
///    body for layout).
/// 2. Stage 1: per-128-bit-lane `_mm_unpacklo_epi16` /
///    `_mm_unpackhi_epi16` cascade with the chroma broadcast into a
///    `__m256i` covers the 1→4 fan-out for the low 16 Y lanes
///    (`Y[0..16]`, fed by `c0..c3`) and the high 16 Y lanes
///    (`Y[16..32]`, fed by `c4..c7`).
///
/// 4:1:1 has no source-alpha variant, so the const-generic surface
/// stays 1-D (`ALPHA` only).
///
/// # Safety
///
/// 1. **AVX2 must be available on the current CPU.**
/// 2. `width % 4 == 0`.
/// 3. `y.len() >= width`, `u_quarter.len() >= width / 4`,
///    `v_quarter.len() >= width / 4`.
/// 4. `out.len() >= width * (if ALPHA { 4 } else { 3 })`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn yuv_411_to_rgb_or_rgba_row<const ALPHA: bool>(
  y: &[u8],
  u_quarter: &[u8],
  v_quarter: &[u8],
  out: &mut [u8],
  width: usize,
  matrix: ColorMatrix,
  full_range: bool,
) {
  debug_assert_eq!(width & 3, 0, "YUV 4:1:1 requires width % 4 == 0");
  debug_assert!(y.len() >= width);
  debug_assert!(u_quarter.len() >= width / 4);
  debug_assert!(v_quarter.len() >= width / 4);
  let bpp: usize = if ALPHA { 4 } else { 3 };
  debug_assert!(out.len() >= width * bpp);

  let coeffs = scalar::Coefficients::for_matrix(matrix);
  let (y_off, y_scale, c_scale) = scalar::range_params_n::<8, 8>(full_range);
  const RND: i32 = 1 << 14;

  // SAFETY: AVX2 availability is the caller's obligation per the
  // `# Safety` section. All pointer adds below are bounded by the
  // `while x + 32 <= width` loop condition and the caller-promised
  // slice lengths.
  unsafe {
    let rnd_v = _mm256_set1_epi32(RND);
    let y_off_v = _mm256_set1_epi16(y_off as i16);
    let y_scale_v = _mm256_set1_epi32(y_scale);
    let c_scale_v = _mm256_set1_epi32(c_scale);
    let mid128 = _mm256_set1_epi16(128);
    let cru = _mm256_set1_epi32(coeffs.r_u());
    let crv = _mm256_set1_epi32(coeffs.r_v());
    let cgu = _mm256_set1_epi32(coeffs.g_u());
    let cgv = _mm256_set1_epi32(coeffs.g_v());
    let cbu = _mm256_set1_epi32(coeffs.b_u());
    let cbv = _mm256_set1_epi32(coeffs.b_v());
    let alpha_u8 = _mm256_set1_epi8(-1); // 0xFF as i8

    let mut x = 0usize;
    while x + 32 <= width {
      // Load 32 Y bytes.
      let y_vec = _mm256_loadu_si256(y.as_ptr().add(x).cast());

      // Load 8 chroma bytes per 32 Y pixels (4:1:1 quarter-rate).
      // Use `_mm_loadl_epi64` to put the 8 bytes in the low 8 lanes of
      // a __m128i, then widen to i16x8 and zero-extend to a __m256i
      // (low 128 lanes hold chroma c0..c7, high 128 lanes are zero —
      // we don't read from there).
      let u_8 = _mm_loadl_epi64(u_quarter.as_ptr().add(x / 4).cast());
      let v_8 = _mm_loadl_epi64(v_quarter.as_ptr().add(x / 4).cast());

      // Widen 8 u8 → 8 i16, subtract 128. We only need an __m128i for
      // the i16 form here; we'll split into two i32x4 halves for the
      // Q15 multiplies.
      let u_i16_128 = _mm_sub_epi16(_mm_cvtepu8_epi16(u_8), _mm256_castsi256_si128(mid128));
      let v_i16_128 = _mm_sub_epi16(_mm_cvtepu8_epi16(v_8), _mm256_castsi256_si128(mid128));

      // Promote to i32x8 (sign-extending) and scale via Q15. We use
      // 256-bit width here so the per-channel chroma arithmetic stays
      // on i32x8, matching the AVX2 4:2:0 layout.
      let u_i32 = _mm256_cvtepi16_epi32(u_i16_128);
      let v_i32 = _mm256_cvtepi16_epi32(v_i16_128);

      let u_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(u_i32, c_scale_v),
        rnd_v,
      ));
      let v_d = q15_shift(_mm256_add_epi32(
        _mm256_mullo_epi32(v_i32, c_scale_v),
        rnd_v,
      ));

      // Per-channel chroma → i32x8 (8 chroma values per channel).
      let r_i32 = q15_shift(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(cru, u_d), _mm256_mullo_epi32(crv, v_d)),
        rnd_v,
      ));
      let g_i32 = q15_shift(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(cgu, u_d), _mm256_mullo_epi32(cgv, v_d)),
        rnd_v,
      ));
      let b_i32 = q15_shift(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(cbu, u_d), _mm256_mullo_epi32(cbv, v_d)),
        rnd_v,
      ));

      // Saturating-pack each i32x8 → i16. `_mm256_packs_epi32(x, x)`
      // produces lane-split [c0..3, c0..3, c4..7, c4..7] within a
      // __m256i; permute4x64::<0xD8>` reorders 64-bit lanes
      // [0,2,1,3] → [c0..3, c4..7, c0..3, c4..7] putting c0..c7 in
      // the low 128 lanes and a useless duplicate in the high 128
      // lanes (which we never read).
      let r_i16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(r_i32, r_i32));
      let g_i16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(g_i32, g_i32));
      let b_i16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(b_i32, b_i32));
      // Now lanes 0..7 of r_i16 / g_i16 / b_i16 hold c0..c7 (i16).
      // Extract that __m128i for the 1→4 fan-out cascade below.
      let r_c8 = _mm256_castsi256_si128(r_i16);
      let g_c8 = _mm256_castsi256_si128(g_i16);
      let b_c8 = _mm256_castsi256_si128(b_i16);

      // 1→4 nearest-neighbor upsample. Two-stage cascade.
      //
      // CRITICAL: `_mm256_unpacklo_epi16` operates per-128-bit lane and
      // only consumes lanes 0..3 of each input lane. A naive broadcast
      // (`inserti128` of `r_c8` into both halves) would therefore expand
      // c0..c3 in BOTH lanes and silently drop c4..c7 — pixels 16..31
      // would reuse the first half's chroma. To use c4..c7 in the high
      // lane, we splice it into the low 64 bits of the high lane via
      // `_mm_unpackhi_epi64(r_c8, r_c8)` (or equivalently a 64-bit
      // shuffle), then `inserti128` that into the high lane.
      //
      // Stage 1 (i16x8 → i16x16, each lane duplicated once): per-lane
      // unpacklo on a vector whose low lane low-64 holds c0..c3 and
      // high lane low-64 holds c4..c7 yields `[c0,c0,c1,c1,c2,c2,c3,c3]`
      // in the low lane and `[c4,c4,c5,c5,c6,c6,c7,c7]` in the high
      // lane.
      let r_c8_hi = _mm_unpackhi_epi64(r_c8, r_c8); // [c4,c5,c6,c7, c4,c5,c6,c7]
      let g_c8_hi = _mm_unpackhi_epi64(g_c8, g_c8);
      let b_c8_hi = _mm_unpackhi_epi64(b_c8, b_c8);
      let r_bcast = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(r_c8), r_c8_hi);
      let g_bcast = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(g_c8), g_c8_hi);
      let b_bcast = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(b_c8), b_c8_hi);
      let r_dup8 = _mm256_unpacklo_epi16(r_bcast, r_bcast);
      let g_dup8 = _mm256_unpacklo_epi16(g_bcast, g_bcast);
      let b_dup8 = _mm256_unpacklo_epi16(b_bcast, b_bcast);
      // r_dup8 = lo lane [c0,c0,c1,c1,c2,c2,c3,c3], hi lane [c4,c4,c5,c5,c6,c6,c7,c7].

      // Stage 2: re-apply per-lane unpack on stage-1 output.
      let u_lo = _mm256_unpacklo_epi16(r_dup8, r_dup8);
      let u_hi = _mm256_unpackhi_epi16(r_dup8, r_dup8);
      // u_lo: lo lane = [c0,c0,c0,c0, c1,c1,c1,c1]; hi lane = [c4×4, c5×4].
      // u_hi: lo lane = [c2,c2,c2,c2, c3,c3,c3,c3]; hi lane = [c6×4, c7×4].
      // Reassemble for natural Y-order:
      //   r_lo16 (Y[0..16]) = [c0×4, c1×4, c2×4, c3×4]
      //                     = lo-lane(u_lo) ++ lo-lane(u_hi)
      //                     = _mm256_permute2x128::<0x20>(u_lo, u_hi).
      //   r_hi16 (Y[16..32]) = [c4×4, c5×4, c6×4, c7×4]
      //                     = hi-lane(u_lo) ++ hi-lane(u_hi)
      //                     = _mm256_permute2x128::<0x31>(u_lo, u_hi).
      let r_lo16 = _mm256_permute2x128_si256::<0x20>(u_lo, u_hi);
      let r_hi16 = _mm256_permute2x128_si256::<0x31>(u_lo, u_hi);

      let g_u_lo = _mm256_unpacklo_epi16(g_dup8, g_dup8);
      let g_u_hi = _mm256_unpackhi_epi16(g_dup8, g_dup8);
      let g_lo16 = _mm256_permute2x128_si256::<0x20>(g_u_lo, g_u_hi);
      let g_hi16 = _mm256_permute2x128_si256::<0x31>(g_u_lo, g_u_hi);

      let b_u_lo = _mm256_unpacklo_epi16(b_dup8, b_dup8);
      let b_u_hi = _mm256_unpackhi_epi16(b_dup8, b_dup8);
      let b_lo16 = _mm256_permute2x128_si256::<0x20>(b_u_lo, b_u_hi);
      let b_hi16 = _mm256_permute2x128_si256::<0x31>(b_u_lo, b_u_hi);

      // Y path: widen 32 Y to two i16x16 vectors, scale.
      let y_low_i16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_vec));
      let y_high_i16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_vec));
      let y_scaled_lo = scale_y(y_low_i16, y_off_v, y_scale_v, rnd_v);
      let y_scaled_hi = scale_y(y_high_i16, y_off_v, y_scale_v, rnd_v);

      // Saturating i16 add Y + chroma per channel.
      let b_lo = _mm256_adds_epi16(y_scaled_lo, b_lo16);
      let b_hi = _mm256_adds_epi16(y_scaled_hi, b_hi16);
      let g_lo = _mm256_adds_epi16(y_scaled_lo, g_lo16);
      let g_hi = _mm256_adds_epi16(y_scaled_hi, g_hi16);
      let r_lo = _mm256_adds_epi16(y_scaled_lo, r_lo16);
      let r_hi = _mm256_adds_epi16(y_scaled_hi, r_hi16);

      // Saturate-narrow to u8x32 per channel (lane fixup included).
      let b_u8 = narrow_u8x32(b_lo, b_hi);
      let g_u8 = narrow_u8x32(g_lo, g_hi);
      let r_u8 = narrow_u8x32(r_lo, r_hi);

      if ALPHA {
        write_rgba_32(r_u8, g_u8, b_u8, alpha_u8, out.as_mut_ptr().add(x * 4));
      } else {
        write_rgb_32(r_u8, g_u8, b_u8, out.as_mut_ptr().add(x * 3));
      }

      x += 32;
    }

    // Scalar tail. 4:1:1 requires width % 4 == 0; the SIMD loop strides
    // 32, so widths in {4, 8, 12, ..., 28, 36, 40, ...} can leave a
    // multiple-of-4 tail.
    if x < width {
      let tail_w = width - x;
      let tail_u = &u_quarter[x / 4..width / 4];
      let tail_v = &v_quarter[x / 4..width / 4];
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
