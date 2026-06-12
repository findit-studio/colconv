//! NEON kernels for the Tier 9 packed-float-RGB (`Rgbf32`) source.
//!
//! Each kernel processes 4 `f32` lanes per iteration via NEON's
//! `float32x4_t` registers. The conversions are componentwise:
//!
//! ```text
//!   clamped = max(min(v, 1.0), 0.0)
//!   scaled  = clamped * out_max         // round-to-nearest (vcvtnq_*)
//!   out_int = saturating_cast(scaled)   // already in-range after clamp
//! ```
//!
//! Operating in `f32` lanes means the loop is **lane-grouped**, not
//! pixel-grouped: the 4-lane vector covers e.g. `[R0, G0, B0, R1]`.
//! That is fine for the integer-output kernels (we narrow the lane
//! vector to 4 bytes / 4 u16 elements with `vst*` straight into the
//! `R, G, B, R, …` packed output) and trivially fine for the lossless
//! `f32` pass-through (just `vst1q_f32`).
//!
//! For `<const BE: bool>` kernels, each 4-lane f32 load is replaced by
//! an endian-aware u32x4 load (via `load_endian_u32x4::<BE>`) followed
//! by a `vreinterpretq_f32_u32` cast. For LE (BE=false) this is a
//! pure load; for BE it adds a `vrev32q_u8` byte-swap.

#[cfg_attr(miri, allow(unused_imports))]
use core::arch::aarch64::*;

use super::{endian::load_endian_u32x4, scalar};

/// Load 4 `f32` lanes from `ptr` in endian-aware fashion.
/// `BE = false` → host-native load (identical to `vld1q_f32`).
/// `BE = true`  → load as u32 with byte-swap, then reinterpret as f32.
///
/// # Safety
///
/// * NEON must be available.
/// * `ptr` must be valid for 16 bytes.
#[inline(always)]
unsafe fn load_f32x4<const BE: bool>(ptr: *const f32) -> float32x4_t {
  unsafe {
    let u = load_endian_u32x4::<BE>(ptr as *const u8);
    vreinterpretq_f32_u32(u)
  }
}

/// f32 RGB → u8 RGB. Clamp `[0, 1]` x 255, saturating round-to-nearest
/// cast.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Safety
///
/// 1. NEON must be available (`is_aarch64_feature_detected!("neon")`).
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgbf32_to_rgb_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(255.0);

    // Iterate in **lane-multiples-of-3 = pixel-aligned** chunks. We
    // process 12 f32 lanes (= 4 pixels = 12 output bytes) per iter so
    // every chunk lands on a pixel boundary.
    let total_lanes = width * 3;
    let mut lane = 0usize;
    while lane + 12 <= total_lanes {
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

      let s0 = vmulq_f32(vminq_f32(vmaxq_f32(v0, zero), one), scale);
      let s1 = vmulq_f32(vminq_f32(vmaxq_f32(v1, zero), one), scale);
      let s2 = vmulq_f32(vminq_f32(vmaxq_f32(v2, zero), one), scale);

      let u0 = vqmovn_u32(vcvtnq_u32_f32(s0));
      let u1 = vqmovn_u32(vcvtnq_u32_f32(s1));
      let u2 = vqmovn_u32(vcvtnq_u32_f32(s2));

      // Narrow each u16x4 to u8x4 via vqmovn_u16(vcombine_u16(x, x))
      // and emit 12 bytes via three 4-byte stores.
      let b0 = vqmovn_u16(vcombine_u16(u0, u0));
      let b1 = vqmovn_u16(vcombine_u16(u1, u1));
      let b2 = vqmovn_u16(vcombine_u16(u2, u2));

      let mut tmp = [0u8; 8];
      vst1_u8(tmp.as_mut_ptr(), b0);
      rgb_out
        .get_unchecked_mut(lane..lane + 4)
        .copy_from_slice(&tmp[..4]);
      vst1_u8(tmp.as_mut_ptr(), b1);
      rgb_out
        .get_unchecked_mut(lane + 4..lane + 8)
        .copy_from_slice(&tmp[..4]);
      vst1_u8(tmp.as_mut_ptr(), b2);
      rgb_out
        .get_unchecked_mut(lane + 8..lane + 12)
        .copy_from_slice(&tmp[..4]);

      lane += 12;
    }

    // Scalar tail handles the leftover 0–3 pixels.
    let pix_done = lane / 3;
    let tail_pix = width - pix_done;
    if tail_pix > 0 {
      scalar::rgbf32_to_rgb_row::<BE>(
        &rgb_in[pix_done * 3..width * 3],
        &mut rgb_out[pix_done * 3..width * 3],
        tail_pix,
      );
    }
  }
}

/// f32 RGB → u8 RGBA (alpha forced to `0xFF`).
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgbf32_to_rgba_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(255.0);
    let alpha = vdupq_n_u8(0xFF);

    // Process 4 pixels per iteration — 12 input lanes → 16 output bytes
    // (R0 G0 B0 A0 R1 G1 B1 A1 …) using `vst4q_u8` with the 4 channels
    // gathered into separate u8x16 vectors.
    //
    // For a 4-pixel iteration we need 4 R lanes, 4 G lanes, 4 B lanes
    // — gathered with strided `vld3q_f32` then converted lanewise.
    let mut x = 0usize;
    while x + 16 <= width {
      let mut r_bytes = [0u8; 16];
      let mut g_bytes = [0u8; 16];
      let mut b_bytes = [0u8; 16];
      // Inner loop covers 16 pixels (4 NEON 4-pixel sub-blocks). We
      // fall back to per-pixel scalar conversion of the loaded lanes
      // — the f32→u8 cast itself is the cost, not the gather.
      for sub in 0..4 {
        let base = (x + sub * 4) * 3;
        // Fast path: on-disk encoding (`BE`) matches host-native, so the raw
        // `vld3q_f32` reads host-native bytes that already carry the right f32
        // values. Slow path (`BE != HOST_NATIVE_BE`): each f32 must be byte-
        // swapped before deinterleave; load via the endian-aware helper into
        // a contiguous buffer, then unstride into per-channel vectors.
        let (r_v, g_v, b_v) = if BE == HOST_NATIVE_BE {
          let v_rgb = vld3q_f32(rgb_in.as_ptr().add(base));
          (v_rgb.0, v_rgb.1, v_rgb.2)
        } else {
          let raw0 = load_f32x4::<BE>(rgb_in.as_ptr().add(base));
          let raw1 = load_f32x4::<BE>(rgb_in.as_ptr().add(base + 4));
          let raw2 = load_f32x4::<BE>(rgb_in.as_ptr().add(base + 8));
          // Manual deinterleave: contiguous host-native f32 layout is
          // [R0,G0,B0,R1, G1,B1,R2,G2, B2,R3,G3,B3] across raw{0,1,2}.
          let mut r_arr = [0.0f32; 4];
          let mut g_arr = [0.0f32; 4];
          let mut b_arr = [0.0f32; 4];
          vst1q_f32(r_arr.as_mut_ptr(), raw0);
          vst1q_f32(g_arr.as_mut_ptr(), raw1);
          vst1q_f32(b_arr.as_mut_ptr(), raw2);
          let r_deint = [r_arr[0], r_arr[3], g_arr[2], b_arr[1]];
          let g_deint = [r_arr[1], g_arr[0], g_arr[3], b_arr[2]];
          let b_deint = [r_arr[2], g_arr[1], b_arr[0], b_arr[3]];
          (
            vld1q_f32(r_deint.as_ptr()),
            vld1q_f32(g_deint.as_ptr()),
            vld1q_f32(b_deint.as_ptr()),
          )
        };

        let r_clamped = vmulq_f32(vminq_f32(vmaxq_f32(r_v, zero), one), scale);
        let g_clamped = vmulq_f32(vminq_f32(vmaxq_f32(g_v, zero), one), scale);
        let b_clamped = vmulq_f32(vminq_f32(vmaxq_f32(b_v, zero), one), scale);

        let r_u32 = vcvtnq_u32_f32(r_clamped);
        let g_u32 = vcvtnq_u32_f32(g_clamped);
        let b_u32 = vcvtnq_u32_f32(b_clamped);

        // Narrow u32x4 → u8x4, store directly into the per-channel
        // staging arrays. Each sub block contributes 4 bytes.
        let r_u16 = vqmovn_u32(r_u32);
        let g_u16 = vqmovn_u32(g_u32);
        let b_u16 = vqmovn_u32(b_u32);
        // vqmovn produces a 4-element u16 vector; combine with itself
        // to make 8, narrow with vqmovn_u16, then write 4 bytes from
        // the low half.
        let r_u8 = vqmovn_u16(vcombine_u16(r_u16, r_u16));
        let g_u8 = vqmovn_u16(vcombine_u16(g_u16, g_u16));
        let b_u8 = vqmovn_u16(vcombine_u16(b_u16, b_u16));
        let mut tmp = [0u8; 8];
        vst1_u8(tmp.as_mut_ptr(), r_u8);
        r_bytes[sub * 4..sub * 4 + 4].copy_from_slice(&tmp[..4]);
        vst1_u8(tmp.as_mut_ptr(), g_u8);
        g_bytes[sub * 4..sub * 4 + 4].copy_from_slice(&tmp[..4]);
        vst1_u8(tmp.as_mut_ptr(), b_u8);
        b_bytes[sub * 4..sub * 4 + 4].copy_from_slice(&tmp[..4]);
      }
      let r = vld1q_u8(r_bytes.as_ptr());
      let g = vld1q_u8(g_bytes.as_ptr());
      let b = vld1q_u8(b_bytes.as_ptr());
      let rgba = uint8x16x4_t(r, g, b, alpha);
      vst4q_u8(rgba_out.as_mut_ptr().add(x * 4), rgba);
      x += 16;
    }

    if x < width {
      scalar::rgbf32_to_rgba_row::<BE>(
        &rgb_in[x * 3..width * 3],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// f32 RGB → u16 RGB. Clamp `[0, 1]` x 65535, saturating cast.
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_row`] but `rgb_out` is `&mut [u16]` with
/// `len() >= 3 * width` u16 elements.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgbf32_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(65535.0);

    // Process pixel-aligned chunks of 4 pixels = 12 lanes per iter.
    let total_lanes = width * 3;
    let mut lane = 0usize;
    while lane + 12 <= total_lanes {
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

      let s0 = vmulq_f32(vminq_f32(vmaxq_f32(v0, zero), one), scale);
      let s1 = vmulq_f32(vminq_f32(vmaxq_f32(v1, zero), one), scale);
      let s2 = vmulq_f32(vminq_f32(vmaxq_f32(v2, zero), one), scale);
      let u0 = vqmovn_u32(vcvtnq_u32_f32(s0));
      let u1 = vqmovn_u32(vcvtnq_u32_f32(s1));
      let u2 = vqmovn_u32(vcvtnq_u32_f32(s2));

      vst1_u16(rgb_out.as_mut_ptr().add(lane), u0);
      vst1_u16(rgb_out.as_mut_ptr().add(lane + 4), u1);
      vst1_u16(rgb_out.as_mut_ptr().add(lane + 8), u2);
      lane += 12;
    }
    let pix_done = lane / 3;
    let tail_pix = width - pix_done;
    if tail_pix > 0 {
      scalar::rgbf32_to_rgb_u16_row::<BE>(
        &rgb_in[pix_done * 3..width * 3],
        &mut rgb_out[pix_done * 3..width * 3],
        tail_pix,
      );
    }
  }
}

/// f32 RGB → u16 RGBA (alpha forced to `0xFFFF`).
///
/// When `BE = true` the input `f32` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_u16_row`] but the output is `&mut [u16]`
/// with `len() >= 4 * width` u16 elements.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgbf32_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_u16_out row too short");

  unsafe {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let scale = vdupq_n_f32(65535.0);
    let alpha_v = vdupq_n_u16(0xFFFF);

    let mut x = 0usize;
    while x + 8 <= width {
      // 8 pixels = 24 input lanes processed in three vld3q_f32 blocks
      // (each loads 4 pixels). We split into two 4-pixel sub-blocks.
      //
      // For each 4-pixel sub-block: deinterleave R, G, B → three
      // f32x4 vectors, clamp+scale+convert, then narrow each 4-element
      // u32 vector to 4 u16. Build R/G/B/A as u16x8 vectors and
      // interleave with `vst4q_u16`.
      let mut r_h = [0u16; 8];
      let mut g_h = [0u16; 8];
      let mut b_h = [0u16; 8];
      for sub in 0..2 {
        let base = (x + sub * 4) * 3;
        // Fast path: `BE == HOST_NATIVE_BE` means on-disk encoding matches the
        // host's native byte order, so `vld3q_f32` (which always reads host-
        // native bytes) decodes the encoded f32s correctly. Slow path: the
        // encoded bytes are foreign — load each f32 through the endian-aware
        // helper (which byte-swaps when `BE != HOST_NATIVE_BE`) into a
        // contiguous buffer, then deinterleave into per-channel vectors.
        let (r_v, g_v, b_v) = if BE == HOST_NATIVE_BE {
          let v_rgb = vld3q_f32(rgb_in.as_ptr().add(base));
          (v_rgb.0, v_rgb.1, v_rgb.2)
        } else {
          let raw0 = load_f32x4::<BE>(rgb_in.as_ptr().add(base));
          let raw1 = load_f32x4::<BE>(rgb_in.as_ptr().add(base + 4));
          let raw2 = load_f32x4::<BE>(rgb_in.as_ptr().add(base + 8));
          let mut r_arr = [0.0f32; 4];
          let mut g_arr = [0.0f32; 4];
          let mut b_arr = [0.0f32; 4];
          vst1q_f32(r_arr.as_mut_ptr(), raw0);
          vst1q_f32(g_arr.as_mut_ptr(), raw1);
          vst1q_f32(b_arr.as_mut_ptr(), raw2);
          let r_deint = [r_arr[0], r_arr[3], g_arr[2], b_arr[1]];
          let g_deint = [r_arr[1], g_arr[0], g_arr[3], b_arr[2]];
          let b_deint = [r_arr[2], g_arr[1], b_arr[0], b_arr[3]];
          (
            vld1q_f32(r_deint.as_ptr()),
            vld1q_f32(g_deint.as_ptr()),
            vld1q_f32(b_deint.as_ptr()),
          )
        };

        let r_s = vmulq_f32(vminq_f32(vmaxq_f32(r_v, zero), one), scale);
        let g_s = vmulq_f32(vminq_f32(vmaxq_f32(g_v, zero), one), scale);
        let b_s = vmulq_f32(vminq_f32(vmaxq_f32(b_v, zero), one), scale);
        let r_u = vqmovn_u32(vcvtnq_u32_f32(r_s));
        let g_u = vqmovn_u32(vcvtnq_u32_f32(g_s));
        let b_u = vqmovn_u32(vcvtnq_u32_f32(b_s));
        vst1_u16(r_h.as_mut_ptr().add(sub * 4), r_u);
        vst1_u16(g_h.as_mut_ptr().add(sub * 4), g_u);
        vst1_u16(b_h.as_mut_ptr().add(sub * 4), b_u);
      }
      let r = vld1q_u16(r_h.as_ptr());
      let g = vld1q_u16(g_h.as_ptr());
      let b = vld1q_u16(b_h.as_ptr());
      let rgba = uint16x8x4_t(r, g, b, alpha_v);
      vst4q_u16(rgba_out.as_mut_ptr().add(x * 4), rgba);
      x += 8;
    }

    if x < width {
      scalar::rgbf32_to_rgba_u16_row::<BE>(
        &rgb_in[x * 3..width * 3],
        &mut rgba_out[x * 4..width * 4],
        width - x,
      );
    }
  }
}

/// f32 RGB → f32 RGB lossless pass-through.
///
/// When `BE = true` the input values are byte-swapped to host-native
/// before being written (big-endian input → host-native output).
///
/// # Safety
///
/// Same as [`rgbf32_to_rgb_row`] but `rgb_out` is `&mut [f32]` with
/// `len() >= 3 * width` f32 elements.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgbf32_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  unsafe {
    let total = width * 3;
    let mut i = 0usize;
    // Fast path: when the requested encoding (BE) matches the host's native
    // endian, the bytes can be copied verbatim — `vld1q_f32` reads host-native
    // bytes which is exactly what we need to emit. Otherwise we must decode
    // through `load_f32x4::<BE>` (which byte-swaps when BE != host-native) so
    // the stored host-native f32 round-trips back to the same value.
    if BE == HOST_NATIVE_BE {
      while i + 4 <= total {
        let v = vld1q_f32(rgb_in.as_ptr().add(i));
        vst1q_f32(rgb_out.as_mut_ptr().add(i), v);
        i += 4;
      }
      while i < total {
        *rgb_out.get_unchecked_mut(i) = *rgb_in.get_unchecked(i);
        i += 1;
      }
    } else {
      // Encoding doesn't match host: decode each lane to host-native.
      while i + 4 <= total {
        let v = load_f32x4::<BE>(rgb_in.as_ptr().add(i));
        vst1q_f32(rgb_out.as_mut_ptr().add(i), v);
        i += 4;
      }
      while i < total {
        let bits = (*rgb_in.get_unchecked(i)).to_bits();
        let host_bits = if BE {
          u32::from_be(bits)
        } else {
          u32::from_le(bits)
        };
        *rgb_out.get_unchecked_mut(i) = f32::from_bits(host_bits);
        i += 1;
      }
    }
  }
}

// ---- Tier 9 — Rgbf16 NEON entry points ------------------------------------
//
// Strategy: widen a chunk of `f16` lanes to `f32` into a stack buffer, then
// delegate to the existing NEON Rgbf32 downstream kernels. The chunk size is
// 4 pixels (= 12 f16 values) which matches the Rgbf32 loop granularity.
//
// `vcvt_f32_f16` widens 4 x f16 to 4 x f32 in a single FCVT instruction.
//
// For BE: we load the u16 bits via `load_endian_u16x4::<BE>` (loads 4 u16 with
// byte-swap for BE) into a `uint16x4_t`, then reinterpret as `float16x4_t`
// before widening with `vcvt_f32_f16`.  `load_endian_u16x4` reads exactly
// 8 bytes regardless of `BE`, matching the 4 x f16 region the kernel owns
// (a 16-byte load via `load_endian_u16x8` would tail-overread).

use super::endian::load_endian_u16x4;

/// `BE` value that makes the f32 row loaders treat their input as host-native
/// (a no-op byte-swap). Used by f16→f32 widen-then-convert paths whose stack
/// buffer is already host-native after `vcvt_f32_f16`. On a LE target, host-
/// native == LE so `BE = false`; on a BE target, host-native == BE so
/// `BE = true`. Without this routing the downstream `rgbf32_to_*::<false>`
/// would byte-swap an already-decoded host-native f32 buffer on BE hosts.
///
/// Also used by the `rgbf32_to_rgb_f32_row` pass-through fast path: the raw
/// `vld1q_f32`/`vst1q_f32` copy is byte-correct only when the source encoding
/// (`BE`) matches the host's native endian, so the kernel falls through to
/// the endian-aware `load_f32x4::<BE>` slow path otherwise.
///
/// Same gate applies to the `rgbf32_to_rgba_row` / `rgbf32_to_rgba_u16_row`
/// `vld3q_f32` deinterleave fast path: `vld3q_f32` reads host-native bytes,
/// so it's only correct when the on-disk encoding matches host-native.
/// Otherwise the kernel falls through to the endian-aware `load_f32x4::<BE>`
/// path with a manual deinterleave.
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

/// Widen 4 half-precision floats (`f16x4`, i.e. 8 bytes starting at `ptr`)
/// to 4 single-precision floats into `out[0..4]`.
///
/// For `BE = true` the f16 values are stored big-endian (bytes swapped);
/// the byte-swap is applied before the widening conversion. The loader reads
/// exactly 8 bytes regardless of `BE` so the caller's `ptr` only needs 8
/// readable bytes (a 16-byte load via `load_endian_u16x8` would tail-overread
/// the 4 x f16 region the kernel actually owns).
///
/// # Safety
///
/// * NEON must be available.
/// * `ptr` must be valid for 4 x u16 reads (8 bytes).
/// * `out` must be valid for 4 x f32 writes.
#[inline(always)]
unsafe fn widen_f16x4<const BE: bool>(ptr: *const half::f16, out: *mut f32) {
  unsafe {
    // 8-byte load (4 x u16), byte-swapped per-lane when BE = true so the
    // resulting `uint16x4_t` carries host-native f16 bit patterns ready for
    // `vcvt_f32_f16`.
    let u16x4 = load_endian_u16x4::<BE>(ptr as *const u8);
    let f16x4 = vreinterpret_f16_u16(u16x4);
    let f32x4 = vcvt_f32_f16(f16x4);
    vst1q_f32(out, f32x4);
  }
}

/// Widen `n` half-precision floats (at most 4) from `src` to `f32` in `dst`.
/// `n` must be in `[0, 4]` — `n == 0` is a no-op (the caller passes
/// `total_lanes - lane`, which is `0` when `total_lanes` is a multiple of 4
/// and the SIMD loop consumed the whole row).
///
/// For `BE = true` the source f16 bits are decoded from big-endian to
/// host-native before widening; for `BE = false` they are read as host-
/// native (identical to a plain LE load on every shipping target). This
/// matches the SIMD body's `widen_f16x4::<BE>` semantics so partial-pixel
/// tail bytes round-trip identically to the full-vector path.
#[inline(always)]
unsafe fn widen_f16_tail<const BE: bool>(src: &[half::f16], dst: &mut [f32], n: usize) {
  for i in 0..n {
    unsafe {
      let raw = src.get_unchecked(i).to_bits();
      let host_bits = if BE {
        u16::from_be(raw)
      } else {
        u16::from_le(raw)
      };
      *dst.get_unchecked_mut(i) = half::f16::from_bits(host_bits).to_f32();
    }
  }
}

/// f16 RGB → u8 RGB.
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// 1. NEON must be available.
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn rgbf16_to_rgb_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  // Process 4 pixels (12 f16 lanes = 12 f32 lanes) per iteration.
  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 12 <= total_lanes {
    let mut buf = [0.0f32; 12];
    unsafe {
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane), buf.as_mut_ptr());
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane + 4), buf.as_mut_ptr().add(4));
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane + 8), buf.as_mut_ptr().add(8));
      // Buffer is host-native f32 after vcvt_f32_f16; route through the f32
      // kernel with HOST_NATIVE_BE so its loaders perform a no-op swap.
      rgbf32_to_rgb_row::<HOST_NATIVE_BE>(&buf, rgb_out.get_unchecked_mut(lane..lane + 12), 4);
    }
    lane += 12;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf16_to_rgb_row::<BE>(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f16 RGB → u8 RGBA (alpha forced to `0xFF`).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn rgbf16_to_rgba_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 12 <= total_lanes {
    let mut buf = [0.0f32; 12];
    unsafe {
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane), buf.as_mut_ptr());
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane + 4), buf.as_mut_ptr().add(4));
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane + 8), buf.as_mut_ptr().add(8));
      // Buffer is host-native f32; route via HOST_NATIVE_BE (see widen_f16x4).
      rgbf32_to_rgba_row::<HOST_NATIVE_BE>(
        &buf,
        rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16),
        4,
      );
    }
    lane += 12;
    pix += 4;
  }
  if pix < width {
    scalar::rgbf16_to_rgba_row::<BE>(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f16 RGB → u16 RGB.
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [u16]` with
/// `len() >= 3 * width` u16 elements.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn rgbf16_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 12 <= total_lanes {
    let mut buf = [0.0f32; 12];
    unsafe {
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane), buf.as_mut_ptr());
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane + 4), buf.as_mut_ptr().add(4));
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane + 8), buf.as_mut_ptr().add(8));
      // Buffer is host-native f32; route via HOST_NATIVE_BE (see widen_f16x4).
      rgbf32_to_rgb_u16_row::<HOST_NATIVE_BE>(&buf, rgb_out.get_unchecked_mut(lane..lane + 12), 4);
    }
    lane += 12;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf16_to_rgb_u16_row::<BE>(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f16 RGB → u16 RGBA (alpha forced to `0xFFFF`).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_u16_row`] but the output is `&mut [u16]` with
/// `len() >= 4 * width` u16 elements.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn rgbf16_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_u16_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 12 <= total_lanes {
    let mut buf = [0.0f32; 12];
    unsafe {
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane), buf.as_mut_ptr());
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane + 4), buf.as_mut_ptr().add(4));
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane + 8), buf.as_mut_ptr().add(8));
      // Buffer is host-native f32; route via HOST_NATIVE_BE (see widen_f16x4).
      rgbf32_to_rgba_u16_row::<HOST_NATIVE_BE>(
        &buf,
        rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16),
        4,
      );
    }
    lane += 12;
    pix += 4;
  }
  if pix < width {
    scalar::rgbf16_to_rgba_u16_row::<BE>(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f16 RGB → f32 RGB (lossless widen).
///
/// When `BE = true` the input `half::f16` values are big-endian encoded.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [f32]` with
/// `len() >= 3 * width` f32 elements.
#[inline]
#[target_feature(enable = "neon,fp16")]
pub(crate) unsafe fn rgbf16_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 4 <= total_lanes {
    unsafe {
      widen_f16x4::<BE>(rgb_in.as_ptr().add(lane), rgb_out.as_mut_ptr().add(lane));
    }
    lane += 4;
  }
  // Scalar tail for the last 0-3 lanes (partial pixel at most).
  unsafe {
    widen_f16_tail::<BE>(
      rgb_in.get_unchecked(lane..),
      rgb_out.get_unchecked_mut(lane..),
      total_lanes - lane,
    );
  }
}

/// f16 RGB → f16 RGB lossless pass-through.
///
/// When `BE = true` the input values are byte-swapped to host-native order
/// on output.
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [half::f16]` with
/// `len() >= 3 * width` f16 elements.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rgbf16_to_rgb_f16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f16_out row too short");

  // Bit-exact copy / byte-swap: delegate to scalar.
  scalar::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
}
