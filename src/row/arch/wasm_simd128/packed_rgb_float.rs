//! WASM simd128 kernels for the Tier 9 packed-float-RGB (`Rgbf32`)
//! source. simd128 is a 128-bit ISA so each register holds 4 `f32`
//! lanes — same shape as the SSE4.1 backend.
//!
//! `i32x4_trunc_sat_f32x4` performs a saturating truncate-toward-zero
//! cast; round-to-nearest-even isn't a primitive on simd128 so we
//! preface the cast with `f32x4_nearest` (round to nearest even, the
//! IEEE 754 default) — matching the scalar path's
//! `round_ties_even_nonneg` helper.

use core::arch::wasm32::*;

use super::{endian::load_endian_u32x4, scalar};

/// `BE` value that makes the f32 row loaders treat their input as host-native
/// (a no-op byte-swap). Used by f16→f32 widen-then-convert paths whose stack
/// buffer is already host-native after `half::f16::to_f32()`. On a LE target,
/// host-native == LE so `BE = false`; on a BE target, host-native == BE so
/// `BE = true`. Without this routing the downstream `rgbf32_to_*::<false>`
/// would byte-swap an already-decoded host-native f32 buffer on BE hosts.
/// (`wasm32-*` is LE today, but keeping the routing endian-agnostic future-
/// proofs against any BE wasm target.)
const HOST_NATIVE_BE: bool = cfg!(target_endian = "big");

// ---- helpers ------------------------------------------------------------------

#[inline(always)]
fn clamp_scale_to_i32(v: v128, zero: v128, one: v128, scale: v128) -> v128 {
  let clamped = f32x4_min(f32x4_max(v, zero), one);
  let scaled = f32x4_mul(clamped, scale);
  // Round to nearest even (IEEE default), then saturating-truncate to i32.
  let rounded = f32x4_nearest(scaled);
  i32x4_trunc_sat_f32x4(rounded)
}

/// Load 4 f32 values from `ptr`, byte-swapping each 32-bit element when
/// `BE = true`.  The returned `v128` holds f32 bit patterns in host-native
/// order so downstream float arithmetic is correct.
///
/// # Safety
///
/// `ptr` must point to at least 16 readable bytes.  simd128 must be
/// available (compile-time `target_feature`).
#[inline(always)]
unsafe fn load_f32x4<const BE: bool>(ptr: *const f32) -> v128 {
  // load_endian_u32x4 byte-swaps each 32-bit lane when BE=true, giving us
  // host-native f32 bit patterns.
  unsafe { load_endian_u32x4::<BE>(ptr as *const u8) }
}

// ---- Tier 9 — Rgbf32 wasm-simd128 kernels ------------------------------------

/// f32 RGB → u8 RGB.
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgb_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(255.0);

  let total_lanes = width * 3;
  let mut lane = 0usize;
  // 4 pixels = 12 lanes per iter.
  while lane + 12 <= total_lanes {
    unsafe {
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

      let i0 = clamp_scale_to_i32(v0, zero, one, scale);
      let i1 = clamp_scale_to_i32(v1, zero, one, scale);
      let i2 = clamp_scale_to_i32(v2, zero, one, scale);

      // i32x4 → i16x8 (saturating signed narrow); each yields 4 i16 lanes.
      let h01 = i16x8_narrow_i32x4(i0, i1);
      let h22 = i16x8_narrow_i32x4(i2, i2);

      // i16x8 → u8x16 (saturating unsigned narrow). 12 valid bytes per iter.
      let bytes = u8x16_narrow_i16x8(h01, h22);

      let mut tmp = [0u8; 16];
      v128_store(tmp.as_mut_ptr() as *mut v128, bytes);
      rgb_out
        .get_unchecked_mut(lane..lane + 12)
        .copy_from_slice(&tmp[..12]);
    }
    lane += 12;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf32_to_rgb_row::<BE>(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f32 RGB → u8 RGBA (alpha forced to `0xFF`).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgba_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_out row too short");

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(255.0);

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 12 <= total_lanes {
    unsafe {
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

      let i0 = clamp_scale_to_i32(v0, zero, one, scale);
      let i1 = clamp_scale_to_i32(v1, zero, one, scale);
      let i2 = clamp_scale_to_i32(v2, zero, one, scale);

      let h01 = i16x8_narrow_i32x4(i0, i1);
      let h22 = i16x8_narrow_i32x4(i2, i2);
      let bytes = u8x16_narrow_i16x8(h01, h22);

      let mut tmp = [0u8; 16];
      v128_store(tmp.as_mut_ptr() as *mut v128, bytes);
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16);
      for p in 0..4 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFF;
      }
    }
    lane += 12;
    pix += 4;
  }
  if pix < width {
    scalar::rgbf32_to_rgba_row::<BE>(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f32 RGB → u16 RGB.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgb_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_u16_out row too short");

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(65535.0);

  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 12 <= total_lanes {
    unsafe {
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

      let i0 = clamp_scale_to_i32(v0, zero, one, scale);
      let i1 = clamp_scale_to_i32(v1, zero, one, scale);
      let i2 = clamp_scale_to_i32(v2, zero, one, scale);

      // i32x4 → u16x8 saturating narrow (`u16x8_narrow_i32x4` saturates
      // negatives to 0 and clamps at 65535). After clamp+scale our
      // values are already in [0, 65535] so saturation is a no-op
      // semantically but still required by the type system.
      let u01 = u16x8_narrow_i32x4(i0, i1);
      let u22 = u16x8_narrow_i32x4(i2, i2);

      let mut tmp = [0u16; 16];
      v128_store(tmp.as_mut_ptr() as *mut v128, u01);
      v128_store(tmp.as_mut_ptr().add(8) as *mut v128, u22);
      rgb_out
        .get_unchecked_mut(lane..lane + 12)
        .copy_from_slice(&tmp[..12]);
    }
    lane += 12;
  }
  let pix_done = lane / 3;
  if pix_done < width {
    scalar::rgbf32_to_rgb_u16_row::<BE>(
      &rgb_in[pix_done * 3..width * 3],
      &mut rgb_out[pix_done * 3..width * 3],
      width - pix_done,
    );
  }
}

/// f32 RGB → u16 RGBA (alpha forced to `0xFFFF`).
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgba_u16_row<const BE: bool>(
  rgb_in: &[f32],
  rgba_out: &mut [u16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgba_out.len() >= width * 4, "rgba_u16_out row too short");

  let zero = f32x4_splat(0.0);
  let one = f32x4_splat(1.0);
  let scale = f32x4_splat(65535.0);

  let total_lanes = width * 3;
  let mut lane = 0usize;
  let mut pix = 0usize;
  while lane + 12 <= total_lanes {
    unsafe {
      let v0 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane));
      let v1 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 4));
      let v2 = load_f32x4::<BE>(rgb_in.as_ptr().add(lane + 8));

      let i0 = clamp_scale_to_i32(v0, zero, one, scale);
      let i1 = clamp_scale_to_i32(v1, zero, one, scale);
      let i2 = clamp_scale_to_i32(v2, zero, one, scale);

      let u01 = u16x8_narrow_i32x4(i0, i1);
      let u22 = u16x8_narrow_i32x4(i2, i2);

      let mut tmp = [0u16; 16];
      v128_store(tmp.as_mut_ptr() as *mut v128, u01);
      v128_store(tmp.as_mut_ptr().add(8) as *mut v128, u22);
      let dst = rgba_out.get_unchecked_mut(pix * 4..pix * 4 + 16);
      for p in 0..4 {
        dst[p * 4] = tmp[p * 3];
        dst[p * 4 + 1] = tmp[p * 3 + 1];
        dst[p * 4 + 2] = tmp[p * 3 + 2];
        dst[p * 4 + 3] = 0xFFFF;
      }
    }
    lane += 12;
    pix += 4;
  }
  if pix < width {
    scalar::rgbf32_to_rgba_u16_row::<BE>(
      &rgb_in[pix * 3..width * 3],
      &mut rgba_out[pix * 4..width * 4],
      width - pix,
    );
  }
}

/// f32 RGB → f32 RGB lossless pass-through / byte-swap.
///
/// - `BE = false`: fast `v128_load` → `v128_store` copy (no math).
/// - `BE = true`:  load each element as u32, byte-swap, store as f32.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf32_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[f32],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf32 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");

  if !BE {
    let total = width * 3;
    let mut i = 0usize;
    while i + 4 <= total {
      unsafe {
        let v = v128_load(rgb_in.as_ptr().add(i) as *const v128);
        v128_store(rgb_out.as_mut_ptr().add(i) as *mut v128, v);
      }
      i += 4;
    }
    while i < total {
      unsafe {
        *rgb_out.get_unchecked_mut(i) = *rgb_in.get_unchecked(i);
      }
      i += 1;
    }
  } else {
    // BE: byte-swap each f32 element via u32 lane reinterpretation.
    let total = width * 3;
    let mut i = 0usize;
    while i + 4 <= total {
      unsafe {
        // load_endian_u32x4::<true> byte-swaps each 32-bit lane.
        let swapped = load_f32x4::<BE>(rgb_in.as_ptr().add(i));
        v128_store(rgb_out.as_mut_ptr().add(i) as *mut v128, swapped);
      }
      i += 4;
    }
    while i < total {
      unsafe {
        let bits = rgb_in.get_unchecked(i).to_bits();
        *rgb_out.get_unchecked_mut(i) = f32::from_bits(u32::from_be(bits));
      }
      i += 1;
    }
  }
}

// ---- Tier 9 — Rgbf16 wasm-simd128 entry points ----------------------------
//
// wasm-simd128 has no native f16 widening instruction. Strategy: widen each
// f16 element to f32 via `half::f16::to_f32()` (scalar) into a stack-allocated
// `[f32; CHUNK_PIXELS * 3]` buffer, then call the existing wasm-simd128
// Rgbf32 downstream kernels for the f32→u8/u16/f32 work.
//
// For BE inputs the byte-swap is applied before widening so the widened f32
// buffer is already host-native; downstream f32 kernels are called with
// `HOST_NATIVE_BE` so their loaders perform a no-op byte-swap (correct on
// both LE and BE hosts).
//
// CHUNK_PIXELS = 4 (= 12 f32 lanes), matching the simd128 Rgbf32 loop stride.

/// f16 RGB → u8 RGB (wasm-simd128).
///
/// # Safety
///
/// 1. simd128 must be available (compile-time `target_feature`).
/// 2. `rgb_in.len() >= 3 * width`; `rgb_out.len() >= 3 * width`.
/// 3. `rgb_in` / `rgb_out` must not alias.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf16_to_rgb_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [u8],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_out row too short");

  // Process 4 pixels (12 f16 lanes) per iteration.
  let total_lanes = width * 3;
  let mut lane = 0usize;
  while lane + 12 <= total_lanes {
    let mut buf = [0.0f32; 12];
    for k in 0..12 {
      let f = unsafe { rgb_in.get_unchecked(lane + k) };
      let raw = f.to_bits();
      let bits = if BE {
        u16::from_be(raw)
      } else {
        u16::from_le(raw)
      };
      buf[k] = half::f16::from_bits(bits).to_f32();
    }
    unsafe {
      // Buffer is now host-native f32; route via HOST_NATIVE_BE so the f32
      // loaders perform a no-op byte-swap on both LE and BE hosts.
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

/// f16 RGB → u8 RGBA (alpha `0xFF`) (wasm-simd128).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    for k in 0..12 {
      let f = unsafe { rgb_in.get_unchecked(lane + k) };
      let raw = f.to_bits();
      let bits = if BE {
        u16::from_be(raw)
      } else {
        u16::from_le(raw)
      };
      buf[k] = half::f16::from_bits(bits).to_f32();
    }
    unsafe {
      // Buffer is host-native f32; route via HOST_NATIVE_BE.
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

/// f16 RGB → u16 RGB (wasm-simd128).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [u16]` with
/// `len() >= 3 * width` u16 elements.
#[inline]
#[target_feature(enable = "simd128")]
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
    for k in 0..12 {
      let f = unsafe { rgb_in.get_unchecked(lane + k) };
      let raw = f.to_bits();
      let bits = if BE {
        u16::from_be(raw)
      } else {
        u16::from_le(raw)
      };
      buf[k] = half::f16::from_bits(bits).to_f32();
    }
    unsafe {
      // Buffer is host-native f32; route via HOST_NATIVE_BE.
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

/// f16 RGB → u16 RGBA (alpha `0xFFFF`) (wasm-simd128).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_u16_row`] but `rgba_out.len() >= 4 * width`.
#[inline]
#[target_feature(enable = "simd128")]
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
    for k in 0..12 {
      let f = unsafe { rgb_in.get_unchecked(lane + k) };
      let raw = f.to_bits();
      let bits = if BE {
        u16::from_be(raw)
      } else {
        u16::from_le(raw)
      };
      buf[k] = half::f16::from_bits(bits).to_f32();
    }
    unsafe {
      // Buffer is host-native f32; route via HOST_NATIVE_BE.
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

/// f16 RGB → f32 RGB (lossless widen) (wasm-simd128).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [f32]` with
/// `len() >= 3 * width` f32 elements.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf16_to_rgb_f32_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [f32],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f32_out row too short");
  // Pure scalar widen; the downstream f32 copy is trivial via copy_from_slice
  // and we avoid an extra pass through the data.
  let total_lanes = width * 3;
  for i in 0..total_lanes {
    unsafe {
      let f = rgb_in.get_unchecked(i);
      let raw = f.to_bits();
      let bits = if BE {
        u16::from_be(raw)
      } else {
        u16::from_le(raw)
      };
      *rgb_out.get_unchecked_mut(i) = half::f16::from_bits(bits).to_f32();
    }
  }
}

/// f16 RGB → f16 RGB lossless pass-through / byte-swap (wasm-simd128).
///
/// # Safety
///
/// Same as [`rgbf16_to_rgb_row`] but `rgb_out` is `&mut [half::f16]` with
/// `len() >= 3 * width` f16 elements.
#[inline]
#[target_feature(enable = "simd128")]
pub(crate) unsafe fn rgbf16_to_rgb_f16_row<const BE: bool>(
  rgb_in: &[half::f16],
  rgb_out: &mut [half::f16],
  width: usize,
) {
  debug_assert!(rgb_in.len() >= width * 3, "rgbf16 row too short");
  debug_assert!(rgb_out.len() >= width * 3, "rgb_f16_out row too short");
  scalar::rgbf16_to_rgb_f16_row::<BE>(rgb_in, rgb_out, width);
}
