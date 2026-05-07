use super::*;
use crate::row::arch::x86_avx512::endian::*;

// Helper: extract __m512i to Vec<u16> (32 lanes).
#[cfg(target_arch = "x86_64")]
unsafe fn m512i_to_u16x32(v: core::arch::x86_64::__m512i) -> std::vec::Vec<u16> {
  let mut out = std::vec![0u16; 32];
  unsafe { core::arch::x86_64::_mm512_storeu_si512(out.as_mut_ptr().cast(), v) };
  out
}

// Helper: extract __m512i to Vec<u32> (16 lanes).
#[cfg(target_arch = "x86_64")]
unsafe fn m512i_to_u32x16(v: core::arch::x86_64::__m512i) -> std::vec::Vec<u32> {
  let mut out = std::vec![0u32; 16];
  unsafe { core::arch::x86_64::_mm512_storeu_si512(out.as_mut_ptr().cast(), v) };
  out
}

// ---- LE loader on LE host (no-op) ------------------------------------------

#[test]
#[cfg(target_endian = "little")]
fn avx512_load_le_u16x32_noop_on_le_host() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  // Build 64 bytes: pairs [lo, hi] for values 0x0102..0x4142.
  let mut input = [0u8; 64];
  for i in 0usize..32 {
    // LE encoding: low byte first
    input[i * 2] = ((i + 1) as u8).wrapping_add(1); // low byte
    input[i * 2 + 1] = (i + 1) as u8; // high byte
  }
  let v = unsafe { load_le_u16x32(input.as_ptr()) };
  let got = unsafe { m512i_to_u16x32(v) };
  let expected: std::vec::Vec<u16> = (0u16..32)
    .map(|i| (((i + 1) as u16) << 8) | ((i + 2) as u16))
    .collect();
  assert_eq!(
    got, expected,
    "AVX-512 load_le_u16x32 must not swap on LE host"
  );
}

#[test]
#[cfg(target_endian = "big")]
fn avx512_load_le_u16x32_swaps_on_be_host() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let mut input = [0u8; 64];
  for i in 0usize..32 {
    input[i * 2] = ((i + 1) as u8).wrapping_add(1);
    input[i * 2 + 1] = (i + 1) as u8;
  }
  let v = unsafe { load_le_u16x32(input.as_ptr()) };
  let got = unsafe { m512i_to_u16x32(v) };
  let expected: std::vec::Vec<u16> = (0u16..32)
    .map(|i| (((i + 1) as u16) << 8) | ((i + 2) as u16))
    .collect();
  assert_eq!(got, expected, "AVX-512 load_le_u16x32 must swap on BE host");
}

// ---- BE loader on LE host (swap) -------------------------------------------

#[test]
#[cfg(target_endian = "little")]
fn avx512_load_be_u16x32_swaps_on_le_host() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  // BE encoding: high byte first.
  let mut input = [0u8; 64];
  for i in 0usize..32 {
    input[i * 2] = (i + 1) as u8; // high byte
    input[i * 2 + 1] = ((i + 1) as u8).wrapping_add(1); // low byte
  }
  let v = unsafe { load_be_u16x32(input.as_ptr()) };
  let got = unsafe { m512i_to_u16x32(v) };
  let expected: std::vec::Vec<u16> = (0u16..32)
    .map(|i| (((i + 1) as u16) << 8) | ((i + 2) as u16))
    .collect();
  assert_eq!(got, expected, "AVX-512 load_be_u16x32 must swap on LE host");
}

#[test]
#[cfg(target_endian = "big")]
fn avx512_load_be_u16x32_noop_on_be_host() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let mut input = [0u8; 64];
  for i in 0usize..32 {
    input[i * 2] = (i + 1) as u8;
    input[i * 2 + 1] = ((i + 1) as u8).wrapping_add(1);
  }
  let v = unsafe { load_be_u16x32(input.as_ptr()) };
  let got = unsafe { m512i_to_u16x32(v) };
  let expected: std::vec::Vec<u16> = (0u16..32)
    .map(|i| (((i + 1) as u16) << 8) | ((i + 2) as u16))
    .collect();
  assert_eq!(
    got, expected,
    "AVX-512 load_be_u16x32 must not swap on BE host"
  );
}

// ---- u32x16 LE loader on LE host (no-op) -----------------------------------

#[test]
#[cfg(target_endian = "little")]
fn avx512_load_le_u32x16_noop_on_le_host() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let input: [u8; 64] = [
    0x04, 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05, 0x0c, 0x0b, 0x0a, 0x09, 0x10, 0x0f, 0x0e, 0x0d,
    0x14, 0x13, 0x12, 0x11, 0x18, 0x17, 0x16, 0x15, 0x1c, 0x1b, 0x1a, 0x19, 0x20, 0x1f, 0x1e, 0x1d,
    0x24, 0x23, 0x22, 0x21, 0x28, 0x27, 0x26, 0x25, 0x2c, 0x2b, 0x2a, 0x29, 0x30, 0x2f, 0x2e, 0x2d,
    0x34, 0x33, 0x32, 0x31, 0x38, 0x37, 0x36, 0x35, 0x3c, 0x3b, 0x3a, 0x39, 0x40, 0x3f, 0x3e, 0x3d,
  ];
  let v = unsafe { load_le_u32x16(input.as_ptr()) };
  let got = unsafe { m512i_to_u32x16(v) };
  assert_eq!(
    got,
    [
      0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10, 0x11121314, 0x15161718, 0x191a1b1c,
      0x1d1e1f20, 0x21222324, 0x25262728, 0x292a2b2c, 0x2d2e2f30, 0x31323334, 0x35363738,
      0x393a3b3c, 0x3d3e3f40,
    ],
    "AVX-512 load_le_u32x16 must not swap on LE host"
  );
}

// ---- u32x16 BE loader on LE host (swap) ------------------------------------

#[test]
#[cfg(target_endian = "little")]
fn avx512_load_be_u32x16_swaps_on_le_host() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let input: [u8; 64] = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30,
    0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40,
  ];
  let v = unsafe { load_be_u32x16(input.as_ptr()) };
  let got = unsafe { m512i_to_u32x16(v) };
  assert_eq!(
    got,
    [
      0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10, 0x11121314, 0x15161718, 0x191a1b1c,
      0x1d1e1f20, 0x21222324, 0x25262728, 0x292a2b2c, 0x2d2e2f30, 0x31323334, 0x35363738,
      0x393a3b3c, 0x3d3e3f40,
    ],
    "AVX-512 load_be_u32x16 must swap on LE host"
  );
}

// ---- Generic dispatcher consistency ----------------------------------------

#[test]
#[cfg(target_endian = "little")]
fn avx512_load_endian_u16x32_le_dispatcher() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let mut input = [0u8; 64];
  for i in 0usize..32 {
    input[i * 2] = ((i + 1) as u8).wrapping_add(1);
    input[i * 2 + 1] = (i + 1) as u8;
  }
  let direct = unsafe { load_le_u16x32(input.as_ptr()) };
  let via_dispatch = unsafe { load_endian_u16x32::<false>(input.as_ptr()) };
  let d = unsafe { m512i_to_u16x32(direct) };
  let g = unsafe { m512i_to_u16x32(via_dispatch) };
  assert_eq!(
    d, g,
    "load_endian_u16x32::<false> must match load_le_u16x32"
  );
}

#[test]
#[cfg(target_endian = "little")]
fn avx512_load_endian_u16x32_be_dispatcher() {
  if !std::arch::is_x86_feature_detected!("avx512bw") {
    return;
  }
  let mut input = [0u8; 64];
  for i in 0usize..32 {
    input[i * 2] = (i + 1) as u8;
    input[i * 2 + 1] = ((i + 1) as u8).wrapping_add(1);
  }
  let direct = unsafe { load_be_u16x32(input.as_ptr()) };
  let via_dispatch = unsafe { load_endian_u16x32::<true>(input.as_ptr()) };
  let d = unsafe { m512i_to_u16x32(direct) };
  let g = unsafe { m512i_to_u16x32(via_dispatch) };
  assert_eq!(d, g, "load_endian_u16x32::<true> must match load_be_u16x32");
}
