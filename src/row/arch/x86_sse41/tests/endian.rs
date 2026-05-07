use crate::row::arch::x86_sse41::endian::*;

// Helper: extract __m128i to a stack array of 8 u16 lanes.
#[cfg(target_arch = "x86_64")]
unsafe fn m128i_to_u16x8(v: core::arch::x86_64::__m128i) -> [u16; 8] {
  let mut out = [0u16; 8];
  unsafe { core::arch::x86_64::_mm_storeu_si128(out.as_mut_ptr().cast(), v) };
  out
}

// Helper: extract __m128i to a stack array of 4 u32 lanes.
#[cfg(target_arch = "x86_64")]
unsafe fn m128i_to_u32x4(v: core::arch::x86_64::__m128i) -> [u32; 4] {
  let mut out = [0u32; 4];
  unsafe { core::arch::x86_64::_mm_storeu_si128(out.as_mut_ptr().cast(), v) };
  out
}

// ---- LE loader on LE host (no-op) ------------------------------------------

/// On a LE host, `load_le_u16x8` must NOT swap bytes.
#[test]
#[cfg(target_endian = "little")]
fn sse41_load_le_u16x8_noop_on_le_host() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let input: [u8; 16] = [
    0x02, 0x01, // u16[0] = 0x0102
    0x04, 0x03, // u16[1] = 0x0304
    0x06, 0x05, // u16[2] = 0x0506
    0x08, 0x07, // u16[3] = 0x0708
    0x0a, 0x09, // u16[4] = 0x090a
    0x0c, 0x0b, // u16[5] = 0x0b0c
    0x0e, 0x0d, // u16[6] = 0x0d0e
    0x10, 0x0f, // u16[7] = 0x0f10
  ];
  let v = unsafe { load_le_u16x8(input.as_ptr()) };
  let got = unsafe { m128i_to_u16x8(v) };
  assert_eq!(
    got,
    [
      0x0102, 0x0304, 0x0506, 0x0708, 0x090a, 0x0b0c, 0x0d0e, 0x0f10
    ],
    "SSE4.1 load_le_u16x8 must not swap on LE host"
  );
}

/// On a BE host, `load_le_u16x8` MUST swap bytes.
#[test]
#[cfg(target_endian = "big")]
fn sse41_load_le_u16x8_swaps_on_be_host() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let input: [u8; 16] = [
    0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07, 0x0a, 0x09, 0x0c, 0x0b, 0x0e, 0x0d, 0x10, 0x0f,
  ];
  let v = unsafe { load_le_u16x8(input.as_ptr()) };
  let got = unsafe { m128i_to_u16x8(v) };
  assert_eq!(
    got,
    [
      0x0102, 0x0304, 0x0506, 0x0708, 0x090a, 0x0b0c, 0x0d0e, 0x0f10
    ],
    "SSE4.1 load_le_u16x8 must swap on BE host"
  );
}

// ---- BE loader on LE host (swap) -------------------------------------------

/// On a LE host, `load_be_u16x8` MUST swap bytes.
#[test]
#[cfg(target_endian = "little")]
fn sse41_load_be_u16x8_swaps_on_le_host() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let input: [u8; 16] = [
    0x01, 0x02, // u16[0] = 0x0102 BE
    0x03, 0x04, // u16[1] = 0x0304
    0x05, 0x06, // u16[2] = 0x0506
    0x07, 0x08, // u16[3] = 0x0708
    0x09, 0x0a, // u16[4] = 0x090a
    0x0b, 0x0c, // u16[5] = 0x0b0c
    0x0d, 0x0e, // u16[6] = 0x0d0e
    0x0f, 0x10, // u16[7] = 0x0f10
  ];
  let v = unsafe { load_be_u16x8(input.as_ptr()) };
  let got = unsafe { m128i_to_u16x8(v) };
  assert_eq!(
    got,
    [
      0x0102, 0x0304, 0x0506, 0x0708, 0x090a, 0x0b0c, 0x0d0e, 0x0f10
    ],
    "SSE4.1 load_be_u16x8 must swap on LE host"
  );
}

/// On a BE host, `load_be_u16x8` must NOT swap.
#[test]
#[cfg(target_endian = "big")]
fn sse41_load_be_u16x8_noop_on_be_host() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let input: [u8; 16] = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
  ];
  let v = unsafe { load_be_u16x8(input.as_ptr()) };
  let got = unsafe { m128i_to_u16x8(v) };
  assert_eq!(
    got,
    [
      0x0102, 0x0304, 0x0506, 0x0708, 0x090a, 0x0b0c, 0x0d0e, 0x0f10
    ],
    "SSE4.1 load_be_u16x8 must not swap on BE host"
  );
}

// ---- u32x4 LE loader on LE host (no-op) ------------------------------------

#[test]
#[cfg(target_endian = "little")]
fn sse41_load_le_u32x4_noop_on_le_host() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let input: [u8; 16] = [
    0x04, 0x03, 0x02, 0x01, // u32[0] = 0x01020304 LE
    0x08, 0x07, 0x06, 0x05, // u32[1] = 0x05060708
    0x0c, 0x0b, 0x0a, 0x09, // u32[2] = 0x090a0b0c
    0x10, 0x0f, 0x0e, 0x0d, // u32[3] = 0x0d0e0f10
  ];
  let v = unsafe { load_le_u32x4(input.as_ptr()) };
  let got = unsafe { m128i_to_u32x4(v) };
  assert_eq!(
    got,
    [0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10],
    "SSE4.1 load_le_u32x4 must not swap on LE host"
  );
}

// ---- u32x4 BE loader on LE host (swap) -------------------------------------

#[test]
#[cfg(target_endian = "little")]
fn sse41_load_be_u32x4_swaps_on_le_host() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let input: [u8; 16] = [
    0x01, 0x02, 0x03, 0x04, // u32[0] = 0x01020304 BE
    0x05, 0x06, 0x07, 0x08, // u32[1] = 0x05060708
    0x09, 0x0a, 0x0b, 0x0c, // u32[2] = 0x090a0b0c
    0x0d, 0x0e, 0x0f, 0x10, // u32[3] = 0x0d0e0f10
  ];
  let v = unsafe { load_be_u32x4(input.as_ptr()) };
  let got = unsafe { m128i_to_u32x4(v) };
  assert_eq!(
    got,
    [0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10],
    "SSE4.1 load_be_u32x4 must swap on LE host"
  );
}

// ---- Generic dispatcher consistency ----------------------------------------

#[test]
#[cfg(target_endian = "little")]
fn sse41_load_endian_u16x8_le_dispatcher() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let input: [u8; 16] = [
    0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07, 0x0a, 0x09, 0x0c, 0x0b, 0x0e, 0x0d, 0x10, 0x0f,
  ];
  let direct = unsafe { load_le_u16x8(input.as_ptr()) };
  let via_dispatch = unsafe { load_endian_u16x8::<false>(input.as_ptr()) };
  let d = unsafe { m128i_to_u16x8(direct) };
  let g = unsafe { m128i_to_u16x8(via_dispatch) };
  assert_eq!(d, g, "load_endian_u16x8::<false> must match load_le_u16x8");
}

#[test]
#[cfg(target_endian = "little")]
fn sse41_load_endian_u16x8_be_dispatcher() {
  if !std::arch::is_x86_feature_detected!("sse4.1") {
    return;
  }
  let input: [u8; 16] = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
  ];
  let direct = unsafe { load_be_u16x8(input.as_ptr()) };
  let via_dispatch = unsafe { load_endian_u16x8::<true>(input.as_ptr()) };
  let d = unsafe { m128i_to_u16x8(direct) };
  let g = unsafe { m128i_to_u16x8(via_dispatch) };
  assert_eq!(d, g, "load_endian_u16x8::<true> must match load_be_u16x8");
}
