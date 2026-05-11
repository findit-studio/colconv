//! Packed YUV 4:2:2 high-bit-depth source family `Y2xx` — common
//! frame template + format aliases.
//!
//! Each row contains `width × 4` u8 bytes laid out as YUYV-shaped
//! u16 quadruples (`Y₀, U, Y₁, V`). Active bits are MSB-aligned;
//! low `(16 - BITS)` bits are zero.
//!
//! | Format | BITS | FFmpeg pix_fmt              | Active bit width | Low bits |
//! |--------|------|-----------------------------|------------------|----------|
//! | Y210   | 10   | `AV_PIX_FMT_Y210{LE,BE}`    | bits[15:6]       | bits[5:0] = 0 |
//! | Y212   | 12   | `AV_PIX_FMT_Y212{LE,BE}`    | bits[15:4]       | bits[3:0] = 0 |
//! | Y216   | 16   | `AV_PIX_FMT_Y216{LE,BE}`    | bits[15:0]       | n/a (full range) |
//!
//! Width must be even (4:2:2 chroma subsampling).
//!
//! # Endian contract — `<const BE: bool = false>`
//!
//! Each frame type carries a `<const BE: bool>` parameter that defaults to
//! `false` (LE-encoded bytes). The parameter encodes the **byte order of the
//! plane bytes**, matching the FFmpeg `*LE` / `*BE` pixel-format suffix:
//!
//! - `BE = false` (default; e.g. `Y210LeFrame`) — plane bytes are LE-encoded,
//!   matching `AV_PIX_FMT_Y210LE`. On a little-endian host (every CI runner
//!   today) LE bytes _are_ host-native, so `&[u16]` is also a host-native
//!   `u16` slice; on a big-endian host the bytes have to be byte-swapped
//!   back to host-native before arithmetic.
//! - `BE = true` (e.g. `Y210BeFrame`) — plane bytes are BE-encoded, matching
//!   `AV_PIX_FMT_Y210BE`. On a little-endian host the bytes are byte-swapped
//!   before arithmetic; on a big-endian host they are host-native.
//!
//! Downstream row kernels handle the byte-swap (or no-op) under the hood —
//! callers do **not** pre-swap. The `BE` parameter on `Frame` propagates
//! through the walker (`y210_to::<BE>(...)`) into the sinker dispatch
//! (`MixedSinker<Y210<BE>>`), which forwards `BE` as the runtime
//! `big_endian` argument of the `*_row_endian` kernels.
//!
//! Stride is in **u16 elements** (not bytes). Callers holding a raw
//! FFmpeg byte buffer should cast via `bytemuck::cast_slice` (which
//! checks alignment at runtime) and divide `linesize[0]` by 2 before
//! constructing. Direct pointer casts to `&[u16]` are undefined behaviour
//! if the byte buffer is not 2-byte aligned.
//!
//! Used by Ship 11b (Y210), Ship 11c (Y212 — wiring-only), and
//! Ship 11d (Y216 — separate kernel family with i64 chroma path).

use derive_more::{IsVariant, TryUnwrap, Unwrap};
use thiserror::Error;

/// Validated wrapper around a packed YUV 4:2:2 high-bit-depth plane
/// for the `Y210` / `Y212` / `Y216` family
/// (`AV_PIX_FMT_Y210{LE,BE}` / `Y212{LE,BE}` / `Y216{LE,BE}`).
///
/// `BITS` selects the active sample width: 10, 12, or 16. The
/// `<const BE: bool>` parameter selects the plane byte order: `false`
/// (default) → LE-encoded bytes (`AV_PIX_FMT_Y2xxLE`), `true` →
/// BE-encoded bytes (`AV_PIX_FMT_Y2xxBE`). Construct via
/// [`Self::try_new`] (fallible) or [`Self::new`] (panics on invalid
/// input). For `BITS ∈ {10, 12}` the optional
/// [`Self::try_new_checked`] additionally verifies that every
/// sample's low `(16 - BITS)` bits are zero (matches the
/// `P010::try_new_checked` pattern).
///
/// The `&[u16]` plane is the **LE- or BE-encoded byte layout**
/// reinterpreted as `u16`, matching the FFmpeg `*LE`/`*BE`
/// pixel-format convention. Downstream row kernels handle the
/// byte-swap (or no-op) under the hood — callers do **not** pre-swap.
/// Callers holding raw FFmpeg byte buffers should cast via
/// `bytemuck::cast_slice` and divide `linesize[0]` by 2 before
/// constructing.
#[derive(Debug, Clone, Copy)]
pub struct Y2xxFrame<'a, const BITS: u32, const BE: bool = false> {
  packed: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

/// Y210 alias — 10-bit MSB-aligned packed YUV 4:2:2, **LE-encoded** plane
/// bytes (`AV_PIX_FMT_Y210LE`). Concrete alias resolving to
/// `Y2xxFrame<'a, 10, false>` so existing call sites
/// (`Y210Frame::try_new(...)`) compile unchanged. For BE plane bytes,
/// use [`Y210BeFrame`].
pub type Y210Frame<'a> = Y2xxFrame<'a, 10, false>;

/// LE-encoded `Y210Frame` (`AV_PIX_FMT_Y210LE`). Equivalent to
/// [`Y210Frame`]; provided as an explicit alias for callers who want to
/// document the endianness at the type level.
pub type Y210LeFrame<'a> = Y2xxFrame<'a, 10, false>;

/// BE-encoded `Y210Frame` (`AV_PIX_FMT_Y210BE`). Plane bytes are
/// big-endian-encoded `u16` samples; downstream row kernels byte-swap under
/// the hood. Drives the `MixedSinker<Y210<true>>` monomorphization.
pub type Y210BeFrame<'a> = Y2xxFrame<'a, 10, true>;

/// Y212 alias — 12-bit MSB-aligned packed YUV 4:2:2, **LE-encoded** plane
/// bytes (`AV_PIX_FMT_Y212LE`). For BE plane bytes, use [`Y212BeFrame`].
pub type Y212Frame<'a> = Y2xxFrame<'a, 12, false>;

/// LE-encoded `Y212Frame` (`AV_PIX_FMT_Y212LE`).
pub type Y212LeFrame<'a> = Y2xxFrame<'a, 12, false>;

/// BE-encoded `Y212Frame` (`AV_PIX_FMT_Y212BE`).
pub type Y212BeFrame<'a> = Y2xxFrame<'a, 12, true>;

/// Y216 alias — 16-bit packed YUV 4:2:2 (full-range u16 samples,
/// no MSB-alignment shift), **LE-encoded** plane bytes
/// (`AV_PIX_FMT_Y216LE`). For Y216, [`Self::try_new_checked`] is
/// equivalent to [`Self::try_new`] (no low bits to verify). For BE plane
/// bytes, use [`Y216BeFrame`].
pub type Y216Frame<'a> = Y2xxFrame<'a, 16, false>;

/// LE-encoded `Y216Frame` (`AV_PIX_FMT_Y216LE`).
pub type Y216LeFrame<'a> = Y2xxFrame<'a, 16, false>;

/// BE-encoded `Y216Frame` (`AV_PIX_FMT_Y216BE`).
pub type Y216BeFrame<'a> = Y2xxFrame<'a, 16, true>;

/// Errors returned by [`Y2xxFrame::try_new`] and
/// [`Y2xxFrame::try_new_checked`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, TryUnwrap, Unwrap, Error)]
#[non_exhaustive]
pub enum Y2xxFrameError {
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `BITS ∉ {10, 12, 16}`.
  #[error("Y2xxFrame: unsupported BITS {bits}; must be 10, 12, or 16")]
  UnsupportedBits {
    /// `BITS` const-generic value.
    bits: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `width == 0` or `height == 0`.
  #[error("Y2xxFrame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `width % 2 != 0`. 4:2:2 subsampling requires even width.
  #[error("Y2xxFrame: width {width} is odd; 4:2:2 chroma subsampling requires even width")]
  OddWidth {
    /// Configured width.
    width: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `stride < width * 2` (u16 elements). Each row needs at least
  /// `width × 2` u16 elements (= `width × 4` bytes) to hold all
  /// pixels.
  #[error("Y2xxFrame: stride {stride} u16 elements is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride in u16 elements (`width × 2`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `packed.len() < expected`. The packed plane is too short for
  /// the declared geometry (in u16 elements).
  #[error("Y2xxFrame: plane too short: expected >= {expected} u16 elements, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in u16 elements (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in u16 elements.
    actual: usize,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `stride * height` overflows `u32`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("Y2xxFrame: stride × height overflows u32 (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
  #[unwrap(ignore)]
  #[try_unwrap(ignore)]
  /// `width × 2` overflows `u32`. Only reachable on 32-bit targets
  /// with extreme widths.
  #[error("Y2xxFrame: width {width} × 2 overflows u32 (per-row u16 element count)")]
  WidthOverflow {
    /// Configured width.
    width: u32,
  },
  /// `try_new_checked` only: a sample's low `(16 - BITS)` bits are
  /// non-zero. Diagnoses callers feeding non-MSB-aligned data
  /// (e.g. low-bit-packed yuv422p10le mistakenly handed to a Y210
  /// path). Y216 doesn't emit this since all 16 bits are active.
  #[error("Y2xxFrame: sample with non-zero low bits found; expected MSB-aligned data")]
  SampleLowBitsSet,
}

impl<'a, const BITS: u32, const BE: bool> Y2xxFrame<'a, BITS, BE> {
  /// Validates and constructs a [`Y2xxFrame`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Y2xxFrameError> {
    if BITS != 10 && BITS != 12 && BITS != 16 {
      return Err(Y2xxFrameError::UnsupportedBits { bits: BITS });
    }
    if width == 0 || height == 0 {
      return Err(Y2xxFrameError::ZeroDimension { width, height });
    }
    if !width.is_multiple_of(2) {
      return Err(Y2xxFrameError::OddWidth { width });
    }
    let min_stride = match width.checked_mul(2) {
      Some(n) => n,
      None => return Err(Y2xxFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Y2xxFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(Y2xxFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(Y2xxFrameError::PlaneTooShort {
        expected: plane_min,
        actual: packed.len(),
      });
    }
    Ok(Self {
      packed,
      width,
      height,
      stride,
    })
  }

  /// Like [`Self::try_new`] but additionally rejects samples whose
  /// low `(16 - BITS)` bits are non-zero. Only meaningful for
  /// `BITS ∈ {10, 12}`; for `BITS = 16` this delegates to
  /// [`Self::try_new`] (no low bits to check).
  ///
  /// Per the byte-encoding contract on the type-level docs, samples are
  /// validated **after** byte-order normalization (`u16::from_le` for
  /// `BE = false`, `u16::from_be` for `BE = true`) so the bit check
  /// operates on the intended logical sample value on every host. On
  /// little-endian hosts `from_le` is a no-op and `from_be` byte-swaps;
  /// on big-endian hosts the roles flip. Without this normalization a
  /// valid `Y210LE` plane on a BE host would have its MSB-aligned
  /// samples appear byte-swapped (low bits set in the host-native
  /// reading) and the validator would falsely reject every row — and
  /// vice-versa for `Y210BE` on an LE host.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new_checked(
    packed: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Y2xxFrameError> {
    let frame = Self::try_new(packed, width, height, stride)?;
    if BITS < 16 {
      let low_mask: u16 = (1u16 << (16 - BITS)) - 1;
      let row_elems = (width * 2) as usize;
      let h = height as usize;
      let stride_us = stride as usize;
      for row in 0..h {
        let start = row * stride_us;
        for &sample in &packed[start..start + row_elems] {
          // Normalize from wire byte order to host-native before the
          // bit check. `BE = false` → `from_le` (no-op on LE host,
          // byte-swap on BE host); `BE = true` → `from_be` (byte-swap
          // on LE host, no-op on BE host).
          let host = if BE {
            u16::from_be(sample)
          } else {
            u16::from_le(sample)
          };
          if host & low_mask != 0 {
            return Err(Y2xxFrameError::SampleLowBitsSet);
          }
        }
      }
    }
    Ok(frame)
  }

  /// Panicking convenience over [`Self::try_new`]. Per-variant
  /// panic messages mirror [`crate::frame::V210Frame::new`] for
  /// debuggability — generic "validation failed" doesn't tell a
  /// caller whether the issue was odd width, short plane, or
  /// stride-too-small.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        Y2xxFrameError::UnsupportedBits { .. } => panic!("invalid Y2xxFrame: unsupported BITS"),
        Y2xxFrameError::ZeroDimension { .. } => panic!("invalid Y2xxFrame: zero dimension"),
        Y2xxFrameError::OddWidth { .. } => panic!("invalid Y2xxFrame: odd width"),
        Y2xxFrameError::StrideTooSmall { .. } => panic!("invalid Y2xxFrame: stride too small"),
        Y2xxFrameError::PlaneTooShort { .. } => panic!("invalid Y2xxFrame: plane too short"),
        Y2xxFrameError::GeometryOverflow { .. } => panic!("invalid Y2xxFrame: geometry overflow"),
        Y2xxFrameError::WidthOverflow { .. } => panic!("invalid Y2xxFrame: width overflow"),
        // SampleLowBitsSet is only emitted by try_new_checked, never by try_new.
        // Listed for exhaustiveness so a future variant addition forces an explicit choice.
        Y2xxFrameError::SampleLowBitsSet => {
          panic!("invalid Y2xxFrame: sample low bits set (unreachable from try_new)")
        }
      },
    }
  }

  /// Packed plane: `(Y₀, U, Y₁, V)` u16 quadruples — `width × 2`
  /// u16 elements per row (= `width × 4` bytes). 4:2:2 chroma is
  /// shared between each Y pair; samples are MSB-aligned with the
  /// low `(16 - BITS)` bits zero (`BITS ∈ {10, 12}`).
  ///
  /// The slice carries the **LE-encoded byte layout** reinterpreted
  /// as `u16` (FFmpeg `*LE` convention) — see the type-level docs.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn packed(&self) -> &'a [u16] {
    self.packed
  }
  /// Frame width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn width(&self) -> u32 {
    self.width
  }
  /// Frame height in rows.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn height(&self) -> u32 {
    self.height
  }
  /// Stride in u16 elements (NOT bytes — this is the number of
  /// u16 slots per row).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
  /// Returns the compile-time BE flag — `true` if plane bytes are BE-encoded
  /// (`AV_PIX_FMT_Y2xxBE`), `false` if LE-encoded (`AV_PIX_FMT_Y2xxLE`).
  /// Runtime mirror of the `<const BE: bool>` type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}
