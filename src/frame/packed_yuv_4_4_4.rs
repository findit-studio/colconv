//! Packed YUV 4:4:4 frame types — Tier 5.
//!
//! This module is the container for the Tier 5 packed-YUV-4:4:4
//! family across all bit depths (8 / 10 / 12 / 16-bit): `v410`,
//! `xv36`, `vuya` / `vuyx`, `ayuv64`. Ship 12a adds [`V410Frame`]
//! and [`V30XFrame`] (10-bit, sibling formats with opposite padding
//! positions); Ship 12b adds [`Xv36Frame`] (12-bit MSB-aligned);
//! Ship 12c adds [`VuyaFrame`] and [`VuyxFrame`] (8-bit native, with
//! source α / α-as-padding semantics); Ship 12d adds [`Ayuv64Frame`]
//! (16-bit native, source α).

use derive_more::IsVariant;
use thiserror::Error;

/// Validated wrapper around a packed YUV 4:4:4 10-bit `V410` plane.
///
/// `V410` is the **MSB-padded** packed YUV 4:4:4 layout — the same
/// bits Microsoft V410 fourcc, NVIDIA Video Codec SDK, Apple
/// AVFoundation, and the FFmpeg `AV_CODEC_ID_V410` codec all describe.
/// Current FFmpeg (8.1+) exposes this layout as `AV_PIX_FMT_XV30LE`
/// (the `AV_PIX_FMT_V410` symbol was renamed to `XV30` — same bit
/// pattern, new name). Each pixel occupies one 32-bit word with the
/// following little-endian layout (MSB → LSB):
///
/// | Bits  | Field |
/// |-------|-------|
/// | 31:30 | padding (zero) |
/// | 29:20 | V (10 bits) |
/// | 19:10 | Y (10 bits) |
/// | 9:0   | U (10 bits) |
///
/// **If your data uses LSB padding instead** (`AV_PIX_FMT_V30XLE`,
/// `(msb) 10V 10Y 10U 2X (lsb)`), use [`V30XFrame`] — it is a
/// type-distinct sibling with the same shape but shifted bit
/// positions.
///
/// Each row holds exactly `width` u32 words (`stride >= width`); the
/// plane occupies `stride * height` u32 elements.
///
/// # Endian contract — `<const BE: bool = false>`
///
/// The `<const BE: bool>` parameter selects the per-word byte order:
/// `false` (default) → LE-encoded u32 words (V410 wire format,
/// QuickTime / FFmpeg `AV_PIX_FMT_V410`); `true` → BE-encoded u32
/// words (matches QuickTime-style BE V410 streams). Each u32 word is
/// byte-swapped under the hood by the row kernels — callers do **not**
/// pre-swap.
///
/// # Aliases
/// - [`V410LeFrame`] = `V410Frame<'a, false>` — explicit LE.
/// - [`V410BeFrame`] = `V410Frame<'a, true>` — explicit BE.
#[derive(Debug, Clone, Copy)]
pub struct V410Frame<'a, const BE: bool = false> {
  packed: &'a [u32],
  width: u32,
  height: u32,
  stride: u32,
}

/// LE-encoded `V410Frame` (`AV_PIX_FMT_V410` / `AV_PIX_FMT_XV30LE`).
/// Equivalent to the default `V410Frame<'a>`; provided as an explicit
/// alias for callers who want to document the endianness at the type
/// level.
pub type V410LeFrame<'a> = V410Frame<'a, false>;

/// BE-encoded `V410Frame`. Per-word u32s are big-endian-encoded;
/// downstream row kernels byte-swap each word before bit-extraction.
pub type V410BeFrame<'a> = V410Frame<'a, true>;

/// Errors returned by [`V410Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum V410FrameError {
  /// `width == 0` or `height == 0`.
  #[error("V410Frame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `stride < width`. Each row needs at least `width` u32 words.
  #[error("V410Frame: stride {stride} u32 elements is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride (= `width`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short for
  /// the declared geometry.
  #[error("V410Frame: plane too short: expected >= {expected} u32 elements, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in u32 elements (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in u32 elements.
    actual: usize,
  },
  /// `stride * height` overflows `usize`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("V410Frame: stride × height overflows usize (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
}

impl<'a, const BE: bool> V410Frame<'a, BE> {
  /// Validates and constructs a [`V410Frame`].
  ///
  /// The `<const BE: bool>` parameter selects whether the supplied
  /// `packed` slice is interpreted as LE-encoded u32 words
  /// (`BE = false`, default) or BE-encoded u32 words (`BE = true`).
  /// The byte-swap is performed inside the row kernels — this
  /// constructor does no I/O on the words.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u32],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, V410FrameError> {
    if width == 0 || height == 0 {
      return Err(V410FrameError::ZeroDimension { width, height });
    }
    if stride < width {
      return Err(V410FrameError::StrideTooSmall {
        min_stride: width,
        stride,
      });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(V410FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(V410FrameError::PlaneTooShort {
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

  /// Panicking convenience over [`Self::try_new`]. Per-variant panic
  /// messages mirror [`crate::frame::V210Frame::new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u32], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        V410FrameError::ZeroDimension { .. } => panic!("invalid V410Frame: zero dimension"),
        V410FrameError::StrideTooSmall { .. } => panic!("invalid V410Frame: stride too small"),
        V410FrameError::PlaneTooShort { .. } => panic!("invalid V410Frame: plane too short"),
        V410FrameError::GeometryOverflow { .. } => panic!("invalid V410Frame: geometry overflow"),
      },
    }
  }

  /// Packed plane: `stride * height` total u32 elements, with
  /// `width` active pixels per row and `stride` u32 elements per
  /// row. Each word holds one pixel `(U, Y, V, padding)` per the
  /// V410 layout described above.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn packed(&self) -> &'a [u32] {
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
  /// Stride in u32 elements (NOT bytes — the number of u32 slots
  /// per row, ≥ `width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
  /// Returns the compile-time BE flag — `true` if the plane u32 words
  /// are BE-encoded, `false` if LE-encoded. Runtime mirror of the
  /// `<const BE: bool>` type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}

/// Validated wrapper around a packed YUV 4:4:4 10-bit `V30X` plane.
///
/// `V30X` (FFmpeg `AV_PIX_FMT_V30XLE`) packs **one pixel per 32-bit word**
/// with the following little-endian layout (MSB → LSB):
///
/// | Bits  | Field |
/// |-------|-------|
/// | 31:22 | V (10 bits) |
/// | 21:12 | Y (10 bits) |
/// | 11:2  | U (10 bits) |
/// | 1:0   | padding (zero) |
///
/// This is a sibling of [`V410Frame`]: the pixel data is identical but
/// V30X places the 2-bit padding at the **LSB** (bits \[1:0\]), whereas V410
/// places it at the **MSB** (bits \[31:30\]). Bit-extraction shifts differ by
/// exactly 2.
///
/// Each row holds exactly `width` u32 words (`stride >= width`); the
/// plane occupies `stride * height` u32 elements.
#[derive(Debug, Clone, Copy)]
pub struct V30XFrame<'a> {
  packed: &'a [u32],
  width: u32,
  height: u32,
  stride: u32,
}

/// Errors returned by [`V30XFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum V30XFrameError {
  /// `width == 0` or `height == 0`.
  #[error("V30XFrame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `stride < width`. Each row needs at least `width` u32 words.
  #[error("V30XFrame: stride {stride} u32 elements is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride (= `width`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short for
  /// the declared geometry.
  #[error("V30XFrame: plane too short: expected >= {expected} u32 elements, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in u32 elements (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in u32 elements.
    actual: usize,
  },
  /// `stride * height` overflows `usize`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("V30XFrame: stride × height overflows usize (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
}

impl<'a> V30XFrame<'a> {
  /// Validates and constructs a [`V30XFrame`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u32],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, V30XFrameError> {
    if width == 0 || height == 0 {
      return Err(V30XFrameError::ZeroDimension { width, height });
    }
    if stride < width {
      return Err(V30XFrameError::StrideTooSmall {
        min_stride: width,
        stride,
      });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(V30XFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(V30XFrameError::PlaneTooShort {
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

  /// Panicking convenience over [`Self::try_new`]. Per-variant panic
  /// messages mirror [`crate::frame::V210Frame::new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u32], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        V30XFrameError::ZeroDimension { .. } => panic!("invalid V30XFrame: zero dimension"),
        V30XFrameError::StrideTooSmall { .. } => panic!("invalid V30XFrame: stride too small"),
        V30XFrameError::PlaneTooShort { .. } => panic!("invalid V30XFrame: plane too short"),
        V30XFrameError::GeometryOverflow { .. } => panic!("invalid V30XFrame: geometry overflow"),
      },
    }
  }

  /// Packed plane: `stride * height` total u32 elements, with
  /// `width` active pixels per row and `stride` u32 elements per
  /// row. Each word holds one pixel `(U, Y, V, padding)` per the
  /// V30X layout described above.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn packed(&self) -> &'a [u32] {
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
  /// Stride in u32 elements (NOT bytes — the number of u32 slots
  /// per row, ≥ `width`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

/// Validated wrapper around a packed YUV 4:4:4 12-bit `XV36` plane.
///
/// `XV36` (FFmpeg `AV_PIX_FMT_XV36LE`) packs **four u16 channels per
/// pixel** as `U(16) ‖ Y(16) ‖ V(16) ‖ A(16)` little-endian. Each
/// channel uses the high 12 bits of its u16 with the low 4 bits zero
/// (MSB-aligned at 12-bit, same encoding as `Y212`). The `X` prefix
/// means the A slot is **padding** — reads are tolerated but values
/// are discarded; RGBA outputs always force α = max (`0xFF` u8 /
/// `0x0FFF` u16 native-depth).
///
/// Per-pixel layout (LE, MSB → LSB inside each channel u16):
///
/// | u16 slot | Field | Active bits |
/// |----------|-------|-------------|
/// | 0        | U     | bits[15:4]  |
/// | 1        | Y     | bits[15:4]  |
/// | 2        | V     | bits[15:4]  |
/// | 3        | A     | bits[15:4] (padding) |
///
/// Each row holds exactly `width × 4` u16 elements (`stride >=
/// width × 4`); the plane occupies `stride * height` u16 elements.
///
/// # Endian contract — `<const BE: bool = false>`
///
/// The `<const BE: bool>` parameter selects the per-channel u16 byte
/// order: `false` (default) → LE-encoded bytes (`AV_PIX_FMT_XV36LE`),
/// `true` → BE-encoded bytes (`AV_PIX_FMT_XV36BE`). Each u16 channel
/// is byte-swapped under the hood by the row kernels — callers do
/// **not** pre-swap.
///
/// # Aliases
/// - [`Xv36LeFrame`] = `Xv36Frame<'a, false>` — explicit LE.
/// - [`Xv36BeFrame`] = `Xv36Frame<'a, true>` — explicit BE.
#[derive(Debug, Clone, Copy)]
pub struct Xv36Frame<'a, const BE: bool = false> {
  packed: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

/// LE-encoded `Xv36Frame` (`AV_PIX_FMT_XV36LE`). Equivalent to the
/// default `Xv36Frame<'a>`; provided as an explicit alias.
pub type Xv36LeFrame<'a> = Xv36Frame<'a, false>;

/// BE-encoded `Xv36Frame` (`AV_PIX_FMT_XV36BE`). Per-channel u16s are
/// big-endian-encoded; downstream row kernels byte-swap each channel.
pub type Xv36BeFrame<'a> = Xv36Frame<'a, true>;

/// Errors returned by [`Xv36Frame::try_new`] and
/// [`Xv36Frame::try_new_checked`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Xv36FrameError {
  /// `width == 0` or `height == 0`.
  #[error("Xv36Frame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `width × 4` overflows `u32`. Only reachable on 32-bit targets
  /// with extreme widths.
  #[error("Xv36Frame: width {width} × 4 overflows u32 (per-row u16 element count)")]
  WidthOverflow {
    /// Configured width.
    width: u32,
  },
  /// `stride < width × 4` (u16 elements). Each row needs at least
  /// `width × 4` u16 elements (= `width × 8` bytes) to hold all
  /// pixels.
  #[error("Xv36Frame: stride {stride} u16 elements is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride in u16 elements (`width × 4`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short.
  #[error("Xv36Frame: plane too short: expected >= {expected} u16 elements, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in u16 elements (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in u16 elements.
    actual: usize,
  },
  /// `stride * height` overflows `usize`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("Xv36Frame: stride × height overflows usize (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
  /// `try_new_checked` only: a sample's low 4 bits are non-zero
  /// after normalizing the byte-storage `u16` to the logical sample
  /// value (`u16::from_be` for `Xv36BeFrame`, `u16::from_le` for
  /// `Xv36LeFrame`). Diagnoses callers feeding low-bit-packed data
  /// (e.g. `yuv444p12le` mistakenly handed to an XV36 path).
  ///
  /// `value` is the **logical** sample (post-normalization) so the
  /// reported nibble is comparable across hosts and BE/LE flags.
  #[error(
    "Xv36Frame: sample {value:#06x} at element {index} has non-zero low 4 bits (expected MSB-aligned XV36 data)"
  )]
  SampleLowBitsSet {
    /// Element index (in `u16` slots) within the packed plane.
    index: usize,
    /// Offending sample value, normalized to host-native via
    /// `u16::from_be`/`u16::from_le` per the `BE` flag.
    value: u16,
  },
}

impl<'a, const BE: bool> Xv36Frame<'a, BE> {
  /// Validates and constructs an [`Xv36Frame`].
  ///
  /// `<const BE: bool>` selects LE (`false`, default) vs BE (`true`)
  /// per-channel u16 byte order; row kernels perform the byte-swap.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Xv36FrameError> {
    if width == 0 || height == 0 {
      return Err(Xv36FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(n) => n,
      None => return Err(Xv36FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Xv36FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(Xv36FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(Xv36FrameError::PlaneTooShort {
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
  /// low 4 bits are non-zero. Validates the MSB-alignment invariant
  /// (low 4 bits zero per the XV36 encoding).
  ///
  /// Per the BE/LE byte-storage contract documented on the type,
  /// each `u16` slot is normalized via `u16::from_be` (when
  /// `BE = true`) or `u16::from_le` (when `BE = false`) before the
  /// low-nibble check, so the test operates on the intended logical
  /// sample value on every host. Without this normalization a valid
  /// `Xv36BeFrame` sample such as `0xABC0` (BE bytes `[0xAB, 0xC0]`)
  /// reads as host-native `0xC0AB` on a little-endian host and the
  /// validator would falsely reject every row; conversely, true low-
  /// bit-set BE samples could be judged against the wrong nibble.
  /// Mirrors the `PnFrame::try_new_checked` BE-normalization pattern
  /// (PR #89 `b9a6c19`). The reported `value` is the normalized
  /// logical sample.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn try_new_checked(
    packed: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Xv36FrameError> {
    let frame = Self::try_new(packed, width, height, stride)?;
    let row_elems = (width * 4) as usize;
    let h = height as usize;
    let stride_us = stride as usize;
    for row in 0..h {
      let start = row * stride_us;
      for (col, &sample) in packed[start..start + row_elems].iter().enumerate() {
        // Normalize byte-storage word to host-native logical sample
        // before the low-nibble check (no-op on matching-endian host,
        // byte-swap otherwise).
        let logical = if BE {
          u16::from_be(sample)
        } else {
          u16::from_le(sample)
        };
        if logical & 0x000F != 0 {
          return Err(Xv36FrameError::SampleLowBitsSet {
            index: start + col,
            value: logical,
          });
        }
      }
    }
    Ok(frame)
  }

  /// Panicking convenience over [`Self::try_new`]. Per-variant panic
  /// messages mirror [`crate::frame::V410Frame::new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        Xv36FrameError::ZeroDimension { .. } => panic!("invalid Xv36Frame: zero dimension"),
        Xv36FrameError::WidthOverflow { .. } => panic!("invalid Xv36Frame: width overflow"),
        Xv36FrameError::StrideTooSmall { .. } => panic!("invalid Xv36Frame: stride too small"),
        Xv36FrameError::PlaneTooShort { .. } => panic!("invalid Xv36Frame: plane too short"),
        Xv36FrameError::GeometryOverflow { .. } => panic!("invalid Xv36Frame: geometry overflow"),
        // SampleLowBitsSet is only emitted by try_new_checked.
        Xv36FrameError::SampleLowBitsSet { .. } => {
          panic!("invalid Xv36Frame: sample low bits set (unreachable from try_new)")
        }
      },
    }
  }

  /// Packed plane: `stride * height` total u16 elements, with
  /// `width × 4` active u16 elements per row (4 channels per pixel)
  /// and `stride` u16 elements per row.
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
  /// Stride in u16 elements (NOT bytes — the number of u16 slots per
  /// row, ≥ `width × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
  /// Returns the compile-time BE flag — `true` if per-channel u16s
  /// are BE-encoded (`AV_PIX_FMT_XV36BE`), `false` if LE-encoded
  /// (`AV_PIX_FMT_XV36LE`). Runtime mirror of the `<const BE: bool>`
  /// type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}

/// Validated wrapper around a packed YUV 4:4:4 8-bit `VUYA` plane.
///
/// `VUYA` (FFmpeg `AV_PIX_FMT_VUYA`) packs **four bytes per pixel**
/// as `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)` little-endian, where the A byte is
/// the **source alpha** (passed through to RGBA outputs). For the
/// α-as-padding sibling — A is ignored on read and RGBA outputs
/// force α=`0xFF` — see [`VuyxFrame`].
///
/// Per-pixel byte layout:
///
/// | Byte offset | Field |
/// |-------------|-------|
/// | 0           | V     |
/// | 1           | U     |
/// | 2           | Y     |
/// | 3           | A (source alpha) |
///
/// Each row holds exactly `width × 4` bytes (`stride >= width × 4`);
/// the plane occupies `stride * height` bytes total.
#[derive(Debug, Clone, Copy)]
pub struct VuyaFrame<'a> {
  packed: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

/// Errors returned by [`VuyaFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum VuyaFrameError {
  /// `width == 0` or `height == 0`.
  #[error("VuyaFrame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `width × 4` overflows `u32`. Only reachable on 32-bit targets
  /// with extreme widths.
  #[error("VuyaFrame: width {width} × 4 overflows u32 (per-row byte count)")]
  WidthOverflow {
    /// Configured width.
    width: u32,
  },
  /// `stride < width × 4` (bytes). Each row needs at least
  /// `width × 4` bytes to hold all pixels.
  #[error("VuyaFrame: stride {stride} bytes is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride in bytes (`width × 4`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short.
  #[error("VuyaFrame: plane too short: expected >= {expected} bytes, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in bytes (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in bytes.
    actual: usize,
  },
  /// `stride * height` overflows `usize`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("VuyaFrame: stride × height overflows usize (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
}

impl<'a> VuyaFrame<'a> {
  /// Validates and constructs a [`VuyaFrame`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, VuyaFrameError> {
    if width == 0 || height == 0 {
      return Err(VuyaFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(n) => n,
      None => return Err(VuyaFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(VuyaFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(VuyaFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(VuyaFrameError::PlaneTooShort {
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

  /// Panicking convenience over [`Self::try_new`]. Per-variant panic
  /// messages mirror [`crate::frame::V410Frame::new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        VuyaFrameError::ZeroDimension { .. } => panic!("invalid VuyaFrame: zero dimension"),
        VuyaFrameError::WidthOverflow { .. } => panic!("invalid VuyaFrame: width overflow"),
        VuyaFrameError::StrideTooSmall { .. } => panic!("invalid VuyaFrame: stride too small"),
        VuyaFrameError::PlaneTooShort { .. } => panic!("invalid VuyaFrame: plane too short"),
        VuyaFrameError::GeometryOverflow { .. } => {
          panic!("invalid VuyaFrame: geometry overflow")
        }
      },
    }
  }

  /// Packed plane: `stride * height` total bytes, with `width × 4`
  /// active bytes per row (4 channels per pixel) and `stride` bytes
  /// per row. Byte layout per pixel: `V(8) ‖ U(8) ‖ Y(8) ‖ A(8)`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn packed(&self) -> &'a [u8] {
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
  /// Stride in bytes (the number of bytes per row, ≥ `width × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

/// Validated wrapper around a packed YUV 4:4:4 8-bit `VUYX` plane.
///
/// `VUYX` (FFmpeg `AV_PIX_FMT_VUYX`) packs **four bytes per pixel**
/// as `V(8) ‖ U(8) ‖ Y(8) ‖ X(8)` little-endian. The `X` byte is
/// **padding** — values are ignored on read and RGBA outputs always
/// force α = `0xFF`. For the source-alpha sibling where the fourth
/// byte carries meaningful alpha, see [`VuyaFrame`].
///
/// Per-pixel byte layout:
///
/// | Byte offset | Field |
/// |-------------|-------|
/// | 0           | V     |
/// | 1           | U     |
/// | 2           | Y     |
/// | 3           | X (padding) |
///
/// Each row holds exactly `width × 4` bytes (`stride >= width × 4`);
/// the plane occupies `stride * height` bytes total.
#[derive(Debug, Clone, Copy)]
pub struct VuyxFrame<'a> {
  packed: &'a [u8],
  width: u32,
  height: u32,
  stride: u32,
}

/// Errors returned by [`VuyxFrame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum VuyxFrameError {
  /// `width == 0` or `height == 0`.
  #[error("VuyxFrame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `width × 4` overflows `u32`. Only reachable on 32-bit targets
  /// with extreme widths.
  #[error("VuyxFrame: width {width} × 4 overflows u32 (per-row byte count)")]
  WidthOverflow {
    /// Configured width.
    width: u32,
  },
  /// `stride < width × 4` (bytes). Each row needs at least
  /// `width × 4` bytes to hold all pixels.
  #[error("VuyxFrame: stride {stride} bytes is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride in bytes (`width × 4`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short.
  #[error("VuyxFrame: plane too short: expected >= {expected} bytes, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in bytes (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in bytes.
    actual: usize,
  },
  /// `stride * height` overflows `usize`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("VuyxFrame: stride × height overflows usize (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
}

impl<'a> VuyxFrame<'a> {
  /// Validates and constructs a [`VuyxFrame`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u8],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, VuyxFrameError> {
    if width == 0 || height == 0 {
      return Err(VuyxFrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(n) => n,
      None => return Err(VuyxFrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(VuyxFrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(VuyxFrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(VuyxFrameError::PlaneTooShort {
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

  /// Panicking convenience over [`Self::try_new`]. Per-variant panic
  /// messages mirror [`crate::frame::V410Frame::new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u8], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        VuyxFrameError::ZeroDimension { .. } => panic!("invalid VuyxFrame: zero dimension"),
        VuyxFrameError::WidthOverflow { .. } => panic!("invalid VuyxFrame: width overflow"),
        VuyxFrameError::StrideTooSmall { .. } => panic!("invalid VuyxFrame: stride too small"),
        VuyxFrameError::PlaneTooShort { .. } => panic!("invalid VuyxFrame: plane too short"),
        VuyxFrameError::GeometryOverflow { .. } => {
          panic!("invalid VuyxFrame: geometry overflow")
        }
      },
    }
  }

  /// Packed plane: `stride * height` total bytes, with `width × 4`
  /// active bytes per row (4 channels per pixel) and `stride` bytes
  /// per row. Byte layout per pixel: `V(8) ‖ U(8) ‖ Y(8) ‖ X(8)`
  /// (X = padding).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn packed(&self) -> &'a [u8] {
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
  /// Stride in bytes (the number of bytes per row, ≥ `width × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
}

/// Validated wrapper around a packed YUV 4:4:4 16-bit `AYUV64` plane.
///
/// `AYUV64` (FFmpeg `AV_PIX_FMT_AYUV64LE`) packs **four u16 channels
/// per pixel** as `A(16) ‖ Y(16) ‖ U(16) ‖ V(16)` little-endian.
/// Each channel uses the full u16 range (16-bit native — no padding
/// bits). Source α is real and pass-through to RGBA outputs.
///
/// Per-pixel layout (LE, MSB → LSB inside each u16):
///
/// | u16 slot | Field | Active bits           |
/// |----------|-------|-----------------------|
/// | 0        | A     | bits\[15:0\] (source) |
/// | 1        | Y     | bits\[15:0\] (16-bit) |
/// | 2        | U     | bits\[15:0\] (16-bit) |
/// | 3        | V     | bits\[15:0\] (16-bit) |
///
/// Each row holds exactly `width × 4` u16 elements (`stride >=
/// width × 4`); the plane occupies `stride * height` u16 elements.
///
/// # Endian contract — `<const BE: bool = false>`
///
/// The `<const BE: bool>` parameter selects the per-channel u16 byte
/// order: `false` (default) → LE-encoded bytes (`AV_PIX_FMT_AYUV64LE`),
/// `true` → BE-encoded bytes (`AV_PIX_FMT_AYUV64BE`). Each u16 channel
/// is byte-swapped under the hood by the row kernels — callers do
/// **not** pre-swap.
///
/// # Aliases
/// - [`Ayuv64LeFrame`] = `Ayuv64Frame<'a, false>` — explicit LE.
/// - [`Ayuv64BeFrame`] = `Ayuv64Frame<'a, true>` — explicit BE.
#[derive(Debug, Clone, Copy)]
pub struct Ayuv64Frame<'a, const BE: bool = false> {
  packed: &'a [u16],
  width: u32,
  height: u32,
  stride: u32,
}

/// LE-encoded `Ayuv64Frame` (`AV_PIX_FMT_AYUV64LE`). Equivalent to
/// the default `Ayuv64Frame<'a>`; provided as an explicit alias.
pub type Ayuv64LeFrame<'a> = Ayuv64Frame<'a, false>;

/// BE-encoded `Ayuv64Frame` (`AV_PIX_FMT_AYUV64BE`). Per-channel u16s
/// are big-endian-encoded; downstream row kernels byte-swap each
/// channel.
pub type Ayuv64BeFrame<'a> = Ayuv64Frame<'a, true>;

/// Errors returned by [`Ayuv64Frame::try_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Error)]
#[non_exhaustive]
pub enum Ayuv64FrameError {
  /// `width == 0` or `height == 0`.
  #[error("Ayuv64Frame: zero dimension width={width} height={height}")]
  ZeroDimension {
    /// Configured width.
    width: u32,
    /// Configured height.
    height: u32,
  },
  /// `width × 4` overflows `u32`. Only reachable on 32-bit targets
  /// with extreme widths.
  #[error("Ayuv64Frame: width {width} × 4 overflows u32 (per-row u16 element count)")]
  WidthOverflow {
    /// Configured width.
    width: u32,
  },
  /// `stride < width × 4` (u16 elements). Each row needs at least
  /// `width × 4` u16 elements (= `width × 8` bytes) to hold all
  /// pixels.
  #[error("Ayuv64Frame: stride {stride} u16 elements is below the minimum {min_stride}")]
  StrideTooSmall {
    /// Minimum required stride in u16 elements (`width × 4`).
    min_stride: u32,
    /// Caller-supplied stride.
    stride: u32,
  },
  /// `packed.len() < expected`. The packed plane is too short.
  #[error("Ayuv64Frame: plane too short: expected >= {expected} u16 elements, got {actual}")]
  PlaneTooShort {
    /// Minimum required plane length in u16 elements (`stride * height`).
    expected: usize,
    /// Caller-supplied plane length in u16 elements.
    actual: usize,
  },
  /// `stride * height` overflows `usize`. Only reachable on 32-bit
  /// targets with extreme dimensions.
  #[error("Ayuv64Frame: stride × height overflows usize (stride={stride}, rows={rows})")]
  GeometryOverflow {
    /// Configured stride.
    stride: u32,
    /// Configured height.
    rows: u32,
  },
}

impl<'a, const BE: bool> Ayuv64Frame<'a, BE> {
  /// Validates and constructs an [`Ayuv64Frame`].
  ///
  /// `<const BE: bool>` selects LE (`false`, default) vs BE (`true`)
  /// per-channel u16 byte order; row kernels perform the byte-swap.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn try_new(
    packed: &'a [u16],
    width: u32,
    height: u32,
    stride: u32,
  ) -> Result<Self, Ayuv64FrameError> {
    if width == 0 || height == 0 {
      return Err(Ayuv64FrameError::ZeroDimension { width, height });
    }
    let min_stride = match width.checked_mul(4) {
      Some(n) => n,
      None => return Err(Ayuv64FrameError::WidthOverflow { width }),
    };
    if stride < min_stride {
      return Err(Ayuv64FrameError::StrideTooSmall { min_stride, stride });
    }
    let plane_min = match (stride as usize).checked_mul(height as usize) {
      Some(n) => n,
      None => {
        return Err(Ayuv64FrameError::GeometryOverflow {
          stride,
          rows: height,
        });
      }
    };
    if packed.len() < plane_min {
      return Err(Ayuv64FrameError::PlaneTooShort {
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

  /// Panicking convenience over [`Self::try_new`]. Per-variant panic
  /// messages mirror [`crate::frame::V410Frame::new`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(packed: &'a [u16], width: u32, height: u32, stride: u32) -> Self {
    match Self::try_new(packed, width, height, stride) {
      Ok(f) => f,
      Err(e) => match e {
        Ayuv64FrameError::ZeroDimension { .. } => panic!("invalid Ayuv64Frame: zero dimension"),
        Ayuv64FrameError::WidthOverflow { .. } => panic!("invalid Ayuv64Frame: width overflow"),
        Ayuv64FrameError::StrideTooSmall { .. } => panic!("invalid Ayuv64Frame: stride too small"),
        Ayuv64FrameError::PlaneTooShort { .. } => panic!("invalid Ayuv64Frame: plane too short"),
        Ayuv64FrameError::GeometryOverflow { .. } => {
          panic!("invalid Ayuv64Frame: geometry overflow")
        }
      },
    }
  }

  /// Packed plane: `stride * height` total u16 elements, with
  /// `width × 4` active u16 elements per row (4 channels per pixel)
  /// and `stride` u16 elements per row. Channel layout per pixel:
  /// `A(16) ‖ Y(16) ‖ U(16) ‖ V(16)`.
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
  /// Stride in u16 elements (NOT bytes — the number of u16 slots per
  /// row, ≥ `width × 4`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn stride(&self) -> u32 {
    self.stride
  }
  /// Returns the compile-time BE flag — `true` if per-channel u16s
  /// are BE-encoded (`AV_PIX_FMT_AYUV64BE`), `false` if LE-encoded
  /// (`AV_PIX_FMT_AYUV64LE`). Runtime mirror of the `<const BE: bool>`
  /// type parameter.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn is_be(&self) -> bool {
    BE
  }
}
