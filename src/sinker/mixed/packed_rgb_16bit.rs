//! Sinker impl stubs for 16-bit packed-RGB **source** formats (Tier 8 finish).
//!
//! Sources:
//! - [`Rgb48`]  — `R, G, B` u16 per pixel (`AV_PIX_FMT_RGB48LE`).
//! - [`Bgr48`]  — `B, G, R` u16 per pixel (`AV_PIX_FMT_BGR48LE`).
//! - [`Rgba64`] — `R, G, B, A` u16 per pixel (`AV_PIX_FMT_RGBA64LE`).
//! - [`Bgra64`] — `B, G, R, A` u16 per pixel (`AV_PIX_FMT_BGRA64LE`).
//!
//! **Task 1 stub:** `PixelSink::process` bodies are `unimplemented!()`.
//! Full kernel dispatch and Strategy A+ logic will be wired in Task 10
//! once the scalar kernels (Task 2) and SIMD backends (Tasks 3–8) and
//! dispatcher (Task 9) are in place.

use super::{MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match};
use crate::{
  PixelSink,
  yuv::{
    Bgr48, Bgr48Row, Bgr48Sink, Bgra64, Bgra64Row, Bgra64Sink, Rgb48, Rgb48Row, Rgb48Sink, Rgba64,
    Rgba64Row, Rgba64Sink,
  },
};

// ---- Rgb48 ----------------------------------------------------------------

impl Rgb48Sink for MixedSinker<'_, Rgb48> {}

impl PixelSink for MixedSinker<'_, Rgb48> {
  type Input<'r> = Rgb48Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Rgb48Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();

    let packed_expected = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 3,
    })?;
    if row.rgb48().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Rgb48Packed,
        row: idx,
        expected: packed_expected,
        actual: row.rgb48().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    unimplemented!("Rgb48 sinker kernels not yet wired (Task 10)")
  }
}

// ---- Bgr48 ----------------------------------------------------------------

impl Bgr48Sink for MixedSinker<'_, Bgr48> {}

impl PixelSink for MixedSinker<'_, Bgr48> {
  type Input<'r> = Bgr48Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Bgr48Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();

    let packed_expected = w.checked_mul(3).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 3,
    })?;
    if row.bgr48().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bgr48Packed,
        row: idx,
        expected: packed_expected,
        actual: row.bgr48().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    unimplemented!("Bgr48 sinker kernels not yet wired (Task 10)")
  }
}

// ---- Rgba64 ---------------------------------------------------------------

impl Rgba64Sink for MixedSinker<'_, Rgba64> {}

impl PixelSink for MixedSinker<'_, Rgba64> {
  type Input<'r> = Rgba64Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Rgba64Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();

    let packed_expected = w.checked_mul(4).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 4,
    })?;
    if row.rgba64().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Rgba64Packed,
        row: idx,
        expected: packed_expected,
        actual: row.rgba64().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    unimplemented!("Rgba64 sinker kernels not yet wired (Task 10)")
  }
}

// ---- Bgra64 ---------------------------------------------------------------

impl Bgra64Sink for MixedSinker<'_, Bgra64> {}

impl PixelSink for MixedSinker<'_, Bgra64> {
  type Input<'r> = Bgra64Row<'r>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Bgra64Row<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();

    let packed_expected = w.checked_mul(4).ok_or(MixedSinkerError::GeometryOverflow {
      width: w,
      height: h,
      channels: 4,
    })?;
    if row.bgra64().len() != packed_expected {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Bgra64Packed,
        row: idx,
        expected: packed_expected,
        actual: row.bgra64().len(),
      });
    }
    if idx >= self.height {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: self.height,
      });
    }

    unimplemented!("Bgra64 sinker kernels not yet wired (Task 10)")
  }
}
