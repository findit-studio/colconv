//! [`PixelSink`] implementations for Monoblack and Monowhite sources.

use super::{MixedSinker, MixedSinkerError, RowSlice, check_dimensions_match};
use crate::{
  PixelSink,
  row::dispatch::mono1bit as dispatch,
  yuv::{Monoblack, MonoblackRow, MonoblackSink, Monowhite, MonowhiteRow, MonowhiteSink},
};

// ---- Monoblack impl ---------------------------------------------------------

impl MonoblackSink for MixedSinker<'_, Monoblack> {}

impl PixelSink for MixedSinker<'_, Monoblack> {
  type Input<'i> = MonoblackRow<'i>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Self::Input<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let min_bytes = w.div_ceil(8);
    if row.y().len() < min_bytes {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: min_bytes,
        actual: row.y().len(),
      });
    }

    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: h,
      });
    }

    let Self {
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(buf) = rgb.as_deref_mut() {
      dispatch::monoblack_to_rgb_or_rgba_row::<false>(
        row.y(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      dispatch::monoblack_to_rgb_or_rgba_row::<true>(
        row.y(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      dispatch::monoblack_to_rgb_u16_or_rgba_u16_row::<false>(
        row.y(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      dispatch::monoblack_to_rgb_u16_or_rgba_u16_row::<true>(
        row.y(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma.as_deref_mut() {
      dispatch::monoblack_to_luma_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma_u16.as_deref_mut() {
      dispatch::monoblack_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      dispatch::monoblack_to_hsv_row(
        row.y(),
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    Ok(())
  }
}

// ---- Monowhite impl ---------------------------------------------------------

impl MonowhiteSink for MixedSinker<'_, Monowhite> {}

impl PixelSink for MixedSinker<'_, Monowhite> {
  type Input<'i> = MonowhiteRow<'i>;
  type Error = MixedSinkerError;

  fn begin_frame(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
    check_dimensions_match(self.width, self.height, width, height)
  }

  fn process(&mut self, row: Self::Input<'_>) -> Result<(), Self::Error> {
    let w = self.width;
    let h = self.height;
    let idx = row.row();
    let use_simd = self.simd;

    let min_bytes = w.div_ceil(8);
    if row.y().len() < min_bytes {
      return Err(MixedSinkerError::RowShapeMismatch {
        which: RowSlice::Y,
        row: idx,
        expected: min_bytes,
        actual: row.y().len(),
      });
    }

    if idx >= h {
      return Err(MixedSinkerError::RowIndexOutOfRange {
        row: idx,
        configured_height: h,
      });
    }

    let Self {
      rgb,
      rgba,
      rgb_u16,
      rgba_u16,
      luma,
      luma_u16,
      hsv,
      ..
    } = self;

    let one_plane_start = idx * w;
    let one_plane_end = one_plane_start + w;

    if let Some(buf) = rgb.as_deref_mut() {
      dispatch::monowhite_to_rgb_or_rgba_row::<false>(
        row.y(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba.as_deref_mut() {
      dispatch::monowhite_to_rgb_or_rgba_row::<true>(
        row.y(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgb_u16.as_deref_mut() {
      dispatch::monowhite_to_rgb_u16_or_rgba_u16_row::<false>(
        row.y(),
        &mut buf[one_plane_start * 3..one_plane_end * 3],
        w,
        use_simd,
      );
    }

    if let Some(buf) = rgba_u16.as_deref_mut() {
      dispatch::monowhite_to_rgb_u16_or_rgba_u16_row::<true>(
        row.y(),
        &mut buf[one_plane_start * 4..one_plane_end * 4],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma.as_deref_mut() {
      dispatch::monowhite_to_luma_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(buf) = luma_u16.as_deref_mut() {
      dispatch::monowhite_to_luma_u16_row(
        row.y(),
        &mut buf[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    if let Some(hsv) = hsv.as_mut() {
      dispatch::monowhite_to_hsv_row(
        row.y(),
        &mut hsv.h[one_plane_start..one_plane_end],
        &mut hsv.s[one_plane_start..one_plane_end],
        &mut hsv.v[one_plane_start..one_plane_end],
        w,
        use_simd,
      );
    }

    Ok(())
  }
}
