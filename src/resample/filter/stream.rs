//! Scalar streaming separable **filter** engine — the signed twin of
//! [`AreaStream`](super::super::AreaStream).
//!
//! Like the area engine it H-reduces each source row through the
//! horizontal window into a staged `out_w * channels` row, then
//! V-accumulates that staged row under the vertical windows and emits an
//! output row on its last contributing source row. Two things differ:
//!
//! - **Signed weights.** Coefficients are floating-point and may be
//!   negative (Catmull-Rom, Lanczos), so accumulation is in floating
//!   point, not exact integer. `f64` accumulators carry every element type
//!   and the products are exact. The `u8` stream resamples on PIL's 8bpc
//!   fixed-point coefficient grid ([`FilterSample::coeffs`] selects
//!   `coeffs_q8`), so its scalar finalize is **byte-exact** with Pillow;
//!   `u16` / `f32` use the full-precision coefficients and land inside the
//!   +-1-LSB PIL parity budget at the single narrowing.
//!
//! - **More than two pending output rows.** A box span touches a source
//!   cell at most twice, so [`AreaStream`](super::super::AreaStream)
//!   assumes <=2 open accumulators. A filter window is wide
//!   (`ceil(2 * support_v) + 1` source rows for a Lanczos3 downscale), so
//!   a source row contributes to many concurrently-open output rows. This
//!   engine keeps a **ring of accumulator rows**, one per currently-open
//!   output row, indexed `oy % ring_cap`.
//!
//! Finalize clamps to the type range for the integer streams (`u8` ->
//! `0..=255`, `u16` -> `0..=65535`) and is the identity (round) for the
//! `f32` stream — matching PIL's per-mode behavior (`L`/`I;16` clamp, `F`
//! is unclamped).

use std::vec::Vec;

use super::{
  super::{OutOfSequenceRow, PlanGeometry, ResampleError},
  FilterAxis,
};

/// The sample element a [`FilterStream`] resamples. Abstracts the element
/// width, the per-type finalize (clamp-and-round for the integer streams,
/// identity for `f32`), and the per-type SIMD H-pass dispatch (a `u8` /
/// `u16` / `f32` row widens to the shared `f64` accumulation domain
/// differently). Both passes accumulate in `f64`, so there is a single
/// accumulator type shared by every element; the supertrait
/// [`crate::row::FilterSimdElem`] supplies the per-element kernel
/// selection the H-pass dispatcher routes through.
pub(crate) trait FilterSample: crate::row::FilterSimdElem {
  /// Whether this element routes a SIMD H-pass that consumes the plan-time
  /// [`FilterPaddedSpans`](crate::row::FilterPaddedSpans) staging arena.
  /// `true` for the SIMD-backed tiers (`u8` / `u16` / `f32` — the default);
  /// `false` for the scalar-only `u32` tier, whose H-pass widens to the exact
  /// `f64` domain with no per-backend SIMD kernel. [`FilterStream::new`] skips
  /// building the arena when this is `false`, routing the dispatcher to the
  /// unpadded scalar reduce (the identical exact `f64` dot product) with no
  /// dead staging allocation.
  const NEEDS_SIMD_STAGING: bool = true;
  /// Selects this element's H/V coefficient set from a [`FilterAxis`].
  /// `u8` returns PIL's 8bpc fixed-point (`PRECISION_BITS = 22`) snapped
  /// set, so both passes' `clip8` quantize byte-for-byte with Pillow's
  /// integer pipeline; `u16` and `f32` return the full-precision float set
  /// (their 32bpc parity is already inside budget and must not move). The
  /// two sets share `starts` / `offsets`, so every consumer (scalar
  /// H-pass, the SIMD staging arena, and the V-pass span) slices a window
  /// identically out of whichever set this returns.
  fn coeffs(axis: &FilterAxis) -> &[f32];

  /// `(first source sample, window)` for output index `j` from this
  /// element's coefficient set — the V-pass twin of [`Self::coeffs`].
  /// Defaults to the full-precision [`FilterAxis::span`] (`u16` / `f32`);
  /// `u8` overrides to the 8bpc fixed-point [`FilterAxis::span_q8`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn v_span(axis: &FilterAxis, j: usize) -> (usize, &[f32]) {
    axis.span(j)
  }

  /// Runtime-dispatched H-pass of one source `row` into `h_tmp` (the raw
  /// `f64` dot products, `out_w * channels` wide). Routes to the highest
  /// available SIMD tier when `use_simd` and `padded` permit, else the
  /// scalar reference. The signed twin of
  /// [`AreaSample::h_reduce`](super::super::AreaSample::h_reduce).
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn h_reduce(
    row: &[Self],
    channels: usize,
    h: &FilterAxis,
    padded: Option<&crate::row::FilterPaddedSpans>,
    h_tmp: &mut [f64],
    use_simd: bool,
  ) {
    crate::row::filter_h_reduce_row(
      row,
      channels,
      &h.starts,
      &h.offsets,
      Self::coeffs(h),
      padded,
      h_tmp,
      use_simd,
    );
  }

  /// Runtime-dispatched V-pass AXPY `acc[i] += w * h_tmp[i]` in `f64`,
  /// element-wise (mul+add), so it stays bit-equal to the scalar reference
  /// for every backend.
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn v_accumulate(acc: &mut [f64], h_tmp: &[f64], w: f32, use_simd: bool) {
    crate::row::filter_v_accumulate(acc, h_tmp, w, use_simd);
  }

  /// Quantize one horizontal-pass accumulator to the intermediate PIL
  /// would store between its two passes, **kept in the `f64` domain** so
  /// the vertical pass reads it directly. This is the crux of matching
  /// Pillow's two-pass behavior per mode:
  /// - `u8` (PIL 8bpc `L` / `RGB`): the intermediate image is itself
  ///   `u8`, so round-half **and clamp to `0..=255`** (PIL `clip8`).
  /// - `u16` (PIL 32bpc `I` resampler): the intermediate is a wide
  ///   integer, so round-half-up but **do not clamp** — a negative-lobe
  ///   overshoot must survive into the vertical pass (clamping only at the
  ///   final output).
  /// - `f32` (PIL 32bpc `F`): the intermediate is `f32`, so **narrow to
  ///   `f32`** (no rounding, no clamp).
  fn quantize_intermediate(acc: f64) -> f64;
  /// Resolve one finished accumulator to an output sample: round-half and
  /// clamp to the type range for the integer streams; narrow for `f32`
  /// (PIL `F`-mode is unclamped).
  fn finalize(acc: f64) -> Self;
}

impl FilterSample for u8 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn coeffs(axis: &FilterAxis) -> &[f32] {
    // PIL's 8bpc resize is fixed-point: read the `PRECISION_BITS = 22`
    // snapped set so `f64`-accumulating `K_i/2^22 * pixel_i` reproduces
    // PIL's `ss/2^22`, byte-exact through both passes' `clip8`.
    &axis.coeffs_q8
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn v_span(axis: &FilterAxis, j: usize) -> (usize, &[f32]) {
    axis.span_q8(j)
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn quantize_intermediate(acc: f64) -> f64 {
    // PIL `clip8` on the u8 H-pass image: round-half-up, clamp to 0..=255.
    floor_f64_local(acc + 0.5).clamp(0.0, 255.0)
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn finalize(acc: f64) -> u8 {
    floor_f64_local(acc + 0.5).clamp(0.0, 255.0) as u8
  }
}

impl FilterSample for u16 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn coeffs(axis: &FilterAxis) -> &[f32] {
    // PIL 32bpc `I` resampler uses double coefficients; the full-precision
    // `f32` set already matches within one LSB and must not move.
    &axis.coeffs
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn quantize_intermediate(acc: f64) -> f64 {
    // PIL 32bpc `I` resampler: the H-pass intermediate is rounded to a
    // wide integer with NO range clamp (clamping is final-output only), so
    // a negative-lobe overshoot survives into the vertical pass.
    floor_f64_local(acc + 0.5)
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn finalize(acc: f64) -> u16 {
    floor_f64_local(acc + 0.5).clamp(0.0, 65535.0) as u16
  }
}

impl FilterSample for u32 {
  // Scalar-only `f64`-domain tier: no SIMD H-pass kernel, so the stream skips
  // the `FilterPaddedSpans` arena and the dispatcher uses the unpadded scalar
  // reduce (the same exact dot product the padded fallback would compute).
  const NEEDS_SIMD_STAGING: bool = false;
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn coeffs(axis: &FilterAxis) -> &[f32] {
    // Wide-integer (`u32`) resampler: use the full-precision coefficient set,
    // like the `u16` (PIL 32bpc `I`) path. The `f64` accumulation carries a
    // `u32` sample exactly, so the dot product is exact.
    &axis.coeffs
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn quantize_intermediate(acc: f64) -> f64 {
    // Wide-integer intermediate (the `u16` `I`-mode rule, widened to `u32`):
    // round-half-up but **do not clamp** — a negative-lobe overshoot must
    // survive into the vertical pass (clamping is final-output only). Exact:
    // a binned `u32` magnitude (`< 2^32`) is integer-valued in `f64`.
    floor_f64_local(acc + 0.5)
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn finalize(acc: f64) -> u32 {
    // Round-half-up and clamp to the full `u32` range; `u32::MAX` is exactly
    // representable in `f64`, and the saturating `as u32` cast pins the upper
    // edge.
    floor_f64_local(acc + 0.5).clamp(0.0, u32::MAX as f64) as u32
  }
}

impl FilterSample for f32 {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn coeffs(axis: &FilterAxis) -> &[f32] {
    // PIL 32bpc `F` resampler uses double coefficients; the full-precision
    // `f32` set already matches within f32 precision and must not move.
    &axis.coeffs
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn quantize_intermediate(acc: f64) -> f64 {
    // PIL 32bpc `F` resampler: the intermediate is `f32`. Narrow to f32
    // precision (no rounding, no clamp) so the vertical pass reads the same
    // value PIL's float intermediate holds.
    f64::from(acc as f32)
  }
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn finalize(acc: f64) -> f32 {
    // PIL `F`-mode does no clamp and no integer rounding — narrow only.
    acc as f32
  }
}

/// `f64` floor, gated like the kernel module's `floor_f64`.
#[cfg_attr(not(tarpaulin), inline(always))]
fn floor_f64_local(x: f64) -> f64 {
  #[cfg(feature = "std")]
  {
    f64::floor(x)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::floor(x)
  }
}

/// Streaming separable filter accumulator over [`FilterSample`] elements.
///
/// H-reduces each source row through the horizontal window set into a
/// staged `out_w * channels` row, then distributes that staged row into
/// every currently-open output accumulator whose vertical window includes
/// this source row, emitting + resetting an output row on its last
/// vertical tap. Accumulators live in a ring sized to the maximum number
/// of vertical windows that can overlap a single source row.
///
/// Source rows must arrive strictly in order from row 0 each frame;
/// [`Self::feed_row`] enforces it ([`ResampleError::OutOfSequenceRow`]),
/// matching the area stream's contract.
///
/// Gated to the families that route through it; the gate widens as
/// formats wire in.
#[derive(Debug)]
pub(crate) struct FilterStream<S: FilterSample> {
  /// Owned horizontal windows.
  h: FilterAxis,
  /// Owned vertical windows.
  v: FilterAxis,
  channels: usize,
  /// Output samples per row (`out_w * channels`).
  row_len: usize,
  /// H-reduced current source row, `row_len` wide, in the `f64`
  /// accumulation domain but **quantized to PIL's two-pass intermediate**
  /// per [`FilterSample::quantize_intermediate`] (u8 clip8, u16
  /// round-no-clamp, f32 narrow-to-f32). Quantizing this intermediate —
  /// not carrying the raw f64 sum through both passes — is what keeps the
  /// output within +-1 LSB of Pillow's two-pass result.
  h_tmp: Vec<f64>,
  /// Ring of in-flight output-row accumulators, `ring_cap * row_len`
  /// f64. Output row `oy` uses slot `oy % ring_cap`; a slot is zeroed as
  /// its row is emitted, ready for the next row that maps to it.
  ring: Vec<f64>,
  /// Plan-time SIMD staging for the H-pass
  /// ([`crate::row::FilterPaddedSpans`]); `None` routes the dispatcher to
  /// the scalar reference. Built once from the horizontal windows.
  h_padded: Option<crate::row::FilterPaddedSpans>,
  /// Number of accumulator rows in [`Self::ring`] — the maximum count of
  /// vertical windows that can overlap one source row, so an open output
  /// row never aliases another still-open one.
  ring_cap: usize,
  /// Finalized staging row handed to `emit`, `row_len` wide.
  out_tmp: Vec<S>,
  /// Next output row to finalize.
  cur_out: usize,
  /// Next source row the frame expects; rows are strictly sequential.
  next_y: usize,
}

impl<S: FilterSample> FilterStream<S> {
  /// Creates a filter stream for `channels` interleaved channels of the
  /// `h`/`v` window geometry over a `src_w x src_h` source.
  ///
  /// # Errors
  ///
  /// [`ResampleError::Overflow`] if `out_w * channels` or the ring arena
  /// length is unrepresentable; [`ResampleError::AllocationFailed`] if an
  /// arena reservation is refused (the planner's recoverable-allocation
  /// contract — output-width buffers are caller-proportional).
  pub(crate) fn new(
    h: &FilterAxis,
    v: &FilterAxis,
    src_w: usize,
    src_h: usize,
    channels: usize,
  ) -> Result<Self, ResampleError> {
    let geometry = || PlanGeometry::new(src_w, src_h, h.out_len(), v.out_len());
    let row_len = h
      .out_len()
      .checked_mul(channels)
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;
    // The ring must hold every output row whose vertical window is open
    // at once. `v.max_overlap()` is exactly that peak (computed by the
    // plan builder's window sweep), so no two open output rows ever map to
    // the same ring slot. `max(1)` keeps a degenerate zero-window plan
    // from sizing an empty ring.
    let ring_cap = v.max_overlap().max(1);
    let ring_len = ring_cap
      .checked_mul(row_len)
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;
    let alloc = |_| ResampleError::AllocationFailed(geometry());
    let h = h.try_clone()?;
    let v = v.try_clone()?;
    // The arena is an optional accelerator: a refused allocation (or an
    // unrepresentable padded length) leaves `None`, routing the H-pass to
    // the scalar reference rather than failing stream creation. It stages
    // this element's coefficient set ([`FilterSample::coeffs`]) — the `u8`
    // SIMD H-pass thus consumes the same 8bpc fixed-point coefficients its
    // scalar path does, so both stay byte-exact with PIL.
    // The SIMD H-pass staging arena is an accelerator only the SIMD-backed
    // tiers consume; the scalar-only `u32` filter tier always uses the
    // unpadded reducer, so skip building (and retaining) it for that tier.
    let h_padded = if S::NEEDS_SIMD_STAGING {
      crate::row::FilterPaddedSpans::build(&h.starts, &h.offsets, S::coeffs(&h))
    } else {
      None
    };
    Ok(Self {
      h,
      v,
      channels,
      row_len,
      h_tmp: try_zeroed_f64(row_len).map_err(alloc)?,
      ring: try_zeroed_f64(ring_len).map_err(alloc)?,
      ring_cap,
      h_padded,
      out_tmp: try_zeroed::<S>(row_len).map_err(alloc)?,
      cur_out: 0,
      next_y: 0,
    })
  }

  /// Next source row this stream expects — the sinker-level preflight
  /// checks every requested stream against the incoming row index before
  /// any stream is fed, keeping a multi-stream `process` call atomic.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn next_y(&self) -> usize {
    self.next_y
  }

  /// Restarts the stream for a new frame.
  pub(crate) fn reset(&mut self) {
    self.ring.fill(0.0);
    self.cur_out = 0;
    self.next_y = 0;
  }

  /// H-reduces source `row` (channels-interleaved, source width) into
  /// [`Self::h_tmp`]: per output sample, the signed-weighted sum of its
  /// horizontal window's source samples, **quantized to PIL's intermediate
  /// representation** ([`FilterSample::quantize_intermediate`]). This is
  /// PIL's horizontal-into-intermediate-image step; the vertical pass then
  /// reconstructs from this quantized intermediate.
  ///
  /// The signed-weighted sum runs through the runtime SIMD dispatcher
  /// ([`FilterSample::h_reduce`]) — which leaves the **raw `f64` dot
  /// product** in `h_tmp`, matching the scalar reference within the float
  /// tolerance — then this pass quantizes each in place. Because the
  /// integer dispatchers are not bit-exact, the quantized intermediate may
  /// land `±1` LSB of the scalar path, which the engine's PIL parity budget
  /// absorbs.
  fn h_reduce(&mut self, row: &[S], use_simd: bool) {
    S::h_reduce(
      row,
      self.channels,
      &self.h,
      self.h_padded.as_ref(),
      &mut self.h_tmp,
      use_simd,
    );
    for t in &mut self.h_tmp {
      *t = S::quantize_intermediate(*t);
    }
  }

  /// Feeds source row `y` (channels-interleaved, source width) and
  /// invokes `emit(out_y, finalized_row)` for every output row this
  /// source row completes. Rows beyond the plan's coverage are accepted
  /// and ignored.
  ///
  /// `use_simd` selects the SIMD H-pass when the host backend and the
  /// staging arena permit; the V-pass is element-wise (mul+add) and so is
  /// bit-equal to scalar on every backend. The flag threads through from
  /// [`AreaStream::feed_row`](super::super::AreaStream::feed_row)'s shared
  /// signature.
  ///
  /// # Errors
  ///
  /// [`ResampleError::OutOfSequenceRow`] when `y` is not the next
  /// expected source row; the stream state is untouched so the caller can
  /// resume with the expected row.
  pub(crate) fn feed_row(
    &mut self,
    y: usize,
    row: &[S],
    use_simd: bool,
    mut emit: impl FnMut(usize, &[S]),
  ) -> Result<(), ResampleError> {
    if y != self.next_y {
      return Err(ResampleError::OutOfSequenceRow(OutOfSequenceRow::new(
        self.next_y,
        y,
      )));
    }
    self.next_y += 1;
    let out_h = self.v.out_len();
    if self.cur_out >= out_h {
      return Ok(());
    }

    self.h_reduce(row, use_simd);

    // Distribute the staged H-row into every open output accumulator
    // whose vertical window contains `y`. Windows start at non-decreasing
    // `y`, so the open set is the contiguous range from `cur_out` up to
    // the first window that has not started yet.
    let mut oy = self.cur_out;
    while oy < out_h {
      let (vstart, vcoeffs) = S::v_span(&self.v, oy);
      if vstart > y {
        // This and every later window starts after `y`; none open yet.
        break;
      }
      let idx = y - vstart;
      // `vstart <= y`; if `y` is past this window's last tap the window
      // is already finished (only possible transiently at the head before
      // `cur_out` advances), so skip it.
      let Some(&w) = vcoeffs.get(idx) else {
        oy += 1;
        continue;
      };
      let slot = (oy % self.ring_cap) * self.row_len;
      let acc_row = &mut self.ring[slot..slot + self.row_len];
      S::v_accumulate(acc_row, &self.h_tmp, w, use_simd);
      oy += 1;
    }

    // Emit every output row whose last vertical tap is this `y`. They
    // finish in `cur_out` order; stop at the first row still open.
    while self.cur_out < out_h {
      let (vstart, vcoeffs) = S::v_span(&self.v, self.cur_out);
      let last = vstart + vcoeffs.len();
      // The window has not started, or has taps beyond `y` yet to arrive.
      if vstart > y || last == 0 || last - 1 != y {
        break;
      }
      let slot = (self.cur_out % self.ring_cap) * self.row_len;
      {
        let acc_row = &mut self.ring[slot..slot + self.row_len];
        for (o, a) in self.out_tmp.iter_mut().zip(acc_row.iter_mut()) {
          *o = S::finalize(*a);
          *a = 0.0;
        }
      }
      emit(self.cur_out, &self.out_tmp);
      self.cur_out += 1;
    }
    Ok(())
  }
}

/// Zeroed `f64` buffer via fallible reservation (the parent module's
/// `try_zeroed` is generic over `Clone + Default`; this is the f64 view
/// without re-exporting it across the module boundary).
fn try_zeroed_f64(n: usize) -> Result<Vec<f64>, std::collections::TryReserveError> {
  let mut buf = Vec::new();
  buf.try_reserve_exact(n)?;
  buf.resize(n, 0.0);
  Ok(buf)
}

/// Zeroed `S` buffer via fallible reservation.
fn try_zeroed<S: Copy + Default>(n: usize) -> Result<Vec<S>, std::collections::TryReserveError> {
  let mut buf = Vec::new();
  buf.try_reserve_exact(n)?;
  buf.resize(n, S::default());
  Ok(buf)
}
