//! Resampling strategies for [`MixedSinker`](crate::sinker::MixedSinker).
//!
//! A [`Resampler`] is injected at sinker construction and decides the
//! sinker's **output geometry** once, before any output buffer
//! attaches: [`Resampler::plan`] returns `Ok(None)` for the identity
//! (output geometry == source geometry — the sinker takes the direct
//! conversion path) or `Ok(Some(plan))` carrying the output geometry
//! that every output buffer is then validated against. The walker-side
//! contract is unchanged either way:
//! [`PixelSink::begin_frame`](crate::PixelSink::begin_frame) keeps
//! validating frames against the **source** geometry.
//!
//! The trait is sealed — resampling strategies ship with this crate.
//! [`NoopResampler`], the default `R` of
//! [`MixedSinker`](crate::sinker::MixedSinker), is the identity
//! strategy; [`AreaResampler`] plans area (box-coverage) downscales.
//! The streaming engine that executes non-identity plans lands with
//! the per-format route dispatch.

use std::vec::Vec;

use derive_more::{IsVariant, TryUnwrap, Unwrap};
use thiserror::Error;

/// Decides a [`MixedSinker`](crate::sinker::MixedSinker)'s output
/// geometry from its source geometry, once, at construction.
///
/// Sealed — only strategies shipped by this crate implement it.
pub trait Resampler: sealed::Sealed {
  /// Builds the resampling plan for a `src_w x src_h` source frame.
  ///
  /// `Ok(None)` means the resampling is the identity: output geometry
  /// equals source geometry and the sinker takes the direct conversion
  /// path with no resampling state at all.
  ///
  /// # Errors
  ///
  /// Strategy-specific validation of the requested output geometry —
  /// see [`ResampleError`] for the variants.
  fn plan(&self, src_w: usize, src_h: usize) -> Result<Option<ResamplePlan>, ResampleError>;
}

mod sealed {
  pub trait Sealed {}
}

/// Identity strategy and the default `R` of
/// [`MixedSinker`](crate::sinker::MixedSinker): [`Resampler::plan`]
/// always returns `Ok(None)`, so output geometry equals source
/// geometry and the sinker behaves exactly like a non-resampling sink.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoopResampler;

impl sealed::Sealed for NoopResampler {}

impl Resampler for NoopResampler {
  #[cfg_attr(not(tarpaulin), inline(always))]
  fn plan(&self, _src_w: usize, _src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
    Ok(None)
  }
}

/// Area (box-coverage) downscale strategy — the `cv2.INTER_AREA`
/// convention that analysis pipelines are calibrated against. Plans
/// exact integer coverage spans on both axes, fractional ratios
/// included (1920 -> 336 is a x40/7 scale). Requesting the source
/// geometry plans the identity (`Ok(None)`); upscaling on either axis
/// is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AreaResampler {
  out_w: usize,
  out_h: usize,
}

impl AreaResampler {
  /// Strategy producing an `out_w x out_h` output frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn to(out_w: usize, out_h: usize) -> Self {
    Self { out_w, out_h }
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

impl sealed::Sealed for AreaResampler {}

impl Resampler for AreaResampler {
  fn plan(&self, src_w: usize, src_h: usize) -> Result<Option<ResamplePlan>, ResampleError> {
    if self.out_w == 0 || self.out_h == 0 {
      return Err(ResampleError::ZeroOutputDimension(
        ZeroOutputDimension::new(self.out_w, self.out_h),
      ));
    }
    if self.out_w > src_w || self.out_h > src_h {
      return Err(ResampleError::UpscaleUnsupported(UpscaleUnsupported::new(
        src_w, src_h, self.out_w, self.out_h,
      )));
    }
    if self.out_w == src_w && self.out_h == src_h {
      return Ok(None);
    }
    ResamplePlan::area(src_w, src_h, self.out_w, self.out_h).map(Some)
  }
}

/// Zero-filled buffer via fallible reservation: `resize` after an
/// exact reserve cannot reallocate, so refusal is the only failure
/// and it surfaces as the error instead of aborting.
#[cfg(feature = "yuv-planar")]
fn try_zeroed<T: Clone + Default>(n: usize) -> Result<Vec<T>, std::collections::TryReserveError> {
  let mut buf = Vec::new();
  buf.try_reserve_exact(n)?;
  buf.resize(n, T::default());
  Ok(buf)
}

/// Greatest common divisor (Euclid); both inputs are nonzero axis
/// dimensions by the time the planner runs.
fn gcd(mut a: usize, mut b: usize) -> usize {
  while b != 0 {
    (a, b) = (b, a % b);
  }
  a
}

/// Why one axis of span planning failed; [`ResamplePlan::area`] maps
/// these onto [`ResampleError`] with the full two-axis geometry.
#[derive(Debug)]
enum AxisError {
  /// A grid product or arena length is unrepresentable.
  Overflow,
  /// An arena reservation was refused by the allocator.
  Alloc,
}

/// Per-axis area-coverage spans of a [`ResamplePlan`]: for each output
/// index, the first contributing source cell plus the integer overlap
/// weight of every contributing cell.
///
/// Geometry lives on the axis's `x out` integer grid — output pixel
/// `j` covers `[j * src, (j + 1) * src)` and source cell `i` covers
/// `[i * out, (i + 1) * out)` — so weights are exact for fractional
/// ratios and every span sums to `src`, the normalization denominator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AxisSpans {
  /// First contributing source cell per output index.
  starts: Vec<usize>,
  /// Prefix offsets into `weights`; `out_len() + 1` entries.
  offsets: Vec<usize>,
  /// Concatenated per-span overlap weights.
  weights: Vec<usize>,
}

impl AxisSpans {
  /// Exact weight-arena length for a `src -> out` area axis:
  /// `src + out - gcd(src, out)` — every source cell contributes to
  /// exactly one span, plus one shared straddle cell for each of the
  /// `out - 1` interior output boundaries that does NOT fall on a
  /// cell boundary (the aligned ones number `gcd - 1`).
  fn area_taps(src: usize, out: usize) -> Option<usize> {
    // `gcd <= min(src, out)`, so subtracting first cannot underflow
    // and keeps a representable count (e.g. usize::MAX source, one
    // output) from being misreported as overflow.
    (src - gcd(src, out)).checked_add(out)
  }

  /// Builds the exact area spans for one `src -> out` axis.
  ///
  /// Coordinate arithmetic runs in `u64` so 32-bit targets plan any
  /// geometry whose buffers fit — `usize`-sized `src` and `out` can
  /// only overflow the `src * out` grid product on 64-bit hosts. The
  /// up-front check bounds every loop term, so the loop body stays in
  /// plain arithmetic; per-cell weights are at most `out` and starts
  /// at most `src`, so the casts back to `usize` are lossless.
  ///
  /// Arena sizes are precomputed exactly ([`Self::area_taps`]) and
  /// reserved fallibly, so a hostile source dimension from untrusted
  /// metadata surfaces [`AxisError::Alloc`] instead of aborting
  /// inside infallible allocation; the pushes below never reallocate.
  fn area(src: usize, out: usize) -> Result<Self, AxisError> {
    let src64 = src as u64;
    let out64 = out as u64;
    src64.checked_mul(out64).ok_or(AxisError::Overflow)?;
    let taps = Self::area_taps(src, out).ok_or(AxisError::Overflow)?;
    let offsets_len = out.checked_add(1).ok_or(AxisError::Overflow)?;
    let mut starts = Vec::new();
    starts
      .try_reserve_exact(out)
      .map_err(|_| AxisError::Alloc)?;
    let mut offsets = Vec::new();
    offsets
      .try_reserve_exact(offsets_len)
      .map_err(|_| AxisError::Alloc)?;
    let mut weights = Vec::new();
    weights
      .try_reserve_exact(taps)
      .map_err(|_| AxisError::Alloc)?;
    offsets.push(0);
    for j in 0..out64 {
      let lo = j * src64;
      let hi = lo + src64;
      let start = lo / out64;
      starts.push(start as usize);
      for i in start..hi.div_ceil(out64) {
        let cell_lo = i * out64;
        let cell_hi = cell_lo + out64;
        weights.push((cell_hi.min(hi) - cell_lo.max(lo)) as usize);
      }
      offsets.push(weights.len());
    }
    debug_assert_eq!(weights.len(), taps);
    Ok(Self {
      starts,
      offsets,
      weights,
    })
  }

  /// Number of output samples on this axis.
  // Consumed by the area streaming engine, which is gated to the
  // families that route through it (currently yuv-planar).
  #[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn out_len(&self) -> usize {
    self.starts.len()
  }

  /// `(first source cell, overlap weights)` for output index `j`;
  // Consumed by the area streaming engine, which is gated to the
  // families that route through it (currently yuv-planar).
  #[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
  /// `j` must be below [`Self::out_len`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn span(&self, j: usize) -> (usize, &[usize]) {
    (
      self.starts[j],
      &self.weights[self.offsets[j]..self.offsets[j + 1]],
    )
  }
}

/// Output-geometry product of [`Resampler::plan`], built once at
/// sinker construction. Carries the per-axis area spans the streaming
/// engine consumes; the source dimensions double as the spans'
/// normalization denominators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResamplePlan {
  src_w: usize,
  src_h: usize,
  out_w: usize,
  out_h: usize,
  h: AxisSpans,
  v: AxisSpans,
}

impl ResamplePlan {
  /// Builds the exact area plan for `src -> out`. The strategy has
  /// already validated zero, upscale, and identity geometry.
  fn area(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Result<Self, ResampleError> {
    let fail = |e: AxisError| match e {
      AxisError::Overflow => ResampleError::Overflow(PlanGeometry::new(src_w, src_h, out_w, out_h)),
      AxisError::Alloc => {
        ResampleError::AllocationFailed(PlanGeometry::new(src_w, src_h, out_w, out_h))
      }
    };
    // Sequential on purpose: the second axis is not built when the
    // first has already failed.
    let h = AxisSpans::area(src_w, out_w).map_err(fail)?;
    let v = AxisSpans::area(src_h, out_h).map_err(fail)?;
    Ok(Self {
      src_w,
      src_h,
      out_w,
      out_h,
      h,
      v,
    })
  }

  /// Source width in pixels — the horizontal spans' normalization
  /// denominator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_w(&self) -> usize {
    self.src_w
  }

  /// Source height in pixels — the vertical spans' normalization
  /// denominator.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_h(&self) -> usize {
    self.src_h
  }

  /// Horizontal-axis spans.
  // Consumed by the area streaming engine, which is gated to the
  // families that route through it (currently yuv-planar).
  #[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn h(&self) -> &AxisSpans {
    &self.h
  }

  /// Vertical-axis spans.
  // Consumed by the area streaming engine, which is gated to the
  // families that route through it (currently yuv-planar).
  #[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn v(&self) -> &AxisSpans {
    &self.v
  }

  /// Output width in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Output height in pixels.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }

  /// Output `(width, height)` pair.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_dims(&self) -> (usize, usize) {
    (self.out_w, self.out_h)
  }
}

/// Source vs requested-output geometry payload for
/// [`ResampleError::UpscaleUnsupported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpscaleUnsupported {
  /// Source width.
  src_w: usize,
  /// Source height.
  src_h: usize,
  /// Requested output width.
  out_w: usize,
  /// Requested output height.
  out_h: usize,
}

impl UpscaleUnsupported {
  /// Constructs a new `UpscaleUnsupported` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Self {
    Self {
      src_w,
      src_h,
      out_w,
      out_h,
    }
  }

  /// Source width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_w(&self) -> usize {
    self.src_w
  }

  /// Source height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_h(&self) -> usize {
    self.src_h
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

/// Requested-output geometry payload for
/// [`ResampleError::ZeroOutputDimension`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroOutputDimension {
  /// Requested output width.
  out_w: usize,
  /// Requested output height.
  out_h: usize,
}

impl ZeroOutputDimension {
  /// Constructs a new `ZeroOutputDimension` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(out_w: usize, out_h: usize) -> Self {
    Self { out_w, out_h }
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

/// `round(a / d)` with ties rounding up, overflow-free for every
/// `u64` pair (`d > 0`): the naive `(a + d / 2) / d` wraps when `a`
/// sits near `u64::MAX`, while `r >= d - d / 2` compares the remainder
/// against `ceil(d / 2)` without any widening arithmetic.
// Consumed by the area streaming engine (gated to routed families)
// and its std-gated tests; allowed to idle in the remaining combos.
#[cfg_attr(not(feature = "yuv-planar"), allow(dead_code))]
fn round_div_half_up(a: u64, d: u64) -> u64 {
  let q = a / d;
  let r = a % d;
  q + u64::from(r >= d - d / 2)
}

/// Streaming separable area accumulator for `u8` samples: H-reduces
/// each source row through the plan's horizontal spans, accumulates it
/// under the vertical span weights, and finalizes an output row the
/// moment its last contributing source row arrives — the walker hands
/// rows in order and [`PixelSink`](crate::PixelSink) has no
/// end-of-frame hook, so emission must ride the last contribution.
///
/// Arithmetic is exact: weights are the plan's integer coverage
/// lengths, accumulation is `u64`, and the single divide per output
/// sample rounds half-up by `src_w * src_h`. Exactness makes the math
/// order-independent, which is what lets future SIMD tiers match the
/// scalar reference bit-for-bit.
///
/// Source rows must arrive strictly in order from row 0 each frame —
/// the accumulator state is meaningless otherwise — and
/// [`Self::feed_row`] enforces it, so a direct
/// [`process`](crate::PixelSink::process) caller replaying or
/// reordering rows gets an error instead of silently corrupted
/// output.
///
/// Gated to the families that route through it (currently
/// `yuv-planar`); the gate widens as formats wire in.
#[cfg(feature = "yuv-planar")]
#[derive(Debug)]
pub(crate) struct AreaStream {
  channels: usize,
  /// `src_w * src_h` — the exact normalization denominator.
  denom: u64,
  /// H-reduced current source row, `out_w * channels`.
  h_tmp: Vec<u64>,
  /// In-flight output-row accumulators, `out_w * channels`.
  acc: Vec<u64>,
  /// Finalized staging row handed to `emit`, `out_w * channels`.
  out_tmp: Vec<u8>,
  /// Next output row to finalize.
  cur_out: usize,
  /// Next source row the frame expects; rows are strictly sequential.
  next_y: usize,
}

#[cfg(feature = "yuv-planar")]
impl AreaStream {
  /// Creates a stream for `channels` interleaved channels of the
  /// plan's geometry. Fails with [`ResampleError::Overflow`] when the
  /// normalization denominator (or `denom * 255`, the accumulator
  /// bound that keeps every sum exact in `u64`) is unrepresentable.
  /// `h`/`v` are the plane's own span sets and `src_w`/`src_h` its
  /// own grid (the chroma planes of a subsampled format run smaller
  /// grids — and possibly the upsample direction — against the same
  /// output geometry).
  pub(crate) fn new(
    h: &AxisSpans,
    v: &AxisSpans,
    src_w: usize,
    src_h: usize,
    channels: usize,
  ) -> Result<Self, ResampleError> {
    let geometry = || PlanGeometry::new(src_w, src_h, h.out_len(), v.out_len());
    let denom = (src_w as u64)
      .checked_mul(src_h as u64)
      .filter(|d| *d <= u64::MAX / 255)
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;
    let n = h
      .out_len()
      .checked_mul(channels)
      .ok_or_else(|| ResampleError::Overflow(geometry()))?;
    // Row buffers follow the planner's recoverable-allocation
    // contract: output-width rows are caller-proportional, not
    // small-constant, so refusal surfaces as an error rather than an
    // abort on the first processed row.
    let alloc = |_| ResampleError::AllocationFailed(geometry());
    Ok(Self {
      channels,
      denom,
      h_tmp: try_zeroed(n).map_err(alloc)?,
      acc: try_zeroed(n).map_err(alloc)?,
      out_tmp: try_zeroed(n).map_err(alloc)?,
      cur_out: 0,
      next_y: 0,
    })
  }

  /// Next source row this stream expects — the sinker-level preflight
  /// checks every requested stream against the incoming row index
  /// before any stream is fed, keeping a multi-stream `process` call
  /// atomic.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) const fn next_y(&self) -> usize {
    self.next_y
  }

  /// Restarts the stream for a new frame.
  pub(crate) fn reset(&mut self) {
    self.acc.fill(0);
    self.cur_out = 0;
    self.next_y = 0;
  }

  /// Feeds source row `y` (`channels`-interleaved, source width) and
  /// invokes `emit(out_y, finalized_row)` for every output row this
  /// source row completes. Rows beyond the plan's coverage are
  /// accepted and ignored.
  ///
  /// # Errors
  ///
  /// [`ResampleError::OutOfSequenceRow`] when `y` is not the next
  /// expected source row; the stream state is untouched so the caller
  /// can resume with the expected row.
  pub(crate) fn feed_row(
    &mut self,
    h: &AxisSpans,
    v: &AxisSpans,
    y: usize,
    row: &[u8],
    mut emit: impl FnMut(usize, &[u8]),
  ) -> Result<(), ResampleError> {
    if y != self.next_y {
      return Err(ResampleError::OutOfSequenceRow(OutOfSequenceRow::new(
        self.next_y,
        y,
      )));
    }
    self.next_y += 1;
    if self.cur_out >= v.out_len() {
      return Ok(());
    }
    let c = self.channels;
    for j in 0..h.out_len() {
      let (start, weights) = h.span(j);
      let base = j * c;
      for ch in 0..c {
        let mut sum = 0u64;
        for (i, &w) in weights.iter().enumerate() {
          sum += w as u64 * row[(start + i) * c + ch] as u64;
        }
        self.h_tmp[base + ch] = sum;
      }
    }
    // A source row contributes to at most two output rows (a downscale
    // span covers a source cell at most twice); the loop runs the
    // second pass only when the next span starts on this same row.
    loop {
      // With rows strictly sequential, `y` always lies in the current
      // span; the two defensive exits keep the no-panic contract if
      // that invariant is ever broken by a future edit.
      let (start, weights) = v.span(self.cur_out);
      let Some(idx) = y.checked_sub(start) else {
        return Ok(());
      };
      let Some(&w) = weights.get(idx) else {
        return Ok(());
      };
      let w = w as u64;
      for (a, t) in self.acc.iter_mut().zip(self.h_tmp.iter()) {
        *a += w * *t;
      }
      if idx + 1 != weights.len() {
        return Ok(());
      }
      for (o, a) in self.out_tmp.iter_mut().zip(self.acc.iter_mut()) {
        *o = round_div_half_up(*a, self.denom) as u8;
        *a = 0;
      }
      emit(self.cur_out, &self.out_tmp);
      self.cur_out += 1;
      if self.cur_out >= v.out_len() || v.span(self.cur_out).0 != y {
        return Ok(());
      }
    }
  }
}

/// Row-sequencing payload for [`ResampleError::OutOfSequenceRow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfSequenceRow {
  /// Source row the stream expected next.
  expected: usize,
  /// Source row that was fed.
  got: usize,
}

impl OutOfSequenceRow {
  /// Constructs a new `OutOfSequenceRow` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(expected: usize, got: usize) -> Self {
    Self { expected, got }
  }

  /// Source row the stream expected next.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn expected(&self) -> usize {
    self.expected
  }

  /// Source row that was fed.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn got(&self) -> usize {
    self.got
  }
}

/// Geometry payload shared by [`ResampleError::Overflow`] and
/// [`ResampleError::AllocationFailed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanGeometry {
  /// Source width.
  src_w: usize,
  /// Source height.
  src_h: usize,
  /// Requested output width.
  out_w: usize,
  /// Requested output height.
  out_h: usize,
}

impl PlanGeometry {
  /// Constructs a new `PlanGeometry` payload.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new(src_w: usize, src_h: usize, out_w: usize, out_h: usize) -> Self {
    Self {
      src_w,
      src_h,
      out_w,
      out_h,
    }
  }

  /// Source width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_w(&self) -> usize {
    self.src_w
  }

  /// Source height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn src_h(&self) -> usize {
    self.src_h
  }

  /// Requested output width.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_w(&self) -> usize {
    self.out_w
  }

  /// Requested output height.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn out_h(&self) -> usize {
    self.out_h
  }
}

/// Errors returned by [`Resampler::plan`] while validating the
/// requested output geometry at sinker construction, and by the
/// streaming engine when a direct caller violates the row-sequencing
/// contract.
///
/// The plan-validation variants surface before the sinker exists and
/// before any output buffer attaches; the sequencing variant leaves
/// the stream untouched. All are recoverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, IsVariant, TryUnwrap, Unwrap, Error)]
#[non_exhaustive]
pub enum ResampleError {
  /// The requested output geometry exceeds the source geometry on at
  /// least one axis and the strategy only downscales.
  #[error(
    "resample output {}x{} exceeds source {}x{} (upscaling is unsupported)",
    .0.out_w(), .0.out_h(), .0.src_w(), .0.src_h()
  )]
  UpscaleUnsupported(UpscaleUnsupported),

  /// The requested output geometry has a zero side.
  #[error(
    "resample output dimensions must be nonzero, got {}x{}",
    .0.out_w(), .0.out_h()
  )]
  ZeroOutputDimension(ZeroOutputDimension),

  /// Building the span tables would overflow `usize`: a per-axis
  /// `source x output` product is unrepresentable. Only reachable
  /// with extreme dimensions (32-bit targets foremost).
  #[error(
    "resample plan geometry overflows usize: source {}x{}, output {}x{}",
    .0.src_w(), .0.src_h(), .0.out_w(), .0.out_h()
  )]
  Overflow(PlanGeometry),

  /// A span-table arena reservation was refused by the allocator. The
  /// per-axis tables are `O(source + output)` entries, so a hostile
  /// source dimension from untrusted metadata lands here as a
  /// recoverable error instead of aborting inside infallible
  /// allocation.
  #[error(
    "resample plan allocation failed: source {}x{}, output {}x{}",
    .0.src_w(), .0.src_h(), .0.out_w(), .0.out_h()
  )]
  AllocationFailed(PlanGeometry),

  /// A streaming sinker was fed a source row out of order — rows must
  /// arrive strictly sequentially from row 0 each frame. Walker-driven
  /// processing never trips this; it guards direct
  /// [`process`](crate::PixelSink::process) callers replaying or
  /// reordering rows.
  #[error(
    "resample stream fed source row {}, expected row {}",
    .0.got(), .0.expected()
  )]
  OutOfSequenceRow(OutOfSequenceRow),
}

#[cfg(all(test, feature = "std"))]
mod tests;
