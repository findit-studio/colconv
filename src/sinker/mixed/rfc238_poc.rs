//! RFC #238 — staged resampling pipeline, vertical-slice
//! proof-of-concept.
//!
//! The RFC proposes resolving "which colour domain does an area
//! downscale average in?" by modelling the convert→resample path as a
//! *staged pipeline* and splicing the resample at a chosen stage. This
//! module is the exploratory vertical slice that validates the
//! architecture against **one** format — `Yuva420p` — across the three
//! averaging domains the RFC names, each spliced at its earliest-valid
//! pipeline stage:
//!
//! | Domain | Splice stage | What is averaged |
//! |--------|--------------|------------------|
//! | `Encoded` | YUV **codes** (pre-convert) | Y, U, V, A native codes |
//! | `Linear` | linear-light **RGB** | linearised R, G, B (+ straight A) |
//! | `Premultiplied` | premultiplied **encoded** RGBA | α-weighted R, G, B (+ A) |
//!
//! (See the `AveragingDomain` enum for the per-variant detail.)
//!
//! `Yuva420p`'s convert is **affine** (a Q15 `YUV→RGB` matrix + offset +
//! clamp, no transfer step), so the Encoded and Linear domains land at
//! materially different RGB — averaging YUV codes then converting is not
//! the same as converting, linearising, averaging, and re-encoding. That
//! divergence is the point of offering the choice, and the PoC asserts
//! it.
//!
//! # The byte-identity anchor
//!
//! The `Premultiplied` domain routes straight to the **existing**
//! packed-YUVA premultiplied area tail (`packed_yuva444_resample`) — the
//! same code an `AlphaMode::Premultiplied` `Yuva420p` sink already runs by
//! default. So selecting this domain produces bytes **identical** to
//! the current behaviour, proving the strategy dispatch routes correctly
//! with zero behaviour change. This is the keystone result: a new
//! pipeline abstraction that reproduces an existing path bit-for-bit can
//! be trusted to host the new ones.
//!
//! # Standing PoC limitations (deliberate, documented for the RFC)
//!
//! - **Transfer function is an sRGB stand-in.** The Linear domain needs
//!   a YUV EOTF/OETF pair, which colconv does not yet carry (there is an
//!   sRGB-shape OETF in `row::scalar::xyz12`, but it is `xyz`-gated and
//!   has no inverse). The `TransferFunction::SRGB` constant supplies a
//!   small, self-contained sRGB curve and its exact inverse as the
//!   stand-in.
//!   The production refinement is the precise BT.709 / BT.1886 curve
//!   selected per `ColorMatrix`; the *architecture* (decode → linearise
//!   → bin → re-encode) is what this validates, and is curve-agnostic.
//! - **Frame-buffered, not streamed.** The PoC accumulates the source
//!   planes and runs the staged pipeline once the frame completes, so
//!   the new domains reuse the real area-binning stream (`AreaStream`)
//!   but skip the row-streaming bookkeeping (the 4:2:0 two-row chroma stage)
//!   that the production native tier already solves. Streaming
//!   integration is orthogonal to the domain-splice question and is left
//!   to the real implementation.
//!
//! Only `Yuva420p` is wired. No other format is touched.

use crate::{
  ColorMatrix,
  resample::{AreaStream, ResamplePlan},
  row::yuva420p_to_rgba_row,
  sinker::mixed::MixedSinkerError,
};

/// The colour domain an RFC #238 area downscale averages in. Each
/// variant names the pipeline stage the resample is spliced at; see the
/// [module docs](self) for the splice table and the affine-convert
/// rationale for why the choice is observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AveragingDomain {
  /// Average the **encoded YUV codes** before conversion: bin Y, U, V,
  /// A natively, then run the `YUV→RGBA` convert once per output pixel.
  /// Assumes straight alpha (colour and α are independent). This is the
  /// fused, libswscale-class semantics and the #235 straight-alpha
  /// native resolution.
  Encoded,
  /// Average in **linear light**: decode each source pixel to RGB,
  /// linearise via the inverse transfer function, bin the linear RGB
  /// (and bin α straight), then re-encode per output pixel. The
  /// physically-correct light-mixing domain.
  Linear,
  /// Average **premultiplied encoded** RGBA: convert at source
  /// resolution, premultiply by α, bin, then un-premultiply per output
  /// row. Routes to the existing packed-YUVA premultiplied tail — the
  /// byte-identity anchor.
  Premultiplied,
}

impl AveragingDomain {
  /// Lowercase identifier for the domain (`"encoded"` / `"linear"` /
  /// `"premultiplied"`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Encoded => "encoded",
      Self::Linear => "linear",
      Self::Premultiplied => "premultiplied",
    }
  }
}

/// A minimal opto-electronic transfer-function pair for the Linear
/// averaging domain — the **PoC stand-in** (see the [module docs](self)).
///
/// [`eotf`](Self::eotf) maps an encoded `[0, 1]` value to linear light;
/// [`oetf`](Self::oetf) is its inverse. The two are exact analytic
/// inverses, so a decode-then-re-encode round-trip with no averaging is
/// (modulo float rounding) the identity — the property the Linear path
/// relies on to stay close to the encoded path when α and chroma are
/// flat.
///
/// Only the sRGB-shape curve ([`Self::SRGB`]) is provided; production
/// would carry the per-`ColorMatrix` BT.709 / BT.1886 curve here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferFunction {
  /// Linear-segment slope below the toe threshold.
  toe_slope: f32,
  /// Encoded value at and below which the linear toe applies (the OETF
  /// breakpoint, in encoded space).
  oetf_toe: f32,
  /// Linear value at and below which the linear toe applies (the EOTF
  /// breakpoint, in linear space).
  eotf_toe: f32,
  /// Gain of the power segment.
  gain: f32,
  /// Offset of the power segment.
  offset: f32,
  /// Display gamma (the EOTF exponent); the OETF uses its reciprocal.
  gamma: f32,
}

impl TransferFunction {
  /// The sRGB transfer pair — the PoC stand-in for the Linear domain.
  ///
  /// EOTF (encoded → linear): `c <= 0.04045 ? c / 12.92 :
  /// ((c + 0.055) / 1.055)^2.4`.
  /// OETF (linear → encoded): `c <= 0.0031308 ? 12.92 * c :
  /// 1.055 * c^(1/2.4) - 0.055`.
  pub const SRGB: Self = Self {
    toe_slope: 12.92,
    oetf_toe: 0.04045,
    eotf_toe: 0.003_130_8,
    gain: 1.055,
    offset: 0.055,
    gamma: 2.4,
  };

  /// Inverse transfer (EOTF): encoded `[0, 1]` → linear light. Inputs
  /// outside `[0, 1]` extrapolate analytically (the integer narrow
  /// clamps downstream).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn eotf(self, c: f32) -> f32 {
    if c <= self.oetf_toe {
      c / self.toe_slope
    } else {
      powf32((c + self.offset) / self.gain, self.gamma)
    }
  }

  /// Forward transfer (OETF): linear light → encoded `[0, 1]`. The exact
  /// inverse of [`Self::eotf`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn oetf(self, c: f32) -> f32 {
    if c <= self.eotf_toe {
      self.toe_slope * c
    } else {
      self.gain * powf32(c, 1.0 / self.gamma) - self.offset
    }
  }
}

/// Portable `f32::powf` across `std` and `no_std + alloc`. Mirrors the
/// `row::scalar::xyz12` helper of the same name; duplicated here because
/// that one is `xyz`-gated and `yuva` does not imply `xyz`. A shared
/// transfer module would unify the two.
#[cfg_attr(not(tarpaulin), inline(always))]
fn powf32(x: f32, y: f32) -> f32 {
  #[cfg(feature = "std")]
  {
    f32::powf(x, y)
  }
  #[cfg(all(not(feature = "std"), feature = "alloc"))]
  {
    libm::powf(x, y)
  }
}

/// Per-frame source-plane accumulator for the frame-buffered PoC path
/// (see the [module docs](self)). The `Yuva420p` sink appends each
/// source row here; the staged pipeline runs once the last row lands.
///
/// `Vec` resolves to `alloc::vec::Vec` under `no_std` (the crate aliases
/// `alloc` to `std` in `lib.rs`), so this is `no_std + alloc` clean.
#[derive(Debug)]
pub struct PocFrameBuffer {
  domain: AveragingDomain,
  matrix: ColorMatrix,
  full_range: bool,
  src_w: usize,
  src_h: usize,
  y: std::vec::Vec<u8>,
  u: std::vec::Vec<u8>,
  v: std::vec::Vec<u8>,
  a: std::vec::Vec<u8>,
}

impl PocFrameBuffer {
  /// Allocates the per-frame plane buffers for an `src_w x src_h`
  /// `Yuva420p` frame (chroma half-resolution each way).
  fn new(
    domain: AveragingDomain,
    matrix: ColorMatrix,
    full_range: bool,
    src_w: usize,
    src_h: usize,
  ) -> Self {
    let luma = src_w * src_h;
    let chroma = (src_w / 2) * (src_h / 2);
    Self {
      domain,
      matrix,
      full_range,
      src_w,
      src_h,
      y: vec_zeroed(luma),
      u: vec_zeroed(chroma),
      v: vec_zeroed(chroma),
      a: vec_zeroed(luma),
    }
  }
}

/// `alloc`-portable zeroed `u8` vector for the plane buffers.
fn vec_zeroed(n: usize) -> std::vec::Vec<u8> {
  std::vec![0u8; n]
}

/// Entry point for the `Yuva420p` RFC #238 PoC resample. Called per
/// source row; buffers the planes and runs the staged pipeline at frame
/// completion. The `domain` selects the splice stage; outputs are the
/// caller's straight-RGBA buffer (the single output the PoC wires).
///
/// Frame-buffered rather than streamed (a PoC simplification — see the
/// [module docs](self)).
#[allow(clippy::too_many_arguments)]
pub(super) fn yuva420p_poc_resample(
  domain: AveragingDomain,
  frame: &mut Option<PocFrameBuffer>,
  rgba: &mut Option<&mut [u8]>,
  plan: &ResamplePlan,
  y_row: &[u8],
  u_half: &[u8],
  v_half: &[u8],
  a_row: &[u8],
  matrix: ColorMatrix,
  full_range: bool,
  idx: usize,
  w: usize,
  h: usize,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  // Lazily create the frame buffer on row 0; reset between frames is the
  // sink's `begin_frame` clearing `frame` to `None`.
  let buf = frame.get_or_insert_with(|| PocFrameBuffer::new(domain, matrix, full_range, w, h));
  // Append this source row into the plane buffers.
  let cw = w / 2;
  buf.y[idx * w..(idx + 1) * w].copy_from_slice(&y_row[..w]);
  buf.a[idx * w..(idx + 1) * w].copy_from_slice(&a_row[..w]);
  if idx.is_multiple_of(2) {
    let cidx = idx / 2;
    buf.u[cidx * cw..(cidx + 1) * cw].copy_from_slice(&u_half[..cw]);
    buf.v[cidx * cw..(cidx + 1) * cw].copy_from_slice(&v_half[..cw]);
  }

  // Run the staged pipeline only once the whole frame is buffered.
  if idx + 1 != h {
    return Ok(());
  }
  let Some(out) = rgba.as_deref_mut() else {
    return Ok(());
  };
  match buf.domain {
    AveragingDomain::Encoded => encoded_domain(buf, out, plan, use_simd),
    AveragingDomain::Linear => linear_domain(buf, out, plan, use_simd),
    // Premultiplied is routed to the existing tail by the sink before
    // this entry point; reaching it here is a contract bug.
    AveragingDomain::Premultiplied => {
      unreachable!("Premultiplied routes to packed_yuva444_resample, not the PoC frame path")
    }
  }
}

/// **Encoded domain** — bin the native Y / U / V / A codes, then convert
/// once per output pixel. Y and A bin on the luma grid (`plan`); U and V
/// bin on their own chroma grid (to output-chroma resolution). The binned
/// planes form a `Yuva420p` frame at output geometry, which
/// `yuva420p_to_rgba_row` converts to straight RGBA. Straight alpha:
/// colour and α bin independently.
fn encoded_domain(
  buf: &PocFrameBuffer,
  out: &mut [u8],
  plan: &ResamplePlan,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let ow = plan.out_w();
  let oh = plan.out_h();
  let cw = buf.src_w / 2;
  let ch = buf.src_h / 2;
  let ocw = ow / 2;
  let och = oh / 2;

  // Bin each native plane INDEPENDENTLY at its own resolution: Y and A on
  // the luma grid (`plan`), U and V on their own chroma grid
  // (`cw x ch -> ocw x och`). The binned planes form a `Yuva420p` frame
  // at output geometry, which the convert below upsamples + colour-mixes.
  // (This is the "downscale the YUVA frame as a YUVA frame" model: the
  // chroma bins to output-chroma resolution, distinct from the native
  // RGB-convert tier's `area_chroma_420` which bins chroma straight to
  // output-luma res because it converts to 4:4:4 RGB inline.)
  let y_out = bin_plane_u8(&buf.y, plan, buf.src_w, buf.src_h, use_simd)?;
  let a_out = bin_plane_u8(&buf.a, plan, buf.src_w, buf.src_h, use_simd)?;
  let cplan = ResamplePlan::area(cw, ch, ocw, och)?;
  let u_out = bin_plane_u8(&buf.u, &cplan, cw, ch, use_simd)?;
  let v_out = bin_plane_u8(&buf.v, &cplan, cw, ch, use_simd)?;

  // Convert the downscaled YUVA frame to straight RGBA, one output row at
  // a time. The chroma row index is `oy / 2` (4:2:0 at output res).
  for oy in 0..oh {
    let cy = oy / 2;
    yuva420p_to_rgba_row(
      &y_out[oy * ow..(oy + 1) * ow],
      &u_out[cy * ocw..(cy + 1) * ocw],
      &v_out[cy * ocw..(cy + 1) * ocw],
      &a_out[oy * ow..(oy + 1) * ow],
      &mut out[oy * 4 * ow..(oy + 1) * 4 * ow],
      ow,
      buf.matrix,
      buf.full_range,
      use_simd,
    );
  }
  Ok(())
}

/// **Linear domain** — decode each source pixel to RGB, linearise, bin
/// the linear RGB (and bin α straight), then re-encode per output pixel.
///
/// Decode uses the full-resolution `YUV→RGBA` convert (4:2:0 chroma
/// upsampled, real source α), so the linear bin runs on per-source-pixel
/// RGB. The RGB and α are binned on the luma grid; α never enters the
/// linearisation (it is straight). Uses [`TransferFunction::SRGB`] as the
/// PoC stand-in.
fn linear_domain(
  buf: &PocFrameBuffer,
  out: &mut [u8],
  plan: &ResamplePlan,
  use_simd: bool,
) -> Result<(), MixedSinkerError> {
  let tf = TransferFunction::SRGB;
  let n = buf.src_w * buf.src_h;

  // Decode the whole source frame to full-resolution straight RGBA, then
  // split into a linear-light RGB plane (3ch f32) and a straight α plane
  // (1ch u8).
  let mut rgba = vec_zeroed(n * 4);
  decode_full_res_rgba(buf, &mut rgba, use_simd);
  let mut lin_rgb = std::vec![0f32; n * 3];
  let mut alpha = vec_zeroed(n);
  for (px, (lin, a)) in rgba
    .chunks_exact(4)
    .zip(lin_rgb.chunks_exact_mut(3).zip(alpha.iter_mut()))
  {
    for c in 0..3 {
      lin[c] = tf.eotf(px[c] as f32 / 255.0);
    }
    *a = px[3];
  }

  // Bin the linear RGB (f32, exact mean) and the straight α (u8) on the
  // luma grid.
  let lin_binned = bin_plane_f32(&lin_rgb, 3, plan, buf.src_w, buf.src_h, use_simd)?;
  let a_binned = bin_plane_u8(&alpha, plan, buf.src_w, buf.src_h, use_simd)?;

  // Re-encode the binned linear RGB per output pixel; α passes through.
  debug_assert_eq!(a_binned.len(), plan.out_w() * plan.out_h());
  for (out_px, (lin, &a)) in out
    .chunks_exact_mut(4)
    .zip(lin_binned.chunks_exact(3).zip(a_binned.iter()))
  {
    for c in 0..3 {
      let enc = tf.oetf(lin[c]) * 255.0 + 0.5;
      out_px[c] = enc.clamp(0.0, 255.0) as u8;
    }
    out_px[3] = a;
  }
  Ok(())
}

/// Decodes the buffered `Yuva420p` source frame to full-resolution
/// straight RGBA via the production `yuva420p_to_rgba_row` kernel (4:2:0
/// chroma upsampled, real source α).
fn decode_full_res_rgba(buf: &PocFrameBuffer, rgba: &mut [u8], use_simd: bool) {
  let w = buf.src_w;
  let cw = w / 2;
  for sy in 0..buf.src_h {
    let cy = sy / 2;
    yuva420p_to_rgba_row(
      &buf.y[sy * w..(sy + 1) * w],
      &buf.u[cy * cw..(cy + 1) * cw],
      &buf.v[cy * cw..(cy + 1) * cw],
      &buf.a[sy * w..(sy + 1) * w],
      &mut rgba[sy * 4 * w..(sy + 1) * 4 * w],
      w,
      buf.matrix,
      buf.full_range,
      use_simd,
    );
  }
}

/// Bins one single-channel `u8` plane to output resolution through the
/// real `AreaStream<u8>`. `plan` carries the plane's own grid
/// (`src_w x src_h` are the plane's dims and the normalization
/// denominator). Returns an owned output-resolution plane.
fn bin_plane_u8(
  plane: &[u8],
  plan: &ResamplePlan,
  src_w: usize,
  src_h: usize,
  use_simd: bool,
) -> Result<std::vec::Vec<u8>, MixedSinkerError> {
  let ow = plan.out_w();
  let oh = plan.out_h();
  let mut stream = AreaStream::<u8>::new(plan.h(), plan.v(), src_w, src_h, 1)?;
  let mut out = vec_zeroed(ow * oh);
  for sy in 0..src_h {
    stream.feed_row(
      sy,
      &plane[sy * src_w..(sy + 1) * src_w],
      use_simd,
      |oy, row| {
        out[oy * ow..(oy + 1) * ow].copy_from_slice(row);
      },
    )?;
  }
  Ok(out)
}

/// Bins one `channels`-interleaved `f32` plane to output resolution
/// through the real `AreaStream<f32>` (the exact-mean float bin).
/// Returns an owned output-resolution interleaved plane.
fn bin_plane_f32(
  plane: &[f32],
  channels: usize,
  plan: &ResamplePlan,
  src_w: usize,
  src_h: usize,
  use_simd: bool,
) -> Result<std::vec::Vec<f32>, MixedSinkerError> {
  let ow = plan.out_w();
  let oh = plan.out_h();
  let mut stream = AreaStream::<f32>::new(plan.h(), plan.v(), src_w, src_h, channels)?;
  let mut out = std::vec![0f32; ow * oh * channels];
  let stride = src_w * channels;
  for sy in 0..src_h {
    stream.feed_row(
      sy,
      &plane[sy * stride..(sy + 1) * stride],
      use_simd,
      |oy, row| {
        let dst = oy * ow * channels;
        out[dst..dst + ow * channels].copy_from_slice(row);
      },
    )?;
  }
  Ok(out)
}
