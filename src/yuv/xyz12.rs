//! Tier 12 — packed CIE XYZ 12-bit source (`AV_PIX_FMT_XYZ12LE` /
//! `AV_PIX_FMT_XYZ12BE`).
//!
//! This is the only Tier 12 source format: 12-bit CIE XYZ in packed
//! `X, Y, Z` u16 triples. Used by Digital Cinema Package distribution
//! masters per SMPTE ST 428-1 *D-Cinema Distribution Master — Image
//! Characteristics*.
//!
//! Unlike every other source format in colconv, the input is **CIE
//! XYZ in a 2.6-gamma-encoded space**, not RGB or YUV. The full
//! conversion chain is:
//!
//! ```text
//! xyz_u12  →  xyz_linear (f32)  →  rgb_linear (f32) via M_xyz_to_rgb
//!         →  rgb_gamma (f32) via OETF  →  bgr_u8 / rgb_u8 / etc
//! ```
//!
//! - Step 1 (DCDM inverse-OETF): `xyz_lin = (x_u12 / 4095)^2.6 / 0.91653`
//!   per SMPTE ST 428-1 §8.
//! - Step 2 (3×3 matmul): `[R G B] = M_xyz_to_rgb · [X Y Z]`. `M`
//!   depends on the chosen target gamut — see [`DcpTargetGamut`].
//! - Step 3 (OETF — gamma encode): sRGB-shape OETF for u8 / u16
//!   integer outputs; **skipped** for the lossless `with_rgb_f32` and
//!   `with_xyz_f32` paths.
//! - Step 4 (range scale + integer narrow): `clamp(rgb_gamma, 0, 1) ×
//!   255` (or 65535) + round-half-up.
//!
//! The walker takes the target gamut as a value parameter (not a const
//! generic) — DCP-delivery target choice is a runtime decision, and
//! the 3×3 matrix is a small per-frame constant.
//!
//! ## Endianness
//!
//! `Xyz12Frame<BE>` carries the wire-format endianness as a const
//! generic; the walker forwards `BE` to the row marker so kernels can
//! const-branch on byte-swap. Type aliases [`Xyz12LeFrame`] and
//! [`Xyz12BeFrame`] cover the FFmpeg `XYZ12LE` / `XYZ12BE` variants.

use crate::{DcpTargetGamut, PixelSink, SourceFormat, frame::Xyz12Frame, sealed::Sealed};

/// Zero-sized marker type for the packed **XYZ12** source format
/// (`AV_PIX_FMT_XYZ12LE` / `AV_PIX_FMT_XYZ12BE`).
///
/// The const-generic `BE: bool` parameter selects the wire-format
/// endianness for downstream type-level reasoning. Default is `false`
/// (LE); use [`Xyz12Be`] for big-endian.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Xyz12<const BE: bool = false>;

impl<const BE: bool> Sealed for Xyz12<BE> {}
impl<const BE: bool> SourceFormat for Xyz12<BE> {}

/// Type alias for the LE marker variant. Matches `Xyz12LeFrame`.
pub type Xyz12Le = Xyz12<false>;
/// Type alias for the BE marker variant. Matches `Xyz12BeFrame`.
pub type Xyz12Be = Xyz12<true>;

/// One row of an [`Xyz12Frame`] — `width * 3` packed `u16` X/Y/Z
/// samples, each in the **high-bit-packed** layout per FFmpeg
/// `AV_PIX_FMT_XYZ12LE/BE` (active 12 bits in `[15:4]`, low 4 bits
/// reserved zero).
///
/// Carries the per-frame [`DcpTargetGamut`] choice so downstream row
/// kernels can apply the correct XYZ → RGB matrix without a separate
/// dispatch parameter. Per-target Q15 luma weights `(k_r, k_g, k_b)`
/// are also derived once at the walker call site (see
/// [`luma_weights_q15_for_gamut`]) so the `with_luma` /
/// `with_luma_u16` sinker accessors can apply the gamut-matched
/// coefficients without going through the YUV-leaning [`ColorMatrix`]
/// enum (which has no DCI-P3 entry — codex round-2 finding).
#[derive(Debug, Clone, Copy)]
pub struct Xyz12Row<'a, const BE: bool = false> {
  xyz: &'a [u16],
  row: usize,
  target_gamut: DcpTargetGamut,
  luma_q15: (i32, i32, i32),
}

impl<'a, const BE: bool> Xyz12Row<'a, BE> {
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub(crate) fn new(
    xyz: &'a [u16],
    row: usize,
    target_gamut: DcpTargetGamut,
    luma_q15: (i32, i32, i32),
  ) -> Self {
    Self {
      xyz,
      row,
      target_gamut,
      luma_q15,
    }
  }

  /// Packed source row — `width * 3` u16 samples in `X, Y, Z` order.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn xyz(&self) -> &'a [u16] {
    self.xyz
  }

  /// Output row index within the frame.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn row(&self) -> usize {
    self.row
  }

  /// Target RGB gamut chosen at the walker call site.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn target_gamut(&self) -> DcpTargetGamut {
    self.target_gamut
  }

  /// Q15 luma weights `(k_r, k_g, k_b)` matched to the target gamut.
  /// Each `k` is the corresponding RGB coefficient × 32768, rounded
  /// to nearest, with the constraint `k_r + k_g + k_b ≈ 32768`. Used
  /// by `with_luma` / `with_luma_u16` to derive Y' from gamma-encoded
  /// RGB without going through the YUV-leaning [`ColorMatrix`] enum.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn luma_q15(&self) -> (i32, i32, i32) {
    self.luma_q15
  }

  /// XYZ12 always emits full-range RGB after the OETF; the constant
  /// `true` is provided as a convenience for sinker plumbing that takes
  /// a `full_range` flag uniformly across source formats.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn full_range(&self) -> bool {
    true
  }

  /// Whether the source samples are big-endian on the wire (mirrors
  /// the const-generic parameter).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn big_endian(&self) -> bool {
    BE
  }
}

/// Maps a [`DcpTargetGamut`] to the Q15 luma coefficients
/// `(k_r, k_g, k_b)` used by the `with_luma` / `with_luma_u16` sinker
/// accessors.
///
/// Coefficients are the Y row of each gamut's `M_rgb_to_xyz` (so the
/// luma weights are exactly `(Y_red, Y_green, Y_blue)` with the
/// gamut's white point at `Y = 1`), scaled to Q15 (×32768) and rounded
/// to nearest. The triple sums to `32768 ± 1` LSB by construction.
///
/// Numbers (derived in `examples/derive_xyz_matrices.rs`):
///
/// - **Rec.709 / sRGB** (D65) — `Y = 0.2126 R + 0.7152 G + 0.0722 B`
///   → `(6966, 23436, 2366)`.
/// - **DCI-P3 (theatrical, DCI white)** —
///   `Y ≈ 0.2095 R + 0.7216 G + 0.0689 B` →
///   `(6865, 23645, 2258)`, sum = 32768. *Distinct from BT.709* —
///   codex round-2 medium finding: prior code reused the Bt709
///   triple, biasing luma values for saturated content under the
///   DCI-P3 path.
/// - **Rec.2020** (D65) — `Y = 0.2627 R + 0.6780 G + 0.0593 B`
///   → `(8607, 22217, 1944)`. (Matches `ColorMatrix::Bt2020Ncl`.)
#[cfg_attr(not(tarpaulin), inline(always))]
pub(crate) const fn luma_weights_q15_for_gamut(g: DcpTargetGamut) -> (i32, i32, i32) {
  match g {
    // Rec.709 / sRGB: Y = 0.2126 R + 0.7152 G + 0.0722 B (D65).
    DcpTargetGamut::Rec709 => (6966, 23436, 2366),
    // DCI-P3 theatrical (DCI white): Y row of the P3-DCI rgb_to_xyz
    // matrix derived in `examples/derive_xyz_matrices.rs`
    // (`Y_red = 0.2094916779`, `Y_green = 0.7215952542`,
    // `Y_blue = 0.0689130679`); each coefficient × 32768 rounded to
    // nearest gives `(6865, 23645, 2258)` with `sum = 32768` exactly.
    DcpTargetGamut::DciP3 => (6865, 23645, 2258),
    // Rec.2020: Y = 0.2627 R + 0.6780 G + 0.0593 B (D65).
    DcpTargetGamut::Rec2020 => (8607, 22217, 1944),
  }
}

/// Sinks that consume rows of an [`Xyz12`] source.
pub trait Xyz12Sink<const BE: bool = false>:
  for<'a> PixelSink<Input<'a> = Xyz12Row<'a, BE>>
{
}

/// Walks an [`Xyz12Frame`] row by row, dispatching each row to the
/// sink along with the chosen target RGB gamut.
///
/// The `target_gamut` parameter selects the XYZ → RGB matrix used at
/// every per-pixel matmul. It is a runtime value (not a const generic)
/// because the DCP delivery target is a per-frame decision; the cost
/// of the 3×3 `[[f32; 3]; 3]` indirection is amortised over the
/// per-pixel matmul + 6 `powf` calls and is unmeasurable.
///
/// The const-generic `BE: bool` parameter is taken from the frame's
/// own const generic and forwarded to the row marker so kernels can
/// const-branch on byte-swap; no runtime overhead.
pub fn xyz12_to<const BE: bool, S: Xyz12Sink<BE>>(
  src: &Xyz12Frame<'_, BE>,
  target_gamut: DcpTargetGamut,
  sink: &mut S,
) -> Result<(), S::Error> {
  sink.begin_frame(src.width(), src.height())?;

  let w = src.width() as usize;
  let h = src.height() as usize;
  let stride = src.stride() as usize;
  let row_elems: usize = w * 3;
  let plane = src.xyz();
  let luma_q15 = luma_weights_q15_for_gamut(target_gamut);

  for row in 0..h {
    let start = row * stride;
    let xyz = &plane[start..start + row_elems];
    sink.process(Xyz12Row::<BE>::new(xyz, row, target_gamut, luma_q15))?;
  }
  Ok(())
}
