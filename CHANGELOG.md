# CHANGELOG

## Unreleased — Tier 15 — Mono1bit (monoblack/monowhite) source formats

Closes Tier 15. Two new source-side pixel formats:
- `Monoblack` (`AV_PIX_FMT_MONOBLACK`) — 1-bit-per-pixel, MSB first, bit=0=black
- `Monowhite` (`AV_PIX_FMT_MONOWHITE`) — same byte layout, INVERTED polarity

Const-generic `MonoFrame<const INVERT: bool>` with type aliases for the two
polarities. Native SIMD across all 5 backends (NEON, SSE4.1, AVX2, AVX-512,
wasm-simd128). 16-64 px / iter via bit-mask broadcast + test-nonzero pattern.
Each format exposes 7 sinker accessors: with_rgb, with_rgba, with_rgb_u16,
with_rgba_u16, with_luma, with_luma_u16, with_hsv. ~1730+ tests.

**Fix (codex high):** u16 outputs (`with_luma_u16`, `with_rgb_u16`, `with_rgba_u16`)
now zero-extend the 8-bit luma value to u16 — white maps to 0x00FF, not 0xFFFF.
This matches Gray8's `with_luma_u16` / `with_rgb_u16` / `with_rgba_u16` contract
(zero-extend computed 8-bit luma rather than rescaling). Alpha in `with_rgba_u16`
is also zero-extended (0x00FF), consistent with a `0xFF` u8 alpha zero-extended to
u16. Callers mixing Monoblack/Monowhite with other 8-bit sources will see consistent
white luma (255) across all formats in u16 outputs.

## Unreleased — Tier 7 — Legacy low-bit packed RGB source formats

Closes Tier 7. Six new source-side pixel formats covering FFmpeg's legacy
16-bits-per-pixel packed RGB family:

- **Rgb565** (`AV_PIX_FMT_RGB565LE`) — R5 G6 B5 packed in u16
- **Bgr565** (`AV_PIX_FMT_BGR565LE`) — B5 G6 R5 packed in u16
- **Rgb555** (`AV_PIX_FMT_RGB555LE`) — 1 padding bit + R5 G5 B5
- **Bgr555** (`AV_PIX_FMT_BGR555LE`) — 1 padding bit + B5 G5 R5
- **Rgb444** (`AV_PIX_FMT_RGB444LE`) — 4 padding bits + R4 G4 B4
- **Bgr444** (`AV_PIX_FMT_BGR444LE`) — 4 padding bits + B4 G4 R4

Each format exposes 7 sinker accessors. Channel expansion via bit-replication
(5-bit: `(c << 3) | (c >> 2)`; 6-bit: `(c << 2) | (c >> 4)`; 4-bit: `(c << 4) | c`)
preserves 0→0 and max→255 mapping. Native SIMD across all 5 backends
(NEON 8 px/iter, SSE4.1 8 px/iter, AVX2 16 px/iter, AVX-512 32 px/iter,
wasm-simd128 8 px/iter) — bit-shift + mask + replicate pattern translates
directly across architectures. ~223 new tests: 36 frame validation, 24 scalar
correctness, 24 NEON / 24 SSE4.1 / 24 AVX2 / 24 AVX-512 / 24 wasm parity tests
(SIMD tests gated by appropriate feature detection / arch cfg), plus 43
sinker integration tests with SIMD-vs-scalar parity, alpha-forcing semantics,
buffer-too-short error paths, and channel-order verification.


## Unreleased — Tier 13 — Pal8 (palette) source format

Closes Tier 13. New source-side pixel format `Pal8` (`AV_PIX_FMT_PAL8`):
8-bit indexed-color with 256-entry BGRA palette, MAM completeness path.

- Frame: `Pal8Frame` carries `&[u8]` index buffer + `&[[u8; 4]; 256]` palette
  (FFmpeg's BGRA palette layout); width/height/stride validated; checked-mul
  32-bit overflow guard
- Walker: `pal8_to<S: Pal8Sink>` in `src/raw/` parallel to `bayer_to`
- Scalar kernels: 4 per-pixel palette-lookup kernels (rgb, rgba, rgb_u16,
  rgba_u16) with BGRA→RGB reorder + full-range u16 widening
- NEON SIMD backend (`src/row/arch/neon/pal8.rs`): hybrid scalar-gather +
  NEON deinterleave/store strategy. For each 16-pixel block: scalar gather of
  64 bytes into a stack buffer (reordering BGRA→RGBA), then `vld4q_u8` splits
  4 channels for free, `vst3q_u8`/`vst4q_u8` stores them interleaved without
  a temporary. For u16 output: `vmovl_u8` + `vshlq_n_u16`/`vorrq_u16` apply
  the `(v << 8) | v` full-range widening in SIMD. Scalar tail for `width % 16`.
- Dispatcher (`src/row/dispatch/pal8.rs`) updated: aarch64 paths select the
  NEON backend via `neon_available()` when `use_simd=true`.
- Criterion benchmark (`benches/pal8_simd.rs`) comparing scalar vs. NEON for
  all 4 kernels at widths 256, 1280, 1920. Results on Apple M-series aarch64:

  | kernel           | width | scalar      | NEON        | speedup |
  |------------------|-------|-------------|-------------|---------|
  | pal8_to_rgb      |   256 |   139.7 ns  |   123.5 ns  |  1.13×  |
  | pal8_to_rgb      |  1280 |   667.6 ns  |   490.5 ns  |  1.36×  |
  | pal8_to_rgb      |  1920 |   985.7 ns  |   744.6 ns  |  1.32×  |
  | pal8_to_rgba     |   256 |   165.0 ns  |    83.7 ns  |  1.97×  |
  | pal8_to_rgba     |  1280 |   794.2 ns  |   404.9 ns  |  1.96×  |
  | pal8_to_rgba     |  1920 |  1197.8 ns  |   608.5 ns  |  1.97×  |
  | pal8_to_rgb_u16  |   256 |   152.1 ns  |    83.3 ns  |  1.83×  |
  | pal8_to_rgb_u16  |  1280 |   735.9 ns  |   404.0 ns  |  1.82×  |
  | pal8_to_rgb_u16  |  1920 |  1085.2 ns  |   601.6 ns  |  1.80×  |
  | pal8_to_rgba_u16 |   256 |   180.7 ns  |   133.0 ns  |  1.36×  |
  | pal8_to_rgba_u16 |  1280 |   896.9 ns  |   680.5 ns  |  1.32×  |
  | pal8_to_rgba_u16 |  1920 |  1342.5 ns  |   999.5 ns  |  1.34×  |

  Analysis: SIMD wins 1.3–2.0× across all kernels. The RGBA kernels benefit
  most (~2×) because `vst4q_u8` eliminates 4 independent byte-stride writes
  that the scalar loop serializes. The gather itself is scalar and is the
  primary bottleneck; the SIMD benefit comes entirely from cheaper
  deinterleave + interleaved store, not from vectorized LUT access. The full
  4-bank TBL approach would eliminate the stack buffer but add 16 `vqtbl4q_u8`
  + select-merge operations per block — unlikely to win given the gather is
  only ~50% of total cycle budget at 1920px width.
- Sinker: `MixedSinker<'_, Pal8>` with 7 accessors. Strategy A+ for
  rgb+rgba combos (single palette lookup, RGB stripped from RGBA buffer).
- 36 new tests: 6 frame validation + 5 scalar inline + 13 sinker integration
  + 12 NEON parity/semantic tests (boundary widths 1, 8, 15, 16, 17, 32, 33,
  128, 130)

## Unreleased — Tier 10 float — Gbrpf32 / Gbrapf32 / Gbrpf16 / Gbrapf16 source formats

Closes Tier 10's float family (P1 VFX priority — EXR / 30-60% format share).
Four new source-side pixel formats end-to-end:

- **Gbrpf32** (`AV_PIX_FMT_GBRPF32LE`) — planar G/B/R f32, no source α
- **Gbrapf32** (`AV_PIX_FMT_GBRAPF32LE`) — planar G/B/R/A f32, source α
- **Gbrpf16** (`AV_PIX_FMT_GBRPF16LE`) — planar G/B/R `half::f16`, no source α
- **Gbrapf16** (`AV_PIX_FMT_GBRAPF16LE`) — planar G/B/R/A `half::f16`, source α

Each format exposes 12 sinker accessors. Native SIMD across all 5 backends
(NEON, SSE4.1, AVX2, AVX-512, wasm-simd128). f32 → f16 narrowing uses F16C
on x86 (`is_x86_feature_detected!("f16c")`) and NEON `fp16` on aarch64
(`is_aarch64_feature_detected!("fp16")`); scalar `half::f16` fallback on wasm
and feature-absent paths.

Rounding contracts: round-half-up for f32 → u8 / u16; IEEE-754
round-to-nearest-even for f32 → f16 (F16C `_MM_FROUND_TO_NEAREST_INT`).

α handling for gbrap*: Strategy A+ — chroma kernel runs once → expand to
RGBA → α-extract scatters source α into slot 3. New helpers:
`copy_alpha_plane_f32_to_u8`, `copy_alpha_plane_f32_to_u16`,
`copy_alpha_plane_f32`, plus `copy_alpha_plane_f16` for f16-native paths.

~85 new tests: per-format scalar correctness (~24), per-backend SIMD
parity (5 × 8 kernels = 40), round-half-up regression (5), cross-format
planar parity (6), Strategy A+ byte-equivalence (4), HDR pass-through (4),
32-bit dispatcher overflow guards (4).

## Unreleased — Tier 8 finish — Rgb48 / Bgr48 / Rgba64 / Bgra64 source formats

Closes Tier 8's high-bit-depth packed RGB family (P1 mastering — DPX / film
scan / 16-bit RGB pipelines). Four new source-side pixel formats:

- **Rgb48** (`AV_PIX_FMT_RGB48LE`) — 16-bit packed R/G/B, no source α
- **Bgr48** (`AV_PIX_FMT_BGR48LE`) — 16-bit packed B/G/R, no source α
- **Rgba64** (`AV_PIX_FMT_RGBA64LE`) — 16-bit packed R/G/B/A, source α
- **Bgra64** (`AV_PIX_FMT_BGRA64LE`) — 16-bit packed B/G/R/A, source α

Each format exposes 7 sinker accessors (`with_rgb`, `with_rgba`,
`with_rgb_u16`, `with_rgba_u16`, `with_luma`, `with_luma_u16`, `with_hsv`).
Native SIMD across all 5 backends (NEON 8 px/iter via `vld3q_u16`/`vld4q_u16`,
SSE4.1 8 px/iter, AVX2 16 px/iter via `_mm256_permute4x64_epi64` lane fix,
AVX-512 32 px/iter, wasm-simd128 8 px/iter).

α handling for Rgba64/Bgra64: Strategy A+ — single deinterleave pass +
`expand_rgb_to_rgba_row` + α-extract via new helpers
`copy_alpha_packed_u16x4_to_u8_at_3` and `copy_alpha_packed_u16x4_at_3`.
Standalone fast path for alpha-only callers.

~120 new tests: per-format scalar correctness, SIMD-vs-scalar parity across
all 5 backends, lane-order regression (asymmetric R/G/B/A inputs), Strategy
A+ byte-equivalence, frame validation (WidthOverflow + GeometryOverflow),
sinker integration with row shape mismatch errors.

## 0.23.0 — Tier 11 finish: Grayf32 / Ya8 / Ya16 source formats

**Additive feature; no public API change for existing callers.**

Completes Tier 11 by adding full `MixedSinker` support for three new source
formats: `Grayf32` (32-bit float luma), `Ya8` (8-bit luma + alpha), and
`Ya16` (16-bit luma + alpha). Each format exposes RGB, RGBA, RGB-u16,
RGBA-u16, luma, luma-u16, and HSV outputs via the same builder/accessor
pattern used by all other formats.

Key implementation details:

- **Grayf32 output model**: clamp `[0, 1]`, scale × 255 (u8) or × 65535
  (u16) with MXCSR-independent rounding; lossless f32 pass-throughs
  (`rgb_f32`, `luma_f32`) added to `MixedSinker` struct.
- **Ya8 / Ya16 RGBA**: Strategy A+ — RGB kernel first, then
  `copy_alpha_ya_{u8,u16}` patches the α channel in-place, avoiding
  double per-pixel math when both RGB and RGBA buffers are requested.
- **Gray fast-path HSV**: H = 0, S = 0, V = Y (no RGB→HSV path) for all
  three formats.
- **SIMD coverage**: 5-backend SIMD (NEON, SSE4.1, AVX2, AVX-512,
  wasm-simd128) dispatchers wired through `src/row/dispatch/`.
- **New fields** on `MixedSinker`: `rgb_f32`, `luma_f32` with
  `with_*/set_*` accessors and corresponding buffer-too-short error variants.

Depends on PR #71 (Tier 11 MVP), PR #72 (Rgbf16), and PR #73 (Tier 10b)
merging first.

## 0.22.0 — Tier 10b — Gbrp{9,10,12,14,16} + Gbrap{10,12,14,16} high-bit-depth planar GBR

**Additive feature; no public API change for existing source formats.**

### Added

- `Gbrp9`, `Gbrp10`, `Gbrp12`, `Gbrp14`, `Gbrp16` — planar GBR high-bit-depth
  formats (`AV_PIX_FMT_GBRP{9,10,12,14,16}LE`) with native u16 planes. Accessors:
  `with_rgb`, `with_rgb_u16`, `with_rgba`, `with_rgba_u16`, `with_luma`,
  `with_luma_u16`, `with_hsv`.
- `Gbrap10`, `Gbrap12`, `Gbrap14`, `Gbrap16` — planar GBR+A
  high-bit-depth formats (`AV_PIX_FMT_GBRAP{10,12,14,16}LE`) with source alpha
  plane. (FFmpeg has no 9-bit Gbrap variant.) Same accessors; alpha sourced from the A plane at native depth. Strategy
  A+ combo path for simultaneous RGB + RGBA output.
- 5 SIMD backends (NEON, SSE4.1, AVX2, AVX-512, wasm_simd128) for all new
  high-bit kernels. Each backend is byte-identical to the scalar reference for
  these pure-shuffle / right-shift kernels.

## 0.21.0 — Tier 9 finish — Rgbf16 packed half-precision float RGB source

**Additive feature; no public API change for existing source formats.**

One new source format:

- `Rgbf16` (`AV_PIX_FMT_RGBF16`) — packed `R, G, B` 16-bit half-precision float.
  Linear-RGB convention; HDR > 1.0 saturates on integer outputs and is preserved
  bit-exact on `with_rgb_f16` / `with_rgb_f32`.

Architecture:
- f16 → f32 widening at row entry via hardware (`vcvt_f32_f16` on AArch64,
  `_mm{,256,512}_cvtph_ps` on x86 + F16C) or scalar `half::f16::to_f32`.
- All downstream conversions (u8/u16/f32) reuse the existing `rgbf32_to_*_row`
  kernels — zero math duplication.

Accessors on `MixedSinker<'_, Rgbf16>`:
- `with_rgb` / `with_rgba` (clamp [0,1] × 255 → u8)
- `with_rgb_u16` / `with_rgba_u16` (clamp [0,1] × 65535 → u16; full-range scaling
  — same divergence-from-integer-source-family note as `Rgbf32`)
- `with_rgb_f16` (NEW — lossless half-float pass-through)
- `with_rgb_f32` (lossless widening)
- `with_luma` / `with_luma_u16` / `with_hsv` (staged through u8 RGB scratch)

Test coverage:
- 9 frame constructor tests (mirroring Rgbf32Frame).
- 6 scalar parity tests (rgbf16 vs rgbf32-after-widen, bit-exact).
- 6 SIMD scalar-equivalence tests per backend × 5 backends = 30 SIMD tests.
- Sinker integration tests mirroring the Rgbf32 sinker pattern.

F16C runtime detection on x86 with scalar fallback when F16C is unavailable
(very rare on shipping hardware — Ivy Bridge 2012+ has F16C).

## 0.20.0 — Tier 10 — Gbrp + Gbrap planar GBR sources (8-bit MVP)

**Additive feature; no public API change for existing source formats.**

Two new source formats:

- `Gbrp` (`AV_PIX_FMT_GBRP`) — three full-resolution u8 planes in
  **G, B, R** order (FFmpeg convention).
- `Gbrap` (`AV_PIX_FMT_GBRAP`) — four planes (G, B, R, A) at 8 bits,
  with a real per-pixel alpha plane (1:1 with the colour planes).

`MixedSinker` impls expose the standard `with_rgb` / `with_rgba` /
`with_luma` / `with_luma_u16` / `with_hsv` channels:

- **`with_rgb`** — interleaves the three planes into packed `R, G, B`
  via the new `gbr_to_rgb_row` kernel. Output is identical to running
  the same pixel data through `Rgb24` (verified by parity tests).
- **`with_rgba`** — for `Gbrp`: standalone path uses
  `gbr_to_rgba_opaque_row` (α = `0xFF`). For `Gbrap`: standalone uses
  `gbra_to_rgba_row` (real α from the source A plane). The combo
  case `with_rgb + with_rgba` runs Strategy A (Gbrp) / A+ (Gbrap) —
  expand the already-computed RGB row to RGBA and (Gbrap) overwrite
  the α slot from the source plane, avoiding a second per-pixel
  interleave.
- **`with_luma` / `with_luma_u16`** — derive luma from staged packed
  RGB via the existing `rgb_to_luma_row` (no new luma kernel).
- **`with_hsv`** — derive HSV from staged packed RGB via the existing
  `rgb_to_hsv_row`.

Per-arch SIMD on every backend (NEON / SSE4.1 / AVX2 / AVX-512 /
wasm-simd128) for the three new kernels (`gbr_to_rgb_row`,
`gbra_to_rgba_row`, `gbr_to_rgba_opaque_row`). Each backend is
byte-identical to the scalar reference (these kernels are pure
shuffles — no rounding involved).

The `walker!` macro is reused unchanged via thin Y/U/V → G/B/R alias
methods on `GbrpFrame` / `GbrapFrame` (crate-private) plus accessor
re-naming on the row types.

**Out of scope (deferred to Tier 10b):**

- High-bit GBR (`Gbrp9` / `Gbrp10` / `Gbrp12` / `Gbrp14` / `Gbrp16`,
  `Gbrap10` / `Gbrap12` / `Gbrap14` / `Gbrap16`) — same shape, will
  follow the BITS-generic `YuvNp` template.
- Float planar (`gbrpf32`, `gbrapf16`) — folded into the Tier 9 float
  follow-up.

## 0.19.0 — `with_luma_u16` for all 8-bit source formats

**Additive feature; no public API change for existing accessors.**

All 14 8-bit source formats now uniformly support `with_luma_u16` /
`set_luma_u16` accessors, matching the convention already established
by 16-bit and high-bit formats (Y216, XV36, AYUV64, Yuva*p9/10/12/14/16,
etc.). Output is mechanically `out[x] = src_y_byte[x] as u16` —
zero-extension at the type level, no shift, no scaling.

Formats wired:

- **Planar:** Yuv420p, Yuv422p, Yuv440p, Yuv444p
- **Semi-planar:** Nv12, Nv16, Nv21, Nv24, Nv42
- **Packed YUV 4:2:2:** Yuyv422, Uyvy422, Yvyu422
- **Packed YUV 4:4:4:** Vuya, Vuyx

The 9 planar/semi-planar formats share a single
`y_plane_to_luma_u16_row` kernel (Y is a contiguous u8 plane in all of
them); the 5 packed formats each get their own format-specific kernel
that gathers Y from the right byte offset (`packed[x*2]` / `packed[x*2+1]`
for 4:2:2, `packed[x*4 + 2]` for 4:4:4). All kernels are scalar +
5-backend SIMD (NEON, SSE4.1, AVX2, AVX-512, wasm-simd128) with byte-
identical scalar-equivalence per backend.

This closes a long-standing asymmetry where downstream consumers
operating on `&mut [u16]` luma (scene-detect, bilateral filters,
percentile/histogram work) had to special-case 8-bit sources or drop
u16 luma support entirely. Surfaced from Ship 12c plan-time discussion.

## 0.18.0 — Strategy A+ across all source-α formats (Tier 5 closure megaship PR 4 / final)

**Performance improvement; no public API change; output byte-identical to v0.17.x.**

For source-α formats (VUYA, AYUV64, Yuva420p / 422p / 444p, Yuva*p9 / 10 / 12 / 14 / 16),
the `with_rgb + with_rgba` combo case previously ran the chroma kernel
TWICE — once for RGB output, once for RGBA-with-source-α. Strategy A+
now runs the chroma kernel ONCE, expands RGB → RGBA via the existing
`expand_rgb_to_rgba_row` helper, then overwrites the α slot with a cheap
α-extract pass from the source. Cost: 1× chroma + 1× expand + 1× α-overwrite
instead of 2× chroma. Impact is largest for 16-bit (u16) outputs, where
the chroma kernel uses i64 arithmetic (Q15 sums overflow i32 at 16-bit
input) — that kernel now runs once instead of twice.

The standalone `with_rgba`-only path is unchanged — it already runs a
single inline-α kernel, which remains optimal.

New module: `src/row/scalar/alpha_extract.rs` with 6 helpers, plus per-arch
SIMD parity in `src/row/arch/{neon,x86_sse41,x86_avx2,x86_avx512,wasm_simd128}/alpha_extract.rs`
(30 SIMD impls total) plus the runtime dispatcher
`src/row/dispatch/alpha_extract.rs`.

Output is byte-identical to v0.17.x — verified by 30 per-format A+
correctness tests asserting `sinker(combo) == inline-α-kernel-direct`
for every (range, ColorMatrix) combination.

This is **PR 4 of the 4-PR Tier 5 closure megaship — the final PR**:
- ✅ PR 1 (v0.16.0): AYUV64 + Tier 5 closure
- ✅ PR 2 (v0.16.1): Multi-channel lane-order test backport
- ✅ PR 3 (v0.17.0): 8-bit planar scale-constant migration
- ✅ PR 4 (this release): Strategy A+ across source-α formats — **closes the megaship**

After this release the source-α format family shares a uniform combo-case
optimization with the rest of the wired families. The Tier 5 (packed
YUV 4:4:4 + α) family is now feature-complete and performance-optimized.

## 0.17.0 — 8-bit planar scale-constant migration (Tier 5 closure megaship PR 3)

**Behavior change at limited-range:** all 8-bit YUV→RGB paths now use
`range_params_n::<8, 8>(...)` for scale constants, replacing the legacy
hardcoded `range_params(...)` (Ship 8 era). The new constants are
derived from BITS_IN/BITS_OUT and match the rest of the
10/12/16-bit family.

For limited-range output, individual channel bytes may differ from
v0.16.x by ≤1 LSB. Full-range output is byte-identical (both functions
reduce to scale=1<<15 at full range).

This unifies scale-constant computation across all bit depths into a
single source of truth. Cross-format planar parity tests
(`Vuya ↔ Yuva444p`) now exercise both full-range AND limited-range
without the prior workaround.

Migrated kernels: `yuv_planar_8bit`, `semi_planar_8bit`,
`packed_yuv_8bit` across the scalar reference and all 5 SIMD backends
(NEON, SSE4.1, AVX2, AVX-512, wasm-simd128) — 30 call sites total.

The legacy `range_params` function (hardcoded 8-bit constants) has been
removed; all callers now use `range_params_n::<8, 8>`.

This is **PR 3 of the 4-PR Tier 5 closure megaship**:
- ✅ PR 1 (v0.16.0): AYUV64 + Tier 5 closure
- ✅ PR 2 (v0.16.1): Multi-channel lane-order test backport
- ✅ PR 3 (this release): 8-bit planar scale-constant migration
- PR 4 (queued): Strategy A+ design + impl across source-α formats

## 0.16.1 — Multi-channel lane-order regression test backport

Pure test additions; no public API or behavior change.

Backports the multi-channel lane-order regression test pattern from
Ship 12d (AYUV64) to all existing SIMD-deinterleave formats × every
backend. The new pattern encodes the pixel index in TWO channels
independently (Y + U for non-α formats, Y + A for VUYA) and asserts:

- Luma output is in natural pixel order (catches Y-channel reorder bugs)
- SIMD RGB output is byte-identical to scalar RGB output (catches
  per-channel asymmetric mask bugs that would diverge from scalar)

Formats covered: V210, V410, V30X, XV36, VUYA, Y210, Y212, Y216.
(VUYX is excluded — its deinterleave kernel is shared with VUYA via
`vuya_to_rgb_or_rgba_row<ALPHA, ALPHA_SRC>`.)

Two real bugs were caught in this PR's audit:
- V410 SSE4.1 lane-order test was using W=4 but the SSE4.1 V410 kernel
  main loop is `while x + 8 <= width` — the test never entered the
  SIMD path. Fixed by bumping W to 16 (≥2 SIMD iterations).
- V30X SSE4.1 had the same issue and was fixed identically.

All lane-order tests now use W = 2× SIMD entry threshold, so each
test exercises ≥2 full SIMD main-loop iterations rather than only
the scalar tail.

This is **PR 2 of the 4-PR Tier 5 closure megaship**:
- ✅ PR 1 (Ship 12d v0.16.0): AYUV64 + Tier 5 closure
- ✅ PR 2 (this release): Multi-channel lane-order test backport
- PR 3 (next): 8-bit planar `range_params` → `range_params_n::<8, 8>` migration
- PR 4 (queued): Strategy A+ design + impl across source-α formats

After this release, all SIMD-deinterleave families share the same
lane-order regression test discipline. Asymmetric per-channel mask
bugs are now catchable across the entire packed-YUV family.

### Follow-up — coverage gap (not blocking)

Code-quality reviewers noted that for 4:2:2 formats (Y2xx, Y216) the
encoded values `Y[2k] = 2k+1` and `U[k] = 2k+1` happen to be equal,
so Part 1 (luma assertion) alone cannot distinguish a Y0↔U slot swap
bug. Part 2 (SIMD-vs-scalar parity) catches this bug class fully, so
the tests are correct, but a future cleanup could change the U-encoding
formula to a non-overlapping range (e.g. `U[k] = 0x80 + k`) so Part 1
provides additional coverage independent of the scalar reference.

---

## 0.16.0 — Tier 5 closed: AYUV64 (16-bit packed YUV 4:4:4 + α)

- Added `Ayuv64` source marker (FFmpeg `AV_PIX_FMT_AYUV64LE`).
- 16-bit packed YUV 4:4:4 with **source α** — A 16-bit A component at slot 0,
  followed by Y/U/V at slots 1/2/3 (channel order differs from
  VUYA's V/U/Y/A).
- u8 output path: i32 chroma; α depth-converted u16 → u8 via `>> 8`.
- u16 output path: **i64 chroma** (BT.2020 sums overflow i32 at
  BITS=16; reuses `chroma_i64x*` helpers from Y216 / `yuv_420p16` /
  Yuva444p16). α written direct as u16 (no conversion).
- 5-backend SIMD: NEON (16/16 px/iter), SSE4.1 (16/8), AVX2 (32/16),
  AVX-512 (64/32, F+BW baseline), wasm-simd128 (16/8).
- 7 sinker accessors: `with_rgb`, `with_rgba`, `with_rgb_u16`,
  `with_rgba_u16`, `with_luma`, `with_luma_u16`, `with_hsv`.
- Cross-format invariant: `Ayuv64 ↔ Yuva444p16` planar parity test
  validates source-α pass-through at both u8 and u16 paths
  (limited-range — no scale-constant divergence; first place
  i64 chroma + `ALPHA_SRC` at u16 meets cross-format parity for
  a packed format).
- Day-1 multi-channel lane-order regression tests (encode pixel
  index in BOTH Y AND A) on every backend — catches per-channel
  asymmetric mask bugs that the Y-only Ship 12c pattern would
  miss.

### Tier 5 closed

This release closes Tier 5 (Packed YUV 4:4:4). All four tranches
shipped:
- Ship 12a (v0.13): V410 + V30X (10-bit, MSB / LSB padded)
- Ship 12b (v0.14): XV36 (12-bit MSB-aligned + α-as-padding)
- Ship 12c (v0.15): VUYA + VUYX (8-bit source α / α-as-padding)
- Ship 12d (v0.16): AYUV64 (16-bit + source α) — this release

Three follow-up cleanup PRs are queued (per the Tier 5 closure
megaship plan):
- Multi-channel lane-order backport to existing Tier 4 / 5
  formats × every backend
- 8-bit planar `range_params` → `range_params_n::<8, 8>` migration
- Strategy A+ design + impl (post-Strategy-A α-overwrite hook
  across all source-α formats)

Next tier-of-formats work: Tier 9 + Tier 10 floats (rgbf16 /
rgbf32 / gbrpf32 — VFX archetype's biggest unmet need).

---

## 0.15.0 — Tier 5 third tranche: VUYA + VUYX (8-bit packed YUV 4:4:4)

- Added `Vuya` and `Vuyx` source markers (FFmpeg `AV_PIX_FMT_VUYA` and
  `AV_PIX_FMT_VUYX`).
- `Vuya` is 8-bit packed YUV 4:4:4 with **source alpha** — the per-
  pixel A byte is passed through to RGBA outputs.
- `Vuyx` shares the byte layout but treats the A slot as **padding**
  — RGBA outputs always force α=`0xFF` regardless of source.
- 5-backend SIMD: NEON (16 px/iter), SSE4.1 (16 px/iter), AVX2
  (32 px/iter), AVX-512 (64 px/iter), wasm-simd128 (16 px/iter).
- u8 RGB / RGBA / luma / HSV outputs only — no u16 paths (8-bit
  source).
- Cross-format invariants: `Vuya ↔ Yuva444p` planar parity test
  validates the source-α pass-through; `Vuyx` force-α-max test
  validates padding-byte ignore.
- AVX2 and AVX-512 backends ship with day-1 lane-order regression
  tests (the pattern that surfaced Ship 12b's AVX2 deinterleave bug
  retroactively — these tests catch it on first commit).

---

## 0.14.0 — Ship 12b (Tier 5 XV36, second tranche)

- Add `Xv36Frame` (12-bit packed YUV 4:4:4 with α-as-padding; FFmpeg
  `AV_PIX_FMT_XV36LE`). Each pixel is a u16 quadruple
  `U(16) ‖ Y(16) ‖ V(16) ‖ A(16)` with each channel using high 12 bits
  (low 4 bits zero, MSB-aligned). The `X` prefix means the A slot is
  padding; RGBA outputs force α = max regardless of source A.
- 5-backend SIMD: NEON (8 px/iter), SSE4.1 (8 px/iter), AVX2
  (16 px/iter), AVX-512 (32 px/iter), wasm-simd128 (8 px/iter). Each
  backend uses a u16x4 deinterleave (`vld4q_u16` on NEON / four-way
  u16 shuffle on x86 / wasm) + right-shift by 4 to drop padding bits.
- `MixedSinker<Xv36>` with `with_rgb` / `with_rgba` / `with_rgb_u16` /
  `with_rgba_u16` / `with_luma` / **`with_luma_u16`** / `with_hsv`.
- Retroactively wired `with_luma_u16` for V410 and V30X (Ship 12a
  formats) for cross-format symmetry — kernels were already shipped
  in 12a; only sinker accessor was missing.
- Xv36 ↔ Yuv444p12 planar parity oracle validates the SIMD path
  byte-for-byte against the established planar 4:4:4 12-bit reference.
- Tier 5 remaining: 12c VUYA / VUYX, 12d AYUV64.

---

## 0.13.0 — Ship 12a (Tier 5 V410 + V30X, first tranche)

- Add `V410Frame` (10-bit packed YUV 4:4:4 in 32-bit words; FFmpeg
  `AV_PIX_FMT_V410` = XV30 alias) — first Tier 5 tranche.
- Add `V30XFrame` (10-bit packed YUV 4:4:4 in 32-bit words; FFmpeg
  `AV_PIX_FMT_V30XLE`) — sibling of V410 with opposite padding position
  (`(msb) 10V | 10Y | 10U | 2X (lsb)` instead of V410's
  `(msb) 2X | 10V | 10Y | 10U (lsb)`).
- 5-backend SIMD for both formats: NEON (4 px/iter), SSE4.1 (8 px/iter),
  AVX2 (8 px/iter), AVX-512 (16 px/iter), wasm-simd128 (4 px/iter).
- `MixedSinker<V410>` and `MixedSinker<V30X>` with `with_rgb` /
  `with_rgba` / `with_rgb_u16` / `with_rgba_u16` / `with_luma` /
  `with_hsv`. (`with_luma_u16` deferred — no library consumer ask.)
- Cross-tranche infrastructure: 4 new `RowSlice` variants (`V410Packed`,
  `Xv36Packed`, `VuyaPacked`, `Ayuv64Packed`) + `V30XPacked` (added
  with V30X) for Ship 12b/c/d.
- V410 ↔ Yuv444p10 + V30X ↔ Yuv444p10 planar parity oracles validate
  both SIMD paths byte-for-byte against the established planar 4:4:4
  reference.
- Opens Tier 5 (remaining tranches: 12b XV36, 12c VUYA, 12d AYUV64).

---

## 0.12.0 — Ship 11d (Tier 4 Y216, closes Tier 4)

- Add `Y216Frame` (16-bit packed YUV 4:2:2, full-range u16 samples).
- Parallel `y216_*` kernel family separate from `y2xx_n_to_*<BITS>` for the
  i64 chroma u16 path (BITS=16 overflows i32 on Q15 chroma sums).
- 5-backend SIMD: NEON, SSE4.1, AVX2, AVX-512, wasm-simd128.
- `MixedSinker<Y216>` with `with_rgb` / `with_rgba` / `with_rgb_u16` /
  `with_rgba_u16` / `with_luma` (generic) / `with_luma_u16` / `with_hsv`.
- Closes Tier 4 (packed YUV 4:2:2 high-bit-depth: v210, Y210, Y212, Y216).

---

# UNRELEASED

## Tier 14 (in progress) — Bayer demosaic + WB + CCM

New RAW source family for camera-RAW pipelines (RED R3D, Blackmagic
BRAW, Nikon NRAW, FFmpeg `bayer_*`). `colconv` covers demosaic
onwards: vendor SDKs decode the camera bitstream into a Bayer plane,
`colconv` runs bilinear demosaic + per-channel white balance + 3×3
color-correction in a single per-row kernel.

### New types (all in `colconv::raw`)

- `BayerPattern` — `enum { Bggr, Rggb, Grbg, Gbrg }`,
  `#[non_exhaustive]`, `IsVariant`-derived.
- `BayerDemosaic` — `enum { Bilinear }`, `#[non_exhaustive]`,
  `Default = Bilinear`. Future variants (Malvar-He-Cutler, etc.)
  will land without a breaking change.
- `WhiteBalance { r, g, b: f32 }` — per-channel gain newtype with
  `::try_new` (validating: rejects NaN / ±∞ / negative via
  [`WhiteBalanceError`]), panicking `::new`, `::neutral`,
  accessors, `Default = neutral()`. `WbChannel` enum names which
  channel failed validation.
- `ColorCorrectionMatrix` — 3×3 newtype with `::try_new`
  (validating: rejects any non-finite element via
  [`ColorCorrectionMatrixError`]; negative entries are allowed
  because real CCMs subtract crosstalk), panicking `::new`,
  `::identity`, `as_array`, `Default = identity()`.

### New frame types (in `colconv::frame`)

- `BayerFrame<'a>` — single `&[u8]` plane. Odd widths and heights
  are accepted (cropped Bayer planes are real workflow output; the
  walker / kernel handle partial 2×2 tiles via edge clamping).
- `BayerFrame16<'a, const BITS: u32>` — `&[u16]` **low-packed** at
  `BITS` ∈ {10, 12, 14, 16} (active samples in the low `BITS` bits,
  valid range `[0, (1 << BITS) - 1]`). Matches the planar
  `Yuv420p10/12/14/16` convention in packing; diverges in
  validation: `BayerFrame16::try_new` validates **every active
  sample's range** as part of construction (returning
  `BayerFrame16Error::SampleOutOfRange` for out-of-range data),
  not just geometry. RAW pipelines often surface trusted-but-
  mispacked input from sensor SDKs, and the demosaic kernel has no
  well-defined behavior on out-of-range samples; mandatory
  validation makes the `bayer16_to` walker fully fallible — no
  data-dependent panic surface. Aliases: `Bayer10Frame` /
  `Bayer12Frame` / `Bayer14Frame` / `Bayer16Frame`. Odd dimensions
  accepted.
- `BayerFrameError` / `BayerFrame16Error` — structured error enums,
  `#[non_exhaustive]`, `IsVariant`-derived.

### New walkers / kernels

- `raw::bayer_to(src, pattern, demosaic, wb, ccm, sink)` and
  `raw::bayer16_to::<BITS, _>(...)` walkers — zero per-row and
  per-frame allocation. Walker fuses `M = CCM · diag(wb)` once at
  entry; row scratch is the source plane itself (`above` / `mid` /
  `below` row borrows with **mirror-by-2** boundary handling at
  top / bottom edges — `row 0 → above = row 1`, `row h-1 → below =
  row h-2`; replicate fallback only when `height < 2`). Same
  contract surfaces through `BayerRow::above()` / `below()` so
  custom sinks see the mirror borrows directly.
- Public dispatchers: `row::bayer_to_rgb_row`,
  `row::bayer16_to_rgb_row<BITS>`,
  `row::bayer16_to_rgb_u16_row<BITS>`. Each runs release-mode
  preflight (`above` / `below` length match `mid`, `rgb_out >=
  3 * width` via `checked_mul`, `BITS` const-asserted), matching
  the `yuv_*_to_rgb_row` boundary contract so future unsafe SIMD
  kernels inherit a hardened entry point. `use_simd` is currently
  a no-op — per-arch SIMD ships in a follow-up PR.
- Sink subtraits: `BayerSink`, `BayerSink16<BITS>`. Source markers:
  `Bayer`, `Bayer16<BITS>` plus `Bayer10` / `Bayer12` / `Bayer14` /
  `Bayer16Bit` aliases.

### SIMD coverage

| Kernel                                  | NEON | SSE4.1 | AVX2 | AVX-512 | wasm simd128 |
| --------------------------------------- | :--: | :----: | :--: | :-----: | :----------: |
| `bayer_to_rgb_row` (8-bit)              |  ⏳  |   ⏳   |  ⏳  |    ⏳   |      ⏳      |
| `bayer16_to_rgb_row<BITS>` (→u8)        |  ⏳  |   ⏳   |  ⏳  |    ⏳   |      ⏳      |
| `bayer16_to_rgb_u16_row<BITS>` (→u16)   |  ⏳  |   ⏳   |  ⏳  |    ⏳   |      ⏳      |

Scalar reference path lands first; per-arch SIMD backends are
scheduled as a dedicated follow-up PR (`feat/bayer-simd`).

### MixedSinker integration

- `MixedSinker<Bayer>` and `MixedSinker<Bayer16<BITS>>` impls — both
  expose `with_rgb` / `with_luma` / `with_hsv`; the `Bayer16<BITS>`
  variant additionally exposes `with_rgb_u16` for native-depth RGB
  output. RGB scratch buffer is grown lazily for the HSV-without-RGB
  path (mirrors the YUV impls).
- **Luma colorimetry is caller-configurable** via the new
  `LumaCoefficients` enum (`Bt709` / `Bt2020` / `Bt601` / `DciP3` /
  `AcesAp1` / `Custom(CustomLumaCoefficients)`, `#[non_exhaustive]`).
  YUV source impls memcpy luma directly off the Y plane and are
  unaffected; Bayer impls *derive* luma from the demosaiced RGB and
  therefore need to know which gamut weights to apply. Choose the
  preset matching the gamut your `ColorCorrectionMatrix` targets:
  passing a Rec.2020 CCM and using BT.709 luma weights produces a
  valid-shaped but numerically wrong luma plane for non-grayscale
  content (downstream scene-cut detectors, brightness thresholders
  and perceptual-diff tools see the wrong values; uniform gray is
  invariant — every preset agrees on gray, which is what made the
  hard-coded BT.709 path go undetected by uniform-gray tests).
  Default is `Bt709` to match the implicit weights every YUV → RGB →
  luma pipeline uses. API:
  `MixedSinker::<Bayer>::new(w, h).with_luma_coefficients(LumaCoefficients::Bt2020)`.
  `Custom` wraps the validated `CustomLumaCoefficients` newtype
  (private fields, mirrors the `WhiteBalance` / `ColorCorrectionMatrix`
  pattern): construct via `LumaCoefficients::try_custom(r, g, b)` /
  `CustomLumaCoefficients::try_new(r, g, b)` which return
  `Result<_, LumaCoefficientsError>` after rejecting NaN / ±∞ /
  negative / `> MAX_COEFFICIENT (10.0)` inputs. The bound is much
  tighter than `WhiteBalance::MAX_GAIN (1e6)` because the luma
  kernel multiplies into a `u32` accumulator (not `f32` as in
  WB/CCM) — `1e6` would overflow the per-row sum, `10.0` keeps it
  six orders of magnitude clear of `u32::MAX`. Custom weights are
  not normalized to sum to 1.0 — caller is responsible (otherwise
  the luma plane is brightness-scaled). All five published presets
  resolve to Q8 triples summing to exactly 256 so the kernel's
  `>> 8` divisor is exact (the published ACES AP1 weights round
  naïvely to `(70, 173, 14) = 257`; `cg` is shaved by 1 LSB to
  make the triple sum to 256 with the smallest perceptual error).

### Tests

- Frame-validation tests (8-bit + high-bit-depth, including
  `BayerFrame16::try_new` rejecting samples whose value exceeds
  `(1 << BITS) - 1` under the low-packed convention; both above-
  max and the common MSB-aligned packing-mismatch case).
- 5 type-helper tests (WB / CCM defaults, fuse arithmetic).
- 11 end-to-end walker + kernel tests (8-bit + 12-bit, solid R / G
  / B channels, uniform-byte invariant, pattern swap RGGB↔BGGR,
  walker row-count). Solid-channel assertions cover the **full
  frame** including borders — boundary handling uses mirror-by-2
  (`row -1 → row 1`, `row h → row h-2`, same on columns) which
  preserves CFA parity, so a constant-channel Bayer mosaic stays
  constant everywhere instead of bleeding wrong-color samples into
  the missing-channel averages at edges.
- 6 luma-coefficient tests covering both Bayer and Bayer16 paths:
  solid-red rows produce distinct luma values for each preset (54
  / 67 / 77 / 59 / 70 for BT.709 / BT.2020 / BT.601 / DCI-P3 /
  ACES AP1 — guards against silent collapse to one preset);
  `try_custom(1.0, 0.0, 0.0)` round-trips the red channel back to
  255; default is `Bt709`; uniform gray is invariant across all
  presets (regression-pin for the original
  `*_with_luma_uniform_byte` semantics); preset Q8 triples each
  sum to exactly 256.
- 8 `CustomLumaCoefficients` validation tests: accepts standard
  weights / zeroes / `MAX_COEFFICIENT` boundary; rejects NaN /
  ±∞ / negative / `MAX_COEFFICIENT + 1.0` / `1e9` per channel
  with the matching `LumaCoefficientsError` variant; `try_custom`
  routes errors through; `::new` panics loudly on hostile input;
  end-to-end "all three channels at MAX_COEFFICIENT, all pixels
  255" stays inside the `u32` accumulator and clamps to 255.

## Ship 8b — source-side YUVA (alpha-preserving RGBA output)

The follow-up to Ship 8: source-side alpha. Where Ship 8 padded the
output alpha lane to `0xFF` / `(1 << BITS) - 1` regardless of source,
Ship 8b adds **YUVA source types** that carry an alpha plane through
to the RGBA output. The first vertical slice ships `Yuva444p10`
(ProRes 4444 + α territory — the highest-value VFX format from the
Format Share table § 2a-1 row 10).

### Strategy B (forked kernels) over Strategy A (separate splice)

Two implementation strategies were considered:

- **Strategy A** (deferred) — run the existing RGBA kernel (alpha =
  opaque), then a second-pass helper reads source alpha + overwrites
  the alpha byte. Memory traffic 6W per pixel; ~50 LOC + 1 helper.
- **Strategy B** (adopted) — extend each kernel's const-`ALPHA`
  template with a third `ALPHA_SRC: bool` generic. Source-alpha is
  loaded inside the kernel, masked, and stored straight into the
  alpha lane in the same pass. Memory traffic 5W per pixel (single
  pass); ~3,000 LOC across 30+ kernels for an L1-noise ~10% perf
  win in the alpha-present case.

Strategy B was picked for best alpha-present throughput on the
high-bandwidth 4:4:4 + α format that motivated the work. Existing
`*_to_rgb_*` and `*_to_rgba_*` public wrappers are backward-compat
shims passing `ALPHA_SRC = false` and `None` to the templates — zero
overhead when alpha-source is off; existing call sites compile
unchanged.

### Vertical slice 1: `Yuva444p10` (3 PRs)

The first format follows the same staging pattern as Ship 8 high-bit
tranches (5/6/7): scalar prep first (call-site stable), then u8 SIMD,
then u16 SIMD.

| # | Tranche | Status |
|---|---|---|
| 1 | scalar prep + Frame + walker + dispatchers + sinker integration | ✅ shipped (PR #32) — `Yuva444pFrame16<BITS=10>`, `Yuva444p10Frame` alias, `yuva444p10_to` walker, `MixedSinker<Yuva444p10>`, scalar tests |
| 1b | u8 RGBA SIMD across all 5 backends | ✅ shipped (PR #33) |
| 1c | u16 RGBA SIMD across all 5 backends | ✅ shipped (PR #34) |

### Surface added

- **`Yuva444pFrame16<'a, const BITS: u32>`** — mirrors `Yuv444pFrame16`
  with an extra `a` slice + `a_stride`. Const-asserted `BITS == 10`
  in this slice; other bit depths land in subsequent vertical slices.
  `try_new` validates dimensions + plane lengths; `try_new_checked`
  additionally validates every active sample range.
- **`Yuva444p10Frame<'a>`** type alias.
- **`Yuva444p10`** marker + `Yuva444p10Row<'a>` (carries `a` slice)
  + `Yuva444p10Sink` trait + `yuva444p10_to` walker.
- **`MixedSinker<Yuva444p10>`** with `with_rgba` / `set_rgba` (u8) +
  `with_rgba_u16` / `set_rgba_u16` (u16) per-format builders, plus
  `with_rgb` / `with_rgb_u16` / `with_luma` / `with_hsv` alpha-drop
  paths that reuse the `Yuv444p10` row dispatchers verbatim.
- **Public dispatchers** in `colconv::row`: `yuva444p10_to_rgba_row`
  and `yuva444p10_to_rgba_u16_row` — same SIMD-via-`use_simd` shape
  as `yuv444p10_to_rgba_*`.

### Strategy B template extension

The four 4:4:4 const-`ALPHA` templates gained the `ALPHA_SRC` third
generic in this slice (only the BITS-generic planar variant is in
scope for this vertical slice; other 4:4:4 variants land later):

- `scalar::yuv_444p_n_to_rgb_or_rgba_row<BITS, ALPHA, ALPHA_SRC>` (u8)
- `scalar::yuv_444p_n_to_rgb_or_rgba_u16_row<BITS, ALPHA, ALPHA_SRC>` (u16)
- Same SIMD templates × 5 backends (NEON / SSE4.1 / AVX2 / AVX-512 /
  wasm simd128) — refactor in PRs #33 (u8) and #34 (u16).

Per-pixel store branched on three combinations:

| `ALPHA` | `ALPHA_SRC` | Per-pixel alpha |
|---|---|---|
| false | false | RGB-only (no alpha lane) |
| true | false | RGBA, alpha = `0xFF` u8 / `(1 << BITS) - 1` u16 (existing path) |
| true | true | RGBA, alpha = `(a_src[x] & bits_mask::<BITS>())` from source plane; depth-converted via `>> (BITS - 8)` for u8 output, native depth for u16 output |

`!ALPHA_SRC || ALPHA` const-asserted at every template top.

### Hardenings (Codex review fixes)

- **Source alpha is masked with `bits_mask::<BITS>()` before depth
  conversion** — `Yuva444p10Frame::try_new` accepts unchecked u16
  samples; without masking an overrange `1024` at BITS=10 would shift
  to `256` and cast to u8 zero, silently turning over-range alpha
  into transparent output. Same masking pattern that Y/U/V already
  use. Pinned by 2 regression tests at the sinker layer.
- **`MixedSinker<Yuva444p10>` wires alpha-drop paths** for `with_rgb`
  / `with_rgb_u16` / `with_luma` / `with_hsv` (declared on the
  generic `MixedSinker<F>` impl) — initial implementation only wrote
  RGBA buffers, leaving the others as silent stale-buffer bugs.
  Pinned by 4 cross-format byte-equivalence tests against
  `MixedSinker<Yuv444p10>`.

### Tests

- **Per-backend SIMD equivalence tests**: 30 per backend × 5 backends
  for `Yuva444p10` (5 u8 added in PR #33 + 5 u16 added in PR #34).
  Solid-alpha + random-alpha + tail-width coverage. All x86 tests
  carry `is_x86_feature_detected!` early-return guards.
- **Sinker integration tests**: 17 (PR #32 added 7 covering alpha
  pass-through / opacity contracts / buffer-too-short error paths;
  PR #32 review-fix added 7 covering alpha-drop paths + Strategy A
  combine; PR #32 review-fix added 2 covering overrange-alpha
  masking).
- **Test count growth**: 578 → 588 on aarch64-darwin host (583 after
  PR #33, 588 after PR #34); +5 NEON tests run at each tranche; the
  +20 x86/wasm tests fire on their respective CI runners.

### Notes

- **Sink-side YUVA + Ship 8 sinks are now end-to-end for the format**:
  with `Yuva444p10Frame` source and `MixedSinker<Yuva444p10>` sink,
  the alpha plane flows through to `with_rgba` / `with_rgba_u16`
  output. `with_rgb` / `with_rgb_u16` / `with_luma` / `with_hsv`
  are alpha-drop (reuse `Yuv444p10` row kernels).
- **Subsequent vertical slices (Ship 8b‑2 onward)** will mass-apply
  the established Strategy B template to other Yuva format families:
  `Yuva420p*` (4:2:0 with α — `yuva420p`, `yuva420p9/10/16`),
  `Yuva422p*` (4:2:2 with α — `yuva422p`, `yuva422p9/10/16`), and
  the remaining `Yuva444p*` variants (8-bit, 9-bit, 16-bit). The
  template's third generic + per-backend wrapper pattern is now
  proven; subsequent slices reuse it mechanically.

## Ship 8 — alpha + RGBA output (`with_rgba` / `with_rgba_u16`)

Adds packed RGBA output across the YUV format inventory. Every YUV
source is now sinkable to packed `R, G, B, A` u8 (alpha = `0xFF`) and,
for native-depth high-bit-depth sources, to packed u16 RGBA (alpha =
`(1 << BITS) - 1` for BITS-generic kernels, `0xFFFF` for the
dedicated 16-bit kernel family). The sink-side RGBA gap was the
single biggest unmet ask — image rendering, masking, and
alpha-aware composition all consume packed RGBA, and every
downstream of `colconv` benefits.

### Surface added

- **Per-format builders** on `MixedSinker<F>`: `with_rgba` /
  `set_rgba` (u8) for every wired format; `with_rgba_u16` /
  `set_rgba_u16` for the high-bit-depth families. Attaching RGBA
  to a sink that doesn't write it is a **compile error** (no
  silent stale-buffer bug) — each format's builder lives on its
  format-specific impl block, only added once `process` is wired.
- **Per-format public dispatchers** in `colconv::row`: `*_to_rgba_row`
  + `*_to_rgba_u16_row` siblings of every `*_to_rgb_*` dispatcher.
  Same SIMD-via-`use_simd` shape; same scalar reference contract.
- **Strategy A combine**: when both `with_rgb` and `with_rgba` are
  attached, `process` runs the YUV→RGB kernel once and fans out to
  RGBA via `expand_rgb_to_rgba_row` / `expand_rgb_u16_to_rgba_u16_row<BITS>`
  (memory-bound copy + alpha pad, ~7W bytes/row) instead of running
  the YUV math twice. ~2× speedup for the both-buffers caller.

### Mass-apply tracker

Each tranche shipped as a separate PR (or sub-PR series) to keep
review weight tractable. **All RGBA work is staged so the const-ALPHA
template lands per-format with a stable public-API signature; SIMD
backends are wired in follow-up sub-PRs without breaking call sites.**

| # | Tranche | Formats | Status |
|---|---|---|---|
| 1 | 4:2:0 planar | `Yuv420p` | ✅ shipped (PR #16) |
| 2 | 4:2:0 semi-planar | `Nv12`, `Nv21` | ✅ shipped (PR #17) — shared `<SWAP_UV, ALPHA>` template |
| 3 | 4:2:2 planar + semi-planar | `Yuv422p`, `Nv16` | ✅ shipped (PR #18) — wiring-only, reuses tranche-1+2 kernels |
| 4a | 4:4:4 planar | `Yuv444p` | ✅ shipped (PR #19) — kernel refactor across all 5 backends |
| 4b | 4:4:4 semi-planar | `Nv24`, `Nv42` | ✅ shipped (PR #20) — `<SWAP_UV, ALPHA>` template + Strategy A combine retro-applied to all 8 wired families |
| 4c | 4:4:0 planar | `Yuv440p` | ✅ shipped (PR #22) — wiring-only (reuses `yuv_444_to_rgba_row`) |
| 5 | High-bit 4:2:0 | `Yuv420p9/10/12/14/16`, `P010/P012/P016` | ✅ shipped — **5** scalar prep + dispatchers (PR #24); **5a** u8 SIMD across all 5 backends (PR #25); **5b** u16 SIMD + sinker integration (PR #26) |
| 6 | High-bit 4:2:2 | `Yuv422p9/10/12/14/16`, `P210/P212/P216` | ✅ shipped (PR #28) — sinker-only; reuses tranche-5 row kernels via the established 4:2:2 → 4:2:0 dispatcher pattern. (`Yuv440p10/12` deferred to tranche 7 alongside the 4:4:4 work it depends on.) |
| 7 | High-bit 4:4:4 + 4:4:0 | `Yuv444p9/10/12/14/16`, `P410/P412/P416`, `Yuv440p10/12` | ✅ shipped — **7** scalar prep + dispatchers (PR #29); **7b** u8 SIMD across all 5 backends (PR #30); **7c** u16 SIMD + sinker integration incl. `Yuv440p10/12` reusing 4:4:4 dispatchers (PR #31) |
| 8 | RAW | `Bayer`, `Bayer16<BITS>` | (deferred — RAW already has `with_luma_coefficients`) |

### SIMD coverage

**All 7 tranches (Ship 8 complete)**: 5 backends (NEON, SSE4.1, AVX2,
AVX-512, wasm simd128) have the const-ALPHA `<…, ALPHA>` template
wired for both u8 and u16 RGBA paths across every high-bit kernel
family (4:2:0 in tranche 5; 4:4:4 + Pn-444 in tranche 7). 4:2:2 and
4:4:0 sinkers reuse 4:2:0 / 4:4:4 dispatchers respectively — no new
SIMD code needed for those subsampling families. Per-arch RGBA store
helpers added in tranche 5: `vst4q_u8` / `vst4q_u16` (NEON),
`write_rgba_16` / `write_rgba_u16_8` (SSE4.1, AVX2 via re-export),
`write_rgba_64` / `write_rgba_u16_32` + `write_quarter_rgba`
(AVX-512), `u8x16_splat` / `i16x8_shuffle`-based `write_rgba_u16_8`
(wasm). Reused verbatim across tranches 5–7.

### Cleanup PRs

- **PR #21** — refactored inline `mod tests` blocks out of per-arch
  backend source files into sibling `tests.rs` files (NEON / SSE4.1 /
  AVX2 / AVX-512 / wasm simd128 + scalar + sinker/mixed). Pure
  layout reorg, no behavior change.
- **PR #23** — narrowed visibility of internal helpers and tightened
  module boundaries surfaced by the Strategy A retroactive refactor.
- **PR #27** — split the remaining inline `mod tests` blocks
  (`src/frame.rs`, `src/raw/types.rs`, `src/raw/bayer.rs`,
  `src/raw/bayer16.rs`) into sibling files. Same shape as PR #21.

### Tests (cumulative through PR #31, Ship 8 complete)

- **534 tests pass on aarch64-darwin** (host) at Ship 8 close;
  trajectory: 507 (PR #28, 4:2:2 sinker) → 513 (PR #29, 4:4:4 scalar
  prep) → 519 (PR #30, 4:4:4 u8 SIMD) → 534 (PR #31, 4:4:4 u16 SIMD
  + sinker).
- Per-arch RGBA equivalence tests: ~30 per high-bit family across all
  5 backends — tranche 5 added 4:2:0 (u8 + u16, BITS=9/10/12/14 + 16
  + Pn); tranche 7b/7c added 4:4:4 (u8 + u16, BITS=9/10/12/14 + 16 +
  Pn-444). All matrices × ranges × natural-block + tail widths.
- Sinker integration tests: 8 in PR #26 (4:2:0), 8 in PR #28 (4:2:2),
  6 in PR #29 (4:4:4 scalar), 9 in PR #31 (4:4:4 + Yuv440p10 cross-
  family kernel-reuse proof). Cover standalone-RGBA, Strategy A
  combine, and buffer-too-short error variants.
- All x86 `#[test]` functions exercising new SIMD kernels include
  `is_x86_feature_detected!` early-return guards (per the PR #25 CI
  fallout — without them, ASAN sanitizer saw `SIGILL` and Miri
  reported UB on runners lacking the feature).

### Notes

- **Strategy B deferred**: a third const generic on every kernel
  (`<SWAP_UV, RGB_OUT, RGBA_OUT>`) eliminating the L1-hot RGB readback
  in the Strategy A path was considered and rejected as ~2,500 LOC
  for L1-noise improvement. See `docs/color-conversion-functions.md`
  § Ship 8 → Combined RGB + RGBA path for the design notes.
- **Source-side YUVA** (Ship 8b — separate follow-up): not part of
  this Ship. Adds YUVA frame types (`Yuv420pAFrame`, etc.) so the
  alpha plane flows through to RGBA output instead of being padded
  to opaque. Ship 8 only addresses the sink-side RGBA gap.

## Ship 7 — u16 semi-planar 4:2:2 / 4:4:4 (P210 / P212 / P216 / P410 / P412 / P416)

Six new high-bit-packed semi-planar formats from the FFmpeg HW-decode
download space (CUDA / NVDEC / QSV emit these for HDR 4:2:2 and 4:4:4
content).

### New formats

- **`P210`** / **`P212`** / **`P216`** — 4:2:2 semi-planar at 10 / 12 /
  16 bits. Const-generic `PnFrame422<BITS>` with aliases. Per-row
  layout is identical to P010/P012/P016 (half-width interleaved UV =
  `width` u16 elements per row); only the walker reads chroma row
  `r` instead of `r / 2` (4:2:2 vs 4:2:0). MixedSinker impls reuse
  the existing `p010_to_rgb_*` / `p012_to_rgb_*` / `p016_to_rgb_*`
  row primitives — **zero new SIMD code** for 4:2:2.
- **`P410`** / **`P412`** / **`P416`** — 4:4:4 semi-planar at 10 / 12 /
  16 bits. Const-generic `PnFrame444<BITS>` with aliases. UV is
  full-width (`2 * width` u16 elements per row, one `U, V` pair per
  pixel — no horizontal chroma subsampling). New row-primitive
  family `p_n_444_to_rgb_*<BITS>` (BITS ∈ {10, 12}, Q15 i32 pipeline)
  + dedicated `p_n_444_16_to_rgb_*` (16-bit, parallel i64-chroma path
  for u16 output).

Frame error type: `PnFrameError` extended with the same variants for
both new families. The `OddWidth` variant message was reworded
format-agnostically (`"horizontally-subsampled chroma requires even
width"`) since it now surfaces from both `PnFrame::try_new` (4:2:0)
and `PnFrame422::try_new` (4:2:2). `PnFrame444` has no parity
constraint and never emits this variant.

### SIMD coverage (4:4:4 family)

| Kernel                                  | NEON | SSE4.1 | AVX2 | AVX-512 | wasm simd128 |
| --------------------------------------- | :--: | :----: | :--: | :-----: | :----------: |
| `p_n_444_to_rgb_row<BITS>`              |  ✅  |   ✅   |  ✅  |    ✅   |      ✅      |
| `p_n_444_to_rgb_u16_row<BITS>`          |  ✅  |   ✅   |  ✅  |    ✅   |      ✅      |
| `p_n_444_16_to_rgb_row`                 |  ✅  |   ✅   |  ✅  |    ✅   |      ✅      |
| `p_n_444_16_to_rgb_u16_row`             |  ✅  |   ✅   |  ✅  |    ✅   |      ✅      |

**Native SIMD on every supported backend** for both u8 and u16 output.
Block sizes per iteration:

| Backend       | u8 / u16-low | u16 i64 (P416) |
| ------------- | :----------: | :------------: |
| NEON          | 16 px        | 8 px           |
| SSE4.1        | 16 px        | 8 px           |
| AVX2          | 32 px        | 16 px          |
| AVX-512       | 64 px        | 32 px          |
| wasm simd128  | 16 px        | 8 px           |

UV deinterleave per-arch: `vld2q_u16` (NEON), `_mm_shuffle_epi8` +
permutes (SSE4.1), `_mm256_shuffle_epi8` + `_mm256_permute2x128_si256`
(AVX2), `_mm512_shuffle_epi8` + `_mm512_permutexvar_epi64` (AVX-512),
`u8x16_swizzle` (wasm simd128).

The 16-bit u16-output i64 chroma path uses **native `_mm512_srai_epi64`
on AVX-512** and **native `i64x2_shr` on wasm** — no bias trick. AVX2
and SSE4.1 use the `srai64_15_x4` / `srai64_15` bias trick (those ISAs
lack arithmetic i64 right shift). NEON uses native `vshrq_n_s64`.

### MixedSinker integration

6 new `MixedSinker<F>` impls (P210 / P212 / P216 / P410 / P412 /
P416). New `RowSlice` variants for the 4:4:4 chroma rows:
`UvFull10`, `UvFull12`, `UvFull16`. The 4:2:2 impls reuse the
existing `UvHalf10/12/16` variants since the per-row layout is
identical to 4:2:0.

### Tests

- 6 new sanity gray-to-gray `MixedSinker` integration tests.
- 3 new walker-level SIMD-vs-scalar equivalence tests for P410 / P412
  / P416 at width 1922 (forces tail handling), pseudo-random chroma,
  full + limited range, all matrices.
- 25 new per-arch SIMD scalar-equivalence tests for the new
  `p_n_444_to_rgb_*<BITS>` and `p_n_444_16_to_rgb_*` kernels —
  5 tests × 5 backends (NEON, SSE4.1, AVX2, AVX-512, wasm simd128).
  Cover all 6 ColorMatrix variants × full + limited range at the
  backend's natural width, plus tail widths {1, 3, 7, 8, 9, 15, 16,
  17, 31, 33, 47, 63, 65, 95, 127, 129, 1920, 1921} forcing
  scalar-tail fallback at every block-size boundary.
- **Total suite: 318 passed on aarch64** (up from 304 at Ship 6b);
  +20 tests fire on x86_64 (15 SSE4.1 / AVX2 / AVX-512) / wasm32 (5)
  CI runners.

## Ship 6b — 9-bit family + 4:4:0 family (Tier 1 completion)

Closes the remaining FFmpeg `AVPixelFormat` Tier 1 gap. Six new
formats, all reusing existing kernel families:

### New formats

- **`Yuv420p9` / `Yuv422p9` / `Yuv444p9`** — 9-bit planar at 4:2:0 /
  4:2:2 / 4:4:4. Aliases over `Yuv420pFrame16<9>` /
  `Yuv422pFrame16<9>` / `Yuv444pFrame16<9>`. Reuses the const-generic
  `yuv_420p_n_to_rgb_*<BITS>` and `yuv_444p_n_to_rgb_*<BITS>` kernel
  families — only the AND mask (`0x1FF`) and the Q15 scale change at
  `BITS = 9`. Niche format (AVC High 9 profile only); no HEVC / VP9 /
  AV1 producers.
- **`Yuv440p`** — 4:4:0 planar at 8 bits (`AV_PIX_FMT_YUV440P` /
  `AV_PIX_FMT_YUVJ440P`). Full-width chroma, half-height — the
  axis-flipped twin of `Yuv422p`. Reuses `yuv_444_to_rgb_row`
  verbatim; only the walker reads chroma row `r / 2`. Mostly seen
  from JPEG decoders that subsample vertically only.
- **`Yuv440p10` / `Yuv440p12`** — 4:4:0 planar at 10 / 12 bits.
  `Yuv440pFrame16<BITS>` with aliases. Reuses the const-generic
  `yuv_444p_n_to_rgb_*<BITS>` family. No 9 / 14 / 16-bit variants
  exist in FFmpeg, so `try_new` rejects them.
- New `RowSlice` variants for the 9-bit shape rows: `Y9`, `UHalf9`,
  `VHalf9`, `UFull9`, `VFull9`.

### SIMD

All 6 new formats inherit native SIMD coverage from the underlying
const-generic kernel families. No new SIMD code paths — only the
compile-time `BITS` validators were widened from `{10, 12, 14}` to
`{9, 10, 12, 14}` across scalar + 5 backends.

| Kernel dispatch                                       | NEON | SSE4.1 | AVX2 | AVX-512 | wasm simd128 |
| ----------------------------------------------------- | :--: | :----: | :--: | :-----: | :----------: |
| `yuv_420p_n_to_rgb_*<9>` (4:2:0 / 4:2:2)              |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p_n_to_rgb_*<9>` (4:4:4)                      |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444_to_rgb_row` (via `Yuv440p`)                  |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p_n_to_rgb_*<10/12>` (via `Yuv440p10/12`)     |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |

### Notes

- 4:4:0 is rare in modern codecs (mostly JPEG vertical-only
  subsampling) but ships as a first-class citizen for completeness.
- 9-bit is niche but trivially cheap to add (zero new kernels);
  shipping it closes the Tier 1 row in the format matrix.
- Skipped: `Yuv411p` / `Yuv410p` (legacy DV / Cinepak — uncommon
  enough that adding them now would be speculative work).

## Ship 6 — Yuv422p / Yuv444p at 8/10/12/14/16 bit

All three priorities landed in a single PR:
- **A (HW→SW gap)** — `Yuv444p16` (NVDEC / CUDA 4:4:4 HDR download target)
- **B (Pro video)** — `Yuv422p10/12/14`, `Yuv444p10/12/14` (ProRes, DNxHD)
- **C (Common SW)** — `Yuv422p`, `Yuv444p` 8-bit (libx264 defaults)

### New formats

- **`Yuv422p`** — 4:2:2 planar, 8-bit. New `Yuv422pFrame` + marker +
  walker + `MixedSinker<Yuv422p>` impl. Per-row kernel reused from
  `Yuv420p` verbatim (4:2:0 vs 4:2:2 differs only in the vertical
  walker). No new SIMD kernels.
- **`Yuv422p10` / `Yuv422p12` / `Yuv422p14`** — 4:2:2 planar at 10 /
  12 / 14 bit. Const-generic `Yuv422pFrame16<BITS>` with aliases.
  Per-row kernels reused from the `Yuv420p_n<BITS>` family.
- **`Yuv422p16`** — 4:2:2 planar at 16 bit. Alias over
  `Yuv422pFrame16<'_, 16>`. Per-row kernels reused from the parallel
  i64-chroma `yuv_420p16_to_rgb_*` family.
- **`Yuv444p`** — 4:4:4 planar, 8-bit. New `Yuv444pFrame` + marker +
  walker + `MixedSinker<Yuv444p>` + dedicated `yuv_444_to_rgb_row`
  kernel family. No width parity constraint (4:4:4 chroma is 1:1
  with Y, not paired).
- **`Yuv444p10` / `Yuv444p12` / `Yuv444p14`** — 4:4:4 planar at 10 /
  12 / 14 bit. Const-generic `Yuv444pFrame16<BITS>` with aliases.
  New const-generic `yuv_444p_n_to_rgb_row<BITS>` +
  `yuv_444p_n_to_rgb_u16_row<BITS>` kernel family.
- **`Yuv444p16`** — 4:4:4 planar at 16 bit. Alias over
  `Yuv444pFrame16<'_, 16>`. Dedicated parallel i64-chroma kernel
  family `yuv444p16_to_rgb_*` (same rationale as `Yuv420p16` — the
  blue coefficient overflows i32 at 16 bits).
- New `RowSlice` variants for the full-width 4:4:4 chroma rows:
  `UFull`, `VFull`, `UFull10/12/14`, `VFull10/12/14`.

### SIMD

Every new 4:4:4 kernel ships native SIMD on every backend — no
scalar fallbacks or cross-tier delegations:

| Kernel family                     | NEON | SSE4.1 | AVX2 | AVX-512 | wasm simd128 |
| --------------------------------- | :--: | :----: | :--: | :-----: | :----------: |
| `yuv_444_to_rgb_row` (8-bit)      |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p_n_to_rgb_row<BITS>`     |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p_n_to_rgb_u16_row<BITS>` |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p16_to_rgb_row`           |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |
| `yuv_444p16_to_rgb_u16_row`       |  ✅  |   ✅   |  ✅  |   ✅    |      ✅      |

Yuv422p family reuses Yuv420p kernels (4:2:2 differs only in the
vertical walker):

| Yuv422p kernel dispatch                                      | NEON | SSE4.1 | AVX2 | AVX-512 | wasm |
| ------------------------------------------------------------ | :--: | :----: | :--: | :-----: | :--: |
| `yuv_420_to_rgb_row` (via `Yuv422p`)                         |  ✅  |   ✅   |  ✅  |   ✅    |  ✅  |
| `yuv420p{10,12,14,16}_to_rgb_*` (via `Yuv422p{10,12,14,16}`) |  ✅  |   ✅   |  ✅  |   ✅    |  ✅  |

Block sizes (u8 output): 16 pixels (NEON / SSE4.1 / wasm), 32
pixels (AVX2), 64 pixels (AVX-512). The 16-bit u16-output variants
run at 8 pixels per iter on SSE4.1 and wasm (i64-lane width), 16 on
AVX2, 32 on AVX-512.

### Bonus: native 16-bit u16 kernels on AVX2 + wasm (resolves Ship 4c leftover)

This PR also replaces the **three residual u16-output delegations**
from Ship 4b/4c — `yuv_420p16_to_rgb_u16_row`, `p16_to_rgb_u16_row`,
and the newly added `yuv_444p16_to_rgb_u16_row` — with native
implementations on AVX2 and wasm simd128:

- **AVX2**: all three previously delegated to SSE4.1. The delegation
  was rational when `_mm256_srai_epi64` was unavailable, but the
  `srai64_15` bias trick scales cleanly to 256 bits via
  `_mm256_srli_epi64` + offset. New AVX2 kernels process 16 pixels
  per iter — 2× the SSE4.1 rate.
- **wasm simd128**: all three previously fell through to scalar. The
  "no native i64 arithmetic shift" rationale became stale once
  `i64x2_shr_s` stabilized. New wasm kernels use `i64x2_mul` +
  `i64x2_shr` at 8 pixels per iter.

Every 16-bit u16-output path is now native on every backend.

### Tests

37 new tests total:
- 11 `MixedSinker` integration tests (10 `gray → gray` sanity checks
  covering every new format × u8/u16 output, plus a `yuv422p ↔
  yuv420p` equivalence check that pins the shared-row-kernel
  contract).
- 6 NEON arch equivalence tests for `yuv_444p_n` and `yuv_444p16`
  across all six matrices, full/limited range, and odd-width tails
  (1, 3, 15, 17, 32, 33, 1920, 1921).
- 10 per-arch `yuv_444_to_rgb_row` scalar-equivalence tests (2 per
  backend × 5 backends).
- 10 per-arch `yuv_444p_n<BITS>` scalar-equivalence tests on x86 +
  wasm (4 kernels × SSE4.1 / AVX2 / AVX-512 / wasm, covering 10/12/14
  and widths straddling each backend's block size).

Total suite: **273 passed on aarch64** (up from 254 at v0.5). x86
and wasm tests run in CI on their respective targets.

### Benches

10 new benches:
- `yuv_422p_to_rgb`, `yuv_422p10_to_rgb`, `yuv_422p12_to_rgb`,
  `yuv_422p14_to_rgb`, `yuv_422p16_to_rgb` — reuse the 4:2:0 row
  primitives; output numerically identical to the 4:2:0 benches at
  the same width.
- `yuv_444p_to_rgb`, `yuv_444p10_to_rgb`, `yuv_444p12_to_rgb`,
  `yuv_444p14_to_rgb`, `yuv_444p16_to_rgb` — dedicated 4:4:4
  kernels. NEON 4× over scalar on the 8-bit kernel (~1.6 GiB/s
  scalar → ~6.4 GiB/s NEON at 1080p).

## Ship 5 — NV16 / NV24 / NV42

### New formats

- **`Nv16`** — 4:2:2 semi-planar, UV-ordered. New `Nv16Frame` type
  plus `MixedSinker<Nv16>` impl. Per-row kernel is shared with
  `Nv12` (the 4:2:0 → 4:2:2 difference is purely in the vertical
  walker — one UV row per Y row instead of one per two).
- **`Nv24`** — 4:4:4 semi-planar, UV-ordered. New `Nv24Frame` type,
  `MixedSinker<Nv24>` impl, and a dedicated kernel family
  (`nv24_to_rgb_row`). No width parity constraint (4:4:4 chroma is
  1:1 with Y).
- **`Nv42`** — 4:4:4 semi-planar, VU-ordered. Shares kernels with
  `Nv24` via a `SWAP_UV` const generic (mirrors the `Nv21` / `Nv12`
  pairing).
- New `RowSlice::UvFull` / `RowSlice::VuFull` variants for the
  full-width chroma rows.

### SIMD

Native NV24 / NV42 kernels across all five arches:

| Backend   | Block (Y × UV bytes) | Relative to SSE4.1 |
| --------- | -------------------- | ------------------ |
| NEON      | 16 × 32              | 1×                 |
| SSE4.1    | 16 × 32              | 1× (baseline)      |
| AVX2      | 32 × 64              | ~2×                |
| AVX-512   | 64 × 128             | ~4×                |
| wasm      | 16 × 32              | 1×                 |

The 4:4:4 layout simplifies the main loop vs NV12/NV21 — no
horizontal chroma duplication since UV is 1:1 with Y.

### AVX-512 16-bit u16 native kernels (follow-up to Ship 4b)

- `yuv_420p16_to_rgb_u16_row` and `p16_to_rgb_u16_row` on AVX-512
  now run a native 32-pixel-per-iter kernel using
  `_mm512_srai_epi64` + `_mm512_mul_epi32` +
  `_mm512_permutex2var_epi32` reassembly, replacing the 8-pixel
  SSE4.1 delegation that Ship 4b shipped. ~4× throughput
  improvement on AVX-512 CPUs.
- AVX2 u16 paths still delegate to SSE4.1 (AVX2 lacks
  `_mm256_srai_epi64`; reimplementing the `srai64_15` bias trick at
  256 bits would have marginal gain).

### Tests

34 new tests total:
- 15 frame validation tests for `Nv16Frame` / `Nv24Frame` /
  `Nv42Frame` including odd-width + odd-height acceptance for 4:4:4,
  `u32` overflow on the `2 × width` chroma stride.
- 13 MixedSinker integration tests including the cross-format
  parity checks `nv16_matches_nv12_mixed_sinker_with_duplicated_chroma`
  and `nv42_matches_nv24_mixed_sinker_with_swapped_chroma` (the
  latter uses width 33 to exercise the no-parity contract), plus
  error-path tests mirroring the NV12 suite.
- 6 NEON arch equivalence tests across 6 matrices × full/limited
  range × odd-width tails (1, 3, 15, 17, 33).

Total suite: 254 passed on aarch64 (up from 204 at v0.4b).

### Benches

`nv16_to_rgb`, `nv24_to_rgb`, `nv42_to_rgb` — same `simd` /
`scalar` split and 720p / 1080p / 4K widths as the rest of the
family.

# 0.1.2 (January 6th, 2022)

FEATURES


