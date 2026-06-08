# Changelog

All notable changes to colconv are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/spec/v2.0.0.html); pre-1.0
breaking changes bump the `x` in `0.x.y`.

## Unreleased

## 0.1.0 — 2026-06-08

Initial public release. colconv is a `no_std`-friendly SIMD-dispatched
color-conversion library covering the FFmpeg `AVPixelFormat` space.

### Architecture

- **Runtime SIMD dispatch.** Every kernel ships AVX-512, AVX2, SSE4.1,
  and a scalar fallback. Backend is selected once at startup via
  `is_x86_feature_detected!` with no per-row branching. The scalar
  path is the reference implementation every other tier is
  equivalence-tested against.
- **Sink-based output API.** Consumers pick which derived outputs a
  source frame produces (`RGB`, `RGBA`, `Luma`, `HSV`, custom);
  kernels for unselected outputs don't compile. Cross-format
  helpers (Q15 ranges, `rgb_expand`, HSV conversion, raw type
  surface, alpha extraction, Y-plane to luma) are always available.
- **Per-format opt-in.** 18 format-family feature gates forward to
  the matching `mediaframe/<family>` so source-format markers and
  `Frame` types compile in lockstep with the kernel code.
- **`no_std` + `alloc`.** Capability tiers are additive — depend on
  `no_std + no_alloc`, `no_std + alloc` (pulls `libm` for the scalar
  path), or full `std` (default).

### Format coverage

| feature | source formats |
|---|---|
| `yuv-planar` | YUV planar 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4 at 8/10/12/14/16-bit |
| `yuv-semi-planar` | NV12 / NV16 / NV21 / NV24 / NV42 + 16-bit P210/P212/P216/P410/P412/P416 |
| `yuva` | YUVA 4:2:0 / 4:2:2 / 4:4:4 (auto-enables `yuv-planar`) |
| `yuv-packed` | UYVY / YUYV / 4:1:1 packed |
| `yuv-444-packed` | AYUV64 / VUYA / VUYX / Y410 / V410 / V30X / XV30 / XV36 |
| `y2xx` | Y210 / Y212 / Y216 (10/12/16-bit packed 4:2:2) |
| `v210` | V210 (10-bit packed 4:2:2, 6-pixels-per-block) |
| `rgb` | packed RGB / RGBA / BGR / BGRA 8-bit + RGB48 / BGR48 / RGBA64 / BGRA64 16-bit |
| `rgb-float` | packed RGB f16 / f32 |
| `rgb-legacy` | RGB565 / BGR565 / RGB555 / BGR555 / RGB444 / BGR444 |
| `gbr` | planar GBR / GBRA at 8-bit, 9/10/12/14/16-bit, f16, f32 |
| `gray` | Y8 / Y16 / Yf16 / Yf32 / Ya8 / Ya16 |
| `bayer` | Bayer 8/16-bit (RGGB / GRBG / GBRG / BGGR) with optional WB + CCM |
| `xyz` | XYZ12 (DCDM / DCP) |
| `mono` | 1-bit mono (monoblack / monowhite) + PAL8 palette |

### Configuration

- `ColorMatrix` / range / endianness handled per-kernel; BT.709 and
  BT.601 (limited and full) lane orderings supported across every
  family.
- Bayer family supports optional white balance and color correction
  matrices, with validated `try_new` constructors that reject
  finite-but-extreme inputs that would overflow during the per-pixel
  matmul.

### Quality

- Scalar reference implementation drives equivalence tests across
  every SIMD backend (AVX-512 / AVX2 / SSE4.1 / NEON / WASM SIMD-128).
- Big-endian parity coverage for every `*LE` host-endian-sensitive
  format.
- Comprehensive sinker-layer fixture and dispatcher-layer regression
  suite.
