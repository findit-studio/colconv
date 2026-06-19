#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Error: TARGET is not provided"
  exit 1
fi

TARGET="$1"

# Install cross-compilation toolchain on Linux
if [ "$(uname)" = "Linux" ]; then
  case "$TARGET" in
    aarch64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-aarch64-linux-gnu
      ;;
    i686-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-multilib
      ;;
    powerpc64-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-powerpc64-linux-gnu
      ;;
    s390x-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-s390x-linux-gnu
      ;;
    riscv64gc-unknown-linux-gnu)
      sudo apt-get update && sudo apt-get install -y gcc-riscv64-linux-gnu
      ;;
  esac
fi

rustup toolchain install nightly --component miri
rustup override set nightly
cargo miri setup

export MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-disable-isolation -Zmiri-symbolic-alignment-check"

# 32-bit targets share one 4 GB address space across the whole lib
# test binary, and miri's default partial address reuse
# (-Zmiri-address-reuse-rate=0.5) exhausts it near the end of the
# suite — the failure then surfaces as "no more free addresses" in
# whichever test happens to run last. Full reuse keeps the run inside
# the space; 64-bit cells keep the stricter default.
case "$TARGET" in
  i686-*)
    export MIRIFLAGS="$MIRIFLAGS -Zmiri-address-reuse-rate=1.0 -Zmiri-address-reuse-cross-thread-rate=1.0"
    ;;
esac

# Full address reuse alone is still not enough to fit the whole lib test
# binary in one i686 process, so split the i686 lib run into four disjoint,
# exhaustive test-filter passes — each a fresh process with its own 4 GB
# address space. The partition (verified exhaustive + non-overlapping over
# the full test list):
#   1) everything outside `sinker::`
#   2) `sinker::` minus the `resample_*` test modules
#   3) the `resample_y*` modules (yuv / yuva / y2xx / ya)
#   4) the remaining `resample_*` modules
# Each test runs in exactly one pass, so coverage is identical to the single
# run; `set -e` propagates any failure. 64-bit cells run the binary in one
# pass (`--tests` is a no-op here — there are no integration test targets).
case "$TARGET" in
  i686-*)
    cargo miri test --lib --target "$TARGET" -- --skip sinker::
    cargo miri test --lib --target "$TARGET" -- \
      sinker:: --skip sinker::mixed::tests::resample_
    cargo miri test --lib --target "$TARGET" -- \
      sinker::mixed::tests::resample_y
    cargo miri test --lib --target "$TARGET" -- \
      sinker::mixed::tests::resample_ --skip sinker::mixed::tests::resample_y
    ;;
  *)
    cargo miri test --lib --tests --target "$TARGET"
    ;;
esac
