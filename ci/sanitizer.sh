#!/bin/bash
set -ex

export ASAN_OPTIONS="detect_odr_violation=0 detect_leaks=0"

# Give the test harness's worker threads an 8 MiB stack. The sanitizer
# (debug + instrumented) builds use markedly larger stack frames, so a
# constrained worker-thread stack can overflow even on output that the
# release build runs comfortably — defence-in-depth alongside the bounded
# `MixedSinker` inline size (see `mixed_sinker_inline_size_stays_small`).
export RUST_MIN_STACK=8388608

TARGET="x86_64-unknown-linux-gnu"

# Run address sanitizer
RUSTFLAGS="-Z sanitizer=address" \
cargo test --tests --target "$TARGET" --all-features

# Run leak sanitizer
RUSTFLAGS="-Z sanitizer=leak" \
cargo test --tests --target "$TARGET" --all-features

# Run memory sanitizer (requires -Zbuild-std for instrumented std)
RUSTFLAGS="-Z sanitizer=memory" \
cargo -Zbuild-std test --tests --target "$TARGET" --all-features

# Run thread sanitizer (requires -Zbuild-std for instrumented std)
RUSTFLAGS="-Z sanitizer=thread" \
cargo -Zbuild-std test --tests --target "$TARGET" --all-features
