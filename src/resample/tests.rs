use super::*;

#[test]
fn noop_plan_is_identity() {
  assert_eq!(NoopResampler.plan(1920, 1080), Ok(None));
}

#[test]
fn area_identity_plans_none() {
  assert_eq!(AreaResampler::to(8, 8).plan(8, 8), Ok(None));
}

#[test]
fn area_zero_output_rejected() {
  let err = AreaResampler::to(4, 0).plan(8, 8).unwrap_err();
  match err {
    ResampleError::ZeroOutputDimension(e) => {
      assert_eq!(e.out_w(), 4);
      assert_eq!(e.out_h(), 0);
    }
    other => panic!("expected ZeroOutputDimension, got {other:?}"),
  }
}

#[test]
fn area_upscale_rejected_per_axis() {
  // Width within bounds, height exceeding: either axis alone trips it.
  let err = AreaResampler::to(4, 16).plan(8, 8).unwrap_err();
  match err {
    ResampleError::UpscaleUnsupported(e) => {
      assert_eq!((e.src_w(), e.src_h()), (8, 8));
      assert_eq!((e.out_w(), e.out_h()), (4, 16));
    }
    other => panic!("expected UpscaleUnsupported, got {other:?}"),
  }
}

#[test]
fn area_plan_reports_geometry() {
  let plan = AreaResampler::to(336, 189)
    .plan(1920, 1080)
    .expect("valid downscale")
    .expect("non-identity");
  assert_eq!(plan.out_w(), 336);
  assert_eq!(plan.out_h(), 189);
  assert_eq!(plan.out_dims(), (336, 189));
  assert_eq!((plan.src_w(), plan.src_h()), (1920, 1080));
}

#[test]
fn area_integer_ratio_spans() {
  // 8 -> 4 on both axes: every output pixel covers exactly two source
  // cells with equal weight (cell width = out on the scaled grid).
  let plan = AreaResampler::to(4, 4)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  for axis in [plan.h(), plan.v()] {
    assert_eq!(axis.out_len(), 4);
    for j in 0..4 {
      let (start, weights) = axis.span(j);
      assert_eq!(start, 2 * j);
      assert_eq!(weights, &[4, 4]);
    }
  }
}

#[test]
fn area_fractional_spans_8_to_3() {
  // Scale 8/3: out pixel j covers [8j, 8j+8) on the x3 grid, source
  // cell i covers [3i, 3i+3). Overlaps hand-derived; each span sums to
  // the denominator (the source dimension, 8).
  let plan = AreaResampler::to(3, 8)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");

  let h = plan.h();
  assert_eq!(h.out_len(), 3);
  assert_eq!(h.span(0), (0, &[3usize, 3, 2][..]));
  assert_eq!(h.span(1), (2, &[1usize, 3, 3, 1][..]));
  assert_eq!(h.span(2), (5, &[2usize, 3, 3][..]));

  // Vertical axis is identity-sized (8 -> 8): unit spans, full weight.
  let v = plan.v();
  assert_eq!(v.out_len(), 8);
  for j in 0..8 {
    assert_eq!(v.span(j), (j, &[8usize][..]));
  }
}

#[test]
fn area_spans_handle_the_upsample_direction() {
  // The coverage math is direction-agnostic; the native tier feeds
  // chroma grids through it (8x8 -> 6x6 frame means 4 -> 6 chroma).
  // Hand-derived: out pixel intervals of length 4 on the x6 grid.
  let spans = AxisSpans::area(4, 6).expect("upsample coverage is valid");
  assert_eq!(spans.out_len(), 6);
  let expected: [(usize, &[usize]); 6] = [
    (0, &[4]),
    (0, &[2, 2]),
    (1, &[4]),
    (2, &[4]),
    (2, &[2, 2]),
    (3, &[4]),
  ];
  for (j, &(start, weights)) in expected.iter().enumerate() {
    assert_eq!(spans.span(j), (start, weights), "span {j}");
  }
}

#[test]
fn area_spans_partition_the_source() {
  // The NaFlex case: 1920x1080 -> 336x189 (x40/7 per axis). Every span
  // sums to the source dimension, starts strictly increase, tap counts
  // stay within ceil(scale) + 1, and coverage ends exactly at the last
  // source cell.
  let plan = AreaResampler::to(336, 189)
    .plan(1920, 1080)
    .expect("valid")
    .expect("non-identity");

  for (axis, src, out) in [(plan.h(), 1920, 336), (plan.v(), 1080, 189)] {
    assert_eq!(axis.out_len(), out);
    let mut prev_start = None;
    for j in 0..out {
      let (start, weights) = axis.span(j);
      assert_eq!(weights.iter().sum::<usize>(), src, "span {j} sum");
      assert!(weights.iter().all(|&w| w > 0), "span {j} zero tap");
      assert!(weights.len() == 6 || weights.len() == 7, "span {j} taps");
      if let Some(p) = prev_start {
        assert!(start > p, "span {j} start not increasing");
      }
      prev_start = Some(start);
    }
    let (last_start, last_weights) = axis.span(out - 1);
    assert_eq!(last_start + last_weights.len(), src);
    assert_eq!(axis.span(0).0, 0);
  }
}

/// Direct (non-separable) 2-D area reference: per output pixel,
/// sum `w_y * w_x * sample` over the full source block via the plan's
/// own spans, then round-half-up by the `src_w * src_h` denominator.
/// The streaming engine must reproduce this exactly.
#[cfg(feature = "yuv-planar")]
fn direct_area_2d(plan: &ResamplePlan, src: &[u8], channels: usize) -> std::vec::Vec<u8> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let denom = (src_w as u64) * (plan.src_h() as u64);
  let mut out = std::vec![0u8; out_w * out_h * channels];
  for oy in 0..out_h {
    let (vy, vw) = plan.v().span(oy);
    for ox in 0..out_w {
      let (hx, hw) = plan.h().span(ox);
      for c in 0..channels {
        let mut acc = 0u64;
        for (dy, &wy) in vw.iter().enumerate() {
          for (dx, &wx) in hw.iter().enumerate() {
            let s = src[((vy + dy) * src_w + hx + dx) * channels + c] as u64;
            acc += (wy as u64) * (wx as u64) * s;
          }
        }
        out[(oy * out_w + ox) * channels + c] = ((acc + denom / 2) / denom) as u8;
      }
    }
  }
  out
}

#[cfg(feature = "yuv-planar")]
fn stream_collect(plan: &ResamplePlan, src: &[u8], channels: usize) -> std::vec::Vec<u8> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let mut stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), channels)
    .expect("realistic geometry");
  let mut out = std::vec![0u8; out_w * out_h * channels];
  let mut emitted = std::vec::Vec::new();
  for y in 0..plan.src_h() {
    let row = &src[y * src_w * channels..(y + 1) * src_w * channels];
    stream
      .feed_row(y, row, true, |oy, finalized| {
        emitted.push(oy);
        out[oy * out_w * channels..(oy + 1) * out_w * channels].copy_from_slice(finalized);
      })
      .expect("rows arrive in order");
  }
  assert_eq!(emitted, (0..out_h).collect::<std::vec::Vec<_>>());
  out
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_matches_direct_2d_reference_fractional() {
  let plan = AreaResampler::to(3, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let src: std::vec::Vec<u8> = (0..64u8).collect();
  assert_eq!(
    stream_collect(&plan, &src, 1),
    direct_area_2d(&plan, &src, 1)
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_matches_direct_2d_reference_multichannel() {
  let plan = AreaResampler::to(4, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  // 3 interleaved channels with distinct ramps.
  let mut src = std::vec![0u8; 8 * 8 * 3];
  for (i, px) in src.chunks_exact_mut(3).enumerate() {
    px[0] = i as u8;
    px[1] = (3 * i % 251) as u8;
    px[2] = 255 - i as u8;
  }
  assert_eq!(
    stream_collect(&plan, &src, 3),
    direct_area_2d(&plan, &src, 3)
  );
}

/// `u16` analogue of [`direct_area_2d`]: exact 2D area mean over the
/// full 16-bit sample range, so the reference exercises the wider
/// (`u64`) horizontal accumulator the `u8` path never reaches.
#[cfg(feature = "yuv-planar")]
fn direct_area_2d_u16(plan: &ResamplePlan, src: &[u16], channels: usize) -> std::vec::Vec<u16> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let denom = (src_w as u64) * (plan.src_h() as u64);
  let mut out = std::vec![0u16; out_w * out_h * channels];
  for oy in 0..out_h {
    let (vy, vw) = plan.v().span(oy);
    for ox in 0..out_w {
      let (hx, hw) = plan.h().span(ox);
      for c in 0..channels {
        let mut acc = 0u64;
        for (dy, &wy) in vw.iter().enumerate() {
          for (dx, &wx) in hw.iter().enumerate() {
            let s = src[((vy + dy) * src_w + hx + dx) * channels + c] as u64;
            acc += (wy as u64) * (wx as u64) * s;
          }
        }
        out[(oy * out_w + ox) * channels + c] = ((acc + denom / 2) / denom) as u16;
      }
    }
  }
  out
}

#[cfg(feature = "yuv-planar")]
fn stream_collect_u16(plan: &ResamplePlan, src: &[u16], channels: usize) -> std::vec::Vec<u16> {
  let (out_w, out_h) = plan.out_dims();
  let src_w = plan.src_w();
  let mut stream = AreaStream::<u16>::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), channels)
    .expect("realistic geometry");
  let mut out = std::vec![0u16; out_w * out_h * channels];
  let mut emitted = std::vec::Vec::new();
  for y in 0..plan.src_h() {
    let row = &src[y * src_w * channels..(y + 1) * src_w * channels];
    stream
      .feed_row(y, row, true, |oy, finalized| {
        emitted.push(oy);
        out[oy * out_w * channels..(oy + 1) * out_w * channels].copy_from_slice(finalized);
      })
      .expect("rows arrive in order");
  }
  assert_eq!(emitted, (0..out_h).collect::<std::vec::Vec<_>>());
  out
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_u16_matches_direct_2d_reference_fractional() {
  let plan = AreaResampler::to(3, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  // Full-range ramp 0, 1040, …, 65520 — samples above 255 prove the
  // u16 horizontal accumulator carries the high bits a u8 path drops.
  let src: std::vec::Vec<u16> = (0..64u16).map(|i| i * 1040).collect();
  assert_eq!(
    stream_collect_u16(&plan, &src, 1),
    direct_area_2d_u16(&plan, &src, 1)
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_u16_matches_direct_2d_reference_multichannel() {
  let plan = AreaResampler::to(4, 3)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  // 3 interleaved channels with distinct full-range ramps.
  let mut src = std::vec![0u16; 8 * 8 * 3];
  for (i, px) in src.chunks_exact_mut(3).enumerate() {
    px[0] = (i as u16) * 1000;
    px[1] = ((7 * i) % 211) as u16 * 300;
    px[2] = 65535 - (i as u16) * 1000;
  }
  assert_eq!(
    stream_collect_u16(&plan, &src, 3),
    direct_area_2d_u16(&plan, &src, 3)
  );
}

/// Integer-ratio 2× downscale: every output pixel is an exact
/// round-half-up mean of its 2×2 source block. Hand-computed
/// expectations pin the u16 finalize independent of the reference.
#[cfg(feature = "yuv-planar")]
#[test]
fn stream_u16_exact_2x2_block_mean() {
  let plan = AreaResampler::to(2, 2)
    .plan(4, 4)
    .expect("valid")
    .expect("non-identity");
  // 4×4 source; each 2×2 quadrant chosen so its mean rounds half-up.
  let src: std::vec::Vec<u16> = std::vec![
    60000, 60002, 10, 11, //
    60004, 60006, 12, 13, //
    1, 2, 65535, 65533, //
    3, 5, 65531, 65529, //
  ];
  // Quadrant means: (60000+60002+60004+60006)/4 = 60003;
  // (10+11+12+13)/4 = 11.5 -> 12; (1+2+3+5)/4 = 2.75 -> 3;
  // (65535+65533+65531+65529)/4 = 65532.
  assert_eq!(
    stream_collect_u16(&plan, &src, 1),
    std::vec![60003u16, 12, 3, 65532]
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_identity_vertical_axis_emits_every_row() {
  // 8x8 -> 3x8: vertical axis is identity-sized, so every source row
  // finalizes exactly one output row.
  let plan = AreaResampler::to(3, 8)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let src: std::vec::Vec<u8> = (0..64u8).collect();
  assert_eq!(
    stream_collect(&plan, &src, 1),
    direct_area_2d(&plan, &src, 1)
  );
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_creation_fails_recoverably_on_huge_row_buffers() {
  // Row buffers that can never be reserved must surface
  // AllocationFailed via capacity overflow — the magnitude has to sit
  // in the CAPACITY-OVERFLOW zone for the SMALLEST arena element
  // (h_tmp is u32, so entries x 4 bytes must exceed isize::MAX);
  // anything smaller is a real near-exabyte allocator request that
  // hosts refuse but sanitizers and miri abort on. Real spans,
  // magnitude driven through the channel count: 4 outputs x
  // usize::MAX / 16 channels is representable as a length but puts
  // every per-element arena past capacity.
  let plan = AreaResampler::to(4, 4)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let err = AreaStream::<u8>::new(
    plan.h(),
    plan.v(),
    plan.src_w(),
    plan.src_h(),
    usize::MAX / 16,
  )
  .unwrap_err();
  assert!(err.is_allocation_failed(), "got {err:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn area_chroma_420_reports_allocation_failure_as_such() {
  // Allocator refusal on the paired vertical axis must surface as
  // AllocationFailed, not be misclassified as geometry Overflow. The
  // magnitude must sit in the CAPACITY-OVERFLOW zone (entries x 8
  // bytes above isize::MAX), where try_reserve fails deterministically
  // WITHOUT touching the allocator — a smaller huge request would be
  // a real multi-exabyte allocation that hosts refuse but miri aborts
  // on. usize::MAX/8 starts entries overflow capacity while every
  // arithmetic check still passes.
  let err = ResamplePlan::area_chroma_420(4, 8, 2, usize::MAX / 8).unwrap_err();
  assert!(err.is_allocation_failed(), "got {err:?}");
}

#[cfg(feature = "yuv-planar")]
#[test]
fn area_halved_weights_the_odd_tail_row_by_its_luma_coverage() {
  // 4:2:0 vertical pairing over luma height 9 -> 3 outputs: chroma
  // cells span luma-row pairs except the single-row tail. On the x3
  // grid: out0 = (0, [6, 3]), out1 = (1, [3, 6]), out2 = (3, [6, 3])
  // — the tail cell carries HALF a full cell's weight, and every span
  // sums to the luma height (the denominator).
  let spans = AxisSpans::area_halved(9, 3).expect("valid");
  assert_eq!(spans.out_len(), 3);
  assert_eq!(spans.span(0), (0, &[6usize, 3][..]));
  assert_eq!(spans.span(1), (1, &[3usize, 6][..]));
  assert_eq!(spans.span(2), (3, &[6usize, 3][..]));

  // Even luma heights reduce to uniform double-width cells: identical
  // normalized weighting to the plain chroma-grid spans, scaled x2.
  let even = AxisSpans::area_halved(8, 4).expect("valid");
  for j in 0..4 {
    assert_eq!(even.span(j), (j, &[8usize][..]));
  }
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_rejects_out_of_order_duplicate_and_skipped_rows() {
  let plan = AreaResampler::to(4, 4)
    .plan(8, 8)
    .expect("valid")
    .expect("non-identity");
  let row = [0u8; 8];
  let mut stream = AreaStream::new(plan.h(), plan.v(), plan.src_w(), plan.src_h(), 1).unwrap();

  // Out of order from the start: row 1 before row 0.
  let err = stream.feed_row(1, &row, true, |_, _| {}).unwrap_err();
  match err {
    ResampleError::OutOfSequenceRow(e) => {
      assert_eq!((e.expected(), e.got()), (0, 1));
    }
    other => panic!("expected OutOfSequenceRow, got {other:?}"),
  }
  // The rejected row must not have touched stream state.
  stream.feed_row(0, &row, true, |_, _| {}).unwrap();

  // Duplicate.
  let err = stream.feed_row(0, &row, true, |_, _| {}).unwrap_err();
  assert!(err.is_out_of_sequence_row());

  // Skipped.
  let err = stream.feed_row(2, &row, true, |_, _| {}).unwrap_err();
  match err {
    ResampleError::OutOfSequenceRow(e) => {
      assert_eq!((e.expected(), e.got()), (1, 2));
    }
    other => panic!("expected OutOfSequenceRow, got {other:?}"),
  }

  // reset() restarts the sequence.
  stream.reset();
  stream.feed_row(0, &row, true, |_, _| {}).unwrap();
}

#[test]
fn round_div_half_up_is_exact_and_overflow_free() {
  // Equivalence with (a + d/2) / d on small values, both parities.
  for d in 1u64..=9 {
    for a in 0u64..=200 {
      assert_eq!(round_div_half_up(a, d), (a + d / 2) / d, "a={a} d={d}");
    }
  }
  // Boundary: the naive form would wrap; the q/r form must not.
  let d = u64::MAX / 255;
  assert_eq!(round_div_half_up(d * 255, d), 255);
  assert_eq!(round_div_half_up(u64::MAX, u64::MAX), 1);
  assert_eq!(round_div_half_up(u64::MAX - 1, u64::MAX), 1);
  assert_eq!(round_div_half_up(u64::MAX / 2, u64::MAX), 0);
  assert_eq!(round_div_half_up(u64::MAX / 2 + 1, u64::MAX), 1);
}

#[cfg(feature = "yuv-planar")]
#[test]
fn stream_constant_input_is_constant() {
  let plan = AreaResampler::to(3, 2)
    .plan(7, 5)
    .expect("valid")
    .expect("non-identity");
  let src = std::vec![173u8; 7 * 5];
  assert!(stream_collect(&plan, &src, 1).iter().all(|&v| v == 173));
}

#[cfg(any(feature = "yuv-planar", feature = "rgb"))]
/// Same LCG as `ci/gen_cv2_goldens.py`: the parity sources are
/// synthesized identically on both sides, so the fixture carries only
/// cv2's outputs.
fn lcg_fill(buf: &mut [u8], seed: u32) {
  let mut state = seed;
  for b in buf {
    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *b = (state >> 8) as u8;
  }
}

#[cfg(any(feature = "yuv-planar", feature = "rgb"))]
#[test]
fn area_matches_cv2_inter_area_within_one_lsb() {
  // cv2's INTER_AREA uses its own fixed-point/float internals, so the
  // contract is +-1 LSB against our exact integer area mean — checked
  // across integer and fractional ratios, gray and interleaved RGB.
  for &(src_w, src_h, out_w, out_h, channels, seed, golden) in super::cv2_goldens::ALL {
    let mut src = std::vec![0u8; src_w * src_h * channels];
    lcg_fill(&mut src, seed);
    let plan = AreaResampler::to(out_w, out_h)
      .plan(src_w, src_h)
      .expect("valid downscale")
      .expect("non-identity");
    let mut stream =
      AreaStream::new(plan.h(), plan.v(), src_w, src_h, channels).expect("realistic geometry");
    let mut ours = std::vec![0u8; out_w * out_h * channels];
    for y in 0..src_h {
      let row = &src[y * src_w * channels..(y + 1) * src_w * channels];
      stream
        .feed_row(y, row, true, |oy, finalized| {
          ours[oy * out_w * channels..(oy + 1) * out_w * channels].copy_from_slice(finalized);
        })
        .expect("rows in order");
    }
    assert_eq!(golden.len(), ours.len(), "{src_w}x{src_h}->{out_w}x{out_h}");
    for (i, (a, b)) in ours.iter().zip(golden.iter()).enumerate() {
      assert!(
        a.abs_diff(*b) <= 1,
        "{src_w}x{src_h}->{out_w}x{out_h} c{channels} idx {i}: ours {a} vs cv2 {b}"
      );
    }
  }
}

#[test]
fn plan_error_display_names_geometry() {
  let upscale = ResampleError::UpscaleUnsupported(UpscaleUnsupported::new(1920, 1080, 3840, 2160));
  assert!(upscale.is_upscale_unsupported());
  let msg = format!("{upscale}");
  assert!(msg.contains("1920x1080"), "{msg}");
  assert!(msg.contains("3840x2160"), "{msg}");

  let zero = ResampleError::ZeroOutputDimension(ZeroOutputDimension::new(0, 189));
  assert!(zero.is_zero_output_dimension());
  assert!(format!("{zero}").contains("0x189"));

  let overflow = ResampleError::Overflow(PlanGeometry::new(usize::MAX, 2, 3, 1));
  let alloc = ResampleError::AllocationFailed(PlanGeometry::new(usize::MAX, 1, 1, 1));
  assert!(alloc.is_allocation_failed());
  assert!(format!("{alloc}").contains("allocation"));
  assert!(overflow.is_overflow());
  let msg = format!("{overflow}");
  assert!(msg.contains("3x1"), "{msg}");
}

#[test]
fn area_plans_products_beyond_32_bits() {
  // 70_000 x 62_000 exceeds u32::MAX: span coordinates run in u64, so
  // 32-bit targets plan any geometry whose buffers fit (regression
  // cover runs under the miri i686 jobs).
  let plan = AreaResampler::to(62_000, 1)
    .plan(70_000, 2)
    .expect("coordinates must not overflow 32-bit usize")
    .expect("non-identity");
  let h = plan.h();
  assert_eq!(h.out_len(), 62_000);
  let (first_start, first) = h.span(0);
  assert_eq!(first_start, 0);
  assert_eq!(first.iter().sum::<usize>(), 70_000);
  let (last_start, last) = h.span(61_999);
  assert_eq!(last_start + last.len(), 70_000);
}

#[test]
fn area_taps_formula_is_exact() {
  // Weight-arena size has the closed form src + out - gcd(src, out):
  // every source cell contributes once, plus one shared straddle cell
  // per unaligned output boundary. Cross-checked against the arenas
  // the builder actually materializes.
  for src in 1..=24usize {
    for out in 1..=src {
      let expected = AxisSpans::area_taps(src, out).unwrap();
      let plan = AreaResampler::to(out, 1).plan(src, 1);
      if out == src {
        // Identity plans short-circuit before building spans; check
        // the formula degenerates to src.
        assert_eq!(expected, src);
        continue;
      }
      let plan = plan.expect("valid").expect("non-identity");
      let total: usize = (0..out).map(|j| plan.h().span(j).1.len()).sum();
      assert_eq!(total, expected, "src={src} out={out}");
    }
  }
  // Pure arithmetic at hostile magnitudes — no allocation involved.
  assert_eq!(AxisSpans::area_taps(usize::MAX, 1), Some(usize::MAX));
}

#[test]
fn area_tiny_output_huge_source_fails_recoverably() {
  // A hostile source dimension from untrusted metadata must surface a
  // structured error, not abort inside infallible allocation: the
  // weight arena for usize::MAX taps trips Vec::try_reserve_exact
  // capacity overflow deterministically.
  let err = AreaResampler::to(1, 1).plan(usize::MAX, 1).unwrap_err();
  assert!(err.is_allocation_failed(), "got {err:?}");
}

#[test]
fn area_overflow_rejected() {
  // Hostile dimensions must surface a structured error, and which one
  // is pointer-width-dependent. On 64-bit targets the u64 span-grid
  // product src_w * out_w overflows and is rejected as Overflow. On
  // 32-bit targets that product always fits u64 — the grid
  // coordinates were widened precisely so 32-bit dims cannot overflow
  // them — so the same call runs on to the span arenas, whose
  // capacity-overflow reservation (entries times element size above
  // isize::MAX, no allocator touch) surfaces as AllocationFailed.
  let err = AreaResampler::to(usize::MAX / 2, 1)
    .plan(usize::MAX - 1, 1)
    .unwrap_err();
  #[cfg(target_pointer_width = "64")]
  assert!(err.is_overflow(), "got {err:?}");
  #[cfg(target_pointer_width = "32")]
  assert!(err.is_allocation_failed(), "got {err:?}");
}

#[cfg(any(feature = "yuv-planar", feature = "rgb"))]
#[test]
fn h_pass_simd_matches_scalar_bit_exact() {
  // Differential against the scalar reference at the h_tmp level —
  // the u8 finalize divide downstream could mask a small H-sum error,
  // so the raw u32 sums are compared directly. Geometries cover
  // single-chunk spans with tails (1920->336: 6-7 taps), multi-chunk
  // spans (640->7: ~92 taps), 2-tap spans whose final span lands on
  // the row-end staging path (4096->4095), tiny all-staged rows, and
  // an output width past the u16 weight bound (70000->66000), where
  // the dispatcher falls back to scalar.
  // Miri keeps every allocation's address live for the whole process,
  // and the i686 cell has 4 GB of address space shared by the entire
  // suite — the large geometries would exhaust it (the failure then
  // surfaces in whatever unrelated test runs last). The small cases
  // still cover chunked spans, staged row-ends, and both channel
  // counts under Miri; the full list runs on every native lane.
  let cases: &[(usize, usize, usize)] = if cfg!(miri) {
    &[(64, 7, 1), (64, 7, 3), (12, 11, 3), (8, 3, 1), (5, 4, 3)]
  } else {
    &[
      (1920, 336, 1),
      (1920, 336, 3),
      (640, 7, 1),
      (640, 7, 3),
      (4096, 4095, 1),
      (4096, 4095, 3),
      (5, 4, 3),
      (8, 3, 1),
      (70_000, 66_000, 1),
      // Wide-path boundaries for the AVX kernels (padded span length =
      // taps rounded up to a multiple of 8): the AVX-512 step consumes
      // 32 taps, AVX2 consumes 16, and the remainder falls to the
      // 128-bit step. These ratios place the remainder at each of 0 /
      // 8 / 16 / 24 taps after the wide chunks, for both channel
      // counts. Validated on real AVX2 hardware by the avx2-max
      // coverage job and under SDE for AVX-512.
      (256, 8, 1), // padded 32: one AVX-512 chunk, no remainder
      (256, 8, 3),
      (264, 8, 1), // padded 40: AVX-512 32 + 8 remainder
      (264, 8, 3),
      (336, 8, 1), // padded 48: AVX-512 32 + 16 remainder
      (336, 8, 3),
      (400, 8, 1), // padded 56: AVX-512 32 + 24 remainder
      (400, 8, 3),
      (120, 8, 1), // padded 16: one AVX2 chunk; AVX-512 tail-only
      (120, 8, 3),
      (160, 8, 1), // padded 24: AVX2 16 + 8; AVX-512 tail-only
      (160, 8, 3),
    ]
  };
  for &(src_w, out_w, channels) in cases {
    let plan = AreaResampler::to(out_w, 1)
      .plan(src_w, 2)
      .expect("valid geometry")
      .expect("strict downscale");
    let mut scalar = AreaStream::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut simd = AreaStream::new(plan.h(), plan.v(), src_w, 2, channels).unwrap();
    let mut row = std::vec![0u8; src_w * channels];
    for y in 0..2usize {
      lcg_fill(&mut row, (src_w * 31 + out_w * 7 + channels + y) as u32);
      let mut scalar_rows = std::vec::Vec::new();
      let mut simd_rows = std::vec::Vec::new();
      scalar
        .feed_row(y, &row, false, |oy, r| {
          scalar_rows.push((oy, r.to_vec()));
        })
        .unwrap();
      simd
        .feed_row(y, &row, true, |oy, r| {
          simd_rows.push((oy, r.to_vec()));
        })
        .unwrap();
      assert_eq!(
        scalar.h_tmp, simd.h_tmp,
        "h_tmp diverged: src_w={src_w} out_w={out_w} c={channels} y={y}"
      );
      assert_eq!(
        scalar.acc, simd.acc,
        "acc diverged: src_w={src_w} out_w={out_w} c={channels} y={y}"
      );
      assert_eq!(
        scalar_rows, simd_rows,
        "emitted rows diverged: src_w={src_w} out_w={out_w} c={channels}"
      );
    }
  }
}
