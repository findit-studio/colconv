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

  let overflow = ResampleError::Overflow(PlanOverflow::new(usize::MAX, 2, 3, 1));
  assert!(overflow.is_overflow());
  let msg = format!("{overflow}");
  assert!(msg.contains("3x1"), "{msg}");
}

#[test]
fn area_overflow_rejected() {
  // src_w * out_w cannot be represented: the span grid for the
  // horizontal axis would overflow usize.
  let err = AreaResampler::to(usize::MAX / 2, 1)
    .plan(usize::MAX - 1, 1)
    .unwrap_err();
  assert!(err.is_overflow(), "got {err:?}");
}
