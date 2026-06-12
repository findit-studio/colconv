use super::*;

#[test]
fn noop_plan_is_identity() {
  assert_eq!(NoopResampler.plan(1920, 1080), Ok(None));
}

#[test]
fn fixed_downscale_fixture_plans_requested_geometry() {
  let plan = test_support::FixedDownscale::new(4, 2)
    .plan(8, 8)
    .expect("fixture never fails")
    .expect("fixture always plans");
  assert_eq!(plan.out_dims(), (4, 2));
}

#[test]
fn always_fails_fixture_surfaces_zero_output_dimension() {
  assert!(matches!(
    test_support::AlwaysFails.plan(8, 8),
    Err(ResampleError::ZeroOutputDimension(_))
  ));
}

#[test]
fn plan_reports_output_geometry() {
  let plan = ResamplePlan::new(336, 189);
  assert_eq!(plan.out_w(), 336);
  assert_eq!(plan.out_h(), 189);
  assert_eq!(plan.out_dims(), (336, 189));
}

#[test]
fn upscale_unsupported_reports_both_geometries() {
  let payload = UpscaleUnsupported::new(1920, 1080, 3840, 2160);
  assert_eq!(payload.src_w(), 1920);
  assert_eq!(payload.src_h(), 1080);
  assert_eq!(payload.out_w(), 3840);
  assert_eq!(payload.out_h(), 2160);

  let err = ResampleError::UpscaleUnsupported(payload);
  assert!(err.is_upscale_unsupported());
  let msg = format!("{err}");
  assert!(msg.contains("1920x1080"), "{msg}");
  assert!(msg.contains("3840x2160"), "{msg}");
}

#[test]
fn zero_output_dimension_reports_geometry() {
  let payload = ZeroOutputDimension::new(0, 189);
  assert_eq!(payload.out_w(), 0);
  assert_eq!(payload.out_h(), 189);

  let err = ResampleError::ZeroOutputDimension(payload);
  assert!(err.is_zero_output_dimension());
  let msg = format!("{err}");
  assert!(msg.contains("0x189"), "{msg}");
}
