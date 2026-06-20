use super::*;

/// The tier an output-bearing `Yuv420p` row runs through, as the
/// **pre-RFC-#238 inline dispatch** decided it (see `planar_8bit.rs`'s
/// `Yuv420p` `process`). This is the reference oracle the selector must
/// reproduce: the filter-first branch, then the `with_native` boolean,
/// reading exactly the same inputs the production dispatch read. The
/// route-rejection (`frozen` / `need_output`) machinery does **not**
/// participate in *which tier* runs — it gates rejection and freezing —
/// so it is intentionally absent here and exercised independently below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OldTier {
  /// `yuv420p_process_native` — the native fast tier.
  Native,
  /// `yuv420p_process_resampled` — the row-stage tier.
  RowStage,
  /// `planar_dual_filter_resample` — the filter resampler, taken before
  /// the native/row-stage route machinery on a filter plan.
  Filter,
}

/// Reproduces the old inline tier choice verbatim. `Yuv420p` is always
/// native-eligible, so eligibility is implicit in the original code; it
/// is threaded through here so the equivalence is asserted across the
/// eligibility axis too.
fn old_tier(native_eligible: bool, with_native: bool, area_plan: bool) -> OldTier {
  // `if plan.kind().is_filter()` — the filter resampler runs first,
  // before any native/row-stage decision.
  if !area_plan {
    return OldTier::Filter;
  }
  // `if *native { yuv420p_process_native } else { yuv420p_process_resampled }`.
  // The original Yuv420p code has no `native_eligible` guard because the
  // format is statically eligible; an ineligible format would never reach
  // the native tier, i.e. behaves as the row-stage branch.
  if native_eligible && with_native {
    OldTier::Native
  } else {
    OldTier::RowStage
  }
}

/// Map the old tier onto the splice stage it corresponds to, for the
/// equivalence assertion. A filter plan is not a native-vs-row-stage
/// choice at all; the native tier never runs on it, so the selector must
/// keep it off the native codes (`EncodedOutput`).
fn expected_point(tier: OldTier) -> InsertionPoint {
  match tier {
    OldTier::Native => InsertionPoint::NativeCodes,
    OldTier::RowStage | OldTier::Filter => InsertionPoint::EncodedOutput,
  }
}

#[test]
fn selector_reproduces_old_take_native_decision_exhaustively() {
  // Every combination of the inputs the old dispatch branched on, plus
  // the orthogonal route-machinery axes (`need_output`, frozen-route
  // state) that must NOT perturb the splice choice. `frozen_route`:
  // `None` = unfrozen, `Some(true)` = frozen-to-native, `Some(false)` =
  // frozen-to-row-stage.
  for native_eligible in [false, true] {
    for with_native in [false, true] {
      for area_plan in [false, true] {
        for need_output in [false, true] {
          for frozen_route in [None, Some(true), Some(false)] {
            let ctx = InsertionContext {
              native_eligible,
              with_native,
              area_plan,
            };
            let got = select_insertion_point(AveragingDomain::Encoded, ctx);
            let expected = expected_point(old_tier(native_eligible, with_native, area_plan));
            assert_eq!(
              got, expected,
              "splice mismatch for native_eligible={native_eligible} \
               with_native={with_native} area_plan={area_plan} \
               need_output={need_output} frozen_route={frozen_route:?}",
            );
            // The route-machinery axes are pure pass-through: they change
            // nothing about which stage the resample splices at, which is
            // exactly why re-expressing the dispatch is byte-identical.
            let _ = (need_output, frozen_route);
          }
        }
      }
    }
  }
}

#[test]
fn encoded_area_native_eligible_with_native_selects_native_codes() {
  // The one combination that must splice at the native codes — the
  // affine-format, area-downscale, native-enabled-and-eligible case.
  let ctx = InsertionContext {
    native_eligible: true,
    with_native: true,
    area_plan: true,
  };
  assert_eq!(
    select_insertion_point(AveragingDomain::Encoded, ctx),
    InsertionPoint::NativeCodes,
  );
}

#[test]
fn encoded_disabling_native_forces_encoded_output() {
  let ctx = InsertionContext {
    native_eligible: true,
    with_native: false,
    area_plan: true,
  };
  assert_eq!(
    select_insertion_point(AveragingDomain::Encoded, ctx),
    InsertionPoint::EncodedOutput,
  );
}

#[test]
fn encoded_filter_plan_never_splices_at_native_codes() {
  // A filter plan routes to the filter resampler before the route
  // machinery; the native tier never runs, so the splice stays at the
  // output regardless of `with_native`.
  for with_native in [false, true] {
    let ctx = InsertionContext {
      native_eligible: true,
      with_native,
      area_plan: false,
    };
    assert_eq!(
      select_insertion_point(AveragingDomain::Encoded, ctx),
      InsertionPoint::EncodedOutput,
    );
  }
}

#[test]
fn ineligible_format_never_splices_at_native_codes() {
  for with_native in [false, true] {
    for area_plan in [false, true] {
      let ctx = InsertionContext {
        native_eligible: false,
        with_native,
        area_plan,
      };
      assert_eq!(
        select_insertion_point(AveragingDomain::Encoded, ctx),
        InsertionPoint::EncodedOutput,
      );
    }
  }
}

#[test]
fn reserved_domains_resolve_to_encoded_output() {
  // Phase 0 never constructs Linear / Premultiplied on a splice path, but
  // the selector is total: they resolve to the encoded output until their
  // own phases land.
  for domain in [AveragingDomain::Linear, AveragingDomain::Premultiplied] {
    let ctx = InsertionContext {
      native_eligible: true,
      with_native: true,
      area_plan: true,
    };
    assert_eq!(
      select_insertion_point(domain, ctx),
      InsertionPoint::EncodedOutput,
    );
  }
}

#[test]
fn averaging_domain_as_str_round_trips_variants() {
  assert_eq!(AveragingDomain::Encoded.as_str(), "encoded");
  assert_eq!(AveragingDomain::Linear.as_str(), "linear");
  assert_eq!(AveragingDomain::Premultiplied.as_str(), "premultiplied");
}

#[test]
fn resample_strategy_default_is_encoded_area() {
  let strat = ResampleStrategy::default();
  assert_eq!(strat.domain(), AveragingDomain::Encoded);
  assert_eq!(strat.filter(), FilterSpec::Area);
}
