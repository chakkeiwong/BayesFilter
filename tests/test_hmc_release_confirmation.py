"""Independent statistical and missing/partial-inventory controls for reporting."""
import copy
from fractions import Fraction
import json
import math

import pytest
from scipy.stats import beta

from scripts.analyze_hmc_v7_confirmation import (
    FAMILIES, PRIMARY_L, binomial_upper_tail, delivery_report, one_sided_lower_bound, price_report,
)


def complete_model(family="lgssm_qr", seed=(404, 0)):
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration
    configuration = full_search_configuration(family, seed=seed, wall_seconds=1800,
                                              classification="confirmation")
    return dict(schema="bayesfilter.replicated_acceptance_model_result.v1",
                classification="confirmation", case_id="confirmation-"+family,
                sampling_streams=list(seed),
                completion_status="complete", checkpoint_recomputed=True, expectation_met=True,
                verified_candidate_ids=["member"], candidate_states={"member": "verified"},
                candidates=[dict(leapfrog_steps=l) for l in PRIMARY_L],
                search=json.loads(json.dumps(configuration["search"])))


def inventory(n=32):
    slots, outcomes = [], []
    for f, family in enumerate(FAMILIES):
        for i in range(n):
            slot = dict(slot_id=f"{family}-{i}", family=family, seed=[404, 100*f+i])
            slots.append(slot)
            outcomes.append(dict(**slot, source_manifest_sha256="source", exit_code=0,
                                 model_result=complete_model(family, seed=slot["seed"])))
    return slots, outcomes


@pytest.mark.parametrize("successes", range(33))
def test_rational_lower_bound_matches_independent_beta_reference(successes):
    actual = one_sided_lower_bound(successes, 32, Fraction(1, 60))
    expected = 0. if successes == 0 else beta.ppf(1/60, successes, 33-successes)
    # Independent float64 library reference, 64 ulps of absolute comparison
    # slack; the reporting decision itself uses exact rational inequalities.
    assert abs(actual-expected) <= 64*math.ulp(1.)
    if successes:
        assert binomial_upper_tail(successes, 32, Fraction(actual)) <= Fraction(1, 60)


def test_exact_decision_and_analytic_extremes():
    assert one_sided_lower_bound(0, 32, "1/60") == 0.
    assert one_sided_lower_bound(32, 32, "1/60") == pytest.approx((1/60)**(1/32), abs=2e-15)
    assert binomial_upper_tail(32, 32, "0.8") == Fraction(4, 5)**32
    assert binomial_upper_tail(31, 32, "0.8") < Fraction(1, 60)
    assert binomial_upper_tail(30, 32, "0.8") > Fraction(1, 60)


@pytest.mark.parametrize("bad", [-1, 33, True, .5])
def test_invalid_binomial_counts_are_rejected(bad):
    with pytest.raises(ValueError):
        one_sided_lower_bound(bad, 32, ".05")


def test_missing_and_timed_out_slots_remain_in_original_denominator():
    slots, outcomes = inventory()
    outcomes.pop(0)
    outcomes[0]["exit_code"] = 124
    outcomes[0]["success"] = True  # A caller Boolean cannot turn a timeout into delivery.
    report = delivery_report(slots=slots, outcomes=outcomes, source_manifest_sha256="source")
    assert report["original_denominator"] == 96
    assert report["families"][0]["planned"] == 32
    assert report["families"][0]["successes"] == 30
    assert not report["delivery_criterion_passed"] and not report["all_slots_reported"]
    assert report["dispositions"]["timeout"] == report["dispositions"]["missing"] == 1
    assert not report["release_ready"]


def test_all_successes_and_one_failure_per_family_use_declared_confidence():
    slots, outcomes = inventory()
    good = delivery_report(slots=slots, outcomes=outcomes, source_manifest_sha256="source")
    assert good["delivery_criterion_passed"] and good["all_slots_reported"]
    assert all(row["one_sided_lower_bound"] == pytest.approx((1/60)**(1/32)) for row in good["families"])
    for i in (0, 32, 64): outcomes[i]["exit_code"] = 3
    report = delivery_report(slots=slots, outcomes=outcomes, source_manifest_sha256="source")
    assert report["delivery_criterion_passed"] and report["all_slots_reported"]
    assert all(row["planned"] == 32 and row["successes"] == 31 for row in report["families"])
    assert report["dispositions"]["resource_deferred"] == 3
    assert not report["release_ready"]


@pytest.mark.parametrize("damage", ["partial", "single_pair", "omitted_member", "unreconstructed", "old_policy", "wrong_model"])
def test_partial_or_wrong_procedure_cannot_count_as_delivery(damage):
    slots, outcomes = inventory(1)
    row = outcomes[0]["model_result"]
    if damage == "partial": row["completion_status"] = "partial_budget"
    elif damage == "single_pair": row["search"]["primary_l_grid"] = [1]
    elif damage == "omitted_member": row["candidate_states"]["other"] = "verified"
    elif damage == "unreconstructed": row["checkpoint_recomputed"] = False
    elif damage == "old_policy": row["search"]["replicated_acceptance_policy"]["schema"] = "v5"
    else: row["case_id"] = "fixture-nonlinear"
    outcomes[0]["success"] = True
    report = delivery_report(slots=slots, outcomes=outcomes, source_manifest_sha256="source")
    assert report["families"][0]["successes"] == 0


@pytest.mark.parametrize("damage", ["duplicate_slot", "duplicate_seed", "duplicate_result", "unknown_result", "source", "missing_family"])
def test_invalid_frozen_inventory_is_rejected(damage):
    slots, outcomes = inventory(2)
    if damage == "duplicate_slot": slots.append(copy.deepcopy(slots[0]))
    elif damage == "duplicate_seed": slots[1]["seed"] = slots[0]["seed"]
    elif damage == "duplicate_result": outcomes.append(copy.deepcopy(outcomes[0]))
    elif damage == "unknown_result": outcomes[0]["slot_id"] = "unknown"
    elif damage == "source": outcomes[0]["source_manifest_sha256"] = "different"
    else: slots = [r for r in slots if r["family"] != FAMILIES[-1]]
    with pytest.raises(ValueError):
        delivery_report(slots=slots, outcomes=outcomes, source_manifest_sha256="source")


@pytest.mark.parametrize("damage", ["repeated_model", "changed_seed", "missing_seed", "invalid_seed"])
def test_wrapper_metadata_cannot_relabel_the_model_random_stream(damage):
    slots, outcomes = inventory(2)
    if damage == "repeated_model":
        outcomes[1]["model_result"] = copy.deepcopy(outcomes[0]["model_result"])
    elif damage == "changed_seed":
        outcomes[1]["model_result"]["sampling_streams"][0] += 1
    elif damage == "missing_seed":
        del outcomes[1]["model_result"]["sampling_streams"]
    else:
        outcomes[1]["model_result"]["sampling_streams"] = None
    with pytest.raises(ValueError, match="model result seed"):
        delivery_report(slots=slots, outcomes=outcomes, source_manifest_sha256="source")


@pytest.mark.parametrize("field,value", [
    ("search_family_alpha", .5), ("verification_family_alpha", .5),
    ("diagnostic_family_alpha", .5), ("qualification_region", [.4, .95]),
    ("preferred_region", [.55, .85]), ("base_repetitions", 2),
    ("max_repetitions", 128), ("bet_fractions", [.5]), ("max_candidates", 2),
    ("min_movement_rate", 0.), ("min_normalized_return_displacement", False),
])
def test_changed_statistical_policy_does_not_price_the_declared_procedure(field, value):
    slots, outcomes = inventory(1)
    outcomes[0]["model_result"]["search"]["replicated_acceptance_policy"][field] = value
    report = delivery_report(slots=slots, outcomes=outcomes, source_manifest_sha256="source")
    assert report["families"][0]["successes"] == 0


@pytest.mark.parametrize("field,value", [
    ("epsilon_by_l", [[l, [.01]] for l in PRIMARY_L]),
    ("epsilon_refinement_factors", [1.]), ("repair_reserve_units", 1),
    ("candidate_reserve_units", 1), ("pilot_enabled", True),
    ("max_gradient_work", 1000), ("max_wall_time_seconds", float("inf")),
])
def test_changed_search_controls_cannot_count_as_full_profile_delivery(field, value):
    slots, outcomes = inventory(1)
    outcomes[0]["model_result"]["search"][field] = value
    report = delivery_report(slots=slots, outcomes=outcomes, source_manifest_sha256="source")
    assert report["families"][0]["successes"] == 0


def prices(tmp_path):
    paths = []
    for family in FAMILIES:
        root = tmp_path/family
        model = root/family/"model"
        model.mkdir(parents=True)
        (model/"result.json").write_text(json.dumps(complete_model(family)))
        outer = dict(schema="bayesfilter.hmc_v7_full_search_prices.v1", cases=[family], status="complete",
                     source_manifest_sha256="source", gpu_uuid="device", unstarted_cases=[], wall_seconds=11.,
                     attempts=[dict(case=family, exit_code=0, wall_seconds=10.,
                                    result=dict(full_search_delivered=True))])
        path = root/"result.json"
        path.write_text(json.dumps(outer))
        paths.append(path)
    return paths


def test_complete_forecast_charges_enclosing_overhead_and_never_authorizes(tmp_path):
    report = price_report(prices(tmp_path), replications_per_family=32, available_gpu_seconds=2000)
    assert report["point_forecast_seconds"] == 32*33
    assert report["point_forecast_fits_budget"]
    assert not report["confirmation_authorized"] and not report["release_ready"]


@pytest.mark.parametrize("damage", ["missing_family", "partial_model", "incomplete_outer", "missing_model"])
def test_unpriced_work_has_no_complete_forecast(tmp_path, damage):
    paths = prices(tmp_path)
    if damage == "missing_family": paths.pop()
    elif damage == "partial_model":
        model = paths[0].parent/FAMILIES[0]/"model"/"result.json"
        row = json.loads(model.read_text()); row["completion_status"] = "partial_budget"
        model.write_text(json.dumps(row))
    elif damage == "missing_model": (paths[0].parent/FAMILIES[0]/"model"/"result.json").unlink()
    else:
        row = json.loads(paths[0].read_text()); row["status"] = "incomplete_prices"
        paths[0].write_text(json.dumps(row))
    report = price_report(paths, replications_per_family=32, available_gpu_seconds=2000)
    assert report["point_forecast_seconds"] is None
    assert not report["point_forecast_fits_budget"]


@pytest.mark.parametrize("damage", ["source", "device", "duplicate", "nan", "negative", "undercharge"])
def test_incompatible_or_invalid_prices_are_rejected(tmp_path, damage):
    paths = prices(tmp_path)
    if damage == "duplicate": paths.append(paths[0])
    else:
        row = json.loads(paths[0].read_text())
        if damage == "source": row["source_manifest_sha256"] = "different"
        elif damage == "device": row["gpu_uuid"] = "different"
        elif damage == "nan": row["wall_seconds"] = float("nan")
        elif damage == "negative": row["wall_seconds"] = -1
        else: row["wall_seconds"] = 9.
        paths[0].write_text(json.dumps(row))
    with pytest.raises(ValueError):
        price_report(paths, replications_per_family=32, available_gpu_seconds=2000)
