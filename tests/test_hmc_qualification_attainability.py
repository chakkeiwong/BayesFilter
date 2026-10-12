"""CPU diagnostic checks of evidence-design bounds, never sampler evidence."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from bayesfilter.inference.hmc_acceptance_statistics import (
    replicated_acceptance_statistics, replicated_qualification_attainability)


def policy(**changes):
    values = dict(trial_num_results=64, discarded_prefix=0, base_repetitions=16,
        max_repetitions=16, max_candidates=1, search_family_alpha=.05,
        verification_family_alpha=.05, diagnostic_family_alpha=.05,
        temporal_tolerance=.1, qualification_mode="lower_bound", qualification_region=(.1, 1.))
    values.update(changes)
    return HMCReplicatedAcceptancePolicy(**values)


def preflight(p, rungs=(1,)):
    return replicated_qualification_attainability(policy=p, evidence_rungs=rungs,
                                                   jit_compile=False)


@pytest.mark.parametrize("method,count", [
    ("bounded_betting_mixture_v1", 8), ("bounded_hoeffding_rungs_v1", 2)])
def test_impossible_budget_is_rejected_even_with_perfect_trials(method, count):
    p = policy(method=method, base_repetitions=count, max_repetitions=count)
    report = preflight(p)
    assert report["proven_impossible"]
    assert report["stages"]["measurement"]["maximum_start_lower_endpoint"] < .1
    actual = replicated_acceptance_statistics(tf.ones([count, 4], tf.float64), policy=p,
        stage="measurement", evidence_rungs=(1,), jit_compile=False)
    assert actual["acceptance_decision"] != "passed"


@pytest.mark.parametrize("method", ["bounded_betting_mixture_v1", "bounded_hoeffding_rungs_v1"])
def test_attainable_budget_does_not_claim_power_or_qualify_bad_trials(method):
    p = policy(method=method)
    report = preflight(p)
    assert not report["proven_impossible"]
    assert report["statistical_power"] == "not_established"
    assert report["tuning_artifact_authority"] is False
    for stage in ("measurement", "verification"):
        perfect = replicated_acceptance_statistics(tf.ones([16, 4], tf.float64), policy=p,
            stage=stage, evidence_rungs=(1,), jit_compile=False)
        bad = replicated_acceptance_statistics(tf.zeros([16, 4], tf.float64), policy=p,
            stage=stage, evidence_rungs=(1,), jit_compile=False)
        assert perfect["acceptance_decision"] == "passed"
        assert bad["acceptance_decision"] != "passed"


@pytest.mark.parametrize("method", ["bounded_betting_mixture_v1", "bounded_hoeffding_rungs_v1"])
def test_fresh_verification_budget_and_entire_candidate_allocation_matter(method):
    report = preflight(policy(method=method, verification_family_alpha=1e-14))
    assert not report["stages"]["measurement"]["proven_impossible"]
    assert report["stages"]["verification"]["proven_impossible"]
    assert report["proven_impossible"]
    many = preflight(policy(method=method, max_candidates=10_000_000_000))
    assert many["proven_impossible"]
    assert many["stages"]["measurement"]["one_sided_alpha"] < report["stages"]["measurement"]["one_sided_alpha"]


def test_unused_repetition_cap_does_not_supply_unscheduled_evidence():
    p = policy(base_repetitions=8, max_repetitions=64)
    assert preflight(p)["proven_impossible"]
    report = preflight(p, (1, 2))
    assert report["replication_targets"] == (8, 16)
    assert not report["proven_impossible"]
    with pytest.raises(ValueError, match="strictly increasing"):
        preflight(p, (1, 1))


def test_hoeffding_spends_error_across_all_planned_looks():
    p = policy(method="bounded_hoeffding_rungs_v1", base_repetitions=8, max_repetitions=16)
    report = preflight(p, (1, 2))
    for stage in ("measurement", "verification"):
        assert report["stages"][stage]["one_sided_alpha"] == p.one_sided_alpha(stage)/2


@pytest.mark.parametrize("method", ["bounded_betting_mixture_v1", "bounded_hoeffding_rungs_v1"])
def test_two_sided_report_never_claims_joint_attainability(method):
    p = policy(method=method, qualification_mode="two_sided", qualification_region=(.65, .75),
               base_repetitions=64, max_repetitions=64)
    report = preflight(p)
    assert not report["proven_impossible"]
    assert report["two_sided_joint_attainability"] == "not_checked"
    assert report["stages"]["measurement"]["minimum_start_upper_endpoint"] is not None
    actual = replicated_acceptance_statistics(tf.fill([64, 4], tf.constant(.7, tf.float64)),
        policy=p, stage="measurement", evidence_rungs=(1,), jit_compile=False)
    assert actual["acceptance_decision"] != "passed"
    assert preflight(replace(p, base_repetitions=2, max_repetitions=2))["proven_impossible"]


def test_frozen_consumer_stops_before_numerical_binding_validation(monkeypatch):
    from tests.test_modern_hmc_pipeline import setup
    from tests.test_hmc_candidate_set_execution import make_binding
    from bayesfilter.testing.inference_validation.modern_tuning_consumer import validate_frozen_search
    base, search = setup()
    p = replace(base.config.acceptance_policy, base_repetitions=8, max_repetitions=8)
    binding = make_binding(config=replace(base.config, acceptance_policy=p))
    search = replace(search, replicated_acceptance_policy=p)
    monkeypatch.setattr(binding, "validate", lambda: pytest.fail("target evaluated before preflight"))
    with pytest.raises(ValueError, match="qualification impossible"):
        validate_frozen_search(binding, search)


def test_full_model_driver_stops_before_target_preparation(tmp_path, monkeypatch):
    from scripts import run_modern_hmc_frozen_full_model as driver
    from tests.test_modern_hmc_pipeline import setup
    from bayesfilter.testing.inference_validation import modern_full_model_consumer
    base, search = setup()
    p = replace(base.config.acceptance_policy, base_repetitions=8, max_repetitions=8)
    design = dict(schema="bayesfilter.modern_frozen_full_model_design.v1", plan="diagnostic-control",
        target="NL-PP-SGQF", method="mams", execution=replace(base.config, acceptance_policy=p).payload(),
        search=replace(search, replicated_acceptance_policy=p).payload())
    path = tmp_path/"design.json"
    path.write_text(json.dumps(design))
    output = tmp_path/"run"
    monkeypatch.setattr(modern_full_model_consumer, "full_model_binding",
                        lambda **kw: pytest.fail("full target prepared before preflight"))
    with pytest.raises(ValueError, match="qualification impossible"):
        driver.worker(SimpleNamespace(design=path, output=output, cpu_reference=True,
                                      max_wall_seconds=60))
    report = json.loads((output/"qualification-preflight.json").read_text())
    assert report["proven_impossible"]
    assert json.loads((output/"manifest.json").read_text())["status"] == "repair_required"
