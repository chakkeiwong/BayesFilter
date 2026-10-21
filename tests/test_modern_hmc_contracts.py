"""Independent host-contract and evidence-policy regressions; no GPU claims."""
from copy import deepcopy
from dataclasses import replace
import ast
from pathlib import Path

import pytest

from bayesfilter.inference.hmc_acceptance_protocol import (
    HMCReplicatedAcceptancePolicy, PROTOCOL_SCHEMA, ONE_SIDED_PROTOCOL_SCHEMA,
)
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCCandidateSetScope, HMCTuningCandidateRecord,
)
from bayesfilter.inference.hmc_transition_spec import (
    HMCTransitionProfile, FrozenTransitionSpec, candidate_gradient_work,
)
from bayesfilter.testing.inference_validation.modern_hmc_scope import (
    NONLINEAR_COMPARISON_TARGETS, initial_comparison_matrix, validate_comparison_matrix,
)


def policy(**changes):
    return HMCReplicatedAcceptancePolicy(trial_num_results=64, discarded_prefix=0,
        base_repetitions=16, max_repetitions=256, max_candidates=2,
        search_family_alpha=.05, verification_family_alpha=.05,
        diagnostic_family_alpha=.05, temporal_tolerance=.1, **changes)


def scope(**changes):
    return HMCCandidateSetScope(scope_id="test", search_id="one", target_signature="target",
        mass_signature="identity", coordinate_system="ordinary", start_bank_signature="bank",
        warmup_protocol="discarded", **changes)


def test_legacy_payload_is_not_silently_upgraded():
    p = policy()
    assert p.payload()["schema"] == PROTOCOL_SCHEMA
    assert "qualification_mode" not in p.payload()
    assert HMCReplicatedAcceptancePolicy.from_payload(p.payload()) == p
    s = scope()
    c = HMCTuningCandidateRecord.create(s, leapfrog_steps=5, epsilon=.4, creation_ordinal=1)
    assert s.payload()["schema"].endswith("v1")
    assert c.payload()["schema"].endswith("v1")
    assert "transition_spec" not in c.payload()
    assert candidate_gradient_work(c) == 6


def test_randomized_policy_identity_and_checked_roundtrip():
    s = scope(transition_profile=HMCTransitionProfile("hmc", "uniform_duration"))
    c = HMCTuningCandidateRecord.create(s, leapfrog_steps=5, epsilon=.4, creation_ordinal=1)
    assert c.transition_spec.max_duration == 2.
    assert c.payload()["schema"].endswith("v2")
    assert HMCCandidateSetScope.from_payload(s.payload()) == s
    assert HMCTuningCandidateRecord.from_payload(s, c.payload()) == c
    fixed = replace(s, transition_profile=HMCTransitionProfile("hmc", "fixed"))
    with pytest.raises(ValueError, match="identity"):
        HMCTuningCandidateRecord.from_payload(fixed, c.payload())
    for field, value in (("epsilon", .5), ("max_steps", 6), ("max_duration", 3.)):
        bad = deepcopy(c.payload()); bad["transition_spec"][field] = value
        with pytest.raises(ValueError):
            candidate_gradient_work(bad)


@pytest.mark.parametrize("changes", [{"epsilon":float("nan")}, {"epsilon":0},
    {"max_steps":True}, {"max_steps":0}, {"max_steps":2**31-1}, {"epsilon":1e308, "max_steps":20}])
def test_invalid_transition_cannot_acquire_identity(changes):
    with pytest.raises((ValueError, TypeError)):
        FrozenTransitionSpec(HMCTransitionProfile("hmc", "uniform_duration"),
                             **({"epsilon":.5,"max_steps":4} | changes))


def test_one_sided_policy_is_explicit_and_rejects_relabeling():
    p = policy(qualification_mode="lower_bound", qualification_region=(.5, 1.))
    assert p.payload()["schema"] == ONE_SIDED_PROTOCOL_SCHEMA
    assert HMCReplicatedAcceptancePolicy.from_payload(p.payload()) == p
    bad = p.payload(); bad["schema"] = PROTOCOL_SCHEMA
    with pytest.raises(ValueError):
        HMCReplicatedAcceptancePolicy.from_payload(bad)
    with pytest.raises(ValueError):
        policy(qualification_region=(.5, 1.))
    with pytest.raises(ValueError):
        policy(qualification_mode="lower_bound")


def test_high_acceptance_qualifies_without_preferred_target_repair():
    from bayesfilter.inference.hmc_acceptance_statistics import replicated_acceptance_statistics
    p = policy(qualification_mode="lower_bound", qualification_region=(.5, 1.))
    report = replicated_acceptance_statistics([[1.]*4]*256, policy=p,
        stage="verification", evidence_rungs=(1,4,16), jit_compile=False)
    assert report["acceptance_decision"] == "passed"
    assert report["decision_reason"] == "simultaneous_start_lower_bound_qualification"
    # A good pool cannot hide the first start's low acceptance.
    bad = replicated_acceptance_statistics([[.1,1.,1.,1.]]*256, policy=p,
        stage="verification", evidence_rungs=(1,4,16), jit_compile=False)
    assert bad["acceptance_decision"] != "passed"


def test_modern_preflight_counts_status_and_verification_without_changing_legacy_cost():
    from bayesfilter.inference.hmc_acceptance_protocol import replicated_evidence_preflight
    p = policy(qualification_mode="lower_bound", qualification_region=(.5, 1.))
    kwargs = dict(evidence_rungs=(1,), candidate_cap=2, leapfrog_steps=(3, 5),
                  max_gradient_work=24000)
    legacy = replicated_evidence_preflight(p, **kwargs)
    modern = replicated_evidence_preflight(p, transition_profile=HMCTransitionProfile("mams", "uniform_duration"), **kwargs)
    # Four starts * 64 transitions * 16 repetitions * (4+6) score calls,
    # doubled for measurement plus verification, then again for status calls.
    assert legacy["initial_cohort_mandatory_gradient_work"] == 81920
    assert modern["initial_cohort_mandatory_gradient_work"] == 163840
    assert modern["schema"].endswith("v2")
    assert not modern["first_cohort_fundable"]


def test_required_nonlinear_inventory_cannot_drop_or_merge_cells():
    rows = initial_comparison_matrix()
    assert len(rows) == 20
    assert not validate_comparison_matrix(rows)["all_runs_completed"]
    with pytest.raises(ValueError, match="missing"):
        validate_comparison_matrix(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_comparison_matrix(rows+[rows[0]])
    bad = deepcopy(rows); bad[0]["status"] = "completed"
    with pytest.raises(ValueError, match="evidence"):
        validate_comparison_matrix(bad)
    assert {r["target"] for r in rows if r["target"].startswith("NL-PP")} == {
        "NL-PP-UKF", "NL-PP-SGQF"}


def test_registered_nonlinear_factories_exist_without_importing_historical_recipes():
    root = Path(__file__).resolve().parents[1]
    for target in NONLINEAR_COMPARISON_TARGETS:
        source = root / (target.module.replace(".", "/") + ".py")
        assert source.is_file()
        if target.factory is not None:
            tree = ast.parse(source.read_text())
            assert target.factory in {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
