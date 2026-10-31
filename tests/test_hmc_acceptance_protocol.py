"""Independent invariants for finite-trial acceptance design and budgeting."""
from dataclasses import replace
import json
import os
import subprocess
import sys

import pytest

from bayesfilter.inference.hmc_acceptance_protocol import (
    HMCReplicatedAcceptancePolicy, batch_information_preflight,
    replicated_evidence_preflight,
)


def policy(**changes):
    return replace(HMCReplicatedAcceptancePolicy(
        trial_num_results=65, discarded_prefix=3, base_repetitions=4,
        max_repetitions=16, max_candidates=8, search_family_alpha=.05,
        verification_family_alpha=.05, diagnostic_family_alpha=.05,
        temporal_tolerance=.1), **changes)


def test_protocol_and_preflight_import_without_tensorflow():
    code = """
import sys
from bayesfilter.inference.hmc_acceptance_protocol import batch_information_preflight
from bayesfilter.testing.acceptance_decision_validation import preflight
assert 'tensorflow' not in sys.modules
assert not batch_information_preflight(draws=512, batch_size=11, min_batches=8)['feasible']
"""
    # The suite opts into loading the TensorFlow custom op at package import.
    # Check the normal lazy import in isolation, with a small environment that
    # cannot leak unrelated credentials through a subprocess traceback.
    env = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH")
           if key in os.environ}
    env.update(CUDA_VISIBLE_DEVICES="-1", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20, env=env)


def test_bgs_insufficient_design_and_exact_boundary():
    low = batch_information_preflight(draws=512, batch_size=11, min_batches=8)
    assert low["window_complete_batches"] == (11, 5)
    assert low["minimum_draws_fixed_batch"] == 704
    assert not low["feasible"]
    assert not batch_information_preflight(draws=703, batch_size=11, min_batches=8)["feasible"]
    assert batch_information_preflight(draws=704, batch_size=11, min_batches=8)["feasible"]
    assert "not_established" in low["statistical_sufficiency"]


def test_incremental_costs_include_prefix_all_starts_and_fresh_verification():
    report = replicated_evidence_preflight(policy(), evidence_rungs=(1, 2, 4),
        candidate_cap=8, leapfrog_steps=(3, 9), max_gradient_work=34815)
    # 68 transitions * 4 starts * 4 repetitions * (4 + 10) gradients * 2 stages.
    expected = 68 * 4 * 4 * (4 + 10) * 2
    assert report["initial_cohort_mandatory_gradient_work"] == expected
    assert report["incremental_repetitions"] == (4, 4, 8)
    assert sum(report["incremental_repetitions"]) == 16
    assert report["first_cohort_fundable"] is True
    assert not replicated_evidence_preflight(policy(), evidence_rungs=(1, 2, 4),
        candidate_cap=8, leapfrog_steps=(3, 9), max_gradient_work=expected-1)["first_cohort_fundable"]
    pilot = replicated_evidence_preflight(policy(), evidence_rungs=(1, 2, 4),
        candidate_cap=8, leapfrog_steps=(3, 9), pilot_enabled=True)
    assert pilot["initial_cohort_mandatory_gradient_work"] == expected * 3 // 2


def test_directional_pilot_and_measurement_cannot_spend_search_alpha_twice():
    p = policy()
    number_of_sided_assertions = 2 * 5 * p.max_candidates
    spent = sum(p.one_sided_alpha(stage) * number_of_sided_assertions
                for stage in ("pilot", "measurement"))
    assert spent == pytest.approx(p.search_family_alpha)
    assert p.one_sided_alpha("verification") * number_of_sided_assertions == pytest.approx(p.verification_family_alpha)


def test_identity_binds_estimand_and_policy_and_codec_rejects_relabeling():
    p = policy()
    payload = json.loads(json.dumps(p.payload()))
    assert HMCReplicatedAcceptancePolicy.from_payload(payload) == p
    for changed in (replace(p, trial_num_results=66), replace(p, discarded_prefix=4),
                    replace(p, start_weights=(.125, .125, .25, .5)),
                    replace(p, verification_family_alpha=.04)):
        assert changed.identity != p.identity
    payload["independent_unit"] = "transition"
    with pytest.raises(ValueError, match="metadata"):
        HMCReplicatedAcceptancePolicy.from_payload(payload)


@pytest.mark.parametrize("changes", [
    {"trial_num_results":63}, {"trial_num_results":True}, {"discarded_prefix":-1},
    {"base_repetitions":17}, {"max_candidates":0}, {"search_family_alpha":0},
    {"verification_family_alpha":float('nan')}, {"temporal_tolerance":1.},
    {"start_weights":(.1,.1,.1,.1)}, {"start_weights":(0.,.25,.25,.5)},
    {"qualification_region":(.68,.72)}, {"bet_fractions":(1.1,)},
])
def test_invalid_protocol_cannot_be_silently_repaired(changes):
    with pytest.raises(ValueError):
        policy(**changes)


def test_unrepresentable_error_allocation_cannot_appear_to_be_zero():
    with pytest.raises(ValueError, match="resolution"):
        policy(max_candidates=10**400).one_sided_alpha("verification")


@pytest.mark.parametrize("changes", [
    {"candidate_cap":9}, {"evidence_rungs":(1,2,8)}, {"evidence_rungs":(1,1)},
    {"evidence_rungs":(2,4)}, {"leapfrog_steps":(0,)}, {"pilot_enabled":1},
    {"candidate_cap":1},
])
def test_search_cannot_exceed_the_frozen_information_or_error_contract(changes):
    kwargs = dict(evidence_rungs=(1,2,4), candidate_cap=8, leapfrog_steps=(3,9))
    with pytest.raises(ValueError):
        replicated_evidence_preflight(policy(), **{**kwargs, **changes})


def test_preflight_cli_preserves_unfundable_result_and_refuses_overwrite(tmp_path, capsys):
    from bayesfilter.testing.acceptance_decision_validation import CONFIG_SCHEMA, main
    config = tmp_path / "design.json"
    config.write_text(json.dumps({"schema": CONFIG_SCHEMA, "policy": policy().payload(),
        "evidence_rungs": [1, 2, 4], "candidate_cap": 8, "leapfrog_steps": [3, 9],
        "max_gradient_work": 1,
        "legacy_batch": {"draws": 512, "batch_size": 11, "min_batches": 8}}))
    output = tmp_path / "attempt-01"
    args = ["preflight", "--config", str(config), "--output", str(output)]
    assert main(args) == 2
    report = json.loads((output / "preflight.json").read_text())
    assert report["first_cohort_fundable"] is False
    assert report["legacy_batch"]["minimum_draws_fixed_batch"] == 704
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["artifact_authority"] is False
    assert manifest["source_sha256"]
    with pytest.raises(FileExistsError):
        main(args)
