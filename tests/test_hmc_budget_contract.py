from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

import bayesfilter.hmc_budget_contract as budget_contract
from bayesfilter.hmc_budget_contract import (
    BROAD_FIXED_METRIC_OPERATIONAL_ROUTE,
    HMCOperationalStatisticalWorkPolicy,
    build_private_resolved_hmc_work_manifest,
    build_public_hmc_work_manifest,
    build_serious_public_hmc_work_manifest,
    reconcile_executed_hmc_work,
    validate_executed_hmc_work_reconciliation,
    validate_private_resolved_hmc_work_manifest,
    validate_public_hmc_work_manifest,
)
from bayesfilter.hmc_route_contract import (
    LEGACY_OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID,
)


def test_budget_contract_import_does_not_load_tensorflow_or_tfp() -> None:
    assert importlib.util.find_spec("bayesfilter.hmc_budget_contract") is not None
    child_env = dict(os.environ)
    child_env["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import bayesfilter.hmc_budget_contract; "
            "assert 'tensorflow' not in sys.modules; "
            "assert 'tensorflow_probability' not in sys.modules",
        ),
        check=False,
        capture_output=True,
        text=True,
        env=child_env,
    )
    assert completed.returncode == 0, completed.stderr


def test_screen_and_verification_defaults_are_independent_of_metric_warmup() -> None:
    policy = HMCOperationalStatisticalWorkPolicy()
    small = build_public_hmc_work_manifest(
        target_dimension=17,
        metric_adaptation_steps=(1000,),
        selection_attempts_per_outer_attempt=(1,),
        max_leapfrog_steps=25,
        policy=policy,
    )
    large = build_public_hmc_work_manifest(
        target_dimension=314,
        metric_adaptation_steps=(5000,),
        selection_attempts_per_outer_attempt=(1,),
        max_leapfrog_steps=25,
        policy=policy,
    )

    independent = (
        "initial_candidate_results",
        "candidate_burnin_steps",
        "evidence_extension_checkpoints",
        "fresh_verification_results",
        "fresh_verification_burnin_steps",
    )
    assert all(small[key] == large[key] for key in independent)
    assert small["metric_adaptation_steps"] != large["metric_adaptation_steps"]
    assert small["broad_tune_budget_schedule"] == (250, 500, 1000)
    assert large["broad_tune_budget_schedule"] == (1250, 2500, 5000)
    assert policy.initial_candidate_results != 5000 // 4
    assert policy.fresh_verification_results != 5000 // 2


def test_manifest_declares_operational_scope_not_whole_launch_work() -> None:
    manifest = build_serious_public_hmc_work_manifest(
        target_dimension=314,
        outer_attempt_count=1,
    )

    assert manifest["accounting_scope_id"] == (
        "operational_metric_adaptation_through_fresh_verification.v1"
    )
    assert manifest["whole_launch_hmc_upper_bound"] is False
    assert manifest["pre_scope_hmc_work_included"] is False

    ambiguous = dict(manifest)
    ambiguous["accounting_scope_id"] = "whole_launch"
    ambiguous["manifest_hash"] = budget_contract._canonical_hash(
        {key: value for key, value in ambiguous.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="accounting scope"):
        validate_public_hmc_work_manifest(ambiguous)


@pytest.mark.parametrize(
    "checkpoints, match",
    [
        ((64,), "must exceed"),
        ((256, 128), "strictly increase"),
        ((256, 256), "strictly increase"),
        ((128, 256, 512), "at most two"),
    ],
)
def test_extension_checkpoint_validation(checkpoints, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HMCOperationalStatisticalWorkPolicy(
            evidence_extension_checkpoints=checkpoints,
        )


def test_default_manifest_counts_every_attempt_retune_fallback_and_verification_start() -> None:
    manifest = build_serious_public_hmc_work_manifest(
        target_dimension=314,
        outer_attempt_count=2,
    )

    assert manifest["metric_adaptation_steps"] == (5000, 10000)
    assert manifest["selection_attempts_per_outer_attempt"] == (2, 1)
    assert manifest["route_marker"] == BROAD_FIXED_METRIC_OPERATIONAL_ROUTE
    assert manifest["candidate_count_upper_bound"] == 13
    assert manifest["broad_primary_grid_width"] == 6
    assert manifest["broad_refinement_grid_width_upper_bound"] == 7
    assert manifest["replications_per_candidate"] == 1
    assert manifest["exact_l_tune_start_count_upper_bound"] == 0
    assert manifest["fresh_verification_start_count_upper_bound"] == 4
    assert manifest["evidence_extension_checkpoints"] == ()
    assert manifest["maximum_work"]["extension_candidate_batched_transitions"] == 0
    assert manifest["aggregate_counts_public_safe"] is True
    assert manifest["private_hmc_mechanics_exposed"] is False
    assert validate_public_hmc_work_manifest(manifest) == manifest


def test_extensions_are_conservative_even_when_only_inconclusive_slots_execute() -> None:
    policy = HMCOperationalStatisticalWorkPolicy(
        evidence_extension_checkpoints=(256, 512),
    )
    manifest = build_public_hmc_work_manifest(
        target_dimension=314,
        metric_adaptation_steps=(5000,),
        selection_attempts_per_outer_attempt=(5,),
        max_leapfrog_steps=25,
        policy=policy,
        algorithm_id=LEGACY_OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID,
    )

    expected = 5 * 3 * 3 * ((256 + 16) + (512 + 16))
    assert manifest["maximum_work"]["extension_candidate_batched_transitions"] == expected
    assert manifest["exact_l_tune_start_count_upper_bound"] == 5 * 3 * 3


def test_public_manifest_hash_and_privacy_fail_closed() -> None:
    manifest = dict(
        build_serious_public_hmc_work_manifest(
            target_dimension=314,
            outer_attempt_count=1,
        )
    )
    manifest["target_dimension"] = 315
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_public_hmc_work_manifest(manifest)

    unsafe = dict(
        build_serious_public_hmc_work_manifest(
            target_dimension=314,
            outer_attempt_count=1,
        )
    )
    unsafe["verification_seed"] = (1, 2)
    with pytest.raises(ValueError, match="private field"):
        validate_public_hmc_work_manifest(unsafe)

    arithmetic_tamper = dict(
        build_serious_public_hmc_work_manifest(
            target_dimension=314,
            outer_attempt_count=1,
        )
    )
    arithmetic_tamper["maximum_work"] = {
        **dict(arithmetic_tamper["maximum_work"]),
        "total_batched_transitions": 1,
    }
    arithmetic_tamper["manifest_hash"] = budget_contract._canonical_hash(
        {
            key: value
            for key, value in arithmetic_tamper.items()
            if key != "manifest_hash"
        }
    )
    with pytest.raises(ValueError, match="arithmetic or policy mismatch"):
        validate_public_hmc_work_manifest(arithmetic_tamper)


def test_private_manifest_links_to_public_without_changing_public_payload() -> None:
    public = build_serious_public_hmc_work_manifest(
        target_dimension=314,
        outer_attempt_count=1,
    )
    private = build_private_resolved_hmc_work_manifest(
        public_manifest=public,
        resolved_candidates=({"num_leapfrog_steps": 7, "step_size": 0.02},),
    )

    assert private["public_manifest_hash"] == public["manifest_hash"]
    assert private["private_handoff_only"] is True
    assert private["publicized"] is False
    assert "resolved_candidates" not in public
    assert validate_private_resolved_hmc_work_manifest(
        private,
        expected_public_manifest_hash=public["manifest_hash"],
    ) == private

    corrupted = dict(private)
    corrupted["resolved_candidate_count"] = 2
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_private_resolved_hmc_work_manifest(corrupted)


def test_executed_work_reconciles_partial_start_and_rejects_overrun() -> None:
    public = build_serious_public_hmc_work_manifest(
        target_dimension=314,
        outer_attempt_count=1,
    )
    maximum = public["maximum_work"]
    partial = reconcile_executed_hmc_work(
        public_manifest=public,
        executed_work={
            "fresh_verification_batched_transitions": 64 + 16,
        },
    )
    assert partial["within_public_bounds"] is True
    assert partial["public_manifest_hash"] == public["manifest_hash"]
    assert validate_executed_hmc_work_reconciliation(
        partial,
        expected_public_manifest_hash=public["manifest_hash"],
        public_manifest=public,
    ) == partial

    with pytest.raises(ValueError, match="public manifest is required"):
        validate_executed_hmc_work_reconciliation(
            partial,
            expected_public_manifest_hash=public["manifest_hash"],
        )

    corrupted = dict(partial)
    corrupted["within_public_bounds"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_executed_hmc_work_reconciliation(
            corrupted,
            public_manifest=public,
        )

    arithmetic_tamper = dict(partial)
    arithmetic_tamper["executed_work"] = {
        **dict(arithmetic_tamper["executed_work"]),
        "total_batched_transitions": 0,
    }
    arithmetic_tamper["reconciliation_hash"] = budget_contract._canonical_hash(
        {
            key: value
            for key, value in arithmetic_tamper.items()
            if key != "reconciliation_hash"
        }
    )
    with pytest.raises(ValueError, match="every component count"):
        validate_executed_hmc_work_reconciliation(
            arithmetic_tamper,
            public_manifest=public,
        )

    with pytest.raises(ValueError, match="exceeded public bounds"):
        reconcile_executed_hmc_work(
            public_manifest=public,
            executed_work={
                "fresh_verification_batched_transitions": (
                    maximum["fresh_verification_batched_transitions"] + 1
                )
            },
        )
