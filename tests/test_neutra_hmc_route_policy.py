from __future__ import annotations

from pathlib import Path

import pytest

from bayesfilter.inference.neutra_hmc_policy import (
    NeuTraHMCRoutePolicyError,
    audit_neutra_hmc_route_policy,
    load_neutra_hmc_route_ledger,
    require_neutra_hmc_route_policy,
)


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / (
    "docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-"
    "2026-07-15/c0/route_ledger.json"
)


def test_repository_neutra_hmc_route_ledger_is_complete_and_enforced() -> None:
    ledger = load_neutra_hmc_route_ledger(LEDGER)
    audit = require_neutra_hmc_route_policy(ROOT, ledger)

    assert audit["passed"] is True
    assert "bayesfilter/testing/lgssm_neutra_gap_closure_tf.py" in audit[
        "discovered_routes"
    ]
    assert set(audit["discovered_routes"]) == set(audit["classified_routes"])


def test_policy_rejects_unledgered_qualifying_route(tmp_path: Path) -> None:
    _minimal_repository(tmp_path)
    route = tmp_path / "bayesfilter/testing/new_neutra_route.py"
    route.write_text("# NeuTra\ndef run_hmc_tuning_candidate(): pass\n", encoding="utf-8")
    ledger = _minimal_ledger()

    audit = audit_neutra_hmc_route_policy(tmp_path, ledger)

    assert audit["passed"] is False
    assert audit["errors"] == (
        "unledgered_qualifying_route:bayesfilter/testing/new_neutra_route.py",
    )


def test_policy_rejects_stale_and_duplicate_ledger_paths(tmp_path: Path) -> None:
    _minimal_repository(tmp_path)
    ledger = _minimal_ledger()
    ledger["routes"] = [
        {
            "path": "bayesfilter/testing/missing_neutra_route.py",
            "classification": "historical_or_superseded",
            "reason": "fixture",
        }
    ]
    audit = audit_neutra_hmc_route_policy(tmp_path, ledger)
    assert "stale_or_undiscovered_ledger_route:bayesfilter/testing/missing_neutra_route.py" in audit[
        "errors"
    ]
    assert "stale_ledger_path:bayesfilter/testing/missing_neutra_route.py" in audit[
        "errors"
    ]

    duplicate = _minimal_ledger()
    duplicate["routes"] = [
        {"path": "same.py", "classification": "training_or_non_hmc", "reason": "a"},
        {"path": "same.py", "classification": "training_or_non_hmc", "reason": "b"},
    ]
    with pytest.raises(NeuTraHMCRoutePolicyError, match="duplicate routes path"):
        audit_neutra_hmc_route_policy(tmp_path, duplicate)


def test_policy_rejects_active_route_without_core_or_with_fixed_budget(
    tmp_path: Path,
) -> None:
    _minimal_repository(tmp_path)
    path = tmp_path / "bayesfilter/testing/active_neutra_hmc.py"
    path.write_text(
        "# NeuTra\n"
        "def run_hmc_tuning_candidate():\n"
        "    num_results=1000\n"
        "    num_burnin_steps=1000\n",
        encoding="utf-8",
    )
    ledger = _minimal_ledger()
    ledger["routes"] = [
        {
            "path": "bayesfilter/testing/active_neutra_hmc.py",
            "classification": "active_claim_bearing",
            "core_binding": "direct",
            "active_entry_points": ["run_hmc_tuning_candidate"],
            "required_symbols": [
                "NEUTRA_SEQUENTIAL_HMC_POLICY_ID",
                "run_sequential_neutra_hmc",
            ],
            "policy_id": "bayesfilter_neutra_sequential_hmc_v1",
            "reason": "negative fixture",
        }
    ]

    errors = audit_neutra_hmc_route_policy(tmp_path, ledger)["errors"]

    assert any("active_route_missing_core_symbol" in item for item in errors)
    assert "active_route_fixed_terminal_budget:bayesfilter/testing/active_neutra_hmc.py" in errors


def test_policy_rejects_reachable_local_sampler_bypass(tmp_path: Path) -> None:
    _minimal_repository(tmp_path)
    path = tmp_path / "bayesfilter/testing/active_neutra_hmc.py"
    path.write_text(
        "# NeuTra\n"
        "from bayesfilter.inference.neutra_hmc import (\n"
        "    NEUTRA_SEQUENTIAL_HMC_POLICY_ID, run_sequential_neutra_hmc)\n"
        "def local_sampler():\n"
        "    return tfp.mcmc.sample_chain()\n"
        "def run_hmc_tuning_candidate():\n"
        "    return local_sampler()\n",
        encoding="utf-8",
    )
    ledger = _minimal_ledger()
    ledger["routes"] = [
        {
            "path": "bayesfilter/testing/active_neutra_hmc.py",
            "classification": "active_claim_bearing",
            "core_binding": "direct",
            "active_entry_points": ["run_hmc_tuning_candidate"],
            "required_symbols": [
                "NEUTRA_SEQUENTIAL_HMC_POLICY_ID",
                "run_sequential_neutra_hmc",
            ],
            "policy_id": "bayesfilter_neutra_sequential_hmc_v1",
            "reason": "negative fixture",
        }
    ]

    errors = audit_neutra_hmc_route_policy(tmp_path, ledger)["errors"]

    assert (
        "active_route_local_sampler_bypass:"
        "bayesfilter/testing/active_neutra_hmc.py:sample_chain"
    ) in errors


def test_policy_allows_declared_fixed_nomination_before_sequential_gate(
    tmp_path: Path,
) -> None:
    _minimal_repository(tmp_path)
    path = tmp_path / "bayesfilter/testing/active_neutra_hmc.py"
    path.write_text(
        "# NeuTra\n"
        "NEUTRA_SEQUENTIAL_HMC_POLICY_ID = 'bayesfilter_neutra_sequential_hmc_v1'\n"
        "def run_hmc_tuning_candidate():\n"
        "    probe = run_batched_hmc(config=dict(num_results=64, num_burnin_steps=128))\n"
        "    role = {\"acceptance_role\": \"nomination_only\"}\n"
        "    return run_sequential_neutra_hmc(probe, role)\n",
        encoding="utf-8",
    )
    ledger = _minimal_ledger()
    ledger["routes"] = [
        {
            "path": "bayesfilter/testing/active_neutra_hmc.py",
            "classification": "active_claim_bearing",
            "core_binding": "direct",
            "active_entry_points": ["run_hmc_tuning_candidate"],
            "required_symbols": [
                "NEUTRA_SEQUENTIAL_HMC_POLICY_ID",
                "run_sequential_neutra_hmc",
            ],
            "fixed_budget_role": (
                "kernel_nomination_only_before_shared_sequential_admission"
            ),
            "policy_id": "bayesfilter_neutra_sequential_hmc_v1",
            "reason": "positive nomination fixture",
        }
    ]

    assert audit_neutra_hmc_route_policy(tmp_path, ledger)["passed"] is True


def _minimal_repository(root: Path) -> None:
    (root / "bayesfilter/testing").mkdir(parents=True)
    (root / "docs/benchmarks").mkdir(parents=True)


def _minimal_ledger():
    return {
        "schema": "bayesfilter.neutra_hmc_route_ledger.v1",
        "canonical_policy_id": "bayesfilter_neutra_sequential_hmc_v1",
        "discovery": {
            "roots": ["bayesfilter", "docs/benchmarks"],
            "suffix": ".py",
            "required_case_insensitive_marker": "neutra",
            "behavior_markers": ["run_hmc_tuning_candidate"],
            "exclusions": [],
        },
        "routes": [],
    }
