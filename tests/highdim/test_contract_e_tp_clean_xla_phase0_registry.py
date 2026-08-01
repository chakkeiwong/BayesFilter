from __future__ import annotations

import json
from pathlib import Path

from docs.benchmarks import build_contract_e_tp_clean_xla_phase0_registry as builder


def test_clean_xla_phase0_has_seven_model_items_and_one_shared_fixture() -> None:
    payload = builder.build_payload()
    assert payload["item_kinds"] == {"model_row": 7, "shared_regression_item": 1}
    assert len(payload["rows"]) == 8
    assert len({row["row_id"] for row in payload["rows"]}) == 8
    fixture = next(row for row in payload["rows"] if row["row_id"] == "structural_deterministic_fixture")
    assert fixture["item_kind"] == "shared_regression_item"
    assert all(
        row["item_kind"] == "model_row"
        for row in payload["rows"]
        if row is not fixture
    )


def test_clean_xla_phase0_preserves_blocked_and_negative_boundaries() -> None:
    rows = {row["row_id"]: row for row in builder.build_payload()["rows"]}
    assert rows["zhao_cui_generalized_sv_synthetic_from_estimated_values"]["classification"] == "negative_result"
    assert rows["zhao_cui_generalized_sv_synthetic_from_estimated_values"]["gpu_scheduling"] == "forbidden"
    assert rows["zhao_cui_spatial_sir_austria_j9_T20"]["classification"] == "target_blocked"
    assert rows["zhao_cui_spatial_sir_austria_j9_T20"]["gpu_scheduling"] == "forbidden"
    assert rows["dsge_nawm_client"]["classification"] == "target_blocked"


def test_clean_xla_phase0_does_not_transfer_lgssm_defaults() -> None:
    rows = {row["row_id"]: row for row in builder.build_payload()["rows"]}
    assert rows["benchmark_lgssm_exact_oracle_m3_T50"]["default_audit"]["lookahead"] == "8 (LGSSM reference only)"
    for row_id in (
        "zhao_cui_sv_actual_nongaussian_T1000",
        "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
        "zhao_cui_predator_prey_T20",
    ):
        assert rows[row_id]["default_audit"]["lookahead"].startswith("unset")


def test_clean_xla_phase0_binds_controlling_artifact_hashes_and_policies() -> None:
    payload = builder.build_payload()
    assert all(value["hash_status"] == "verified" for value in payload["controlling_artifacts"].values())
    assert payload["policies"]["contract_e_tp_status"] == "experimental_only"
    assert "same-scalar FD only" in payload["policies"]["fd_tolerance_role"]
    assert payload["policies"]["cross_method_margin"] == "unavailable; descriptive-only"
    assert payload["budget"]["historical_experiments_charged"] == 0
    assert payload["controlling_artifacts"]["prior_target_identity_registry"]["hash_status"] == "verified"
    assert all(row["target_identity_anchor"] for row in payload["rows"])


def test_clean_xla_phase0_source_scan_distinguishes_loops() -> None:
    payload = builder.build_payload()
    source = {item["path"]: item for item in payload["source_dependencies"]}
    assert source["bayesfilter/highdim/ledh_contract_e_tp_lgssm_tf.py"]["functional_loop_calls"]
    assert source["bayesfilter/highdim/ledh_contract_e_tp_scalar_sv_tf.py"]["python_loop_count"] > 0
    assert source["bayesfilter/highdim/ledh_contract_e_tp_predator_prey_tf.py"]["python_loop_count"] > 0
    topology = payload["topology_inventory"]
    assert len(topology["compiled_dynamic_python_loops"]) == 3
    assert topology["gradient_runtime_numpy_scipy_findings"] == []
    assert topology["functional_tensorflow_loops"][0]["status"] == "completed_reference"
