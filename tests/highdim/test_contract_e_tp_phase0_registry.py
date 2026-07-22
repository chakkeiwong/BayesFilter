from __future__ import annotations

import json
from pathlib import Path


REGISTRY = Path(
    "docs/benchmarks/configs/contract_e_tp_all_models_2026_07_15.json"
)


def _registry() -> dict[str, object]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_phase0_registry_freezes_exactly_six_primary_rows() -> None:
    payload = _registry()
    rows = payload["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 6
    assert payload["primary_row_ids"] == [row["row_id"] for row in rows]
    assert len(set(payload["primary_row_ids"])) == 6
    for row in rows:
        assert row["comparison_scope"] == "primary_observed_data_filtering"
        assert row["target_scalar"] == "finite_fixed_program_observed_data_log_likelihood"
        assert row["score_target"] == "total_derivative_of_target_scalar"
        assert row["parameter_dim"] == len(row["parameter_order"])
        assert row["parameter_dim"] == len(row["truth_theta"])
        assert row["dataset"]["raw_observations"]["all_finite"] is True
        assert row["dataset"]["target_observations"]["all_finite"] is True


def test_phase0_registry_keeps_cross_method_margin_separate_from_fd() -> None:
    payload = _registry()
    protocol = payload["comparison_protocol"]
    assert "existing FD policy only" in protocol["same_scalar_derivative"]
    assert "0.05*sqrt(p) as cross-method margin" in protocol["forbidden"]
    assert protocol["pilot_replicates"] == 16
    assert protocol["maximum_replicates_without_plan_revision"] == 64
    assert all(
        row["equivalence_margin_status"] == "descriptive_only_margin_unavailable"
        for row in payload["rows"]
    )


def test_phase0_registry_records_disjoint_seed_roles() -> None:
    seeds = _registry()["role_seeds"]
    preparation = set(seeds["preparation"])
    validation = set(seeds["validation"])
    audit = set(seeds["audit"])
    assert len(preparation) == len(validation) == len(audit) == 16
    assert preparation.isdisjoint(validation)
    assert preparation.isdisjoint(audit)
    assert validation.isdisjoint(audit)
    assert "audit is final-only" in seeds["use"]


def test_phase0_registry_preserves_sir_component_total_score_boundary() -> None:
    payload = _registry()
    sir = next(
        row
        for row in payload["rows"]
        if row["row_id"] == "zhao_cui_spatial_sir_austria_j9_T20"
    )
    assert sir["state_dim"] == 18
    assert sir["parameter_dim"] == 3
    assert sir["zhao_cui"]["route"] == "fixed_ttsirt_source_route"
    assert sir["zhao_cui"]["status"] == (
        "implemented_component_score_full_observed_data_total_score_blocked"
    )
    assert sir["parameter_region"]["box"] == [
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
    ]
    assert "P91 SIR component score as full filtering score" in payload[
        "comparison_protocol"
    ]["forbidden"]


def test_phase0_registry_rejects_oracle_and_retained_grid_substitutions() -> None:
    rows = {row["row_id"]: row for row in _registry()["rows"]}
    lgssm = rows["benchmark_lgssm_exact_oracle_m3_T50"]
    predator = rows["zhao_cui_predator_prey_T20"]
    assert "oracle_adapter_is_not_zhao_cui" in lgssm["zhao_cui"]["status"]
    assert predator["zhao_cui"]["route"] == (
        "forbidden_retained_grid_route_must_not_be_used"
    )


def test_phase0_registry_separates_generalized_sv_target_and_flow_observations() -> None:
    rows = {row["row_id"]: row for row in _registry()["rows"]}
    generalized = rows["zhao_cui_generalized_sv_synthetic_from_estimated_values"]
    dataset = generalized["dataset"]
    assert dataset["transform"] == "identity"
    assert dataset["target_observations"] == dataset["raw_observations"]
    assert dataset["proposal_flow_observations"]["all_finite"] is True
    assert dataset["proposal_flow_observations"]["serialized_tensor_sha256"] != (
        dataset["target_observations"]["serialized_tensor_sha256"]
    )
    assert generalized["target_observation_policy"] == (
        "source_route_prior_mean_generalized_sv"
    )


def test_phase0_registry_has_local_primary_source_hashes_and_claim_boundaries() -> None:
    payload = _registry()
    support = payload["source_support"]
    assert len(support["zhao_cui_paper"]["sha256"]) == 64
    assert len(support["zhao_cui_paper"]["text_sha256"]) == 64
    assert support["zhao_cui_paper"]["classification"] == "DIRECT_METHOD"
    assert "fixed-branch score correctness" in support["zhao_cui_paper"][
        "forbidden_claim"
    ]
    assert support["author_code"]["upstream_commit"] == (
        "80034dccb99eb1d86284a1839b4a12067d13b9da"
    )
