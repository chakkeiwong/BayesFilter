from __future__ import annotations

from pathlib import Path

import pytest

from bayesfilter.inference.neutra_shared_procedure import (
    DEFAULT_COMMON_VARIANT,
    LEGACY_UNLABELED_VARIANT,
    OPERATIONAL_BROAD_GRID_V1,
    PROCEDURE_FAMILY,
    PROCEDURE_VARIANTS,
    STATE_CONTINUING_EPSILON_REPAIR_V1,
    SharedNeuTraProcedureConfig,
    extract_sequential_handoff_from_broad_grid_result,
    procedure_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


def _candidate(
    *,
    leapfrog: int = 25,
    step_size: float = 0.99,
    role: str = "independently_tuned_primary",
    viable: bool = True,
    disposition: str = "provisional_viable",
) -> dict:
    return {
        "viable": viable,
        "tuned_step_size": step_size,
        "evidence": {"disposition": disposition},
        "request": {"role": role, "num_leapfrog_steps": leapfrog},
    }


def _payload(
    candidates,
    *,
    route: str = "operational_broad_fixed_mass_l_epsilon_grid_v1",
    disposition: str = "viable_pair_set",
    ranking: bool = False,
    preserved: bool = True,
    variant: str | None = None,
) -> dict:
    payload = {
        "route": route,
        "disposition": disposition,
        "stochastic_ranking_performed": ranking,
        "all_viable_pairs_preserved": preserved,
        "next_round_candidates": list(candidates),
    }
    if variant is not None:
        payload["procedure_variant"] = variant
    return payload


def test_procedure_metadata_flags_distinguish_variants() -> None:
    generic = procedure_metadata(OPERATIONAL_BROAD_GRID_V1)
    repaired = procedure_metadata(STATE_CONTINUING_EPSILON_REPAIR_V1)
    for payload in (generic, repaired):
        assert payload["procedure_family"] == PROCEDURE_FAMILY
        assert payload["directional_refinement_performed"] is False
        assert payload["same_epsilon_neighbor_guards_performed"] is True
        assert payload["independent_primary_dual_averaging_performed"] is True
        assert payload["fresh_final_screens_performed"] is True
    assert generic["procedure_variant"] == OPERATIONAL_BROAD_GRID_V1
    assert generic["state_continuation_performed"] is False
    assert generic["epsilon_repair_performed"] is False
    assert repaired["procedure_variant"] == STATE_CONTINUING_EPSILON_REPAIR_V1
    assert repaired["state_continuation_performed"] is True
    assert repaired["epsilon_repair_performed"] is True


def test_procedure_metadata_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown NeuTra procedure variant"):
        procedure_metadata("free_form_local_procedure")


def test_shared_config_validates_variant_hash_seed_and_gpu_claim() -> None:
    common = {
        "output_root": ROOT / "unused",
        "frozen_transport_path": ROOT / "unused.json",
        "expected_frozen_transport_sha256": "a" * 64,
        "root_seed": (1, 2),
    }
    config = SharedNeuTraProcedureConfig(**common)
    assert config.variant == DEFAULT_COMMON_VARIANT
    assert config.launch_sequential is False
    with pytest.raises(ValueError, match="unknown NeuTra procedure variant"):
        SharedNeuTraProcedureConfig(**common, variant="ad_hoc")
    with pytest.raises(ValueError, match="SHA-256"):
        SharedNeuTraProcedureConfig(**{**common, "expected_frozen_transport_sha256": "bad"})
    with pytest.raises(ValueError, match="root_seed"):
        SharedNeuTraProcedureConfig(**{**common, "root_seed": (-1, 2)})
    with pytest.raises(ValueError, match="GPU/XLA"):
        SharedNeuTraProcedureConfig(**common, require_gpu=True, jit_compile=False)
    with pytest.raises(ValueError, match="exceed 64"):
        SharedNeuTraProcedureConfig(**common, screen_results=64)
    with pytest.raises(ValueError, match="exceed 64"):
        SharedNeuTraProcedureConfig(**common, final_screen_results=10)
    with pytest.raises(ValueError, match="status_code"):
        SharedNeuTraProcedureConfig(**common, required_status_keys=("floor_count_value",))


def test_handoff_extractor_accepts_unique_primary_and_labels_variant() -> None:
    labeled = extract_sequential_handoff_from_broad_grid_result(
        _payload([_candidate()], variant=OPERATIONAL_BROAD_GRID_V1)
    )
    assert labeled["step_size"] == pytest.approx(0.99)
    assert labeled["num_leapfrog_steps"] == 25
    assert labeled["procedure_variant"] == OPERATIONAL_BROAD_GRID_V1
    legacy = extract_sequential_handoff_from_broad_grid_result(
        _payload([_candidate()])
    )
    assert legacy["procedure_variant"] == LEGACY_UNLABELED_VARIANT


def test_handoff_extractor_rejects_non_unique_union() -> None:
    payload = _payload(
        [
            _candidate(),
            _candidate(leapfrog=24, role="same_epsilon_neighbor_coverage"),
        ]
    )
    with pytest.raises(ValueError, match="exactly one unranked viable pair"):
        extract_sequential_handoff_from_broad_grid_result(payload)


def test_handoff_extractor_rejects_incomplete_or_ranked_sets() -> None:
    with pytest.raises(ValueError, match="complete viable set"):
        extract_sequential_handoff_from_broad_grid_result(
            _payload([_candidate()], ranking=True)
        )
    with pytest.raises(ValueError, match="complete viable set"):
        extract_sequential_handoff_from_broad_grid_result(
            _payload([_candidate()], preserved=False)
        )
    with pytest.raises(ValueError, match="complete viable set"):
        extract_sequential_handoff_from_broad_grid_result(
            _payload([_candidate()], disposition="no_viable_pair")
        )
    with pytest.raises(ValueError, match="complete viable set"):
        extract_sequential_handoff_from_broad_grid_result(
            _payload([_candidate()], route="bespoke_local_route")
        )


def test_handoff_extractor_rejects_non_primary_or_invalid_mechanics() -> None:
    with pytest.raises(ValueError, match="independently tuned primary"):
        extract_sequential_handoff_from_broad_grid_result(
            _payload([_candidate(role="same_epsilon_neighbor_coverage")])
        )
    with pytest.raises(ValueError, match="not viable"):
        extract_sequential_handoff_from_broad_grid_result(
            _payload([_candidate(viable=False)])
        )
    with pytest.raises(ValueError, match="kernel mechanics"):
        extract_sequential_handoff_from_broad_grid_result(
            _payload([_candidate(step_size=-1.0)])
        )
    with pytest.raises(ValueError, match="unknown variant"):
        extract_sequential_handoff_from_broad_grid_result(
            _payload([_candidate()], variant="mystery_procedure")
        )


def test_next_repair_epsilon_preserves_reviewed_bracketing_semantics() -> None:
    from bayesfilter.inference.neutra_state_continuing_broad_grid import (
        next_repair_epsilon,
    )

    step, low, high, action = next_repair_epsilon(
        epsilon=1.0, acceptance_mean=0.90, lower_epsilon=None, upper_epsilon=None
    )
    assert (step, low, high, action) == (1.20, 1.0, None, "increase_epsilon")
    step, low, high, action = next_repair_epsilon(
        epsilon=1.0, acceptance_mean=0.10, lower_epsilon=None, upper_epsilon=None
    )
    assert (step, low, high, action) == (1.0 / 1.20, None, 1.0, "decrease_epsilon")
    step, low, high, action = next_repair_epsilon(
        epsilon=2.0, acceptance_mean=0.90, lower_epsilon=None, upper_epsilon=4.0
    )
    assert action == "geometric_bracket_midpoint"
    assert step == pytest.approx((2.0 * 4.0) ** 0.5)
    assert (low, high) == (2.0, 4.0)
    step, low, high, action = next_repair_epsilon(
        epsilon=1.0, acceptance_mean=0.70, lower_epsilon=0.5, upper_epsilon=2.0
    )
    assert (step, low, high, action) == (1.0, 0.5, 2.0, "calibration_region_reached")


def test_state_continuing_config_records_warm_start_role_and_status_contract() -> None:
    from bayesfilter.inference.neutra_state_continuing_broad_grid import (
        NeuTraStateContinuingBroadGridConfig,
    )

    config = NeuTraStateContinuingBroadGridConfig(
        initial_step_size=0.1,
        root_seed=(1, 2),
        evidence_path="docs/plans/example-plan.md",
        initial_epsilon_by_l={3: 0.8, 25: 0.6},
        required_status_keys=("status_code", "valid_pre_regularized_score"),
    )
    payload = config.payload()
    assert payload["initial_epsilon_role"] == "warm_start_hypothesis_only"
    assert payload["calibration_region"] == (0.68, 0.72)
    assert payload["epsilon_repair_factor"] == pytest.approx(1.20)
    assert payload["max_epsilon_repairs"] == 3
    assert config.warm_start_epsilon(3) == pytest.approx(0.8)
    assert config.warm_start_epsilon(13) == pytest.approx(0.1)
    assert payload["required_status_keys"] == (
        "status_code",
        "valid_pre_regularized_score",
    )
    with pytest.raises(ValueError, match="exceed the 64-draw"):
        NeuTraStateContinuingBroadGridConfig(
            initial_step_size=0.1,
            root_seed=(1, 2),
            evidence_path="docs/plans/example-plan.md",
            final_screen_results=64,
        )


def test_wrappers_delegate_to_the_shared_procedure_module() -> None:
    implementation = (
        ROOT / "bayesfilter/inference/neutra_end_to_end.py"
    ).read_text(encoding="utf-8")
    shared = (
        ROOT / "bayesfilter/inference/neutra_shared_procedure.py"
    ).read_text(encoding="utf-8")
    assert "run_shared_neutra_procedure" in implementation
    assert "extract_sequential_handoff_from_broad_grid_result" in implementation
    assert "run_neutra_operational_broad_grid_tuning" in shared
    assert "run_neutra_state_continuing_broad_grid_tuning" in shared
    for variant in PROCEDURE_VARIANTS:
        assert f'"{variant}"' in shared
    assert '"sampling_launched": sequential_result is not None' in shared
    assert '"retained_sampling_authorized": False' in shared


def test_state_continuing_module_is_tuning_only_and_preserves_repair_mechanics() -> None:
    source = (
        ROOT / "bayesfilter/inference/neutra_state_continuing_broad_grid.py"
    ).read_text(encoding="utf-8")
    assert "tune_hmc_kernel(" not in source
    assert "run_sequential_neutra_hmc" not in source
    assert "next_repair_epsilon" in source
    assert "state_continuing_epsilon_repair_calibration" in source
    assert "state_continuing_primary_fresh_screen" in source
    assert "calibrated parent state" in source or "_calibrated_states" in source
    assert "geometric_bracket_midpoint" in source
    assert "run_operational_broad_grid(" in source
    assert '"all_draws_discarded": True' in source


def test_shared_driver_defaults_to_common_repaired_variant_and_supports_auto_hooks() -> None:
    driver = (
        ROOT / "docs/benchmarks/run_neutra_shared_procedure_20260803.py"
    ).read_text(encoding="utf-8")
    assert '"--variant"' in driver
    assert "DEFAULT_COMMON_VARIANT" in driver
    assert 'default="auto"' in driver
    assert "_spec_status_keys" in driver
    assert "_spec_initial_epsilon_by_l" in driver
    assert "run_shared_neutra_procedure" in driver
    assert 'operational_broad_grid_v1' in driver
    assert 'state_continuing_epsilon_repair_v1' in driver


def test_registry_cells_expose_common_tuning_hook_metadata() -> None:
    from bayesfilter.testing.neutra_model_registry_tf import EXECUTABLE_CELLS

    by_id = {spec.cell_id: spec for spec in EXECUTABLE_CELLS}
    assert by_id["PP-UKF"].common_tuning_initial_epsilon_by_l is not None
    assert by_id["PP-SGQF"].common_tuning_status_keys[:2] == (
        "status_code",
        "valid_pre_regularized_score",
    )
    assert by_id["SIR-SGQF"].common_tuning_status_keys[:2] == (
        "status_code",
        "valid_pre_regularized_score",
    )
    assert by_id["SVX-ZC"].common_tuning_status_keys == (
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    )
    assert by_id["LGSSM-EXACT"].common_tuning_status_keys == (
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    )
