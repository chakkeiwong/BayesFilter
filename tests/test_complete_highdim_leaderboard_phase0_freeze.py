from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts import build_complete_highdim_leaderboard_phase0_freeze as freeze
from scripts import audit_complete_highdim_leaderboard_phase0_freeze as independent


def test_phase0_freeze_binds_six_main_rows_and_one_sidecar() -> None:
    payload = freeze.build_freeze()
    freeze.validate_freeze(payload)

    assert payload["summary"] == {
        "num_main_rows": 6,
        "num_algorithms": 4,
        "num_main_cells": 24,
        "num_sidecar_rows": 1,
        "num_frozen_nonledh_baseline_candidates": 9,
        "num_current_closure_gaps": 15,
        "num_current_program_admitted_cells": 0,
        "num_current_source_ledh_five_seed_aggregates": 0,
        "numeric_leaderboard_complete": False,
    }
    assert payload["main_rows"] == [
        "benchmark_lgssm_exact_oracle_m3_T50",
        "zhao_cui_sv_actual_nongaussian_T1000",
        "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
        "zhao_cui_spatial_sir_austria_j9_T20",
        "zhao_cui_predator_prey_T20",
        "zhao_cui_generalized_sv_synthetic_from_estimated_values",
    ]
    assert payload["algorithms"] == [
        "fixed_sgqf",
        "ukf",
        "zhao_cui_scalar_or_multistate",
        "ledh_pfpf_ot",
    ]
    assert (
        payload["sidecar"]["row_id"]
        == "zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale"
    )
    assert payload["sidecar"]["counted_as_main_row"] is False
    assert payload["sidecar"]["closure_required_by_current_program"] is False
    assert payload["sidecar"]["target_signature_status"] == (
        "outside_main_24_cell_program_not_frozen"
    )
    assert [cell["phase0_closure_status"] for cell in payload["sidecar"]["cells"]] == [
        "not_applicable_scoped_sidecar",
        "not_applicable_scoped_sidecar",
        "historical_scoped_component_candidate_outside_program",
        "historical_scoped_diagnostic_outside_program",
    ]
    assert freeze.SIDECAR_ROW not in payload["main_rows"]
    for cell in payload["starting_cells"]:
        expected_source = (
            freeze.HISTORICAL_LEADERBOARD_PATH
            if cell["algorithm_id"] == "ledh_pfpf_ot"
            else freeze.NONLEDH_BASELINE_PATH
        )
        assert cell["status_source_artifact"] == expected_source
        assert (
            cell["status_source_artifact_sha256"]
            == freeze.EXPECTED_INPUT_HASHES[expected_source]
        )
    independent.audit(payload)


def test_phase0_freeze_binds_owner_approved_sir_target_identity() -> None:
    payload = freeze.build_freeze()

    assert payload["authority_supersession"] == {
        "authority_amendment_path": freeze.AUTHORITY_AMENDMENT_PATH,
        "authority_amendment_sha256": freeze.AUTHORITY_AMENDMENT_SHA256,
        "original_phase0_freeze_sha256": freeze.ORIGINAL_FREEZE_SHA256,
        "supersession_scope": (
            "sir_target_generation_identity_and_exact_row_extension_"
            "classifications_only"
        ),
    }
    assert (
        payload["policies"]["target_generation_identities"][
            "zhao_cui_spatial_sir_austria_j9_T20"
        ]
        == freeze.SIR_TARGET_GENERATION_IDENTITY
    )
    assert "81103" in freeze.SIR_TARGET_GENERATION_IDENTITY
    assert "not_author_matlab_rng1_reproduction" in freeze.SIR_TARGET_GENERATION_IDENTITY


def test_independent_audit_rejects_old_sir_target_identity() -> None:
    payload = copy.deepcopy(freeze.build_freeze())
    payload["policies"]["target_generation_identities"][
        "zhao_cui_spatial_sir_austria_j9_T20"
    ] = "fixed_austria_j9_source_observations_no_synthetic_seed_declared"

    with pytest.raises(ValueError, match="policy freeze mismatch"):
        independent.audit(payload, verify_repository_bytes=False)


def test_independent_audit_rejects_missing_authority_supersession() -> None:
    payload = copy.deepcopy(freeze.build_freeze())
    payload.pop("authority_supersession")

    with pytest.raises(ValueError, match="authority-supersession binding mismatch"):
        independent.audit(payload, verify_repository_bytes=False)


def test_phase0_freeze_rejects_sidecar_promotion() -> None:
    payload = copy.deepcopy(freeze.build_freeze())
    payload["main_rows"][-1] = freeze.SIDECAR_ROW

    with pytest.raises(ValueError, match="main-row freeze mismatch"):
        freeze.validate_freeze(payload)


def test_phase0_freeze_rejects_missing_main_cell() -> None:
    payload = copy.deepcopy(freeze.build_freeze())
    payload["starting_cells"].pop()

    with pytest.raises(ValueError, match="exactly 24 main cells"):
        freeze.validate_freeze(payload)


def test_phase0_freeze_rejects_nonledh_cell_sourced_from_historical_composite() -> None:
    payload = copy.deepcopy(freeze.build_freeze())
    payload["starting_cells"][0][
        "status_source_artifact"
    ] = freeze.HISTORICAL_LEADERBOARD_PATH
    payload["starting_cells"][0]["status_source_artifact_sha256"] = (
        freeze.EXPECTED_INPUT_HASHES[freeze.HISTORICAL_LEADERBOARD_PATH]
    )

    with pytest.raises(ValueError, match="cell source-binding mismatch"):
        freeze.validate_freeze(payload)


def test_independent_audit_rejects_forged_closure_and_policy() -> None:
    payload = copy.deepcopy(freeze.build_freeze())
    payload["starting_cells"][0]["phase0_closure_status"] = (
        "gap_target_matched_value_and_score_evaluator"
    )
    with pytest.raises(ValueError, match="starting-cell status mismatch"):
        independent.audit(payload, verify_repository_bytes=False)

    payload = copy.deepcopy(freeze.build_freeze())
    payload["policies"]["ledh_fd_rule"] = "max_relative_error <= 0.005"
    with pytest.raises(ValueError, match="policy freeze mismatch"):
        independent.audit(payload, verify_repository_bytes=False)


def test_independent_auditor_does_not_import_generator_module() -> None:
    source = Path(independent.__file__).read_text(encoding="utf-8")
    assert "build_complete_highdim_leaderboard_phase0_freeze" not in source


def test_sha_bound_loader_rejects_modified_input(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        freeze._load_sha_bound(path, "0" * 64)  # noqa: SLF001
