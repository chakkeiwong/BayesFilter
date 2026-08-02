from __future__ import annotations

import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_phase0 import (
    P88_ARTIFACT_SHA256,
    P88_BASIS_DIM,
    P88_CORE_COUNT,
    P88_DIMENSION,
    P88_RANK,
    P88_TAU,
    P88_TIME_INDEX,
    reconstruct_p88_phase0,
)
from scripts.run_zhao_cui_austria_sir_fixed_variant_phase0 import run


def test_phase0_reconstructs_exact_p88_t1_density_and_fails_closed_t2_boundary() -> None:
    audit = reconstruct_p88_phase0(Path(__file__).parents[2])
    assert audit.artifact_sha256 == P88_ARTIFACT_SHA256
    assert audit.status == "BLOCK_FIXED_VARIANT_BASELINE_NOT_RECONSTRUCTIBLE"
    assert "missing_explicit_transport_branch_metadata" in audit.blockers
    assert "missing_source_dependency_closure" in audit.blockers
    assert audit.t2_boundary_status == "BLOCK_T2_BOUNDARY_METADATA_MISSING"
    assert len(audit.cores) == P88_CORE_COUNT
    assert tuple(int(core.shape[1]) for core in audit.cores) == (P88_BASIS_DIM,) * P88_DIMENSION
    assert tuple(int(core.shape[2]) for core in audit.cores[:-1]) == (P88_RANK,) * (P88_DIMENSION - 1)
    assert audit.artifact["route_manifest"]["target_id"] == "zhao_cui_sir_austria_d18"
    assert audit.artifact["training_backend"] == "training_base_optimizer"
    assert audit.artifact["training_config"]["defensive_tau"] == P88_TAU
    assert audit.artifact["fit_data_manifest"]["fit_data_mode"] == (
        "source_pushed_computeL_weighted_augmented_samples"
    )
    tf.debugging.assert_all_finite(audit.density.normalizer(), "P88 density normalizer")
    tf.debugging.assert_positive(audit.density.normalizer())


def test_phase0_manifest_records_density_reconstruction_without_t2_claim() -> None:
    audit = reconstruct_p88_phase0(Path(__file__).parents[2])
    payload = audit.manifest_payload()
    assert payload["time_index"] == P88_TIME_INDEX
    assert payload["p88_artifact_path"].endswith(
        "bayesfilter-highdim-zhao-cui-p88-phase2-degree-order3-rank4-lr3e-4-l1-0-fit-2026-06-27.json"
    )
    assert payload["core_count"] == P88_CORE_COUNT
    assert payload["basis_dim"] == (P88_BASIS_DIM,) * P88_DIMENSION
    assert payload["source_observation_hash_binding"] == "absent_in_p88_artifact"
    assert payload["transport_branch_metadata"] == "absent_in_p88_artifact"
    assert payload["source_dependency_closure"] == "absent_in_p88_artifact"
    assert payload["source_relation"]["p88_trainer"] == "extension_or_invention"
    assert payload["source_relation"]["squared_tt_density"] == (
        "source_faithful_mathematical_operation"
    )
    assert payload["density_branch_hash"] == audit.artifact["branch_hash"]


def test_phase0_rejects_tampered_p88_payload_without_rewriting_repository(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / (
        "docs/plans/bayesfilter-highdim-zhao-cui-p88-phase2-degree-order3-rank4-"
        "lr3e-4-l1-0-fit-2026-06-27.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["trained_core_serialization"]["cores"][0]["values"][0][0][0] += 1.0e-3
    target = tmp_path / "docs/plans" / source.name
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="P88 artifact SHA-256 mismatch"):
        reconstruct_p88_phase0(tmp_path)


def test_phase0_runner_manifest_stops_before_phase1(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[2]
    output = tmp_path / "phase0.json"
    payload = run(repository_root, output)
    assert payload["status"] == "BLOCK_FIXED_VARIANT_BASELINE_NOT_RECONSTRUCTIBLE"
    assert payload["density_reconstruction_status"] == (
        "PASS_EXACT_P88_T1_DENSITY_PARITY"
    )
    assert payload["phase1_authorized"] is False
    assert payload["p88_artifact_path"].endswith(
        "bayesfilter-highdim-zhao-cui-p88-phase2-degree-order3-rank4-lr3e-4-l1-0-fit-2026-06-27.json"
    )
    assert payload["result_artifact_path"] == str(output)
    assert payload["blockers"] == [
        "missing_explicit_transport_branch_metadata",
        "missing_source_dependency_closure",
    ]
