from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_phase4_bounded_training_2026_07_14.py"
)


def _load_harness():
    name = "ssl_lstm_neutra_phase4_bounded_training_harness"
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_harness()


def _validation(loss, saturation=0.0):
    return {
        "per_sample_loss": [float(loss)] * harness.VALIDATION_BATCH_SIZE,
        "saturation_fraction": float(saturation),
    }


def _probes(**overrides):
    values = {
        "all_finite": True,
        "roundtrip_max_abs": 1.0e-12,
        "original_neighborhood_max_inverse_radius": 3.0,
        "moderate_shell_max_inverse_radius": 4.0,
    }
    values.update(overrides)
    return values


def test_candidate_ladder_and_seed_namespaces_are_frozen() -> None:
    assert tuple(harness.CANDIDATES) == (
        "affine_a",
        "affine_b",
        "dense_a",
        "dense_b",
        "repair_a",
        "repair_b",
    )
    assert harness.CANDIDATES["affine_a"].role_code == 2101
    assert harness.CANDIDATES["affine_b"].role_code == 2102
    assert harness.CANDIDATES["dense_a"].role_code == 2101
    assert harness.CANDIDATES["dense_b"].role_code == 2102
    assert harness.CANDIDATES["repair_a"].role_code == 2110
    assert harness.CANDIDATES["repair_b"].role_code == 2111
    for candidate in harness.CANDIDATES.values():
        assert candidate.role_code != candidate.validation_role_code
        assert candidate.batch_size == 64
    assert harness.CANDIDATES["dense_a"].steps == 2000
    assert harness.CANDIDATES["repair_a"].learning_rate == pytest.approx(3.0e-4)


def test_paired_loss_upper_bound_passes_improvement_and_rejects_flat() -> None:
    initial = [2.0 + 0.01 * index for index in range(64)]
    improved = [value - 0.5 for value in initial]
    flat = list(initial)
    passing = harness.paired_loss_upper_bound(initial, improved)
    failing = harness.paired_loss_upper_bound(initial, flat)
    assert passing["one_sided_95_upper"] < 0.0
    assert failing["one_sided_95_upper"] == pytest.approx(0.0)
    with pytest.raises(harness.Phase4TrainingError, match="64 paired"):
        harness.paired_loss_upper_bound(initial[:-1], improved[:-1])


def test_probe_bank_has_frozen_roles_and_no_overlap() -> None:
    points, labels, metadata = harness._probe_bank()
    assert tuple(points.shape) == (68, 4)
    assert len(labels) == len(set(labels)) == 68
    assert metadata["original_neighborhood_count"] == 36
    assert metadata["moderate_shell_count"] == 8
    assert metadata["far_tail_count"] == 8
    assert metadata["prior_probe_count"] == 16
    assert metadata["point_count"] == 68
    assert metadata["prior_probe_seed"] == [20260714, 3301]
    assert metadata["sampler_geometry_sha256"]


def test_candidate_decision_separates_hard_and_promotion_vetoes() -> None:
    spec = harness.CANDIDATES["dense_a"]
    pass_interval = {
        "one_sided_95_upper": -0.1,
    }
    decision, hard, promotion = harness._candidate_decision(
        spec=spec,
        initial_validation=_validation(2.0),
        final_validation=_validation(1.0),
        loss_interval=pass_interval,
        probes=_probes(),
        affine_raw_scale_max_abs=None,
    )
    assert decision == "VIABLE_FROZEN_CANDIDATE"
    assert hard == []
    assert promotion == []

    decision, hard, promotion = harness._candidate_decision(
        spec=spec,
        initial_validation=_validation(2.0),
        final_validation=_validation(1.0, saturation=0.10),
        loss_interval={"one_sided_95_upper": 0.1},
        probes=_probes(
            roundtrip_max_abs=1.0e-3,
            original_neighborhood_max_inverse_radius=5.0,
        ),
        affine_raw_scale_max_abs=None,
    )
    assert decision == "INVALID_HARD_VETO"
    assert "roundtrip_residual_above_threshold" in hard
    assert "original_neighborhood_missing_support" in promotion
    assert "heldout_loss_improvement_not_established" in promotion
    assert "dense_scale_saturation_above_cap" in promotion


def test_prior_and_far_tail_radius_are_not_promotion_vetoes() -> None:
    probes = _probes(
        far_tail_max_inverse_radius=100.0,
        prior_probe_max_inverse_radius=100.0,
    )
    decision, hard, promotion = harness._candidate_decision(
        spec=harness.CANDIDATES["dense_a"],
        initial_validation=_validation(2.0),
        final_validation=_validation(1.0),
        loss_interval={"one_sided_95_upper": -0.1},
        probes=probes,
        affine_raw_scale_max_abs=None,
    )
    assert decision == "VIABLE_FROZEN_CANDIDATE"
    assert hard == []
    assert promotion == []


def test_prior_budget_accumulates_completed_and_failed_candidate_receipts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    rows = []
    for index, seconds in enumerate((10.5, 20.25)):
        path = Path(f"candidate-{index}.json")
        payload = {
            "schema": harness.SCHEMA,
            "decision": "VIABLE_FROZEN_CANDIDATE" if index == 0 else "INVALID_HARD_VETO",
            "candidate": {"label": f"candidate-{index}"},
            "run_manifest": {"charged_gpu_seconds": seconds},
        }
        (tmp_path / path).write_text(json.dumps(payload), encoding="utf-8")
        rows.append(path)
    total, lineage = harness._prior_budget(tuple(rows))
    assert total == pytest.approx(30.75)
    assert [row["candidate"] for row in lineage] == ["candidate-0", "candidate-1"]


def test_fresh_directory_rejects_existing_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    output = Path("candidate")
    (tmp_path / output).mkdir()
    (tmp_path / output / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(harness.Phase4TrainingError, match="not fresh"):
        harness._require_fresh_directory(output)


def test_failed_attempts_do_not_claim_trusted_gpu_provenance() -> None:
    assert harness.FAILED_ATTEMPT_TRUST_BASIS == (
        "failure_receipt_only_gpu_provenance_not_established"
    )
    source = HARNESS_PATH.read_text(encoding="utf-8")
    failure_block = source[source.index('except Exception as error:') :]
    assert (
        '"trust_basis": FAILED_ATTEMPT_TRUST_BASIS'
        in failure_block
    )
    assert (
        '"trust_basis": (\n                            '
        '"owner_designated_managed_session_visible_gpu_trusted"'
        not in failure_block
    )
