from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/benchmarks/run_sir_ledh_teacher_comparison_gpu.py"
SPEC = importlib.util.spec_from_file_location("sir_ledh_teacher_comparison", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def test_student_summary_detects_shift_without_equivalence_claim() -> None:
    centered = campaign._student_summary([-1.0, 1.0] * 8)
    shifted = campaign._student_summary([2.0, 2.1] * 8)
    assert centered["contains_zero"] is True
    assert shifted["contains_zero"] is False
    assert centered["bonferroni_family_size"] == 4


def test_nonfinite_comparison_is_unavailable_without_dropping_pairs() -> None:
    left_value = campaign.tf.constant([1.0, float("nan")], campaign.DTYPE)
    left_score = campaign.tf.constant(
        [[1.0, 2.0, 3.0], [4.0, float("nan"), 6.0]], campaign.DTYPE
    )
    right_value = campaign.tf.zeros([2], campaign.DTYPE)
    right_score = campaign.tf.zeros([2, 3], campaign.DTYPE)

    summary = campaign._comparison_summaries(
        left_value, left_score, right_value, right_score
    )

    assert summary["available"] is False
    assert summary["unavailable_reason"] == "NONFINITE_PAIRED_SAMPLES"
    assert summary["invalid_value_pair_count"] == 1
    assert summary["invalid_score_pair_count"] == 1
    assert summary["value"] is None
    assert summary["score"] == [None, None, None]
    assert math.isnan(summary["value_samples"][1])


def test_json_safe_replaces_nonfinite_values_with_null() -> None:
    assert campaign._json_safe(
        {"values": [1.0, float("nan"), float("inf"), -float("inf")]}
    ) == {"values": [1.0, None, None, None]}


def test_model_rows_and_particle_policy_are_frozen() -> None:
    assert campaign.MODEL_ROWS == (
        ("two_node", (1, 2, 5), 87100),
        ("austria_d18", (2, 5, 20), 87200),
    )
    assert campaign.TEACHER_PARTICLES == (128, 256)
    assert campaign.LEDH_PARTICLES == 256
    assert campaign.REPLICATES == 16
    assert campaign.REQUIRED_GPU_MEMORY_LIMIT_MIB == 8192
    assert campaign.GPU_MEMORY_POLICY is None


def test_phase3_replay_tolerances_are_stricter_than_scientific_comparison() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "value_delta > 1.0e-10" in source
    assert "maximum_score_delta > 1.0e-10" in source
    assert "Phase 3 CPU/GPU prepared identities differ" in source


def test_two_node_target_is_spatially_coupled() -> None:
    model = campaign._model("two_node")
    assert model.state_dim() == 4
    assert model.observation_dim() == 2
    assert model.physical_model.base_model.neighbor_sets == ((1,), (0,))


def test_t1_preparation_has_one_unused_xla_safe_transition_slot() -> None:
    model = campaign._model("two_node")
    observations = campaign._observations(model, 1, 87099)
    seeds = campaign.tf.range(87100, 87100 + campaign.REPLICATES, dtype=campaign.tf.int32)
    prepared = campaign._prepared(model, observations, seeds)
    assert prepared["transition_noise"].shape == (
        campaign.REPLICATES,
        1,
        campaign.LEDH_PARTICLES,
        model.state_dim(),
    )
    campaign.tf.debugging.assert_equal(
        prepared["transition_noise"], campaign.tf.zeros_like(prepared["transition_noise"])
    )


def test_resource_exhaustion_writes_failure_artifacts_and_exits_nonzero(
    monkeypatch, tmp_path: Path
) -> None:
    output_root = tmp_path / "synthetic_oom"
    phase3_prepared = tmp_path / "prepared.json"
    phase3_cpu_result = tmp_path / "cpu.json"
    phase3_prepared.write_text("{}\n", encoding="utf-8")
    phase3_cpu_result.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        campaign,
        "GPU_MEMORY_POLICY",
        {
            "mode": "fixed_logical_device_limit",
            "physical_devices": ({"device": "/physical_device:GPU:0"},),
            "memory_limit_mib_per_physical_device": 8192,
        },
    )
    monkeypatch.setattr(
        campaign.tf.config,
        "list_physical_devices",
        lambda kind: [SimpleNamespace(name="/physical_device:GPU:0")],
    )
    monkeypatch.setattr(
        campaign,
        "_phase3_replay",
        lambda *_: (_ for _ in ()).throw(
            campaign.tf.errors.ResourceExhaustedError(None, None, "synthetic OOM")
        ),
    )
    monkeypatch.setattr(
        campaign.sys,
        "argv",
        [
            str(campaign.SCRIPT) if hasattr(campaign, "SCRIPT") else str(SCRIPT),
            "--output-root",
            str(output_root),
            "--phase3-prepared",
            str(phase3_prepared),
            "--phase3-cpu-result",
            str(phase3_cpu_result),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        campaign.main()

    assert raised.value.code == 1
    failure = json.loads((output_root / "failure.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((output_root / "progress.json").read_text(encoding="utf-8"))
    hashes = json.loads((output_root / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert failure["failure_classification"] == "TENSORFLOW_RESOURCE_EXHAUSTED"
    assert failure["active_stage"] == "phase3_final_source_replay"
    assert failure["continuation_veto"] is True
    assert manifest["status"] == "FAILED"
    assert manifest["failure_file"] == str(output_root / "failure.json")
    assert progress["status"] == "FAILED"
    assert set(hashes["artifacts"]) == {
        "failure.json",
        "progress.json",
        "run_manifest.json",
    }
