from __future__ import annotations

import os
import json
import inspect
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.testing import lgssm_neutra_serious_validation_tf as parent
from bayesfilter.testing import lgssm_neutra_target_specific_protocol_tf as campaign
from bayesfilter.testing import lgssm_neutra_strict_training_tf as strict_training


def test_target_specific_protocol_has_no_numpy_dependency() -> None:
    source = inspect.getsource(campaign)
    assert "import numpy" not in source
    assert "from numpy" not in source


def test_strict_training_closure_and_cli_dispatch_are_numpy_free() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import json; "
                "from bayesfilter.testing import lgssm_neutra_strict_training_tf as m; "
                "from bayesfilter.runtime.gpu_memory_policy import "
                "configure_tensorflow_gpu_memory_growth; "
                "import tensorflow as tf; "
                "configure_tensorflow_gpu_memory_growth(tf, require_gpu=False); "
                "print(json.dumps(m.audit_imported_bayesfilter_closure()))"
            ),
        ),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    closure = json.loads(completed.stdout)
    assert closure["passed"] is True
    assert closure["violations"] == []
    assert not any(
        row["path"] == "bayesfilter/runtime/runner.py"
        for row in closure["modules"]
    )
    source = Path(
        "docs/benchmarks/run_lgssm_neutra_target_specific_protocol_2026_07_14.py"
    ).read_text(encoding="utf-8")
    train_branch = source.split('if args.stage == "train":', 1)[1].split(
        'if args.artifact_root is not None', 1
    )[0]
    assert "lgssm_neutra_strict_training_tf" in train_branch
    assert "lgssm_neutra_target_specific_protocol_tf" not in train_branch
    campaign_source = inspect.getsource(campaign.run_gpu_training_job)
    assert "lgssm_neutra_strict_training_tf" in campaign_source
    assert "train_plain_dense_iaf" not in campaign_source


def _row(recipe_id: str, values, *, passed: bool = True):
    return {
        "job_kind": "screen",
        "job_id": recipe_id,
        "passed": passed,
        "vetoes": (() if passed else ("synthetic_veto",)),
        "heldout_common_batches": (
            None
            if not passed
            else {
                "rows": tuple(
                    {
                        "seed": seed,
                        "reverse_kl_objective_mean": float(value),
                    }
                    for seed, value in zip(campaign.HELDOUT_SEEDS, values)
                )
            }
        ),
    }


def test_contract_has_explicit_recipes_and_long_budget() -> None:
    contract = campaign.campaign_contract_payload()

    assert tuple(contract["screen"]["recipe_order"]) == campaign.SCREEN_RECIPE_ORDER
    assert len(contract["screen"]["recipes"]) == 4
    assert contract["final_training"]["steps"] == 5000
    assert contract["final_training"]["batch_size"] == 128
    assert contract["final_training"]["checkpoint_every"] == 50
    assert contract["final_training"]["screen_weights_reused"] is False


def test_seed_ledger_is_disjoint() -> None:
    assert campaign.validate_seed_ledger()["passed"] is True


def test_selection_prefers_source_anchor_within_paired_mcse() -> None:
    base = np.arange(8, dtype=np.float64)
    rows = (
        _row(campaign.SOURCE_ANCHOR, base),
        _row("lower_lr1e3", base + np.array([-1.0, 0.8] * 4)),
        _row("shallow_2stage_lr5e3", base + 2.0),
        _row("wide_2x_lr5e3", base + 3.0),
    )

    result = campaign.select_screen_recipe(rows)

    assert result["provisional_lowest_mean_recipe_id"] == "lower_lr1e3"
    assert result["selected_recipe_id"] == campaign.SOURCE_ANCHOR
    assert result["source_anchor_preference_applied"] is True
    assert result["ranking_statistically_supported"] is False


def test_selection_uses_lower_mean_when_difference_exceeds_paired_mcse() -> None:
    base = np.arange(8, dtype=np.float64)
    rows = (
        _row(campaign.SOURCE_ANCHOR, base),
        _row("lower_lr1e3", base - 2.0),
        _row("shallow_2stage_lr5e3", base + 2.0),
        _row("wide_2x_lr5e3", base + 3.0),
    )

    result = campaign.select_screen_recipe(rows)

    assert result["selected_recipe_id"] == "lower_lr1e3"
    assert result["source_anchor_preference_applied"] is False


def test_zero_survivors_is_terminal_without_recipe() -> None:
    rows = tuple(_row(item, (), passed=False) for item in campaign.SCREEN_RECIPE_ORDER)

    result = campaign.select_screen_recipe(rows)

    assert result["selected_recipe_id"] is None
    assert result["status"] == "zero_surviving_recipes_terminal"
    assert len(result["candidate_failure_table"]) == 4


def test_source_anchor_failure_selects_lowest_mean_without_mcse_fallback() -> None:
    zeros = np.zeros(8, dtype=np.float64)
    rows = (
        _row(campaign.SOURCE_ANCHOR, (), passed=False),
        _row("lower_lr1e3", np.array([-1.0, 1.1] * 4)),
        _row("shallow_2stage_lr5e3", np.ones(8)),
        _row("wide_2x_lr5e3", zeros),
    )

    result = campaign.select_screen_recipe(rows)

    assert result["provisional_lowest_mean_recipe_id"] == "wide_2x_lr5e3"
    assert result["selected_recipe_id"] == "wide_2x_lr5e3"


def test_selection_rejects_reordered_or_duplicate_rows() -> None:
    base = np.arange(8, dtype=np.float64)
    rows = tuple(_row(item, base) for item in campaign.SCREEN_RECIPE_ORDER)

    with pytest.raises(campaign.TargetSpecificProtocolError, match="exact recipe order"):
        campaign.select_screen_recipe(tuple(reversed(rows)))
    with pytest.raises(campaign.TargetSpecificProtocolError, match="exact recipe order"):
        campaign.select_screen_recipe((*rows[:-1], rows[-2]))


def test_failed_screen_reference_has_null_heldout(tmp_path, monkeypatch) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(campaign, "_resolve_result_path", lambda *_args: result_path)
    monkeypatch.setattr(
        campaign,
        "_file_reference",
        lambda path: {"path": str(path), "file_sha256": "2" * 64, "byte_count": 2},
    )
    row = _row(campaign.SOURCE_ANCHOR, (), passed=False)
    row["artifact_hash"] = "sha256:" + "1" * 64

    reference = campaign._screen_result_reference(row)

    assert reference["heldout_mean"] is None
    assert reference["heldout_mcse"] is None
    assert reference["vetoes"] == ("synthetic_veto",)


def test_resume_checkpoint_must_be_latest_and_nonterminal(tmp_path) -> None:
    first = tmp_path / "attempt_1"
    training = first / "training"
    training.mkdir(parents=True)
    for step in (50, 100):
        (training / f"checkpoint_step_{step:06d}.json").write_text(
            json.dumps({"completed_steps": step}), encoding="utf-8"
        )

    with pytest.raises(campaign.TargetSpecificProtocolError, match="latest"):
        campaign._validate_resume_checkpoint(
            first_attempt_root=first,
            checkpoint=training / "checkpoint_step_000050.json",
            planned_steps=500,
        )
    campaign._validate_resume_checkpoint(
        first_attempt_root=first,
        checkpoint=training / "checkpoint_step_000100.json",
        planned_steps=500,
    )
    (first / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(campaign.TargetSpecificProtocolError, match="terminal"):
        campaign._validate_resume_checkpoint(
            first_attempt_root=first,
            checkpoint=training / "checkpoint_step_000100.json",
            planned_steps=500,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("recipe", {"recipe_id": "wrong"}, "recipe"),
        ("seed", (1, 2), "seed"),
        ("steps", 499, "step-budget"),
        ("target_signature", "0" * 64, "target signature"),
        ("adapter_signature", "0" * 64, "adapter signature"),
    ),
)
def test_training_result_identity_drift_fails_closed(
    tmp_path, monkeypatch, field, replacement, message
) -> None:
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", tmp_path)
    recipe_id = campaign.SOURCE_ANCHOR
    recipe = campaign.SCREEN_RECIPES[recipe_id]
    result_path = (
        tmp_path / "screen" / "candidates" / recipe_id / "attempt_1" / "result.json"
    )
    result_path.parent.mkdir(parents=True)
    row = {
        "job_kind": "screen",
        "job_id": recipe_id,
        "recipe": recipe.payload(),
        "seed": campaign.SCREEN_SEED,
        "steps": campaign.SCREEN_STEPS,
        "target_signature": parent.EXPECTED_TARGET_SIGNATURE,
        "adapter_signature": parent.EXPECTED_ADAPTER_SIGNATURE,
    }
    row[field] = replacement
    result_path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(campaign.TargetSpecificProtocolError, match=message):
        campaign._resolve_training_job_result("screen", recipe_id)


def test_phase4_affine_only_cannot_enter_phase5(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", tmp_path)
    result_path = tmp_path / "phase4" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "phase": 4,
                "passed": True,
                "viable_candidates": ["affine_control"],
                "learned_viable_candidates": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(campaign.TargetSpecificProtocolError, match="no surviving"):
        campaign._require_target_specific_phase4_survivor()


def test_historical_affine_identity_drift_fails_closed() -> None:
    valid = {
        "phase": 4,
        "candidate_id": "affine_control",
        "passed": True,
        "target_signature": parent.EXPECTED_TARGET_SIGNATURE,
        "adapter_signature": parent.EXPECTED_ADAPTER_SIGNATURE,
        "artifact_hash": campaign.EXPECTED_HISTORICAL_AFFINE_ARTIFACT_HASH,
        "payload": {
            "path": str(
                (
                    campaign.HISTORICAL_ROOT
                    / "phase4/affine_control/frozen_transport.json"
                ).relative_to(campaign.ROOT)
            ),
            "file_sha256": campaign.EXPECTED_HISTORICAL_AFFINE_PAYLOAD_SHA256,
            "byte_count": (
                campaign.HISTORICAL_ROOT
                / "phase4/affine_control/frozen_transport.json"
            ).stat().st_size,
        },
    }
    campaign._validate_historical_affine_result(valid)
    changed = dict(valid)
    changed["candidate_id"] = "dense_seed1201"

    with pytest.raises(campaign.TargetSpecificProtocolError, match="identity"):
        campaign._validate_historical_affine_result(changed)


def test_phase4_survivor_revalidates_underlying_training_job(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", tmp_path)
    candidate = "dense_seed1201"
    payload = tmp_path / "payload.json"
    checkpoint = tmp_path / "checkpoint.json"
    progress = tmp_path / "progress.jsonl"
    for path in (payload, checkpoint, progress):
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        campaign,
        "_verify_reference_payload",
        lambda _reference, _label: payload,
    )
    monkeypatch.setattr(
        campaign,
        "_verify_file_reference",
        lambda _reference, expected_path, _label: expected_path,
    )
    references = {
        "payload": {"path": str(payload), "file_sha256": "a" * 64, "byte_count": 2},
        "checkpoint": {
            "path": str(checkpoint),
            "file_sha256": "b" * 64,
            "byte_count": 2,
        },
        "progress": {"path": str(progress), "file_sha256": "c" * 64, "byte_count": 2},
    }
    training_job = {
        "artifact_hash": "sha256:" + "1" * 64,
        **references,
    }
    monkeypatch.setattr(
        campaign,
        "_resolve_training_job_result",
        lambda job_kind, job_id: training_job
        if (job_kind, job_id) == ("final", candidate)
        else None,
    )
    candidate_result_path = tmp_path / "phase4" / candidate / "result.json"
    candidate_result_path.parent.mkdir(parents=True)
    candidate_result = {
        "phase": 4,
        "candidate_id": candidate,
        "passed": True,
        "target_signature": parent.EXPECTED_TARGET_SIGNATURE,
        "adapter_signature": parent.EXPECTED_ADAPTER_SIGNATURE,
        "artifact_hash": "sha256:" + "2" * 64,
        "payload": references["payload"],
        "training": {
            "source_training_job_artifact_hash": "sha256:" + "9" * 64,
            "checkpoint": references["checkpoint"],
            "progress": references["progress"],
        },
    }
    candidate_result_path.write_text(json.dumps(candidate_result), encoding="utf-8")
    aggregate_path = tmp_path / "phase4" / "result.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "passed": True,
                "viable_candidates": ["affine_control", candidate],
                "learned_viable_candidates": [candidate],
                "candidate_results": [
                    {
                        "candidate_id": candidate,
                        "result": {
                            "path": str(candidate_result_path),
                            "file_sha256": campaign._file_sha256(
                                candidate_result_path
                            ),
                            "byte_count": candidate_result_path.stat().st_size,
                        },
                        "result_artifact_hash": candidate_result["artifact_hash"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(campaign.TargetSpecificProtocolError, match="training-job"):
        campaign._require_target_specific_phase4_survivor()


def test_candidate_specific_downstream_gate_rejects_other_learned_arm(monkeypatch) -> None:
    monkeypatch.setattr(
        campaign,
        "_require_target_specific_phase4_survivor",
        lambda: {
            "viable_candidates": ["affine_control", "dense_seed1201"],
            "learned_viable_candidates": ["dense_seed1201"],
        },
    )

    campaign._require_phase4_candidate_for_downstream("dense_seed1201")
    with pytest.raises(campaign.TargetSpecificProtocolError, match="not Phase 4 viable"):
        campaign._require_phase4_candidate_for_downstream("dense_seed1202")


def test_phase4_survivor_rehashes_checkpoint_and_progress(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(campaign, "ARTIFACT_ROOT", tmp_path)
    candidate = "dense_seed1201"
    payload = tmp_path / "payload.json"
    checkpoint = tmp_path / "checkpoint.json"
    progress = tmp_path / "progress.jsonl"
    for path in (payload, checkpoint, progress):
        path.write_text("{}", encoding="utf-8")

    def reference(path):
        return {
            "path": str(path.relative_to(tmp_path)),
            "file_sha256": campaign._file_sha256(path),
            "byte_count": path.stat().st_size,
        }

    references = {
        "payload": reference(payload),
        "checkpoint": reference(checkpoint),
        "progress": reference(progress),
    }
    training_job = {"artifact_hash": "sha256:" + "1" * 64, **references}
    monkeypatch.setattr(
        campaign,
        "_resolve_training_job_result",
        lambda *_args: training_job,
    )
    monkeypatch.setattr(
        campaign,
        "_verify_file_reference",
        lambda _reference, expected_path, _label: expected_path,
    )
    monkeypatch.setattr(
        campaign,
        "_expected_training_job_artifact_path",
        lambda *, job_kind, job_id, artifact: {
            "payload": payload,
            "checkpoint": checkpoint,
            "progress": progress,
        }[artifact],
    )

    def verify(reference_value, expected_path, label):
        if reference_value["file_sha256"] != campaign._file_sha256(expected_path):
            raise campaign.TargetSpecificProtocolError(f"{label} file hash mismatch")
        return expected_path

    monkeypatch.setattr(campaign, "_verify_file_reference", verify)
    candidate_result_path = tmp_path / "phase4" / candidate / "result.json"
    candidate_result_path.parent.mkdir(parents=True)
    candidate_result = {
        "phase": 4,
        "candidate_id": candidate,
        "passed": True,
        "target_signature": parent.EXPECTED_TARGET_SIGNATURE,
        "adapter_signature": parent.EXPECTED_ADAPTER_SIGNATURE,
        "artifact_hash": "sha256:" + "2" * 64,
        "payload": references["payload"],
        "training": {
            "source_training_job_artifact_hash": training_job["artifact_hash"],
            "checkpoint": references["checkpoint"],
            "progress": references["progress"],
        },
    }
    candidate_result_path.write_text(json.dumps(candidate_result), encoding="utf-8")
    aggregate_path = tmp_path / "phase4" / "result.json"

    def write_aggregate():
        aggregate_path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "viable_candidates": ["affine_control", candidate],
                    "learned_viable_candidates": [candidate],
                    "candidate_results": [
                        {
                            "candidate_id": candidate,
                            "result": {
                                "path": str(candidate_result_path),
                                "file_sha256": campaign._file_sha256(
                                    candidate_result_path
                                ),
                                "byte_count": candidate_result_path.stat().st_size,
                            },
                            "result_artifact_hash": candidate_result["artifact_hash"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    write_aggregate()
    campaign._require_target_specific_phase4_survivor()
    training_job["checkpoint"]["file_sha256"] = "0" * 64
    candidate_result["training"]["checkpoint"] = training_job["checkpoint"]
    candidate_result_path.write_text(json.dumps(candidate_result), encoding="utf-8")
    write_aggregate()

    with pytest.raises(campaign.TargetSpecificProtocolError, match="checkpoint"):
        campaign._require_target_specific_phase4_survivor()


def test_expected_training_artifact_paths_are_candidate_canonical(
    tmp_path, monkeypatch
) -> None:
    result = tmp_path / "phase4/training_jobs/dense_seed1201/attempt_2/result.json"
    monkeypatch.setattr(campaign, "_resolve_result_path", lambda *_args: result)
    monkeypatch.setattr(
        campaign,
        "_job_spec",
        lambda **_kwargs: (
            campaign.SCREEN_RECIPES[campaign.SOURCE_ANCHOR],
            campaign.FINAL_SEEDS["dense_seed1201"],
            campaign.FINAL_STEPS,
            tmp_path,
        ),
    )

    assert campaign._expected_training_job_artifact_path(
        job_kind="final", job_id="dense_seed1201", artifact="payload"
    ) == result.parent / "training/frozen_transport.json"
    assert campaign._expected_training_job_artifact_path(
        job_kind="final", job_id="dense_seed1201", artifact="checkpoint"
    ) == result.parent / "training/checkpoint_step_005000.json"
    assert campaign._expected_training_job_artifact_path(
        job_kind="final", job_id="dense_seed1201", artifact="progress"
    ) == result.parent / "training/training_progress.jsonl"


def test_parent_context_is_explicit_and_resettable() -> None:
    contract = campaign.campaign_contract_payload()
    plan = campaign.PLAN_PATH
    root = parent.ROOT / "docs/benchmarks/artifacts/context-test-not-created"

    parent.configure_execution_context(
        plan_path=plan,
        artifact_root=root,
        contract_payload=contract,
    )
    try:
        assert parent.PLAN_PATH == plan.resolve()
        assert parent.ARTIFACT_ROOT == root.resolve()
        assert parent.campaign_contract_payload()["schema"] == contract["schema"]
    finally:
        parent.reset_execution_context()

    assert parent.PLAN_PATH == parent._DEFAULT_PLAN_PATH
    assert parent.ARTIFACT_ROOT == parent._DEFAULT_ARTIFACT_ROOT


def test_parent_context_rejects_paths_outside_repo(tmp_path) -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        parent.configure_execution_context(
            plan_path=campaign.PLAN_PATH,
            artifact_root=tmp_path,
            contract_payload=campaign.campaign_contract_payload(),
        )


def test_parent_affine_payload_lookup_honors_result_reference(tmp_path, monkeypatch) -> None:
    root = parent.ROOT / "docs/benchmarks/artifacts/context-payload-test-not-created"
    result_path = root / "phase4" / "affine_control" / "result.json"
    payload = parent.ROOT / "docs/benchmarks/artifacts/synthetic-affine-payload.json"
    monkeypatch.setattr(parent, "ARTIFACT_ROOT", root)
    monkeypatch.setattr(
        parent,
        "_read_mapping",
        lambda path, _label: {"payload": {"path": str(payload.relative_to(parent.ROOT))}}
        if path == result_path
        else {},
    )
    monkeypatch.setattr(type(result_path), "is_file", lambda self: self == result_path)

    assert parent.candidate_payload_path("affine_control") == payload.resolve()
