"""Focused CPU checks for the three-mode corrected-HMC runner contract."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_weighted_neutra_three_mode_hmc_2026_08_12.py"
SPEC = importlib.util.spec_from_file_location("three_mode_hmc_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_runner_binds_the_reviewed_three_mode_checkpoint_and_plan() -> None:
    assert RUNNER.CHECKPOINT.name == "trainer_states.json"
    assert "component-aware-width128-depth6-updates10000-r1" in RUNNER.CHECKPOINT.as_posix()
    assert RUNNER.EXPECTED_CHECKPOINT_SHA256 == (
        "b39c682030fb3ba8bafe863c747674db40b5d7c13e164c8445ddfab649ad93f6"
    )
    assert RUNNER.PLAN.is_file()
    assert "--checkpoint" in SCRIPT.read_text(encoding="utf-8")
    assert 'choices=("reviewed", "fresh-replication")' in SCRIPT.read_text(
        encoding="utf-8"
    )


def _loaded_candidate(**overrides: object) -> SimpleNamespace:
    values = {
        "checkpoint_sha256": RUNNER.EXPECTED_CHECKPOINT_SHA256,
        "selected_step": RUNNER.EXPECTED_SELECTED_STEP,
        "config": SimpleNamespace(
            hidden_layers=RUNNER.EXPECTED_HIDDEN_LAYERS,
            stages=RUNNER.EXPECTED_STAGES,
            jit_compile=True,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_reviewed_candidate_eligibility_accepts_exact_identity() -> None:
    RUNNER._require_reviewed_candidate(
        _loaded_candidate(),
        {"target_signature": RUNNER.EXPECTED_TARGET_SIGNATURE},
    )


def test_target_identity_comparison_allows_roundoff_but_rejects_change() -> None:
    assert RUNNER._nested_numeric_close(
        {"covariance": [[1.0, 0.25]]},
        {"covariance": [[1.0 + 5.0e-15, 0.25 - 5.0e-15]]},
    )
    assert not RUNNER._nested_numeric_close(
        {"covariance": [[1.0, 0.25]]},
        {"covariance": [[1.0, 0.250001]]},
    )


@pytest.mark.parametrize(
    ("loaded", "target", "mismatch"),
    (
        (
            _loaded_candidate(checkpoint_sha256="0" * 64),
            {"target_signature": RUNNER.EXPECTED_TARGET_SIGNATURE},
            "checkpoint_sha256",
        ),
        (
            _loaded_candidate(selected_step=1000),
            {"target_signature": RUNNER.EXPECTED_TARGET_SIGNATURE},
            "selected_step",
        ),
        (
            _loaded_candidate(
                config=SimpleNamespace(hidden_layers=(64, 64), stages=3, jit_compile=True)
            ),
            {"target_signature": RUNNER.EXPECTED_TARGET_SIGNATURE},
            "hidden_layers",
        ),
        (
            _loaded_candidate(),
            {"target_signature": "wrong-target"},
            "target_signature",
        ),
    ),
)
def test_reviewed_candidate_eligibility_rejects_mismatch(
    loaded: SimpleNamespace, target: dict[str, object], mismatch: str
) -> None:
    with pytest.raises(RuntimeError, match=mismatch):
        RUNNER._require_reviewed_candidate(loaded, target)


def test_hmc_tuning_grid_forbids_one_step_trajectory() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "leapfrog_grid=(3, 5, 10, 15, 20, 25)" in source
    assert "if leapfrog < 2:" in source
    assert "L=1 is forbidden" in source
    assert "three_mode_hmc_candidate_rejected_at_tuning" in source


def test_retained_loader_rejects_missing_archived_chunks(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "weighted-three-mode-manifest.json").write_text(
        '{"retained_chunks": []}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="no retained chunks"):
        RUNNER._load_retained_samples(
            object(), {"archive": {"root": archive.as_posix()}}
        )
