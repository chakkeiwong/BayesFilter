from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.select_zhao_cui_austria_sir_lane_b_t1 import (
    BASELINE_ID,
    EXPECTED_ARMS,
    build_selection,
)


def _write_arm(root: Path, arm_id: str, *, viable: bool, rms: float) -> None:
    output = root / arm_id
    output.mkdir(parents=True)
    (output / "result.json").write_text(
        json.dumps(
            {
                "status": "VIABLE_T1_PILOT_ARM" if viable else "REJECTED_T1_PILOT_ARM",
                "baseline_id": BASELINE_ID,
                "arm": {"arm_id": arm_id},
                "gates": {"viable": viable},
                "validation_metrics": {"normalized_log_density_rms": rms},
                "artifact_manifest": f"{arm_id}/artifact/manifest.json" if viable else None,
                "artifact_identity": (arm_id * 64)[:64] if viable else None,
                "run_manifest": {"source_sha256": {"shared.py": "same"}},
            },
            sort_keys=True,
        )
        + "\n"
    )


def test_selection_requires_every_frozen_arm_and_chooses_minimum_viable(tmp_path: Path) -> None:
    for index, arm_id in enumerate(EXPECTED_ARMS):
        _write_arm(tmp_path, arm_id, viable=index in {1, 3}, rms=1.0 - 0.1 * index)
    selection = build_selection(tmp_path)
    assert selection["status"] == "SELECTED_VIABLE_T1_PILOT_ARM"
    assert selection["selected_arm_id"] == EXPECTED_ARMS[3]
    assert selection["untouched_seed_used"] is False


def test_selection_fails_closed_on_missing_arm(tmp_path: Path) -> None:
    for arm_id in EXPECTED_ARMS[:-1]:
        _write_arm(tmp_path, arm_id, viable=False, rms=1.0)
    with pytest.raises(ValueError, match="requires all frozen pilot arms"):
        build_selection(tmp_path)


def test_selection_rejects_mixed_source_closure(tmp_path: Path) -> None:
    for arm_id in EXPECTED_ARMS:
        _write_arm(tmp_path, arm_id, viable=True, rms=1.0)
    path = tmp_path / EXPECTED_ARMS[-1] / "result.json"
    payload = json.loads(path.read_text())
    payload["run_manifest"]["source_sha256"]["shared.py"] = "different"
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="identical source closure"):
        build_selection(tmp_path)
