from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from docs.benchmarks.run_genut_str_ukf_leaderboard import _summarize_rows


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = (
    ROOT
    / "docs/benchmarks/artifacts/genut_str_ukf_leaderboard_20260722/leaderboard_extension.json"
)


def test_nonfinite_claim_rows_cannot_be_partially_summarized() -> None:
    rows = [
        {"value": -1.0, "score": [0.0] * 5, "finite": True},
        {"value": None, "score": [None, 0.0, 0.0, 0.0, 0.0], "finite": False},
    ]
    with pytest.raises(ValueError, match="non-finite candidate rows"):
        _summarize_rows(rows)


def test_blocked_str_ukf_genut_cell_is_included_without_partial_estimates() -> None:
    payload = json.loads(EXTENSION.read_text(encoding="utf-8"))
    cell = payload["leaderboard"]
    assert payload["target"]["cell_id"] == "STR-UKF"
    assert payload["method"]["particle_count"] > 1000
    assert cell["included"] is True
    assert cell["admitted"] is False
    assert cell["cell_status"] == "blocked_nonfinite_full_horizon_multiseed_claim"
    assert cell["value"] is None
    assert cell["value_ci95"] is None
    assert cell["score"] is None
    assert cell["score_ci95"] is None
    assert math.isfinite(payload["same_target_comparator"]["value"])
