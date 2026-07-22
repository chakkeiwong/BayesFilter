from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesfilter.highdim.ledh_historical_raw_policy import (
    HISTORICAL_RAW_BARYCENTRIC_STATUS,
    require_historical_raw_diagnostic_opt_in,
)
from docs.benchmarks import benchmark_ledh_compact_score_gpu_xla as harness


def test_phase6_shared_opt_in_returns_exact_historical_status() -> None:
    assert (
        require_historical_raw_diagnostic_opt_in(
            True,
            route_name="test raw route",
        )
        == HISTORICAL_RAW_BARYCENTRIC_STATUS
    )


def test_phase6_harness_rejects_before_raw_score_callable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = False

    def raw_sentinel(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("raw score callable must not run without explicit opt-in")

    output = tmp_path / "no-opt-in.json"
    monkeypatch.setattr(harness, "_score_only", raw_sentinel)
    with pytest.raises(ValueError, match="historical raw-barycentric diagnostic"):
        harness.main(
            [
                "--row",
                "predator-prey",
                "--stage",
                "score-only",
                "--output",
                str(output),
            ]
        )

    assert called is False
    terminal = json.loads(output.read_text(encoding="utf-8"))
    assert terminal["artifact_status"] == "failed"
    assert terminal["terminal_artifact"] is True

