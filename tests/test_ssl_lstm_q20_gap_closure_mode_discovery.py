from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_gap_closure_mode_discovery_2026_08_18.py"


def _load():
    spec = importlib.util.spec_from_file_location("gap_closure_mode_discovery", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_start_design_is_replayable_and_covers_both_families() -> None:
    module = _load()
    first = module.build_starts()
    second = module.build_starts()
    assert first == second
    assert len(first) == 32
    assert {row["family"] for row in first} == {
        "prior_gaussian",
        "prior_scale_hypercube_corner",
    }


def test_new_competing_stationary_cluster_triggers_enrichment() -> None:
    module = _load()
    known = {"plus": (1.0, 0.0), "minus": (-1.0, 0.0)}
    rows = (
        {"known_reference": True, "position": known["plus"], "log_prob": 0.0, "score_inf_norm": 0.0},
        {"known_reference": True, "position": known["minus"], "log_prob": -0.1, "score_inf_norm": 0.0},
        {"position": (0.0, 2.0), "log_prob": -2.0, "score_inf_norm": 1.0e-8},
    )
    result = module.classify_stationary_endpoints(rows, known)
    assert result["new_competing_cluster_triggered"] is True
    assert result["new_competing_cluster_count"] == 1


def test_nonstationary_or_known_endpoint_does_not_trigger() -> None:
    module = _load()
    known = {"plus": (1.0, 0.0), "minus": (-1.0, 0.0)}
    rows = (
        {"known_reference": True, "position": known["plus"], "log_prob": 0.0, "score_inf_norm": 0.0},
        {"known_reference": True, "position": known["minus"], "log_prob": -0.1, "score_inf_norm": 0.0},
        {"position": (1.0001, 0.0), "log_prob": -0.01, "score_inf_norm": 1.0e-8},
        {"position": (0.0, 2.0), "log_prob": -1.0, "score_inf_norm": 1.0e-2},
    )
    result = module.classify_stationary_endpoints(rows, known)
    assert result["new_competing_cluster_triggered"] is False
