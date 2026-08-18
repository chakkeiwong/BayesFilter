"""Static checks for the generic five-stage campaign supervisor."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_neutra_generic_five_stage_campaign_2026_08_15.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("generic_five_stage_campaign", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_has_matched_per_model_budgets() -> None:
    runner = _load_runner()
    assert set(runner.TARGET_BUDGETS) == {"funnel", "gaussian", "banana", "mixture"}
    for target, budget in runner.TARGET_BUDGETS.items():
        selected_path = budget["affine"] + budget["simple"] + 3 * budget["progressive"] + budget["joint"]
        assert selected_path == budget["cold"], target
    assert runner.TARGET_BUDGETS["funnel"]["cold"] == 5000
    assert runner.TARGET_BUDGETS["gaussian"]["cold"] == 1000


def test_campaign_replicates_only_after_first_seed_pass() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'for route in ("staged", "cold")' in source
    assert 'if any(row["passed"] for row in first_seed)' in source
    assert '"--resume"' in source
    assert '"TF_FORCE_GPU_ALLOW_GROWTH": "true"' in source
