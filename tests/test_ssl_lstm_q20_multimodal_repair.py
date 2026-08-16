"""Static contract tests for the bounded SSL-LSTM multimodal repair runner."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_multimodal_repair_2026_08_10.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-multimodal-repair-plan-2026-08-10.md"


def _assignments() -> dict[str, object]:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def test_runner_uses_bounded_xla_cpu_diagnostic_and_no_nuts() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "jit_compile=True" in source
    assert "ReplicaExchangeMC" not in source  # TFP mechanics stay in shared helper.
    assert "NoUTurnSampler" not in source
    assert "NUTS" not in source
    assert 'choices=("synthetic", "ssl-canary")' in source


def test_ssl_canary_is_small_and_cannot_launch_material_sampling() -> None:
    values = _assignments()
    assert values["SSL_TRANSITIONS"] == 4
    assert values["SSL_CHAINS"] == 2
    assert values["SSL_LEAPFROG"] == 3
    assert values["SSL_CAP_SECONDS"] == 2400.0
    source = RUNNER.read_text(encoding="utf-8")
    assert "material" not in source.lower()
    assert "posterior sampling" in source


def test_plan_declares_evidence_contract_vetoes_and_nonclaims() -> None:
    text = PLAN.read_text(encoding="utf-8")
    for required in (
        "## Research intent ledger",
        "## Evidence contract",
        "## Default and numeric audit",
        "## Skeptical plan audit",
        "## Pre-mortem",
        "AIS",
        "Swap acceptance",
        "Must not be concluded",
    ):
        assert required in text

