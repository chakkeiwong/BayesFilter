"""Regression test for authority/provenance serialization completeness."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _authority_dicts(path: Path) -> list[ast.Dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[ast.Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "value_score_authority" in keys:
            result.append(node)
    return result


def test_authority_payloads_include_score_provenance() -> None:
    paths = (
        ROOT / "bayesfilter/inference/hmc.py",
        ROOT / "bayesfilter/inference/hmc_transition_archive.py",
    )
    payloads = [node for path in paths for node in _authority_dicts(path)]
    assert payloads
    for node in payloads:
        keys = {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert "score_provenance" in keys, f"missing at line {node.lineno}"

