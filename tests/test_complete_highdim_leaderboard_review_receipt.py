from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import verify_complete_highdim_leaderboard_review_receipt as verifier


def _receipt(root: Path, artifact: Path) -> dict:
    return {
        "schema_version": "bayesfilter.complete_highdim_leaderboard.review_receipt.v1",
        "reviewed_path": artifact.relative_to(root).as_posix(),
        "reviewed_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "reviewer_type": "fresh_codex_readonly_substitute",
        "iteration": 1,
        "verdict": "AGREE",
    }


def test_review_receipt_passes_only_for_exact_current_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "subplan.md"
    receipt = tmp_path / "receipt.json"
    artifact.write_text("reviewed\n", encoding="utf-8")
    receipt.write_text(json.dumps(_receipt(tmp_path, artifact)), encoding="utf-8")

    verifier.verify(artifact, receipt, root=tmp_path)

    artifact.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        verifier.verify(artifact, receipt, root=tmp_path)


def test_review_receipt_rejects_unapproved_reviewer(tmp_path: Path) -> None:
    artifact = tmp_path / "subplan.md"
    receipt = tmp_path / "receipt.json"
    artifact.write_text("reviewed\n", encoding="utf-8")
    payload = _receipt(tmp_path, artifact)
    payload["reviewer_type"] = "self_review"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="reviewer type"):
        verifier.verify(artifact, receipt, root=tmp_path)
