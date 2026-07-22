#!/usr/bin/env python3
"""Verify an exact-SHA review receipt immediately before gated execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


ALLOWED_REVIEWERS = {
    "claude_opus_max_readonly",
    "fresh_codex_readonly_substitute",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(artifact: Path, receipt_path: Path, *, root: Path) -> dict:
    artifact = artifact.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("review receipt must be a JSON object")
    try:
        relative_artifact = str(artifact.relative_to(root.resolve(strict=True)))
    except ValueError as error:
        raise ValueError("reviewed artifact is outside the workspace") from error
    if receipt.get("verdict") != "AGREE":
        raise ValueError("review verdict is not AGREE")
    if receipt.get("reviewed_path") != relative_artifact:
        raise ValueError("review receipt names a different artifact")
    if receipt.get("reviewed_sha256") != _sha256(artifact):
        raise ValueError("review receipt is stale after artifact mutation")
    if receipt.get("reviewer_type") not in ALLOWED_REVIEWERS:
        raise ValueError("reviewer type is not allowed")
    iteration = receipt.get("iteration")
    if not isinstance(iteration, int) or not 1 <= iteration <= 5:
        raise ValueError("review iteration must be an integer from 1 through 5")
    if receipt.get("schema_version") != (
        "bayesfilter.complete_highdim_leaderboard.review_receipt.v1"
    ):
        raise ValueError("review receipt schema is invalid")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    verify(args.artifact, args.receipt, root=args.root)
    print(f"REVIEW_RECEIPT_PASS {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

