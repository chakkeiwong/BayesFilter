#!/usr/bin/env python3
"""Validate P0 registry artifacts and fail closed on posterior overclaim."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CELL_IDS = {
    "SVX-SGQF",
    "SVX-ZC",
    "KSC-UKF",
    "PP-SGQF",
    "PP-UKF",
    "PP-ZC",
    "STR-UKF",
    "STR-ZC",
    "SIR-SGQF",
    "SIR-UKF",
    "SIR-ZC",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root

    required = {
        "target_registry.json",
        "cell_ledger.json",
        "assumption_ledger.json",
        "command_manifest.json",
        "budget_ledger.json",
        "execution_events.jsonl",
        "run_manifest.json",
        "artifact_hashes.json",
        "source_support.json",
        "citation_venue_metadata.json",
        "backward_snowball.json",
        "forward_snowball.json",
        "claim_support.json",
        "omitted_paper_risks.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise SystemExit(f"missing P0 artifacts: {missing}")

    registry = _load(root / "target_registry.json")
    cells = registry["cells"]
    ids = {row["cell_id"] for row in cells}
    if ids != CELL_IDS or len(cells) != len(CELL_IDS):
        raise SystemExit(f"cell mismatch: {sorted(ids)}")
    for row in cells:
        if row["state"] != "TARGET_BLOCKED":
            raise SystemExit(f"unexpected admitted state: {row['cell_id']} {row['state']}")
        if row["target_signature"] is not None:
            raise SystemExit(f"blocked cell has posterior signature: {row['cell_id']}")
        if len(row["scope_identity"]) != 64:
            raise SystemExit(f"bad scope identity: {row['cell_id']}")
        if not row["blockers"]:
            raise SystemExit(f"blocked cell has no blockers: {row['cell_id']}")

    if next(row for row in cells if row["cell_id"] == "STR-ZC")["route_classification"] != "EXTENSION_OR_INVENTION_BY_DEFINITION":
        raise SystemExit("STR-ZC classification drift")
    for cell_id in ("SVX-ZC", "PP-ZC"):
        classification = next(row for row in cells if row["cell_id"] == cell_id)["route_classification"]
        if "EXTENSION_OR_INVENTION" not in classification:
            raise SystemExit(f"current {cell_id} route overclaims source faithfulness")
    sir_zc = next(row for row in cells if row["cell_id"] == "SIR-ZC")
    if not any(item["code"] == "PARAMETER_INFERENCE_IS_EXTENSION" for item in sir_zc["blockers"]):
        raise SystemExit("SIR-ZC must preserve paper-target extension boundary")

    budget = _load(root / "budget_ledger.json")
    buckets = budget["per_cell_gpu_buckets"]
    if buckets["total"] != 40:
        raise SystemExit("per-cell GPU budget must be 40 hours")
    if sum(value for key, value in buckets.items() if key != "total") != 40:
        raise SystemExit("per-cell GPU bucket arithmetic mismatch")
    if budget["program_ceiling"] != {"cpu_hours": 136, "gpu_hours": 442}:
        raise SystemExit("program budget mismatch")

    commands = _load(root / "command_manifest.json")["commands"]
    if commands[0]["phase"] != "P0" or commands[0]["status"] != "EXECUTED_BY_THIS_BUILDER":
        raise SystemExit("P0 command record mismatch")
    if any(row["command"] is not None for row in commands[1:]):
        raise SystemExit("future command invented before owning implementation")

    source_support = _load(root / "source_support.json")
    if {row["source_id"] for row in source_support["sources"]} != {
        "zhao_cui_jmlr_2024",
        "zhao_cui_author_code",
    }:
        raise SystemExit("source-support ledger mismatch")
    metadata = _load(root / "citation_venue_metadata.json")
    quarantined = [row for row in metadata["records"] if row["status"].startswith("QUARANTINED")]
    if len(quarantined) != 1:
        raise SystemExit("mislabeled OpenAlex cache must remain quarantined")
    if _load(root / "forward_snowball.json")["status"] != "NOT_QUERIED_NETWORK_NOT_NEEDED_FOR_P0_IMPLEMENTATION_GATE":
        raise SystemExit("forward-snowball limitation not explicit")
    if len(_load(root / "claim_support.json")["claims"]) < 5:
        raise SystemExit("claim-support ledger incomplete")
    if len(_load(root / "omitted_paper_risks.json")["records"]) < 3:
        raise SystemExit("omission-risk ledger incomplete")

    hashes = _load(root / "artifact_hashes.json")["artifacts"]
    for name, expected in hashes.items():
        actual = _hash(root / name)
        if actual != expected:
            raise SystemExit(f"artifact hash mismatch: {name}")

    print(json.dumps({
        "passed": True,
        "cell_count": len(cells),
        "target_blocked_count": len(cells),
        "target_signatures_issued": 0,
        "per_cell_gpu_hours": buckets["total"],
        "program_gpu_hours": budget["program_ceiling"]["gpu_hours"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
