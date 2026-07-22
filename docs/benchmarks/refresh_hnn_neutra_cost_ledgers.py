#!/usr/bin/env python3
"""Refresh derived HNN-NeuTra cost ledgers without rerunning experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    from bayesfilter.testing.hnn_neutra_exact_comparison_tf import cost_ledger

    for requested_root in parse_args().root:
        root = requested_root.resolve()
        result_path = root / "result.json"
        manifest_path = root / "run_manifest.json"
        hashes_path = root / "artifact_hashes.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("schema") != "bayesfilter.hnn_neutra_exact_comparison_result.v1":
            raise ValueError(f"unexpected result schema: {result_path}")

        previous = result["cost_ledger"]
        refreshed = cost_ledger(
            supervision=result["supervision"],
            training=result["training"],
            tuning=result["tuning"],
            runs=result["runs"],
            matched=result["matched_mechanics"],
        )
        result["cost_ledger"] = refreshed
        repairs = list(result.get("derived_artifact_repairs", ()))
        repairs.append(
            {
                "schema": "bayesfilter.hnn_neutra_derived_artifact_repair.v1",
                "repaired_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "derived_cost_ledger_only_no_chain_or_timing_rerun",
                "reason": (
                    "separate HNN preparation break-even from independently "
                    "tuned reuse-campaign break-even"
                ),
                "previous_cost_ledger_sha256": _stable_sha256(previous),
                "refreshed_cost_ledger_sha256": _stable_sha256(refreshed),
                "implementation": (
                    "bayesfilter/testing/hnn_neutra_exact_comparison_tf.py::cost_ledger"
                ),
            }
        )
        result["derived_artifact_repairs"] = repairs
        result_path.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        hashes = {
            "result_sha256": _file_sha256(result_path),
            "run_manifest_sha256": _file_sha256(manifest_path),
        }
        hashes_path.write_text(
            json.dumps(hashes, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "root": str(requested_root),
                    "preparation_break_even_transition_batches": refreshed[
                        "preparation_break_even_transition_batches"
                    ],
                    "reuse_campaign_break_even_transition_batches": refreshed[
                        "reuse_campaign_break_even_transition_batches"
                    ],
                },
                sort_keys=True,
            )
        )
    return 0


def _stable_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
