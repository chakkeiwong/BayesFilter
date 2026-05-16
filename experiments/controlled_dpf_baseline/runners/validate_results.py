"""Validate clean-room controlled DPF baseline result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.controlled_dpf_baseline.results import read_records, validate_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--require-finite-success-metrics", action="store_true")
    parser.add_argument("--require-smoke-only", action="store_true")
    parser.add_argument("--require-fixed-grid", action="store_true")
    args = parser.parse_args(argv)

    records = read_records(args.records_json)
    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    errors = validate_records(
        records,
        expected_records=args.expected_records,
        require_finite_success_metrics=args.require_finite_success_metrics,
        require_smoke_only=args.require_smoke_only,
        require_fixed_grid=args.require_fixed_grid,
    )
    if summary.get("planned_records") != len(records):
        errors.append("summary planned_records does not match records length")
    if summary.get("expected_records") not in (None, args.expected_records):
        errors.append("summary expected_records does not match validation request")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "validated {count} records from {path}".format(
            count=len(records),
            path=args.records_json,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
