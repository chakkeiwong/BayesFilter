"""Diagnostic-only analysis of paired consumer timings and live QR operations."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    arguments = parser.parse_args()
    root = arguments.attempt
    result = json.loads((root / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Incomplete run")
    blocks = {(block["lane"], block["block"]): block["rows"] for block in result["blocks"]}
    if len(blocks) != 16:
        raise ValueError("Missing or duplicate paired blocks")
    rows = []
    for direction in (-1, 0):
        for arm in ("plain", "conditional", "physical"):
            timings = {lane: [] for lane in ("before", "after")}
            for block_index in range(8):
                for lane in timings:
                    matches = [row for row in blocks[(lane, block_index)]
                               if row["arm"] == arm and row["direction"] == direction]
                    if len(matches) != 1:
                        raise ValueError("Missing or duplicate endpoint timing")
                    row = matches[0]
                    if not all(value["passed"] for value in row["comparisons"].values()):
                        raise ValueError("Failed same-source tieout")
                    timings[lane].append(row["synchronized_seconds"])
            means = {lane: statistics.mean(values) for lane, values in timings.items()}
            rows.append(dict(arm=arm, direction=direction, before_seconds=means["before"],
                after_seconds=means["after"], ratio=means["after"]/means["before"],
                paired_difference_seconds=[after-before for before, after in
                                           zip(timings["before"], timings["after"])],
                descriptive_only=True))
    compiled = {}
    for lane in ("before", "after"):
        terminal = result[lane]["terminal"]
        if not terminal["source_unchanged"] or set(terminal["trace_counts"].values()) != {1}:
            raise ValueError("Invalid source or trace count")
        compiled[lane] = {}
        for arm in ("plain", "conditional", "physical"):
            path = root / lane / f"{arm}.optimized-hlo.txt"
            hlo = path.read_text()
            compiled[lane][arm] = dict(live_qr_metadata='op_type="Qr"' in hlo,
                qr_metadata_line_count=sum('op_type="Qr"' in line for line in hlo.splitlines()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    qr_expected = {("before", "plain"): False, ("before", "conditional"): False,
                   ("before", "physical"): True, ("after", "plain"): True,
                   ("after", "conditional"): True, ("after", "physical"): True}
    if any(compiled[lane][arm]["live_qr_metadata"] != expected
           for (lane, arm), expected in qr_expected.items()):
        raise ValueError("Compiled QR pattern differs from the tested explanation")
    serial = [row["serialization_seconds"] for group in blocks.values() for row in group]
    summary = dict(status="complete", timings=rows, optimized_hlo=compiled,
        synchronized_call_count=len(serial), all_same_source_tieouts_passed=True,
        serialization_mean_seconds=statistics.mean(serial), serialization_max_seconds=max(serial),
        compiled_qr_retention_confirmed=True, qr_per_filter_call=5*24*64,
        general_performance_ranking_supported=False,
        remaining_campaign_seconds=15070.08603961207-result["charged_gpu_process_seconds"],
        charged_gpu_process_seconds=result["charged_gpu_process_seconds"])
    (root / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=summary["status"], timings=rows,
                         charged_gpu_process_seconds=summary["charged_gpu_process_seconds"])))


if __name__ == "__main__":
    main()
