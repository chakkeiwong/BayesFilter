"""Create a reproducible paired summary from a completed Phase 2 result.

This is a reporting-only diagnostic.  It reads preserved raw branch records and
does not alter candidate selection, targets, weights, or score computation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


T_CRITICAL_DF11 = 2.200985


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _interval(values: Sequence[float]) -> Mapping[str, object]:
    count = len(values)
    mean = sum(values) / count if count else None
    if count > 1 and mean is not None:
        centered = [value - mean for value in values]
        sample_sd = math.sqrt(sum(value * value for value in centered) / (count - 1))
        half_width = T_CRITICAL_DF11 * sample_sd / math.sqrt(count)
        lower, upper = mean - half_width, mean + half_width
    else:
        sample_sd = lower = upper = None
    return {
        "n": count,
        "mean": mean,
        "sample_sd": sample_sd,
        "descriptive_95_t_low": lower,
        "descriptive_95_t_high": upper,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "positive_count": sum(value > 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
    }


def summarize(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    by_label: dict[str, dict[int, Mapping[str, object]]] = {}
    for record in records:
        by_label.setdefault(str(record["label"]), {})[int(record["branch_index"])] = record
    reference = by_label.get("ukf_apf_k1", {})
    comparisons: dict[str, object] = {}
    for label, candidate in sorted(by_label.items()):
        if label == "ukf_apf_k1":
            continue
        shared = sorted(set(reference).intersection(candidate))
        ess = [
            float(reference[index]["minimum_ess"])
            - float(candidate[index]["minimum_ess"])
            for index in shared
        ]
        log_likelihood = [
            float(reference[index]["log_likelihood"])
            - float(candidate[index]["log_likelihood"])
            for index in shared
        ]
        comparisons[label] = {
            "branch_indices": shared,
            "ess_k1_minus_comparator": _interval(ess),
            "log_likelihood_k1_minus_comparator": _interval(log_likelihood),
        }
    return {
        "method": "paired_descriptive_student_t_interval",
        "t_critical": T_CRITICAL_DF11,
        "degrees_of_freedom": 11,
        "interpretation": "intervals are descriptive with twelve paired branches; they do not establish superiority or posterior correctness",
        "comparisons": comparisons,
    }


def _render(payload: Mapping[str, object]) -> str:
    summary = payload["summary"]
    lines = [
        "# C2 Phase 2 Paired Summary",
        "",
        f"Source result: `{payload['source_result']}`",
        f"Source SHA-256: `{payload['source_sha256']}`",
        "",
        "The intervals are paired descriptive t intervals (df=11). They are not",
        "evidence of superiority, posterior correctness, or default readiness.",
        "",
        "| Comparator | ESS K1-minus-comparator mean [95%] | log L K1-minus-comparator mean [95%] |",
        "| --- | --- | --- |",
    ]
    for label, comparison in summary["comparisons"].items():
        ess = comparison["ess_k1_minus_comparator"]
        log_likelihood = comparison["log_likelihood_k1_minus_comparator"]

        def fmt(value: object) -> str:
            return "n/a" if value is None else f"{float(value):.6g}"

        lines.append(
            f"| `{label}` | {fmt(ess['mean'])} "
            f"[{fmt(ess['descriptive_95_t_low'])}, {fmt(ess['descriptive_95_t_high'])}] | "
            f"{fmt(log_likelihood['mean'])} "
            f"[{fmt(log_likelihood['descriptive_95_t_low'])}, {fmt(log_likelihood['descriptive_95_t_high'])}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("input result has no raw records")
    summary_payload = {
        "schema_version": "c2_mixture_ukf_apf_phase2_paired_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_result": str(source),
        "source_sha256": _sha256(source),
        "source_schema_version": payload.get("schema_version"),
        "summary": summarize(records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(_render(summary_payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "source_sha256": summary_payload["source_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
