"""Aggregate corrected theta pilot receipts without ranking stochastic arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "docs/benchmarks/aggregate_ssl_lstm_q20_parameter_authority_corrected_phase32_2026_08_25.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"


class AggregateError(RuntimeError):
    """Raised when incompatible pilot receipts are supplied."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise AggregateError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="ascii")


def _summary(values: list[float]) -> Mapping[str, Any]:
    if not values:
        raise AggregateError("empty metric vector")
    mean = sum(values) / len(values)
    if len(values) == 1:
        sd = 0.0
    else:
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return {
        "count": len(values),
        "mean": mean,
        "sample_standard_deviation": sd,
        "mcse_of_mean": sd / math.sqrt(len(values)),
        "values": values,
        "interpretation": "descriptive_only_no_ranking",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pilot-root", action="append", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise AggregateError("output root must be repository-relative")
    for root in args.pilot_root:
        if root.is_absolute() or ".." in root.parts:
            raise AggregateError("pilot roots must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise AggregateError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    pilots = []
    for root in args.pilot_root:
        path = ROOT / root / "pilot.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS_THETA_MEASURE_PILOT":
            raise AggregateError(f"pilot is not passing: {root}")
        pilots.append((root, path, payload))
    reference = pilots[0][2]
    reference_signature = reference["arms"]["M0"]["target_signature"]
    rows = []
    for root, path, payload in pilots:
        for arm in ("C0", "M0"):
            arm_payload = payload["arms"].get(arm)
            if arm_payload is None:
                raise AggregateError(f"missing arm {arm}: {root}")
            if arm_payload["target_signature"] != reference_signature:
                raise AggregateError(f"target signature mismatch: {root} {arm}")
            if arm_payload["protocol"].get("measure") != "theta_R4":
                raise AggregateError(f"measure mismatch: {root} {arm}")
            if not all(bool(value) for value in arm_payload["gates"].values()):
                raise AggregateError(f"hard gate mismatch: {root} {arm}")
            rows.append(
                {
                    "root": root.as_posix(),
                    "pilot_sha256": _sha(path),
                    "arm": arm,
                    "seed": arm_payload["configuration"]["seed"],
                    "protocol_hash": arm_payload["configuration"]["protocol_hash"],
                    "terminal_ess_fraction": float(arm_payload["diagnostics"]["terminal_ess_fraction"]),
                    "weighted_negative_mode_fraction": float(arm_payload["diagnostics"]["weighted_negative_mode_fraction"]),
                    "log_unnormalized_mass_estimate": float(arm_payload["diagnostics"]["log_unnormalized_mass_estimate"]),
                }
            )
    summaries = {}
    for arm in ("C0", "M0"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        summaries[arm] = {
            "terminal_ess_fraction": _summary([row["terminal_ess_fraction"] for row in arm_rows]),
            "weighted_negative_mode_fraction": _summary([row["weighted_negative_mode_fraction"] for row in arm_rows]),
            "log_unnormalized_mass_estimate": _summary([row["log_unnormalized_mass_estimate"] for row in arm_rows]),
        }
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_replication_aggregate.v1",
        "status": "PASS_THETA_REPLICATION_HARD_GATES_DESCRIPTIVE_UNCERTAINTY",
        "role": "fresh_theta_multi_seed_replication_report",
        "target_signature": reference_signature,
        "measure": "theta_R4",
        "seed_count": len(pilots),
        "rows": rows,
        "summaries": summaries,
        "run_manifest": {
            "program": PLAN.as_posix(),
            "runner": RUNNER.as_posix(),
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER)},
        },
        "nonclaims": [
            "Three seeds provide descriptive uncertainty only; no arm ranking is supported.",
            "M0 remains a candidate and not an SMC-U authority or posterior estimate.",
            "No IID whitening, mode-discovery, LEDH, HMC, or default claim.",
        ],
    }
    _write_json(output / "result.json", result)
    lines = [
        "# Corrected Theta Replication Aggregate",
        "",
        f"Status: `{result['status']}`",
        "",
        "All supplied seeds passed hard theta-measure gates. Means, SDs, and MCSEs below are descriptive only.",
        "",
    ]
    for arm in ("C0", "M0"):
        lines.append(f"## {arm}")
        for metric, summary in summaries[arm].items():
            lines.append(
                f"- {metric}: mean={summary['mean']:.8g}, sd={summary['sample_standard_deviation']:.8g}, mcse={summary['mcse_of_mean']:.8g}"
            )
        lines.append("")
    lines.append("No stochastic ranking or authority/posterior claim is made.")
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
