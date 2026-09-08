"""Aggregate corrected theta pilot receipts across a particle-size ladder."""

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
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"


class LadderError(RuntimeError):
    """Raised when size-ladder receipts are incompatible."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pilot-root", action="append", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise LadderError("output root must be repository-relative")
    if any(root.is_absolute() or ".." in root.parts for root in args.pilot_root):
        raise LadderError("pilot roots must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise LadderError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    rows: list[Mapping[str, Any]] = []
    target_signature: str | None = None
    protocol_hashes: set[str] = set()
    for root in args.pilot_root:
        pilot_path = ROOT / root / "pilot.json"
        payload = json.loads(pilot_path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS_THETA_MEASURE_PILOT":
            raise LadderError(f"pilot is not passing: {root}")
        for arm_name in ("C0", "M0"):
            arm = payload["arms"].get(arm_name)
            if arm is None or arm.get("protocol", {}).get("measure") != "theta_R4":
                raise LadderError(f"missing or wrong-measure arm: {root} {arm_name}")
            if not all(bool(value) for value in arm["gates"].values()):
                raise LadderError(f"hard gate failure: {root} {arm_name}")
            signature = arm["target_signature"]
            if target_signature is None:
                target_signature = signature
            elif signature != target_signature:
                raise LadderError(f"target signature mismatch: {root} {arm_name}")
            protocol_hash = arm["configuration"]["protocol_hash"]
            protocol_hashes.add(protocol_hash)
            diagnostics = arm["diagnostics"]
            rows.append(
                {
                    "root": root.as_posix(),
                    "particle_count": int(arm["configuration"]["particles"]),
                    "calibration_particles": int(payload["calibration"]["particle_count"]),
                    "arm": arm_name,
                    "seed": arm["configuration"]["seed"],
                    "pilot_sha256": _sha(pilot_path),
                    "protocol_hash": protocol_hash,
                    "target_signature": signature,
                    "terminal_ess_fraction": float(diagnostics["terminal_ess_fraction"]),
                    "weighted_negative_mode_fraction": float(diagnostics["weighted_negative_mode_fraction"]),
                    "log_unnormalized_mass_estimate": float(diagnostics["log_unnormalized_mass_estimate"]),
                    "terminal_negative_root_count": int(diagnostics["terminal_negative_root_count"]),
                    "terminal_positive_root_count": int(diagnostics["terminal_positive_root_count"]),
                    "terminal_unique_root_count": int(diagnostics["stages"][-1]["unique_root_count"]),
                    "wall_seconds": float(payload["run_manifest"]["wall_seconds"]),
                }
            )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_support_ladder.v1",
        "status": "PASS_THETA_SUPPORT_LADDER_HARD_GATES_DESCRIPTIVE",
        "role": "particle_size_support_diagnostic",
        "measure": "theta_R4",
        "target_signature": target_signature,
        "protocol_hashes": sorted(protocol_hashes),
        "rows": rows,
        "nonclaims": [
            "Different particle sizes and stateless seeds are not a replicated superiority comparison.",
            "ESS, mode fractions, masses, and root counts are descriptive support diagnostics.",
            "No SMC-U authority, posterior correctness, IID whitening, mode theorem, LEDH, HMC, or default claim.",
        ],
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
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    lines = [
        "# Corrected Theta Particle-Size Support Ladder",
        "",
        f"Status: `{result['status']}`",
        "",
        "All size roots passed hard theta-measure gates. Values are descriptive across sizes and seeds; no ranking is claimed.",
        "",
        "| N | Arm | ESS fraction | Negative-mode fraction | Mass | Unique roots |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["particle_count"], item["arm"])):
        lines.append(
            f"| {row['particle_count']} | {row['arm']} | {row['terminal_ess_fraction']:.6f} | "
            f"{row['weighted_negative_mode_fraction']:.6f} | {row['log_unnormalized_mass_estimate']:.6f} | "
            f"{row['terminal_unique_root_count']} |"
        )
    lines.extend(
        [
            "",
            "The N=256 M0 root is nominated for a downstream boundary stress test because it has the largest retained root count, not because it is statistically superior.",
            "",
            "No posterior, IID whitening, mode-discovery, LEDH, HMC, or default claim is made.",
        ]
    )
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
