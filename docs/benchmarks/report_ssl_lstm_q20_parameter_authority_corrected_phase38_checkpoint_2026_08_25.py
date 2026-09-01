"""Select predeclared checkpoints from Phase 38 traces without touching audit data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"


class Phase38Error(RuntimeError):
    """Raised when a Phase 38 receipt is incomplete or incompatible."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _read(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise Phase38Error(f"missing receipt: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS_NEUTRA_BOUNDARY_ROLE_LIMITED":
        raise Phase38Error(f"source arm is not passing: {path}")
    if payload.get("plan_version") != "v2.1-training-measure-bound":
        raise Phase38Error(f"wrong plan version: {path}")
    authority = payload.get("authority", {})
    if authority.get("measure") != "theta_R4" or authority.get("parameter_dim") != 4:
        raise Phase38Error(f"wrong target measure: {path}")
    if payload.get("split", {}).get("selection_frozen_before_audit") is not True:
        raise Phase38Error(f"selection/audit ordering is not frozen: {path}")
    return payload


def _checkpoint_row(arm_name: str, item: Mapping[str, Any]) -> Mapping[str, Any]:
    training = item.get("training", {})
    required = (
        "validation_loss",
        "validation_latent_mean_max_abs",
        "validation_latent_covariance_max_abs_offdiag",
        "audit_checkpoint",
    )
    if any(key not in item for key in required):
        raise Phase38Error(f"missing checkpoint fields for {arm_name} step {training.get('step')}")
    audit = item["audit_checkpoint"]
    for key in ("loss", "latent_mean_max_abs", "latent_covariance_max_abs_offdiag"):
        if key not in audit:
            raise Phase38Error(f"missing audit field {key} for {arm_name} step {training.get('step')}")
    step = int(training["step"])
    validation_loss = float(item["validation_loss"])
    validation_mean = float(item["validation_latent_mean_max_abs"])
    validation_cov = float(item["validation_latent_covariance_max_abs_offdiag"])
    score = validation_loss + validation_mean + validation_cov
    return {
        "arm": arm_name,
        "step": step,
        "validation_loss": validation_loss,
        "validation_latent_mean_max_abs": validation_mean,
        "validation_latent_covariance_max_abs_offdiag": validation_cov,
        "selection_score": score,
        "audit_loss": float(audit["loss"]),
        "audit_latent_mean_max_abs": float(audit["latent_mean_max_abs"]),
        "audit_latent_covariance_max_abs_offdiag": float(audit["latent_covariance_max_abs_offdiag"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--identity-root", required=True, type=Path)
    parser.add_argument("--affine-root", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.output_root, args.identity_root, args.affine_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase38Error("paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase38Error(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    sources = {
        "identity": _read(ROOT / args.identity_root / "result.json"),
        "affine": _read(ROOT / args.affine_root / "result.json"),
    }
    rows = []
    for precondition, payload in sources.items():
        for arm_name, arm in payload["arms"].items():
            trace = arm.get("training_trace", [])
            checkpoint_rows = [_checkpoint_row(arm_name, item) for item in trace if "audit_checkpoint" in item]
            if not checkpoint_rows:
                raise Phase38Error(f"no audit checkpoints in {precondition}/{arm_name}")
            checkpoint_steps = tuple(int(item["step"]) for item in checkpoint_rows)
            if 200 not in checkpoint_steps:
                raise Phase38Error(f"terminal step 200 missing in {precondition}/{arm_name}")
            selected = min(checkpoint_rows, key=lambda row: (row["selection_score"], row["step"]))
            terminal = next(row for row in checkpoint_rows if row["step"] == 200)
            rows.append(
                {
                    "precondition": precondition,
                    "arm": arm_name,
                    "checkpoint_steps": list(checkpoint_steps),
                    "selected": selected,
                    "terminal": terminal,
                    "audit_selection_score_delta_selected_minus_terminal": (
                        selected["audit_latent_mean_max_abs"]
                        + selected["audit_latent_covariance_max_abs_offdiag"]
                        - terminal["audit_latent_mean_max_abs"]
                        - terminal["audit_latent_covariance_max_abs_offdiag"]
                    ),
                }
            )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_checkpoint_repair.v1",
        "status": "PASS_CHECKPOINT_SELECTION_AUDIT_RECEIPT",
        "plan_version": "v2.1-training-measure-bound",
        "role": "validation_selected_checkpoint_diagnostic",
        "selection_rule": "min(validation_loss + validation_latent_mean_max_abs + validation_latent_covariance_max_abs_offdiag); ties choose smallest step",
        "selection_data": "validation_only",
        "audit_data_used_for_selection": False,
        "rows": rows,
        "nonclaims": [
            "Checkpoint selection is a finite diagnostic and does not prove transport quality or IID whitening.",
            "The four arm rows are not a replicated superiority comparison.",
            "No posterior correctness, HMC readiness, canonical LEDH status, or default promotion.",
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
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "identity_trace": _sha(ROOT / args.identity_root / "result.json"),
                "affine_trace": _sha(ROOT / args.affine_root / "result.json"),
            },
        },
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    lines = [
        "# Corrected Theta Checkpoint Repair",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Selection rule: `{result['selection_rule']}`",
        "",
        "| Precondition | Arm | selected step | selected validation score | selected audit mean | selected audit covariance | terminal audit mean | terminal audit covariance |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        selected = row["selected"]
        terminal = row["terminal"]
        lines.append(
            f"| {row['precondition']} | {row['arm']} | {selected['step']} | {selected['selection_score']:.6f} | "
            f"{selected['audit_latent_mean_max_abs']:.6f} | {selected['audit_latent_covariance_max_abs_offdiag']:.6f} | "
            f"{terminal['audit_latent_mean_max_abs']:.6f} | {terminal['audit_latent_covariance_max_abs_offdiag']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Selection used validation rows only. Audit rows were evaluated after selection and remain descriptive.",
            "",
            "No IID Gaussian, posterior, HMC, canonical LEDH, or default-readiness claim is made.",
        ]
    )
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
