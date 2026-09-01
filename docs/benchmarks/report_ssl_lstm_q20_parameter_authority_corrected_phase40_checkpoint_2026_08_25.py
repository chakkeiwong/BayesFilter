"""Select checkpoints from the v2.2 root-group-stratified Phase 40 traces."""

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
EXPECTED_VERSION = "v2.2-root-group-stratified"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v3_root_group_stratified_split"


class Phase40Error(RuntimeError):
    """Raised when a v2.2 boundary receipt is incomplete."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _read(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS_NEUTRA_BOUNDARY_ROLE_LIMITED":
        raise Phase40Error(f"non-passing source arm: {path}")
    if payload.get("plan_version") != EXPECTED_VERSION or payload.get("schema") != EXPECTED_SCHEMA:
        raise Phase40Error(f"source is not the active v2.2 schema: {path}")
    authority = payload.get("authority", {})
    if authority.get("measure") != "theta_R4" or authority.get("parameter_dim") != 4:
        raise Phase40Error(f"wrong target measure: {path}")
    split = payload.get("split", {})
    required = ("root_disjoint", "row_partition_complete", "row_partition_disjoint")
    if split.get("policy") != "root_group_stratified_v1" or not all(split.get(k) is True for k in required):
        raise Phase40Error(f"root-group split contract failed: {path}")
    if split.get("selection_frozen_before_audit") is not True:
        raise Phase40Error(f"audit ordering is not frozen: {path}")
    return payload


def _row(arm_name: str, item: Mapping[str, Any]) -> Mapping[str, Any]:
    training = item.get("training", {})
    audit = item.get("audit_checkpoint")
    required = (
        "validation_loss",
        "validation_latent_mean_max_abs",
        "validation_latent_covariance_max_abs_offdiag",
    )
    if audit is None or any(key not in item for key in required):
        raise Phase40Error(f"incomplete checkpoint for {arm_name} step {training.get('step')}")
    if any(key not in audit for key in ("loss", "latent_mean_max_abs", "latent_covariance_max_abs_offdiag")):
        raise Phase40Error(f"incomplete audit checkpoint for {arm_name} step {training.get('step')}")
    step = int(training["step"])
    validation_loss = float(item["validation_loss"])
    validation_mean = float(item["validation_latent_mean_max_abs"])
    validation_cov = float(item["validation_latent_covariance_max_abs_offdiag"])
    return {
        "arm": arm_name,
        "step": step,
        "validation_loss": validation_loss,
        "validation_latent_mean_max_abs": validation_mean,
        "validation_latent_covariance_max_abs_offdiag": validation_cov,
        "selection_score": validation_loss + validation_mean + validation_cov,
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
            raise Phase40Error("paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase40Error(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    source_paths = {
        "identity": ROOT / args.identity_root / "result.json",
        "affine": ROOT / args.affine_root / "result.json",
    }
    sources = {name: _read(path) for name, path in source_paths.items()}
    rows: list[Mapping[str, Any]] = []
    for precondition, payload in sources.items():
        for arm_name, arm in payload["arms"].items():
            checkpoints = [_row(arm_name, item) for item in arm["training_trace"] if "audit_checkpoint" in item]
            steps = tuple(row["step"] for row in checkpoints)
            if 200 not in steps:
                raise Phase40Error(f"terminal checkpoint missing: {precondition}/{arm_name}")
            selected = min(checkpoints, key=lambda row: (row["selection_score"], row["step"]))
            terminal = next(row for row in checkpoints if row["step"] == 200)
            rows.append(
                {
                    "precondition": precondition,
                    "arm": arm_name,
                    "checkpoint_steps": list(steps),
                    "selected": selected,
                    "terminal": terminal,
                    "audit_moment_sum_delta_selected_minus_terminal": (
                        selected["audit_latent_mean_max_abs"]
                        + selected["audit_latent_covariance_max_abs_offdiag"]
                        - terminal["audit_latent_mean_max_abs"]
                        - terminal["audit_latent_covariance_max_abs_offdiag"]
                    ),
                }
            )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_checkpoint_repair.v2_root_group_stratified",
        "status": "PASS_V2_2_CHECKPOINT_SELECTION_AUDIT_RECEIPT",
        "plan_version": EXPECTED_VERSION,
        "role": "validation_selected_checkpoint_diagnostic_root_group_stratified",
        "selection_rule": "min(validation_loss + validation_latent_mean_max_abs + validation_latent_covariance_max_abs_offdiag); ties choose smallest step",
        "selection_data": "validation_only",
        "audit_data_used_for_selection": False,
        "root_group_split_required": True,
        "rows": rows,
        "nonclaims": [
            "Checkpoint selection is descriptive and does not prove transport quality or IID whitening.",
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
                "identity_trace": _sha(source_paths["identity"]),
                "affine_trace": _sha(source_paths["affine"]),
            },
        },
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="ascii",
    )
    lines = [
        "# v2.2 Root-Group-Stratified Checkpoint Repair",
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
            "Selection used validation rows only; root groups are disjoint across train, validation, and audit.",
            "",
            "No IID Gaussian, posterior, HMC, canonical LEDH, or default-readiness claim is made.",
        ]
    )
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
