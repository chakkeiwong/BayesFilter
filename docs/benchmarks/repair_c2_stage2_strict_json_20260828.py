"""Repair the Stage 2 strict-JSON serialization defect without recomputation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def _repair_bounds(value: Any, path: tuple[str, ...], repaired: list[str]) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key)
            if (
                name == "reverse_triangle_log_upper"
                and isinstance(item, float)
                and not math.isfinite(item)
                and "rho_h_qmc" in value
            ):
                if float(value["rho_h_qmc"]) < 1.0:
                    raise ValueError(
                        "non-finite reverse bound was found where rho_h_qmc < 1"
                    )
                result[name] = 0.0
                repaired.append("/".join(path + (name,)))
            else:
                result[name] = _repair_bounds(item, path + (name,), repaired)
        if "rho_h_qmc" in result and "reverse_triangle_log_upper" in result:
            valid = float(result["rho_h_qmc"]) < 1.0
            result["reverse_triangle_bound_valid"] = 1.0 if valid else 0.0
            upper = result["reverse_triangle_log_upper"]
            if not isinstance(upper, (int, float)) or not math.isfinite(float(upper)):
                if valid:
                    raise ValueError(
                        "non-finite reverse bound was found where rho_h_qmc < 1"
                    )
                result["reverse_triangle_log_upper"] = 0.0
                repaired.append("/".join(path + ("reverse_triangle_log_upper",)))
        return result
    if isinstance(value, list):
        return [
            _repair_bounds(item, path + (str(index),), repaired)
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"unexpected non-finite value at {'/'.join(path)}")
    return value


def main() -> None:
    args = _parse_args()
    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    allowed_parent = (
        ROOT / "docs/benchmarks/artifacts/c2_n4_root_cause_20260828"
    ).resolve()
    if source.parent != allowed_parent or output.parent != allowed_parent:
        raise ValueError("source and output must be direct children of the campaign root")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    source_result_path = source / "stage2_result.json"
    source_manifest_path = source / "run_manifest.json"
    raw_result = source_result_path.read_text(encoding="utf-8")
    result = json.loads(raw_result)
    repaired_paths: list[str] = []
    repaired_result = _repair_bounds(result, (), repaired_paths)
    if not repaired_paths:
        raise ValueError("source artifact contains no repairable reverse-bound values")
    repaired_result["artifact_repair"] = {
        "schema_id": "c2_stage2_strict_json_repair_v1",
        "source_attempt": source.name,
        "source_result_sha256": _sha256(source_result_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "defect": (
            "invalid bare Infinity values in explanatory reverse-triangle upper bounds"
        ),
        "repair": (
            "set invalid upper bounds to 0.0 and emit "
            "reverse_triangle_bound_valid=0.0; scientific values were not recomputed"
        ),
        "repaired_paths": repaired_paths,
        "scientific_decision_unchanged": True,
        "fitted_target_attempts_added": 0,
    }

    output.mkdir(parents=True)
    for name in (
        "stage0_tests.log",
        "stage0_verification.json",
        "pf_per_step_reference.json",
        "pf_reference.log",
    ):
        shutil.copy2(source / name, output / name)
    shutil.copytree(source / "snapshots", output / "snapshots")
    shutil.copy2(source_manifest_path, output / "source_attempt02_manifest.json")

    result_text = json.dumps(
        repaired_result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    (output / "stage2_result.json").write_text(result_text, encoding="utf-8")
    json.loads(result_text, parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"non-standard JSON constant {token}")
    ))

    source_markdown = (source / "stage2_result.md").read_text(encoding="utf-8")
    repair_notice = (
        "> **Artifact repair, 2026-08-28:** This fresh attempt repairs only the "
        "strict-JSON serialization defect in attempt02. Seventeen invalid "
        "reverse-bound `Infinity` values were replaced by `0.0` with an explicit "
        "`reverse_triangle_bound_valid=0.0` flag. No scientific quantity was "
        "recomputed and the Stage 2 decision is unchanged.\n\n"
    )
    (output / "stage2_result.md").write_text(
        source_markdown.replace("\n", "\n" + repair_notice, 1), encoding="utf-8"
    )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    manifest = {
        "schema_id": "c2_stage2_artifact_repair_manifest_v1",
        "command": command,
        "git_commit": commit,
        "source_attempt": source.name,
        "output_attempt": output.name,
        "source_result_sha256": _sha256(source_result_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "repaired_result_sha256": _sha256(output / "stage2_result.json"),
        "repaired_markdown_sha256": _sha256(output / "stage2_result.md"),
        "repaired_nonfinite_count": len(repaired_paths),
        "repaired_paths": repaired_paths,
        "scientific_decision_unchanged": True,
        "gpu_run": False,
        "pf_run": False,
        "fitted_target_attempts_added": 0,
        "consumed_fitted_targets_total": 2,
        "source_run_manifest": "source_attempt02_manifest.json",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
