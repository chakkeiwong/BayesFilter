"""Master smoke driver for the observation-aware TT repair.

Step 0 validates the actual-SV SGQF Gaussian-closure guide.  Only after that
gate passes does the driver run the existing actual-SV TT/moment-teacher tests.
This is a bounded diagnostic; it does not establish filtering correctness or
TT statistical superiority.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/observation-aware-tt-repair-mpl")
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[2]


def _json(value):
    if hasattr(value, "numpy"):
        return _json(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step0() -> dict:
    from bayesfilter.highdim.source_sv_sgqf_tf import generate_source_order_sv_dataset_tf
    from bayesfilter.highdim.sv_mixture_cut4 import (
        actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_filter,
        exact_transformed_sv_independent_panel_fixed_sgqf_filter,
    )

    _, observations = generate_source_order_sv_dataset_tf()
    y = observations[:8]
    params = {"gamma": 0.6, "beta": 0.4, "sigma": 1.0}
    raw = actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_filter(
        y, **params, sparse_level=2
    )
    raw_altered = actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_filter(
        tf.concat([y[:1] * 2.0, y[1:]], axis=0),
        **params, sparse_level=2
    )
    base = exact_transformed_sv_independent_panel_fixed_sgqf_filter(
        y, **params, sparse_level=2
    )
    altered = exact_transformed_sv_independent_panel_fixed_sgqf_filter(
        tf.concat([y[:1] * 2.0, y[1:]], axis=0),
        **params, sparse_level=2
    )
    mean_gap = float(tf.reduce_max(tf.abs(base.mean_path - altered.mean_path)).numpy())
    raw_mean_gap = float(tf.reduce_max(tf.abs(raw.mean_path - raw_altered.mean_path)).numpy())
    covariance = base.covariance_path
    finite = bool(tf.reduce_all(tf.math.is_finite(base.mean_path)).numpy())
    finite = finite and bool(tf.reduce_all(tf.math.is_finite(covariance)).numpy())
    positive = bool(tf.reduce_all(covariance > 0.0).numpy())
    responsive = mean_gap > 1.0e-8
    return {
        "status": "PASS" if finite and positive and responsive else "FAIL",
        "finite_moments": finite,
        "positive_diagonal_covariance": positive,
        "observation_mean_max_gap": mean_gap,
        "responsive_to_observation": responsive,
        "raw_closure_observation_mean_max_gap": raw_mean_gap,
        "raw_closure_responsive": raw_mean_gap > 1.0e-8,
        "base_mean_path": base.mean_path,
        "base_variance_path": tf.linalg.diag_part(covariance),
        "diagnostics": dict(base.diagnostics),
        "raw_closure_diagnostics": dict(raw.diagnostics),
        "nonclaims": [
            "not exact SV filtering",
            "not KSC equivalence",
            "not posterior correctness",
            "not TT superiority",
        ],
    }


def _tt_tests(output: Path) -> dict:
    commands = [
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "tests/highdim/test_zhao_cui_moment_teacher_actual_sv.py",
            "-k", "not scalar_trace_does_not_relax_later_multivariate_static_shapes",
        ],
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "tests/highdim/test_zhao_cui_moment_teacher_actual_sv.py::test_scalar_trace_does_not_relax_later_multivariate_static_shapes",
        ],
    ]
    started = time.perf_counter()
    records = []
    for index, command in enumerate(commands):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        environment["TF_NUM_INTRAOP_THREADS"] = "1"
        environment["TF_NUM_INTEROP_THREADS"] = "1"
        proc = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        (output / f"tt_tests_{index}.log").write_text(proc.stdout + proc.stderr)
        records.append({"command": command, "returncode": proc.returncode})
    return {
        "status": "PASS" if all(item["returncode"] == 0 for item in records) else "FAIL",
        "subprocesses": records,
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    step0 = _step0()
    _write(output / "step0_sgqf.json", step0)
    if step0["status"] != "PASS":
        result = {"status": "BLOCK_TT_REPAIR", "step0": step0, "tt_tests": None}
    else:
        tt = _tt_tests(output)
        result = {
            "status": "PASS" if tt["status"] == "PASS" else "BLOCK_TT_REPAIR",
            "step0": step0,
            "tt_tests": tt,
            "scientific_decision": "auxiliary_SGQF_guide_viable_for_TT_diagnostic",
            "nonclaims": [
                "no exact filtering claim",
                "no observation-aware TT accuracy claim",
                "no statistical ranking or default-readiness claim",
            ],
        }
    result["started_utc"] = started
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write(output / "result.json", result)
    _write(
        output / "run_manifest.json",
        {
            "schema": "bayesfilter.observation_aware_tt_repair_master_manifest.v1",
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "python": sys.version,
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "execution_target": "CPU diagnostic; GPU intentionally hidden",
            "script_sha256": _sha256(Path(__file__).resolve()),
            "result_file": str((output / "result.json").relative_to(ROOT)),
            "step0_file": str((output / "step0_sgqf.json").relative_to(ROOT)),
            "tt_test_logs": [
                str(path.relative_to(ROOT))
                for path in sorted(output.glob("tt_tests_*.log"))
            ],
        },
    )
    print(json.dumps(_json(result), indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
