#!/usr/bin/env python3
"""Diagnostic-only exporter for the source d100 NeuTra Gaussian constants.

This script is an independent source-preparation boundary. It may use NumPy to
load the read-only ``dsge_hmc`` benchmark, but its output is data only. No
BayesFilter candidate, training, selection, HMC, or runtime module imports this
script or NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/home/ubuntu/python/dsge_hmc")
SOURCE_FILE = SOURCE_ROOT / "src/dsge_hmc/benchmarks/neutra_paper_targets.py"
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-weighted-forward-kl-paper-d100-fresh-baseline-plan-2026-08-13.md"
)
if str(SOURCE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    if not PLAN.is_file() or not SOURCE_FILE.is_file():
        raise FileNotFoundError("plan or source target file is missing")
    output_root.mkdir(parents=True)

    from dsge_hmc.benchmarks.neutra_paper_targets import (
        make_paper_ill_conditioned_gaussian,
    )

    target = make_paper_ill_conditioned_gaussian(
        100, eig_source="gamma", gamma_shape=0.8
    )
    sigma = np.asarray(target.sigma, dtype=np.float64)
    precision = np.asarray(target.precision, dtype=np.float64)
    cholesky = np.linalg.cholesky(sigma)
    precision_eigenvalues = np.linalg.eigvalsh(precision)
    identity = np.eye(100, dtype=np.float64)
    constants = {
        "name": "paper_ill_cond_gaussian",
        "dimension": 100,
        "mean": np.asarray(target.mu, dtype=np.float64).tolist(),
        "covariance": sigma.tolist(),
        "precision": precision.tolist(),
        "cholesky": cholesky.tolist(),
        "rng": {
            "library": "numpy.random.RandomState",
            "seed": 10,
            "gamma_shape": 0.8,
            "gamma_scale": 1.0,
            "eigenvalue_role": "precision_eigenvalues",
        },
    }
    semantic_hash = _stable_hash(constants)
    payload = {
        "schema": "bayesfilter.neutra.paper_d100_gaussian_source.v1",
        "plan": PLAN.as_posix(),
        "source_path": SOURCE_FILE.as_posix(),
        "source_sha256": _sha256(SOURCE_FILE),
        "source_git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=SOURCE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_factory": (
            "dsge_hmc.benchmarks.neutra_paper_targets."
            "make_paper_ill_conditioned_gaussian"
        ),
        "diagnostic_only_numpy_source_export": True,
        "candidate_runtime_imports_source_or_numpy": False,
        "constants": constants,
        "constants_hash": semantic_hash,
        "numeric_checks": {
            "covariance_symmetry_max_abs": float(np.max(np.abs(sigma - sigma.T))),
            "precision_symmetry_max_abs": float(
                np.max(np.abs(precision - precision.T))
            ),
            "precision_covariance_identity_max_abs": float(
                np.max(np.abs(precision @ sigma - identity))
            ),
            "cholesky_reconstruction_max_abs": float(
                np.max(np.abs(cholesky @ cholesky.T - sigma))
            ),
            "precision_eigenvalue_min": float(precision_eigenvalues[0]),
            "precision_eigenvalue_max": float(precision_eigenvalues[-1]),
            "realized_condition_number": float(
                precision_eigenvalues[-1] / precision_eigenvalues[0]
            ),
        },
    }
    constants_path = output_root / "paper_ill_cond_gaussian_d100_constants.json"
    _write(constants_path, payload)
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.paper_d100_gaussian_source_hashes.v1",
            "artifacts": {constants_path.name: _sha256(constants_path)},
        },
    )
    print(
        json.dumps(
            {
                "output_root": output_root.as_posix(),
                "constants_hash": semantic_hash,
                "condition_number": payload["numeric_checks"][
                    "realized_condition_number"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
