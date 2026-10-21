#!/usr/bin/env python3
"""Independent CPU diagnostic calibration; no sampler or default promotion.

NumPy/SciPy generate exact reference margins and stationary Gaussian-copula
chains. They never supply BayesFilter tuning or runtime inference decisions.
Each alarm replicate, not each correlated draw, is an independent observation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = "docs/plans/bayesfilter-mams-analytic-controls-2026-10-12.md"
TARGETS = ("gaussian", "student_t", "funnel_noncentered", "residual_funnel")
CORRELATIONS = (0., .9, -.9)


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def binomial_interval(count, total):
    from scipy import stats
    return [float(stats.beta.ppf(.025, count, total-count+1)) if count else 0.,
            float(stats.beta.ppf(.975, count+1, total-count)) if count < total else 1.]


def summarize(rows, planned):
    counts = Counter(row["status"] for row in rows)
    interval = binomial_interval(counts["discrepancy_detected"], len(rows)) if rows else None
    return dict(planned=planned, completed=len(rows), unstarted=planned-len(rows),
                counts=dict(counts), discrepancy_frequency_interval=interval,
                observation_unit="independent replicate alarm, not adjacent chain draw",
                calibration_finding=("incomplete" if len(rows) != planned else
                    "excess_discrepancy_rate_detected" if interval[0] > .05 else
                    "no_excess_detected_with_limited_power"),
                coverage_guarantee=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replications", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("independent CPU calibration must hide GPUs before framework import")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("set memory-growth policy before framework import")
    if not 1 <= args.replications <= 32 or not 512 <= args.draws <= 2048:
        raise ValueError("configuration exceeds the bounded diagnostic plan")
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 1200:
        raise ValueError("calibration ceiling must be within the 1200-second suballocation")
    if args.seed < 0:
        raise ValueError("nonnegative reference seed required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    provenance = ROOT/".validation-source.json"
    source = json.loads(provenance.read_text()) if provenance.exists() else dict(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        snapshot_kind="working_tree_with_source_hashes")
    manifest = dict(**source, plan=PLAN, result_file=str(args.output/"result.json"),
        command=[sys.executable, *sys.argv], environment=sys.executable,
        started_utc=datetime.now(timezone.utc).isoformat(), status="starting",
        seed=args.seed, replications=args.replications, draws_per_chain=args.draws, chains=4,
        correlations=CORRELATIONS, targets=TARGETS, max_wall_seconds=args.max_wall_seconds,
        gpu_intentionally_hidden=True, use_xla=True, data_version="N/A: exact synthetic laws",
        cpu_threads={k: os.environ.get(k) for k in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
        role="independent_reference_diagnostic_calibration_only", ranking_supported=False,
        memory_policy="GPUs hidden; TF_FORCE_GPU_ALLOW_GROWTH=true set before import")
    write_json(args.output/"manifest.json", manifest)
    cells = []
    result = dict(status="running", cells=cells, accuracy_established=False,
                  default_readiness=False, tuning_authority=False, ranking_supported=False)
    try:
        import numpy as np
        from scipy import stats
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.testing.inference_validation.mams_analytic_diagnostics import (
            ExactWhitenedFunnelTarget, analytic_quantities, analytic_reference_report)
        from bayesfilter.testing.inference_validation.targets import ValidationTarget
        from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies
        from scripts.validate_mams_adaptation import fixture

        paths = source_dependencies([__file__, ROOT/"scripts/validate_mams_adaptation.py"], repository_root=ROOT)
        manifest.update(framework_version=tf.__version__, tfp_version=tfp.__version__,
            numpy_version=np.__version__, python_version=sys.version,
            source_hashes={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}, status="running")
        for name in paths:
            path = Path(name)
            copy = args.output/"source"/path.relative_to(ROOT)
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(path.read_bytes())
        write_json(args.output/"manifest.json", manifest)

        class WrongChart(ExactWhitenedFunnelTarget):
            def to_model(self, q):
                return ValidationTarget.to_model(self, q)

        for target_index, target in enumerate(TARGETS):
            adapter, _, _, _ = fixture(target, True)
            contract = analytic_quantities(target, adapter, use_xla=True)
            wrong_contract = (analytic_quantities(target, WrongChart(), use_xla=True)
                              if target == "funnel_noncentered" else None)
            for rho_index, rho in enumerate(CORRELATIONS):
                cell = dict(target=target, rho=rho, positive=[], negative=[],
                            quantities=contract.payload(), negative_control=(
                                "truncate_at_1.9" if target == "student_t" else
                                "wrong_model_chart" if wrong_contract else None))
                cells.append(cell)
                for replication in range(args.replications):
                    if time.monotonic()-started >= args.max_wall_seconds:
                        break
                    seed = [args.seed, target_index, rho_index, replication]
                    rng = np.random.default_rng(np.random.SeedSequence(seed))
                    z = rng.normal(size=(args.draws, 4, adapter.parameter_dim))
                    for i in range(1, len(z)):
                        z[i] = rho*z[i-1]+math.sqrt(1-rho*rho)*z[i]
                    if target == "student_t":
                        q = stats.t.ppf(stats.norm.cdf(z), 5)
                    elif target == "residual_funnel":
                        q = z.copy()
                        q[..., 1:] *= np.exp(-.5*np.tanh(z[..., :1]/2))
                    else:
                        q = z
                    report = analytic_reference_report(q, contract, use_xla=True)
                    cell["positive"].append(dict(replication=replication, seed=seed,
                                                 status=report["status"], report=report))
                    if cell["negative_control"]:
                        negative = analytic_reference_report(np.clip(q, -1.9, 1.9), contract, use_xla=True) if target == "student_t" else analytic_reference_report(q, wrong_contract, use_xla=True)
                        cell["negative"].append(dict(replication=replication, seed=seed,
                                                     status=negative["status"], report=negative))
                    cell["positive_summary"] = summarize(cell["positive"], args.replications)
                    if cell["negative_control"]:
                        counts = Counter(row["status"] for row in cell["negative"])
                        cell["negative_summary"] = dict(planned=args.replications,
                            completed=len(cell["negative"]), counts=dict(counts),
                            false_passes=counts["no_discrepancy_detected"],
                            note="inconclusive and discrepancy are distinct; neither is reference agreement")
                    result["elapsed_seconds"] = time.monotonic()-started
                    write_json(args.output/"progress.json", result)
                cell["positive_summary"] = summarize(cell["positive"], args.replications)
        result["status"] = ("completed" if all(len(c["positive"]) == args.replications for c in cells)
                            else "budget_exhausted")
        manifest["status"] = result["status"]
    except BaseException as exc:
        result.update(status="repair_required", error_type=type(exc).__name__, error=str(exc))
        manifest["status"] = "repair_required"
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        result["elapsed_seconds"] = manifest["elapsed_seconds"]
        write_json(args.output/"result.json", result)
        write_json(args.output/"manifest.json", manifest)
    print(json.dumps(dict(status=result["status"], elapsed_seconds=result["elapsed_seconds"], output=str(args.output))))


if __name__ == "__main__":
    main()
