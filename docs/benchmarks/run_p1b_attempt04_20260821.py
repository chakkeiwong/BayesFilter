"""P1B ladder attempt04: triangular adapted maps + truncation correction.

Plan: docs/plans/bayesfilter-p1b-attempt04-plan-2026-08-21.md.
Pre-gate (run_attempt04_pregate_20260821.py): XLA parity 1.3e-15 PASS;
n=4 r=8 wall 2093s < 2700s stop. Engine: adapted-XLA. Schema v3.
"""
import os, sys
LOG = "/tmp/p1b_attempt04.log"
if __name__ == "__main__" and "--detach" in sys.argv and os.fork() > 0:
    print(f"detached; output -> {LOG}"); sys.exit(0)
if __name__ == "__main__" and "--detach" in sys.argv:
    os.setsid(); fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1); os.dup2(fd, 2)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import datetime as _dt
import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "benchmarks"))

import numpy as np, tensorflow as tf
from run_n4_step_localization_20260819 import case_with_steps
from run_adapted_engine_validation_20260820 import kalman_hint_factory
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
from bayesfilter.highdim.squared_tt_engine_adapted_xla_tf import (
    run_value_filter_branch_axis_adapted_xla,
)

PLAN = "docs/plans/bayesfilter-p1b-attempt04-plan-2026-08-21.md"
PER_STEP_TOLERANCE = 2.5e-3
HORIZON = 8
ROWS = 8192
RANKS = (6, 8, 10)
SEEDS = (42, 142, 242)
STOP_WALL_SECONDS = 45 * 60
KAPPA_C, KAPPA_P = 4.0, 3.0


def main() -> None:
    started = time.time()
    rows_list = []
    stop_reason = None
    for n in (2, 4):
        for r in RANKS:
            for seed in SEEDS:
                adapter, ys, kalman_steps = case_with_steps(n, seed + n)
                hint, observe_t0 = kalman_hint_factory(n, seed + n)
                observe_t0(ys[0].numpy())
                config = EngineConfig(
                    basis_degree=12, rank=r, row_count=ROWS, sweeps=3,
                    ridge=1e-10, tau=1e-6, coordinate_half_width=3.0,
                    seed=91000 + 10 * n + r, row_design="sobol",
                )
                cell_start = time.time()
                try:
                    value, diags = run_value_filter_branch_axis_adapted_xla(
                        adapter, ys, config, predictive_moment_hint=hint,
                        map_kappa_prev=KAPPA_P, map_kappa_current=KAPPA_C,
                    )
                    gap = abs(float(value.numpy()) - sum(kalman_steps))
                    status = "ok"
                    max_ratio = max(
                        d.get("truncation_mass_ratio", 0.0) for d in diags[1:]
                    )
                    if max_ratio > 0.5:
                        status = "flagged:correction_dominates"
                except Exception as error:
                    gap, diags, status, max_ratio = float("nan"), [], f"veto:{error}", None
                wall = time.time() - cell_start
                row = {
                    "n": n, "rank": r, "seed": seed, "gap": gap,
                    "per_step_gap": gap / HORIZON if np.isfinite(gap) else None,
                    "passes_declared": bool(
                        np.isfinite(gap)
                        and gap / HORIZON <= PER_STEP_TOLERANCE
                        and status == "ok"
                    ),
                    "wall_seconds": wall, "status": status,
                    "max_truncation_ratio": max_ratio,
                    "max_fit_rms": (
                        max(d["weighted_fit_rms"] for d in diags) if diags else None
                    ),
                    "min_map_shrink": (
                        min((d.get("map_shrink", 1.0) for d in diags[1:]), default=None)
                        if diags else None
                    ),
                }
                rows_list.append(row)
                print(json.dumps(row), flush=True)
                if wall > STOP_WALL_SECONDS:
                    stop_reason = f"resource bound at n={n} r={r} ({wall:.0f}s)"
                    break
            if stop_reason:
                break
        if stop_reason:
            break

    r_star = {}
    for n in (2, 4):
        passing = [
            r for r in RANKS
            if all(
                row["passes_declared"]
                for row in rows_list
                if row["n"] == n and row["rank"] == r
            )
            and any(row["n"] == n and row["rank"] == r for row in rows_list)
        ]
        r_star[str(n)] = min(passing) if passing else None

    result = {
        "schema_version": "p1b_lgssm_value_ladder.v3",
        "plan": PLAN,
        "engine": "adapted_triangular_xla",
        "map_kappa_current": KAPPA_C, "map_kappa_prev": KAPPA_P,
        "truncation_correction": True,
        "row_design": "sobol", "rows": ROWS, "ranks_run": list(RANKS),
        "per_step_tolerance": PER_STEP_TOLERANCE, "horizon": HORIZON,
        "cells": rows_list, "r_star": r_star, "stop_reason": stop_reason,
        "wall_time_seconds": time.time() - started,
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "tensorflow_version": tf.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "nonclaims": [
            "LGSSM-family fixtures only; exact-hint M2-joint moments",
            "3 seeds: pass/fail vs declared screen only, no ranking language",
            "kappas untuned defaults; tuning v1.1 owns claim-bearing scopes",
        ],
    }
    out = ROOT / "docs/benchmarks/artifacts/p1b_lgssm_value_ladder_20260817/attempt04/result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"r_star": r_star, "stop_reason": stop_reason}), flush=True)


if __name__ == "__main__":
    main()
