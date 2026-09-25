#!/usr/bin/env python3
"""RQMC LEDH Initialization: Phase 2 campaign driver.

Executes the 45 claim-bearing runs (3 models x 5 arms x 3 seeds) specified by
docs/plans/rqmc-ledh-initialization-master-program-2026-09-02.md, in the order
the program requires, and enforces the program's stop conditions.

Ordering constraint (from the program's Continuation Veto section): LGSSM T50 is
the first model tested, and the campaign stops if every RQMC arm is
statistically inferior to the MC baseline on LGSSM. That check can only run
once LGSSM's 15 runs exist, so this driver runs LGSSM first, evaluates the veto,
and only then proceeds to KSC SV T10 and Predator-Prey T20. A plain triple loop
over (model, arm, seed) would not honour that gate.

Infrastructure failure protocol (from the program): up to 2 attempts per run
configuration; a cell that still fails is recorded as an infrastructure failure
and the campaign continues. If more than 9 of the 45 runs (>20%) end as
infrastructure failures, the campaign stops.

A nonzero runner exit that still wrote result.json means the runner's own
promotion veto fired (non-finite value or program_valid False). That is
scientific evidence, not an infrastructure fault, so it is recorded and not
retried.

Usage:
    python docs/benchmarks/drive_rqmc_ledh_phase2.py \\
        --output_root docs/benchmarks/artifacts/rqmc_ledh_init_v1_20260904
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNNER = Path("docs/benchmarks/run_rqmc_ledh_initialization.py")
PLAN = Path("docs/plans/rqmc-ledh-initialization-master-program-2026-09-02.md")

# LGSSM first: the continuation veto is evaluated on it before other models run.
MODEL_ORDER = ("lgssm_T50", "ksc_sv_T10", "predator_prey_T20")
ARMS = ("mc", "sobol_matousek", "sobol_owen", "halton_owen", "genut_guided")
SEEDS = (98301, 98302, 98303)

# Explicit map: --model values are mixed-case, tuning directories are lowercase.
TUNING_ARTIFACTS = {
    "lgssm_T50": Path(
        "docs/benchmarks/artifacts/ledh_trust_region_lgssm_t50_20260903/result.json"
    ),
    "ksc_sv_T10": Path(
        "docs/benchmarks/artifacts/ledh_trust_region_ksc_sv_t10_20260903/result.json"
    ),
    "predator_prey_T20": Path(
        "docs/benchmarks/artifacts/ledh_trust_region_predator_prey_t20_20260903/result.json"
    ),
}

MAX_ATTEMPTS = 2
INFRASTRUCTURE_FAILURE_BUDGET = 9  # >20% of 45
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 42
CONDA_ENV = "tftwogpu"
GPU_INDEX = "1"  # PCI_BUS_ID order: 1 -> RTX 4080 SUPER


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    return env


def _run_one(
    model: str, arm: str, seed: int, output_root: Path, log_dir: Path
) -> dict[str, Any]:
    """Run one cell, retrying up to MAX_ATTEMPTS on infrastructure failure."""
    run_id = f"{model}_{arm}_seed{seed}"
    output_dir = output_root / "runs" / run_id
    tuning = TUNING_ARTIFACTS[model]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        command = [
            "conda", "run", "-n", CONDA_ENV, "python", str(RUNNER),
            "--model", model,
            "--arm", arm,
            "--seed", str(seed),
            "--tuning_artifact", str(tuning),
            "--output", str(output_dir),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, env=_child_env()
        )
        wall = time.perf_counter() - started

        log_path = log_dir / f"{run_id}_attempt{attempt}.log"
        log_path.write_text(
            completed.stdout + "\n===STDERR===\n" + completed.stderr, encoding="utf-8"
        )

        result_path = output_dir / "result.json"
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            finite = bool(payload.get("finite", False))
            program_valid = bool(payload.get("program_valid", False))
            status = "complete" if (completed.returncode == 0 and finite and program_valid) else "veto"
            tag = "ok  " if status == "complete" else "VETO"
            print(
                f"  [{tag}] {run_id:44s} value={payload.get('value', float('nan')):11.4f} "
                f"attempt={attempt} wall={wall:.0f}s"
            )
            return {
                "run_id": run_id, "model": model, "arm": arm, "seed": seed,
                "status": status, "attempts": attempt,
                "value": payload.get("value"),
                "finite": finite, "program_valid": program_valid,
                "fraction_coordinatewise_cap_active": payload.get(
                    "fraction_coordinatewise_cap_active"
                ),
                "driver_wall_seconds": wall,
                "result_path": str(result_path), "log_path": str(log_path),
            }

        print(
            f"  [fail] {run_id:44s} attempt={attempt} rc={completed.returncode} "
            f"(no result.json)"
        )

    return {
        "run_id": run_id, "model": model, "arm": arm, "seed": seed,
        "status": "infrastructure_failure", "attempts": MAX_ATTEMPTS,
        "value": None, "finite": None, "program_valid": None,
        "fraction_coordinatewise_cap_active": None,
        "driver_wall_seconds": None, "result_path": None,
        "log_path": str(log_dir / f"{run_id}_attempt{MAX_ATTEMPTS}.log"),
    }


def _lgssm_continuation_veto(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the program's LGSSM-gated continuation veto.

    Fires only when EVERY RQMC arm is statistically inferior to MC on LGSSM,
    i.e. each arm's paired bootstrap 95% CI for (arm - mc) excludes any
    positive difference (ci_upper < 0). One arm that is merely indistinguishable
    from MC is enough to continue the campaign.
    """
    import numpy as np

    def arm_values(arm: str) -> np.ndarray:
        out = []
        for seed in SEEDS:
            match = [
                r for r in rows
                if r["arm"] == arm and r["seed"] == seed and r["status"] == "complete"
            ]
            out.append(match[0]["value"] if match else float("nan"))
        return np.asarray(out, dtype=float)

    mc = arm_values("mc")
    if np.any(np.isnan(mc)):
        return {
            "fires": False,
            "reason": "mc_baseline_incomplete_cannot_evaluate_veto",
            "per_arm": {},
        }

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    per_arm: dict[str, Any] = {}
    inferior_flags: list[bool] = []
    for arm in ARMS:
        if arm == "mc":
            continue
        values = arm_values(arm)
        if np.any(np.isnan(values)):
            per_arm[arm] = {"status": "incomplete", "statistically_inferior": False}
            inferior_flags.append(False)
            continue
        differences = values - mc
        n = len(differences)
        boot = np.array([
            differences[rng.choice(n, n, replace=True)].mean()
            for _ in range(BOOTSTRAP_SAMPLES)
        ])
        ci_lower = float(np.percentile(boot, 2.5))
        ci_upper = float(np.percentile(boot, 97.5))
        inferior = ci_upper < 0.0
        per_arm[arm] = {
            "status": "evaluated",
            "mean_difference": float(differences.mean()),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "statistically_inferior": inferior,
        }
        inferior_flags.append(inferior)

    fires = bool(inferior_flags) and all(inferior_flags)
    return {
        "fires": fires,
        "reason": (
            "all_rqmc_arms_statistically_inferior_to_mc_on_lgssm" if fires
            else "at_least_one_rqmc_arm_not_statistically_inferior_on_lgssm"
        ),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "per_arm": per_arm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", required=True, type=Path)
    args = parser.parse_args()

    output_root = args.output_root
    log_dir = output_root / "logs"
    (output_root / "runs").mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    for model, path in TUNING_ARTIFACTS.items():
        if not (ROOT / path).exists():
            print(f"FATAL: missing tuning artifact for {model}: {path}")
            return 2

    planned = len(MODEL_ORDER) * len(ARMS) * len(SEEDS)
    print("=" * 78)
    print("RQMC LEDH Phase 2: claim-bearing campaign")
    print("=" * 78)
    print(f"Plan:        {PLAN}")
    print(f"Output root: {output_root}")
    print(f"Models:      {' -> '.join(MODEL_ORDER)}  (LGSSM first, veto-gated)")
    print(f"Arms:        {', '.join(ARMS)}")
    print(f"Seeds:       {', '.join(str(s) for s in SEEDS)}")
    print(f"Total runs:  {planned}")
    print()

    campaign_started = time.perf_counter()
    all_rows: list[dict[str, Any]] = []
    veto_record: dict[str, Any] | None = None
    stopped_early: str | None = None

    for model in MODEL_ORDER:
        print(f"--- {model} " + "-" * max(0, 70 - len(model)))
        model_rows: list[dict[str, Any]] = []
        for arm in ARMS:
            for seed in SEEDS:
                row = _run_one(model, arm, seed, output_root, log_dir)
                model_rows.append(row)
                all_rows.append(row)

        infra = sum(1 for r in all_rows if r["status"] == "infrastructure_failure")
        if infra > INFRASTRUCTURE_FAILURE_BUDGET:
            stopped_early = (
                f"infrastructure_failure_budget_exhausted "
                f"({infra} > {INFRASTRUCTURE_FAILURE_BUDGET})"
            )
            print(f"\nCONTINUATION VETO: {stopped_early}")
            break

        if model == "lgssm_T50":
            print()
            veto_record = _lgssm_continuation_veto(model_rows)
            print(f"LGSSM continuation-veto check: {veto_record['reason']}")
            for arm, detail in veto_record["per_arm"].items():
                if detail.get("status") == "evaluated":
                    print(
                        f"  {arm:18s} mean_diff={detail['mean_difference']:+9.4f} "
                        f"95% CI=[{detail['ci_lower']:+9.4f}, {detail['ci_upper']:+9.4f}] "
                        f"inferior={detail['statistically_inferior']}"
                    )
            if veto_record["fires"]:
                stopped_early = veto_record["reason"]
                print(f"\nCONTINUATION VETO: {stopped_early}")
                break
            print()

    wall = time.perf_counter() - campaign_started
    complete = sum(1 for r in all_rows if r["status"] == "complete")
    vetoed = sum(1 for r in all_rows if r["status"] == "veto")
    infra = sum(1 for r in all_rows if r["status"] == "infrastructure_failure")

    summary = {
        "schema_version": "rqmc_ledh_phase2_campaign.v1",
        "plan": str(PLAN),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "conda_env": CONDA_ENV,
        "cuda_visible_devices": GPU_INDEX,
        "model_order": list(MODEL_ORDER),
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "planned_runs": planned,
        "attempted_runs": len(all_rows),
        "complete_runs": complete,
        "vetoed_runs": vetoed,
        "infrastructure_failures": infra,
        "lgssm_continuation_veto": veto_record,
        "stopped_early": stopped_early,
        "campaign_wall_seconds": wall,
        "rows": all_rows,
    }
    summary_path = output_root / "campaign_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("=" * 78)
    print("Phase 2 campaign finished")
    print("=" * 78)
    print(f"Attempted:               {len(all_rows)} / {planned}")
    print(f"Complete:                {complete}")
    print(f"Vetoed (scientific):     {vetoed}")
    print(f"Infrastructure failures: {infra}")
    print(f"Stopped early:           {stopped_early or 'no'}")
    print(f"Wall time:               {wall / 60:.1f} min")
    print(f"Summary:                 {summary_path}")
    print()

    if stopped_early:
        return 1
    return 0 if len(all_rows) == planned else 1


if __name__ == "__main__":
    sys.exit(main())
