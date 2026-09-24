#!/usr/bin/env python3
"""Rebuild the RQMC Phase 2 campaign summary from the per-run result files.

The original summary was written when 9 of 45 cells had failed (missing Phase 1
tuning artifacts on this branch, plus the sobol_matousek d==1 Lloyd defect).
Those cells were repaired in Phase 2B, so the summary must be regenerated from
the artifacts actually on disk rather than edited by hand.

Preserves the original campaign metadata and the LGSSM continuation-veto record,
and carries each run's `arm_degenerate_with` flag into the row so downstream
analysis cannot count one computation as two agreeing arms.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

MODELS = ("lgssm_T50", "ksc_sv_T10", "predator_prey_T20")
ARMS = ("mc", "sobol_matousek", "sobol_owen", "halton_owen", "genut_guided")
SEEDS = (98301, 98302, 98303)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign_root", required=True, type=Path)
    args = parser.parse_args()

    root = args.campaign_root
    runs = root / "runs"
    original_path = root / "campaign_summary.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for arm in ARMS:
            for seed in SEEDS:
                run_id = f"{model}_{arm}_seed{seed}"
                result_path = runs / run_id / "result.json"
                if not result_path.exists():
                    rows.append({
                        "run_id": run_id, "model": model, "arm": arm, "seed": seed,
                        "status": "missing", "value": None, "finite": None,
                        "program_valid": None, "arm_degenerate_with": None,
                    })
                    continue

                payload = json.loads(result_path.read_text(encoding="utf-8"))
                finite = bool(payload.get("finite", False))
                program_valid = bool(payload.get("program_valid", False))
                rows.append({
                    "run_id": run_id, "model": model, "arm": arm, "seed": seed,
                    "status": "complete" if (finite and program_valid) else "veto",
                    "value": payload.get("value"),
                    "finite": finite,
                    "program_valid": program_valid,
                    "fraction_coordinatewise_cap_active": payload.get(
                        "fraction_coordinatewise_cap_active"
                    ),
                    "arm_degenerate_with": payload.get("arm_degenerate_with"),
                    "arm_degeneracy_reason": payload.get("arm_degeneracy_reason"),
                    "result_path": str((ROOT / result_path).relative_to(ROOT)),
                })

    complete = sum(1 for r in rows if r["status"] == "complete")
    vetoed = sum(1 for r in rows if r["status"] == "veto")
    missing = sum(1 for r in rows if r["status"] == "missing")

    degenerate = sorted({
        (r["model"], r["arm"], r["arm_degenerate_with"])
        for r in rows if r.get("arm_degenerate_with")
    })

    summary = dict(original)
    summary.update({
        "schema_version": "rqmc_ledh_phase2_campaign.v2",
        "rebuilt_from_run_artifacts": True,
        "rebuild_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "original_git_commit": original.get("git_commit"),
        "phase2b_repair_applied": True,
        "planned_runs": len(MODELS) * len(ARMS) * len(SEEDS),
        "attempted_runs": len(rows),
        "complete_runs": complete,
        "vetoed_runs": vetoed,
        "infrastructure_failures": missing,
        "degenerate_arms": [
            {"model": m, "arm": a, "identical_to": d} for m, a, d in degenerate
        ],
        "rows": rows,
    })

    summary_path = root / "campaign_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Rebuilt {summary_path}")
    print(f"  complete={complete}  vetoed={vetoed}  missing={missing}")
    if degenerate:
        print("  degenerate arms (one computation under two names):")
        for m, a, d in degenerate:
            print(f"    {m}: {a} == {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
