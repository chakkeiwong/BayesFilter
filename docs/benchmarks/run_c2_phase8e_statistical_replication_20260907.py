"""Launch the bounded primary C2 Phase 8E replication campaign.

The launcher is an orchestration/reporting lane.  It does not implement the
filter and does not select a schedule.  It creates a fresh output root,
generates the predeclared fixture set, runs the frozen five-family driver for
each fixture, and invokes the strict fixture-cluster analyzer.  A candidate
failure is retained in the per-fixture result and the campaign still writes
the aggregate analysis so the failure is inspectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "docs/benchmarks/generate_c2_fresh_fixture_phase8c_20260905.py"
RUNNER = ROOT / "docs/benchmarks/run_c2_exact_likelihood_laplace_phase8b_20260904.py"
ANALYZER = ROOT / "docs/benchmarks/analyze_c2_phase8e_statistical_replication_20260907.py"
PLAN = ROOT / "docs/plans/c2-phase8e-statistical-replication-20260907.md"
BOUNDARIES = ROOT / "docs/benchmarks/fixtures/c2_phase8e_observation_bin_boundaries_v1.json"
PHASE_ID = "c2_exact_likelihood_laplace_phase8e_statistical_replication_v1"
REPLICATION_SET_ID = "c2_phase8e_primary_n4096_v1"
DEFAULT_ANALYSIS_SEED = 20260907
DEFAULT_MODEL_SEED = 20260912
DEFAULT_OBSERVATION_SEEDS = tuple(range(424300, 424312))
DEFAULT_PARTICLE_COUNT = 4096
DEFAULT_BRANCH_COUNT = 2
MAX_FIXTURES = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run(command: Sequence[str], *, log_path: Path, environment: Mapping[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            tuple(str(value) for value in command),
            cwd=ROOT,
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--particle-count", type=int, default=DEFAULT_PARTICLE_COUNT)
    parser.add_argument("--branch-count", type=int, default=DEFAULT_BRANCH_COUNT)
    parser.add_argument("--state-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--analysis-seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    parser.add_argument(
        "--observation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_OBSERVATION_SEEDS),
    )
    return parser.parse_args()


def _validate_inputs(args: argparse.Namespace) -> tuple[Path, tuple[int, ...]]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    observation_seeds = tuple(int(value) for value in args.observation_seeds)
    if len(observation_seeds) < 1 or len(observation_seeds) > MAX_FIXTURES:
        raise ValueError(f"observation fixture count must be in [1, {MAX_FIXTURES}]")
    if len(set(observation_seeds)) != len(observation_seeds):
        raise ValueError("observation seeds must be unique")
    if int(args.particle_count) != DEFAULT_PARTICLE_COUNT:
        raise ValueError("this launcher is frozen to the Phase 8E N=4096 primary arm")
    if int(args.branch_count) != DEFAULT_BRANCH_COUNT:
        raise ValueError("this launcher is frozen to two paired proposal branches")
    if int(args.state_seed) != DEFAULT_MODEL_SEED:
        raise ValueError("this launcher is frozen to the reviewed model/state seed")
    if observation_seeds != DEFAULT_OBSERVATION_SEEDS:
        raise ValueError(
            "the primary launcher requires the complete reviewed observation seed set"
        )
    if int(args.analysis_seed) != DEFAULT_ANALYSIS_SEED:
        raise ValueError("this launcher is frozen to the reviewed analysis seed")
    for path in (GENERATOR, RUNNER, ANALYZER, PLAN, BOUNDARIES):
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase 8E input {path}")
    if _git("status", "--porcelain=v1"):
        raise RuntimeError("Phase 8E requires a clean integration worktree")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite campaign root {output_root}")
    return output_root, observation_seeds


def run(args: argparse.Namespace) -> tuple[Path, int]:
    output_root, observation_seeds = _validate_inputs(args)
    output_root.mkdir(parents=True)
    fixture_root = output_root / "fixtures"
    run_root = output_root / "runs"
    log_root = output_root / "logs"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    environment.setdefault("MPLCONFIGDIR", str(output_root / "matplotlib"))
    generation_environment = dict(environment)
    generation_environment["CUDA_VISIBLE_DEVICES"] = "-1"
    commit = _git("rev-parse", "HEAD")
    launch_manifest: dict[str, Any] = {
        "schema_version": "c2_phase8e_statistical_replication_campaign_v1",
        "phase": PHASE_ID,
        "replication_set_id": REPLICATION_SET_ID,
        "status": "RUNNING",
        "started_unix_seconds": time.time(),
        "command": [sys.executable, *sys.argv],
        "workspace": {
            "git_commit": commit,
            "git_branch": _git("branch", "--show-current"),
            "git_status_before": "clean",
        },
        "plan": {"path": str(PLAN), "sha256": _sha256(PLAN)},
        "sources": {
            "launcher": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            "generator": {"path": str(GENERATOR), "sha256": _sha256(GENERATOR)},
            "runner": {"path": str(RUNNER), "sha256": _sha256(RUNNER)},
            "analyzer": {"path": str(ANALYZER), "sha256": _sha256(ANALYZER)},
            "boundaries": {"path": str(BOUNDARIES), "sha256": _sha256(BOUNDARIES)},
        },
        "design": {
            "particle_count": int(args.particle_count),
            "branch_count": int(args.branch_count),
            "model_seed": int(args.state_seed),
            "analysis_seed": int(args.analysis_seed),
            "observation_seeds": list(observation_seeds),
            "schedule_id": "quarter_long_12",
            "schedule_ladder_id": "baseline_plus_reviewed_repair_hypotheses",
        },
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
            "tf_force_gpu_allow_growth": environment["TF_FORCE_GPU_ALLOW_GROWTH"],
        },
        "fixtures": [],
        "runs": [],
    }
    launch_path = output_root / "launch_manifest.json"
    _write_once(launch_path, launch_manifest)

    run_failures = 0
    run_paths: list[Path] = []
    for observation_seed in observation_seeds:
        fixture_path = fixture_root / f"fixture_state{int(args.state_seed)}_obs{observation_seed}.json"
        generation_command = (
            sys.executable,
            str(GENERATOR.relative_to(ROOT)),
            "--output",
            str(fixture_path),
            "--state-seed",
            str(args.state_seed),
            "--observation-seed",
            str(observation_seed),
        )
        generation_log = log_root / f"generate_obs{observation_seed}.log"
        generation_code = _run(
            generation_command,
            log_path=generation_log,
            environment=generation_environment,
        )
        fixture_entry: dict[str, Any] = {
            "observation_seed": observation_seed,
            "command": list(generation_command),
            "log": str(generation_log),
            "returncode": generation_code,
        }
        if generation_code == 0 and fixture_path.is_file():
            fixture_entry["path"] = str(fixture_path)
            fixture_entry["sha256"] = _sha256(fixture_path)
            run_path = run_root / f"obs{observation_seed}"
            run_command = (
                sys.executable,
                str(RUNNER.relative_to(ROOT)),
                "--output-root",
                str(run_path),
                "--fixture-path",
                str(fixture_path),
                "--plan-path",
                str(PLAN),
                "--phase-id",
                PHASE_ID,
                "--replication-set-id",
                REPLICATION_SET_ID,
                "--analysis-seed",
                str(args.analysis_seed),
                "--exclude-gaussian-hint",
                "--include-repair-schedules",
                "--fixed-schedule-config",
                "quarter_long_12",
                "--particle-count",
                str(args.particle_count),
                "--branch-count",
                str(args.branch_count),
            )
            run_log = log_root / f"run_obs{observation_seed}.log"
            run_code = _run(run_command, log_path=run_log, environment=environment)
            run_entry = {
                "observation_seed": observation_seed,
                "path": str(run_path),
                "command": list(run_command),
                "log": str(run_log),
                "returncode": run_code,
                "result_present": (run_path / "result.json").is_file(),
            }
            launch_manifest["runs"].append(run_entry)
            run_paths.append(run_path)
            if run_code != 0:
                run_failures += 1
        else:
            run_failures += 1
        launch_manifest["fixtures"].append(fixture_entry)

    analysis_path = output_root / "phase8e_analysis.json"
    analysis_command = (
        sys.executable,
        str(ANALYZER.relative_to(ROOT)),
        "--output",
        str(analysis_path),
        "--expected-analysis-seed",
        str(args.analysis_seed),
        "--observation-bin-boundaries",
        str(BOUNDARIES),
        *tuple(str(path) for path in run_paths),
    )
    analysis_log = log_root / "analyze_phase8e.log"
    analysis_code = _run(analysis_command, log_path=analysis_log, environment=environment)
    analysis_status: str | None = None
    if analysis_path.is_file():
        try:
            analysis_status = str(json.loads(analysis_path.read_text(encoding="utf-8"))["status"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            analysis_status = "UNREADABLE_ANALYSIS"
    launch_manifest["analysis"] = {
        "path": str(analysis_path),
        "command": list(analysis_command),
        "log": str(analysis_log),
        "returncode": analysis_code,
        "status": analysis_status,
    }
    if run_failures or analysis_code != 0 or analysis_status in {
        None,
        "UNREADABLE_ANALYSIS",
        "BLOCKED_ARTIFACT_PARSE",
        "VETO_PHASE8E_VALIDITY",
    }:
        launch_manifest["status"] = "COMPLETED_WITH_SUBPROCESS_FAILURES"
    elif analysis_status == "PASS_VALIDITY_WITH_PROMOTION_VETO":
        launch_manifest["status"] = "COMPLETED_PROMOTION_VETO"
    elif analysis_status == "NOMINATED_STATISTICAL_CONTRAST_DIAGNOSTIC":
        launch_manifest["status"] = "COMPLETED_STATISTICAL_NOMINATION"
    else:
        launch_manifest["status"] = "COMPLETED_DESCRIPTIVE_ONLY"
    launch_manifest["finished_unix_seconds"] = time.time()
    launch_manifest["wall_seconds"] = (
        launch_manifest["finished_unix_seconds"]
        - launch_manifest["started_unix_seconds"]
    )
    launch_manifest["subprocess_failure_count"] = run_failures + int(analysis_code != 0)
    launch_path.unlink()
    _write_once(output_root / "campaign_manifest.json", launch_manifest)
    print(
        json.dumps(
            {
                "analysis_status": analysis_status,
                "campaign_manifest": str(output_root / "campaign_manifest.json"),
                "subprocess_failure_count": launch_manifest["subprocess_failure_count"],
            },
            sort_keys=True,
        )
    )
    hard_failure = bool(
        run_failures
        or analysis_code != 0
        or analysis_status in {
            None,
            "UNREADABLE_ANALYSIS",
            "BLOCKED_ARTIFACT_PARSE",
            "VETO_PHASE8E_VALIDITY",
        }
    )
    return output_root, int(hard_failure)


def main() -> int:
    args = _parse_args()
    _, return_code = run(args)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
