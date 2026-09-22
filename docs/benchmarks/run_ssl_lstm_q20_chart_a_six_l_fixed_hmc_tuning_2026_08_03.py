#!/usr/bin/env python3
"""Run the Chart A fixed-HMC leapfrog grid in CPU/XLA process shards."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE_HARNESS = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_fixed_hmc_api_cpu_xla_validation_2026_08_02.py"
)
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-plan-2026-08-03.md"
)
SCHEMA = "bayesfilter.ssl_lstm.q20_chart_a_six_l_fixed_hmc_tuning.v1"
CANONICAL_GRID = (5, 10, 15, 20, 25, 3)
TUNE_SEED_BASE = (20260625, 100)
SCREEN_SEED_BASE = (20260625, 200)
VERIFICATION_SEED_BASE = (20260625, 300)
SUPERVISOR_CPU = 127
ASSIGNMENTS = (
    {"candidate_index": 0, "leapfrog": 5, "cpus": tuple(range(0, 6))},
    {"candidate_index": 1, "leapfrog": 10, "cpus": tuple(range(6, 12))},
    {"candidate_index": 2, "leapfrog": 15, "cpus": tuple(range(64, 72))},
    {"candidate_index": 3, "leapfrog": 20, "cpus": tuple(range(12, 28))},
    {"candidate_index": 4, "leapfrog": 25, "cpus": tuple(range(72, 88))},
    {"candidate_index": 5, "leapfrog": 3, "cpus": tuple(range(88, 94))},
)


class CampaignError(RuntimeError):
    pass


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CampaignError(f"artifact already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(payload))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_base_harness() -> Any:
    spec = importlib.util.spec_from_file_location("q20_fixed_hmc_base_harness", BASE_HARNESS)
    if spec is None or spec.loader is None:
        raise CampaignError("could not load the q=20 fixed-HMC base harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assignment(leapfrog: int) -> Mapping[str, Any]:
    for row in ASSIGNMENTS:
        if int(row["leapfrog"]) == int(leapfrog):
            return row
    raise CampaignError(f"leapfrog value is outside the reviewed grid: {leapfrog}")


def _shifted_seed(seed: tuple[int, int], offset: int) -> tuple[int, int]:
    return seed[0], seed[1] + int(offset)


def _config_for_worker(*, leapfrog: int, candidate_index: int) -> Any:
    from bayesfilter.inference.fixed_transport_hmc_tuning import (
        FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        FixedTransportHMCKernelTuningConfig,
    )

    return FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.5,
        leapfrog_grid=(int(leapfrog),),
        chain_count=4,
        target_accept_prob=0.70,
        acceptance_band=(0.65, 0.75),
        repair_band=(0.55, 0.85),
        selection_policy="acceptance_target_distance",
        selection_replications=1,
        budget_schedule=(8, 16, 32),
        tune_num_results=8,
        screen_num_results=16,
        screen_num_burnin_steps=4,
        verification_num_results=64,
        verification_num_burnin_steps=16,
        tune_seed_base=_shifted_seed(TUNE_SEED_BASE, candidate_index * 100),
        screen_seed_base=_shifted_seed(SCREEN_SEED_BASE, candidate_index * 100),
        verification_seed_base=_shifted_seed(
            VERIFICATION_SEED_BASE, candidate_index
        ),
        target_status_trace_policy="per_chain_step",
        target_scope=(
            "ssl_lstm_neutra_state_complexity_batch_native:q20:"
            "fixed_hmc_api:chart-a:claim_tuning_grid6"
        ),
        use_xla=True,
        tuning_policy=FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY,
        output_filename="tuning-result.json",
    )


def _validate_static_contract() -> None:
    if tuple(int(row["leapfrog"]) for row in ASSIGNMENTS) != CANONICAL_GRID:
        raise CampaignError("assignment order differs from canonical grid")
    all_cpus = tuple(cpu for row in ASSIGNMENTS for cpu in row["cpus"])
    if len(all_cpus) != 58 or len(set(all_cpus)) != 58:
        raise CampaignError("worker allocation must contain 58 unique CPUs")
    if SUPERVISOR_CPU in all_cpus:
        raise CampaignError("supervisor CPU overlaps a worker")
    expected_cores = {3: 6, 5: 6, 10: 6, 15: 8, 20: 16, 25: 16}
    if {int(row["leapfrog"]): len(row["cpus"]) for row in ASSIGNMENTS} != expected_cores:
        raise CampaignError("worker core counts differ from reviewed allocation")
    for index, leapfrog in enumerate(CANONICAL_GRID):
        config = _config_for_worker(leapfrog=leapfrog, candidate_index=index)
        if config.tune_seed_base != _shifted_seed(TUNE_SEED_BASE, index * 100):
            raise CampaignError("tune seed sharding mismatch")
        if config.screen_seed_base != _shifted_seed(SCREEN_SEED_BASE, index * 100):
            raise CampaignError("screen seed sharding mismatch")
        if config.verification_seed_base != _shifted_seed(
            VERIFICATION_SEED_BASE, index
        ):
            raise CampaignError("verification seed sharding mismatch")


def _worker_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": str(int(threads)),
            "TF_NUM_INTEROP_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _run_worker(args: argparse.Namespace) -> int:
    assignment = _assignment(args.leapfrog)
    candidate_index = int(assignment["candidate_index"])
    cpus = tuple(int(cpu) for cpu in assignment["cpus"])
    if args.candidate_index != candidate_index or args.threads != len(cpus):
        raise CampaignError("worker CLI differs from reviewed assignment")
    if tuple(sorted(os.sched_getaffinity(0))) != cpus:
        raise CampaignError("worker affinity differs from reviewed assignment")

    base_harness = _load_base_harness()
    tf = base_harness._configure_tensorflow(args.threads)
    from bayesfilter.inference.fixed_transport_hmc_tuning import (
        tune_fixed_transport_hmc_kernel,
    )

    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("worker output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    base, transport, provenance = base_harness._build_chart(
        "chart-a", threads=args.threads
    )
    config = _config_for_worker(
        leapfrog=args.leapfrog, candidate_index=candidate_index
    )
    result = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=transport,
        initial_position=tf.zeros((4,), tf.float64),
        config=config,
        output_dir=output,
    )
    artifact = Path(result.artifact_path) if result.artifact_path else None
    summary = {
        "schema": SCHEMA,
        "role": "candidate_worker",
        "status": "CANDIDATE_PASSED_TUNER" if result.passed else "NO_VIABLE_CANDIDATE",
        "chart": "chart-a",
        "canonical_candidate_index": candidate_index,
        "leapfrog_steps": int(args.leapfrog),
        "started_utc": started_utc,
        "wall_seconds": time.perf_counter() - started,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "configured_intra_op_threads": int(args.threads),
        "configured_inter_op_threads": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "jit_compile": True,
        "dtype": "float64",
        "runtime_numerical_backend": "tensorflow_tfp_only",
        "chart_provenance": provenance,
        "shifted_seed_bases": {
            "tune": config.tune_seed_base,
            "screen": config.screen_seed_base,
            "verification": config.verification_seed_base,
        },
        "candidate_passed_tuner": result.passed,
        "selected_step_size": (
            None if result.selected_candidate is None else result.selected_candidate.selected_step_size
        ),
        "selected_acceptance_rate": (
            None
            if result.selected_candidate is None
            else result.selected_candidate.selected_acceptance_rate
        ),
        "hard_vetoes": result.hard_vetoes,
        "repair_triggers": result.repair_triggers,
        "tuning_artifact_path": result.artifact_path,
        "tuning_artifact_sha256": None if artifact is None else _sha256(artifact),
        "ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "plan": PLAN.as_posix(),
        "source_hashes": {
            "supervisor": _sha256(SCRIPT),
            "base_harness": _sha256(BASE_HARNESS),
            "tensorflow_tuner": _sha256(
                ROOT / "bayesfilter/inference/fixed_transport_hmc_tuning_tf.py"
            ),
        },
        "nonclaims": [
            "Chart A fixed-HMC kernel candidate tuning only",
            "no sequential HMC, convergence, posterior, chart B, or default claim",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "leapfrog_steps": args.leapfrog,
                "wall_seconds": summary["wall_seconds"],
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


def _canonical_config(worker_payload: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(worker_payload["config"])
    config["leapfrog_grid"] = list(CANONICAL_GRID)
    config["tune_seed_base"] = list(TUNE_SEED_BASE)
    config["screen_seed_base"] = list(SCREEN_SEED_BASE)
    config["verification_seed_base"] = list(VERIFICATION_SEED_BASE)
    return config


def _deduplicated(values: Sequence[Sequence[str]]) -> list[str]:
    result: list[str] = []
    for group in values:
        for value in group:
            if value not in result:
                result.append(value)
    return result


def _merge_results(
    output: Path, worker_summaries: Mapping[int, Mapping[str, Any]]
) -> Mapping[str, Any]:
    payloads = {}
    for row in ASSIGNMENTS:
        leapfrog = int(row["leapfrog"])
        summary = worker_summaries[leapfrog]
        path = Path(str(summary["tuning_artifact_path"]))
        payloads[leapfrog] = _read_json(path)

    first = payloads[CANONICAL_GRID[0]]
    identity_fields = (
        "transformed_adapter_signature",
        "base_adapter_signature",
        "fixed_transport_manifest_hash",
        "target_dimension",
        "identity_z_mass_artifact_payload",
        "identity_z_mass_artifact_signature",
        "diagnostic_roles",
        "nonclaims",
    )
    for payload in payloads.values():
        for field in identity_fields:
            if payload.get(field) != first.get(field):
                raise CampaignError(f"worker result mismatch for {field}")

    candidates = []
    for index, leapfrog in enumerate(CANONICAL_GRID):
        payload = payloads[leapfrog]
        if len(payload.get("candidates", [])) != 1:
            raise CampaignError(f"L={leapfrog} did not emit exactly one candidate")
        candidate = dict(payload["candidates"][0])
        candidate["candidate_index"] = index
        if int(candidate["num_leapfrog_steps"]) != leapfrog:
            raise CampaignError("candidate leapfrog identity mismatch")
        candidates.append(candidate)

    viable = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if candidate.get("passed") is True
    ]
    selected_index = None
    if viable:
        selected_index = min(
            viable,
            key=lambda item: (
                abs(float(item[1]["verification_diagnostics"]["acceptance_rate"]) - 0.70),
                int(item[1]["num_leapfrog_steps"]),
                float(item[1]["selected_step_size"]),
                item[0],
            ),
        )[0]
    selected_leapfrog = (
        None if selected_index is None else int(candidates[selected_index]["num_leapfrog_steps"])
    )
    final_kernel = (
        None
        if selected_leapfrog is None
        else payloads[selected_leapfrog]["final_kernel_payload"]
    )
    merged_path = output / "merged-tuning-result.json"
    merged = {
        "schema": first["schema"],
        "config": _canonical_config(first),
        **{field: first[field] for field in identity_fields if field != "nonclaims"},
        "candidates": candidates,
        "selected_candidate_index": selected_index,
        "final_status": "passed" if selected_index is not None else "no_viable_candidate",
        "final_kernel_payload": final_kernel,
        "final_kernel_hash": None if final_kernel is None else _stable_hash(final_kernel),
        "artifact_path": str(merged_path),
        "fixed_grid_scale_selection": None,
        "hard_vetoes": _deduplicated(
            [payload.get("hard_vetoes", []) for payload in payloads.values()]
        ),
        "repair_triggers": _deduplicated(
            [payload.get("repair_triggers", []) for payload in payloads.values()]
        ),
        "passed": selected_index is not None,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "nonclaims": first["nonclaims"],
        "parallel_shard_provenance": {
            "schema": SCHEMA,
            "canonical_grid": CANONICAL_GRID,
            "seed_equivalence": "public_candidate_index_offsets_preserved",
            "selection_equivalence": "public_select_candidate_tuple_preserved",
            "worker_tuning_artifact_sha256": {
                str(leapfrog): worker_summaries[leapfrog]["tuning_artifact_sha256"]
                for leapfrog in CANONICAL_GRID
            },
        },
    }
    _write_json(merged_path, merged)
    return merged


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "worktree_dirty": bool(status), "status": status}


def _terminate(processes: Mapping[int, subprocess.Popen[str]]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 30.0
    for process in processes.values():
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_supervisor(args: argparse.Namespace) -> int:
    _validate_static_contract()
    if tuple(sorted(os.sched_getaffinity(0))) != (SUPERVISOR_CPU,):
        raise CampaignError("supervisor must be pinned to CPU 127")
    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("supervisor output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    affinity_probes = {}
    for row in ASSIGNMENTS:
        cpus = tuple(int(cpu) for cpu in row["cpus"])
        specification = f"{cpus[0]}-{cpus[-1]}"
        probe = subprocess.run(
            ["taskset", "-c", specification, "true"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        affinity_probes[str(row["leapfrog"])] = {
            "specification": specification,
            "exit_code": probe.returncode,
        }
        if probe.returncode != 0:
            raise CampaignError(f"cannot assign CPUs {specification}")

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    processes: dict[int, subprocess.Popen[str]] = {}
    logs: dict[int, Any] = {}
    commands = {}
    for row in ASSIGNMENTS:
        leapfrog = int(row["leapfrog"])
        cpus = tuple(int(cpu) for cpu in row["cpus"])
        worker_root = output / f"l{leapfrog}"
        command = [
            "taskset",
            "-c",
            f"{cpus[0]}-{cpus[-1]}",
            sys.executable,
            str(SCRIPT),
            "--mode",
            "worker",
            "--leapfrog",
            str(leapfrog),
            "--candidate-index",
            str(row["candidate_index"]),
            "--threads",
            str(len(cpus)),
            "--output-root",
            str(worker_root.relative_to(ROOT)),
        ]
        log = (output / f"l{leapfrog}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=_worker_environment(len(cpus)),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes[leapfrog] = process
        logs[leapfrog] = log
        commands[str(leapfrog)] = command

    terminal = {}
    timed_out = False
    pending = dict(processes)
    try:
        while pending:
            if time.perf_counter() - started >= args.cap_seconds:
                timed_out = True
                _terminate(pending)
            for leapfrog, process in list(pending.items()):
                code = process.poll()
                if code is None:
                    continue
                terminal[str(leapfrog)] = {
                    "pid": process.pid,
                    "exit_code": code,
                    "elapsed_at_observation_seconds": time.perf_counter() - started,
                }
                logs[leapfrog].close()
                del pending[leapfrog]
            if pending and not timed_out:
                time.sleep(2.0)
            elif pending:
                time.sleep(0.1)
    finally:
        _terminate(pending)
        for leapfrog, log in logs.items():
            if not log.closed:
                log.close()

    summaries = {}
    for row in ASSIGNMENTS:
        leapfrog = int(row["leapfrog"])
        path = output / f"l{leapfrog}" / "summary.json"
        if path.is_file():
            summaries[leapfrog] = _read_json(path)
    all_completed = bool(
        not timed_out
        and len(terminal) == len(ASSIGNMENTS)
        and all(row["exit_code"] == 0 for row in terminal.values())
        and len(summaries) == len(ASSIGNMENTS)
    )
    merged = _merge_results(output, summaries) if all_completed else None
    wall = time.perf_counter() - started
    summary = {
        "schema": SCHEMA,
        "role": "six_l_supervisor",
        "status": (
            "GRID_COMPLETED_CANDIDATE_NOMINATED"
            if merged is not None and merged["passed"]
            else "GRID_COMPLETED_NO_VIABLE_CANDIDATE"
            if merged is not None
            else "GRID_INCOMPLETE"
        ),
        "chart": "chart-a",
        "started_utc": started_utc,
        "wall_seconds": wall,
        "cap_seconds": args.cap_seconds,
        "timed_out": timed_out,
        "canonical_grid": CANONICAL_GRID,
        "assignments": ASSIGNMENTS,
        "total_worker_cores": sum(len(row["cpus"]) for row in ASSIGNMENTS),
        "supervisor_cpu": SUPERVISOR_CPU,
        "commands": commands,
        "affinity_probes": affinity_probes,
        "terminal": terminal,
        "worker_summaries": summaries,
        "merged_tuning_result_path": (
            None if merged is None else str(output / "merged-tuning-result.json")
        ),
        "merged_tuning_result_sha256": (
            None
            if merged is None
            else _sha256(output / "merged-tuning-result.json")
        ),
        "selected_candidate_index": (
            None if merged is None else merged["selected_candidate_index"]
        ),
        "selected_kernel": None if merged is None else merged["final_kernel_payload"],
        "sequential_hmc_launched": False,
        "git": _git_manifest(),
        "source_hashes": {
            "supervisor": _sha256(SCRIPT),
            "base_harness": _sha256(BASE_HARNESS),
            "tensorflow_tuner": _sha256(
                ROOT / "bayesfilter/inference/fixed_transport_hmc_tuning_tf.py"
            ),
            "plan": _sha256(ROOT / PLAN),
        },
        "plan": PLAN.as_posix(),
        "nonclaims": [
            "Chart A fixed-HMC kernel candidate tuning only",
            "no sequential HMC, convergence, posterior, chart B, or default claim",
        ],
    }
    _write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "wall_seconds": wall,
                "selected_candidate_index": summary["selected_candidate_index"],
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if all_completed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "worker", "supervisor"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--leapfrog", type=int)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cap-seconds", type=float, default=43_200.0)
    args = parser.parse_args()
    if args.mode == "preflight":
        _validate_static_contract()
        print(json.dumps({"status": "PREFLIGHT_PASS", "total_worker_cores": 58}))
        return 0
    if args.mode == "worker":
        if args.leapfrog is None or args.candidate_index is None or args.threads is None:
            parser.error("worker mode requires --leapfrog, --candidate-index, and --threads")
        return _run_worker(args)
    if any(value is not None for value in (args.leapfrog, args.candidate_index, args.threads)):
        parser.error("supervisor mode does not accept worker arguments")
    if not 0.0 < args.cap_seconds <= 43_200.0:
        parser.error("--cap-seconds must be in (0, 43200]")
    return _run_supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
