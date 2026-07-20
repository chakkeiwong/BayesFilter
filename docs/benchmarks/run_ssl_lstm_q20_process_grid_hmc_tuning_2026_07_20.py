#!/usr/bin/env python3
"""Bounded q=20 plain-target fixed-metric HMC tuning supervisor.

The parent launch environment is validated before BayesFilter can import
TensorFlow. GPU device initialization and target construction occur only in
spawned workers after their memory-growth policy exists.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _require_parent_launch_environment() -> None:
    """Fail before BayesFilter can import TensorFlow in material CLI runs."""

    if "--mode" not in sys.argv:
        return
    index = sys.argv.index("--mode")
    if index + 1 >= len(sys.argv):
        return
    mode = sys.argv[index + 1]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mode == "contract-smoke":
        if visible != "-1":
            raise RuntimeError("contract-smoke requires CUDA_VISIBLE_DEVICES=-1")
        return
    if visible not in {"0", "1"}:
        raise RuntimeError("material parent requires explicit CUDA_VISIBLE_DEVICES=0 or 1")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("material parent requires TF_FORCE_GPU_ALLOW_GROWTH=true")


_require_parent_launch_environment()


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_fixed_metric_grid_search import (  # noqa: E402
    DEFAULT_L_GRID,
    REPLICATION_COUNT,
    FixedMetricCandidateWorkerRequest,
    FixedMetricGridExecutionConfig,
    FixedMetricGridSearchConfig,
    FixedMetricSearchLineage,
    refinement_l_values,
    run_fixed_metric_grid_search,
)
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy  # noqa: E402
from bayesfilter.testing.ssl_lstm_q20_fixed_metric_worker import (  # noqa: E402
    HOST_RAM_CAP_BYTES,
    SCREEN_BURNIN,
    TARGET_SIGNATURE,
    TUNE_BURNIN,
    TUNE_RESULTS,
    expected_lineage_payload,
    q20_fixed_kernel_hmc_test,
    q20_hmc_rate_probe,
    run_q20_candidate_worker,
)


SCHEMA = "bayesfilter.ssl_lstm_q20.process_grid_hmc_tuning.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-process-grid-hmc-tuning-plan-2026-07-20.md"
)
RESULT = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-process-grid-hmc-tuning-result-2026-07-20.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
WORKER_FACTORY = (
    "bayesfilter.testing.ssl_lstm_q20_fixed_metric_worker:"
    "q20_fixed_metric_worker_factory"
)
GPU_CAP_SECONDS = 8.0 * 3600.0
PROJECTION_MARGIN = 1.50
HMC_TEST_RESULTS = 64
HMC_TEST_BURNIN = 64
HMC_TEST_SEED = (20260720, 9800)
HOST_RAM_CAP_GIB = 64


class Q20TuningError(RuntimeError):
    pass


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(label: str, payload: Any) -> str:
    return hashlib.sha256(canonical({"label": label, "payload": payload})).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Q20TuningError(f"expected a JSON object: {path}")
    return value


def repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise Q20TuningError(f"{label} must remain inside the repository")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    resolved = repo_path(path, label="output")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise Q20TuningError(f"output already exists: {path}")
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(resolved)


def source_bindings() -> Mapping[str, Any]:
    paths = {
        "plan": PLAN,
        "supervisor": SCRIPT,
        "worker": Path("bayesfilter/testing/ssl_lstm_q20_fixed_metric_worker.py"),
        "grid": Path("bayesfilter/inference/hmc_fixed_metric_grid_search.py"),
        "acceptance": Path("bayesfilter/inference/hmc_verification.py"),
        "hmc": Path("bayesfilter/inference/hmc.py"),
        "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        "filter": Path("bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py"),
    }
    hashes = {name: sha256(ROOT / path) for name, path in paths.items()}
    return {
        "paths": {name: path.as_posix() for name, path in paths.items()},
        "sha256": hashes,
        "execution_source_signature": signature(
            "bayesfilter.ssl_lstm_q20.process_grid_sources.v1", hashes
        ),
    }


def lineage() -> FixedMetricSearchLineage:
    return FixedMetricSearchLineage(**expected_lineage_payload())


def worker_environment(gpu: int, telemetry_dir: Path) -> tuple[tuple[str, str], ...]:
    if int(gpu) not in {0, 1}:
        raise Q20TuningError("gpu must be 0 or 1")
    directory = repo_path(telemetry_dir, label="worker telemetry directory")
    return tuple(
        sorted(
            {
                "CUDA_VISIBLE_DEVICES": str(int(gpu)),
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "BAYESFILTER_Q20_WORKER_TELEMETRY_DIR": directory.as_posix(),
                "PYTHONPYCACHEPREFIX": "/tmp/bayesfilter-q20-grid-pycache",
            }.items()
        )
    )


@contextmanager
def installed_environment(items: Sequence[tuple[str, str]]):
    previous = {key: os.environ.get(key) for key, _ in items}
    os.environ.update(dict(items))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def candidate_request(
    leapfrog: int,
    *,
    round_index: int = 0,
    config: FixedMetricGridSearchConfig | None = None,
) -> FixedMetricCandidateWorkerRequest:
    return FixedMetricCandidateWorkerRequest(
        round_index=int(round_index),
        num_leapfrog_steps=int(leapfrog),
        config=(
            FixedMetricGridSearchConfig(refinement_rounds=0)
            if config is None
            else config
        ),
        lineage=lineage(),
        acceptance_policy=HMCAcceptancePolicy(),
    )


def candidate_transition_leapfrogs(
    leapfrog: int,
    *,
    screen_results: int = 64,
    extension_results: int = 128,
    include_all_extensions: bool,
) -> int:
    tune_transitions = TUNE_BURNIN + TUNE_RESULTS
    screens = REPLICATION_COUNT * (SCREEN_BURNIN + int(screen_results))
    extensions = (
        REPLICATION_COUNT * (SCREEN_BURNIN + int(extension_results))
        if include_all_extensions
        else 0
    )
    return int(leapfrog) * (tune_transitions + screens + extensions)


def projected_seconds(
    leapfrog_values: Sequence[int],
    *,
    seconds_per_transition_leapfrog: float,
    effective_workers: float,
    include_all_extensions: bool,
    cold_seconds_per_worker: float = 0.0,
    worker_process_count: int | None = None,
) -> float:
    rate = float(seconds_per_transition_leapfrog)
    workers = float(effective_workers)
    if not 0.0 < rate or not 0.0 < workers:
        raise ValueError("rate and effective_workers must be positive")
    work = sum(
        candidate_transition_leapfrogs(
            value, include_all_extensions=include_all_extensions
        )
        for value in leapfrog_values
    )
    processes = (
        math_ceil(workers)
        if worker_process_count is None
        else int(worker_process_count)
    )
    if processes <= 0:
        raise ValueError("worker_process_count must be positive")
    return PROJECTION_MARGIN * (
        work * rate / workers + float(cold_seconds_per_worker) * processes
    )


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if float(integer) == value else integer + 1


def validate_resource_launch(
    *, cap_seconds: float, prior_seconds: float, projection_seconds: float
) -> None:
    cap = float(cap_seconds)
    prior = float(prior_seconds)
    projection = float(projection_seconds)
    if cap <= 0.0 or prior < 0.0 or projection <= 0.0:
        raise Q20TuningError("resource values must be positive with nonnegative prior")
    if cap > GPU_CAP_SECONDS:
        raise Q20TuningError("cap exceeds the prospective eight-GPU-hour ceiling")
    if prior + projection > cap:
        raise Q20TuningError("prospective resource projection exceeds remaining cap")


def run_manifest(
    *, args: argparse.Namespace, started: float, output: Path
) -> Mapping[str, Any]:
    return {
        "git_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ("git", "status", "--porcelain"), cwd=ROOT, text=True
            ).strip()
        ),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "cpu_gpu_status": (
            "cpu_hidden_contract"
            if args.mode == "contract-smoke"
            else "trusted_gpu_xla_spawn_workers"
        ),
        "selected_physical_gpu": getattr(args, "gpu", None),
        "tf_force_gpu_allow_growth": (
            None
            if args.mode == "contract-smoke"
            else "true_required_and_worker_verified"
        ),
        "jit_compile": args.mode != "contract-smoke",
        "dtype": "float64",
        "tf32": "recorded_by_worker; FP64 target",
        "random_seeds": {
            "grid_root": list(FixedMetricGridSearchConfig().root_seed),
            "hmc_test": list(HMC_TEST_SEED),
        },
        "wall_seconds": time.perf_counter() - started,
        "output": output.as_posix(),
        "plan": PLAN.as_posix(),
        "result": RESULT.as_posix(),
        "trust_basis": (
            "cpu_hidden_contract_only"
            if args.mode == "contract-smoke"
            else "owner_designated_managed_session_visible_gpu_trusted"
        ),
    }


def contract_payload() -> Mapping[str, Any]:
    config = FixedMetricGridSearchConfig(refinement_rounds=0)
    policy = HMCAcceptancePolicy()
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "status": "PASSED",
        "q": 20,
        "target_signature": TARGET_SIGNATURE,
        "target_type": "plain_q20_posterior_no_transport",
        "lineage": lineage().payload(),
        "config": config.payload(),
        "acceptance_policy": policy.payload(),
        "tune": {
            "num_results": TUNE_RESULTS,
            "num_burnin_steps": TUNE_BURNIN,
            "num_adaptation_steps": TUNE_BURNIN,
            "target_acceptance": policy.target,
        },
        "screen_burnin_steps": SCREEN_BURNIN,
        "gpu_cap_seconds": GPU_CAP_SECONDS,
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
        "host_ram_cap_gib": HOST_RAM_CAP_GIB,
        "material_execution_authorized": False,
        "source_bindings": source_bindings(),
        "nonclaims": (
            "contract smoke only",
            "GPU hidden; no GPU initialization or target construction",
            "no HMC execution or tuning result",
        ),
    }


def _spawn_call(
    function: Any,
    argument: Any,
    *,
    environment: Sequence[tuple[str, str]],
) -> Any:
    context = multiprocessing.get_context("spawn")
    with installed_environment(environment):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=context
        ) as executor:
            return executor.submit(function, argument).result()


def run_rate_probe(args: argparse.Namespace) -> Mapping[str, Any]:
    environment = worker_environment(args.gpu, args.telemetry_dir)
    probe = _spawn_call(
        q20_hmc_rate_probe,
        (20260720, 9700),
        environment=environment,
    )
    rate = float(probe["warm_seconds_per_transition_leapfrog_max"])
    projections = {
        "round0_no_extensions_one_worker_seconds": projected_seconds(
            DEFAULT_L_GRID,
            seconds_per_transition_leapfrog=rate,
            effective_workers=1.0,
            include_all_extensions=False,
            cold_seconds_per_worker=float(probe["first_call_seconds"]),
        ),
        "round0_all_extensions_one_worker_seconds": projected_seconds(
            DEFAULT_L_GRID,
            seconds_per_transition_leapfrog=rate,
            effective_workers=1.0,
            include_all_extensions=True,
            cold_seconds_per_worker=float(probe["first_call_seconds"]),
        ),
    }


def run_rate_topology(args: argparse.Namespace) -> Mapping[str, Any]:
    environment = worker_environment(args.gpu, args.telemetry_dir)
    context = multiprocessing.get_context("spawn")
    seeds = tuple((20260720, 9710 + index) for index in range(args.workers))
    started = time.perf_counter()
    with installed_environment(environment):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context
        ) as executor:
            futures = [executor.submit(q20_hmc_rate_probe, seed) for seed in seeds]
            probes = tuple(future.result() for future in futures)
    wall = time.perf_counter() - started
    contended_rate = max(
        float(probe["warm_seconds_per_transition_leapfrog_max"])
        for probe in probes
    )
    contended_cold = max(float(probe["first_call_seconds"]) for probe in probes)
    projection = projected_seconds(
        DEFAULT_L_GRID,
        seconds_per_transition_leapfrog=contended_rate,
        effective_workers=float(args.workers),
        include_all_extensions=True,
        cold_seconds_per_worker=contended_cold,
        worker_process_count=args.workers,
    )
    return {
        "schema": SCHEMA,
        "mode": "rate-topology",
        "status": "PASSED",
        "workers": args.workers,
        "wall_seconds": wall,
        "probes": probes,
        "contended_warm_seconds_per_transition_leapfrog_max": contended_rate,
        "contended_cold_seconds_max": contended_cold,
        "round0_all_extensions_projected_seconds": projection,
        "source_bindings": source_bindings(),
        "nonclaims": (
            "bounded concurrent rate and resource canary only",
            "no universal topology ranking or tuning evidence",
        ),
    }
    return {
        "schema": SCHEMA,
        "mode": "rate-probe",
        "status": "PASSED",
        "probe": probe,
        "projections": projections,
        "source_bindings": source_bindings(),
        "nonclaims": (
            "current-source timing and mechanics probe only",
            "projection is a resource diagnostic, not tuning evidence",
        ),
    }


def _run_candidate_requests(
    requests: Sequence[FixedMetricCandidateWorkerRequest],
    *,
    max_workers: int,
    environment: Sequence[tuple[str, str]],
) -> tuple[Any, ...]:
    context = multiprocessing.get_context("spawn")
    results: list[Any | None] = [None] * len(requests)
    with installed_environment(environment):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(int(max_workers), len(requests)),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(run_q20_candidate_worker, request): index
                for index, request in enumerate(requests)
            }
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
    if any(result is None for result in results):
        raise Q20TuningError("candidate worker did not return a result")
    return tuple(results)


def run_topology(args: argparse.Namespace) -> Mapping[str, Any]:
    l_values = tuple(int(item) for item in args.l_values)
    requests = tuple(candidate_request(value) for value in l_values)
    environment = worker_environment(args.gpu, args.telemetry_dir)
    validate_resource_launch(
        cap_seconds=args.cap_seconds,
        prior_seconds=args.prior_seconds,
        projection_seconds=args.projected_seconds,
    )
    started = time.perf_counter()
    candidates = _run_candidate_requests(
        requests,
        max_workers=args.workers,
        environment=environment,
    )
    wall = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "mode": "topology",
        "status": "PASSED",
        "workers": args.workers,
        "l_values": l_values,
        "wall_seconds": wall,
        "candidate_payloads": tuple(candidate.payload() for candidate in candidates),
        "candidate_signatures": tuple(candidate.signature for candidate in candidates),
        "surviving_l_values": tuple(
            candidate.num_leapfrog_steps for candidate in candidates if candidate.survivor
        ),
        "resource_contract": {
            "cap_seconds": args.cap_seconds,
            "prior_seconds": args.prior_seconds,
            "projected_seconds": args.projected_seconds,
        },
        "source_bindings": source_bindings(),
        "nonclaims": (
            "bounded topology and real-target candidate canary",
            "worker timing differences are descriptive only",
            "no convergence or posterior claim",
        ),
    }


def run_grid(args: argparse.Namespace) -> Mapping[str, Any]:
    validate_resource_launch(
        cap_seconds=args.cap_seconds,
        prior_seconds=args.prior_seconds,
        projection_seconds=args.projected_seconds,
    )
    config = FixedMetricGridSearchConfig(refinement_rounds=0)
    environment = worker_environment(args.gpu, args.telemetry_dir)
    execution = FixedMetricGridExecutionConfig(
        mode="process_parallel",
        max_workers=args.workers,
        worker_factory_locator=WORKER_FACTORY,
        worker_environment=environment,
    )
    started = time.perf_counter()
    result = run_fixed_metric_grid_search(
        config=config,
        lineage=lineage(),
        acceptance_policy=HMCAcceptancePolicy(),
        execution=execution,
    )
    wall = time.perf_counter() - started
    status = (
        "TUNING_SUCCEEDED"
        if result.survivors
        else "SHARED_EXECUTION_INVALID"
        if result.shared_invalidity_reasons
        else "NO_SURVIVOR"
    )
    return {
        "schema": SCHEMA,
        "mode": "grid",
        "status": status,
        "q": 20,
        "target_type": "plain_q20_posterior_no_transport",
        "grid_private": result.payload(),
        "grid_public": result.public_summary(),
        "round0_complete": len(result.round0_candidates) == len(DEFAULT_L_GRID),
        "refinement_authorized": False,
        "refinement_reason": (
            "round0 broad grid is the prospective tuning gate; refinement is optional "
            "candidate enrichment and was not charged under this bounded run"
        ),
        "wall_seconds": wall,
        "resource_contract": {
            "cap_seconds": args.cap_seconds,
            "prior_seconds": args.prior_seconds,
            "projected_seconds": args.projected_seconds,
            "charged_seconds_this_command": wall,
        },
        "source_bindings": source_bindings(),
        "all_tuning_draws_discarded": True,
        "nonclaims": (
            "plain-target fixed-metric HMC tuning only",
            "no NeuTra transport or NeuTra-HMC result",
            "no convergence, posterior correctness, or candidate ranking claim",
        ),
    }


def representative_from_grid(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("schema") != SCHEMA or payload.get("mode") != "grid":
        raise Q20TuningError("HMC test requires this supervisor's grid result")
    if payload.get("status") != "TUNING_SUCCEEDED":
        raise Q20TuningError("HMC test requires successful tuning")
    if payload.get("round0_complete") is not True:
        raise Q20TuningError("HMC test requires a complete broad Round-0 grid")
    private = payload.get("grid_private")
    if not isinstance(private, Mapping):
        raise Q20TuningError("grid result lacks private candidate mechanics")
    survivors = private.get("survivor_pairs")
    if not isinstance(survivors, list) or not survivors:
        raise Q20TuningError("grid result lacks survivors")
    ordered = sorted(
        survivors,
        key=lambda item: (
            int(item["num_leapfrog_steps"]), float(item["tuned_step_size"])
        ),
    )
    selected = ordered[0]
    return {
        "selection_rule": "smallest_L_then_step_tie_break_no_stochastic_ranking",
        "num_leapfrog_steps": int(selected["num_leapfrog_steps"]),
        "step_size": float(selected["tuned_step_size"]),
        "lineage": private["lineage"],
        "grid_source_signature": payload["source_bindings"][
            "execution_source_signature"
        ],
    }


def run_hmc_test(args: argparse.Namespace) -> Mapping[str, Any]:
    grid_path = repo_path(args.grid_result, label="grid result")
    grid = strict_json(grid_path)
    representative = representative_from_grid(grid)
    current_sources = source_bindings()
    if representative["grid_source_signature"] != current_sources[
        "execution_source_signature"
    ]:
        raise Q20TuningError("execution source changed after tuning")
    validate_resource_launch(
        cap_seconds=args.cap_seconds,
        prior_seconds=args.prior_seconds,
        projection_seconds=args.projected_seconds,
    )
    environment = worker_environment(args.gpu, args.telemetry_dir)
    test_input = {
        **representative,
        "seed": HMC_TEST_SEED,
    }
    test = _spawn_call(
        q20_fixed_kernel_hmc_test,
        test_input,
        environment=environment,
    )
    return {
        "schema": SCHEMA,
        "mode": "hmc-test",
        "status": (
            "HMC_TEST_PASSED" if test["status"] == "PASSED" else "HMC_TEST_VETOED"
        ),
        "grid_result_path": args.grid_result.as_posix(),
        "grid_result_sha256": sha256(grid_path),
        "representative": representative,
        "test": test,
        "resource_contract": {
            "cap_seconds": args.cap_seconds,
            "prior_seconds": args.prior_seconds,
            "projected_seconds": args.projected_seconds,
        },
        "source_bindings": current_sources,
        "raw_samples_retained": False,
        "nonclaims": (
            "short four-chain HMC mechanics test only",
            "no convergence, posterior correctness, or model-adequacy claim",
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "contract-smoke",
            "rate-probe",
            "rate-topology",
            "topology",
            "grid",
            "hmc-test",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--telemetry-dir", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--l-values", type=int, nargs="+", default=(3, 5))
    parser.add_argument("--cap-seconds", type=float, default=GPU_CAP_SECONDS)
    parser.add_argument("--prior-seconds", type=float, default=0.0)
    parser.add_argument("--projected-seconds", type=float)
    parser.add_argument("--grid-result", type=Path)
    parser.add_argument("--authorize-material-run", action="store_true")
    args = parser.parse_args(argv)
    if args.mode != "contract-smoke":
        if not args.authorize_material_run:
            parser.error("material modes require --authorize-material-run")
        if args.gpu is None or args.telemetry_dir is None:
            parser.error("material modes require --gpu and --telemetry-dir")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
            parser.error("--gpu must match parent CUDA_VISIBLE_DEVICES")
    if args.mode == "rate-topology" and args.workers not in {2, 4}:
        parser.error("rate-topology requires two or four workers")
    if args.mode in {"topology", "grid", "hmc-test"} and (
        args.projected_seconds is None or args.projected_seconds <= 0.0
    ):
        parser.error(f"{args.mode} requires positive --projected-seconds")
    if args.mode == "hmc-test" and args.grid_result is None:
        parser.error("hmc-test requires --grid-result")
    if args.mode != "hmc-test" and args.grid_result is not None:
        parser.error("--grid-result is valid only for hmc-test")
    if args.mode == "topology":
        l_values = tuple(args.l_values)
        if len(l_values) < args.workers:
            parser.error("topology requires at least one L value per worker")
        if len(set(l_values)) != len(l_values) or not set(l_values).issubset(
            set(DEFAULT_L_GRID)
        ):
            parser.error("topology L values must be distinct reviewed Round-0 values")
    repo_path(args.output, label="output")
    if args.telemetry_dir is not None:
        repo_path(args.telemetry_dir, label="telemetry directory")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    if args.mode == "contract-smoke":
        payload = dict(contract_payload())
    elif args.mode == "rate-probe":
        payload = dict(run_rate_probe(args))
    elif args.mode == "rate-topology":
        payload = dict(run_rate_topology(args))
    elif args.mode == "topology":
        payload = dict(run_topology(args))
    elif args.mode == "grid":
        payload = dict(run_grid(args))
    else:
        payload = dict(run_hmc_test(args))
    payload["run_manifest"] = run_manifest(
        args=args, started=started, output=args.output
    )
    write_json(args.output, payload)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "mode": payload["mode"],
                "status": payload["status"],
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
