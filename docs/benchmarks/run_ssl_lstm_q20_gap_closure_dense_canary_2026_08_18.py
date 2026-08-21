#!/usr/bin/env python3
"""Run one bounded hot-endpoint replica-ladder tuning arm."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-plan-2026-08-18.md"
)
RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_q20_gap_closure_dense_canary_2026_08_18.py"
)
CHECKPOINT_RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_checkpoint_2026_08_10.py"
)
MATERIAL_RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_material_2026_08_11.py"
)
DISTRIBUTED_HELPER = Path("bayesfilter/testing/distributed_replica_exchange_tf.py")
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/"
    "r1/geometry.json"
)
FAILED_MATERIAL = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r8-material-24x1-eight-hour/material.json"
)
ARTIFACT_PARENT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10"
)
RATIO_0P40_RESULT = ARTIFACT_PARENT / "r9-hot-tuning-ratio-0p40/canary.json"
RATIO_0P35_RESULT = ARTIFACT_PARENT / "r9-hot-tuning-ratio-0p35/canary.json"
RESUMED_MATERIAL = ARTIFACT_PARENT / "r11-material-24x1-resumed/material.json"
WARMUP_DIAGNOSIS = ARTIFACT_PARENT / "r12-warmup-failure-decomposition/diagnosis.json"
DENSE_MASS_0P35 = ARTIFACT_PARENT / "r13-dense-mass-step-0p35/canary.json"

PARAMETER_DIM = 4
CHAINS = 4
REPLICAS = 6
ROWS = CHAINS * REPLICAS
WORKERS = ROWS
ROWS_PER_WORKER = 1
LEAPFROG = 8
TRANSITIONS = 100
CHUNK_SIZE = 10
BASE_STEP = 0.05
ACCEPTANCE_LOWER = 0.35
ACCEPTANCE_UPPER = 0.99
MASTER_SEED = (20260811, 8201)
CAP_SECONDS = 2400.0
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
FAILED_MATERIAL_SHA256 = "9e6771652842b6f96e304509a042949dc2513923ef8279021a7783b4fd82b9d9"
RATIO_0P40_SHA256 = "b58381e92dc609ff2b33dade8901c62558df5cbc23d82c6fb25a5eb6a261e570"
RATIO_0P35_SHA256 = "22349ca8141f2b89d921adbc22eafb5774901eadf50f07d0a05d6fc4618394b2"
RESUMED_MATERIAL_SHA256 = "0fbec0c372008d406953908a30b6aa66a27d843781a93dba5ae52cd98235c66b"
WARMUP_DIAGNOSIS_SHA256 = "4946e5b33bf3258d00ef86ea2bf6067b7928e74ce85c871990975e760c15832c"
DENSE_MASS_0P35_SHA256 = "3b785f78ca2e18e44162756cde7a69088c8bb3723f2549dee1106f4567dc63f0"
CANDIDATES = {
    "ratio-0p40": {"ratio": 0.40, "hot_step_multiplier": 1.0, "cpu_start": 32, "output": "r9-hot-tuning-ratio-0p40"},
    "ratio-0p35": {"ratio": 0.35, "hot_step_multiplier": 1.0, "cpu_start": 64, "output": "r9-hot-tuning-ratio-0p35"},
    "hot-step-1p5": {"ratio": 0.50, "hot_step_multiplier": 1.5, "cpu_start": 32, "output": "r10-hot-step-1p5"},
    "hot-step-2p0": {"ratio": 0.50, "hot_step_multiplier": 2.0, "cpu_start": 64, "output": "r10-hot-step-2p0"},
    "dense-mass-step-0p35": {"ratio": 0.50, "base_step": 0.35, "hot_step_multiplier": 1.0, "dense_mass": True, "cpu_start": 32, "output": "r1-dense-mass-step-0p35"},
    "dense-mass-step-0p70": {"ratio": 0.50, "base_step": 0.70, "hot_step_multiplier": 1.0, "dense_mass": True, "cpu_start": 64, "output": "r13-dense-mass-step-0p70"},
    "dense-mass-step-0p35-l4": {"ratio": 0.50, "base_step": 0.35, "leapfrog": 4, "hot_step_multiplier": 1.0, "dense_mass": True, "cpu_start": 32, "output": "r14-dense-mass-step-0p35-l4"},
}


class HotEndpointCanaryError(RuntimeError):
    """Raised when a tuning arm cannot produce valid evidence."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists() and not overwrite:
        raise HotEndpointCanaryError(f"refusing to overwrite {path}")
    encoded = json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def _write_tensor(path: Path, value: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise HotEndpointCanaryError(f"refusing to overwrite tensor: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(absolute)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, _abs(path))
    if spec is None or spec.loader is None:
        raise HotEndpointCanaryError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pool_config(cpu_ids: tuple[int, ...]) -> Any:
    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePoolConfig

    return TFBatchValueScorePoolConfig(
        factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:"
            "batch_native_complexity_target_worker_factory"
        ),
        factory_config={
            "q": 20,
            "principal_sqrt_backend": "tensorflow_eigh",
            "jit_compile": True,
        },
        dimension=PARAMETER_DIM,
        worker_count=WORKERS,
        cores_per_worker=1,
        batch_sizes=(1,),
        batch_per_worker=1,
        worker_cpu_ids=cpu_ids,
        timeout_seconds=900.0,
    )


def run(candidate: str) -> Mapping[str, Any]:
    started = time.perf_counter()
    config = CANDIDATES[candidate]
    ratio = float(config["ratio"])
    base_step = float(config.get("base_step", BASE_STEP))
    dense_mass = bool(config.get("dense_mass", False))
    leapfrog = int(config.get("leapfrog", LEAPFROG))
    hot_step_multiplier = float(config["hot_step_multiplier"])
    cpu_start = int(config["cpu_start"])
    worker_cpu_ids = tuple(range(cpu_start, cpu_start + WORKERS))
    parent_cpu_ids = tuple(range(cpu_start, cpu_start + 32))
    output_root = ARTIFACT_PARENT / str(config["output"])
    progress = output_root / "progress.json"
    final = output_root / "canary.json"
    if _abs(final).exists():
        raise HotEndpointCanaryError("refusing to overwrite tuning canary")
    if tuple(sorted(os.sched_getaffinity(0))) != parent_cpu_ids:
        raise HotEndpointCanaryError("parent CPU affinity mismatch")
    if _sha(GEOMETRY) != GEOMETRY_SHA256 or _sha(FAILED_MATERIAL) != FAILED_MATERIAL_SHA256:
        raise HotEndpointCanaryError("bound evidence identity mismatch")
    if candidate.startswith("hot-step-") and (
        _sha(RATIO_0P40_RESULT) != RATIO_0P40_SHA256
        or _sha(RATIO_0P35_RESULT) != RATIO_0P35_SHA256
    ):
        raise HotEndpointCanaryError("bound ratio-tuning evidence identity mismatch")
    if dense_mass and (
        _sha(RESUMED_MATERIAL) != RESUMED_MATERIAL_SHA256
        or _sha(WARMUP_DIAGNOSIS) != WARMUP_DIAGNOSIS_SHA256
    ):
        raise HotEndpointCanaryError("bound warm-up repair evidence identity mismatch")
    if candidate.endswith("-l4") and _sha(DENSE_MASS_0P35) != DENSE_MASS_0P35_SHA256:
        raise HotEndpointCanaryError("bound dense-mass L8 evidence identity mismatch")
    failed = json.loads(_abs(FAILED_MATERIAL).read_text(encoding="utf-8"))
    if failed.get("status") != "MATERIAL_REPLICA_WARMUP_NOT_READY":
        raise HotEndpointCanaryError("repair trigger status changed")

    betas = tuple(ratio**index for index in range(REPLICAS))
    base_steps = tuple(base_step / math.sqrt(beta) for beta in betas)
    steps = base_steps[:-1] + (base_steps[-1] * hot_step_multiplier,)
    _write_json(progress, {"status": "HOT_ENDPOINT_CANARY_STARTING", "candidate": candidate}, overwrite=True)

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise HotEndpointCanaryError("CPU-only canary found visible GPU")

    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool
    from bayesfilter.testing.distributed_replica_exchange_tf import (
        distributed_replica_exchange_transition,
        initialize_distributed_replica_state,
    )

    checkpoint_support = _load(f"hot_endpoint_checkpoint_{candidate}", CHECKPOINT_RUNNER)
    material_support = _load(f"hot_endpoint_material_{candidate}", MATERIAL_RUNNER)
    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    chart = checkpoint_support._chart(tf, geometry)
    source_precisions = tf.stack(
        [
            tf.constant(
                geometry["source_curvature"][label]["records"][-1]["precision"],
                tf.float64,
            )
            for label in ("plus", "minus")
        ]
    )
    mapped_precisions = tf.stack(
        [
            tf.matmul(chart["factor"], tf.matmul(value, chart["factor"]), transpose_a=True)
            for value in source_precisions
        ]
    )
    mass_matrix = tf.reduce_mean(mapped_precisions, axis=0) if dense_mass else None
    chain_centers = tf.gather(chart["latent_centers"], (0, 1, 0, 1))
    initial_state = tf.repeat(chain_centers[tf.newaxis, :, :], REPLICAS, axis=0)
    initial_physical = chart["center"] + tf.matmul(
        tf.reshape(initial_state, (ROWS, PARAMETER_DIM)), chart["factor"], transpose_b=True
    )
    initial_physical = tf.reshape(initial_physical, (REPLICAS, CHAINS, PARAMETER_DIM))
    state_rows = []
    pre_rows = []
    identity_rows = []
    accept_rows = []
    log_accept_rows = []
    valid_rows = []
    finite_log_accept_rows = []
    swap_proposed_rows = []
    swap_accepted_rows = []
    transition_seconds = []
    chunk_receipts = []

    with TFBatchValueScorePool(_pool_config(worker_cpu_ids)) as pool:
        def evaluator(rows: Any, request_id: str):
            latent = tf.ensure_shape(tf.convert_to_tensor(rows, tf.float64), (ROWS, PARAMETER_DIM))
            theta = chart["center"] + tf.matmul(latent, chart["factor"], transpose_b=True)
            value, score, status, metadata = pool.evaluate_with_status(theta, request_id=request_id)
            return (
                tf.convert_to_tensor(value, tf.float64) + chart["log_abs_determinant"],
                tf.matmul(tf.convert_to_tensor(score, tf.float64), chart["factor"]),
                status,
                metadata,
            )

        initialized = initialize_distributed_replica_state(initial_state, evaluator=evaluator)
        identity = checkpoint_support._worker_identity(initialized["evaluation_metadata"])
        identity_passed = bool(
            identity["worker_count"] == WORKERS
            and tuple(identity["assigned_cpu_ids"]) == worker_cpu_ids
            and tuple(identity["target_signatures"]) == (TARGET_SIGNATURE,)
            and tuple(identity["adapter_signatures"]) == (ADAPTER_SIGNATURE,)
            and tuple(identity["status_jit_compile_values"]) == (True,)
        )
        if not identity_passed:
            raise HotEndpointCanaryError("worker identity failed")
        current = {
            name: initialized[name]
            for name in ("state", "base_target_log_prob", "base_score", "identities_at_temperature")
        }
        for transition_index in range(TRANSITIONS):
            transition_started = time.perf_counter()
            transition = distributed_replica_exchange_transition(
                **current,
                inverse_temperatures=betas,
                step_sizes=steps,
                num_leapfrog_steps=leapfrog,
                transition_index=transition_index,
                master_seed=MASTER_SEED,
                evaluator=evaluator,
                mass_matrix=mass_matrix,
            )
            transition_seconds.append(time.perf_counter() - transition_started)
            invalid = tf.logical_not(transition["hmc_path_valid"])
            finite_log_accept_or_invalid = tf.reduce_all(
                tf.logical_or(
                    tf.math.is_finite(transition["hmc_log_accept_ratio"]),
                    tf.logical_and(
                        invalid,
                        tf.math.is_inf(transition["hmc_log_accept_ratio"])
                        & (transition["hmc_log_accept_ratio"] < 0.0),
                    ),
                )
            )
            matrix = tf.cast(transition["swap_is_accepted_matrix"], tf.int32)
            transition_valid = bool(
                tf.reduce_all(
                    tf.stack(
                        (
                            tf.reduce_all(tf.math.is_finite(transition["state"])),
                            tf.reduce_all(tf.math.is_finite(transition["base_target_log_prob"])),
                            tf.reduce_all(tf.math.is_finite(transition["base_score"])),
                            finite_log_accept_or_invalid,
                            tf.reduce_all(tf.logical_not(tf.boolean_mask(transition["hmc_is_accepted"], invalid))),
                            tf.reduce_all(tf.reduce_sum(matrix, axis=0) == tf.ones((REPLICAS, CHAINS), tf.int32)),
                            tf.reduce_all(tf.reduce_sum(matrix, axis=1) == tf.ones((REPLICAS, CHAINS), tf.int32)),
                        )
                    )
                ).numpy()
            )
            if not transition_valid:
                raise HotEndpointCanaryError(f"transition {transition_index} hard gate failed")
            state_rows.append(transition["state"])
            pre_rows.append(transition["pre_swap_state"])
            identity_rows.append(transition["identities_at_temperature"])
            accept_rows.append(transition["hmc_is_accepted"])
            log_accept_rows.append(transition["hmc_log_accept_ratio"])
            valid_rows.append(transition["hmc_path_valid"])
            finite_log_accept_rows.append(finite_log_accept_or_invalid)
            swap_proposed_rows.append(transition["swap_is_proposed_adjacent"])
            swap_accepted_rows.append(transition["swap_is_accepted_adjacent"])
            current = {
                name: transition[name]
                for name in ("state", "base_target_log_prob", "base_score", "identities_at_temperature")
            }
            if (transition_index + 1) % CHUNK_SIZE == 0:
                start = transition_index + 1 - CHUNK_SIZE
                receipt = _write_tensor(
                    output_root / f"chunk-{start // CHUNK_SIZE:02d}-state.tftensor",
                    tf.stack(state_rows[start : transition_index + 1]),
                    tf,
                )
                chunk_receipts.append(receipt)
                _write_json(
                    progress,
                    {
                        "status": "HOT_ENDPOINT_CANARY_RUNNING",
                        "candidate": candidate,
                        "completed_transitions": transition_index + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                        "last_receipt": receipt,
                    },
                    overwrite=True,
                )
            if time.perf_counter() - started > CAP_SECONDS:
                raise HotEndpointCanaryError("canary wall cap exceeded")

    states = tf.stack(state_rows)
    pre_states = tf.stack(pre_rows)
    identities = tf.stack(identity_rows)
    accepted = tf.stack(accept_rows)
    log_accept = tf.stack(log_accept_rows)
    valid = tf.stack(valid_rows)
    swap_proposed = tf.stack(swap_proposed_rows)
    swap_accepted = tf.stack(swap_accepted_rows)
    physical = chart["center"] + tf.matmul(
        tf.reshape(states, (-1, PARAMETER_DIM)), chart["factor"], transpose_b=True
    )
    physical = tf.reshape(physical, (TRANSITIONS, REPLICAS, CHAINS, PARAMETER_DIM))
    pre_physical = chart["center"] + tf.matmul(
        tf.reshape(pre_states, (-1, PARAMETER_DIM)), chart["factor"], transpose_b=True
    )
    pre_physical = tf.reshape(pre_physical, tf.shape(physical))
    travel = material_support._window_round_trips(tf, identities)
    forgetting = material_support._hot_forgetting(tf, pre_physical, physical, initial_physical)
    acceptance = material_support._acceptance_summary(tf, log_accept)
    proposed_counts = tf.reduce_sum(tf.cast(swap_proposed, tf.int32), axis=(0, 2))
    accepted_counts = tf.reduce_sum(tf.cast(swap_accepted, tf.int32), axis=(0, 2))
    communication_passed = tf.reduce_all(accepted_counts > 0)
    selection_passed = bool(
        forgetting["all_chains_passed"].numpy()
        and communication_passed.numpy()
        and acceptance["all_temperature_chain_means_in_band"].numpy()
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_hot_endpoint_tuning_canary.v1",
        "status": "HOT_ENDPOINT_CANARY_PASSED" if selection_passed else "HOT_ENDPOINT_CANARY_NOT_NOMINATED",
        "candidate": candidate,
        "selection_passed": selection_passed,
        "configuration": {
            "ratio": ratio,
            "hot_step_multiplier": hot_step_multiplier,
            "base_step": base_step,
            "mass_matrix": mass_matrix,
            "mass_policy": "mean_two_checked_mapped_local_precisions" if dense_mass else "identity",
            "inverse_temperatures": betas,
            "step_sizes": steps,
            "num_leapfrog_steps": leapfrog,
            "transitions": TRANSITIONS,
            "chains": CHAINS,
            "workers": WORKERS,
            "worker_cpu_ids": worker_cpu_ids,
            "master_seed": MASTER_SEED,
            "jit_compile": True,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        },
        "hard_gates": {
            "worker_identity_passed": identity_passed,
            "invalid_path_count": tf.reduce_sum(
                tf.cast(tf.logical_not(valid), tf.int32)
            ),
            "invalid_paths_self_rejected": tf.reduce_all(
                tf.logical_not(
                    tf.boolean_mask(accepted, tf.logical_not(valid))
                )
            ),
            "log_acceptance_finite_or_invalid_negative_infinity": tf.reduce_all(
                tf.stack(finite_log_accept_rows)
            ),
            "wall_time_within_cap": time.perf_counter() - started <= CAP_SECONDS,
        },
        "selection_screens": {
            "hot_forgetting_all_chains": forgetting["all_chains_passed"],
            "every_adjacent_pair_communicated": communication_passed,
            "acceptance_means_in_band": acceptance["all_temperature_chain_means_in_band"],
        },
        "hot_forgetting": forgetting,
        "travel": travel,
        "acceptance": acceptance,
        "swap_proposals_by_pair": proposed_counts,
        "swap_acceptances_by_pair": accepted_counts,
        "timing": {
            "transition_seconds": transition_seconds,
            "mean_transition_seconds": sum(transition_seconds) / len(transition_seconds),
            "wall_seconds": time.perf_counter() - started,
        },
        "chunk_receipts": chunk_receipts,
        "bindings": {
            "geometry_sha256": _sha(GEOMETRY),
            "failed_material_sha256": _sha(FAILED_MATERIAL),
            "ratio_0p40_sha256": (
                _sha(RATIO_0P40_RESULT) if candidate.startswith("hot-step-") else None
            ),
            "ratio_0p35_sha256": (
                _sha(RATIO_0P35_RESULT) if candidate.startswith("hot-step-") else None
            ),
            "resumed_material_sha256": _sha(RESUMED_MATERIAL) if dense_mass else None,
            "warmup_diagnosis_sha256": _sha(WARMUP_DIAGNOSIS) if dense_mass else None,
            "dense_mass_0p35_l8_sha256": _sha(DENSE_MASS_0P35) if candidate.endswith("-l4") else None,
        },
        "run_manifest": {
            "launch_git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "command": " ".join(sys.argv),
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "artifact_root": output_root.as_posix(),
            "plan_file": PLAN.as_posix(),
            "source_sha256": {"runner": _sha(RUNNER), "checkpoint_runner": _sha(CHECKPOINT_RUNNER), "material_runner": _sha(MATERIAL_RUNNER), "distributed_helper": _sha(DISTRIBUTED_HELPER)},
        },
        "nonclaims": (
            "100-transition tuning canary only",
            "R-hat, occupancy, and runtime are not selection criteria",
            "no convergence, posterior, predictive, superiority, or default claim",
        ),
    }
    _write_json(final, payload)
    _write_json(
        progress,
        {"status": payload["status"], "candidate": candidate, "result": final.as_posix(), "elapsed_seconds": time.perf_counter() - started},
        overwrite=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=tuple(CANDIDATES))
    args = parser.parse_args()
    config = CANDIDATES[args.candidate]
    output_root = ARTIFACT_PARENT / str(config["output"])
    final = output_root / "canary.json"
    progress = output_root / "progress.json"
    started = time.perf_counter()
    try:
        payload = run(args.candidate)
    except BaseException as error:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20_physical_hot_endpoint_tuning_failure.v1",
            "status": "HOT_ENDPOINT_CANARY_HARNESS_FAILED",
            "candidate": args.candidate,
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_seconds": time.perf_counter() - started,
        }
        if not _abs(final).exists():
            _write_json(final, failure)
        _write_json(progress, {**failure, "result": final.as_posix()}, overwrite=True)
        raise
    print(json.dumps({"status": payload["status"], "candidate": args.candidate}, sort_keys=True))


if __name__ == "__main__":
    main()
