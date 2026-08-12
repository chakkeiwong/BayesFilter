#!/usr/bin/env python3
"""Run the bounded physical annealed-SMC mechanics canary."""

from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-physical-annealed-smc-repair-plan-2026-08-10.md")
RESULT = Path("docs/plans/bayesfilter-ssl-lstm-q20-physical-annealed-smc-repair-result-2026-08-10.md")
RUNNER = Path("docs/benchmarks/run_ssl_lstm_q20_physical_annealed_smc_canary_2026_08_10.py")
SMC_HELPER = Path("bayesfilter/testing/annealed_smc_tf.py")
AIS_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_ais_repair_2026_08_10.py"
AIS_MATERIAL = Path("docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r3/material.json")
OUTPUT_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r1")
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "canary.json"

PARTICLES = 100
WORKERS = 25
ROWS_PER_WORKER = 4
CORES_PER_WORKER = 4
TARGET_ESS_FRACTION = 0.80
BISECTION_ITERATIONS = 24
BETA_TOLERANCE = 1.0e-6
MAX_STAGES = 24
HMC_STEP_SIZE = 0.03
HMC_LEAPFROG = 4
RUNNER_CAP_SECONDS = 3500.0
STAGE_TIMEOUT_SECONDS = 900.0
AIS_MATERIAL_SHA256 = "1c95aa6712dd08567a7cd2b51ada5755a3de14f5ea7f50a054de5a25abac79ff"
SEED_DOMAIN_OFFSET = 0

_WORKER_CACHE: dict[str, Any] | None = None


class SMCCanaryError(RuntimeError):
    """Raised when the SMC canary cannot preserve valid evidence."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _load_ais_runner() -> Any:
    name = "ssl_lstm_q20_physical_ais_runner_for_smc"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, AIS_RUNNER)
    if spec is None or spec.loader is None:
        raise SMCCanaryError("cannot load AIS-bound target/chart runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _worker_context(task: Mapping[str, Any]) -> Mapping[str, Any]:
    global _WORKER_CACHE
    cpu_ids = tuple(int(value) for value in task["cpu_ids"])
    os.sched_setaffinity(0, set(cpu_ids))
    if _WORKER_CACHE is not None:
        return _WORKER_CACHE

    ais = _load_ais_runner()
    ais._worker_environment()
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(CORES_PER_WORKER)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise SMCCanaryError("SMC CPU worker found a visible GPU")
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )
    from bayesfilter.testing.annealed_smc_tf import make_bridge_hmc_step
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob,
        sample_gaussian_mixture,
    )

    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    if (
        target.target_signature() != ais.TARGET_SIGNATURE
        or target.adapter_signature() != ais.ADAPTER_SIGNATURE
    ):
        raise SMCCanaryError("SMC worker target identity mismatch")
    proposal = task["proposal"]
    means = tf.constant(proposal["means"], tf.float64)
    precisions = tf.constant(proposal["precisions"], tf.float64)
    covariances = tf.linalg.inv(precisions)
    probabilities = tf.constant((0.5, 0.5), tf.float64)
    center = tf.reduce_mean(means, axis=0)
    displacement = means - center
    pooled = tf.reduce_mean(covariances, axis=0) + tf.einsum(
        "ni,nj->ij", displacement, displacement
    ) / 2.0
    eigenvalues, eigenvectors = tf.linalg.eigh(pooled)
    factor = tf.matmul(
        eigenvectors * tf.sqrt(eigenvalues)[tf.newaxis, :],
        eigenvectors,
        transpose_b=True,
    )
    log_jacobian = tf.reduce_sum(tf.math.log(eigenvalues)) / 2.0

    def proposal_log_prob(z: tf.Tensor) -> tf.Tensor:
        theta = center + tf.matmul(z, factor, transpose_b=True)
        return (
            gaussian_mixture_log_prob(
                theta, probabilities, means, covariances
            )
            + log_jacobian
        )

    def target_log_prob(z: tf.Tensor) -> tf.Tensor:
        theta = center + tf.matmul(z, factor, transpose_b=True)
        value, _score, status = target.neutra_batch_log_prob_and_grad_status(theta)
        valid = tf.logical_and(
            tf.convert_to_tensor(status["status_code"], tf.int32) == 0,
            tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
        )
        return tf.where(
            valid,
            tf.convert_to_tensor(value, tf.float64) + log_jacobian,
            tf.fill(tf.shape(value), tf.constant(float("nan"), tf.float64)),
        )

    hmc_step = make_bridge_hmc_step(
        proposal_log_prob,
        target_log_prob,
        path_count=ROWS_PER_WORKER,
        dimension=ais.PARAMETER_DIM,
        step_size=HMC_STEP_SIZE,
        num_leapfrog_steps=HMC_LEAPFROG,
        jit_compile=True,
    )
    _WORKER_CACHE = {
        "ais": ais,
        "tf": tf,
        "target": target,
        "means": means,
        "covariances": covariances,
        "probabilities": probabilities,
        "center": center,
        "factor": factor,
        "proposal_log_prob": proposal_log_prob,
        "target_log_prob": target_log_prob,
        "sample_gaussian_mixture": sample_gaussian_mixture,
        "hmc_step": hmc_step,
    }
    return _WORKER_CACHE


def _serialize_evaluation(context: Mapping[str, Any], z: Any) -> Mapping[str, bytes]:
    tf = context["tf"]
    target = context["target"]
    state = tf.convert_to_tensor(z, tf.float64)
    theta = context["center"] + tf.matmul(state, context["factor"], transpose_b=True)
    proposal = context["proposal_log_prob"](state)
    value, _score, status = target.neutra_batch_log_prob_and_grad_status(theta)
    target_value = tf.convert_to_tensor(value, tf.float64) + tf.reduce_sum(
        tf.math.log(tf.linalg.eigvalsh(context["factor"]))
    )
    valid = tf.logical_and(
        tf.convert_to_tensor(status["status_code"], tf.int32) == 0,
        tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
    )

    def encode(value: Any) -> bytes:
        return bytes(tf.io.serialize_tensor(value).numpy())

    return {
        "z": encode(state),
        "theta": encode(theta),
        "proposal_log_prob": encode(proposal),
        "target_log_prob": encode(target_value),
        "status_code": encode(status["status_code"]),
        "valid": encode(valid),
        "sign": encode(theta[:, 2] < 0.0),
    }


def _worker_task(task: Mapping[str, Any]) -> Mapping[str, Any]:
    context = _worker_context(task)
    tf = context["tf"]
    started = time.perf_counter()
    action = str(task["action"])
    if action == "initialize":
        theta, labels = context["sample_gaussian_mixture"](
            ROWS_PER_WORKER,
            context["probabilities"],
            context["means"],
            context["covariances"],
            seed=tuple(int(value) for value in task["seed"]),
        )
        z = tf.transpose(
            tf.linalg.solve(
                context["factor"], tf.transpose(theta - context["center"])
            )
        )
        output = dict(_serialize_evaluation(context, z))
        output["component_labels"] = bytes(tf.io.serialize_tensor(labels).numpy())
        output["is_accepted"] = bytes(
            tf.io.serialize_tensor(tf.ones(ROWS_PER_WORKER, tf.bool)).numpy()
        )
        output["log_accept_ratio"] = bytes(
            tf.io.serialize_tensor(tf.zeros(ROWS_PER_WORKER, tf.float64)).numpy()
        )
        proposed_finite = tf.ones(ROWS_PER_WORKER, tf.bool)
    elif action == "mutate":
        state = tf.io.parse_tensor(task["z"], out_type=tf.float64)
        state = tf.ensure_shape(state, [ROWS_PER_WORKER, 4])
        mutation = context["hmc_step"](
            state,
            tf.constant(float(task["beta"]), tf.float64),
            tf.constant(task["seed"], tf.int32),
        )
        output = dict(_serialize_evaluation(context, mutation["state"]))
        output["is_accepted"] = bytes(
            tf.io.serialize_tensor(mutation["is_accepted"]).numpy()
        )
        output["log_accept_ratio"] = bytes(
            tf.io.serialize_tensor(mutation["log_accept_ratio"]).numpy()
        )
        proposed_finite = tf.math.is_finite(mutation["proposed_target_log_prob"])
    else:
        raise SMCCanaryError(f"unknown worker action: {action}")
    output.update(
        {
            "worker_index": int(task["worker_index"]),
            "pid": os.getpid(),
            "cpu_ids": tuple(task["cpu_ids"]),
            "actual_cpu_ids": tuple(sorted(os.sched_getaffinity(0))),
            "runtime_seconds": time.perf_counter() - started,
            "target_signature": context["target"].target_signature(),
            "adapter_signature": context["target"].adapter_signature(),
            "proposed_finite": bytes(tf.io.serialize_tensor(proposed_finite).numpy()),
        }
    )
    return output


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
        raise SMCCanaryError(f"refusing to overwrite artifact: {path}")
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(absolute)


def _write_tensor(path: Path, value: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise SMCCanaryError(f"refusing to overwrite artifact: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def _base_tasks(proposal: Mapping[str, Any], action: str) -> list[dict[str, Any]]:
    return [
        {
            "action": action,
            "worker_index": index,
            "cpu_ids": tuple(range(index * CORES_PER_WORKER, (index + 1) * CORES_PER_WORKER)),
            "proposal": proposal,
        }
        for index in range(WORKERS)
    ]


def _run_tasks(
    tasks: list[Mapping[str, Any]],
    *,
    timeout_seconds: float,
) -> list[Mapping[str, Any]]:
    context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=len(tasks),
        mp_context=context,
        max_tasks_per_child=1,
    )
    try:
        futures = [executor.submit(_worker_task, task) for task in tasks]
        done, pending = concurrent.futures.wait(
            futures,
            timeout=timeout_seconds,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        if pending:
            for future in pending:
                future.cancel()
            raise SMCCanaryError("SMC worker stage exceeded timeout")
        rows = sorted(
            (future.result() for future in done), key=lambda row: row["worker_index"]
        )
        executor.shutdown(wait=True)
        return rows
    except BaseException:
        for process in (getattr(executor, "_processes", None) or {}).values():
            if process.is_alive():
                process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        raise


def _decode_rows(rows: list[Mapping[str, Any]], tf: Any) -> Mapping[str, Any]:
    dtypes = {
        "z": tf.float64,
        "theta": tf.float64,
        "proposal_log_prob": tf.float64,
        "target_log_prob": tf.float64,
        "status_code": tf.int32,
        "valid": tf.bool,
        "sign": tf.bool,
        "is_accepted": tf.bool,
        "log_accept_ratio": tf.float64,
        "proposed_finite": tf.bool,
    }
    return {
        name: tf.concat(
            [tf.io.parse_tensor(row[name], out_type=dtype) for row in rows], axis=0
        )
        for name, dtype in dtypes.items()
    }


def _stage_receipts(prefix: str, values: Mapping[str, Any], tf: Any) -> Mapping[str, Any]:
    return {
        name: _write_tensor(OUTPUT_ROOT / f"{prefix}-{name}.tftensor", value, tf)
        for name, value in values.items()
    }


def run_canary() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise SMCCanaryError("refusing to overwrite SMC canary")
    if _sha(AIS_MATERIAL) != AIS_MATERIAL_SHA256:
        raise SMCCanaryError("AIS comparator identity mismatch")
    ais = _load_ais_runner()
    proposal = ais._proposal_payload()
    try:
        tasks = _base_tasks(proposal, "initialize")
        for index, task in enumerate(tasks):
            task["seed"] = (20260810, SEED_DOMAIN_OFFSET + 21000 + index)
        initial_rows = _run_tasks(tasks, timeout_seconds=STAGE_TIMEOUT_SECONDS)

        for row in initial_rows:
            if tuple(row["actual_cpu_ids"]) != tuple(row["cpu_ids"]):
                raise SMCCanaryError("SMC initialization worker affinity mismatch")

        # Workers are fully spawned and initialized before coordinator TF import.
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.testing.annealed_smc_tf import (
            normalized_weight_diagnostics,
            select_next_beta,
            systematic_resample_indices,
        )

        values = _decode_rows(initial_rows, tf)
        component_labels = tf.concat(
            [
                tf.io.parse_tensor(row["component_labels"], out_type=tf.int32)
                for row in initial_rows
            ],
            axis=0,
        )
        roots = tf.range(PARTICLES, dtype=tf.int32)
        root_signs = tf.identity(values["sign"])
        log_weights = tf.zeros(PARTICLES, tf.float64)
        beta = tf.constant(0.0, tf.float64)
        log_normalizer = tf.constant(0.0, tf.float64)
        stage_rows: list[Mapping[str, Any]] = []
        total_resampling = 0
        total_sign_changes = 0
        initial_receipts = _stage_receipts(
            "initial",
            {
                **values,
                "component_labels": component_labels,
                "roots": roots,
                "root_signs": root_signs,
            },
            tf,
        )
        _write_json(
            PROGRESS,
            {"status": "SMC_CANARY_RUNNING", "completed_stages": 0, "beta": 0.0},
            overwrite=True,
        )

        for stage_index in range(MAX_STAGES):
            if time.perf_counter() - started >= RUNNER_CAP_SECONDS:
                raise SMCCanaryError("SMC canary reached runner wall cap")
            log_ratio = values["target_log_prob"] - values["proposal_log_prob"]
            selection = select_next_beta(
                beta,
                log_ratio,
                log_weights,
                target_ess_fraction=TARGET_ESS_FRACTION,
                bisection_iterations=BISECTION_ITERATIONS,
                beta_tolerance=BETA_TOLERANCE,
            )
            next_beta = tf.convert_to_tensor(selection["next_beta"], tf.float64)
            delta = next_beta - beta
            if not bool((delta > 0.0).numpy()):
                raise SMCCanaryError("adaptive beta selection stalled")
            incremental = delta * log_ratio
            log_weights = log_weights + incremental
            diagnostics = normalized_weight_diagnostics(log_weights)
            log_normalizer = log_normalizer + tf.reduce_logsumexp(incremental) - tf.math.log(
                tf.cast(PARTICLES, tf.float64)
            )
            terminal = bool((next_beta >= 1.0 - BETA_TOLERANCE).numpy())
            pre = {
                "z": values["z"],
                "theta": values["theta"],
                "proposal_log_prob": values["proposal_log_prob"],
                "target_log_prob": values["target_log_prob"],
                "sign": values["sign"],
                "roots": roots,
                "root_signs": tf.gather(root_signs, roots),
                "log_weights": log_weights,
                "normalized_weights": diagnostics["normalized_weights"],
            }
            prefix = f"stage-{stage_index:02d}"
            pre_receipts = dict(_stage_receipts(f"{prefix}-pre", pre, tf))
            stage_payload: dict[str, Any] = {
                "schema": "bayesfilter.ssl_lstm.q20_physical_annealed_smc.stage.v2",
                "stage_index": stage_index,
                "previous_beta": beta,
                "beta": next_beta,
                "delta_beta": delta,
                "terminal_pre_resampling": terminal,
                "pre_resampling_ess_fraction": diagnostics["effective_sample_size_fraction"],
                "pre_resampling_maximum_weight": diagnostics["maximum_normalized_weight"],
                "log_normalizer": log_normalizer,
                "receipts": {"pre": pre_receipts, "post": {}},
            }
            if terminal:
                stage_payload.update(
                    {
                        "resampled": False,
                        "unique_root_count": tf.size(tf.unique(roots).y),
                        "surviving_positive_root_count": tf.size(
                            tf.unique(tf.boolean_mask(roots, tf.logical_not(tf.gather(root_signs, roots)))).y
                        ),
                        "surviving_negative_root_count": tf.size(
                            tf.unique(tf.boolean_mask(roots, tf.gather(root_signs, roots))).y
                        ),
                    }
                )
                _write_json(OUTPUT_ROOT / f"{prefix}.json", stage_payload)
                stage_rows.append(stage_payload)
                beta = tf.constant(1.0, tf.float64)
                _write_json(
                    PROGRESS,
                    {
                        "status": "SMC_CANARY_AGGREGATING",
                        "completed_stages": len(stage_rows),
                        "beta": 1.0,
                    },
                    overwrite=True,
                )
                break

            parents = systematic_resample_indices(
                diagnostics["normalized_log_weights"],
                seed=(20260810, SEED_DOMAIN_OFFSET + 22000 + stage_index),
            )
            if (
                int(tf.reduce_min(parents).numpy()) < 0
                or int(tf.reduce_max(parents).numpy()) >= PARTICLES
            ):
                raise SMCCanaryError("global resampling indices out of bounds")
            resampled_z = tf.gather(values["z"], parents)
            resampled_roots = tf.gather(roots, parents)
            resampled_signs = tf.gather(values["sign"], parents)
            tasks = _base_tasks(proposal, "mutate")
            for worker_index, task in enumerate(tasks):
                shard = resampled_z[
                    worker_index * ROWS_PER_WORKER : (worker_index + 1) * ROWS_PER_WORKER
                ]
                task.update(
                    {
                        "z": bytes(tf.io.serialize_tensor(shard).numpy()),
                        "beta": float(next_beta.numpy()),
                        "seed": (
                            20260810,
                            SEED_DOMAIN_OFFSET
                            + 23000
                            + stage_index * WORKERS
                            + worker_index,
                        ),
                    }
                )
            mutation_rows = _run_tasks(tasks, timeout_seconds=STAGE_TIMEOUT_SECONDS)
            for row in mutation_rows:
                if tuple(row["actual_cpu_ids"]) != tuple(row["cpu_ids"]):
                    raise SMCCanaryError("SMC mutation worker affinity mismatch")
            mutated = _decode_rows(mutation_rows, tf)
            valid = tf.logical_and(
                mutated["valid"],
                tf.logical_and(
                    tf.math.is_finite(mutated["target_log_prob"]),
                    tf.math.is_finite(mutated["proposal_log_prob"]),
                ),
            )
            if not bool(tf.reduce_all(valid).numpy()):
                raise SMCCanaryError("SMC mutation produced an invalid target state")
            sign_changes = tf.reduce_sum(
                tf.cast(resampled_signs != mutated["sign"], tf.int32)
            )
            total_sign_changes += int(sign_changes.numpy())
            total_resampling += 1
            roots = resampled_roots
            values = mutated
            log_weights = tf.zeros(PARTICLES, tf.float64)
            post = {
                **mutated,
                "parents": parents,
                "roots": roots,
                "resampled_signs": resampled_signs,
            }
            stage_payload["receipts"]["post"] = dict(
                _stage_receipts(f"{prefix}-post", post, tf)
            )
            stage_payload.update(
                {
                    "resampled": True,
                    "unique_parent_count": tf.size(tf.unique(parents).y),
                    "unique_root_count": tf.size(tf.unique(roots).y),
                    "surviving_positive_root_count": tf.size(
                        tf.unique(tf.boolean_mask(roots, tf.logical_not(tf.gather(root_signs, roots)))).y
                    ),
                    "surviving_negative_root_count": tf.size(
                        tf.unique(tf.boolean_mask(roots, tf.gather(root_signs, roots))).y
                    ),
                    "mean_hmc_acceptance": tf.reduce_mean(tf.cast(mutated["is_accepted"], tf.float64)),
                    "hmc_sign_changes": sign_changes,
                    "nonfinite_hmc_proposal_count": tf.reduce_sum(
                        tf.cast(tf.logical_not(mutated["proposed_finite"]), tf.int32)
                    ),
                    "worker_runtime_max_seconds": max(
                        float(row["runtime_seconds"]) for row in mutation_rows
                    ),
                }
            )
            _write_json(OUTPUT_ROOT / f"{prefix}.json", stage_payload)
            stage_rows.append(stage_payload)
            beta = next_beta
            _write_json(
                PROGRESS,
                {
                    "status": "SMC_CANARY_RUNNING",
                    "completed_stages": len(stage_rows),
                    "beta": beta,
                    "unique_roots": stage_payload["unique_root_count"],
                    "positive_roots": stage_payload["surviving_positive_root_count"],
                    "negative_roots": stage_payload["surviving_negative_root_count"],
                    "elapsed_seconds": time.perf_counter() - started,
                },
                overwrite=True,
            )

        if not bool((beta >= 1.0 - BETA_TOLERANCE).numpy()):
            raise SMCCanaryError("SMC canary did not reach beta one")
        terminal_stage = stage_rows[-1]
        positive_roots = int(_safe(terminal_stage["surviving_positive_root_count"]))
        negative_roots = int(_safe(terminal_stage["surviving_negative_root_count"]))
        all_valid = bool(
            tf.reduce_all(
                tf.logical_and(
                    values["valid"],
                    tf.logical_and(
                        tf.math.is_finite(values["target_log_prob"]),
                        tf.math.is_finite(values["proposal_log_prob"]),
                    ),
                )
            ).numpy()
        )
        gates = {
            "reached_beta_one": True,
            "all_states_and_targets_valid": all_valid,
            "at_least_one_global_resampling": total_resampling >= 1,
            "positive_root_ancestry_survives": positive_roots >= 1,
            "negative_root_ancestry_survives": negative_roots >= 1,
            "stage_count_at_most_24": len(stage_rows) <= MAX_STAGES,
            "wall_time_within_3600_seconds": time.perf_counter() - started <= 3600.0,
        }
        passed = all(gates.values())
        payload = {
            "schema": "bayesfilter.ssl_lstm.q20_physical_annealed_smc.canary.v1",
            "status": "SMC_CANARY_PASSED" if passed else "SMC_CANARY_FAILED",
            "role": "adaptive_global_resampling_ancestry_mechanics_and_timing_canary",
            "configuration": {
                "particles": PARTICLES,
                "workers": WORKERS,
                "rows_per_worker": ROWS_PER_WORKER,
                "cores_per_worker": CORES_PER_WORKER,
                "target_ess_fraction": TARGET_ESS_FRACTION,
                "bisection_iterations": BISECTION_ITERATIONS,
                "maximum_stages": MAX_STAGES,
                "hmc_step_size": HMC_STEP_SIZE,
                "hmc_leapfrog_steps": HMC_LEAPFROG,
                "resampling": "global_systematic_every_nonterminal_stage",
                "terminal_policy": "measure_beta_one_before_resampling",
            },
            "gates": gates,
            "stage_count": len(stage_rows),
            "beta_path": [row["beta"] for row in stage_rows],
            "global_resampling_count": total_resampling,
            "total_hmc_sign_changes": total_sign_changes,
            "terminal_pre_resampling_ess_fraction": terminal_stage["pre_resampling_ess_fraction"],
            "terminal_pre_resampling_maximum_weight": terminal_stage["pre_resampling_maximum_weight"],
            "surviving_positive_root_count": positive_roots,
            "surviving_negative_root_count": negative_roots,
            "initial_receipts": initial_receipts,
            "run_manifest": {
                "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
                "git_dirty": bool(subprocess.run(("git", "status", "--porcelain"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()),
                "command": " ".join(sys.argv),
                "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
                "python": platform.python_version(),
                "tensorflow": tf.__version__,
                "tensorflow_probability": tfp.__version__,
                "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
                "jit_compile": True,
                "cpu_ids": list(range(100)),
                "wall_time_seconds": time.perf_counter() - started,
                "artifact_root": OUTPUT_ROOT.as_posix(),
                "plan_file": PLAN.as_posix(),
                "result_file": RESULT.as_posix(),
                "source_sha256": {
                    "plan": _sha(PLAN),
                    "runner": _sha(RUNNER),
                    "smc_helper": _sha(SMC_HELPER),
                    "ais_runner": hashlib.sha256(AIS_RUNNER.read_bytes()).hexdigest(),
                    "ais_material": _sha(AIS_MATERIAL),
                },
            },
            "nonclaims": (
                "the canary does not estimate posterior mode mass",
                "surviving sign ancestry does not prove sign-changing mutation",
                "the two-mode proposal does not prove exhaustive mode discovery",
            ),
        }
        _write_json(FINAL, payload)
        _write_json(
            PROGRESS,
            {
                "status": payload["status"],
                "completed_stages": len(stage_rows),
                "beta": 1.0,
                "result": FINAL.as_posix(),
                "elapsed_seconds": time.perf_counter() - started,
            },
            overwrite=True,
        )
        return payload
    finally:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=OUTPUT_ROOT.as_posix())
    parser.add_argument("--target-ess-fraction", type=float, default=TARGET_ESS_FRACTION)
    parser.add_argument("--seed-domain-offset", type=int, default=0)
    parser.add_argument("--plan-file", default=PLAN.as_posix())
    parser.add_argument("--result-file", default=RESULT.as_posix())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = Path(args.output_root)
    if output.is_absolute() or ".." in output.parts:
        raise SMCCanaryError("output root must be a repository-relative path")
    if not 0.0 < float(args.target_ess_fraction) <= 1.0:
        raise SMCCanaryError("target ESS fraction must be in (0,1]")
    if int(args.seed_domain_offset) < 0:
        raise SMCCanaryError("seed domain offset must be nonnegative")
    OUTPUT_ROOT = output
    PROGRESS = OUTPUT_ROOT / "progress.json"
    FINAL = OUTPUT_ROOT / "canary.json"
    TARGET_ESS_FRACTION = float(args.target_ess_fraction)
    SEED_DOMAIN_OFFSET = int(args.seed_domain_offset)
    PLAN = Path(args.plan_file)
    RESULT = Path(args.result_file)
    result = run_canary()
    print(json.dumps({"status": result["status"]}, sort_keys=True))
