#!/usr/bin/env python3
"""Compare one exact replica transition across one-row and two-row shards."""

from __future__ import annotations

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
    "docs/plans/bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-plan-2026-08-10.md"
)
RESULT = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-result-2026-08-10.md"
)
RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_12x2_numerical_materiality_canary_2026_08_11.py"
)
CHECKPOINT_RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_checkpoint_2026_08_10.py"
)
HELPER = Path("bayesfilter/testing/distributed_replica_exchange_tf.py")
POOL_HELPER = Path("bayesfilter/inference/tf_batch_value_score_pool.py")
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/"
    "r1/geometry.json"
)
FAILED_CANARY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r5-topology-12x2-canary/canary.json"
)
PAIR_DIAGNOSIS = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r5-cache-parity-diagnosis/diagnosis.json"
)
R6_FAILURE = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r6-12x2-numerical-materiality-canary/canary.json"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r7-12x2-numerical-materiality-canary"
)
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "canary.json"

ROWS = 24
DIMENSION = 4
ONE_ROW_WORKERS = 24
TWO_ROW_WORKERS = 12
ONE_ROW_CPU_IDS = tuple(range(32, 56))
TWO_ROW_CPU_IDS = tuple(range(32, 44))
PARENT_CPU_IDS = tuple(range(32, 64))
MASTER_SEED = (20260810, 7301)
TRANSITION_INDEX = 0
MATERIAL_TRANSITIONS = 1300
MATERIAL_MARGIN = 1.5
MATERIAL_CAP_SECONDS = 20000.0
CAP_SECONDS = 360.0
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
FAILED_CANARY_SHA256 = "08e9d29fee2af56aeadc3622f01a6f97487384c4446e01f16fc00dedb2ecb3ac"
PAIR_DIAGNOSIS_SHA256 = "1a29bd118fb75481aa86dde0dd6a3353d4f7b729b6e9c6cf0bf55ac2e5774363"
R6_FAILURE_SHA256 = "408645656995f123f334a4c92e1c8eb779cd9dd2633540a89e3029b0cd93caa9"


class NumericalMaterialityCanaryError(RuntimeError):
    """Raised when the bounded comparison cannot produce valid evidence."""


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
        raise NumericalMaterialityCanaryError(f"refusing to overwrite {path}")
    encoded = json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def _load_checkpoint_runner() -> Any:
    name = "physical_checkpoint_support_for_numerical_materiality"
    spec = importlib.util.spec_from_file_location(name, _abs(CHECKPOINT_RUNNER))
    if spec is None or spec.loader is None:
        raise NumericalMaterialityCanaryError("cannot load checkpoint support")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pool_config(*, workers: int, rows_per_worker: int, cpu_ids: tuple[int, ...]) -> Any:
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
        dimension=DIMENSION,
        worker_count=workers,
        cores_per_worker=1,
        batch_sizes=(rows_per_worker,),
        batch_per_worker=rows_per_worker,
        worker_cpu_ids=cpu_ids,
        timeout_seconds=900.0,
    )


def _all_finite(tf: Any, values: tuple[Any, ...]) -> bool:
    return bool(
        tf.reduce_all(
            tf.stack(
                tuple(tf.reduce_all(tf.math.is_finite(value)) for value in values)
            )
        ).numpy()
    )


def _max_abs(tf: Any, left: Any, right: Any) -> Any:
    return tf.reduce_max(
        tf.abs(tf.convert_to_tensor(left, tf.float64) - tf.convert_to_tensor(right, tf.float64))
    )


def _max_scaled(tf: Any, candidate: Any, reference: Any) -> Any:
    candidate = tf.convert_to_tensor(candidate, tf.float64)
    reference = tf.convert_to_tensor(reference, tf.float64)
    return tf.reduce_max(tf.abs(candidate - reference) / tf.maximum(1.0, tf.abs(reference)))


def _run_route(
    *,
    tf: Any,
    pool: Any,
    support: Any,
    chart: Mapping[str, Any],
    initial_state: Any,
    route_name: str,
    pair_shift: int,
) -> Mapping[str, Any]:
    from bayesfilter.testing.distributed_replica_exchange_tf import (
        distributed_replica_exchange_transition,
        initialize_distributed_replica_state,
        leapfrog_proposal,
    )

    permutation = tf.roll(tf.range(ROWS, dtype=tf.int32), shift=-pair_shift, axis=0)
    inverse_permutation = tf.argsort(permutation)
    evaluation_seconds = []
    evaluation_valid = []
    evaluation_metadata = []

    def evaluator(rows: Any, request_id: str):
        started = time.perf_counter()
        latent = tf.ensure_shape(tf.convert_to_tensor(rows, tf.float64), (ROWS, DIMENSION))
        theta = chart["center"] + tf.matmul(latent, chart["factor"], transpose_b=True)
        if pair_shift:
            theta = tf.gather(theta, permutation)
        value, score, status, metadata = pool.evaluate_with_status(
            theta, request_id=f"{route_name}-{request_id}"
        )
        if pair_shift:
            value = tf.gather(value, inverse_permutation)
            score = tf.gather(score, inverse_permutation)
            status = {
                key: tf.gather(item, inverse_permutation) for key, item in status.items()
            }
        value = tf.convert_to_tensor(value, tf.float64) + chart["log_abs_determinant"]
        score = tf.matmul(tf.convert_to_tensor(score, tf.float64), chart["factor"])
        valid = tf.logical_and(
            tf.convert_to_tensor(status["status_code"], tf.int32) == 0,
            tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
        )
        evaluation_seconds.append(time.perf_counter() - started)
        evaluation_valid.append(bool(tf.reduce_all(valid).numpy()))
        evaluation_metadata.append(metadata)
        return value, score, status, metadata

    initialized = initialize_distributed_replica_state(initial_state, evaluator=evaluator)
    transition_started = time.perf_counter()
    transition = distributed_replica_exchange_transition(
        state=initialized["state"],
        base_target_log_prob=initialized["base_target_log_prob"],
        base_score=initialized["base_score"],
        identities_at_temperature=initialized["identities_at_temperature"],
        inverse_temperatures=support.BETAS,
        step_sizes=support.STEPS,
        num_leapfrog_steps=support.LEAPFROG,
        transition_index=TRANSITION_INDEX,
        master_seed=MASTER_SEED,
        evaluator=evaluator,
    )
    transition_seconds = time.perf_counter() - transition_started

    cache_started = time.perf_counter()
    cache_value, cache_score, _cache_status, _cache_metadata = evaluator(
        tf.reshape(transition["state"], (ROWS, DIMENSION)), "terminal-cache"
    )
    cache_seconds = time.perf_counter() - cache_started
    cache_value = tf.reshape(cache_value, (len(support.BETAS), support.CHAINS))
    cache_score = tf.reshape(cache_score, (len(support.BETAS), support.CHAINS, DIMENSION))

    endpoint_value, endpoint_score, _endpoint_status, _endpoint_metadata = evaluator(
        tf.reshape(transition["proposed_state"], (ROWS, DIMENSION)),
        "proposal-endpoint",
    )
    endpoint_value = tf.reshape(endpoint_value, (len(support.BETAS), support.CHAINS))
    endpoint_score = tf.reshape(
        endpoint_score, (len(support.BETAS), support.CHAINS, DIMENSION)
    )
    reverse = leapfrog_proposal(
        transition["proposed_state"],
        endpoint_value,
        endpoint_score,
        -transition["final_momentum"],
        inverse_temperatures=support.BETAS,
        step_sizes=support.STEPS,
        num_leapfrog_steps=support.LEAPFROG,
        evaluator=evaluator,
        request_prefix=f"{route_name}-reverse",
    )
    state_scale = tf.maximum(1.0, tf.reduce_max(tf.abs(initialized["state"])))
    momentum_scale = tf.maximum(
        1.0, tf.reduce_max(tf.abs(transition["initial_momentum"]))
    )
    estimated_cost = transition_seconds + cache_seconds / 5.0
    return {
        "route": route_name,
        "pair_shift": pair_shift,
        "initial": initialized,
        "transition": transition,
        "cache_value": cache_value,
        "cache_score": cache_score,
        "transition_seconds": transition_seconds,
        "cache_seconds": cache_seconds,
        "estimated_checkpoint_cost_per_transition_seconds": estimated_cost,
        "projected_material_seconds_with_50pct_margin": (
            estimated_cost * MATERIAL_TRANSITIONS * MATERIAL_MARGIN
        ),
        "all_evaluations_status_valid": all(evaluation_valid),
        "all_outputs_finite": _all_finite(
            tf,
            (
                initialized["base_target_log_prob"],
                initialized["base_score"],
                transition["state"],
                transition["hmc_log_accept_ratio"],
                cache_value,
                cache_score,
                reverse["proposed_state"],
                reverse["final_momentum"],
            ),
        ),
        "terminal_cache_value_max_abs_error": _max_abs(
            tf, cache_value, transition["base_target_log_prob"]
        ),
        "terminal_cache_score_max_abs_error": _max_abs(
            tf, cache_score, transition["base_score"]
        ),
        "reverse_state_max_abs_error": _max_abs(
            tf, reverse["proposed_state"], initialized["state"]
        ),
        "reverse_state_scaled_error": (
            _max_abs(tf, reverse["proposed_state"], initialized["state"]) / state_scale
        ),
        "reverse_momentum_max_abs_error": _max_abs(
            tf, reverse["final_momentum"], -transition["initial_momentum"]
        ),
        "reverse_momentum_scaled_error": (
            _max_abs(tf, reverse["final_momentum"], -transition["initial_momentum"])
            / momentum_scale
        ),
        "worker_identity": support._worker_identity(evaluation_metadata[0]),
        "evaluation_seconds": evaluation_seconds,
    }


def _comparison(tf: Any, candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Mapping[str, Any]:
    left = candidate["transition"]
    right = reference["transition"]
    probability_left = tf.exp(tf.minimum(left["hmc_log_accept_ratio"], 0.0))
    probability_right = tf.exp(tf.minimum(right["hmc_log_accept_ratio"], 0.0))
    proposed = tf.logical_and(
        left["swap_is_proposed_adjacent"], right["swap_is_proposed_adjacent"]
    )
    proposed_swap_log_error = tf.abs(
        tf.boolean_mask(left["swap_log_accept_ratio_adjacent"], proposed)
        - tf.boolean_mask(right["swap_log_accept_ratio_adjacent"], proposed)
    )
    return {
        "candidate": candidate["route"],
        "hmc_path_valid_identical": bool(
            tf.reduce_all(left["hmc_path_valid"] == right["hmc_path_valid"]).numpy()
        ),
        "hmc_accept_decisions_identical": bool(
            tf.reduce_all(left["hmc_is_accepted"] == right["hmc_is_accepted"]).numpy()
        ),
        "swap_accept_decisions_identical": bool(
            tf.reduce_all(
                left["swap_is_accepted_matrix"] == right["swap_is_accepted_matrix"]
            ).numpy()
        ),
        "initial_target_max_abs_error": _max_abs(
            tf, candidate["initial"]["base_target_log_prob"], reference["initial"]["base_target_log_prob"]
        ),
        "initial_target_max_scaled_error": _max_scaled(
            tf, candidate["initial"]["base_target_log_prob"], reference["initial"]["base_target_log_prob"]
        ),
        "initial_score_max_abs_error": _max_abs(
            tf, candidate["initial"]["base_score"], reference["initial"]["base_score"]
        ),
        "initial_score_max_scaled_error": _max_scaled(
            tf, candidate["initial"]["base_score"], reference["initial"]["base_score"]
        ),
        "proposal_state_max_abs_error": _max_abs(
            tf, left["proposed_state"], right["proposed_state"]
        ),
        "proposal_state_max_scaled_error": _max_scaled(
            tf, left["proposed_state"], right["proposed_state"]
        ),
        "log_accept_ratio_max_abs_error": _max_abs(
            tf, left["hmc_log_accept_ratio"], right["hmc_log_accept_ratio"]
        ),
        "acceptance_probability_max_abs_error": _max_abs(
            tf, probability_left, probability_right
        ),
        "swap_log_accept_ratio_max_abs_error": _max_abs(
            tf, proposed_swap_log_error, tf.zeros_like(proposed_swap_log_error)
        ),
        "retained_state_max_abs_error": _max_abs(tf, left["state"], right["state"]),
    }


def run_canary() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise NumericalMaterialityCanaryError("refusing to overwrite materiality canary")
    if tuple(sorted(os.sched_getaffinity(0))) != PARENT_CPU_IDS:
        raise NumericalMaterialityCanaryError("parent CPU affinity mismatch")
    bindings = {
        "geometry_sha256": _sha(GEOMETRY),
        "failed_12x2_canary_sha256": _sha(FAILED_CANARY),
        "pair_diagnosis_sha256": _sha(PAIR_DIAGNOSIS),
        "r6_reporting_failure_sha256": _sha(R6_FAILURE),
    }
    if bindings != {
        "geometry_sha256": GEOMETRY_SHA256,
        "failed_12x2_canary_sha256": FAILED_CANARY_SHA256,
        "pair_diagnosis_sha256": PAIR_DIAGNOSIS_SHA256,
        "r6_reporting_failure_sha256": R6_FAILURE_SHA256,
    }:
        raise NumericalMaterialityCanaryError("bound evidence identity mismatch")
    _write_json(PROGRESS, {"status": "NUMERICAL_MATERIALITY_CANARY_STARTING"}, overwrite=True)

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise NumericalMaterialityCanaryError("CPU-only canary found visible GPU")

    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool

    support = _load_checkpoint_runner()
    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    chart = support._chart(tf, geometry)
    chain_centers = tf.gather(chart["latent_centers"], (0, 1, 0, 1))
    initial_state = tf.repeat(chain_centers[tf.newaxis, :, :], len(support.BETAS), axis=0)

    with TFBatchValueScorePool(
        _pool_config(workers=ONE_ROW_WORKERS, rows_per_worker=1, cpu_ids=ONE_ROW_CPU_IDS)
    ) as pool:
        canonical = _run_route(
            tf=tf,
            pool=pool,
            support=support,
            chart=chart,
            initial_state=initial_state,
            route_name="one_row_reference",
            pair_shift=0,
        )
    _write_json(
        PROGRESS,
        {"status": "ONE_ROW_REFERENCE_COMPLETE", "elapsed_seconds": time.perf_counter() - started},
        overwrite=True,
    )
    with TFBatchValueScorePool(
        _pool_config(workers=TWO_ROW_WORKERS, rows_per_worker=2, cpu_ids=TWO_ROW_CPU_IDS)
    ) as pool:
        contiguous = _run_route(
            tf=tf,
            pool=pool,
            support=support,
            chart=chart,
            initial_state=initial_state,
            route_name="two_row_contiguous",
            pair_shift=0,
        )
        shifted = _run_route(
            tf=tf,
            pool=pool,
            support=support,
            chart=chart,
            initial_state=initial_state,
            route_name="two_row_shifted_pairing",
            pair_shift=1,
        )

    comparisons = (
        _comparison(tf, contiguous, canonical),
        _comparison(tf, shifted, canonical),
        _comparison(tf, shifted, contiguous),
    )
    validity_passed = all(
        route["all_evaluations_status_valid"] and route["all_outputs_finite"]
        for route in (canonical, contiguous, shifted)
    )
    decision_passed = all(
        row["hmc_path_valid_identical"]
        and row["hmc_accept_decisions_identical"]
        and row["swap_accept_decisions_identical"]
        for row in comparisons
    )
    # Forward/reverse error is compared with a floating-point scale rather than
    # the former uncalibrated cache tolerances.  This is a canary screen, not a
    # posterior-accuracy claim.
    reverse_scale = 100.0 * math.sqrt(sys.float_info.epsilon)
    reversibility_passed = all(
        float(route["reverse_state_scaled_error"].numpy()) <= reverse_scale
        and float(route["reverse_momentum_scaled_error"].numpy()) <= reverse_scale
        for route in (canonical, contiguous, shifted)
    )
    numerical_materiality_passed = validity_passed and decision_passed and reversibility_passed
    cost_passed = (
        contiguous["projected_material_seconds_with_50pct_margin"]
        <= MATERIAL_CAP_SECONDS
    )
    nominated = bool(numerical_materiality_passed and cost_passed)
    status = (
        "NUMERICAL_MATERIALITY_AND_COST_NOMINATED"
        if nominated
        else (
            "NUMERICAL_MATERIALITY_PASSED_COST_FAILED"
            if numerical_materiality_passed
            else "NUMERICAL_MATERIALITY_FAILED"
        )
    )

    def route_summary(route: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            key: value
            for key, value in route.items()
            if key not in ("initial", "transition", "cache_value", "cache_score")
        }

    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_12x2_numerical_materiality.v1",
        "status": status,
        "configuration": {
            "inverse_temperatures": support.BETAS,
            "step_sizes": support.STEPS,
            "num_leapfrog_steps": support.LEAPFROG,
            "chains": support.CHAINS,
            "master_seed": MASTER_SEED,
            "transition_index": TRANSITION_INDEX,
            "jit_compile": True,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        },
        "evidence_contract": {
            "primary": "identical path-validity, HMC decisions, and swap decisions under identical randomness",
            "hard_vetoes": "nonfinite/status-invalid output or forward/reverse scaled error above declared floating-point screen",
            "explanatory": "all continuous value, score, endpoint, energy, acceptance-probability, cache, and timing differences",
            "reverse_scaled_error_limit": reverse_scale,
            "reverse_limit_provenance": "100 * sqrt(binary64 machine epsilon), canary engineering screen",
            "nonclaim": "one transition does not prove invariant-measure equality, convergence, posterior validity, or performance superiority",
        },
        "gates": {
            "all_routes_finite_and_status_valid": validity_passed,
            "all_identical_randomness_decisions_match": decision_passed,
            "all_forward_reverse_errors_within_scale": reversibility_passed,
            "numerical_materiality_passed": numerical_materiality_passed,
            "contiguous_12x2_material_projection_within_20000_seconds": bool(cost_passed),
        },
        "routes": {
            "one_row_reference": route_summary(canonical),
            "two_row_contiguous": route_summary(contiguous),
            "two_row_shifted_pairing": route_summary(shifted),
        },
        "comparisons": comparisons,
        "bindings": bindings,
        "run_manifest": {
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ("git", "status", "--porcelain"), cwd=ROOT, check=True,
                    capture_output=True, text=True,
                ).stdout.strip()
            ),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "wall_seconds": time.perf_counter() - started,
            "artifact_root": OUTPUT_ROOT.as_posix(),
            "plan_file": PLAN.as_posix(),
            "result_file": RESULT.as_posix(),
            "source_sha256": {
                "runner": _sha(RUNNER),
                "checkpoint_runner": _sha(CHECKPOINT_RUNNER),
                "helper": _sha(HELPER),
                "pool_helper": _sha(POOL_HELPER),
            },
        },
        "nonclaims": (
            "no material sampling or posterior archive",
            "no convergence, exhaustive-mode, predictive, or default-readiness claim",
            "MCSE context is not used as a proof of detailed balance",
        ),
    }
    _write_json(FINAL, payload)
    _write_json(
        PROGRESS,
        {"status": status, "result": FINAL.as_posix(), "elapsed_seconds": time.perf_counter() - started},
        overwrite=True,
    )
    return payload


def main() -> None:
    started = time.perf_counter()
    try:
        payload = run_canary()
    except BaseException as error:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20_physical_12x2_numerical_materiality_failure.v1",
            "status": "NUMERICAL_MATERIALITY_CANARY_HARNESS_FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_seconds": time.perf_counter() - started,
        }
        if not _abs(FINAL).exists():
            _write_json(FINAL, failure)
        _write_json(PROGRESS, {**failure, "result": FINAL.as_posix()}, overwrite=True)
        raise
    print(json.dumps({"status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
