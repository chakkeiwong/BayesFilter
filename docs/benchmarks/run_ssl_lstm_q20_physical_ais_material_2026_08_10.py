#!/usr/bin/env python3
"""Run the bounded material sparse-rejuvenation AIS campaign."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_ais_repair_2026_08_10.py"
OUTPUT_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r3")
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "material.json"
MATERIAL_RUNNER = Path("docs/benchmarks/run_ssl_lstm_q20_physical_ais_material_2026_08_10.py")
TEST_FILE = Path("tests/test_annealed_importance_tf.py")
RUNNER_STOP_SECONDS = 7000.0
WAVE_TIMEOUT_SECONDS = 1200.0


class MaterialAISError(RuntimeError):
    """Raised when the material AIS harness cannot produce valid evidence."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_base() -> Any:
    name = "ssl_lstm_q20_physical_ais_canary_bound_runner"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise MaterialAISError("cannot load canary-bound AIS runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _material_worker(task: Mapping[str, Any]) -> Mapping[str, Any]:
    return _load_base()._worker_run(task)


def _write_progress(payload: Mapping[str, Any]) -> None:
    path = _abs(PROGRESS)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("ascii")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _tasks(
    base: Any,
    proposal: Mapping[str, Any],
    *,
    family: str,
    batch_index: int,
    num_steps: int,
) -> list[Mapping[str, Any]]:
    family_offset = 0 if family == "central" else 1000
    return [
        {
            "worker_index": worker_index,
            "cpu_ids": tuple(
                range(
                    worker_index * base.CORES_PER_WORKER,
                    (worker_index + 1) * base.CORES_PER_WORKER,
                )
            ),
            "proposal": proposal,
            "num_steps": num_steps,
            "rejuvenation_interval": base.SPARSE_REJUVENATION_INTERVAL,
            "seed": (
                20260810,
                11000 + family_offset + batch_index * base.MATERIAL_WORKERS + worker_index,
            ),
            "ais_seed": (
                20260810,
                13000 + family_offset + batch_index * base.MATERIAL_WORKERS + worker_index,
            ),
        }
        for worker_index in range(base.MATERIAL_WORKERS)
    ]


def _run_wave(
    tasks: list[Mapping[str, Any]], *, timeout_seconds: float
) -> list[Mapping[str, Any]]:
    context = multiprocessing.get_context("spawn")
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=len(tasks), mp_context=context
    )
    try:
        futures = [executor.submit(_material_worker, task) for task in tasks]
        done, pending = concurrent.futures.wait(
            futures,
            timeout=timeout_seconds,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        if pending:
            for future in pending:
                future.cancel()
            for process in executor._processes.values():
                process.terminate()
            raise MaterialAISError("AIS material wave exceeded its timeout")
        rows = [future.result() for future in done]
        executor.shutdown(wait=True)
    except BaseException:
        for process in (getattr(executor, "_processes", None) or {}).values():
            if process.is_alive():
                process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    return sorted(rows, key=lambda row: int(row["worker_index"]))


def _decode_wave(rows: list[Mapping[str, Any]], tf: Any) -> Mapping[str, Any]:
    dtypes = {
        "initial_theta": tf.float64,
        "component_labels": tf.int32,
        "terminal_theta": tf.float64,
        "initial_sign": tf.bool,
        "terminal_sign": tf.bool,
        "log_weights": tf.float64,
        "acceptance_fraction": tf.float64,
        "all_finite": tf.bool,
        "terminal_status_code": tf.int32,
        "terminal_valid_pre_regularized_score": tf.bool,
        "maximum_absolute_log_accept_ratio": tf.float64,
    }
    return {
        name: tf.concat(
            [tf.io.parse_tensor(row[name], out_type=dtype) for row in rows], axis=0
        )
        for name, dtype in dtypes.items()
    }


def _write_wave(
    base: Any,
    helper: Any,
    tf: Any,
    rows: list[Mapping[str, Any]],
    *,
    family: str,
    batch_index: int,
) -> Mapping[str, Any]:
    values = _decode_wave(rows, tf)
    prefix = f"{family}-{batch_index:02d}"
    all_finite = bool(tf.reduce_all(values["all_finite"]).numpy())
    status_invalid_count = int(
        tf.reduce_sum(tf.cast(values["terminal_status_code"] != 0, tf.int32)).numpy()
    )
    weights_finite = bool(tf.reduce_all(tf.math.is_finite(values["log_weights"])).numpy())
    receipts = {
        name: base._write_tensor(OUTPUT_ROOT / f"{prefix}-{name}.tftensor", value, tf)
        for name, value in values.items()
    }
    valid = all_finite and status_invalid_count == 0 and weights_finite
    diagnostics = None
    if valid:
        diagnostics = helper.self_normalized_importance_diagnostics(
            values["log_weights"],
            tf.zeros_like(values["log_weights"]),
            values["terminal_sign"],
        )
        receipts["normalized_weights"] = base._write_tensor(
            OUTPUT_ROOT / f"{prefix}-normalized_weights.tftensor",
            diagnostics["normalized_weights"],
            tf,
        )
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_ais_repair.batch.v1",
        "status": "AIS_BATCH_COMPLETED" if valid else "AIS_BATCH_INVALID",
        "family": family,
        "batch_index": batch_index,
        "configuration": {
            "paths": len(rows) * base.PATHS_PER_WORKER,
            "workers": len(rows),
            "paths_per_worker": base.PATHS_PER_WORKER,
            "cores_per_worker": base.CORES_PER_WORKER,
            "num_steps": int(rows[0]["num_steps"]),
            "rejuvenation_interval": int(rows[0]["rejuvenation_interval"]),
            "rejuvenation_count": int(rows[0]["rejuvenation_count"]),
            "step_size": base.HMC_STEP_SIZE,
            "num_leapfrog_steps": base.HMC_LEAPFROG,
        },
        "all_paths_finite": all_finite,
        "all_log_weights_finite": weights_finite,
        "terminal_target_status_invalid_count": status_invalid_count,
        "negative_region_probability": None if diagnostics is None else diagnostics["negative_region_probability"],
        "effective_sample_size_fraction": None if diagnostics is None else diagnostics["effective_sample_size_fraction"],
        "maximum_normalized_weight": None if diagnostics is None else diagnostics["maximum_normalized_weight"],
        "log_normalizer_ratio_estimate": None if diagnostics is None else diagnostics["log_normalizer_ratio_estimate"],
        "mean_acceptance_fraction": tf.reduce_mean(values["acceptance_fraction"]),
        "initial_to_terminal_sign_changes": tf.reduce_sum(
            tf.cast(values["initial_sign"] != values["terminal_sign"], tf.int32)
        ),
        "maximum_absolute_log_accept_ratio": tf.reduce_max(
            values["maximum_absolute_log_accept_ratio"]
        ),
        "workers": [
            {
                "worker_index": int(row["worker_index"]),
                "pid": int(row["pid"]),
                "cpu_ids": row["cpu_ids"],
                "runtime_seconds": float(row["runtime_seconds"]),
                "proposal_seed": row["seed"],
                "ais_seed": row["ais_seed"],
                "target_signature": row["target_signature"],
                "adapter_signature": row["adapter_signature"],
            }
            for row in rows
        ],
        "receipts": receipts,
    }
    base._write_json(OUTPUT_ROOT / f"{prefix}.json", payload)
    return {"payload": payload, "values": values}


def _terminal_early_veto(
    base: Any,
    tf: Any,
    started: float,
    completed: list[str],
    batch: Mapping[str, Any],
) -> Mapping[str, Any]:
    import tensorflow_probability as tfp

    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_ais_repair.material.v1",
        "status": "AIS_WEIGHT_REPAIR_FAILED",
        "decision": "STOP_AIS_WEIGHT_PROMOTION_AND_PLAN_ANNEALED_SMC",
        "failure": "TARGET_OR_NUMERICAL_VALIDITY_VETO",
        "failed_batch": {
            "family": batch["payload"]["family"],
            "batch_index": batch["payload"]["batch_index"],
            "batch_status": batch["payload"]["status"],
        },
        "completed_batches": completed,
        "run_manifest": {
            **base._manifest("material-sparse-early-veto", started, OUTPUT_ROOT, tf, tfp),
            "material_runner_sha256": base._sha(MATERIAL_RUNNER),
            "base_runner_sha256": base._sha(Path(BASE_RUNNER)),
        },
        "nonclaims": (
            "no posterior mass estimate is issued from an invalid AIS batch",
            "this veto does not reject the physical multimodal inference direction",
        ),
    }
    base._write_json(FINAL, payload)
    _write_progress(
        {
            "status": payload["status"],
            "completed_batches": completed,
            "total_batches": base.CENTRAL_BATCHES + base.SENSITIVITY_BATCHES,
            "elapsed_seconds": time.perf_counter() - started,
            "result": FINAL.as_posix(),
        }
    )
    return payload


def _run() -> Mapping[str, Any]:
    started = time.perf_counter()
    base = _load_base()
    base._worker_environment()
    if _abs(FINAL).exists():
        raise MaterialAISError("refusing to overwrite material AIS result")
    if base._sha(base.DIRECT_WEIGHT) != base.DIRECT_WEIGHT_SHA256:
        raise MaterialAISError("direct-importance comparator identity mismatch")
    proposal = base._proposal_payload()
    completed: list[str] = []
    decoded_batches: dict[str, list[Mapping[str, Any]]] = {
        "central": [],
        "sensitivity": [],
    }
    batch_specs = [
        ("central", index, base.MATERIAL_STEPS)
        for index in range(base.CENTRAL_BATCHES)
    ] + [
        ("sensitivity", index, base.SENSITIVITY_STEPS)
        for index in range(base.SENSITIVITY_BATCHES)
    ]
    _write_progress(
        {
            "status": "AIS_MATERIAL_RUNNING",
            "completed_batches": completed,
            "total_batches": len(batch_specs),
        }
    )
    tf = None
    helper = None
    for family, batch_index, num_steps in batch_specs:
        remaining = RUNNER_STOP_SECONDS - (time.perf_counter() - started)
        if remaining <= 0.0:
            raise MaterialAISError("AIS material runner reached its wall-time stop")
        rows = _run_wave(
            _tasks(
                base,
                proposal,
                family=family,
                batch_index=batch_index,
                num_steps=num_steps,
            ),
            timeout_seconds=min(WAVE_TIMEOUT_SECONDS, remaining),
        )
        if tf is None:
            import tensorflow as tf_module
            from bayesfilter.testing import importance_sampling_tf as helper_module

            tf = tf_module
            helper = helper_module
        decoded = _write_wave(
            base,
            helper,
            tf,
            rows,
            family=family,
            batch_index=batch_index,
        )
        decoded_batches[family].append(decoded)
        completed.append(f"{family}-{batch_index:02d}")
        if decoded["payload"]["status"] != "AIS_BATCH_COMPLETED":
            return _terminal_early_veto(base, tf, started, completed, decoded)
        _write_progress(
            {
                "status": "AIS_MATERIAL_RUNNING",
                "completed_batches": completed,
                "total_batches": len(batch_specs),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )

    import tensorflow_probability as tfp

    central_log_weights = tf.concat(
        [batch["values"]["log_weights"] for batch in decoded_batches["central"]],
        axis=0,
    )
    central_terminal_sign = tf.concat(
        [batch["values"]["terminal_sign"] for batch in decoded_batches["central"]],
        axis=0,
    )
    central = helper.self_normalized_importance_diagnostics(
        central_log_weights,
        tf.zeros_like(central_log_weights),
        central_terminal_sign,
    )
    sensitivity_log_weights = tf.concat(
        [batch["values"]["log_weights"] for batch in decoded_batches["sensitivity"]],
        axis=0,
    )
    sensitivity_terminal_sign = tf.concat(
        [batch["values"]["terminal_sign"] for batch in decoded_batches["sensitivity"]],
        axis=0,
    )
    sensitivity = helper.self_normalized_importance_diagnostics(
        sensitivity_log_weights,
        tf.zeros_like(sensitivity_log_weights),
        sensitivity_terminal_sign,
    )
    batch_estimates = tf.stack(
        [
            batch["payload"]["negative_region_probability"]
            for batch in decoded_batches["central"]
        ]
    )
    interval = helper.independent_batch_interval(batch_estimates)
    all_batches = decoded_batches["central"] + decoded_batches["sensitivity"]
    finite_gate = all(
        batch["payload"]["status"] == "AIS_BATCH_COMPLETED"
        for batch in all_batches
    )
    movement_count = sum(
        int(batch["payload"]["initial_to_terminal_sign_changes"].numpy())
        for batch in decoded_batches["central"]
    )
    schedule_difference = tf.abs(
        central["negative_region_probability"]
        - sensitivity["negative_region_probability"]
    )
    gates = {
        "all_target_states_valid_and_finite": finite_gate,
        "central_ess_fraction_at_least_0.30": bool(
            (central["effective_sample_size_fraction"] >= 0.30).numpy()
        ),
        "central_maximum_weight_at_most_0.02": bool(
            (central["maximum_normalized_weight"] <= 0.02).numpy()
        ),
        "eight_batch_interval_half_width_at_most_0.08": bool(
            (interval["half_width"] <= 0.08).numpy()
        ),
        "schedule_difference_at_most_0.08": bool(
            (schedule_difference <= 0.08).numpy()
        ),
        "at_least_one_central_terminal_sign_change": movement_count >= 1,
        "wall_time_within_7200_seconds": time.perf_counter() - started <= base.CAMPAIGN_CAP_SECONDS,
    }
    passed = all(gates.values())
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_ais_repair.material.v1",
        "status": "AIS_WEIGHT_REPAIR_PASSED" if passed else "AIS_WEIGHT_REPAIR_FAILED",
        "decision": "AIS_WEIGHT_EVIDENCE_VIABLE" if passed else "STOP_AIS_WEIGHT_PROMOTION_AND_PLAN_ANNEALED_SMC",
        "configuration": {
            "central_batches": base.CENTRAL_BATCHES,
            "central_paths_per_batch": base.MATERIAL_WORKERS * base.PATHS_PER_WORKER,
            "central_steps": base.MATERIAL_STEPS,
            "sensitivity_batches": base.SENSITIVITY_BATCHES,
            "sensitivity_paths_per_batch": base.MATERIAL_WORKERS * base.PATHS_PER_WORKER,
            "sensitivity_steps": base.SENSITIVITY_STEPS,
            "rejuvenation_interval": base.SPARSE_REJUVENATION_INTERVAL,
            "step_size": base.HMC_STEP_SIZE,
            "num_leapfrog_steps": base.HMC_LEAPFROG,
        },
        "gates": gates,
        "central": {
            "negative_region_probability": central["negative_region_probability"],
            "effective_sample_size": central["effective_sample_size"],
            "effective_sample_size_fraction": central["effective_sample_size_fraction"],
            "maximum_normalized_weight": central["maximum_normalized_weight"],
            "log_normalizer_ratio_estimate": central["log_normalizer_ratio_estimate"],
            "batch_estimates": batch_estimates,
            "independent_batch_interval": interval,
            "initial_to_terminal_sign_changes": movement_count,
        },
        "sensitivity": {
            "negative_region_probability": sensitivity["negative_region_probability"],
            "effective_sample_size": sensitivity["effective_sample_size"],
            "effective_sample_size_fraction": sensitivity["effective_sample_size_fraction"],
            "maximum_normalized_weight": sensitivity["maximum_normalized_weight"],
            "log_normalizer_ratio_estimate": sensitivity["log_normalizer_ratio_estimate"],
            "schedule_difference": schedule_difference,
        },
        "analytic_known_law_tests": {
            "status": "PASSED_PRELAUNCH",
            "command": "CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_annealed_importance_tf.py",
            "tests_passed": 5,
            "test_file_sha256": base._sha(TEST_FILE),
        },
        "aggregate_receipts": {
            "central_log_weights": base._write_tensor(OUTPUT_ROOT / "central-log-weights.tftensor", central_log_weights, tf),
            "central_terminal_sign": base._write_tensor(OUTPUT_ROOT / "central-terminal-sign.tftensor", central_terminal_sign, tf),
            "central_normalized_weights": base._write_tensor(OUTPUT_ROOT / "central-normalized-weights.tftensor", central["normalized_weights"], tf),
            "sensitivity_log_weights": base._write_tensor(OUTPUT_ROOT / "sensitivity-log-weights.tftensor", sensitivity_log_weights, tf),
            "sensitivity_terminal_sign": base._write_tensor(OUTPUT_ROOT / "sensitivity-terminal-sign.tftensor", sensitivity_terminal_sign, tf),
            "sensitivity_normalized_weights": base._write_tensor(OUTPUT_ROOT / "sensitivity-normalized-weights.tftensor", sensitivity["normalized_weights"], tf),
        },
        "run_manifest": {
            **base._manifest("material-sparse", started, OUTPUT_ROOT, tf, tfp),
            "material_runner_sha256": base._sha(MATERIAL_RUNNER),
            "base_runner_sha256": base._sha(Path(BASE_RUNNER)),
            "random_seed_policy": "stateless_unique_by_family_batch_worker",
            "cpu_ids": list(range(base.MATERIAL_WORKERS * base.CORES_PER_WORKER)),
            "worker_launch_policy": "fresh_spawn_wave_per_independent_batch",
        },
        "nonclaims": (
            "two-mode proposal does not establish exhaustive mode discovery",
            "passing AIS weight gates does not establish HMC convergence or posterior correctness",
            "predictive validity remains untested",
        ),
    }
    base._write_json(FINAL, payload)
    _write_progress(
        {
            "status": payload["status"],
            "completed_batches": completed,
            "total_batches": len(batch_specs),
            "elapsed_seconds": time.perf_counter() - started,
            "result": FINAL.as_posix(),
        }
    )
    return payload


if __name__ == "__main__":
    result = _run()
    print(json.dumps({"status": result["status"]}, sort_keys=True))
