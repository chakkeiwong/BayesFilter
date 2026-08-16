#!/usr/bin/env python3
"""Resume the exact physical replica campaign from transition 500."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta
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
RUNNER = Path(
    "docs/benchmarks/"
    "resume_ssl_lstm_q20_physical_distributed_replica_material_2026_08_11.py"
)
CHECKPOINT_RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_checkpoint_2026_08_10.py"
)
MATERIAL_RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_material_2026_08_11.py"
)
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/"
    "r1/geometry.json"
)
R8_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r8-material-24x1-eight-hour"
)
R8_RESULT = R8_ROOT / "material.json"
R9_RATIO_0P40 = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r9-hot-tuning-ratio-0p40/canary.json"
)
R9_RATIO_0P35 = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r9-hot-tuning-ratio-0p35/canary.json"
)
R10_STEP_1P5 = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r10-hot-step-1p5/canary.json"
)
R10_STEP_2P0 = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r10-hot-step-2p0/canary.json"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r11-material-24x1-resumed"
)
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "material.json"
LOG = OUTPUT_ROOT / "run.log"

PARAMETER_DIM = 4
REPLICAS = 6
CHAINS = 4
ROWS = REPLICAS * CHAINS
WORKERS = ROWS
WORKER_CPU_IDS = tuple(range(32, 56))
PARENT_CPU_IDS = tuple(range(32, 64))
CHUNK_SIZE = 10
RESUME_TRANSITION = 500
WARMUP_MILESTONES = (600, 700, 800, 900, 1000)
WARMUP_WINDOW = 300
RETAINED_MILESTONES = (1000, 1250, 1500)
MASTER_SEED = (20260811, 8101)
CAMPAIGN_START = datetime.fromisoformat("2026-08-11T05:10:18+08:00")
CAMPAIGN_END = CAMPAIGN_START + timedelta(hours=8)
FINALIZATION_RESERVE_SECONDS = 300.0
R8_SHA256 = "9e6771652842b6f96e304509a042949dc2513923ef8279021a7783b4fd82b9d9"
R9_0P40_SHA256 = "b58381e92dc609ff2b33dade8901c62558df5cbc23d82c6fb25a5eb6a261e570"
R9_0P35_SHA256 = "22349ca8141f2b89d921adbc22eafb5774901eadf50f07d0a05d6fc4618394b2"
R10_1P5_SHA256 = "4b5f38eb87d6f45642859fb1035078bd63a1bcce1bbc750900c760280a0f167b"
R10_2P0_SHA256 = "f039cfc4b285a10385a1d7c73dd89cbe8f9f4a740a8502ce7de7a3de9cac4235"


class ResumeMaterialError(RuntimeError):
    """Raised when exact continuation or a hard invariant fails."""


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
        raise ResumeMaterialError(f"refusing to overwrite: {path}")
    encoded = json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def _write_tensor(path: Path, value: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise ResumeMaterialError(f"refusing to overwrite tensor: {path}")
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


def _append_log(message: str) -> None:
    absolute = _abs(LOG)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    with absolute.open("a", encoding="ascii") as stream:
        stream.write(f"{time.time():.6f} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, _abs(path))
    if spec is None or spec.loader is None:
        raise ResumeMaterialError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_receipt(tf: Any, receipt: Mapping[str, Any]) -> Any:
    path = Path(str(receipt["path"]))
    raw = _abs(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ResumeMaterialError(f"receipt hash mismatch: {path}")
    tensor = tf.io.parse_tensor(raw, out_type=tf.dtypes.as_dtype(receipt["dtype"]))
    if list(tensor.shape) != list(receipt["shape"]):
        raise ResumeMaterialError(f"receipt shape mismatch: {path}")
    return tensor


def _load_r8_history(tf: Any) -> Mapping[str, Any]:
    material = json.loads(_abs(R8_RESULT).read_text(encoding="utf-8"))
    states = []
    pre_states = []
    identities = []
    log_acceptance = []
    for index, row in enumerate(material["chunk_manifests"]):
        manifest_path = Path(row["path"])
        raw = _abs(manifest_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ResumeMaterialError(f"r8 manifest hash mismatch: {index}")
        manifest = json.loads(raw)
        if manifest["chunk_index"] != index:
            raise ResumeMaterialError("r8 chunk order mismatch")
        if manifest["transition_start_inclusive"] != index * CHUNK_SIZE:
            raise ResumeMaterialError("r8 chunk start mismatch")
        if manifest["transition_stop_exclusive"] != (index + 1) * CHUNK_SIZE:
            raise ResumeMaterialError("r8 chunk stop mismatch")
        receipts = manifest["tensor_receipts"]
        verified = {
            name: _read_receipt(tf, receipt)
            for name, receipt in receipts.items()
        }
        states.append(verified["state"])
        pre_states.append(verified["pre_swap_state"])
        identities.append(verified["identities"])
        log_acceptance.append(verified["hmc_log_accept_ratio"])
        terminal = {
            name: _read_receipt(tf, receipt)
            for name, receipt in manifest["terminal_checkpoint_receipts"].items()
        }
    if len(states) * CHUNK_SIZE != RESUME_TRANSITION:
        raise ResumeMaterialError("r8 history does not end at resume transition")
    return {
        "material": material,
        "state": tf.concat(states, axis=0),
        "pre_swap_state": tf.concat(pre_states, axis=0),
        "identities": tf.concat(identities, axis=0),
        "hmc_log_accept_ratio": tf.concat(log_acceptance, axis=0),
        "terminal": terminal,
    }


def _next_milestone(completed: int, milestones: tuple[int, ...]) -> int:
    for milestone in milestones:
        if int(completed) < milestone:
            return milestone
    return milestones[-1]


def run() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise ResumeMaterialError("refusing to overwrite resumed result")
    if tuple(sorted(os.sched_getaffinity(0))) != PARENT_CPU_IDS:
        raise ResumeMaterialError("parent CPU affinity mismatch")
    bindings = {
        "r8_material_sha256": _sha(R8_RESULT),
        "r9_ratio_0p40_sha256": _sha(R9_RATIO_0P40),
        "r9_ratio_0p35_sha256": _sha(R9_RATIO_0P35),
        "r10_hot_step_1p5_sha256": _sha(R10_STEP_1P5),
        "r10_hot_step_2p0_sha256": _sha(R10_STEP_2P0),
    }
    expected = {
        "r8_material_sha256": R8_SHA256,
        "r9_ratio_0p40_sha256": R9_0P40_SHA256,
        "r9_ratio_0p35_sha256": R9_0P35_SHA256,
        "r10_hot_step_1p5_sha256": R10_1P5_SHA256,
        "r10_hot_step_2p0_sha256": R10_2P0_SHA256,
    }
    if bindings != expected:
        raise ResumeMaterialError("bound campaign evidence mismatch")
    remaining_at_launch = CAMPAIGN_END.timestamp() - time.time()
    if remaining_at_launch <= FINALIZATION_RESERVE_SECONDS:
        raise ResumeMaterialError("eight-hour campaign deadline already exhausted")
    transition_deadline_epoch = CAMPAIGN_END.timestamp() - FINALIZATION_RESERVE_SECONDS
    _write_json(
        PROGRESS,
        {
            "status": "RESUMED_MATERIAL_STARTING",
            "completed_transitions": RESUME_TRANSITION,
            "campaign_end": CAMPAIGN_END.isoformat(),
            "remaining_seconds_at_launch": remaining_at_launch,
        },
        overwrite=True,
    )

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise ResumeMaterialError("CPU-only continuation found visible GPU")
    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool
    from bayesfilter.testing.distributed_replica_exchange_tf import distributed_replica_exchange_transition

    checkpoint = _load("resume_checkpoint_support", CHECKPOINT_RUNNER)
    material_support = _load("resume_material_support", MATERIAL_RUNNER)
    history = _load_r8_history(tf)
    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    chart = checkpoint._chart(tf, geometry)
    state_rows = list(tf.unstack(history["state"], axis=0))
    pre_rows = list(tf.unstack(history["pre_swap_state"], axis=0))
    identity_rows = list(tf.unstack(history["identities"], axis=0))
    log_accept_rows = list(tf.unstack(history["hmc_log_accept_ratio"], axis=0))
    current = dict(history["terminal"])
    warmup_cutoff = 0
    retained_count = 0
    warmup_checks = []
    retained_checks = []
    chunk_receipts = []
    transition_seconds = []
    cache_seconds = []
    cache_value_residuals = []
    cache_score_residuals = []
    terminal_status = "RESUMED_MATERIAL_RUNNING"
    hard_failures = []

    with TFBatchValueScorePool(material_support._pool_config()) as pool:
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

        value, score, status, identity_metadata = evaluator(
            tf.reshape(current["state"], (ROWS, PARAMETER_DIM)), "resume-cache-validation"
        )
        value = tf.reshape(value, (REPLICAS, CHAINS))
        score = tf.reshape(score, (REPLICAS, CHAINS, PARAMETER_DIM))
        cache_valid = tf.logical_and(
            tf.convert_to_tensor(status["status_code"], tf.int32) == 0,
            tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
        )
        if not bool(tf.reduce_all(cache_valid).numpy()):
            raise ResumeMaterialError("resumed terminal target status invalid")
        tf.debugging.assert_equal(value, current["base_target_log_prob"])
        tf.debugging.assert_equal(score, current["base_score"])
        worker_identity = checkpoint._worker_identity(identity_metadata)
        if not checkpoint._identity_passed(worker_identity):
            raise ResumeMaterialError("resumed worker identity failed")

        while True:
            if time.time() >= transition_deadline_epoch:
                terminal_status = "RESUMED_MATERIAL_CAMPAIGN_DEADLINE"
                break
            total = len(state_rows)
            if warmup_cutoff == 0:
                target_total = _next_milestone(total, WARMUP_MILESTONES)
                if total >= WARMUP_MILESTONES[-1]:
                    terminal_status = "RESUMED_MATERIAL_WARMUP_NOT_READY"
                    break
                phase = "warmup"
            else:
                retained_count = total - warmup_cutoff
                target_total = warmup_cutoff + _next_milestone(
                    retained_count, RETAINED_MILESTONES
                )
                if retained_count >= RETAINED_MILESTONES[-1]:
                    terminal_status = "RESUMED_MATERIAL_RETAINED_GATES_FAILED"
                    break
                phase = "retained"
            chunk_goal = min(CHUNK_SIZE, target_total - total)
            chunk_start = total
            new_states = []
            new_pre = []
            new_ids = []
            new_log_accept = []
            new_accept = []
            new_valid = []
            new_swap_matrix = []
            for _ in range(chunk_goal):
                transition_index = len(state_rows)
                transition_started = time.perf_counter()
                transition = distributed_replica_exchange_transition(
                    **current,
                    inverse_temperatures=checkpoint.BETAS,
                    step_sizes=checkpoint.STEPS,
                    num_leapfrog_steps=checkpoint.LEAPFROG,
                    transition_index=transition_index,
                    master_seed=MASTER_SEED,
                    evaluator=evaluator,
                )
                transition_seconds.append(time.perf_counter() - transition_started)
                invalid = tf.logical_not(transition["hmc_path_valid"])
                matrix = tf.cast(transition["swap_is_accepted_matrix"], tf.int32)
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
                gates = (
                    tf.reduce_all(tf.math.is_finite(transition["state"])),
                    tf.reduce_all(tf.math.is_finite(transition["base_target_log_prob"])),
                    tf.reduce_all(tf.math.is_finite(transition["base_score"])),
                    finite_log_accept_or_invalid,
                    tf.reduce_all(tf.logical_not(tf.boolean_mask(transition["hmc_is_accepted"], invalid))),
                    tf.reduce_all(tf.reduce_sum(matrix, axis=0) == tf.ones((REPLICAS, CHAINS), tf.int32)),
                    tf.reduce_all(tf.reduce_sum(matrix, axis=1) == tf.ones((REPLICAS, CHAINS), tf.int32)),
                )
                if not bool(tf.reduce_all(tf.stack(gates)).numpy()):
                    hard_failures.append(f"transition_{transition_index}_hard_gate")
                    terminal_status = "RESUMED_MATERIAL_HARD_GATE_FAILED"
                for destination, name in (
                    (new_states, "state"),
                    (new_pre, "pre_swap_state"),
                    (new_ids, "identities_at_temperature"),
                    (new_log_accept, "hmc_log_accept_ratio"),
                    (new_accept, "hmc_is_accepted"),
                    (new_valid, "hmc_path_valid"),
                    (new_swap_matrix, "swap_is_accepted_matrix"),
                ):
                    destination.append(transition[name])
                state_rows.append(transition["state"])
                pre_rows.append(transition["pre_swap_state"])
                identity_rows.append(transition["identities_at_temperature"])
                log_accept_rows.append(transition["hmc_log_accept_ratio"])
                current = {
                    name: transition[name]
                    for name in ("state", "base_target_log_prob", "base_score", "identities_at_temperature")
                }
                if hard_failures:
                    break
            cache_started = time.perf_counter()
            cache_value, cache_score, cache_status, _cache_metadata = evaluator(
                tf.reshape(current["state"], (ROWS, PARAMETER_DIM)),
                f"chunk-{len(chunk_receipts):04d}-terminal-cache",
            )
            cache_seconds.append(time.perf_counter() - cache_started)
            cache_value = tf.reshape(cache_value, (REPLICAS, CHAINS))
            cache_score = tf.reshape(
                cache_score, (REPLICAS, CHAINS, PARAMETER_DIM)
            )
            cache_valid = tf.logical_and(
                tf.convert_to_tensor(cache_status["status_code"], tf.int32) == 0,
                tf.convert_to_tensor(
                    cache_status["valid_pre_regularized_score"], tf.bool
                ),
            )
            if not bool(tf.reduce_all(cache_valid).numpy()):
                hard_failures.append("terminal_cache_target_status_invalid")
                terminal_status = "RESUMED_MATERIAL_HARD_GATE_FAILED"
            cache_value_residuals.append(
                tf.reduce_max(
                    tf.abs(cache_value - current["base_target_log_prob"])
                )
            )
            cache_score_residuals.append(
                tf.reduce_max(tf.abs(cache_score - current["base_score"]))
            )
            chunk_index = len(chunk_receipts)
            receipts = {
                "state": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-state.tftensor", tf.stack(new_states), tf),
                "pre_swap_state": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-pre_swap_state.tftensor", tf.stack(new_pre), tf),
                "identities": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-identities.tftensor", tf.stack(new_ids), tf),
                "hmc_log_accept_ratio": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-hmc_log_accept_ratio.tftensor", tf.stack(new_log_accept), tf),
                "hmc_is_accepted": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-hmc_is_accepted.tftensor", tf.stack(new_accept), tf),
                "hmc_path_valid": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-hmc_path_valid.tftensor", tf.stack(new_valid), tf),
                "swap_is_accepted_matrix": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-swap_is_accepted_matrix.tftensor", tf.stack(new_swap_matrix), tf),
                "terminal_state": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-terminal-state.tftensor", current["state"], tf),
                "terminal_target": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-terminal-target.tftensor", current["base_target_log_prob"], tf),
                "terminal_score": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-terminal-score.tftensor", current["base_score"], tf),
                "terminal_identities": _write_tensor(OUTPUT_ROOT / f"chunk-{chunk_index:04d}-terminal-identities.tftensor", current["identities_at_temperature"], tf),
            }
            manifest_path = OUTPUT_ROOT / f"chunk-{chunk_index:04d}.json"
            _write_json(
                manifest_path,
                {
                    "schema": "bayesfilter.ssl_lstm.q20_physical_replica_resumed_chunk.v1",
                    "chunk_index": chunk_index,
                    "phase": phase,
                    "transition_start_inclusive": chunk_start,
                    "transition_stop_exclusive": len(state_rows),
                    "receipts": receipts,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            chunk_receipts.append({"path": manifest_path.as_posix(), "sha256": _sha(manifest_path)})
            if hard_failures:
                break

            total = len(state_rows)
            all_state = tf.stack(state_rows)
            all_pre = tf.stack(pre_rows)
            all_ids = tf.stack(identity_rows)
            physical = chart["center"] + tf.matmul(
                tf.reshape(all_state, (-1, PARAMETER_DIM)), chart["factor"], transpose_b=True
            )
            physical = tf.reshape(physical, (total, REPLICAS, CHAINS, PARAMETER_DIM))
            pre_physical = chart["center"] + tf.matmul(
                tf.reshape(all_pre, (-1, PARAMETER_DIM)), chart["factor"], transpose_b=True
            )
            pre_physical = tf.reshape(pre_physical, tf.shape(physical))
            if warmup_cutoff == 0 and total in WARMUP_MILESTONES:
                start = total - WARMUP_WINDOW
                previous = physical[start - 1] if start > 0 else physical[0]
                convergence = material_support._diagnose_warmup(tf, physical[start:total])
                travel = material_support._window_round_trips(tf, all_ids[start:total])
                forgetting = material_support._hot_forgetting(
                    tf, pre_physical[start:total], physical[start:total], previous
                )
                check = {
                    "global_transition": total,
                    "window_start": start,
                    "convergence": convergence,
                    "travel": travel,
                    "hot_forgetting": forgetting,
                    "passed": bool(
                        convergence["passed"]
                        and travel["each_chain_has_required_round_trip"].numpy()
                        and forgetting["all_chains_passed"].numpy()
                    ),
                }
                warmup_checks.append(check)
                if check["passed"]:
                    warmup_cutoff = total
                    _append_log(f"warmup ready at global transition {total}")
                elif total == WARMUP_MILESTONES[-1]:
                    terminal_status = "RESUMED_MATERIAL_WARMUP_NOT_READY"
                    break
            elif warmup_cutoff > 0:
                retained_count = total - warmup_cutoff
                if retained_count in RETAINED_MILESTONES:
                    retained_physical = physical[warmup_cutoff:total]
                    convergence = material_support._diagnose_retained(tf, retained_physical)
                    travel = material_support._window_round_trips(tf, all_ids[warmup_cutoff:total])
                    forgetting = material_support._hot_forgetting(
                        tf,
                        pre_physical[warmup_cutoff:total],
                        retained_physical,
                        physical[warmup_cutoff - 1],
                    )
                    acceptance = material_support._acceptance_summary(
                        tf, tf.stack(log_accept_rows[warmup_cutoff:total])
                    )
                    check = {
                        "retained_draws_per_chain": retained_count,
                        "convergence": convergence,
                        "travel": travel,
                        "hot_forgetting": forgetting,
                        "acceptance": acceptance,
                        "passed": bool(
                            convergence["passed"]
                            and travel["each_chain_has_required_round_trip"].numpy()
                            and forgetting["all_chains_passed"].numpy()
                            and acceptance["all_temperature_chain_means_in_band"].numpy()
                        ),
                    }
                    retained_checks.append(check)
                    if check["passed"]:
                        terminal_status = "RESUMED_MATERIAL_ADMISSION_PASSED"
                        break
                    if retained_count == RETAINED_MILESTONES[-1]:
                        terminal_status = "RESUMED_MATERIAL_RETAINED_GATES_FAILED"
                        break
            _write_json(
                PROGRESS,
                {
                    "status": "RESUMED_MATERIAL_RUNNING",
                    "phase": "retained" if warmup_cutoff else "warmup",
                    "completed_transitions": total,
                    "warmup_cutoff": warmup_cutoff or total,
                    "retained_transitions": total - warmup_cutoff if warmup_cutoff else 0,
                    "elapsed_seconds": time.perf_counter() - started,
                    "campaign_remaining_seconds": CAMPAIGN_END.timestamp() - time.time(),
                    "latest_warmup_check": warmup_checks[-1] if warmup_checks else None,
                    "latest_retained_check": retained_checks[-1] if retained_checks else None,
                    "last_chunk": chunk_receipts[-1],
                },
                overwrite=True,
            )

    total = len(state_rows)
    retained_count = total - warmup_cutoff if warmup_cutoff else 0
    passed = terminal_status == "RESUMED_MATERIAL_ADMISSION_PASSED" and not hard_failures
    retained_receipt = None
    cold_negative_fraction = None
    if retained_count:
        all_state = tf.stack(state_rows)
        physical = chart["center"] + tf.matmul(
            tf.reshape(all_state, (-1, PARAMETER_DIM)), chart["factor"], transpose_b=True
        )
        physical = tf.reshape(physical, (total, REPLICAS, CHAINS, PARAMETER_DIM))
        retained = physical[warmup_cutoff:]
        retained_receipt = _write_tensor(
            OUTPUT_ROOT / "retained-physical-diagnostic.tftensor", retained, tf
        )
        cold_negative_fraction = tf.reduce_mean(
            tf.cast(retained[:, 0, :, 2] < 0.0, tf.float64)
        )
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_replica_resumed_material.v1",
        "status": terminal_status,
        "passed": passed,
        "configuration": {
            "inverse_temperatures": checkpoint.BETAS,
            "step_sizes": checkpoint.STEPS,
            "num_leapfrog_steps": checkpoint.LEAPFROG,
            "master_seed": MASTER_SEED,
            "resume_transition": RESUME_TRANSITION,
            "workers": WORKERS,
            "worker_cpu_ids": WORKER_CPU_IDS,
            "jit_compile": True,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        },
        "budget": {
            "campaign_start": CAMPAIGN_START.isoformat(),
            "campaign_end": CAMPAIGN_END.isoformat(),
            "remaining_seconds_at_launch": remaining_at_launch,
            "runner_wall_seconds": time.perf_counter() - started,
            "campaign_remaining_seconds_at_end": CAMPAIGN_END.timestamp() - time.time(),
        },
        "counts": {
            "global_transitions": total,
            "warmup_cutoff": warmup_cutoff or total,
            "retained_draws_per_chain": retained_count,
        },
        "hard_failures": hard_failures,
        "cache_diagnostics": {
            "evaluation_seconds": cache_seconds,
            "value_max_abs_residuals": cache_value_residuals,
            "score_max_abs_residuals": cache_score_residuals,
            "role": "explanatory_only_status_invalidity_is_a_hard_veto",
        },
        "warmup_checks": warmup_checks,
        "retained_checks": retained_checks,
        "cold_negative_sign_fraction": cold_negative_fraction,
        "occupancy_role": "explanatory_only_not_mass_authority",
        "chunk_manifests": chunk_receipts,
        "retained_physical_diagnostic_receipt": retained_receipt,
        "bindings": bindings,
        "run_manifest": {
            "launch_git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "command": " ".join(sys.argv),
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "worker_identity": worker_identity,
            "source_sha256": {"runner": _sha(RUNNER), "checkpoint_runner": _sha(CHECKPOINT_RUNNER), "material_runner": _sha(MATERIAL_RUNNER)},
        },
        "nonclaims": (
            "tuning canary draws excluded from this continuation",
            "raw occupancy is not posterior mass authority",
            "two-region travel does not prove exhaustive mode discovery",
            "no predictive, superiority, or default-readiness claim",
        ),
    }
    _write_json(FINAL, payload)
    _write_json(
        PROGRESS,
        {
            "status": terminal_status,
            "passed": passed,
            "completed_transitions": total,
            "warmup_cutoff": warmup_cutoff or total,
            "retained_transitions": retained_count,
            "elapsed_seconds": time.perf_counter() - started,
            "result": FINAL.as_posix(),
        },
        overwrite=True,
    )
    return payload


def main() -> None:
    started = time.perf_counter()
    try:
        payload = run()
    except BaseException as error:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20.physical_replica_resumed_failure.v1",
            "status": "RESUMED_MATERIAL_HARNESS_FAILED",
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
