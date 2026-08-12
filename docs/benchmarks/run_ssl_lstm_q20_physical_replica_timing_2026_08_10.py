#!/usr/bin/env python3
"""Profile compile and cached exact replica transitions under bounded CPU topologies."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-plan-2026-08-10.md"
)
RESULT = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-result-2026-08-10.md"
)
RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_q20_physical_replica_timing_2026_08_10.py"
)
GLOBAL_RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_q20_physical_global_repair_2026_08_10.py"
)
REPLICA_HELPER = Path("bayesfilter/testing/replica_exchange_tf.py")
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-"
    "2026-08-10/r1/geometry.json"
)
LOCAL_GATE = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-global-repair-2026-08-10/"
    "r1/physical-local.json"
)
BASELINE = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-global-repair-2026-08-10/"
    "r1/physical-transition.json"
)
SMC_RECOVERY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/"
    "r2/receipt-recovery-v1.json"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-"
    "2026-08-10/r1-timing"
)
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "timing.json"

PARAMETER_DIM = 4
BETAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
STEPS = tuple(0.05 / math.sqrt(beta) for beta in BETAS)
LEAPFROG = 8
CHAINS = 2
TRANSITIONS_PER_CALL = 1
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
EXPECTED_SHA256 = {
    GEOMETRY: "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb",
    LOCAL_GATE: "fdb782346c62d53384611b8a9bed95b72e5329693aab18160890d59d6267a62a",
    BASELINE: "dbbb094337fef40cb5a2d7b8f715e192d4c595b513ae223af89824b5cb82f04d",
    SMC_RECOVERY: "3aea988e7b27381a6b62e7a2d452db8251b9bd7d8b9f5e68ad08fcbe711b6d97",
}


class ReplicaTimingError(RuntimeError):
    """Raised when the timing profile cannot preserve its evidence contract."""


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


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists() and not overwrite:
        raise ReplicaTimingError(f"refusing to overwrite artifact: {path}")
    encoded = json.dumps(
        _safe(payload), sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def _write_tensor(path: Path, value: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise ReplicaTimingError(f"refusing to overwrite tensor: {path}")
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


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _abs(path))
    if spec is None or spec.loader is None:
        raise ReplicaTimingError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        if _sha(path) != expected:
            raise ReplicaTimingError(f"bound input identity mismatch: {path}")
    local = json.loads(_abs(LOCAL_GATE).read_text(encoding="utf-8"))
    baseline = json.loads(_abs(BASELINE).read_text(encoding="utf-8"))
    recovery = json.loads(_abs(SMC_RECOVERY).read_text(encoding="utf-8"))
    if local.get("status") != "PHYSICAL_LOCAL_CANARY_PASSED":
        raise ReplicaTimingError("physical local gate did not pass")
    if baseline.get("status") != "PHYSICAL_TRANSITION_CANARY_COMPLETED":
        raise ReplicaTimingError("physical baseline did not complete")
    if recovery.get("status") != "SMC_RECEIPT_RECOVERY_PASSED":
        raise ReplicaTimingError("SMC receipt recovery did not pass")
    configuration = baseline["configuration"]
    if (
        tuple(configuration["inverse_temperatures"]) != BETAS
        or tuple(configuration["step_sizes"]) != STEPS
        or int(configuration["num_leapfrog_steps"]) != LEAPFROG
        or int(configuration["chains"]) != CHAINS
    ):
        raise ReplicaTimingError("timing configuration drifted from physical baseline")


def _configure_tensorflow(threads: int) -> tuple[Any, Any]:
    os.environ["OMP_NUM_THREADS"] = str(int(threads))
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(int(threads))
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    import tensorflow as tf
    import tensorflow_probability as tfp

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(int(threads))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise ReplicaTimingError("CPU-only timing child found visible GPU")
    return tf, tfp


def _call_summary(trace: Mapping[str, Any], elapsed: float, tf: Any) -> Mapping[str, Any]:
    finite = tf.reduce_all(
        tf.stack(
            [
                tf.reduce_all(tf.math.is_finite(trace[name]))
                for name in (
                    "replica_states",
                    "pre_swap_replica_states",
                    "hmc_log_accept_ratio",
                    "potential_energy",
                )
            ]
        )
    )
    proposed = tf.reduce_sum(
        tf.cast(trace["swap_is_proposed_adjacent"], tf.int32), axis=(0, 2)
    )
    accepted = tf.reduce_sum(
        tf.cast(trace["swap_is_accepted_adjacent"], tf.int32), axis=(0, 2)
    )
    return {
        "elapsed_seconds": elapsed,
        "seconds_per_transition": elapsed / TRANSITIONS_PER_CALL,
        "finite": finite,
        "hmc_acceptance_by_temperature": tf.reduce_mean(
            tf.cast(trace["hmc_is_accepted"], tf.float64), axis=(0, 2)
        ),
        "adjacent_swap_proposals": proposed,
        "adjacent_swap_acceptances": accepted,
        "trace_shapes": {name: list(value.shape) for name, value in trace.items()},
    }


def run_child(*, label: str, threads: int, output_root: Path) -> Mapping[str, Any]:
    started = time.perf_counter()
    final = output_root / "timing.json"
    if _abs(final).exists():
        raise ReplicaTimingError("refusing to overwrite child timing result")
    expected_affinity = tuple(range(32, 32 + int(threads)))
    actual_affinity = tuple(sorted(os.sched_getaffinity(0)))
    if actual_affinity != expected_affinity:
        raise ReplicaTimingError(
            f"child affinity mismatch: expected {expected_affinity}, got {actual_affinity}"
        )
    _verify_inputs()
    tf, tfp = _configure_tensorflow(threads)
    global_runner = _load_module(GLOBAL_RUNNER, f"physical_global_for_timing_{label}")
    replica = _load_module(REPLICA_HELPER, f"physical_replica_helper_for_timing_{label}")
    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    # The historical runner defaults to four threads.  Bind its configuration
    # helper to this child's declared topology before it touches TensorFlow.
    global_runner.THREADS = int(threads)
    target = global_runner._configure()[1]
    if int(global_runner.THREADS) != int(threads):
        raise ReplicaTimingError("historical target helper thread binding failed")
    parity = global_runner._target_parity(tf, target, geometry)
    chart = global_runner._physical_chart(tf, geometry)
    center = chart["center"]
    factor = chart["factor"]
    log_abs_determinant = tf.reduce_sum(tf.math.log(tf.linalg.eigvalsh(factor)))
    initial_state = tf.repeat(
        chart["latent_centers"][tf.newaxis, :, :], len(BETAS), axis=0
    )

    def latent_target(state: Any) -> Any:
        values = tf.convert_to_tensor(state, tf.float64)
        z = tf.reshape(values, (-1, PARAMETER_DIM))
        theta = center + tf.matmul(z, factor, transpose_b=True)
        target_value, _score, _status = target.neutra_batch_log_prob_and_grad_status(theta)
        return tf.reshape(
            tf.convert_to_tensor(target_value, tf.float64) + log_abs_determinant,
            tf.shape(values)[:-1],
        )

    sampler = replica.make_replica_exchange_fixed_hmc_sampler(
        latent_target,
        initial_state,
        inverse_temperatures=BETAS,
        step_sizes=STEPS,
        num_leapfrog_steps=LEAPFROG,
        num_steps=TRANSITIONS_PER_CALL,
        jit_compile=True,
    )
    call_rows = []
    receipts = {}
    current = initial_state
    for call_index in range(2):
        call_started = time.perf_counter()
        trace = sampler(
            current,
            tf.constant((20260810, 5101 + call_index), tf.int32),
        )
        elapsed = time.perf_counter() - call_started
        current = trace["replica_states"][-1]
        summary = _call_summary(trace, elapsed, tf)
        if not bool(summary["finite"].numpy()):
            raise ReplicaTimingError(f"non-finite replica trace in call {call_index}")
        call_rows.append(summary)
        receipts[f"call_{call_index}_replica_states"] = _write_tensor(
            output_root / f"call-{call_index}-replica-states.tftensor",
            trace["replica_states"],
            tf,
        )
        receipts[f"call_{call_index}_pre_swap_states"] = _write_tensor(
            output_root / f"call-{call_index}-pre-swap-states.tftensor",
            trace["pre_swap_replica_states"],
            tf,
        )
        receipts[f"call_{call_index}_hmc_log_accept_ratio"] = _write_tensor(
            output_root / f"call-{call_index}-hmc-log-accept-ratio.tftensor",
            trace["hmc_log_accept_ratio"],
            tf,
        )
        receipts[f"call_{call_index}_swap_matrix"] = _write_tensor(
            output_root / f"call-{call_index}-swap-matrix.tftensor",
            trace["swap_is_accepted_matrix"],
            tf,
        )

    accepted_states = tf.reshape(current, (-1, PARAMETER_DIM))
    theta = center + tf.matmul(accepted_states, factor, transpose_b=True)
    target_values, target_scores, status = target.neutra_batch_log_prob_and_grad_status(theta)
    valid = tf.logical_and(
        tf.convert_to_tensor(status["status_code"], tf.int32) == 0,
        tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
    )
    valid = tf.logical_and(
        valid,
        tf.logical_and(
            tf.math.is_finite(tf.convert_to_tensor(target_values, tf.float64)),
            tf.reduce_all(
                tf.math.is_finite(tf.convert_to_tensor(target_scores, tf.float64)),
                axis=1,
            ),
        ),
    )
    if not bool(tf.reduce_all(valid).numpy()):
        raise ReplicaTimingError("terminal accepted target state/status is invalid")
    tracing_count = int(sampler.experimental_get_tracing_count())
    if tracing_count != 1:
        raise ReplicaTimingError(f"reusable sampler retraced {tracing_count} times")
    hlo = sampler.experimental_get_compiler_ir(
        initial_state, tf.constant((20260810, 5199), tf.int32)
    )(stage="hlo")
    hlo_bytes = str(hlo).encode("utf-8")
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_replica_timing.child.v1",
        "status": "PHYSICAL_REPLICA_TIMING_CHILD_PASSED",
        "label": label,
        "configuration": {
            "threads": int(threads),
            "historical_target_helper_threads": int(global_runner.THREADS),
            "cpu_affinity": actual_affinity,
            "inverse_temperatures": BETAS,
            "step_sizes": STEPS,
            "num_leapfrog_steps": LEAPFROG,
            "chains": CHAINS,
            "transitions_per_call": TRANSITIONS_PER_CALL,
            "call_count": 2,
            "jit_compile": True,
        },
        "call_0_compile_inclusive": call_rows[0],
        "call_1_cached": call_rows[1],
        "sampler_tracing_count": tracing_count,
        "terminal_target_status_all_valid": True,
        "target_parity": parity,
        "bindings": {
            "target_signature": target.target_signature(),
            "adapter_signature": target.adapter_signature(),
            "geometry_sha256": _sha(GEOMETRY),
            "local_gate_sha256": _sha(LOCAL_GATE),
            "baseline_sha256": _sha(BASELINE),
            "smc_recovery_sha256": _sha(SMC_RECOVERY),
        },
        "hlo_sha256": hashlib.sha256(hlo_bytes).hexdigest(),
        "trace_receipts": receipts,
        "run_manifest": {
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ("git", "status", "--porcelain"),
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
            "wall_time_seconds": time.perf_counter() - started,
            "artifact_root": output_root.as_posix(),
            "plan_file": PLAN.as_posix(),
            "result_file": RESULT.as_posix(),
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "global_runner": _sha(GLOBAL_RUNNER),
                "replica_helper": _sha(REPLICA_HELPER),
            },
        },
        "nonclaims": (
            "timing and mechanics canary only",
            "one cached call does not statistically rank CPU topologies",
            "no travel, convergence, posterior, weight, or predictive claim",
        ),
    }
    if (
        payload["bindings"]["target_signature"] != TARGET_SIGNATURE
        or payload["bindings"]["adapter_signature"] != ADAPTER_SIGNATURE
    ):
        raise ReplicaTimingError("target or adapter signature mismatch")
    _write_json(final, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_child(
        label=str(args.label),
        threads=int(args.threads),
        output_root=Path(args.output_root),
    )
    print(json.dumps({"status": result["status"], "label": result["label"]}, sort_keys=True))
