#!/usr/bin/env python3
"""Replay T1 and issue centered-difference XLA core tangents on GPU."""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

import tensorflow as tf  # noqa: E402

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_limit,
)

GPU_MEMORY_LIMIT_MIB = 6 * 1024
MEMORY_POLICY = configure_tensorflow_gpu_memory_limit(
    tf, memory_limit_mib=GPU_MEMORY_LIMIT_MIB, require_gpu=True
)

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (  # noqa: E402
    FINITE_DIFFERENCE_ATOL,
    FINITE_DIFFERENCE_RTOL,
    FINITE_DIFFERENCE_STEP,
    FUNCTIONAL_SCREEN_COLUMNS,
    FUNCTIONAL_SCREEN_ORDER,
    GPU_MEMORY_LIMIT_MIB as ISSUER_GPU_MEMORY_LIMIT_MIB,
    ISSUER_ID,
    ISSUER_SCHEMA,
    MEMORY_CAP_BYTES,
    OFFLINE_ISSUER_DERIVATIVE,
    REQUIRED_ISSUER_SOURCE_PATHS,
    RUNTIME_SCORE_BACKEND,
    SHIFT_DERIVATIVE_POLICY,
    TAU_DERIVATIVE_POLICY,
    TANGENT_FINITE_DIFFERENCE_STEP,
    prepare_t1_replay_inputs,
    replay_t1_training_jvp,
)
from bayesfilter.highdim.zhao_cui_austria_sir_packed_xla_tf import (  # noqa: E402
    MATERIAL_REPLAY_ATOL,
    MATERIAL_REPLAY_POLICY_ID,
    MATERIAL_REPLAY_RTOL,
    PACKED_XLA_POLICY_ID,
    material_positive_value_metrics,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (  # noqa: E402
    LaneBParameterChild,
)


PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)
if GPU_MEMORY_LIMIT_MIB != ISSUER_GPU_MEMORY_LIMIT_MIB:
    raise RuntimeError("T1 runner and issuer GPU memory limits disagree")
PLAN = Path(
    "docs/plans/bayesfilter-zhao-cui-austria-sir-material-replay-xla-repair-plan-2026-08-02.md"
)
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, tf.Tensor):
        if value.shape.rank == 0:
            return value.numpy().item()
        return value.numpy().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_tensor(path: Path, value: tf.Tensor) -> dict[str, object]:
    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(value))
    tf.io.write_file(path.as_posix(), serialized)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(bytes(serialized.numpy())).hexdigest(),
        "dtype": value.dtype.name,
        "shape": value.shape.as_list(),
    }


def _semantic_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parent = load_lane_b_t1_artifact_v1_compat(PARENT_DIR)
    inputs = prepare_t1_replay_inputs(parent)
    replay = replay_t1_training_jvp(parent, inputs=inputs)
    child = LaneBParameterChild(parent, replay.tangent_cores)
    manual_value, manual_score = child.increment_and_score(
        tf.zeros([3], tf.float64)
    )
    tf.debugging.assert_near(manual_value, replay.value, atol=2e-13, rtol=0.0)
    tf.debugging.assert_near(
        manual_score,
        replay.score,
        atol=FINITE_DIFFERENCE_ATOL,
        rtol=FINITE_DIFFERENCE_RTOL,
        message="T1 issued core-tangent/manual score mismatch",
    )
    step = tf.constant(FINITE_DIFFERENCE_STEP, tf.float64)
    finite_difference = (
        replay.finite_difference_plus - replay.finite_difference_minus
    ) / (2.0 * step)
    tf.debugging.assert_near(
        finite_difference,
        replay.score,
        atol=FINITE_DIFFERENCE_ATOL,
        rtol=FINITE_DIFFERENCE_RTOL,
        message="T1 replay finite-difference mismatch",
    )
    scalar_passed, scalar_absolute, scalar_normalized, scalar_log_residual = (
        material_positive_value_metrics(tf.exp(replay.value), tf.exp(parent.value()))
    )
    tf.debugging.assert_equal(
        scalar_passed, True, "T1 material scalar replay gate failed"
    )
    finite_difference_rows = [
        {
            "parameter": parameter,
            "step": FINITE_DIFFERENCE_STEP,
            "value_plus": replay.finite_difference_plus[parameter],
            "value_minus": replay.finite_difference_minus[parameter],
            "finite_difference": finite_difference[parameter],
            "issued_tangent_score": replay.score[parameter],
            "absolute_residual": tf.abs(
                finite_difference[parameter] - replay.score[parameter]
            ),
        }
        for parameter in range(3)
    ]
    memory = tf.config.experimental.get_memory_info("GPU:0")
    if int(memory["peak"]) > MEMORY_CAP_BYTES:
        raise MemoryError("T1 replay/FD tangent exceeded the 6 GiB allocator cap")
    tensors: dict[str, object] = {}
    for axis, bank in enumerate(replay.tangent_cores):
        for parameter, tangent in enumerate(bank):
            name = f"tangent_{axis:02d}_{parameter}"
            tensors[name] = _write_tensor(output / f"{name}.tensor", tangent)
    source_files = {
        relative_path: _sha256(ROOT / relative_path)
        for relative_path in REQUIRED_ISSUER_SOURCE_PATHS
    }
    identity_payload = {
        "schema_version": ISSUER_SCHEMA,
        "issuer_id": ISSUER_ID,
        "replay_id": replay.replay_id,
        "classification": "extension_or_invention",
        "material_replay_policy_id": MATERIAL_REPLAY_POLICY_ID,
        "packed_xla_policy_id": PACKED_XLA_POLICY_ID,
        "parent_identity": parent.identity.hash.value,
        "parent_value": parent.value(),
        "child_identity": child.identity.hash.value,
        "training_cloud_manifest": dict(parent.training_cloud_manifest),
        "calibration_estimate": parent.calibration_estimate.manifest_payload(),
        "shift_derivative_policy": SHIFT_DERIVATIVE_POLICY,
        "tau_derivative_policy": TAU_DERIVATIVE_POLICY,
        "optimizer": {
            "family": "keras3_adam_functional_exact_update_order",
            "learning_rate": parent.settings.learning_rate,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-7,
            "gradient_clip_norm": parent.settings.gradient_clip_norm,
            "train_steps": parent.settings.train_steps,
            "batch_size": parent.settings.batch_size,
            "jit_compile": True,
            "full_program_control_flow": "tensorflow_while_loop",
            "python_numerical_loops": False,
        },
        "replay_gate": {
            "material_functional_atol": MATERIAL_REPLAY_ATOL,
            "material_functional_rtol": MATERIAL_REPLAY_RTOL,
            "maximum_normalized_functional_residual": 1.0,
            "functional_screen_order": FUNCTIONAL_SCREEN_ORDER,
            "functional_screen_columns": FUNCTIONAL_SCREEN_COLUMNS,
            "tangent_finite_difference_step": TANGENT_FINITE_DIFFERENCE_STEP,
            "finite_difference_step": FINITE_DIFFERENCE_STEP,
            "finite_difference_atol": FINITE_DIFFERENCE_ATOL,
            "finite_difference_rtol": FINITE_DIFFERENCE_RTOL,
            "memory_cap_bytes": MEMORY_CAP_BYTES,
            "gpu_memory_limit_mib": GPU_MEMORY_LIMIT_MIB,
        },
        "material_replay_evidence": {
            "functional_replay_metrics": replay.functional_replay_metrics,
            "scalar_absolute_residual": scalar_absolute,
            "scalar_normalized_residual": scalar_normalized,
            "scalar_log_residual": scalar_log_residual,
        },
        "derivative_evidence": {
            "issuer_method": OFFLINE_ISSUER_DERIVATIVE,
            "independent_finite_difference_rows": finite_difference_rows,
        },
        "tangent_tensor_sha256": {
            name: row["sha256"] for name, row in tensors.items()
        },
        "source_sha256": source_files,
        "runtime_score_backend": RUNTIME_SCORE_BACKEND,
        "offline_issuer_derivative": OFFLINE_ISSUER_DERIVATIVE,
        "runtime_autodiff": False,
        "runtime_finite_difference": False,
        "hmc_authorized": False,
    }
    issuer_identity = _semantic_sha256(identity_payload)
    result = {
        "schema_version": ISSUER_SCHEMA,
        "status": "PASS_T1_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT",
        "issuer_identity": issuer_identity,
        "issuer_identity_payload": identity_payload,
        "parent_identity": parent.identity.hash.value,
        "parent_value": parent.value(),
        "replay_id": replay.replay_id,
        "replay_value": replay.value,
        "issued_tangent_score": replay.score,
        "manual_value": manual_value,
        "manual_score": manual_score,
        "finite_difference_rows": finite_difference_rows,
        "functional_screen_order": FUNCTIONAL_SCREEN_ORDER,
        "functional_screen_columns": FUNCTIONAL_SCREEN_COLUMNS,
        "functional_replay_metrics": replay.functional_replay_metrics,
        "scalar_absolute_residual": scalar_absolute,
        "scalar_normalized_residual": scalar_normalized,
        "scalar_log_residual": scalar_log_residual,
        "explanatory_core_residuals": {
            "maximum_absolute_residual": replay.maximum_core_residual,
            "maximum_normalized_residual": replay.maximum_normalized_core_residual,
            "promotion_role": "explanatory_gauge_diagnostic_only",
        },
        "child_identity": child.identity.hash.value,
        "tensors": tensors,
        "hard_gates": {
            "training_and_calibration_cloud_hashes": True,
            "material_functional_replay": True,
            "material_scalar_replay": bool(scalar_passed.numpy()),
            "manual_issued_tangent_parity": True,
            "independent_step_halving_fd_parity": True,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
        },
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": sys.argv,
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "device": [item.name for item in tf.config.list_logical_devices("GPU")],
            "dtype": "float64",
            "jit_compile": True,
            "full_program_control_flow": "tensorflow_while_loop",
            "python_numerical_loops": False,
            "offline_derivative_method": OFFLINE_ISSUER_DERIVATIVE,
            "gpu_memory_policy": dict(MEMORY_POLICY),
            "gpu_allocator": {key: int(value) for key, value in memory.items()},
            "plan": PLAN.as_posix(),
            "plan_sha256": _sha256(ROOT / PLAN),
            "wall_time_seconds": time.monotonic() - started,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "nonclaims": (
            "no T2 or later score",
            "no bitwise core replay claim",
            "no exact-autodiff or JVP claim",
            "issued tangents are centered-finite-difference estimates",
            "no exact physical likelihood theorem",
            "no HMC readiness",
            "no source-faithful parameter algorithm",
        ),
    }
    (output / "result.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
