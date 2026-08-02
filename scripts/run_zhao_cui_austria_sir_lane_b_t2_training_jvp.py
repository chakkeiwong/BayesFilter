#!/usr/bin/env python3
"""Issue the material-replay Lane-B T2 finite-difference tangent artifact."""

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
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_prepared_tf import (  # noqa: E402
    load_t2_prepared_cloud,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_training_jvp_tf import (  # noqa: E402
    T2_FINITE_DIFFERENCE_ATOL,
    T2_FINITE_DIFFERENCE_RTOL,
    T2_FINITE_DIFFERENCE_STEP,
    T2_GPU_MEMORY_LIMIT_MIB,
    T2_ISSUER_SCHEMA,
    T2_MEMORY_CAP_BYTES,
    T2_OFFLINE_ISSUER_DERIVATIVE,
    T2_TANGENT_FINITE_DIFFERENCE_STEP,
    issue_t2_training_jvp_identity_payload,
    make_t2_replay_inputs,
    replay_t2_training_jvp,
    replay_t2_training_value,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (  # noqa: E402
    FUNCTIONAL_SCREEN_COLUMNS,
    FUNCTIONAL_SCREEN_ORDER,
    MATERIAL_REPLAY_ATOL,
    MATERIAL_REPLAY_POLICY_ID,
    MATERIAL_REPLAY_RTOL,
    _semantic_sha256,
    load_t1_training_jvp_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (  # noqa: E402
    LaneBParameterChild,
    load_selected_t2_parameter_parent_compat,
)


PARENT_T1_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)
if GPU_MEMORY_LIMIT_MIB != T2_GPU_MEMORY_LIMIT_MIB:
    raise RuntimeError("T2 runner and issuer GPU memory limits disagree")
PARENT_T2_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "pilot-final-01/t2_p05_r4_b5_lr3e4_l1_1e9/artifact"
)
TRAINING_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "attempt-06-training-prepared-final-closure"
)
CALIBRATION_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "attempt-08-calibration-prepared-final-closure"
)
PLAN = Path(
    "docs/plans/bayesfilter-zhao-cui-austria-sir-material-replay-xla-repair-plan-2026-08-02.md"
)
def _jsonable(value):
    if isinstance(value, tf.Tensor):
        if value.shape.rank == 0:
            return value.numpy().item()
        return value.numpy().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_tensor(path: Path, value: tf.Tensor) -> dict[str, object]:
    serialized = tf.io.serialize_tensor(value)
    tf.io.write_file(path.as_posix(), serialized)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(bytes(serialized.numpy())).hexdigest(),
        "dtype": value.dtype.name,
        "shape": value.shape.as_list(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1-issuer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    parent_t1 = load_lane_b_t1_artifact_v1_compat(PARENT_T1_DIR)
    t1_child, t1_issuer = load_t1_training_jvp_child(
        args.t1_issuer_dir.resolve(), parent=parent_t1
    )
    parent_t2 = load_selected_t2_parameter_parent_compat(
        PARENT_T2_DIR, parent_artifact=parent_t1
    )
    training_cloud, training_payload = load_t2_prepared_cloud(TRAINING_DIR)
    calibration_cloud, calibration_payload = load_t2_prepared_cloud(CALIBRATION_DIR)
    inputs = make_t2_replay_inputs(
        artifact=parent_t2,
        t1_child=t1_child,
        training_cloud=training_cloud,
        calibration_cloud=calibration_cloud,
    )
    replay = replay_t2_training_jvp(
        t1_child,
        parent_t2,
        inputs=inputs,
    )
    t2_child = LaneBParameterChild(parent_t2, replay.tangent_cores)
    manual_increment, manual_increment_score = t2_child.increment_and_score(
        tf.zeros([3], tf.float64)
    )
    tf.debugging.assert_near(manual_increment, replay.increment, atol=2e-13, rtol=0.0)
    tf.debugging.assert_near(
        manual_increment_score,
        replay.increment_score,
        atol=T2_FINITE_DIFFERENCE_ATOL,
        rtol=T2_FINITE_DIFFERENCE_RTOL,
    )
    t1_value, t1_score = t1_child.increment_and_score(
        tf.zeros([3], tf.float64)
    )
    manual_cumulative_value = t1_value + manual_increment
    manual_cumulative_score = t1_score + manual_increment_score
    tf.debugging.assert_near(
        manual_cumulative_value,
        replay.cumulative_value,
        atol=5e-13,
        rtol=0.0,
    )
    tf.debugging.assert_near(
        manual_cumulative_score,
        replay.cumulative_score,
        atol=T2_FINITE_DIFFERENCE_ATOL,
        rtol=T2_FINITE_DIFFERENCE_RTOL,
    )
    fd_rows = []
    step = tf.constant(T2_FINITE_DIFFERENCE_STEP, tf.float64)
    for parameter in range(3):
        plus = replay.finite_difference_plus[parameter]
        minus = replay.finite_difference_minus[parameter]
        observed = (plus - minus) / (2.0 * step)
        tf.debugging.assert_near(
            observed,
            replay.cumulative_score[parameter],
            atol=T2_FINITE_DIFFERENCE_ATOL,
            rtol=T2_FINITE_DIFFERENCE_RTOL,
        )
        fd_rows.append(
            {
                "parameter": parameter,
                "step": T2_FINITE_DIFFERENCE_STEP,
                "value_plus": plus,
                "value_minus": minus,
                "finite_difference": observed,
                "issued_tangent_score": replay.cumulative_score[parameter],
                "absolute_residual": tf.abs(
                    observed - replay.cumulative_score[parameter]
                ),
            }
        )
    memory = tf.config.experimental.get_memory_info("GPU:0")
    if int(memory["peak"]) > T2_MEMORY_CAP_BYTES:
        raise MemoryError("T2 replay/FD tangent exceeded the 6 GiB allocator cap")
    scalar_threshold = MATERIAL_REPLAY_ATOL + MATERIAL_REPLAY_RTOL * abs(
        float(parent_t2.value().numpy())
    )
    scalar_absolute = tf.abs(replay.cumulative_value - parent_t2.value())
    scalar_normalized = scalar_absolute / tf.constant(scalar_threshold, tf.float64)
    tf.debugging.assert_less_equal(
        scalar_normalized, tf.constant(1.0, tf.float64), "T2 material scalar replay failed"
    )
    tensors = {}
    for axis, bank in enumerate(replay.tangent_cores):
        for parameter, tangent in enumerate(bank):
            name = f"tangent_{axis:02d}_{parameter}"
            tensors[name] = _write_tensor(output / f"{name}.tensor", tangent)
    evidence = {
        "functional_replay_metrics": replay.functional_replay_metrics,
        "scalar_absolute_residual": scalar_absolute,
        "scalar_normalized_residual": scalar_normalized,
        "explanatory_maximum_core_residual": replay.maximum_core_residual,
        "explanatory_maximum_normalized_core_residual": replay.maximum_normalized_core_residual,
        "manual_increment": manual_increment,
        "manual_increment_score": manual_increment_score,
        "manual_cumulative_value": manual_cumulative_value,
        "manual_cumulative_score": manual_cumulative_score,
        "finite_difference_rows": fd_rows,
        "offline_issuer_derivative": T2_OFFLINE_ISSUER_DERIVATIVE,
        "raw_core_tangent_increment_score": replay.raw_core_tangent_increment_score,
        "scalar_consistency_radial_correction": replay.scalar_consistency_radial_correction,
        "gpu_allocator_peak_bytes": int(memory["peak"]),
    }
    identity_payload = issue_t2_training_jvp_identity_payload(
        t1_issuer_identity=str(t1_issuer["issuer_identity"]),
        t1_child_identity=t1_child.identity.hash.value,
        parent_t1_identity=parent_t1.identity.hash.value,
        parent_t2=parent_t2,
        t2_child_identity=t2_child.identity.hash.value,
        tangent_tensor_sha256={
            name: str(row["sha256"]) for name, row in tensors.items()
        },
        evidence=_jsonable(evidence),
    )
    result = {
        "schema_version": T2_ISSUER_SCHEMA,
        "status": "PASS_T1_T2_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT",
        "issuer_identity": _semantic_sha256(identity_payload),
        "issuer_identity_payload": identity_payload,
        "parent_t1_identity": parent_t1.identity.hash.value,
        "parent_t2_identity": parent_t2.identity.hash.value,
        "t1_issuer_identity": t1_issuer["issuer_identity"],
        "t2_child_identity": t2_child.identity.hash.value,
        "replay_id": replay.replay_id,
        "replay_increment": replay.increment,
        "issued_increment_tangent_score": replay.increment_score,
        "replay_cumulative_value": replay.cumulative_value,
        "issued_cumulative_tangent_score": replay.cumulative_score,
        "manual_increment": manual_increment,
        "manual_increment_score": manual_increment_score,
        "functional_screen_order": FUNCTIONAL_SCREEN_ORDER,
        "functional_screen_columns": FUNCTIONAL_SCREEN_COLUMNS,
        "functional_replay_metrics": replay.functional_replay_metrics,
        "scalar_absolute_residual": scalar_absolute,
        "scalar_normalized_residual": scalar_normalized,
        "explanatory_core_residuals": {
            "maximum_absolute_residual": replay.maximum_core_residual,
            "maximum_normalized_residual": replay.maximum_normalized_core_residual,
            "promotion_role": "explanatory_gauge_diagnostic_only",
        },
        "finite_difference_rows": fd_rows,
        "training_cloud_manifest": training_cloud.manifest_payload(),
        "calibration_cloud_manifest": calibration_cloud.manifest_payload(),
        "training_prepared_result_sha256": _sha256(TRAINING_DIR / "result.json"),
        "calibration_prepared_result_sha256": _sha256(
            CALIBRATION_DIR / "result.json"
        ),
        "tensors": tensors,
        "hard_gates": {
            "strict_t1_issuer_load": True,
            "strict_t2_prepared_cloud_load": True,
            "material_functional_replay": True,
            "material_scalar_replay": bool((scalar_normalized <= 1.0).numpy()),
            "manual_issued_tangent_parity": True,
            "independent_step_halving_fd_parity": True,
            "memory_under_6_gib": int(memory["peak"]) <= T2_MEMORY_CAP_BYTES,
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
            "full_program_control_flow": "nested_tensorflow_while_loop",
            "python_numerical_loops": False,
            "offline_derivative_method": T2_OFFLINE_ISSUER_DERIVATIVE,
            "gpu_memory_policy": dict(MEMORY_POLICY),
            "gpu_allocator": {key: int(value) for key, value in memory.items()},
            "plan": PLAN.as_posix(),
            "plan_sha256": _sha256(ROOT / PLAN),
            "wall_time_seconds": time.monotonic() - started,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "nonclaims": (
            "no T5 or later score",
            "no exact physical likelihood theorem",
            "no exact-autodiff or JVP claim",
            "issued tangents are centered-finite-difference estimates",
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
