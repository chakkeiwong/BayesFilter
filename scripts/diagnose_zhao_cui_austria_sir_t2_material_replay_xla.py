#!/usr/bin/env python3
"""Run the bounded real-cloud T2 packed-XLA primal diagnostic."""

from __future__ import annotations

import argparse
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


MEMORY_POLICY = configure_tensorflow_gpu_memory_limit(
    tf, memory_limit_mib=6144, require_gpu=True
)

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_prepared_tf import (  # noqa: E402
    load_t2_prepared_cloud,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_training_jvp_tf import (  # noqa: E402
    T2_MEMORY_CAP_BYTES,
    _make_t2_compiled_primal,
    _t2_functional_replay_metrics,
    make_t2_replay_inputs,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (  # noqa: E402
    MATERIAL_REPLAY_ATOL,
    MATERIAL_REPLAY_POLICY_ID,
    MATERIAL_REPLAY_RTOL,
    load_t1_training_jvp_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (  # noqa: E402
    load_selected_t2_parameter_parent_compat,
)


PARENT_T1_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)
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
    training, _training_payload = load_t2_prepared_cloud(TRAINING_DIR)
    calibration, _calibration_payload = load_t2_prepared_cloud(CALIBRATION_DIR)
    inputs = make_t2_replay_inputs(
        artifact=parent_t2,
        t1_child=t1_child,
        training_cloud=training,
        calibration_cloud=calibration,
    )
    cores, increment, cumulative = _make_t2_compiled_primal(
        t1_child, parent_t2, inputs
    )(tf.zeros([3], tf.float64))
    functional_passed, metrics = _t2_functional_replay_metrics(
        cores, parent_t2, inputs
    )
    scalar_absolute = tf.abs(cumulative - parent_t2.value())
    scalar_threshold = tf.constant(MATERIAL_REPLAY_ATOL, tf.float64) + tf.constant(
        MATERIAL_REPLAY_RTOL, tf.float64
    ) * tf.abs(parent_t2.value())
    scalar_normalized = scalar_absolute / scalar_threshold
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = bool(functional_passed.numpy()) and bool(
        (scalar_normalized <= 1.0).numpy()
    ) and int(memory["peak"]) <= T2_MEMORY_CAP_BYTES
    result = {
        "schema": "bayesfilter.zhao_cui_austria_sir_t2_material_replay_xla_diagnostic.v1",
        "status": (
            "PASS_T2_MATERIAL_REPLAY_XLA_DIAGNOSTIC"
            if passed
            else "FAIL_T2_MATERIAL_REPLAY_XLA_DIAGNOSTIC"
        ),
        "parent_t1_identity": parent_t1.identity.hash.value,
        "parent_t2_identity": parent_t2.identity.hash.value,
        "t1_issuer_identity": t1_issuer["issuer_identity"],
        "material_replay_policy_id": MATERIAL_REPLAY_POLICY_ID,
        "functional_replay_metrics": metrics,
        "replay_increment": increment,
        "parent_increment": parent_t2.increment(),
        "replay_cumulative_value": cumulative,
        "parent_cumulative_value": parent_t2.value(),
        "scalar_absolute_residual": scalar_absolute,
        "scalar_normalized_residual": scalar_normalized,
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
            "gpu_memory_policy": dict(MEMORY_POLICY),
            "gpu_allocator": {key: int(value) for key, value in memory.items()},
            "wall_time_seconds": time.monotonic() - started,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "nonclaims": (
            "primal diagnostic only",
            "no T2 tangent or score issuer admission",
            "core residuals are not promotion evidence",
            "no later horizon or HMC",
        ),
    }
    encoded = json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    (output / "result.json").write_text(encoded)
    print(encoded)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
