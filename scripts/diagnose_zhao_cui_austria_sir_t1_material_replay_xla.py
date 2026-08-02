#!/usr/bin/env python3
"""Run the bounded real-cloud T1 packed-XLA primal diagnostic."""

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


GPU_MEMORY_LIMIT_MIB = 6144
MEMORY_POLICY = configure_tensorflow_gpu_memory_limit(
    tf, memory_limit_mib=GPU_MEMORY_LIMIT_MIB, require_gpu=True
)

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (  # noqa: E402
    MEMORY_CAP_BYTES,
    _t1_functional_replay_metrics,
    _make_t1_compiled_primal,
    prepare_t1_replay_inputs,
)
from bayesfilter.highdim.zhao_cui_austria_sir_packed_xla_tf import (  # noqa: E402
    MATERIAL_REPLAY_ATOL,
    MATERIAL_REPLAY_POLICY_ID,
    MATERIAL_REPLAY_RTOL,
    PACKED_XLA_POLICY_ID,
    material_replay_metrics,
    material_positive_value_metrics,
)


PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    parent = load_lane_b_t1_artifact_v1_compat(PARENT_DIR)
    inputs = prepare_t1_replay_inputs(parent)
    replayed, value = _make_t1_compiled_primal(parent, inputs)(
        tf.zeros([3], tf.float64)
    )
    passed, maximum_absolute, maximum_normalized = material_replay_metrics(
        replayed, inputs.parent_packed_cores, inputs.packed_mask
    )
    functional_passed, functional_metrics = _t1_functional_replay_metrics(
        replayed, inputs, tf.constant(parent.settings.tau, tf.float64)
    )
    scalar_passed, scalar_absolute, scalar_normalized, scalar_log_residual = (
        material_positive_value_metrics(
            tf.exp(value), tf.exp(parent.value())
        )
    )
    memory = tf.config.experimental.get_memory_info("GPU:0")
    admitted = tf.logical_and(functional_passed, scalar_passed)
    result = {
        "schema": "bayesfilter.zhao_cui_austria_sir_t1_material_replay_xla_diagnostic.v1",
        "status": (
            "PASS_T1_MATERIAL_REPLAY_XLA_DIAGNOSTIC"
            if bool(admitted.numpy()) and int(memory["peak"]) <= MEMORY_CAP_BYTES
            else "FAIL_T1_MATERIAL_REPLAY_XLA_DIAGNOSTIC"
        ),
        "parent_identity": parent.identity.hash.value,
        "parent_value": parent.value(),
        "replay_value": value,
        "material_replay_policy_id": MATERIAL_REPLAY_POLICY_ID,
        "packed_xla_policy_id": PACKED_XLA_POLICY_ID,
        "material_core_atol": MATERIAL_REPLAY_ATOL,
        "material_core_rtol": MATERIAL_REPLAY_RTOL,
        "maximum_core_residual": maximum_absolute,
        "maximum_normalized_core_residual": maximum_normalized,
        "material_core_replay_explanatory_only": passed,
        "functional_screen_order": (
            "training_full_density",
            "calibration_full_density",
            "training_prefix_marginal",
            "calibration_prefix_marginal",
        ),
        "functional_screen_columns": (
            "maximum_absolute_residual",
            "maximum_normalized_residual",
            "maximum_log_residual",
        ),
        "functional_replay_metrics": functional_metrics,
        "material_functional_replay": functional_passed,
        "scalar_absolute_residual": scalar_absolute,
        "scalar_normalized_residual": scalar_normalized,
        "scalar_log_residual": scalar_log_residual,
        "material_scalar_replay": scalar_passed,
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
            "gpu_memory_policy": dict(MEMORY_POLICY),
            "gpu_allocator": {key: int(item) for key, item in memory.items()},
            "plan": PLAN.as_posix(),
            "wall_time_seconds": time.monotonic() - started,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "nonclaims": (
            "diagnostic primal only",
            "no JVP or finite-difference admission",
            "no T1 issuer admission",
            "no bitwise replay claim",
            "core residuals are explanatory gauge diagnostics only",
            "no T2 or HMC",
        ),
    }
    (output / "result.json").write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    if result["status"] != "PASS_T1_MATERIAL_REPLAY_XLA_DIAGNOSTIC":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
