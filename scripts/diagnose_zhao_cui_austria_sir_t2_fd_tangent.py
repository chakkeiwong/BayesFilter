#!/usr/bin/env python3
"""Localize T2 finite-difference core-tangent score disagreement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


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
    T2_FINITE_DIFFERENCE_STEP,
    make_t2_replay_inputs,
    replay_t2_training_jvp,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (  # noqa: E402
    LaneBT2Artifact,
    issue_lane_b_t2_identity,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (  # noqa: E402
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
    parent_t1 = load_lane_b_t1_artifact_v1_compat(PARENT_T1_DIR)
    t1_child, t1_issuer = load_t1_training_jvp_child(
        args.t1_issuer_dir.resolve(), parent=parent_t1
    )
    parent_t2 = load_selected_t2_parameter_parent_compat(
        PARENT_T2_DIR, parent_artifact=parent_t1
    )
    training, _ = load_t2_prepared_cloud(TRAINING_DIR)
    calibration, _ = load_t2_prepared_cloud(CALIBRATION_DIR)
    inputs = make_t2_replay_inputs(
        artifact=parent_t2,
        t1_child=t1_child,
        training_cloud=training,
        calibration_cloud=calibration,
    )
    replay = replay_t2_training_jvp(t1_child, parent_t2, inputs=inputs)
    t2_child = LaneBParameterChild(parent_t2, replay.tangent_cores)
    replay_identity = issue_lane_b_t2_identity(
        parent_artifact=parent_t1,
        settings=parent_t2.settings,
        frame=parent_t2.frame,
        cores=replay.cores,
        shift_constant=parent_t2.shift_constant,
        calibration_estimate=parent_t2.calibration_estimate,
        validation_estimate=parent_t2.validation_estimate,
        training_cloud_manifest=parent_t2.training_cloud_manifest,
        validation_cloud_manifest=parent_t2.validation_cloud_manifest,
        source_hashes=parent_t2.source_hashes,
    )
    replay_parent = LaneBT2Artifact(
        parent_artifact=parent_t1,
        settings=parent_t2.settings,
        frame=parent_t2.frame,
        cores=replay.cores,
        shift_constant=parent_t2.shift_constant,
        calibration_estimate=parent_t2.calibration_estimate,
        validation_estimate=parent_t2.validation_estimate,
        training_cloud_manifest=parent_t2.training_cloud_manifest,
        validation_cloud_manifest=parent_t2.validation_cloud_manifest,
        source_hashes=parent_t2.source_hashes,
        identity=replay_identity,
    )
    replay_base_child = LaneBParameterChild(replay_parent, replay.tangent_cores)
    origin = tf.zeros([3], tf.float64)
    _t1_value, t1_manual_score = t1_child.increment_and_score(origin)
    _increment, manual_increment_score = t2_child.increment_and_score(origin)
    _replay_increment, replay_base_increment_score = (
        replay_base_child.increment_and_score(origin)
    )
    manual_cumulative_score = t1_manual_score + manual_increment_score
    independent_cumulative_score = (
        replay.finite_difference_plus - replay.finite_difference_minus
    ) / tf.constant(2.0 * T2_FINITE_DIFFERENCE_STEP, tf.float64)
    result = {
        "schema": "bayesfilter.zhao_cui_austria_sir_t2_fd_tangent_diagnostic.v1",
        "status": "COMPLETE_T2_FD_TANGENT_DIAGNOSTIC",
        "t1_issuer_identity": t1_issuer["issuer_identity"],
        "direct_tangent_step_increment_score": replay.increment_score,
        "raw_core_tangent_increment_score": replay.raw_core_tangent_increment_score,
        "scalar_consistency_radial_correction": replay.scalar_consistency_radial_correction,
        "manual_core_tangent_increment_score": manual_increment_score,
        "increment_manual_minus_direct": manual_increment_score
        - replay.increment_score,
        "replay_base_core_tangent_increment_score": replay_base_increment_score,
        "replay_base_minus_direct_increment": replay_base_increment_score
        - replay.increment_score,
        "replay_base_parent_identity": replay_parent.identity.hash.value,
        "direct_tangent_step_cumulative_score": replay.cumulative_score,
        "manual_core_tangent_cumulative_score": manual_cumulative_score,
        "independent_fd_cumulative_score": independent_cumulative_score,
        "manual_minus_direct_cumulative": manual_cumulative_score
        - replay.cumulative_score,
        "independent_minus_direct_cumulative": independent_cumulative_score
        - replay.cumulative_score,
        "functional_replay_metrics": replay.functional_replay_metrics,
        "explanatory_core_residual": replay.maximum_core_residual,
        "gpu_memory_policy": dict(MEMORY_POLICY),
        "gpu_allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "nonclaims": (
            "diagnostic only",
            "no T2 issuer admission",
            "no exact derivative or autodiff claim",
            "no later horizon or HMC",
        ),
    }
    encoded = json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    (output / "result.json").write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
