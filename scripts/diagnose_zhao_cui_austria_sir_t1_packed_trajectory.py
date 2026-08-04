#!/usr/bin/env python3
"""Localize trajectory drift between the admitted and packed T1 programs."""

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

from bayesfilter.highdim.stochastic_density_training import (  # noqa: E402
    TrainableFunctionalTT,
    make_adam_optimizer,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (  # noqa: E402
    balanced_initial_cores,
    make_compiled_train_step,
    trainer_config,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (  # noqa: E402
    ADAM_BETA_1,
    ADAM_BETA_2,
    ADAM_EPSILON,
    MEMORY_CAP_BYTES,
    _t1_functional_replay_metrics,
    prepare_t1_replay_inputs,
)
from bayesfilter.highdim.zhao_cui_austria_sir_packed_xla_tf import (  # noqa: E402
    PACKED_XLA_POLICY_ID,
    pack_cores,
    packed_adam_apply_gradients,
    packed_amplitude,
    packed_per_core_regularizers,
    packed_square_mass,
    packed_tuple_global_norm,
)


DTYPE = tf.float64
CHECKPOINTS = frozenset((1, 16, 32, 64, 96))
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
    settings = parent.settings
    config = trainer_config(settings)
    initial_cores = balanced_initial_cores(settings, config.product_basis)
    authority = TrainableFunctionalTT(config, initial_cores=initial_cores)
    authority_step = make_compiled_train_step(authority, make_adam_optimizer(config))

    learning_rate = tf.cast(tf.constant(settings.learning_rate, tf.float32), DTYPE)
    clip = tf.constant(settings.gradient_clip_norm, DTYPE)
    tau = tf.constant(settings.tau, DTYPE)
    l1_weight = tf.constant(settings.l1_weight, DTYPE)
    l2_weight = tf.constant(settings.l2_weight, DTYPE)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def packed_step(
        cores,
        momentums,
        velocities,
        step,
        basis_values,
        target_sqrt,
        integration_weights,
    ):
        raw_alpha = integration_weights * (tf.square(target_sqrt) + tau)
        alpha = raw_alpha / tf.reduce_sum(raw_alpha)
        with tf.GradientTape() as tape:
            tape.watch(cores)
            amplitude = packed_amplitude(cores, basis_values)
            rho = tf.square(amplitude) + tau
            square_mass = packed_square_mass(cores, inputs.mass_matrices)
            l1, l2 = packed_per_core_regularizers(cores, inputs.packed_mask)
            cross_entropy = -tf.reduce_sum(alpha * tf.math.log(rho))
            log_normalizer = tf.math.log(square_mass + tau)
            regularization = l1_weight * l1 + l2_weight * l2
            total = cross_entropy + log_normalizer + regularization
        gradients = tape.gradient(total, cores)
        gradient_norm = packed_tuple_global_norm(gradients, inputs.packed_mask)
        next_cores, next_m, next_v = packed_adam_apply_gradients(
            cores,
            momentums,
            velocities,
            gradients,
            inputs.packed_mask,
            step=step,
            learning_rate=learning_rate,
            gradient_clip_norm=clip,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
        )
        terms = tf.stack(
            (
                total,
                cross_entropy,
                log_normalizer,
                regularization,
                gradient_norm,
                tf.reduce_min(rho),
                tf.reduce_max(rho),
            )
        )
        return next_cores, next_m, next_v, gradients, terms

    @tf.function(jit_compile=True, reduce_retracing=True)
    def functional_compare(candidate, reference):
        observed_inputs = inputs.__class__(
            training_physical_points=inputs.training_physical_points,
            training_local_points=inputs.training_local_points,
            training_origin_target_sqrt=inputs.training_origin_target_sqrt,
            training_integration_weights=inputs.training_integration_weights,
            calibration_physical_points=inputs.calibration_physical_points,
            calibration_origin_log_likelihood=inputs.calibration_origin_log_likelihood,
            training_basis_values=inputs.training_basis_values,
            calibration_basis_values=inputs.calibration_basis_values,
            mass_matrices=inputs.mass_matrices,
            initial_packed_cores=inputs.initial_packed_cores,
            parent_packed_cores=reference,
            packed_mask=inputs.packed_mask,
            observation=inputs.observation,
            training_batch_indices=inputs.training_batch_indices,
        )
        return _t1_functional_replay_metrics(candidate, observed_inputs, tau)

    packed = inputs.initial_packed_cores * inputs.packed_mask
    momentums = tf.zeros_like(packed)
    velocities = tf.zeros_like(packed)
    rows = []
    for step_index in range(settings.train_steps):
        indices = inputs.training_batch_indices[step_index]
        local_points = tf.gather(inputs.training_local_points, indices)
        basis_values = tf.gather(inputs.training_basis_values, indices)
        target_sqrt = tf.gather(inputs.training_origin_target_sqrt, indices)
        integration_weights = tf.gather(inputs.training_integration_weights, indices)
        authority_terms = authority_step(
            local_points, target_sqrt, integration_weights
        )
        packed, momentums, velocities, packed_gradient, packed_terms = packed_step(
            packed,
            momentums,
            velocities,
            tf.constant(step_index + 1, tf.int32),
            basis_values,
            target_sqrt,
            integration_weights,
        )
        checkpoint = step_index + 1
        if checkpoint in CHECKPOINTS:
            authority_packed = pack_cores(authority.variables)
            authority_term_vector = tf.stack(authority_terms)
            functional_passed, functional_metrics = functional_compare(
                packed, authority_packed
            )
            active_residual = tf.abs(packed - authority_packed) * inputs.packed_mask
            rows.append(
                {
                    "step": checkpoint,
                    "authority_terms": authority_term_vector,
                    "packed_terms": packed_terms,
                    "maximum_term_residual": tf.reduce_max(
                        tf.abs(authority_term_vector - packed_terms)
                    ),
                    "packed_gradient_norm": packed_terms[4],
                    "packed_gradient_maximum_absolute": tf.reduce_max(
                        tf.abs(packed_gradient) * inputs.packed_mask
                    ),
                    "maximum_core_residual": tf.reduce_max(active_residual),
                    "functional_material_pass": functional_passed,
                    "functional_metrics": functional_metrics,
                }
            )

    memory = tf.config.experimental.get_memory_info("GPU:0")
    result = {
        "schema": "bayesfilter.zhao_cui_austria_sir_t1_packed_trajectory_diagnostic.v1",
        "status": "COMPLETE_T1_PACKED_TRAJECTORY_DIAGNOSTIC",
        "parent_identity": parent.identity.hash.value,
        "packed_xla_policy_id": PACKED_XLA_POLICY_ID,
        "term_order": (
            "total_loss",
            "cross_entropy",
            "log_normalizer_before_update",
            "regularization",
            "gradient_norm",
            "rho_min",
            "rho_max",
        ),
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
        "checkpoints": rows,
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
            "candidate_jit_compile": True,
            "candidate_control_flow": "tensorflow_while_loop_kernels",
            "authority_role": "historical_diagnostic_only",
            "authority_python_step_driver": True,
            "gpu_memory_policy": dict(MEMORY_POLICY),
            "gpu_allocator": {key: int(item) for key, item in memory.items()},
            "memory_gate": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "plan": PLAN.as_posix(),
            "wall_time_seconds": time.monotonic() - started,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "nonclaims": (
            "diagnostic localization only",
            "historical tuple authority is not a claim-bearing candidate",
            "no JVP or finite-difference admission",
            "no T1 issuer admission",
            "no T2 or HMC",
        ),
    }
    encoded = json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    (output / "result.json").write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
