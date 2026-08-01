"""Exact-likelihood LGSSM pilot for true-energy-corrected neural-force HMC."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.hmc_convergence import (
    RankNormalizedHMCThresholds,
    rank_normalized_hmc_diagnostics,
    rank_normalized_split_rhat_summary,
)
from bayesfilter.inference.neural_force_campaign import (
    NeuralForceTuningCandidate,
    bind_transformed_neural_force_target,
    generate_neural_force_supervision,
    select_health_aware_tuning_candidate,
    validate_value_only_endpoint_parity,
)
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce,
    NeuralForceHMCConfig,
    sample_neural_force_hmc,
)
from bayesfilter.inference.neural_force_training import (
    FrozenScalarResidualForce,
    ScalarResidualForceTrainingConfig,
    train_scalar_residual_force,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.linear.svd_factor_tf import symmetrize
from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
    load_deterministic_lgssm_exact_target,
)
from bayesfilter.testing.lgssm_neutra_gap_closure_tf import (
    load_plain_hmc_comparator_summary,
    posterior_summary,
    read_tensor_archive,
    write_tensor_archive,
)
from bayesfilter.testing.multidim_triangular_lgssm_tf import raw_truth_from_contract


EXPECTED_TARGET_SIGNATURE = (
    "f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30"
)
EXPECTED_TRANSPORT_SIGNATURE = (
    "bcbe925f2ca77996bfe05cd5b951d1a66f540327789093d0ade8fecdf0773363"
)
DIMENSION = 18
CHAIN_COUNT = 4
PASS_TRUTH_TAIL = 0.05
SEVERE_TRUTH_TAIL = 0.003
_TRANSITION_BASIS = tf.constant(
    [
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
    ],
    dtype=tf.float64,
)


class LGSSMNeuralForcePilotError(RuntimeError):
    """Raised when the exact LGSSM pilot cannot preserve its evidence contract."""


class BatchNativeLGSSMTransformedAdapter:
    """Complete transformed value/score using the admitted batched LGSSM kernel."""

    def __init__(self, *, base_adapter: Any, transport: Any) -> None:
        self.base_adapter = base_adapter
        self.transport = transport
        self.parameter_dim = DIMENSION

    def log_prob_and_grad_batch(
        self, position: Any
    ) -> tuple[tf.Tensor, tf.Tensor]:
        z = tf.convert_to_tensor(position, tf.float64)
        if z.shape.rank != 2 or z.shape[-1] != DIMENSION:
            raise ValueError("transformed LGSSM batch must have shape [row, 18]")
        raw = self.transport.forward_batch(z)
        value, raw_score, status = (
            self.base_adapter.neutra_batch_log_prob_and_grad_status(raw)
        )
        valid = tf.logical_and(
            tf.equal(tf.convert_to_tensor(status["status_code"], tf.int32), 0),
            tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
        )
        with tf.control_dependencies(
            [
                tf.debugging.assert_equal(
                    tf.reduce_all(valid), True, message="batch-native LGSSM target status"
                )
            ]
        ):
            value = tf.identity(value)
        logdet = self.transport.log_abs_det_jacobian_batch(z)
        score = self.transport.pullback_score_batch(z, raw_score)
        score += self.transport.log_abs_det_jacobian_score_batch(z)
        return value + logdet, score


def load_pilot_context(
    *,
    transport_payload_path: str | Path,
) -> Mapping[str, Any]:
    bundle = load_deterministic_lgssm_exact_target(
        expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    payload = json.loads(Path(transport_payload_path).read_text(encoding="utf-8"))
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=EXPECTED_TARGET_SIGNATURE
    )
    if loaded.manifest.transport_hash != EXPECTED_TRANSPORT_SIGNATURE:
        raise LGSSMNeuralForcePilotError("LGSSM transport semantic hash mismatch")
    transformed = BatchNativeLGSSMTransformedAdapter(
        base_adapter=bundle.adapter,
        transport=loaded.transport,
    )

    def endpoint_potential(position: tf.Tensor) -> tf.Tensor:
        z = tf.convert_to_tensor(position, tf.float64)
        raw = loaded.transport.forward_batch(z)
        raw_log_prob = lgssm_raw_log_prob_value_batch(
            bundle.fixture["observations"], raw, bundle.contract
        )
        logdet = loaded.transport.log_abs_det_jacobian_batch(z)
        return -(raw_log_prob + logdet)

    binding = bind_transformed_neural_force_target(
        adapter=transformed,
        endpoint_potential_function=endpoint_potential,
        target_signature=EXPECTED_TARGET_SIGNATURE,
        transport_signature=EXPECTED_TRANSPORT_SIGNATURE,
        dimension=DIMENSION,
    )
    return {
        "bundle": bundle,
        "loaded_transport": loaded,
        "transformed_adapter": transformed,
        "target_binding": binding,
        "plain_hmc_comparator": load_plain_hmc_comparator_summary(),
    }


@tf.function(jit_compile=True, reduce_retracing=True)
def lgssm_raw_log_prob_value_batch(
    observations: Any,
    raw_parameters: Any,
    contract: Mapping[str, Any],
) -> tf.Tensor:
    """Evaluate the registered LGSSM posterior without a parameter gradient."""

    y = tf.convert_to_tensor(observations, tf.float64)
    raw = tf.convert_to_tensor(raw_parameters, tf.float64)
    materialized = _value_only_materialization(raw, contract)
    batch_size = tf.shape(raw)[0]
    state_dim = tf.constant(4, tf.int32)
    observation_dim = tf.constant(4, tf.int32)
    mean = materialized[0]
    covariance = symmetrize(materialized[1])
    identity_state = tf.eye(state_dim, batch_shape=(batch_size,), dtype=tf.float64)
    identity_observation = tf.eye(
        observation_dim, batch_shape=(batch_size,), dtype=tf.float64
    )
    two_pi = tf.constant(2.0 * math.pi, tf.float64)

    def cond(index: tf.Tensor, *_: tf.Tensor) -> tf.Tensor:
        return index < tf.shape(y)[0]

    def body(
        index: tf.Tensor,
        current_mean: tf.Tensor,
        current_covariance: tf.Tensor,
        log_likelihood: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        predicted_mean = materialized[2] + tf.einsum(
            "bij,bj->bi", materialized[3], current_mean
        )
        predicted_covariance = symmetrize(
            materialized[3]
            @ current_covariance
            @ tf.linalg.matrix_transpose(materialized[3])
            + materialized[4]
        )
        innovation = y[index][tf.newaxis, :] - (
            materialized[5]
            + tf.einsum("bij,bj->bi", materialized[6], predicted_mean)
        )
        innovation_covariance = symmetrize(
            materialized[6]
            @ predicted_covariance
            @ tf.linalg.matrix_transpose(materialized[6])
            + materialized[7]
            + tf.constant(1.0e-9, tf.float64) * identity_observation
        )
        eigenvalues, eigenvectors = tf.linalg.eigh(innovation_covariance)
        floored = tf.maximum(eigenvalues, tf.constant(1.0e-12, tf.float64))
        projected = symmetrize(
            eigenvectors
            @ tf.linalg.diag(floored)
            @ tf.linalg.matrix_transpose(eigenvectors)
        )
        precision = eigenvectors @ tf.linalg.diag(1.0 / floored) @ tf.linalg.matrix_transpose(eigenvectors)
        solved = tf.einsum("bij,bj->bi", precision, innovation)
        contribution = -0.5 * (
            tf.cast(observation_dim, tf.float64) * tf.math.log(two_pi)
            + tf.reduce_sum(tf.math.log(floored), axis=1)
            + tf.einsum("bi,bi->b", innovation, solved)
        )
        gain = (
            predicted_covariance
            @ tf.linalg.matrix_transpose(materialized[6])
            @ precision
        )
        next_mean = predicted_mean + tf.einsum("bij,bj->bi", gain, innovation)
        joseph_left = identity_state - gain @ materialized[6]
        next_covariance = symmetrize(
            joseph_left
            @ predicted_covariance
            @ tf.linalg.matrix_transpose(joseph_left)
            + gain
            @ (
                materialized[7]
                + tf.constant(1.0e-9, tf.float64) * identity_observation
            )
            @ tf.linalg.matrix_transpose(gain)
        )
        with tf.control_dependencies(
            [tf.debugging.assert_all_finite(projected, "implemented innovation covariance")]
        ):
            return index + 1, next_mean, next_covariance, log_likelihood + contribution

    result = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0, tf.int32),
            mean,
            covariance,
            tf.zeros([batch_size], tf.float64),
        ),
        parallel_iterations=1,
    )
    prior_value = _gaussian_prior_value_only(raw, contract)
    return result[3] + prior_value


def _value_only_materialization(
    raw: tf.Tensor, contract: Mapping[str, Any]
) -> tuple[tf.Tensor, ...]:
    """Materialize only tensors used by the scalar Kalman value program."""

    batch_size = tf.shape(raw)[0]
    rho_max = tf.constant(float(contract["transform"]["rho_max"]), tf.float64)
    lower_scale = tf.constant(float(contract["transform"]["lower_scale"]), tf.float64)
    transition_values = tf.concat(
        (rho_max * tf.tanh(raw[:, :4]), lower_scale * tf.tanh(raw[:, 4:10])),
        axis=1,
    )
    transition = tf.einsum("bk,kij->bij", transition_values, _TRANSITION_BASIS)
    process_covariance = tf.linalg.diag(tf.exp(2.0 * raw[:, 10:14]))
    observation_covariance = tf.linalg.diag(tf.exp(2.0 * raw[:, 14:18]))
    kron = tf.reshape(
        tf.einsum("bij,bkl->bikjl", transition, transition),
        [batch_size, 16, 16],
    )
    system = tf.eye(16, batch_shape=(batch_size,), dtype=tf.float64) - kron
    initial_covariance = tf.reshape(
        tf.linalg.solve(system, tf.reshape(process_covariance, [batch_size, 16, 1])),
        [batch_size, 4, 4],
    )
    zeros = tf.zeros([batch_size, 4], tf.float64)
    observation_matrix = tf.broadcast_to(
        tf.eye(4, dtype=tf.float64)[tf.newaxis, :, :], [batch_size, 4, 4]
    )
    return (
        zeros,
        symmetrize(initial_covariance),
        zeros,
        transition,
        process_covariance,
        zeros,
        observation_matrix,
        observation_covariance,
    )


def _gaussian_prior_value_only(
    raw: tf.Tensor, contract: Mapping[str, Any]
) -> tf.Tensor:
    center = raw_truth_from_contract(dict(contract))[tf.newaxis, :]
    scales = tf.constant(
        [0.50] * 4 + [0.60] * 6 + [0.35] * 8, tf.float64
    )[tf.newaxis, :]
    return -0.5 * tf.reduce_sum(tf.square((raw - center) / scales), axis=1)


def prepare_supervision(
    *,
    context: Mapping[str, Any],
    warmup_archive_sidecar: str | Path,
    retained_archive_sidecar: str | Path,
    train_rows: int = 2048,
    heldout_rows: int = 1024,
) -> Mapping[str, Any]:
    warmup = read_tensor_archive(warmup_archive_sidecar)
    retained = read_tensor_archive(retained_archive_sidecar)
    warmup_flat = tf.reshape(warmup, [-1, DIMENSION])
    retained_flat = tf.reshape(retained, [-1, DIMENSION])
    if train_rows > int(warmup_flat.shape[0]) or heldout_rows > int(retained_flat.shape[0]):
        raise LGSSMNeuralForcePilotError("supervision request exceeds preserved archive")
    train_positions = warmup_flat[:train_rows]
    heldout_positions = retained_flat[:heldout_rows]
    parity_points = tf.concat((train_positions[:4], heldout_positions[:4]), axis=0)
    # Independent GPU/XLA operation ordering changes this float64 scalar by up
    # to 1.96e-7 on audited posterior/shell/tail points while all materialized
    # model tensors agree within 1.4e-17.
    parity = validate_value_only_endpoint_parity(
        context["target_binding"], parity_points, absolute_tolerance=5.0e-7
    )
    train = generate_neural_force_supervision(context["target_binding"], train_positions)
    heldout = generate_neural_force_supervision(
        context["target_binding"], heldout_positions
    )
    return {
        "train": train,
        "heldout": heldout,
        "parity": parity,
        "source_shapes": {
            "warmup": tuple(int(value) for value in warmup.shape),
            "retained": tuple(int(value) for value in retained.shape),
        },
    }


def train_recipe_grid(
    *,
    supervision: Mapping[str, Any],
    output_root: Path,
) -> Mapping[str, Any]:
    recipes = []
    for width_multiplier in (2, 4):
        for layer_count in (2, 3):
            for learning_rate in (1.0e-3, 5.0e-3):
                for batch_size in (128, 512):
                    recipe_id = (
                        f"w{width_multiplier}_l{layer_count}_lr{learning_rate:g}_b{batch_size}"
                    )
                    config = ScalarResidualForceTrainingConfig(
                        target_signature=EXPECTED_TARGET_SIGNATURE,
                        transport_signature=EXPECTED_TRANSPORT_SIGNATURE,
                        dimension=DIMENSION,
                        hidden_layers=(width_multiplier * DIMENSION,) * layer_count,
                        output_dir=output_root / "screen" / recipe_id,
                        seed=(20260717, 51000 + len(recipes)),
                        steps=500,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        heartbeat_every=100,
                        device="/GPU:0",
                        require_gpu=True,
                    )
                    result = train_scalar_residual_force(
                        train_positions=supervision["train"].positions,
                        train_potentials=supervision["train"].potentials,
                        train_forces=supervision["train"].forces,
                        heldout_positions=supervision["heldout"].positions,
                        heldout_potentials=supervision["heldout"].potentials,
                        heldout_forces=supervision["heldout"].forces,
                        config=config,
                    )
                    recipes.append(
                        {
                            "recipe_id": recipe_id,
                            "width_multiplier": width_multiplier,
                            "layer_count": layer_count,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "heldout": dict(result.metrics["heldout"]),
                            "elapsed_seconds": result.runtime_metadata["elapsed_seconds"],
                            "result_path": str(result.result_path),
                        }
                    )
    viable = tuple(
        row for row in recipes if row["heldout"]["predictions_all_finite"] is True
    )
    if not viable:
        raise LGSSMNeuralForcePilotError("no finite force-training recipe")
    selected = min(
        viable,
        key=lambda row: (
            row["heldout"]["standardized_force_rmse"],
            row["heldout"]["centered_standardized_potential_rmse"],
            row["recipe_id"],
        ),
    )
    final_config = ScalarResidualForceTrainingConfig(
        target_signature=EXPECTED_TARGET_SIGNATURE,
        transport_signature=EXPECTED_TRANSPORT_SIGNATURE,
        dimension=DIMENSION,
        hidden_layers=(selected["width_multiplier"] * DIMENSION,)
        * selected["layer_count"],
        output_dir=output_root / "final" / selected["recipe_id"],
        seed=(20260717, 52000),
        steps=5000,
        batch_size=selected["batch_size"],
        learning_rate=selected["learning_rate"],
        heartbeat_every=250,
        device="/GPU:0",
        require_gpu=True,
    )
    final = train_scalar_residual_force(
        train_positions=supervision["train"].positions,
        train_potentials=supervision["train"].potentials,
        train_forces=supervision["train"].forces,
        heldout_positions=supervision["heldout"].positions,
        heldout_potentials=supervision["heldout"].potentials,
        heldout_forces=supervision["heldout"].forces,
        config=final_config,
    )
    shell_tail = heldout_shell_tail_metrics(
        final.frozen,
        supervision["heldout"].positions,
        supervision["heldout"].potentials,
        supervision["heldout"].forces,
    )
    return {
        "recipes": tuple(recipes),
        "selected_recipe": dict(selected),
        "final": final,
        "shell_tail": shell_tail,
    }


def heldout_shell_tail_metrics(
    force: FrozenScalarResidualForce,
    positions: tf.Tensor,
    potentials: tf.Tensor,
    target_forces: tf.Tensor,
) -> Mapping[str, Any]:
    del potentials
    norms = tf.linalg.norm(positions, axis=-1)
    ordered = tf.sort(norms)
    n = int(norms.shape[0])
    q80 = ordered[int(0.8 * (n - 1))]
    q95 = ordered[int(0.95 * (n - 1))]
    predicted = force.force(positions)
    error = tf.linalg.norm(predicted - target_forces, axis=-1)

    def summary(mask: tf.Tensor) -> Mapping[str, Any]:
        selected = tf.boolean_mask(error, mask)
        return {
            "count": int(tf.size(selected).numpy()),
            "force_error_mean_l2": float(tf.reduce_mean(selected).numpy()),
            "force_error_max_l2": float(tf.reduce_max(selected).numpy()),
            "role": "nomination_or_veto_only",
        }

    return {
        "central": summary(norms < q80),
        "shell": summary(tf.logical_and(norms >= q80, norms < q95)),
        "tail": summary(norms >= q95),
        "all_predictions_finite": bool(tf.reduce_all(tf.math.is_finite(predicted)).numpy()),
    }


def tune_force(
    *,
    force: FrozenPositionOnlyForce,
    target: Any,
    initial_position: tf.Tensor,
    transform: Any,
    step_sizes: Sequence[float],
    leapfrog_steps: Sequence[int],
    seed_offset: int,
    synchronize_timing: bool = False,
) -> Mapping[str, Any]:
    dimension = int(tf.convert_to_tensor(initial_position).shape[-1])
    current = target.function(initial_position)
    rows = []
    evidence_rows = []
    for step_size in step_sizes:
        for steps in leapfrog_steps:
            config = NeuralForceHMCConfig(
                step_size=step_size,
                num_leapfrog_steps=steps,
                inverse_mass_diagonal=(1.0,) * dimension,
                dtype="float64",
            )

            @tf.function(jit_compile=True, reduce_retracing=True)
            def run(position: tf.Tensor, potential: tf.Tensor, seed: tf.Tensor):
                return sample_neural_force_hmc(
                    position,
                    potential,
                    force,
                    target,
                    config,
                    num_warmup=0,
                    num_results=500,
                    seed=seed,
                )

            started = time.monotonic()
            chain = run(
                initial_position,
                current,
                tf.constant((20260717, seed_offset + len(rows)), tf.int32),
            )
            if synchronize_timing:
                synchronize_chain(chain)
            elapsed = time.monotonic() - started
            raw = transform_samples(transform, chain.positions)
            rhat = rank_normalized_split_rhat_summary(raw, rhat_max=1.05)
            health = bool(
                tf.reduce_all(chain.finite_status).numpy()
                and tf.reduce_all(tf.math.is_finite(chain.delta_h)).numpy()
                and tf.reduce_all(chain.log_acceptance_ratio > -1000.0).numpy()
            )
            candidate = NeuralForceTuningCandidate(
                    candidate_id=f"eps{step_size:g}_l{steps}",
                    step_size=step_size,
                    num_leapfrog_steps=steps,
                    health_passed=health,
                    modern_rhat=float(rhat["max_finite_rhat"] or float("inf")),
                    maximum_absolute_delta_h=float(
                        tf.reduce_max(tf.abs(chain.delta_h)).numpy()
                    ),
                    acceptance_rate=float(
                        tf.reduce_mean(tf.cast(chain.accepted, tf.float64)).numpy()
                    ),
                )
            rows.append(candidate)
            evidence_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "step_size": candidate.step_size,
                    "num_leapfrog_steps": candidate.num_leapfrog_steps,
                    "health_passed": candidate.health_passed,
                    "modern_rhat": candidate.modern_rhat,
                    "maximum_absolute_delta_h": candidate.maximum_absolute_delta_h,
                    "acceptance_rate": candidate.acceptance_rate,
                    "elapsed_seconds": elapsed,
                    "timing_synchronized": bool(synchronize_timing),
                    "timing_scope": "cold_compile_plus_500_transition_batches",
                    "endpoint_batch_invocations": 500,
                    "force_batch_invocations": 500 * (steps + 1),
                    "role": "tuning_nomination_only",
                }
            )
    selected = select_health_aware_tuning_candidate(
        rows, rhat_max=1.05, maximum_absolute_delta_h=100.0
    )
    return {
        "selected": selected,
        "rows": tuple(evidence_rows),
    }


def run_sequential_arm(
    *,
    arm_id: str,
    force: FrozenPositionOnlyForce,
    target: Any,
    initial_position: tf.Tensor,
    transform: Any,
    parameter_names: Sequence[str],
    step_size: float,
    num_leapfrog_steps: int,
    output_root: Path,
    seed_base: int,
    precompile: bool = False,
    synchronize_timing: bool = False,
) -> Mapping[str, Any]:
    dimension = int(tf.convert_to_tensor(initial_position).shape[-1])
    config = NeuralForceHMCConfig(
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        inverse_mass_diagonal=(1.0,) * dimension,
        dtype="float64",
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def run_chunk(position: tf.Tensor, potential: tf.Tensor, seed: tf.Tensor):
        return sample_neural_force_hmc(
            position,
            potential,
            force,
            target,
            config,
            num_warmup=0,
            num_results=1000,
            seed=seed,
        )

    total_started = time.monotonic()
    position = tf.convert_to_tensor(initial_position, tf.float64)
    potential = target.function(position)
    tf.reduce_sum(potential).numpy()
    compile_probe_seconds = 0.0
    if precompile:
        compile_started = time.monotonic()
        probe = run_chunk(
            position,
            potential,
            tf.constant((20260717, seed_base - 1), tf.int32),
        )
        synchronize_chain(probe)
        compile_probe_seconds = time.monotonic() - compile_started
    sampling_execution_seconds = 0.0
    warmup_chunks = []
    warmup_traces = []
    warmup_checks = []
    hard_vetoes = []
    for index in range(10):
        chunk_started = time.monotonic()
        chain = run_chunk(
            position,
            potential,
            tf.constant((20260717, seed_base + index), tf.int32),
        )
        if synchronize_timing:
            synchronize_chain(chain)
        sampling_execution_seconds += time.monotonic() - chunk_started
        position = chain.positions[-1]
        potential = chain.potentials[-1]
        warmup_chunks.append(chain.positions)
        warmup_traces.append(chain)
        health = chain_health(chain)
        if not health["passed"]:
            hard_vetoes.append("warmup_chunk_health_failed")
        cumulative = tf.concat(warmup_chunks, axis=0)
        raw = transform_samples(transform, cumulative[-1000:])
        rhat = rank_normalized_split_rhat_summary(raw, rhat_max=1.05)
        passed = bool(index + 1 >= 2 and health["passed"] and rhat["passed"])
        warmup_checks.append(
            {
                "results_per_chain": (index + 1) * 1000,
                "health": health,
                "modern_rhat": rhat,
                "passed": passed,
            }
        )
        if passed or hard_vetoes:
            break
    warmup_passed = bool(warmup_checks[-1]["passed"] and not hard_vetoes)
    retained_chunks = []
    retained_traces = []
    retained_checks = []
    if warmup_passed:
        for index in range(10):
            chunk_started = time.monotonic()
            chain = run_chunk(
                position,
                potential,
                tf.constant((20260717, seed_base + 100 + index), tf.int32),
            )
            if synchronize_timing:
                synchronize_chain(chain)
            sampling_execution_seconds += time.monotonic() - chunk_started
            position = chain.positions[-1]
            potential = chain.potentials[-1]
            retained_chunks.append(chain.positions)
            retained_traces.append(chain)
            health = chain_health(chain)
            if not health["passed"]:
                hard_vetoes.append("retained_chunk_health_failed")
            cumulative = tf.concat(retained_chunks, axis=0)
            raw = transform_samples(transform, cumulative)
            diagnostic = rank_normalized_hmc_diagnostics(
                raw,
                parameter_names=parameter_names,
                thresholds=RankNormalizedHMCThresholds(
                    rhat_max=1.01, bulk_ess_min=1000.0, tail_ess_min=400.0
                ),
            )
            passed = bool(health["passed"] and diagnostic["passed"])
            retained_checks.append(
                {
                    "results_per_chain": (index + 1) * 1000,
                    "health": health,
                    "full_convergence": diagnostic,
                    "passed": passed,
                }
            )
            if passed or hard_vetoes:
                break
    warmup = tf.concat(warmup_chunks, axis=0)
    retained = (
        tf.concat(retained_chunks, axis=0)
        if retained_chunks
        else tf.zeros([0, int(position.shape[0]), dimension], tf.float64)
    )
    raw_warmup = transform_samples(transform, warmup)
    raw_retained = transform_samples(transform, retained)
    archive_root = output_root / arm_id / "samples"
    archive_root.mkdir(parents=True, exist_ok=True)
    archives = {
        "warmup_z": write_tensor_archive(
            archive_root / "warmup_z.tftensor", warmup, metadata={"stage": "warmup", "arm": arm_id}
        ),
        "warmup_raw": write_tensor_archive(
            archive_root / "warmup_raw.tftensor", raw_warmup, metadata={"stage": "warmup", "arm": arm_id}
        ),
        "warmup_trace": write_tensor_archive(
            archive_root / "warmup_trace.tftensor",
            stack_energy_trace(warmup_traces),
            metadata={
                "stage": "warmup",
                "arm": arm_id,
                "columns": (
                    "initial_potential",
                    "final_potential",
                    "initial_kinetic",
                    "final_kinetic",
                    "delta_h",
                    "log_acceptance_ratio",
                ),
            },
        ),
    }
    if retained_chunks:
        archives.update(
            {
                "retained_z": write_tensor_archive(
                    archive_root / "retained_z.tftensor", retained, metadata={"stage": "retained", "arm": arm_id}
                ),
                "retained_raw": write_tensor_archive(
                    archive_root / "retained_raw.tftensor", raw_retained, metadata={"stage": "retained", "arm": arm_id}
                ),
                "retained_trace": write_tensor_archive(
                    archive_root / "retained_trace.tftensor",
                    stack_energy_trace(retained_traces),
                    metadata={
                        "stage": "retained",
                        "arm": arm_id,
                        "columns": (
                            "initial_potential",
                            "final_potential",
                            "initial_kinetic",
                            "final_kinetic",
                            "delta_h",
                            "log_acceptance_ratio",
                        ),
                    },
                ),
            }
        )
    elapsed = time.monotonic() - total_started
    accepted = tf.concat(
        [chain.accepted for chain in (*warmup_traces, *retained_traces)], axis=0
    )
    return {
        "arm_id": arm_id,
        "passed": bool(
            warmup_passed
            and retained_checks
            and retained_checks[-1]["passed"]
            and not hard_vetoes
        ),
        "step_size": step_size,
        "num_leapfrog_steps": num_leapfrog_steps,
        "warmup_results_per_chain": int(warmup.shape[0]),
        "retained_results_per_chain": int(retained.shape[0]),
        "warmup_checks": tuple(warmup_checks),
        "retained_checks": tuple(retained_checks),
        "hard_vetoes": tuple(hard_vetoes),
        "acceptance_rate": float(tf.reduce_mean(tf.cast(accepted, tf.float64)).numpy()),
        "elapsed_seconds": elapsed,
        "sampling_execution_seconds": sampling_execution_seconds,
        "compile_probe_seconds": compile_probe_seconds,
        "compile_included_in_sampling_execution_seconds": not precompile,
        "timing_synchronized": bool(synchronize_timing),
        "timing_scope": "synchronized_warm_chunk_execution" if precompile else "cold_mixed_execution",
        "archives": archives,
        "private_retained_raw": raw_retained,
        "endpoint_batch_invocations": int(warmup.shape[0] + retained.shape[0]),
        "endpoint_scalar_values": int(
            (warmup.shape[0] + retained.shape[0]) * CHAIN_COUNT
        ),
        "force_batch_invocations": int(
            (warmup.shape[0] + retained.shape[0]) * (num_leapfrog_steps + 1)
        ),
        "full_energy_identity_max_error": float(
            tf.reduce_max(
                tf.abs(
                    stack_energy_trace((*warmup_traces, *retained_traces))[..., 4]
                    - (
                        stack_energy_trace((*warmup_traces, *retained_traces))[..., 1]
                        + stack_energy_trace((*warmup_traces, *retained_traces))[..., 3]
                        - stack_energy_trace((*warmup_traces, *retained_traces))[..., 0]
                        - stack_energy_trace((*warmup_traces, *retained_traces))[..., 2]
                    )
                )
            ).numpy()
        ),
    }


def stack_energy_trace(chains: Sequence[Any]) -> tf.Tensor:
    return tf.stack(
        (
            tf.concat([chain.initial_potential for chain in chains], axis=0),
            tf.concat([chain.final_potential for chain in chains], axis=0),
            tf.concat([chain.initial_kinetic for chain in chains], axis=0),
            tf.concat([chain.final_kinetic for chain in chains], axis=0),
            tf.concat([chain.delta_h for chain in chains], axis=0),
            tf.concat([chain.log_acceptance_ratio for chain in chains], axis=0),
        ),
        axis=-1,
    )


def chain_health(chain: Any) -> Mapping[str, Any]:
    finite = bool(
        tf.reduce_all(chain.finite_status).numpy()
        and tf.reduce_all(tf.math.is_finite(chain.delta_h)).numpy()
    )
    energy = bool(tf.reduce_all(chain.log_acceptance_ratio > -1000.0).numpy())
    endpoint_counts = bool(tf.reduce_all(tf.equal(chain.endpoint_call_count, 1)).numpy())
    return {
        "passed": bool(finite and energy and endpoint_counts),
        "all_finite": finite,
        "energy_error_veto_clear": energy,
        "one_new_endpoint_call_per_transition": endpoint_counts,
        "maximum_absolute_delta_h": float(tf.reduce_max(tf.abs(chain.delta_h)).numpy()),
    }


def synchronize_chain(chain: Any) -> None:
    """Block until every returned chain tensor has completed device execution."""

    tensors = tuple(tf.convert_to_tensor(value) for value in chain)
    tf.add_n(
        [tf.reduce_sum(tf.cast(value, tf.float64)) for value in tensors]
    ).numpy()


def transform_samples(transport: Any, samples: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(samples, tf.float64)
    shape = tf.shape(values)
    dimension = int(values.shape[-1])
    flat = tf.reshape(values, [-1, dimension])
    raw = transport.forward_batch(flat)
    return tf.reshape(raw, shape)


def truth_tail_summary(samples: tf.Tensor, truth: tf.Tensor, parameter_names: Sequence[str]) -> Mapping[str, Any]:
    dimension = int(tf.convert_to_tensor(samples).shape[-1])
    values = tf.reshape(tf.convert_to_tensor(samples, tf.float64), [-1, dimension])
    truth = tf.convert_to_tensor(truth, tf.float64)
    less = tf.reduce_sum(tf.cast(values < truth[tf.newaxis, :], tf.float64), axis=0)
    equal = tf.reduce_sum(tf.cast(values == truth[tf.newaxis, :], tf.float64), axis=0)
    total = tf.cast(tf.shape(values)[0], tf.float64)
    cdf = (less + 0.5 * equal + 0.5) / (total + 1.0)
    p_truth = 2.0 * tf.minimum(cdf, 1.0 - cdf)
    rows = []
    for index, name in enumerate(parameter_names):
        value = float(p_truth[index].numpy())
        status = "SEVERE" if value < SEVERE_TRUTH_TAIL else "MARGINAL" if value < PASS_TRUTH_TAIL else "PASS"
        rows.append({"parameter": name, "truth": float(truth[index].numpy()), "p_truth": value, "status": status})
    minimum = min(row["p_truth"] for row in rows)
    return {
        "passed": minimum >= PASS_TRUTH_TAIL,
        "minimum_p_truth": minimum,
        "marginal_parameters": tuple(row["parameter"] for row in rows if row["status"] == "MARGINAL"),
        "severe_parameters": tuple(row["parameter"] for row in rows if row["status"] == "SEVERE"),
        "rows": tuple(rows),
        "definition": "F=(n_less+0.5*n_equal+0.5)/(N+1); p_truth=2*min(F,1-F)",
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items() if not str(key).startswith("private_")}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return json_ready(value.numpy().tolist())
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "EXPECTED_TARGET_SIGNATURE",
    "EXPECTED_TRANSPORT_SIGNATURE",
    "LGSSMNeuralForcePilotError",
    "file_sha256",
    "json_ready",
    "lgssm_raw_log_prob_value_batch",
    "load_pilot_context",
    "prepare_supervision",
    "run_sequential_arm",
    "synchronize_chain",
    "train_recipe_grid",
    "transform_samples",
    "truth_tail_summary",
    "tune_force",
]
