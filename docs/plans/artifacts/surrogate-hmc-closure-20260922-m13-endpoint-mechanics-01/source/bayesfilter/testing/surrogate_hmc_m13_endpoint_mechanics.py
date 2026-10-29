"""Bounded M13 endpoint-corrected surrogate-HMC mechanics diagnostic.

The canonical analytical DCTR score is used only as a frozen proposal field.
Metropolis correction uses a separate deterministic full-support FP64
importance endpoint. This worker does not issue HMC tuning authority or a
posterior claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.ledh_range_bearing_target_tf import make_range_bearing_target
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    NeuralForceHMCConfig,
    PositionOnlyForceEvaluation,
    TargetPotentialEvaluation,
    sample_neural_force_hmc,
)
from bayesfilter.inference.surrogate_hmc_dctr_selection import protection_controls
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_m13_controls import m13_scope, partition_fixture
from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import (
    trajectory_importance_reference,
)

PLAN = "docs/plans/surrogate-hmc-m13-endpoint-mechanics-2026-09-24.md"
PARTICLE_COUNT = 4096
ATTEMPT = 4
DESIGN_PARTITION = 9
CONTROL_MULTIPLIER = 0.01
REFERENCE_PARTICLES = 4096
REFERENCE_REPLICATES = 8
REFERENCE_SEED = tf.constant([2026092402, 9101], tf.int32)
PRIOR_SD = tf.constant(math.log(2.0), tf.float64)


def serial(value):
    if tf.is_tensor(value):
        return serial(value.numpy())
    if hasattr(value, "tolist"):
        return serial(value.tolist())
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def prior_log_density(theta):
    theta = tf.cast(theta, tf.float64)
    normalizer = tf.constant(2.0 * math.log(math.log(2.0) * math.sqrt(2.0 * math.pi)), tf.float64)
    return -0.5 * tf.reduce_sum(tf.square(theta / PRIOR_SD), axis=-1) - normalizer


def endpoint_components(fixture):
    """Return a fixed-seed independent endpoint and its compiled evaluator."""
    reference = trajectory_importance_reference(
        fixture, REFERENCE_PARTICLES, REFERENCE_REPLICATES, full_support=True
    )

    @tf.function(
        input_signature=[tf.TensorSpec([2], tf.float64)],
        jit_compile=True,
        autograph=False,
    )
    def independent_log_likelihood(theta):
        result = reference(theta, REFERENCE_SEED)
        estimates = tf.convert_to_tensor(result["log_likelihood"], tf.float64)
        valid = tf.reduce_all(tf.convert_to_tensor(result["valid"], tf.bool))
        log_likelihood = tf.reduce_logsumexp(estimates) - tf.math.log(
            tf.cast(tf.shape(estimates)[0], tf.float64)
        )
        valid = valid & tf.reduce_all(tf.math.is_finite(estimates)) & tf.math.is_finite(log_likelihood)
        return log_likelihood, valid, result

    def endpoint(position):
        position = tf.convert_to_tensor(position, tf.float32)
        theta = tf.cast(position[0], tf.float64)
        log_likelihood, reference_valid, _ = independent_log_likelihood(theta)
        log_posterior = log_likelihood + prior_log_density(theta)
        value64 = -log_posterior
        valid = reference_valid & tf.math.is_finite(value64)
        value = tf.cast(tf.where(valid, value64, tf.constant(float("inf"), tf.float64)), tf.float32)
        valid_batch = valid[tf.newaxis]
        status = {
            "value_valid": valid_batch,
            "force_valid": valid_batch,
            "numerical_failure": tf.logical_not(valid_batch),
            "outside_declared_support": tf.zeros_like(valid_batch),
        }
        return TargetPotentialEvaluation(value, status)

    return endpoint, independent_log_likelihood


def force_components(target):
    evaluate = target.compile(1, filter_moments=False, all_correction_diagnostics=False)

    @tf.function(
        input_signature=[tf.TensorSpec([1, 2], tf.float32)],
        jit_compile=True,
        autograph=False,
    )
    def force(position):
        value, score, status = evaluate(position)
        valid = tf.convert_to_tensor(status["program_valid"], tf.bool)
        # The proposal integrator expects grad(U).  score is grad(log L), so
        # grad(U) = -score + grad(-log prior) = -score + theta/sigma^2.
        potential_gradient = -score + position / tf.cast(tf.square(PRIOR_SD), tf.float32)
        safe_force = tf.where(valid[:, tf.newaxis], potential_gradient, tf.zeros_like(potential_gradient))
        return PositionOnlyForceEvaluation(safe_force, valid)

    return force, evaluate


def _source_hashes(target):
    names = [
        "bayesfilter/testing/surrogate_hmc_m13_endpoint_mechanics.py",
        "bayesfilter/testing/surrogate_hmc_m13_controls.py",
        "bayesfilter/testing/surrogate_hmc_m13_trajectory_reference.py",
        "bayesfilter/inference/neural_force_hmc.py",
        "bayesfilter/inference/surrogate_hmc_dctr_selection.py",
        "docs/plans/surrogate-hmc-m13-endpoint-mechanics-2026-09-24.md",
    ]
    hashes = dict(target.manifest["source_sha256"])
    hashes.update({name: file_hash(ROOT / name) for name in names})
    return hashes


def mechanics_stage(output):
    started = time.monotonic()
    output = Path(output)
    fixture = partition_fixture("calibration", attempt=ATTEMPT)
    controls = protection_controls(CONTROL_MULTIPLIER)
    target = make_range_bearing_target(
        fixture,
        particle_count=PARTICLE_COUNT,
        controls=controls,
        dtype=tf.float32,
        design_seed=20260923,
        design_partition=DESIGN_PARTITION,
    )
    endpoint, endpoint_evaluator = endpoint_components(fixture)
    proposal_force, force_evaluator = force_components(target)

    initial_position = tf.constant([[0.0, 0.0]], tf.float32)
    initial_endpoint = endpoint(initial_position)
    initial_potential = tf.convert_to_tensor(initial_endpoint.potential, tf.float32)
    initial_valid = bool(tf.reduce_all(initial_endpoint.status["value_valid"]))

    # The fixed-seed endpoint must replay bit-for-bit before it is used by HMC.
    endpoint_replay_a = endpoint(initial_position).potential
    endpoint_replay_b = endpoint(initial_position).potential
    endpoint_deterministic = bool(tf.reduce_all(endpoint_replay_a == endpoint_replay_b))
    initial_force = proposal_force(initial_position)
    initial_force_valid = bool(tf.reduce_all(initial_force.force_valid))

    config = NeuralForceHMCConfig(
        step_size=0.002,
        num_leapfrog_steps=2,
        inverse_mass_diagonal=(1.0, 1.0),
        dtype="float32",
    )
    chain = sample_neural_force_hmc(
        initial_position,
        initial_potential,
        FrozenPositionOnlyForce(
            proposal_force,
            identity=(
                "m13_canonical_dctr_gradient_of_negative_log_posterior_"
                "fp32_tf32_xla_n4096_attempt4_unselected_lm_0p01"
            ),
            frozen=True,
            momentum_dependent=False,
            direct_state_update=False,
            symmetric_schedule=True,
            coordinate_system="raw",
        ),
        FrozenTargetPotential(
            endpoint,
            identity=(
                "m13_independent_fixed_seed_fp64_full_support_defensive_importance_"
                "endpoint_plus_exact_gaussian_prior_v1"
            ),
            coordinate_system="raw",
            includes_chart_log_jacobian=False,
            deterministic=True,
        ),
        config,
        num_warmup=8,
        num_results=16,
        seed=tf.constant([2026092402, 9911], tf.int32),
    )

    finite_chain = bool(tf.reduce_all(chain.finite_status))
    no_divergence = not bool(tf.reduce_any(chain.divergence))
    no_force_fallback = not bool(tf.reduce_any(chain.force_fallback))
    endpoint_status_valid = bool(tf.reduce_all(chain.endpoint_status["value_valid"]))
    finite_positions = bool(tf.reduce_all(tf.math.is_finite(chain.positions)))
    finite_potentials = bool(tf.reduce_all(tf.math.is_finite(chain.potentials)))
    finite_log_ratios = bool(tf.reduce_all(tf.math.is_finite(chain.log_acceptance_ratio)))
    hard_pass = all((initial_valid, initial_force_valid, endpoint_deterministic,
                     finite_chain, no_divergence, no_force_fallback,
                     endpoint_status_valid, finite_positions, finite_potentials,
                     finite_log_ratios))

    report = dict(
        status="M13_SURROGATE_HMC_MECHANICS_PASSED" if hard_pass else "M13_SURROGATE_HMC_MECHANICS_FAILED",
        qualified=hard_pass,
        plan=PLAN,
        model="M13",
        goal=("test canonical analytical DCTR-force surrogate HMC with a separately "
              "declared deterministic endpoint potential"),
        inference_launched=True,
        posterior_qualified=False,
        controls_qualified=False,
        endpoint_qualified=False,
        scope=dict(
            m13=m13_scope(bounded=True, particle_count=PARTICLE_COUNT),
            attempt=ATTEMPT,
            calibration_fixture_generation_seed=fixture.fixture_generation_seed,
            observation_checksum=fixture.observation_checksum,
            target_identity=target.identity,
            target_manifest=dict(target.manifest),
            control_multiplier=CONTROL_MULTIPLIER,
            control_status="unselected_candidate_after_no_stable_control_pair",
        ),
        endpoint=dict(
            identity="m13_independent_fixed_seed_fp64_full_support_defensive_importance_endpoint_plus_exact_gaussian_prior_v1",
            estimator="trajectory_importance_reference",
            particles=REFERENCE_PARTICLES,
            replicates=REFERENCE_REPLICATES,
            seed=serial(REFERENCE_SEED),
            dtype="float64",
            full_support=True,
            prior="independent N(0, log(2)^2)",
            initial_potential=serial(initial_potential),
            deterministic_replay=endpoint_deterministic,
            initial_status=serial(initial_endpoint.status),
            nonclaim="finite fixed-seed Monte Carlo endpoint diagnostic; not an R^2 posterior oracle",
        ),
        force=dict(
            identity="m13_canonical_dctr_gradient_of_negative_log_posterior_fp32_tf32_xla_n4096_attempt4_unselected_lm_0p01",
            semantics="position_only_scalar_potential_gradient",
            score_role="canonical analytical DCTR gradient of finite log likelihood",
            composition="-score_DCTR + theta/(log(2)^2)",
            initial_force=serial(initial_force.force),
            initial_force_valid=initial_force_valid,
            dtype="float32",
            tf32=bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            jit_compile=True,
        ),
        hmc=dict(
            step_size=0.002,
            num_leapfrog_steps=2,
            inverse_mass_diagonal=[1.0, 1.0],
            num_warmup=8,
            num_results=16,
            seed=[2026092402, 9911],
            acceptance_rate=float(tf.reduce_mean(tf.exp(tf.minimum(chain.log_acceptance_ratio, tf.zeros_like(chain.log_acceptance_ratio))))),
            binary_acceptance_rate=float(tf.reduce_mean(tf.cast(chain.accepted, tf.float32))),
            finite_status=finite_chain,
            no_divergence=no_divergence,
            no_force_fallback=no_force_fallback,
            endpoint_status_valid=endpoint_status_valid,
            finite_positions=finite_positions,
            finite_potentials=finite_potentials,
            finite_log_acceptance_ratios=finite_log_ratios,
            positions=serial(chain.positions),
            potentials=serial(chain.potentials),
            accepted=serial(chain.accepted),
            log_acceptance_ratio=serial(chain.log_acceptance_ratio),
            delta_h=serial(chain.delta_h),
            divergence=serial(chain.divergence),
            force_fallback=serial(chain.force_fallback),
            endpoint_status=serial(chain.endpoint_status),
        ),
        hard_checks=dict(
            initial_endpoint_valid=initial_valid,
            initial_force_valid=initial_force_valid,
            endpoint_deterministic_replay=endpoint_deterministic,
            all_transition_finite=finite_chain,
            no_divergence=no_divergence,
            no_force_fallback=no_force_fallback,
            all_endpoint_status_valid=endpoint_status_valid,
            all_positions_finite=finite_positions,
            all_potentials_finite=finite_potentials,
            all_log_acceptance_ratios_finite=finite_log_ratios,
        ),
        explanatory_only=["acceptance_rate", "binary_acceptance_rate", "delta_h"],
        nonclaims=[
            "no posterior convergence or R-hat claim",
            "no HMC tuning authority or default-readiness claim",
            "no M13 control selection or posterior-reference qualification",
            "no all-model surrogate-HMC claim",
        ],
        source_hashes=_source_hashes(target),
        wall_seconds=time.monotonic()-started,
    )
    write_json(output, report)
    return report
