"""Registered LEDH value endpoint using the shared TensorFlow/XLA recurrence.

Execution repair only: the name does not establish canonical-method admission.
The shared native owner preserves the existing numerical algorithm and rejects
unusable resets. No NumPy or Python numerical recurrence executes here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim.ledh_alg1_contract import COVARIANCE_PROVENANCE_KINDS
from bayesfilter.highdim.ledh_canonical_value_program_tf import (
    make_canonical_value_program,
)
from bayesfilter.ops.ledh_random_compat_tf import (
    integer_seed_words,
    philox_replication_state,
)

Tensor = tf.Tensor


@dataclass(frozen=True)
class CanonicalModelCallbacks:
    """Fixed model configuration; time callbacks accept a scalar int32 Tensor.

    Python closure configuration must remain fixed while a reusable owner lives.
    Recreate that owner after changing configuration or use the one-shot API.
    """

    model_id: str
    state_dim: int
    observation_dim: int
    transition_mean_fn: Callable[[Tensor, Tensor], Tensor]
    transition_log_density_fn: Callable[[Tensor, Tensor, Tensor], Tensor]
    process_noise_covariance: Tensor
    process_noise_covariance_provenance: str
    observation_fn: Callable[[Tensor, Tensor], Tensor]
    observation_jacobian_fn: Callable[[Tensor, Tensor], Tensor]
    observation_covariance: Tensor
    observation_log_density_fn: Callable[[Tensor, Tensor, Tensor], Tensor]
    initial_mean: Tensor
    initial_covariance: Tensor
    initial_covariance_provenance: str


def _replication_generator(seed: int) -> tf.random.Generator:
    """Diagnostic/reference stateful generator; not used by the value endpoint.

    Preserve the prior SeedSequence-mixed Philox stream for tests that compare
    native normal transforms with TensorFlow's independent stateful kernel.
    """
    entropy = tf.constant(integer_seed_words(seed), tf.uint32)
    state = tf.bitcast(philox_replication_state(entropy), tf.int64)
    return tf.random.Generator.from_state(state, alg="philox")


def _require_provenance(kind: str, field: str) -> None:
    if kind not in COVARIANCE_PROVENANCE_KINDS:
        raise ValueError(
            f"{field}: covariance provenance {kind!r} is not an accepted "
            f"kind {COVARIANCE_PROVENANCE_KINDS}; identity/constant "
            "placeholders require a reviewed exception (contract C-10)"
        )


def make_seeded_canonical_value_program(
    callbacks: CanonicalModelCallbacks,
    observation_spec: tf.TensorSpec,
    *,
    particle_count: int,
    seed_word_count: int = 1,
    resample_seed_word_count: int = 1,
    flow_substeps: int = 24,
    temper_stages: int = 1,
    annealed_resampling: bool = False,
    flow_prior_cap: float = float("inf"),
    epsilon: float = 2.0,
    sinkhorn_steps: int = 8,
    balance_steps: int = 8,
    ridge: float = 1.0e-5,
    dual_cap_enabled: bool = False,
    trust_region_enabled: bool = False,
    trust_region_lm_damping: float = 1.0e-2,
    trust_region_lm_scale_floor: float = 1.0e-4,
    trust_region_radius: float = 0.5,
    jit_compile: bool = True,
):
    """Build one reusable XLA owner for the existing seeded value program.

    Observations and minimally encoded seed words are dynamic operands. Shapes,
    controls and Python callback closures are fixed for this owner's lifetime;
    no global cache is used. Recreate after a configuration/closure change.
    The three input signatures are [T, observation_dim], [seed_word_count] and
    [resample_seed_word_count]. Use ``integer_seed_words`` for both seeds.
    The shared recurrence generates one process draw per valid prediction;
    no full-horizon random array is materialized. Numeric RNG-state and consumed
    draw diagnostics preserve observability of early termination.
    Return fixed-capacity numeric diagnostics; ``model_id`` and trimmed marginal
    histories are reporting-boundary operations in the convenience wrapper.
    Explicit ``jit_compile=False`` is a diagnostic/reference exception only.
    This execution repair does not confer canonical or scientific admission.
    """
    _require_provenance(callbacks.initial_covariance_provenance, "initial_covariance")
    _require_provenance(callbacks.process_noise_covariance_provenance,
                        "process_noise_covariance")
    if seed_word_count < 1 or resample_seed_word_count < 1:
        raise ValueError("seed word counts must be positive")
    return make_canonical_value_program(
        callbacks, observation_spec, particle_count=particle_count,
        flow_substeps=flow_substeps, temper_stages=temper_stages,
        annealed_resampling=annealed_resampling, flow_prior_cap=flow_prior_cap,
        epsilon=epsilon, sinkhorn_steps=sinkhorn_steps, balance_steps=balance_steps,
        ridge=ridge, dual_cap_enabled=dual_cap_enabled,
        trust_region_enabled=trust_region_enabled,
        trust_region_lm_damping=trust_region_lm_damping,
        trust_region_lm_scale_floor=trust_region_lm_scale_floor,
        trust_region_radius=trust_region_radius, jit_compile=jit_compile,
        seed_word_count=seed_word_count, resample_seed_word_count=resample_seed_word_count,
    )


def canonical_value_and_diagnostics(
    callbacks: CanonicalModelCallbacks,
    observations: Tensor,
    *,
    particle_count: int,
    seed: int,
    flow_substeps: int = 24,
    temper_stages: int = 1,
    annealed_resampling: bool = False,
    flow_prior_cap: float = float("inf"),
    resample_seed: int = 1,
    epsilon: float = 2.0,
    sinkhorn_steps: int = 8,
    balance_steps: int = 8,
    ridge: float = 1.0e-5,
    dual_cap_enabled: bool = False,
    trust_region_enabled: bool = False,
    trust_region_lm_damping: float = 1.0e-2,
    trust_region_lm_scale_floor: float = 1.0e-4,
    trust_region_radius: float = 0.5,
    jit_compile: bool = True,
) -> dict[str, Tensor]:
    """Run the seeded filter through a fresh XLA owner and format diagnostics.

    Each call reads current Python callback configuration. For repeated calls
    with fixed configuration, retain ``make_seeded_canonical_value_program``
    explicitly to amortize compilation. Time callbacks must accept tensor time.
    A failed/incomplete reset returns NaN value, false program_valid and an
    explicit failure code/index. raw_value is diagnostic only, never a fallback
    likelihood. This does not establish canonical-method or scientific status.
    """
    dtype = tf.convert_to_tensor(callbacks.initial_mean).dtype
    observations = tf.convert_to_tensor(observations, dtype)
    seed_words = integer_seed_words(seed)
    resample_words = integer_seed_words(resample_seed) if annealed_resampling else (0,)
    program = make_seeded_canonical_value_program(
        callbacks, tf.TensorSpec(observations.shape, dtype),
        particle_count=particle_count, seed_word_count=len(seed_words),
        resample_seed_word_count=len(resample_words),
        flow_substeps=flow_substeps, temper_stages=temper_stages,
        annealed_resampling=annealed_resampling, flow_prior_cap=flow_prior_cap,
        epsilon=epsilon, sinkhorn_steps=sinkhorn_steps, balance_steps=balance_steps,
        ridge=ridge, dual_cap_enabled=dual_cap_enabled,
        trust_region_enabled=trust_region_enabled,
        trust_region_lm_damping=trust_region_lm_damping,
        trust_region_lm_scale_floor=trust_region_lm_scale_floor,
        trust_region_radius=trust_region_radius, jit_compile=jit_compile,
    )
    result = program(observations, tf.constant(seed_words, tf.uint32),
                     tf.constant(resample_words, tf.uint32))
    completed = result["marginal_steps_completed"]
    return {
        **result,
        "model_id": tf.constant(callbacks.model_id),
        "per_step_marginal_tv_error": result["per_step_marginal_tv_error"][:completed],
        "per_step_marginal_valid": result["per_step_marginal_valid"][:completed],
    }


__all__ = [
    "CanonicalModelCallbacks",
    "canonical_value_and_diagnostics",
    "make_seeded_canonical_value_program",
]
