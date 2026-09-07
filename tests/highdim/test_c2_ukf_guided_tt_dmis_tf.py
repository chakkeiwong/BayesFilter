"""Integration checks for the shared C2 TT/defensive DMIS branch."""

import math

import tensorflow as tf

from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    GaussianHermiteRetainedProposal,
)
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
    C2StochasticVolatilityFrozenAPFModel,
    compile_c2_dmis_proposal_branch,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    prepare_frozen_proposal_apf_program,
)


DTYPE = tf.float64


def _tt(time_index: int) -> GaussianHermiteRetainedProposal:
    first = tf.constant([1.0, 0.12], DTYPE)
    second = tf.constant([1.0, -0.08], DTYPE)
    z_h = tf.reduce_sum(tf.square(first)) * tf.reduce_sum(tf.square(second))
    return GaussianHermiteRetainedProposal(
        prefix_core_values=(
            tf.reshape(first, [1, 2, 1]),
            tf.reshape(second, [1, 2, 1]),
        ),
        suffix_gram=tf.ones([1, 1], DTYPE),
        z_h=z_h,
        tau_abs=tf.constant(0.02, DTYPE),
        coordinate_offset=tf.constant([0.1 * time_index, -0.05 * time_index], DTYPE),
        coordinate_matrix=tf.constant([[1.0, 0.0], [0.1, 0.9]], DTYPE),
        defensive_nu=5.0,
        time_index=time_index,
        source_snapshot_fingerprint=f"{time_index:064x}",
    )


def test_dmis_call_chain_is_finite_and_analytical_score_matches_fd() -> None:
    model = C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=tf.constant([[0.0, 0.04], [-0.02, 0.0]], DTYPE), sigma=1.0
    )
    theta = tf.constant([0.58, math.log(0.4)], DTYPE)
    observations = tf.constant(
        [[0.2, -0.1], [0.35, 0.16], [-0.22, 0.31]], DTYPE
    )
    compilation = compile_c2_dmis_proposal_branch(
        model=model,
        observations=observations,
        theta_reference=theta,
        transition_proposals=(_tt(1), _tt(2)),
        particle_count=32,
        seed=901,
        alpha=0.5,
        nu=8.0,
        jit_compile_sampler=False,
    )
    program = prepare_frozen_proposal_apf_program(model, compilation.branch)
    result = program.evaluate(theta)
    assert bool(result["finite"].numpy())
    tf.debugging.assert_near(
        tf.reduce_logsumexp(compilation.branch.transition_log_base_mass, axis=1),
        tf.zeros([2], DTYPE),
        atol=2e-12,
    )
    finite_difference = []
    for index in range(2):
        direction = tf.one_hot(index, 2, dtype=DTYPE)
        step = tf.constant(1e-5, DTYPE)
        finite_difference.append(
            (
                program.evaluate(theta + step * direction)["log_likelihood"]
                - program.evaluate(theta - step * direction)["log_likelihood"]
            )
            / (2.0 * step)
        )
    tf.debugging.assert_near(result["score"], tf.stack(finite_difference), atol=2e-7, rtol=2e-7)
    assert compilation.manifest["complete_mixture_density"] is True
    assert compilation.manifest["base_mass_policy"] == "component_weight_over_bank_count"
