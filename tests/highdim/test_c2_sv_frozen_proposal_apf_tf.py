"""Value/score and branch-compiler tests for the C2 frozen proposal APF."""

import inspect
import math

import tensorflow as tf

from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    GaussianHermiteRetainedProposal,
)
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
    C2StochasticVolatilityFrozenAPFModel,
    FrozenGaussianStateProposal,
    compile_c2_bootstrap_proposal_branch,
    compile_c2_dmis_proposal_branch,
    compile_c2_independent_proposal_branch,
    compile_c2_transformed_student_proposal_branch,
    stationary_gaussian_proposals,
    transformed_student_proposals,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    prepare_frozen_proposal_apf_program,
    prepare_frozen_proposal_branch,
)


DTYPE = tf.float64


def _model(dimension: int = 2) -> C2StochasticVolatilityFrozenAPFModel:
    if dimension == 1:
        coupling = tf.zeros([1, 1], DTYPE)
    else:
        coupling = tf.constant([[0.0, 0.04], [-0.02, 0.0]], DTYPE)
    return C2StochasticVolatilityFrozenAPFModel(coupling_matrix=coupling, sigma=1.0)


def _centered_difference(function, theta: tf.Tensor, index: int, step=1e-5):
    direction = tf.one_hot(index, int(theta.shape[0]), dtype=DTYPE)
    return (function(theta + step * direction) - function(theta - step * direction)) / (
        2.0 * step
    )


def test_scalar_stationary_covariance_and_derivative_match_closed_form() -> None:
    model = _model(1)
    theta = tf.constant([0.6, math.log(0.4)], DTYPE)
    covariance, derivative = model.stationary_covariance_and_derivative(theta)
    tf.debugging.assert_near(covariance, tf.constant([[1.0 / 0.64]], DTYPE), atol=2e-14)
    tf.debugging.assert_near(
        derivative, tf.constant([[1.2 / (0.64**2)]], DTYPE), atol=2e-14
    )
    diagnostics = model.stability_diagnostics(theta)
    assert float(diagnostics["spectral_radius"].numpy()) == 0.6
    assert float(diagnostics["lyapunov_residual_max"].numpy()) <= 2e-15
    assert float(diagnostics["derivative_lyapunov_residual_max"].numpy()) <= 2e-15


def test_c2_manual_local_scores_match_centered_finite_differences() -> None:
    model = _model(2)
    theta = tf.constant([0.57, math.log(0.43)], DTYPE)
    previous = tf.constant([[-0.4, 0.8], [0.2, -0.7], [1.1, 0.3]], DTYPE)
    current = tf.constant([[0.1, 1.0], [-0.3, 0.2], [0.7, -0.5]], DTYPE)
    observation = tf.constant([0.35, -0.21], DTYPE)

    initial_score = model.initial_log_density_parameter_score(theta, current)
    initial_fd = _centered_difference(
        lambda value: model.initial_log_density(value, current), theta, 0
    )
    tf.debugging.assert_near(initial_score[:, 0], initial_fd, atol=2e-8, rtol=2e-8)
    tf.debugging.assert_equal(initial_score[:, 1], tf.zeros([3], DTYPE))

    transition_score = model.transition_log_density_parameter_score(
        theta, previous, current, 1
    )
    transition_fd = _centered_difference(
        lambda value: model.transition_log_density(value, previous, current, 1),
        theta,
        0,
    )
    tf.debugging.assert_near(
        transition_score[:, 0], transition_fd, atol=2e-9, rtol=2e-9
    )
    tf.debugging.assert_equal(transition_score[:, 1], tf.zeros([3], DTYPE))

    observation_score = model.observation_log_density_parameter_score(
        theta, current, observation, 1
    )
    observation_fd = _centered_difference(
        lambda value: model.observation_log_density(value, current, observation, 1),
        theta,
        1,
    )
    tf.debugging.assert_near(
        observation_score[:, 1], observation_fd, atol=2e-9, rtol=2e-9
    )
    tf.debugging.assert_equal(observation_score[:, 0], tf.zeros([3], DTYPE))

    source = inspect.getsource(C2StochasticVolatilityFrozenAPFModel)
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source


def _observations() -> tf.Tensor:
    return tf.constant([[0.2, -0.1], [0.35, 0.16], [-0.22, 0.31]], DTYPE)


def _hint_proposals() -> tuple[FrozenGaussianStateProposal, ...]:
    return (
        FrozenGaussianStateProposal(
            mean=tf.constant([0.1, -0.2], DTYPE),
            chol=tf.constant([[1.1, 0.0], [0.12, 0.9]], DTYPE),
            time_index=1,
            family="gaussian_hint_marginal",
        ),
        FrozenGaussianStateProposal(
            mean=tf.constant([-0.05, 0.15], DTYPE),
            chol=tf.constant([[0.95, 0.0], [-0.08, 1.05]], DTYPE),
            time_index=2,
            family="gaussian_hint_marginal",
        ),
    )


def _assert_same_scalar_score(model, compilation, theta) -> None:
    program = prepare_frozen_proposal_apf_program(model, compilation.branch)
    result = program.evaluate(theta)
    finite_difference = tf.stack(
        [
            _centered_difference(
                lambda value: program.evaluate(value)["log_likelihood"],
                theta,
                parameter_index,
            )
            for parameter_index in range(model.parameter_dim())
        ]
    )
    tf.debugging.assert_near(result["score"], finite_difference, atol=2e-7, rtol=2e-7)
    tf.debugging.assert_near(
        result["score"], tf.reduce_sum(result["increment_scores"], axis=0), atol=2e-12
    )
    tf.debugging.assert_near(
        result["minimum_ess"], tf.reduce_min(result["ess_by_time"]), atol=2e-12
    )
    tf.debugging.assert_near(
        result["maximum_log_weight_spread"],
        tf.reduce_max(result["log_weight_spread_by_time"]),
        atol=2e-12,
    )
    assert result["ess_by_time"].shape == (compilation.branch.time_steps,)
    assert result["log_weight_spread_by_time"].shape == (
        compilation.branch.time_steps,
    )
    assert result["maximum_normalized_weight_by_time"].shape == (
        compilation.branch.time_steps,
    )
    assert bool(result["finite"].numpy())
    compiled = program.compiled(jit_compile=False)(theta)
    tf.debugging.assert_near(
        compiled["log_likelihood"], result["log_likelihood"], atol=2e-11
    )
    tf.debugging.assert_near(compiled["score"], result["score"], atol=2e-11)


def _assert_auxiliary_laws_come_from_generic_prefix(model, compilation, theta) -> None:
    branch = compilation.branch
    for time_index in range(1, branch.time_steps):
        prefix = prepare_frozen_proposal_branch(
            observations=branch.observations[:time_index],
            states=branch.states[:time_index],
            initial_log_proposal_density=branch.initial_log_proposal_density,
            ancestors=branch.ancestors[: time_index - 1],
            auxiliary_log_probabilities=branch.auxiliary_log_probabilities[
                : time_index - 1
            ],
            transition_log_proposal_density=branch.transition_log_proposal_density[
                : time_index - 1
            ],
        )
        expected = prepare_frozen_proposal_apf_program(model, prefix).evaluate(theta)[
            "final_log_weights"
        ]
        tf.debugging.assert_near(
            branch.auxiliary_log_probabilities[time_index - 1], expected, atol=2e-12
        )


def test_gaussian_independent_compiler_uses_generic_prefix_and_same_scalar_score() -> None:
    model = _model(2)
    theta = tf.constant([0.58, math.log(0.4)], DTYPE)
    compilation = compile_c2_independent_proposal_branch(
        model=model,
        observations=_observations(),
        theta_reference=theta,
        transition_proposals=_hint_proposals(),
        particle_count=64,
        seed=811,
        family="gaussian_hint_marginal",
        jit_compile_sampler=False,
    )
    _assert_auxiliary_laws_come_from_generic_prefix(model, compilation, theta)
    _assert_same_scalar_score(model, compilation, theta)
    tf.debugging.assert_near(
        compilation.branch.initial_log_proposal_density,
        model.initial_log_density(theta, compilation.branch.states[0]),
        atol=2e-12,
    )
    assert compilation.manifest["auxiliary_law"] == (
        "generic_apf_prefix_weights_at_theta_reference"
    )
    assert compilation.manifest["exact_pseudo_marginal_claimed"] is False
    assert len(compilation.compiler_id) == 64
    proposal = _hint_proposals()[0]
    assert proposal.compiled_transform(64, jit_compile=False) is proposal.compiled_transform(
        64, jit_compile=False
    )


def test_bootstrap_compiler_q_matches_frozen_reference_transition() -> None:
    model = _model(2)
    theta = tf.constant([0.58, math.log(0.4)], DTYPE)
    compilation = compile_c2_bootstrap_proposal_branch(
        model=model,
        observations=_observations(),
        theta_reference=theta,
        particle_count=64,
        seed=812,
        jit_compile_sampler=False,
    )
    branch = compilation.branch
    for time_index in range(1, branch.time_steps):
        parents = tf.gather(branch.states[time_index - 1], branch.ancestors[time_index - 1])
        expected = model.transition_log_density(
            theta, parents, branch.states[time_index], time_index
        )
        tf.debugging.assert_near(
            branch.transition_log_proposal_density[time_index - 1], expected, atol=2e-12
        )
    _assert_auxiliary_laws_come_from_generic_prefix(model, compilation, theta)
    _assert_same_scalar_score(model, compilation, theta)


def _rank_one_tt_proposal(time_index: int) -> GaussianHermiteRetainedProposal:
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


def test_tt_proposal_compiler_wires_complete_density_into_generic_program() -> None:
    model = _model(2)
    theta = tf.constant([0.58, math.log(0.4)], DTYPE)
    proposals = (_rank_one_tt_proposal(1), _rank_one_tt_proposal(2))
    compilation = compile_c2_independent_proposal_branch(
        model=model,
        observations=_observations(),
        theta_reference=theta,
        transition_proposals=proposals,
        particle_count=64,
        seed=813,
        family="retained_tt",
        jit_compile_sampler=False,
    )
    for row, proposal in enumerate(proposals):
        expected = proposal.physical_log_density(compilation.branch.states[row + 1])
        tf.debugging.assert_near(
            compilation.branch.transition_log_proposal_density[row], expected, atol=2e-11
        )
        assert bool(compilation.proposal_diagnostics[row]["cdf_bracket_valid"].numpy())
    _assert_same_scalar_score(model, compilation, theta)


def test_stationary_independence_proposal_factory_has_bound_time_indices() -> None:
    model = _model(2)
    theta = tf.constant([0.58, math.log(0.4)], DTYPE)
    proposals = stationary_gaussian_proposals(model, theta, horizon=4)
    assert [proposal.time_index for proposal in proposals] == [1, 2, 3]
    assert all(proposal.family == "stationary_independence" for proposal in proposals)


def test_dmis_equal_banks_use_complete_mixture_density_and_generalized_base_mass() -> None:
    model = _model(2)
    theta = tf.constant([0.58, math.log(0.4)], DTYPE)
    observations = _observations()
    tt_proposals = (_rank_one_tt_proposal(1), _rank_one_tt_proposal(2))
    defensive = transformed_student_proposals(
        model=model, observations=observations, theta_reference=theta, nu=8.0
    )
    compilation = compile_c2_dmis_proposal_branch(
        model=model,
        observations=observations,
        theta_reference=theta,
        transition_proposals=tt_proposals,
        defensive_proposals=defensive,
        particle_count=64,
        seed=814,
        alpha=0.5,
        nu=8.0,
        jit_compile_sampler=False,
    )
    branch = compilation.branch
    bank_count = 32
    expected_base = math.log(0.5 / bank_count)
    tf.debugging.assert_near(
        branch.transition_log_base_mass,
        tf.fill([2, 64], tf.constant(expected_base, DTYPE)),
        atol=2e-14,
    )
    tf.debugging.assert_near(
        tf.reduce_logsumexp(branch.transition_log_base_mass, axis=1),
        tf.zeros([2], DTYPE),
        atol=2e-14,
    )
    for time_index in range(1, branch.time_steps):
        ancestor = branch.ancestors[time_index - 1]
        parents = tf.gather(branch.states[time_index - 1], ancestor)
        tt = tt_proposals[time_index - 1]
        d = defensive[time_index - 1]
        current = branch.states[time_index]
        expected_tt = tf.reduce_logsumexp(
            tf.stack(
                [
                    tf.math.log(tf.constant(0.5, DTYPE))
                    + tt.physical_log_density(current),
                    tf.math.log(tf.constant(0.5, DTYPE))
                    + d.log_density(current, parents),
                ],
                axis=1,
            ),
            axis=1,
        )
        tf.debugging.assert_near(
            branch.transition_log_proposal_density[time_index - 1], expected_tt, atol=2e-11
        )
    _assert_auxiliary_laws_come_from_generic_prefix(model, compilation, theta)
    _assert_same_scalar_score(model, compilation, theta)
    assert compilation.manifest["complete_mixture_density"] is True


def test_transformed_student_single_proposal_uses_parent_conditionals() -> None:
    model = _model(2)
    theta = tf.constant([0.58, math.log(0.4)], DTYPE)
    compilation = compile_c2_transformed_student_proposal_branch(
        model=model,
        observations=_observations(),
        theta_reference=theta,
        nu=8.0,
        particle_count=32,
        seed=815,
        jit_compile_sampler=False,
    )
    _assert_auxiliary_laws_come_from_generic_prefix(model, compilation, theta)
    _assert_same_scalar_score(model, compilation, theta)
    assert all(
        row["family"] == "transformed_observation_student"
        for row in compilation.proposal_diagnostics
    )
