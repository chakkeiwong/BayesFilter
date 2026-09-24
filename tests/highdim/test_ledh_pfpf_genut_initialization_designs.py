from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim import ledh_pfpf_genut_initialization_tf as core
from bayesfilter.highdim import sir_online_score_teacher_tf as sir_score
from bayesfilter.highdim.cubature_genut_candidate import (
    gaussian_genut_design,
    replicate_positive_genut,
)
from bayesfilter.highdim.gaussian_cloud_designs_tf import (
    SUPPORTED_DESIGNS,
    standard_normal_cloud,
)
from bayesfilter.highdim.sir_latent_preclip_reference_tf import (
    reduced_latent_preclip_sir_model,
)
from docs.benchmarks import (
    run_ledh_pfpf_genut_initialization_design_comparison as runner,
)


def _cloud(design: str, *, dimension: int = 3, seed: int = 701) -> tf.Tensor:
    return standard_normal_cloud(
        design,
        num_particles=48,
        dimension=dimension,
        seed=seed,
        salt=19,
    )


@pytest.mark.parametrize("design", SUPPORTED_DESIGNS)
def test_clouds_are_finite_shaped_replayable_and_randomized(design: str) -> None:
    first = _cloud(design)
    replay = _cloud(design)
    other = _cloud(design, seed=702)
    assert first.shape == (48, 3)
    tf.debugging.assert_all_finite(first, "cloud must be finite")
    tf.debugging.assert_equal(first, replay)
    assert not bool(tf.reduce_all(first == other).numpy())


@pytest.mark.parametrize(
    "design", ("antithetic_gaussian", "antithetic_scrambled_halton_gaussian")
)
def test_antithetic_clouds_have_exact_paired_sign_symmetry(design: str) -> None:
    cloud = _cloud(design)
    tf.debugging.assert_equal(cloud[:24], -cloud[24:])
    tf.debugging.assert_near(tf.reduce_mean(cloud, axis=0), tf.zeros([3]), atol=6e-8)


def test_latin_hypercube_has_exactly_one_point_per_marginal_stratum() -> None:
    cloud = _cloud("latin_hypercube_gaussian")
    uniforms = tfp.distributions.Normal(0.0, 1.0).cdf(cloud)
    strata = tf.cast(tf.floor(48.0 * uniforms), tf.int32)
    expected = tf.range(48, dtype=tf.int32)
    for axis in range(3):
        tf.debugging.assert_equal(tf.sort(strata[:, axis]), expected)


def test_orthogonal_cloud_has_orthogonal_directions_within_each_block() -> None:
    dimension = 3
    blocks = tf.reshape(_cloud("orthogonal_gaussian_blocks"), [-1, dimension, dimension])
    directions = tf.math.l2_normalize(blocks, axis=2)
    gram = tf.linalg.matmul(directions, directions, transpose_b=True)
    tf.debugging.assert_near(
        gram,
        tf.broadcast_to(tf.eye(dimension), tf.shape(gram)),
        atol=3e-6,
        rtol=3e-6,
    )


def test_invalid_cloud_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="even"):
        standard_normal_cloud(
            "antithetic_gaussian", num_particles=47, dimension=3, seed=1
        )
    with pytest.raises(ValueError, match="divisible"):
        standard_normal_cloud(
            "orthogonal_gaussian_blocks",
            num_particles=48,
            dimension=5,
            seed=1,
        )
    with pytest.raises(ValueError, match="unsupported"):
        standard_normal_cloud("sobol", num_particles=48, dimension=3, seed=1)


@pytest.mark.parametrize("dimension", (2, 3))
def test_positive_gaussian_genut_is_exactly_replicable_at_n48(dimension: int) -> None:
    cloud = replicate_positive_genut(
        gaussian_genut_design(dim=dimension), num_particles=48
    )
    assert cloud.shape == (48, dimension)
    tf.debugging.assert_near(tf.reduce_mean(cloud, axis=0), tf.zeros([dimension]))
    covariance = tf.einsum("ni,nj->ij", cloud, cloud) / 48.0
    tf.debugging.assert_near(covariance, tf.eye(dimension), atol=2e-7)


def test_experiment_uses_existing_standard_scores_without_autodiff_or_new_sir_score() -> None:
    core_source = inspect.getsource(core)
    runner_source = inspect.getsource(runner)
    for disallowed in ("GradientTape", "ForwardAccumulator", "finite_difference"):
        assert disallowed not in core_source
        assert disallowed not in runner_source
    assert "sir_score.initial_log_density_and_score" in core_source
    assert "sir_score.transition_log_density_and_score" in core_source
    assert "sir_score.observation_log_density_and_score" in core_source
    assert "_initial_target_model_score_marks" in core_source
    assert "_target_model_progressive_score_marks" in core_source


def test_reduced_sir_t1_score_is_existing_observation_score_posterior_mean() -> None:
    model = reduced_latent_preclip_sir_model()
    static_spec = sir_score.static_spec_from_model(model)
    theta = tf.zeros([3], tf.float32)
    observation = tf.constant([[0.35]], tf.float32)
    initial = _cloud("iid_gaussian", dimension=2)
    design = replicate_positive_genut(
        gaussian_genut_design(dim=2), num_particles=48
    )
    value, score, diagnostics = core.finite_value_standard_score_ledh_pfpf_genut(
        core.reduced_sir_spec(),
        theta,
        observation,
        initial,
        tf.zeros([0, 48, 2], tf.float32),
        design,
        sir_static_spec=static_spec,
        sinkhorn_steps=4,
        balance_steps=4,
    )
    assert bool(diagnostics["program_valid"].numpy())
    assert bool(tf.math.is_finite(value).numpy())
    # The first two SIR parameters enter transitions only, which do not exist at T=1.
    tf.debugging.assert_near(score[:2], tf.zeros([2]), atol=1e-7)
    assert bool(tf.math.is_finite(score[2]).numpy())


def test_lgssm_core_replays_exactly_and_compiles_on_cpu_xla() -> None:
    spec = core.diagonal_lgssm_spec()
    theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32)
    observations = tf.zeros([1, 3], tf.float32)
    initial = _cloud("iid_gaussian")
    process = _cloud("scrambled_halton_gaussian")[None, :, :]
    design = replicate_positive_genut(
        gaussian_genut_design(dim=3), num_particles=48
    )

    @tf.function(jit_compile=True, autograph=False)
    def evaluate(theta_value, observations_value, initial_value, process_value, design_value):
        with tf.device("/CPU:0"):
            return core.finite_value_standard_score_ledh_pfpf_genut(
                spec,
                theta_value,
                observations_value,
                initial_value,
                process_value,
                design_value,
                sinkhorn_steps=4,
                balance_steps=4,
            )

    first = evaluate(theta, observations, initial, process, design)
    replay = evaluate(theta, observations, initial, process, design)
    assert bool(first[2]["program_valid"].numpy())
    assert "CPU:0" in first[0].device
    tf.debugging.assert_equal(first[0], replay[0])
    tf.debugging.assert_equal(first[1], replay[1])
    assert evaluate.experimental_get_tracing_count() == 1


def test_runner_campaign_contract_is_complete_and_conservative() -> None:
    assert runner.PARTICLE_COUNT == 48
    assert runner.HORIZON == 6
    assert len(runner.RANDOMIZATION_SEEDS) == 8
    assert runner.SCOPES == ("initial_only", "all_innovations")
    assert len(runner.SUPPORTED_DESIGNS) == 6
    assert runner.BASELINE == "iid_gaussian"
    assert runner.BOOTSTRAP_REPLICATES >= 1000
    markdown_source = inspect.getsource(runner._markdown)  # noqa: SLF001
    assert "not a statistically supported universal ranking" in markdown_source
    assert "Default readiness" in markdown_source


def test_runner_model_generators_have_expected_shapes_and_finite_values() -> None:
    lgssm = runner._lgssm_observations()  # noqa: SLF001
    _model, static_spec, sir = runner._sir_context()  # noqa: SLF001
    assert lgssm.shape == (runner.HORIZON, 3)
    assert sir.shape == (runner.HORIZON, 1)
    assert static_spec.state_dimension == 2
    tf.debugging.assert_all_finite(lgssm, "LGSSM observations must be finite")
    tf.debugging.assert_all_finite(sir, "SIR observations must be finite")
