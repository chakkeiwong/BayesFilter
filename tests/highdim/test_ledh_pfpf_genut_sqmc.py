from __future__ import annotations

import inspect

import tensorflow as tf

from bayesfilter.highdim import ledh_pfpf_genut_initialization_tf as core
from bayesfilter.highdim.cubature_genut_candidate import (
    gaussian_genut_design,
    replicate_positive_genut,
)
from bayesfilter.highdim.gaussian_cloud_designs_tf import standard_normal_cloud
from bayesfilter.highdim.sqmc_tf import randomized_halton_joint


def _records(
    particles: tf.Tensor,
    marks: tf.Tensor,
    *,
    policy: str,
    uniforms: tf.Tensor | None = None,
) -> dict[str, tf.Tensor]:
    count = int(particles.shape[0])
    return core._transition_records(  # noqa: SLF001
        particles,
        tf.fill([count], tf.cast(1.0 / count, particles.dtype)),
        marks,
        tf.zeros_like(particles),
        tf.linspace(0.01, 0.99, count) if uniforms is None else uniforms,
        ancestry_policy=policy,
        state_map_location=tf.zeros([particles.shape[1]], particles.dtype),
        state_map_scale=tf.ones([particles.shape[1]], particles.dtype),
        hilbert_bits=12,
    )


def test_existing_records_preserve_rows_and_attached_marks() -> None:
    particles = tf.reshape(tf.range(18, dtype=tf.float32), [6, 3])
    marks = tf.reshape(tf.range(30, dtype=tf.float32), [6, 5])
    records = _records(particles, marks, policy="existing_one_to_one")
    tf.debugging.assert_equal(records["backward_parents"], particles)
    tf.debugging.assert_equal(records["backward_marks"], marks)
    tf.debugging.assert_equal(records["flow_ancestors"], particles)
    tf.debugging.assert_equal(records["selected_row_identities"], tf.range(6))


def test_hilbert_sort_moves_particles_and_marks_as_one_record() -> None:
    particles = tf.constant(
        ((2.0, -1.0), (-0.4, 0.2), (0.7, 1.3), (-1.2, -0.8)), tf.float32
    )
    marks = tf.cast(tf.range(4)[:, None], tf.float32)
    records = _records(particles, marks, policy="hilbert_one_to_one")
    identities = records["selected_row_identities"]
    tf.debugging.assert_equal(records["backward_parents"], tf.gather(particles, identities))
    tf.debugging.assert_equal(records["backward_marks"], tf.gather(marks, identities))


def test_full_sqmc_is_equivariant_to_input_record_permutation() -> None:
    particles = tf.constant(
        ((-1.1, 0.3), (0.4, 1.2), (1.6, -0.7), (-0.2, -1.4)), tf.float32
    )
    marks = tf.reshape(tf.range(12, dtype=tf.float32), [4, 3])
    uniforms = tf.constant((0.05, 0.30, 0.55, 0.90), tf.float32)
    first = _records(
        particles, marks, policy="hilbert_inverse_cdf", uniforms=uniforms
    )
    permutation = tf.constant((2, 0, 3, 1), tf.int32)
    second = _records(
        tf.gather(particles, permutation),
        tf.gather(marks, permutation),
        policy="hilbert_inverse_cdf",
        uniforms=uniforms,
    )
    tf.debugging.assert_equal(first["backward_parents"], second["backward_parents"])
    tf.debugging.assert_equal(first["backward_marks"], second["backward_marks"])
    tf.debugging.assert_equal(first["flow_ancestors"], second["flow_ancestors"])


def test_core_source_retains_standard_score_and_has_no_proposal_correction() -> None:
    source = inspect.getsource(core)
    assert "_target_model_progressive_score_marks" in source
    assert "_sir_progressive_marks" in source
    assert 'records["backward_parents"]' in source
    assert 'records["flow_ancestors"]' in source
    for forbidden in (
        "GradientTape",
        "ForwardAccumulator",
        "finite_difference",
        "ancestor_proposal_probability",
        "importance_ratio_w_over_r",
    ):
        assert forbidden not in source


def test_full_sqmc_core_compiles_on_cpu_xla_and_reports_ancestry() -> None:
    spec = core.diagonal_lgssm_spec()
    count = 48
    theta = tf.constant((0.72, 0.55, 0.35, 0.35, 0.45), tf.float32)
    observations = tf.zeros([1, 3], tf.float32)
    initial = standard_normal_cloud(
        "scrambled_halton_gaussian",
        num_particles=count,
        dimension=3,
        seed=101,
    )
    _, ancestor_uniforms, uniforms = randomized_halton_joint(
        num_particles=count,
        state_dimension=3,
        seed=101,
        salt=2,
    )
    process = tf.math.ndtri(uniforms)[None, :, :]
    design = replicate_positive_genut(
        gaussian_genut_design(dim=3), num_particles=count
    )

    @tf.function(jit_compile=True, autograph=False)
    def evaluate(theta_value, observations_value, initial_value, process_value, ancestor_value):
        return core.finite_value_standard_score_ledh_pfpf_genut(
            spec,
            theta_value,
            observations_value,
            initial_value,
            process_value,
            design,
            ancestry_policy="hilbert_inverse_cdf",
            process_ancestor_uniforms=ancestor_value,
            state_map_location=tf.zeros([3], tf.float32),
            state_map_scale=tf.ones([3], tf.float32),
            sinkhorn_steps=4,
            balance_steps=4,
        )

    repeated_ancestor_uniforms = tf.fill([1, count], tf.constant(0.01, tf.float32))
    value, score, diagnostics = evaluate(
        theta, observations, initial, process, repeated_ancestor_uniforms
    )
    assert bool(diagnostics["program_valid"].numpy())
    tf.debugging.assert_all_finite(value, "value must be finite")
    tf.debugging.assert_all_finite(score, "score must be finite")
    assert int(diagnostics["ancestry_unique_count"][0].numpy()) == 1
    assert evaluate.experimental_get_tracing_count() == 1
