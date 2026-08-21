from __future__ import annotations

import ast
import inspect
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_candidate import (
    cubature_design,
    gaussian_genut_design,
    replicate_positive_genut,
)
from bayesfilter.highdim.genut_guided_proposal_tf import _restore_cloud_primal
from bayesfilter.highdim.genut_guided_proposal_tf import (
    _dense_sinkhorn_barycentric_value,
    _streaming_sinkhorn_barycentric_value,
)
from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
    finite_value_standard_score_initial_rqmc,
    standard_pairwise_backward_marks,
)
from bayesfilter.highdim.ledh_pfpf_genut_model_callbacks_tf import (
    austria_sir_callbacks,
)
from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)


ROOT = Path(__file__).resolve().parents[2]


def test_sqmc_harness_defaults_to_trust_region() -> None:
    path = ROOT / "docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"RESET_VARIANTS", "DEFAULT_RESET_VARIANTS"}
    }
    assert assignments["RESET_VARIANTS"] == ("legacy", "trust_region")
    assert assignments["DEFAULT_RESET_VARIANTS"] == ("trust_region",)


def test_streaming_sinkhorn_has_no_dense_pairwise_state() -> None:
    source = inspect.getsource(_streaming_sinkhorn_barycentric_value)
    assert "particles[:, None" not in source
    assert "particles[None, :" not in source
    assert "coupling = left[:, None]" not in source
    assert "quotient_coupling" not in source


def test_streaming_sinkhorn_multitile_matches_dense_fp64() -> None:
    count = 12
    particles = tf.random.stateless_normal(
        [count, 3], [2026, 819], dtype=tf.float64
    )
    weights = tf.nn.softmax(
        tf.random.stateless_normal([count], [2026, 820], dtype=tf.float64)
    )
    dense = _dense_sinkhorn_barycentric_value(
        particles,
        weights,
        epsilon=8.0,
        sinkhorn_steps=8,
        balance_steps=8,
    )
    streamed = _streaming_sinkhorn_barycentric_value(
        particles,
        weights,
        epsilon=8.0,
        sinkhorn_steps=8,
        balance_steps=8,
        row_chunk_size=4,
        col_chunk_size=4,
    )
    for name in (
        "barycentric",
        "row_mass",
        "raw_column_mass",
        "quotient_column_residual",
        "cost_scale",
    ):
        tolerance = 2.0e-10 if name != "quotient_column_residual" else 2.0e-8
        tf.debugging.assert_near(
            streamed[name], dense[name], atol=tolerance, rtol=tolerance
        )


def test_streaming_one_block_uses_dense_arithmetic_baseline() -> None:
    count = 12
    particles = tf.random.stateless_normal([count, 3], [2026, 829])
    weights = tf.nn.softmax(tf.random.stateless_normal([count], [2026, 830]))
    dense = _dense_sinkhorn_barycentric_value(
        particles,
        weights,
        epsilon=8.0,
        sinkhorn_steps=8,
        balance_steps=8,
    )
    streamed = _streaming_sinkhorn_barycentric_value(
        particles,
        weights,
        epsilon=8.0,
        sinkhorn_steps=8,
        balance_steps=8,
        row_chunk_size=count,
        col_chunk_size=count,
    )
    for name in dense:
        tf.debugging.assert_equal(streamed[name], dense[name])


def test_streaming_sinkhorn_multitile_compiles_with_cpu_xla() -> None:
    count = 12
    particles = tf.random.stateless_normal([count, 3], [2026, 821])
    weights = tf.nn.softmax(tf.random.stateless_normal([count], [2026, 822]))

    @tf.function(jit_compile=True, autograph=False)
    def compiled(points: tf.Tensor, mass: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        result = _streaming_sinkhorn_barycentric_value(
            points,
            mass,
            epsilon=8.0,
            sinkhorn_steps=2,
            balance_steps=2,
            row_chunk_size=4,
            col_chunk_size=4,
        )
        return result["barycentric"], result["quotient_column_residual"]

    barycentric, residual = compiled(particles, weights)
    assert bool(tf.reduce_all(tf.math.is_finite(barycentric)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(residual)).numpy())


def test_streaming_sinkhorn_multitile_matches_dense_fp32_cpu_xla() -> None:
    count = 12
    particles = tf.random.stateless_normal([count, 3], [2026, 827])
    weights = tf.nn.softmax(tf.random.stateless_normal([count], [2026, 828]))

    @tf.function(jit_compile=True, autograph=False)
    def compiled(
        points: tf.Tensor, mass: tf.Tensor
    ) -> tuple[dict[str, tf.Tensor], dict[str, tf.Tensor]]:
        dense = _dense_sinkhorn_barycentric_value(
            points,
            mass,
            epsilon=8.0,
            sinkhorn_steps=8,
            balance_steps=8,
        )
        streamed = _streaming_sinkhorn_barycentric_value(
            points,
            mass,
            epsilon=8.0,
            sinkhorn_steps=8,
            balance_steps=8,
            row_chunk_size=4,
            col_chunk_size=4,
        )
        return dense, streamed

    dense, streamed = compiled(particles, weights)
    for name in (
        "barycentric",
        "row_mass",
        "raw_column_mass",
        "quotient_column_residual",
        "cost_scale",
    ):
        tf.debugging.assert_near(
            streamed[name], dense[name], atol=2.0e-5, rtol=2.0e-5
        )


def test_austria_score_child_blocking_matches_dense() -> None:
    callbacks = austria_sir_callbacks(latent_preclip_zhao_cui_sir_austria_model())
    count = 36
    theta = tf.zeros([3], tf.float32)
    parents = tf.reshape(tf.linspace(1.0, 36.0 * 18.0, count * 18), [count, 18])
    children = parents + tf.reshape(tf.linspace(-0.5, 0.5, count * 18), [count, 18])
    parent_log_weights = tf.fill([count], -tf.math.log(tf.cast(count, tf.float32)))
    parent_marks = tf.reshape(tf.linspace(-1.0, 1.0, count * 3), [count, 3])
    observation = tf.fill([9], tf.constant(50.0, tf.float32))

    dense = standard_pairwise_backward_marks(
        callbacks,
        theta,
        parents,
        parent_log_weights,
        parent_marks,
        children,
        observation,
        observation_index=1,
    )
    blocked = standard_pairwise_backward_marks(
        callbacks,
        theta,
        parents,
        parent_log_weights,
        parent_marks,
        children,
        observation,
        observation_index=1,
        child_block_size=6,
    )
    tf.debugging.assert_near(blocked, dense, atol=2.0e-5, rtol=2.0e-5)


def test_sqmc_trust_region_reset_route_is_finite() -> None:
    count = 36
    particles = tf.random.stateless_normal([count, 2], [2026, 817])
    weights = tf.nn.softmax(
        0.2 * tf.random.stateless_normal([count], [2026, 818])
    )
    design = replicate_positive_genut(
        gaussian_genut_design(dim=2), num_particles=count
    )

    restored = _restore_cloud_primal(
        particles,
        weights,
        design,
        epsilon=8.0,
        sinkhorn_steps=8,
        balance_steps=8,
        ridge=1.0e-5,
        dual_cap_enabled=True,
        dual_cap_diagonal_steps=3,
        dual_cap_diagonal_strength=0.15,
        dual_cap_pairwise_steps=3,
        dual_cap_pairwise_strength=0.01,
        dual_cap_pairwise_particle_rms_cap=1.5,
        dual_cap_coordinate_cap=0.97,
        dual_cap_coordinate_cap_power=6,
        trust_region_enabled=True,
        trust_region_lm_damping=1.0e-2,
        trust_region_lm_scale_floor=1.0e-4,
        trust_region_radius=0.5,
    )

    assert bool(restored["reset_valid"].numpy())
    assert int(restored["trust_region_solver_id"].numpy()) == 1
    assert bool(tf.reduce_all(tf.math.is_finite(restored["particles"])).numpy())
    assert float(restored["maximum_diagonal_post_cap_particle_rms"].numpy()) < 0.5


def test_sqmc_streaming_reset_matches_dense_with_trust_region() -> None:
    count = 36
    particles = tf.random.stateless_normal([count, 2], [2026, 823])
    weights = tf.nn.softmax(
        0.2 * tf.random.stateless_normal([count], [2026, 824])
    )
    design = replicate_positive_genut(
        gaussian_genut_design(dim=2), num_particles=count
    )
    common = dict(
        epsilon=8.0,
        sinkhorn_steps=8,
        balance_steps=8,
        ridge=1.0e-5,
        dual_cap_enabled=True,
        dual_cap_diagonal_steps=3,
        dual_cap_diagonal_strength=0.15,
        dual_cap_pairwise_steps=3,
        dual_cap_pairwise_strength=0.01,
        dual_cap_pairwise_particle_rms_cap=1.5,
        dual_cap_coordinate_cap=0.97,
        dual_cap_coordinate_cap_power=6,
        trust_region_enabled=True,
        trust_region_lm_damping=1.0e-2,
        trust_region_lm_scale_floor=1.0e-4,
        trust_region_radius=0.5,
    )
    dense = _restore_cloud_primal(
        particles, weights, design, transport_plan_mode="dense", **common
    )
    streamed = _restore_cloud_primal(
        particles,
        weights,
        design,
        transport_plan_mode="streaming",
        transport_row_chunk_size=count,
        transport_col_chunk_size=count,
        **common,
    )
    assert bool(dense["reset_valid"].numpy())
    assert bool(streamed["reset_valid"].numpy())
    assert int(streamed["transport_plan_id"].numpy()) == 1
    tf.debugging.assert_near(
        streamed["particles"], dense["particles"], atol=2.0e-5, rtol=2.0e-5
    )
    tf.debugging.assert_near(
        streamed["post_quotient_column_tv_error"],
        dense["post_quotient_column_tv_error"],
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_austria_streaming_route_matches_dense_value_and_score() -> None:
    count = 36
    horizon = 2
    model = latent_preclip_zhao_cui_sir_austria_model()
    callbacks = austria_sir_callbacks(model)
    _states, observations = model.physical_model.base_model.simulate(
        final_time=horizon, seed=81120
    )
    observations = tf.cast(observations[1 : horizon + 1], tf.float32)
    initial_noise = tf.random.stateless_normal([count, 18], [2026, 825])
    process_noise = tf.random.stateless_normal(
        [horizon, count, 18], [2026, 826]
    )
    ancestor_uniforms = tf.zeros([horizon, count], tf.float32)
    design = cubature_design(dim=18, num_particles=count)
    common = dict(
        ancestry_policy="existing_one_to_one",
        process_ancestor_uniforms=ancestor_uniforms,
        reset_policy="contract_e",
        dual_cap_enabled=True,
        dual_cap_diagonal_steps=3,
        dual_cap_diagonal_strength=0.15,
        dual_cap_pairwise_steps=3,
        dual_cap_pairwise_strength=0.01,
        dual_cap_pairwise_particle_rms_cap=1.5,
        dual_cap_coordinate_cap=0.97,
        dual_cap_coordinate_cap_power=6,
        trust_region_enabled=True,
        trust_region_lm_damping=1.0e-2,
        trust_region_lm_scale_floor=1.0e-4,
        trust_region_radius=0.5,
        score_child_block_size=6,
        epsilon=8.0,
        sinkhorn_steps=8,
        balance_steps=8,
        ridge=1.0e-5,
    )
    inputs = (
        callbacks,
        tf.zeros([3], tf.float32),
        observations,
        initial_noise,
        process_noise,
        design,
    )
    dense = finite_value_standard_score_initial_rqmc(
        *inputs, transport_plan_mode="dense", **common
    )
    streamed = finite_value_standard_score_initial_rqmc(
        *inputs,
        transport_plan_mode="streaming",
        transport_row_chunk_size=count,
        transport_col_chunk_size=count,
        **common,
    )
    assert bool(dense[2]["program_valid"].numpy())
    assert bool(streamed[2]["program_valid"].numpy())
    tf.debugging.assert_near(streamed[0], dense[0], atol=5.0e-4, rtol=5.0e-4)
    tf.debugging.assert_near(streamed[1], dense[1], atol=5.0e-4, rtol=5.0e-4)
    tf.debugging.assert_near(
        streamed[2]["final_particles"],
        dense[2]["final_particles"],
        atol=5.0e-4,
        rtol=5.0e-4,
    )
