from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_tf import (
    prepare_austria_sir_source_order_program,
)
from bayesfilter.highdim.zhao_cui_austria_sir_rank_one_proposal_tf import (
    COMPILER_ROUTE_ID,
    compile_austria_sir_persistent_guide_program,
    compile_austria_sir_rank_one_proposal_branch,
    make_rank_one_branch_tensor_compiler,
)


def _operation_types(concrete) -> set[str]:
    graph_def = concrete.graph.as_graph_def()
    return {node.op for node in graph_def.node} | {
        node.op
        for function in graph_def.library.function
        for node in function.node_def
    }


def test_rank_one_proposal_is_origin_optimal_and_deterministic() -> None:
    first = compile_austria_sir_rank_one_proposal_branch(
        particle_count=32, horizon=3, seed=30901
    )
    second = compile_austria_sir_rank_one_proposal_branch(
        particle_count=32, horizon=3, seed=30901
    )
    assert first.compiler_id == second.compiler_id
    assert first.branch.branch_id == second.branch.branch_id
    assert first.manifest["compiler_route_id"] == COMPILER_ROUTE_ID
    assert first.manifest["rank"] == 1
    assert first.manifest["python_numerical_loop"] is False
    assert first.manifest["numpy_numerical_path"] is False

    program = prepare_austria_sir_source_order_program(first.branch)
    result = program.compiled()(tf.zeros([3], tf.float64))
    particle_count = tf.cast(first.branch.particle_count, tf.float64)
    tf.debugging.assert_near(
        result["ess_by_time"],
        tf.fill([4], particle_count),
        atol=2e-9,
        rtol=2e-12,
    )
    tf.debugging.assert_near(
        result["maximum_normalized_weight_by_time"],
        tf.fill([4], tf.math.reciprocal(particle_count)),
        atol=2e-12,
        rtol=2e-12,
    )
    tf.debugging.assert_near(
        result["log_weight_spread_by_time"],
        tf.zeros([4], tf.float64),
        atol=2e-10,
    )
    assert bool(result["finite"].numpy())


def test_origin_rank_one_t3_candidate_is_rejected_off_origin() -> None:
    compilation = compile_austria_sir_rank_one_proposal_branch(
        particle_count=128, horizon=3, seed=30902
    )
    program = prepare_austria_sir_source_order_program(compilation.branch)
    theta_points = tf.constant(
        [
            [0.03, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
            [0.0, 0.03, 0.0],
            [0.0, -0.03, 0.0],
            [0.0, 0.0, 0.03],
            [0.0, 0.0, -0.03],
            [0.03, -0.03, 0.03],
            [-0.03, 0.03, -0.03],
        ],
        tf.float64,
    )
    viability = []
    for theta in tf.unstack(theta_points):
        result = program.compiled()(theta)
        minimum_ess_fraction = result["minimum_ess"] / tf.cast(
            compilation.branch.particle_count, tf.float64
        )
        maximum_weight = tf.reduce_max(
            result["maximum_normalized_weight_by_time"]
        )
        assert bool(result["finite"].numpy())
        viability.append(
            float(minimum_ess_fraction.numpy()) >= 0.10
            and float(maximum_weight.numpy()) <= 0.10
        )
    assert not all(viability)


def test_rank_one_compiler_graph_has_while_and_no_host_callback() -> None:
    compiler = make_rank_one_branch_tensor_compiler(particle_count=8, horizon=2)
    operations = _operation_types(compiler.get_concrete_function())
    assert operations & {"While", "StatelessWhile"}
    assert not operations & {
        "PyFunc",
        "PyFuncStateless",
        "EagerPyFunc",
        "MapDefun",
    }


def test_persistent_guide_combined_manual_score_matches_same_scalar_tape() -> None:
    program = compile_austria_sir_persistent_guide_program(
        particle_count=8, horizon=2, seed=31111
    )
    claim = program.compiled()
    diagnostic = program.compiled(jit_compile=False)
    theta = tf.constant([0.01, -0.01, 0.02], tf.float64)
    result = claim(theta)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        diagnostic_value = diagnostic(theta)["log_likelihood"]
    tape_score = tape.gradient(diagnostic_value, theta)
    tf.debugging.assert_near(result["score"], tape_score, atol=5e-9, rtol=5e-9)
    tf.debugging.assert_near(
        result["branch_values"],
        tf.reduce_sum(result["branch_log_increments"], axis=0),
        atol=2e-11,
    )
    tf.debugging.assert_near(
        result["branch_scores"],
        tf.reduce_sum(result["branch_increment_scores"], axis=0),
        atol=2e-11,
    )
    assert bool(result["finite"].numpy())


def test_persistent_guide_t3_has_viable_local_branch_over_calibration_box() -> None:
    program = compile_austria_sir_persistent_guide_program(
        particle_count=128, horizon=3, seed=31112
    )
    evaluator = program.compiled()
    theta_points = tf.constant(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [-0.03, 0.0, 0.0],
            [0.0, 0.03, 0.0],
            [0.0, -0.03, 0.0],
            [0.0, 0.0, 0.03],
            [0.0, 0.0, -0.03],
            [0.03, -0.03, 0.03],
            [-0.03, 0.03, -0.03],
        ],
        tf.float64,
    )
    for theta in tf.unstack(theta_points):
        result = evaluator(theta)
        minimum_ess_fraction = tf.reduce_min(
            result["ess_by_time_and_guide"], axis=0
        ) / tf.constant(128.0, tf.float64)
        maximum_weight = tf.reduce_max(
            result["maximum_weight_by_time_and_guide"], axis=0
        )
        viable = (minimum_ess_fraction >= 0.10) & (maximum_weight <= 0.10)
        assert bool(result["finite"].numpy())
        assert bool(tf.reduce_any(viable).numpy())
        assert float(result["branch_effective_count"].numpy()) > 1.0


def test_persistent_guide_compiler_and_evaluator_graphs_are_xla_native() -> None:
    program = compile_austria_sir_persistent_guide_program(
        particle_count=8, horizon=2, seed=31113
    )
    operations = _operation_types(program.compiled().get_concrete_function())
    assert operations & {"While", "StatelessWhile"}
    assert not operations & {
        "PyFunc",
        "PyFuncStateless",
        "EagerPyFunc",
        "MapDefun",
    }


def test_persistent_guide_horizons_share_one_literal_frozen_prefix() -> None:
    short = compile_austria_sir_persistent_guide_program(
        particle_count=8, horizon=2, seed=31114
    )
    full = compile_austria_sir_persistent_guide_program(
        particle_count=8, horizon=20, seed=31114
    )
    prefix = full.prefix(2)

    tf.debugging.assert_equal(short.observations, prefix.observations)
    tf.debugging.assert_equal(short.guide_thetas, prefix.guide_thetas)
    tf.debugging.assert_equal(short.states, prefix.states)
    tf.debugging.assert_equal(
        short.initial_log_proposal_density,
        prefix.initial_log_proposal_density,
    )
    tf.debugging.assert_equal(short.ancestors, prefix.ancestors)
    tf.debugging.assert_equal(
        short.auxiliary_log_probabilities,
        prefix.auxiliary_log_probabilities,
    )
    tf.debugging.assert_equal(
        short.transition_log_proposal_density,
        prefix.transition_log_proposal_density,
    )
    assert prefix.manifest["parent_program_id"] == full.program_id
    assert prefix.manifest["prefix_identity"] == "literal_tensor_prefix_of_parent_program"
