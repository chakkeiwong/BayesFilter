from __future__ import annotations

import json
import os
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
import pytest

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as preparation
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks


def test_preparation_accepts_existing_float32_observation_tensor() -> None:
    chunks = select_transport_chunks(8)
    result = preparation.prepare_contract_e_lgssm_inputs(
        observations=tf.zeros([2, 3], tf.float32),
        estimator_seeds=(81600,),
        num_particles=8,
        fixed_reset_mask=[[True, True]],
        prepared_ridge=[[1.0e-6, 1.0e-6]],
        epsilon=0.5,
        scaling=0.9,
        sinkhorn_steps=2,
        balance_steps=2,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        dtype=tf.float32,
    )
    assert result["prepared"]["observations"].dtype == tf.float32


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase5-tiny-fixture-freeze-v2-2026-07-14.json"
)
ONE_STEP_FIXTURE_PATH = ROOT / "docs/plans" / (
    "bayesfilter-contract-e-canonical-gradient-migration-"
    "phase5-one-step-fixture-freeze-2026-07-14.json"
)
DTYPE = tf.float64
HISTORICAL_KWARGS = {
    "steps": 2,
    "balance_steps": 0,
    "row_chunk_size": 4,
    "col_chunk_size": 4,
}
SHORT_BALANCE_KWARGS = {
    "steps": 2,
    "balance_steps": 1,
    "row_chunk_size": 4,
    "col_chunk_size": 4,
}
VALID_BALANCE_KWARGS = {
    "steps": 2,
    "balance_steps": 100,
    "row_chunk_size": 4,
    "col_chunk_size": 4,
}


def _convert(value: Any) -> Any:
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, str):
        return float(Fraction(value))
    return value


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _one_step_fixture() -> dict[str, Any]:
    return json.loads(ONE_STEP_FIXTURE_PATH.read_text(encoding="utf-8"))


def _prepared(*, reset_mask: list[list[bool]] | None = None) -> dict[str, Any]:
    fixture = _fixture()
    return {
        "observations": _convert(fixture["observations"]),
        "initial_noise": _convert(fixture["initial_noise"]),
        "transition_noise": _convert(fixture["transition_noise"]),
        "fixed_reset_mask": reset_mask or fixture["fixed_reset_mask"],
        "residual_design": _convert(fixture["residual_design"]),
        "prepared_ridge": _convert(fixture["prepared_ridge"]),
        "epsilon": _convert(fixture["transport"]["epsilon"]),
        "scaling": _convert(fixture["transport"]["scaling"]),
    }


def _prepared_from_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "observations": _convert(fixture["observations"]),
        "initial_noise": _convert(fixture["initial_noise"]),
        "transition_noise": _convert(fixture["transition_noise"]),
        "fixed_reset_mask": fixture["fixed_reset_mask"],
        "residual_design": _convert(fixture["residual_design"]),
        "prepared_ridge": _convert(fixture["prepared_ridge"]),
        "epsilon": _convert(fixture["transport"]["epsilon"]),
        "scaling": _convert(fixture["transport"]["scaling"]),
    }


def _prepared_dtype(dtype: tf.dtypes.DType) -> dict[str, tf.Tensor]:
    return canonical._as_prepared_tensors(_prepared(), dtype=dtype)


def _tensors(*, reset_mask: list[list[bool]] | None = None) -> dict[str, tf.Tensor]:
    return canonical._as_prepared_tensors(_prepared(reset_mask=reset_mask))


def _theta() -> tf.Tensor:
    return tf.constant(_convert(_fixture()["center_theta"]), DTYPE)


def _ordered_binary64(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    if bits == 1 << 63:
        bits = 0
    if bits & (1 << 63):
        return (~bits) & ((1 << 64) - 1)
    return bits | (1 << 63)


def _max_ulp_distance(left: tf.Tensor, right: tf.Tensor) -> int:
    if left.dtype == tf.float32 and right.dtype == tf.float32:
        def ordered_binary32(value: float) -> int:
            bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
            if bits == 1 << 31:
                bits = 0
            if bits & (1 << 31):
                return (~bits) & ((1 << 32) - 1)
            return bits | (1 << 31)

        left_values = tf.reshape(left, [-1]).numpy()
        right_values = tf.reshape(right, [-1]).numpy()
        return max(
            abs(ordered_binary32(a) - ordered_binary32(b))
            for a, b in zip(left_values, right_values, strict=True)
        )
    left_values = tf.reshape(left, [-1]).numpy()
    right_values = tf.reshape(right, [-1]).numpy()
    return max(
        abs(_ordered_binary64(a) - _ordered_binary64(b))
        for a, b in zip(left_values, right_values, strict=True)
    )


def _forward_parameter_jacobian(
    theta: tf.Tensor,
    value_fn,
) -> tf.Tensor:
    columns = []
    for index in range(canonical.PARAMETER_COUNT):
        with tf.autodiff.ForwardAccumulator(
            theta,
            tf.one_hot(index, canonical.PARAMETER_COUNT, dtype=theta.dtype),
        ) as accumulator:
            value = value_fn(theta)
        columns.append(accumulator.jvp(value))
    return tf.stack(columns, axis=-1)


def test_fixture_shapes_and_fixed_input_invariants() -> None:
    tensors = _tensors()
    assert tensors["observations"].shape == (2, 3)
    assert tensors["initial_noise"].shape == (2, 4, 3)
    assert tensors["transition_noise"].shape == (2, 2, 4, 3)
    assert tensors["fixed_reset_mask"].shape == (2, 2)
    assert tensors["residual_design"].shape == (2, 2, 4, 3)
    assert tensors["prepared_ridge"].shape == (2, 2)
    tf.debugging.assert_equal(
        tf.reduce_sum(tensors["residual_design"], axis=2),
        tf.zeros([2, 2, 3], DTYPE),
    )
    assert canonical.PARAMETER_NAMES == (
        "phi1",
        "phi2",
        "phi3",
        "q_scale",
        "r_scale",
    )


def test_initialization_dependency_map_is_exact_same_graph_jvp() -> None:
    theta = _theta()
    noise = _tensors()["initial_noise"]
    manual = (
        noise[:, :, :, None]
        * canonical._lgssm_component_tangents(theta, 2)["d_initial_std"][
            None, None, :, :
        ]
    )
    automatic = _forward_parameter_jacobian(
        theta,
        lambda value: noise
        * canonical._lgssm_components(value, 2)["initial_std"][None, None, :],
    )
    assert manual.shape == (2, 4, 3, canonical.PARAMETER_COUNT)
    assert _max_ulp_distance(manual, automatic) == 0
    tangent = canonical._lgssm_component_tangents(theta, 2)["d_initial_std"]
    tf.debugging.assert_equal(tangent[:, 4], tf.zeros([3], DTYPE))
    for coordinate in range(3):
        for parameter in range(3):
            if coordinate != parameter:
                assert float(tangent[coordinate, parameter]) == 0.0


def test_floor_free_normalization_primal_jvp_vjp_and_shift_invariance() -> None:
    logits = tf.constant(
        [[-3.1, -2.3, -4.7, -1.2], [-0.5, -2.25, -1.75, -3.0]], DTYPE
    )
    tangents = tf.reshape(tf.range(40, dtype=DTYPE) / 17.0, [2, 4, 5]) - 1.0
    manual = canonical._normalize_log_weights_jvp_core(logits, tangents)
    for primal_name, tangent_name in (
        ("increment", "increment_tangent"),
        ("normalized_log_weights", "normalized_log_weights_tangent"),
        ("normalized_weights", "normalized_weights_tangent"),
    ):
        columns = []
        for index in range(canonical.PARAMETER_COUNT):
            with tf.autodiff.ForwardAccumulator(
                logits, tangents[..., index]
            ) as accumulator:
                value = canonical._normalize_log_weights_forward_core(logits)[
                    primal_name
                ]
            columns.append(accumulator.jvp(value))
        automatic = tf.stack(columns, axis=-1)
        assert _max_ulp_distance(manual[tangent_name], automatic) == 0

    shift = tf.constant([[8.0], [-4.0]], DTYPE)
    shifted = canonical._normalize_log_weights_forward_core(logits + shift)
    base = canonical._normalize_log_weights_forward_core(logits)
    tf.debugging.assert_near(
        shifted["normalized_log_weights"], base["normalized_log_weights"], atol=0.0
    )
    tf.debugging.assert_near(
        shifted["increment"], base["increment"] + tf.reshape(shift, [-1]), atol=0.0
    )

    increment_bar = tf.constant([0.25, -0.5], DTYPE)
    log_weight_bar = tf.reshape(tf.range(8, dtype=DTYPE), [2, 4]) / 11.0
    reverse = canonical._normalize_log_weights_vjp_core(
        logits, increment_bar, log_weight_bar
    )["logits_bar"]
    with tf.GradientTape() as tape:
        tape.watch(logits)
        forward = canonical._normalize_log_weights_forward_core(logits)
        objective = tf.reduce_sum(forward["increment"] * increment_bar)
        objective += tf.reduce_sum(
            forward["normalized_log_weights"] * log_weight_bar
        )
    automatic_reverse = tape.gradient(objective, logits)
    tf.debugging.assert_near(reverse, automatic_reverse, atol=1.0e-15, rtol=0.0)


def test_two_batch_time_sum_mean_aggregation_and_final_direction_axis() -> None:
    tensors = _tensors(reset_mask=[[False, False], [False, False]])
    theta = _theta()
    primal = canonical._canonical_primal_core(
        theta, tensors, **HISTORICAL_KWARGS
    )
    manual = canonical._canonical_manual_jvp_core(
        theta, tensors, **HISTORICAL_KWARGS
    )
    automatic = _forward_parameter_jacobian(
        theta,
        lambda value: canonical._canonical_primal_core(
            value, tensors, **HISTORICAL_KWARGS
        )["per_batch_log_likelihood"],
    )
    assert manual["per_batch_score"].shape == (2, canonical.PARAMETER_COUNT)
    assert manual["final_particles_tangent"].shape == (2, 4, 3, 5)
    assert manual["final_log_weights_tangent"].shape == (2, 4, 5)
    assert _max_ulp_distance(manual["per_batch_score"], automatic) == 0
    tf.debugging.assert_equal(
        primal["per_batch_log_likelihood"],
        tf.reduce_sum(primal["increment_history"], axis=1),
    )
    tf.debugging.assert_equal(
        primal["objective"], tf.reduce_mean(primal["per_batch_log_likelihood"])
    )
    tf.debugging.assert_equal(
        manual["score"], tf.reduce_mean(manual["per_batch_score"], axis=0)
    )


def test_one_batch_one_step_active_reset_and_all_parameter_sensitivity() -> None:
    fixture = _one_step_fixture()
    tensors = canonical._as_prepared_tensors(_prepared_from_fixture(fixture))
    theta = tf.constant(_convert(fixture["center_theta"]), DTYPE)
    primal = canonical._canonical_primal_core(
        theta, tensors, **HISTORICAL_KWARGS
    )
    manual = canonical._canonical_manual_jvp_core(
        theta, tensors, **HISTORICAL_KWARGS
    )
    automatic = _forward_parameter_jacobian(
        theta,
        lambda value: canonical._canonical_primal_core(
            value, tensors, **HISTORICAL_KWARGS
        )["per_batch_log_likelihood"],
    )
    assert manual["per_batch_score"].shape == (1, 5)
    assert _max_ulp_distance(manual["per_batch_score"], automatic) == 0
    tf.debugging.assert_equal(primal["valid_chart"], [False])
    tf.debugging.assert_equal(
        primal["quotient_marginal_valid_history"], [[False]]
    )
    tf.debugging.assert_greater(primal["minimum_mass"], tf.zeros([1], DTYPE))
    uniform = tf.fill([1, 4], -tf.math.log(tf.constant(4.0, DTYPE)))
    tf.debugging.assert_equal(primal["final_log_weights"], uniform)

    step = tf.constant(1.0 / 1024.0, DTYPE)
    for index in range(canonical.PARAMETER_COUNT):
        perturbed = canonical._canonical_primal_core(
            theta + step * tf.one_hot(index, 5, dtype=DTYPE),
            tensors,
            **HISTORICAL_KWARGS,
        )
        assert float(perturbed["objective"]) != float(primal["objective"])


def test_exact_chunk_mixed_reset_full_graph_matches_ad_within_one_ulp() -> None:
    tensors = _tensors()
    theta = _theta()
    primal = canonical._canonical_primal_core(
        theta, tensors, **SHORT_BALANCE_KWARGS
    )
    manual = canonical._canonical_manual_jvp_core(
        theta, tensors, **SHORT_BALANCE_KWARGS
    )
    automatic = _forward_parameter_jacobian(
        theta,
        lambda value: canonical._canonical_primal_core(
            value, tensors, **SHORT_BALANCE_KWARGS
        )["per_batch_log_likelihood"],
    )
    # Exact one-block tiling changes the reduction tree from the archived K=2
    # fixture, but the manual and automatic derivatives remain within one ULP.
    assert _max_ulp_distance(manual["per_batch_score"], automatic) <= 1
    tf.debugging.assert_equal(primal["valid_chart"], [False, False])
    tf.debugging.assert_greater(primal["minimum_mass"], tf.zeros([2], DTYPE))
    assert primal["sinkhorn_running_branch"].shape == (2, 2, 2)
    assert primal["diameter_max_mask"].shape == (2, 2, 3)
    assert primal["geometry_max_mask"].shape == (2, 2, 4, 3)
    assert primal["geometry_min_mask"].shape == (2, 2, 4, 3)

    uniform = tf.fill([4], -tf.math.log(tf.constant(4.0, DTYPE)))
    tf.debugging.assert_equal(primal["final_log_weights"][1], uniform)
    assert bool(
        tf.reduce_any(tf.not_equal(primal["final_log_weights"][0], uniform)).numpy()
    )


def test_repeated_center_identity_and_executable_invalid_physical_chart() -> None:
    tensors = _tensors()
    theta = _theta()
    first = canonical.canonical_value_and_score_core(
        theta, tensors, **SHORT_BALANCE_KWARGS
    )
    second = canonical.canonical_value_and_score_core(
        theta, tensors, **SHORT_BALANCE_KWARGS
    )
    for name in (
        "objective",
        "per_batch_log_likelihood",
        "score",
        "per_batch_score",
        "valid_chart",
        "minimum_mass",
        "flow_valid_history",
        "geometry_valid_history",
        "quotient_valid_history",
        "reset_valid_history",
        "diameter_max_mask",
        "geometry_max_mask",
        "geometry_min_mask",
        "epsilon0_floor_inactive",
        "sinkhorn_running_branch",
    ):
        tf.debugging.assert_equal(first[name], second[name])

    invalid_theta = tf.tensor_scatter_nd_update(theta, [[3]], [-theta[3]])
    invalid = canonical._canonical_primal_core(
        invalid_theta, tensors, **SHORT_BALANCE_KWARGS
    )
    tf.debugging.assert_equal(invalid["valid_chart"], [False, False])
    tf.debugging.assert_all_finite(
        invalid["per_batch_log_likelihood"], "executable invalid physical chart"
    )


def test_fused_loop_matches_separated_finite_program_value_and_score() -> None:
    tensors = _tensors()
    theta = _theta()
    primal = canonical._canonical_primal_core(
        theta, tensors, **SHORT_BALANCE_KWARGS
    )
    tangent = canonical._canonical_manual_jvp_core(
        theta, tensors, **SHORT_BALANCE_KWARGS
    )
    fused = canonical._canonical_fused_loop_core(
        theta,
        tensors,
        execute_contract_e=True,
        **SHORT_BALANCE_KWARGS,
    )
    tf.debugging.assert_near(
        fused["per_batch_log_likelihood"],
        primal["per_batch_log_likelihood"],
        atol=2e-13,
        rtol=2e-13,
    )
    tf.debugging.assert_near(
        fused["per_batch_score"],
        tangent["per_batch_score"],
        atol=2e-12,
        rtol=2e-12,
    )
    assert int(fused["work_sinkhorn_state_constructions"]) == 2
    assert int(fused["work_terminal_balance_state_constructions"]) == 2
    assert int(fused["work_transport_tile_sweeps"]) == 2
    assert int(fused["work_marginal_tile_sweeps"]) == 0
    assert int(fused["work_diagnostic_solver_reconstructions"]) == 0


def test_all_inactive_factory_executes_zero_ot_work() -> None:
    prepared = _prepared(reset_mask=[[False, False], [False, False]])
    callable_ = canonical.make_canonical_value_and_score_tf(
        prepared, jit_compile=False, **SHORT_BALANCE_KWARGS
    )
    result = callable_(_theta())
    assert int(result["work_sinkhorn_state_constructions"]) == 0
    assert int(result["work_terminal_balance_state_constructions"]) == 0
    assert int(result["work_transport_tile_sweeps"]) == 0
    assert int(result["work_marginal_tile_sweeps"]) == 0
    assert int(result["work_diagnostic_solver_reconstructions"]) == 0
    tf.debugging.assert_equal(result["active_reset_history"], False)
    concrete = callable_.get_concrete_function().graph.as_graph_def()
    reachable_text = str(concrete)
    assert "contract_e_streaming_forward_jvp" not in reachable_text


def test_owned_source_forbids_historical_and_floored_paths() -> None:
    source = Path(canonical.__file__).read_text(encoding="utf-8")
    assert "docs.benchmarks" not in source
    assert "numpy" not in source
    assert "tf.linalg.inv" not in source
    assert "stop_gradient" not in source
    assert "tf.random" not in source
    assert "log(tf.maximum" not in source
    assert "_contract_e_streaming_forward_core" in source
    assert "_contract_e_streaming_jvp_core" in source


def test_float32_shared_core_manual_jvp_matches_forward_autodiff() -> None:
    tensors = _prepared_dtype(tf.float32)
    theta = tf.constant(_convert(_fixture()["center_theta"]), tf.float32)
    primal = canonical._canonical_primal_core(
        theta, tensors, **SHORT_BALANCE_KWARGS
    )
    manual = canonical._canonical_manual_jvp_core(
        theta, tensors, **SHORT_BALANCE_KWARGS
    )
    automatic = _forward_parameter_jacobian(
        theta,
        lambda value: canonical._canonical_primal_core(
            value, tensors, **SHORT_BALANCE_KWARGS
        )["per_batch_log_likelihood"],
    )
    assert manual["per_batch_score"].dtype == tf.float32
    assert _max_ulp_distance(manual["per_batch_score"], automatic) <= 1
    assert _max_ulp_distance(
        manual["score"], tf.reduce_mean(automatic, axis=0)
    ) <= 1
    tf.debugging.assert_equal(primal["valid_chart"], [False, False])


def test_float32_factory_uses_one_shared_value_and_score_core() -> None:
    callable_ = canonical.make_canonical_value_and_score_tf(
        _prepared(), dtype=tf.float32, jit_compile=False, **SHORT_BALANCE_KWARGS
    )
    theta = tf.constant(_convert(_fixture()["center_theta"]), tf.float32)
    first = callable_(theta)
    second = callable_(theta)
    for name in (
        "objective",
        "per_batch_log_likelihood",
        "score",
        "per_batch_score",
        "valid_chart",
        "minimum_mass",
        "flow_valid_history",
        "geometry_valid_history",
        "quotient_valid_history",
        "reset_valid_history",
        "diameter_max_mask",
        "geometry_max_mask",
        "geometry_min_mask",
        "epsilon0_floor_inactive",
        "sinkhorn_running_branch",
    ):
        tf.debugging.assert_equal(first[name], second[name])
    assert len(callable_._list_all_concrete_functions_for_serialization()) == 1
    tf.debugging.assert_all_finite(first["objective"], "float32 objective")
    tf.debugging.assert_all_finite(first["score"], "float32 score")


def test_prepared_input_factory_matches_bound_factory_on_same_finite_program() -> None:
    prepared = canonical._as_prepared_tensors(_prepared(), dtype=tf.float32)
    theta = tf.constant(_convert(_fixture()["center_theta"]), tf.float32)
    bound = canonical.make_canonical_value_and_score_tf(
        prepared,
        dtype=tf.float32,
        jit_compile=False,
        **SHORT_BALANCE_KWARGS,
    )
    prepared_input = canonical.make_canonical_prepared_value_and_score_tf(
        batch_size=2,
        time_steps=2,
        num_particles=4,
        dtype=tf.float32,
        jit_compile=False,
        **SHORT_BALANCE_KWARGS,
    )
    expected = bound(theta)
    actual = prepared_input(theta, prepared)
    for name in (
        "objective",
        "per_batch_log_likelihood",
        "score",
        "per_batch_score",
        "valid_chart",
        "reset_valid_history",
        "quotient_marginal_valid_history",
        "tv_column_error_history",
        "maximum_row_error_history",
        "work_sinkhorn_state_constructions",
        "work_terminal_balance_state_constructions",
        "work_transport_tile_sweeps",
        "work_marginal_tile_sweeps",
        "work_diagnostic_solver_reconstructions",
    ):
        tf.debugging.assert_equal(actual[name], expected[name])
    assert len(prepared_input._list_all_concrete_functions_for_serialization()) == 1


def test_factory_rejects_noncanonical_dtype() -> None:
    with pytest.raises(ValueError, match="float32 or float64"):
        canonical.make_canonical_value_and_score_tf(
            _prepared(),
            dtype=tf.float16,
            jit_compile=False,
            **SHORT_BALANCE_KWARGS,
        )


def test_canonical_factory_rejects_zero_terminal_balance() -> None:
    with pytest.raises(ValueError, match="balance_steps"):
        canonical.make_canonical_value_and_score_tf(
            _prepared(),
            steps=2,
            balance_steps=0,
            row_chunk_size=4,
            col_chunk_size=4,
            jit_compile=False,
        )


def test_active_reset_validity_includes_consumed_plan_marginals() -> None:
    tensors = _tensors()
    theta = _theta()
    unbalanced = canonical._canonical_primal_core(
        theta,
        tensors,
        steps=2,
        balance_steps=0,
        row_chunk_size=4,
        col_chunk_size=4,
    )
    balanced = canonical._canonical_primal_core(
        theta, tensors, **VALID_BALANCE_KWARGS
    )
    active = tensors["fixed_reset_mask"]
    assert not bool(
        tf.reduce_all(
            tf.boolean_mask(
                unbalanced["quotient_marginal_valid_history"], active
            )
        ).numpy()
    )
    tf.debugging.assert_equal(
        tf.boolean_mask(unbalanced["reset_valid_history"], active),
        tf.zeros_like(tf.boolean_mask(active, active)),
    )
    assert bool(
        tf.reduce_all(
            tf.boolean_mask(balanced["quotient_marginal_valid_history"], active)
        ).numpy()
    )
