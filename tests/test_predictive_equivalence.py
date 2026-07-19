from __future__ import annotations

import copy
import inspect
from dataclasses import replace

import pytest
import tensorflow as tf

import bayesfilter.inference.predictive_equivalence as predictive_equivalence
from bayesfilter.inference.predictive_equivalence import (
    MMDInterval,
    PredictiveContractError,
    PredictiveStatisticsConfig,
    SimultaneousIntervals,
    adapt_ssl_lstm_observations,
    chain_batch_long_run_covariance,
    chain_batch_means,
    classify_predictive_evidence,
    cross_chain_linear_mmd,
    cross_chain_mmd_upper_interval,
    fixed_rbf_mmd,
    hierarchical_resample_indices,
    mean_log_variance_influence,
    pooled_pairwise_distance_scale,
    simultaneous_feature_intervals,
    standardize_forecast_paths,
    summarize_forecast_paths,
)


F64 = tf.float64


def _status(value: tf.Tensor) -> str:
    raw = value.numpy()
    return raw.decode("ascii") if isinstance(raw, bytes) else str(raw)


def _bands() -> tuple[tf.Tensor, tf.Tensor]:
    return tf.constant([0.5, 1.5], F64), tf.constant([0.25, 0.75], F64)


def _four_chain_paths(offset: float = 0.0) -> tf.Tensor:
    values = tf.reshape(tf.range(4 * 8 * 3 * 10, dtype=F64), [4, 8, 3, 10])
    chain = tf.range(4, dtype=F64)[:, None, None, None] * 0.017
    draw = tf.range(8, dtype=F64)[None, :, None, None] * 0.013
    replication = tf.range(3, dtype=F64)[None, None, :, None] * 0.011
    return values / 100.0 + chain + draw + replication + offset


def _perturbed_four_chain_paths(strength: float = 0.02) -> tf.Tensor:
    draw_pattern = (
        tf.range(1, 9, dtype=F64)[None, :, None, None] / tf.constant(8.0, F64)
    )
    horizon_pattern = (
        tf.range(1, 11, dtype=F64)[None, None, None, :]
        / tf.constant(10.0, F64)
    )
    return _four_chain_paths() + strength * draw_pattern * horizon_pattern


def _valid_feature_interval(
    *,
    lower: tuple[float, ...] = (-0.1, -0.1),
    upper: tuple[float, ...] = (0.1, 0.1),
    alpha: float = 0.03,
) -> SimultaneousIntervals:
    lower_tensor = tf.constant(lower, F64)
    upper_tensor = tf.constant(upper, F64)
    estimate = (lower_tensor + upper_tensor) / 2.0
    probability = 1.0 - alpha / (2.0 * len(lower))
    critical = tf.sqrt(tf.constant(2.0, F64)) * tf.math.erfinv(
        tf.constant(2.0 * probability - 1.0, F64)
    )
    standard_error = (upper_tensor - lower_tensor) / (2.0 * critical)
    return simultaneous_feature_intervals(
        estimate,
        feature_alpha=float(alpha),
        method="bonferroni_studentized",
        standard_error=standard_error,
        jit_compile=False,
    )


def _valid_mmd_interval(
    *,
    alpha: float = 0.02,
) -> MMDInterval:
    bands, weights = _bands()
    statistic = cross_chain_linear_mmd(
        _four_chain_paths(),
        _perturbed_four_chain_paths(0.02),
        bandwidths=bands,
        mixture_weights=weights,
        chain_pair_schedule=tf.constant([[0, 1], [2, 3]], tf.int32),
        independent_arm_banks_verified=True,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=False,
    )
    return cross_chain_mmd_upper_interval(
        statistic,
        mmd_alpha=float(alpha),
        block_length=2,
        jit_compile=False,
    )


def test_adapter_requires_exact_a2_shape_float64_and_finite() -> None:
    values = tf.reshape(tf.range(2 * 3 * 10, dtype=F64), [2, 3, 10, 1])
    adapted = adapt_ssl_lstm_observations(values)
    assert adapted.shape == (1, 2, 3, 10)
    tf.debugging.assert_near(adapted[0], tf.squeeze(values, -1))

    with pytest.raises(PredictiveContractError, match="float64"):
        adapt_ssl_lstm_observations(tf.cast(values, tf.float32))
    with pytest.raises(PredictiveContractError, match="shape"):
        adapt_ssl_lstm_observations(tf.zeros([2, 3, 10, 2], F64))
    with pytest.raises(PredictiveContractError, match="finite"):
        adapt_ssl_lstm_observations(
            tf.tensor_scatter_nd_update(values, [[0, 0, 0, 0]], [float("nan")])
        )


def test_summary_matches_direct_sample_formulas() -> None:
    base = tf.range(1, 61, dtype=F64)
    paths = tf.reshape(base, [1, 2, 3, 10]) / 10.0
    summary = summarize_forecast_paths(
        paths, PredictiveStatisticsConfig(jit_compile=False)
    )
    flat = tf.reshape(paths, [-1, 10])
    expected_mean = tf.reduce_mean(flat, 0)
    centered = flat - expected_mean
    expected_variance = tf.reduce_sum(tf.square(centered), 0) / 5.0
    expected_covariance = tf.matmul(centered, centered, transpose_a=True) / 5.0
    tf.debugging.assert_near(summary.means, expected_mean)
    tf.debugging.assert_near(summary.variances, expected_variance)
    tf.debugging.assert_near(summary.log_variances, tf.math.log(expected_variance))
    tf.debugging.assert_near(summary.cross_horizon_covariance, expected_covariance)
    assert summary.central_moments.shape == (2, 10)
    assert summary.quantiles.shape == (5, 10)
    assert int(summary.path_count) == 6
    assert _status(summary.status) == "VALID"


def test_summary_default_xla_matches_eager() -> None:
    paths = tf.reshape(tf.range(1, 81, dtype=F64), [1, 2, 4, 10]) / 17.0
    eager = summarize_forecast_paths(
        paths, PredictiveStatisticsConfig(jit_compile=False)
    )
    compiled = summarize_forecast_paths(paths)
    for name in (
        "means",
        "variances",
        "log_variances",
        "central_moments",
        "quantiles",
        "cross_horizon_covariance",
    ):
        tf.debugging.assert_near(getattr(compiled, name), getattr(eager, name), atol=1e-12)


def test_summary_rejects_zero_variance_and_noncanonical_config() -> None:
    with pytest.raises(PredictiveContractError, match="invalid variance"):
        summarize_forecast_paths(
            tf.ones([1, 2, 2, 10], F64),
            PredictiveStatisticsConfig(jit_compile=False),
        )
    with pytest.raises(PredictiveContractError, match="horizon"):
        summarize_forecast_paths(
            tf.zeros([1, 2, 2, 9], F64),
            PredictiveStatisticsConfig(horizon=9, jit_compile=False),
        )


@pytest.mark.parametrize(
    "config,match",
    [
        (object(), "PredictiveStatisticsConfig"),
        (PredictiveStatisticsConfig(horizon=True), "horizon must be an integer"),
        (PredictiveStatisticsConfig(horizon=10.0), "horizon must be an integer"),
        (
            PredictiveStatisticsConfig(quantile_probabilities=(0.05, float("nan"))),
            "finite Python floats",
        ),
        (
            PredictiveStatisticsConfig(quantile_probabilities=(0.05, 1)),
            "finite Python floats",
        ),
        (
            PredictiveStatisticsConfig(central_moment_orders=(3, 3.5)),
            "int32-compatible",
        ),
        (
            PredictiveStatisticsConfig(central_moment_orders=(3, True)),
            "int32-compatible",
        ),
    ],
)
def test_summary_rejects_non_exact_config_types(
    config: object, match: str
) -> None:
    with pytest.raises(PredictiveContractError, match=match):
        summarize_forecast_paths(
            tf.reshape(tf.range(40, dtype=F64), [1, 2, 2, 10]),
            config,  # type: ignore[arg-type]
        )


def test_standardization_strict_scale_and_floor_contract() -> None:
    paths = tf.constant([[1.0, 4.0], [3.0, 8.0]], F64)
    result = standardize_forecast_paths(
        paths,
        tf.constant([1.0, 2.0], F64),
        tf.constant([2.0, 3.0], F64),
        scale_floor=tf.constant(0.5, F64),
        jit_compile=False,
    )
    tf.debugging.assert_near(
        result, tf.constant([[0.0, 2.0 / 3.0], [1.0, 2.0]], F64)
    )
    with pytest.raises(PredictiveContractError, match="floor use"):
        standardize_forecast_paths(
            paths,
            tf.zeros([2], F64),
            tf.constant([0.1, 1.0], F64),
            scale_floor=tf.constant(0.5, F64),
            jit_compile=False,
        )
    allowed = standardize_forecast_paths(
        paths,
        tf.zeros([2], F64),
        tf.constant([0.1, 1.0], F64),
        scale_floor=tf.constant(0.5, F64),
        allow_floor_use=True,
        jit_compile=False,
    )
    tf.debugging.assert_near(allowed[0], tf.constant([2.0, 4.0], F64))


def test_standardization_dynamic_graph_scale_floor_fails_closed() -> None:
    @tf.function(
        input_signature=(
            tf.TensorSpec([2, 2], F64),
            tf.TensorSpec([2], F64),
            tf.TensorSpec([2], F64),
            tf.TensorSpec([], F64),
        ),
        autograph=False,
    )
    def dynamic_call(
        paths: tf.Tensor,
        center: tf.Tensor,
        scale: tf.Tensor,
        floor: tf.Tensor,
    ) -> tf.Tensor:
        return standardize_forecast_paths(
            paths,
            center,
            scale,
            scale_floor=floor,
            allow_floor_use=False,
            jit_compile=False,
        )

    with pytest.raises(tf.errors.InvalidArgumentError, match="scale_floor use"):
        dynamic_call(
            tf.ones([2, 2], F64),
            tf.zeros([2], F64),
            tf.constant([0.1, 1.0], F64),
            tf.constant(0.5, F64),
        )


def test_quadratic_mmd_u_v_semantics_and_signed_u() -> None:
    bands, weights = _bands()
    left = tf.constant([[0.0], [2.0]], F64)
    # With identical finite samples, the V-form is zero while diagonal exclusion
    # makes the unbiased U-form negative; this guards against clipping at zero.
    right = tf.identity(left)
    result = fixed_rbf_mmd(
        left,
        right,
        bandwidths=bands,
        mixture_weights=weights,
        sampling_contract="iid_oracle_fixture",
        iid_samples_verified=True,
        independent_arm_banks_verified=True,
        jit_compile=False,
    )
    assert float(result.squared_mmd_u) < 0.0
    assert float(result.squared_mmd_v_biased) >= float(result.squared_mmd_u)
    assert not result.inference_admissible
    assert result.iid_samples_verified
    assert result.independent_arm_banks_verified
    assert result.per_bandwidth_u.shape == (2,)

    dependent = fixed_rbf_mmd(
        tf.reshape(_four_chain_paths(), [-1, 10]),
        tf.reshape(_four_chain_paths(0.1), [-1, 10]),
        bandwidths=bands,
        mixture_weights=weights,
        sampling_contract="dependent_descriptive_only",
        jit_compile=False,
    )
    assert not dependent.inference_admissible

    label_only = fixed_rbf_mmd(
        left,
        right,
        bandwidths=bands,
        mixture_weights=weights,
        sampling_contract="iid_oracle_fixture",
        jit_compile=False,
    )
    assert not label_only.inference_admissible
    assert _status(label_only.status) == "VALID"


def test_mmd_rejects_bad_kernel_contracts() -> None:
    samples = tf.constant([[0.0], [1.0]], F64)
    with pytest.raises(PredictiveContractError, match="unique"):
        fixed_rbf_mmd(
            samples,
            samples,
            bandwidths=tf.constant([1.0, 1.0], F64),
            mixture_weights=tf.constant([0.5, 0.5], F64),
            sampling_contract="iid_oracle_fixture",
            jit_compile=False,
        )
    with pytest.raises(PredictiveContractError, match="sum to one"):
        fixed_rbf_mmd(
            samples,
            samples,
            bandwidths=tf.constant([1.0, 2.0], F64),
            mixture_weights=tf.constant([0.2, 0.2], F64),
            sampling_contract="iid_oracle_fixture",
            jit_compile=False,
        )
    with pytest.raises(PredictiveContractError, match="rank-two"):
        fixed_rbf_mmd(
            tf.zeros([1, 2, 2, 10], F64),
            tf.ones([1, 2, 2, 10], F64),
            bandwidths=tf.constant([1.0], F64),
            mixture_weights=tf.constant([1.0], F64),
            sampling_contract="iid_oracle_fixture",
            jit_compile=False,
        )


def test_cross_chain_linear_mmd_admission_and_sequence() -> None:
    bands, weights = _bands()
    schedule = tf.constant([[0, 1], [2, 3]], tf.int32)
    result = cross_chain_linear_mmd(
        _four_chain_paths(),
        _perturbed_four_chain_paths(0.2),
        bandwidths=bands,
        mixture_weights=weights,
        chain_pair_schedule=schedule,
        independent_arm_banks_verified=True,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=False,
    )
    assert result.kernel_contrast_sequence.shape == (2, 8)
    assert result.inference_admissible
    assert _status(result.status) == "VALID"
    tf.debugging.assert_near(
        result.squared_mmd_linear,
        tf.reduce_mean(result.kernel_contrast_sequence),
    )

    nonadmitted = cross_chain_linear_mmd(
        _four_chain_paths(),
        _four_chain_paths(0.2),
        bandwidths=bands,
        mixture_weights=weights,
        chain_pair_schedule=schedule,
        independent_arm_banks_verified=False,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=False,
    )
    assert not nonadmitted.inference_admissible
    assert _status(nonadmitted.status) == "INVALID_HARD_VETO"


def test_cross_chain_linear_mmd_supports_two_chain_mechanics_only() -> None:
    bands, weights = _bands()
    left = _four_chain_paths()[:2]
    right = _four_chain_paths(0.2)[:2]
    result = cross_chain_linear_mmd(
        left,
        right,
        bandwidths=bands,
        mixture_weights=weights,
        chain_pair_schedule=tf.constant([[0, 1]], tf.int32),
        independent_arm_banks_verified=True,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=False,
    )
    assert result.kernel_contrast_sequence.shape == (1, 8)
    assert not result.inference_admissible
    assert _status(result.status) == "VALID"
    interval = cross_chain_mmd_upper_interval(
        result, mmd_alpha=0.02, block_length=2
    )
    assert not interval.inference_admissible
    assert _status(interval.status) == "INVALID_HARD_VETO"


def test_cross_chain_linear_mmd_rejects_truthy_semantic_flags() -> None:
    bands, weights = _bands()
    with pytest.raises(PredictiveContractError, match="Python bool"):
        cross_chain_linear_mmd(
            _four_chain_paths(),
            _four_chain_paths(0.2),
            bandwidths=bands,
            mixture_weights=weights,
            chain_pair_schedule=tf.constant([[0, 1], [2, 3]], tf.int32),
            independent_arm_banks_verified="yes",  # type: ignore[arg-type]
            stationarity_verified=True,
            mixing_verified=True,
            jit_compile=False,
        )


@pytest.mark.parametrize(
    "schedule,match",
    [
        (tf.constant([[0, 0], [2, 3]], tf.int32), "both sides"),
        (tf.constant([[0, 1], [1, 2]], tf.int32), "disjoint"),
        (tf.constant([[0, 1], [2, 4]], tf.int32), "out of bounds"),
    ],
)
def test_cross_chain_linear_mmd_rejects_invalid_schedules(
    schedule: tf.Tensor, match: str
) -> None:
    bands, weights = _bands()
    with pytest.raises(PredictiveContractError, match=match):
        cross_chain_linear_mmd(
            _four_chain_paths(),
            _four_chain_paths(0.2),
            bandwidths=bands,
            mixture_weights=weights,
            chain_pair_schedule=schedule,
            independent_arm_banks_verified=True,
            stationarity_verified=True,
            mixing_verified=True,
            jit_compile=False,
        )


def test_chain_batch_means_preserves_chain_and_trailing_axes() -> None:
    values = tf.reshape(tf.range(2 * 8 * 3, dtype=F64), [2, 8, 3])
    batches = chain_batch_means(values, block_length=2)
    assert batches.shape == (2, 4, 3)
    tf.debugging.assert_near(batches[:, 0], tf.reduce_mean(values[:, :2], axis=1))
    with pytest.raises(PredictiveContractError, match="remainder"):
        chain_batch_means(values, block_length=3)
    with pytest.raises(PredictiveContractError, match="two complete"):
        chain_batch_means(values, block_length=8)


def test_long_run_covariance_matches_manual_batch_means_formula() -> None:
    values = tf.reshape(tf.range(2 * 8 * 2, dtype=F64), [2, 8, 2]) / 7.0
    result = chain_batch_long_run_covariance(
        values,
        block_length=2,
        ridge_ladder=(0.0, 1.0e-8),
        condition_number_max=1.0e16,
        jit_compile=False,
    )
    batches = chain_batch_means(values, block_length=2, jit_compile=False)
    centered = batches - tf.reduce_mean(batches, axis=1, keepdims=True)
    per_chain = tf.einsum("cbf,cbg->cfg", centered, centered) / 3.0
    expected_spectral = 2.0 * tf.reduce_mean(per_chain, axis=0)
    expected_mean_covariance = expected_spectral / 16.0
    tf.debugging.assert_near(result.spectral_covariance, expected_spectral)
    tf.debugging.assert_near(result.pooled_mean_covariance, expected_mean_covariance)
    assert result.chain_count == 2
    assert result.draw_count == 8
    assert result.batch_count == 4
    assert result.inference_admissible
    assert _status(result.status) == "VALID"


def test_mean_log_variance_influence_matches_manual_cluster_formula() -> None:
    paths = tf.reshape(tf.range(2 * 4 * 3 * 10, dtype=F64), [2, 4, 3, 10]) / 19.0
    result = mean_log_variance_influence(paths, jit_compile=False)
    flat = tf.reshape(paths, [-1, 10])
    means = tf.reduce_mean(flat, axis=0)
    centered = paths - means
    variances = tf.reduce_sum(tf.square(centered), axis=[0, 1, 2]) / 23.0
    expected_mean_if = tf.reduce_mean(centered, axis=2)
    second_moment = tf.reduce_mean(tf.square(centered), axis=[0, 1, 2])
    expected_log_variance_if = (
        tf.reduce_mean(tf.square(centered), axis=2) / second_moment - 1.0
    )
    tf.debugging.assert_near(result.standardized_means, means)
    tf.debugging.assert_near(result.log_variances, tf.math.log(variances))
    tf.debugging.assert_near(
        result.influence_values,
        tf.concat((expected_mean_if, expected_log_variance_if), axis=-1),
    )
    tf.debugging.assert_near(
        tf.reduce_mean(result.influence_values, axis=[0, 1]),
        tf.zeros([20], F64),
        atol=1.0e-14,
    )
    assert (result.chain_count, result.draw_count, result.forecast_replication_count) == (
        2,
        4,
        3,
    )
    assert result.path_count == 24
    assert _status(result.status) == "VALID"


def test_mean_log_variance_influence_default_xla_matches_eager() -> None:
    paths = tf.sin(_four_chain_paths()) + 0.03 * _four_chain_paths()
    eager = mean_log_variance_influence(paths, jit_compile=False)
    compiled = mean_log_variance_influence(paths)
    for name in (
        "feature_estimate",
        "standardized_means",
        "log_variances",
        "influence_values",
    ):
        tf.debugging.assert_near(getattr(compiled, name), getattr(eager, name), atol=1e-12)


def test_mean_log_variance_influence_fails_on_zero_variance() -> None:
    with pytest.raises(PredictiveContractError, match="zero variance"):
        mean_log_variance_influence(tf.ones((4, 8, 2, 10), F64), jit_compile=False)


def test_pairwise_distance_scale_matches_manual_euclidean_median() -> None:
    paths = tf.constant(
        [
            [
                [[0.0] * 10, [3.0] + [0.0] * 9],
                [[4.0] + [0.0] * 9, [0.0] * 10],
            ]
        ],
        F64,
    )
    result = pooled_pairwise_distance_scale(paths, jit_compile=False)
    # Positive distances are 1, 3, 3, 4, 4; the duplicated zero pair is excluded.
    assert float(result.median_distance) == pytest.approx(3.0)
    assert int(result.positive_pair_count) == 5
    assert int(result.total_pair_count) == 6
    assert result.path_count == 4


def test_pairwise_distance_scale_default_xla_matches_eager() -> None:
    paths = tf.sin(_four_chain_paths())
    eager = pooled_pairwise_distance_scale(paths, jit_compile=False)
    compiled = pooled_pairwise_distance_scale(paths)
    tf.debugging.assert_near(compiled.median_distance, eager.median_distance, atol=1e-12)
    tf.debugging.assert_equal(compiled.positive_pair_count, eager.positive_pair_count)
    tf.debugging.assert_equal(compiled.total_pair_count, eager.total_pair_count)


def test_pairwise_distance_scale_kernel_has_no_dynamic_boolean_mask() -> None:
    source = inspect.getsource(predictive_equivalence._pairwise_distance_scale_kernel)
    assert "boolean_mask" not in source
    assert "tf.logical_and(upper" in source
    assert "count * count" in source


def test_pairwise_distance_scale_fails_on_duplicate_degenerate_cloud() -> None:
    with pytest.raises(PredictiveContractError, match="duplicate-degenerate"):
        pooled_pairwise_distance_scale(tf.zeros((2, 2, 1, 10), F64), jit_compile=False)


def test_long_run_covariance_selects_first_eligible_ridge_for_singular_input() -> None:
    base = tf.reshape(tf.range(4 * 8, dtype=F64), [4, 8, 1])
    values = tf.concat((base, 2.0 * base), axis=-1)
    result = chain_batch_long_run_covariance(
        values,
        block_length=2,
        ridge_ladder=(0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6),
        condition_number_max=1.0e8,
        jit_compile=False,
    )
    assert result.inference_admissible
    assert int(result.selected_ridge_index) == 4
    assert float(result.selected_ridge_multiplier) == pytest.approx(1.0e-6)
    assert bool(tf.reduce_all(result.eigenvalues > 0.0))
    assert float(result.condition_number) <= 1.0e8
    tf.debugging.assert_near(
        result.precision @ result.regularized_covariance,
        tf.eye(2, dtype=F64),
        atol=1.0e-8,
    )


def test_long_run_covariance_fails_closed_when_ridge_ladder_is_insufficient() -> None:
    base = tf.reshape(tf.range(4 * 8, dtype=F64), [4, 8, 1])
    values = tf.concat((base, 2.0 * base), axis=-1)
    result = chain_batch_long_run_covariance(
        values,
        block_length=2,
        ridge_ladder=(0.0,),
        condition_number_max=1.0e12,
        jit_compile=False,
    )
    assert not result.inference_admissible
    assert int(result.selected_ridge_index) == -1
    assert _status(result.status) == "INVALID_HARD_VETO"
    assert bool(tf.reduce_all(tf.math.is_nan(result.precision)))


def test_long_run_covariance_default_xla_matches_eager() -> None:
    draw = tf.reshape(tf.cast(tf.range(8), F64), (1, 8, 1))
    chain = tf.reshape(tf.cast(tf.range(4), F64), (4, 1, 1))
    feature = tf.reshape(tf.constant([1.0, 1.7, -0.4], F64), (1, 1, 3))
    values = tf.sin((draw + 0.3 * chain + 1.0) * feature)
    eager = chain_batch_long_run_covariance(
        values,
        block_length=2,
        jit_compile=False,
    )
    compiled = chain_batch_long_run_covariance(values, block_length=2)
    for name in (
        "spectral_covariance",
        "pooled_mean_covariance",
        "regularized_covariance",
        "precision",
        "eigenvalues",
        "condition_number",
    ):
        tf.debugging.assert_near(getattr(compiled, name), getattr(eager, name), atol=1e-10)
    assert compiled.inference_admissible == eager.inference_admissible
    assert int(compiled.selected_ridge_index) == int(eager.selected_ridge_index)
    assert _status(compiled.status) == _status(eager.status) == "VALID"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"ridge_ladder": (0.0, 0.0)}, "unique"),
        ({"ridge_ladder": (1.0e-8, 0.0)}, "increasing"),
        ({"ridge_ladder": (0.0, -1.0)}, "nonnegative"),
        ({"condition_number_max": 1.0}, "exceed one"),
    ],
)
def test_long_run_covariance_rejects_invalid_policy(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(PredictiveContractError, match=match):
        chain_batch_long_run_covariance(
            tf.ones((4, 8, 2), F64),
            block_length=2,
            jit_compile=False,
            **kwargs,
        )


def test_hierarchical_indices_replay_and_preserve_fixed_chains() -> None:
    kwargs = dict(
        chain_count=4,
        draw_count=8,
        forecast_replication_count=3,
        block_length=2,
        bootstrap_count=5,
        chain_mode="stratified_fixed_chains",
        block_mode="circular",
        jit_compile=False,
    )
    first = hierarchical_resample_indices(seed=tf.constant([7, 11], tf.int32), **kwargs)
    replay = hierarchical_resample_indices(seed=tf.constant([7, 11], tf.int32), **kwargs)
    changed = hierarchical_resample_indices(seed=tf.constant([7, 12], tf.int32), **kwargs)
    tf.debugging.assert_equal(first.chain_indices, replay.chain_indices)
    tf.debugging.assert_equal(first.draw_indices, replay.draw_indices)
    tf.debugging.assert_equal(
        first.forecast_replication_indices, replay.forecast_replication_indices
    )
    expected_chains = tf.broadcast_to(tf.range(4)[None, :], [5, 4])
    tf.debugging.assert_equal(first.chain_indices, expected_chains)
    assert bool(tf.reduce_any(first.draw_indices != changed.draw_indices))
    assert bool(
        tf.reduce_any(
            first.forecast_replication_indices != changed.forecast_replication_indices
        )
    )


def test_hierarchical_indices_reject_chain_resampling_and_remainders() -> None:
    common = dict(
        chain_count=4,
        draw_count=8,
        forecast_replication_count=2,
        bootstrap_count=4,
        seed=tf.constant([1, 2], tf.int32),
        jit_compile=False,
    )
    with pytest.raises(PredictiveContractError, match="forbids resampling"):
        hierarchical_resample_indices(
            block_length=2, chain_mode="resample_chains", **common
        )
    with pytest.raises(PredictiveContractError, match="divide"):
        hierarchical_resample_indices(
            block_length=3, chain_mode="stratified_fixed_chains", **common
        )


def test_bonferroni_intervals_use_family_correction() -> None:
    estimate = tf.constant([0.0, 1.0], F64)
    standard_error = tf.constant([0.1, 0.2], F64)
    intervals = simultaneous_feature_intervals(
        estimate,
        feature_alpha=tf.constant(0.04, F64),
        method="bonferroni_studentized",
        standard_error=standard_error,
    )
    pointwise = tf.sqrt(tf.constant(2.0, F64)) * tf.math.erfinv(
        tf.constant(2.0 * (1.0 - 0.04 / 2.0) - 1.0, F64)
    )
    assert float(intervals.critical_value) > float(pointwise)
    tf.debugging.assert_near(
        intervals.upper - estimate, intervals.critical_value * standard_error
    )


def test_bootstrap_max_intervals_and_zero_se_veto() -> None:
    estimates = tf.constant([0.2, -0.1], F64)
    offsets = tf.linspace(tf.constant(-0.3, F64), tf.constant(0.3, F64), 25)
    bootstrap = estimates[None, :] + tf.stack([offsets, -offsets], axis=1)
    intervals = simultaneous_feature_intervals(
        estimates,
        feature_alpha=0.05,
        method="bootstrap_max_statistic",
        bootstrap_estimates=bootstrap,
    )
    assert intervals.method == "bootstrap_max_statistic"
    assert float(intervals.critical_value) > 0.0
    translated = simultaneous_feature_intervals(
        estimates,
        feature_alpha=0.05,
        method="bootstrap_max_statistic",
        bootstrap_estimates=bootstrap + tf.constant(7.5, F64),
    )
    tf.debugging.assert_near(
        translated.critical_value, intervals.critical_value, atol=1e-12
    )
    tf.debugging.assert_near(
        translated.standard_error, intervals.standard_error, atol=1e-12
    )
    with pytest.raises(PredictiveContractError, match="standard error"):
        simultaneous_feature_intervals(
            estimates,
            feature_alpha=0.05,
            method="bonferroni_studentized",
            standard_error=tf.constant([0.0, 0.1], F64),
        )


def test_cross_chain_mmd_interval_admission_and_zero_variance_veto() -> None:
    bands, weights = _bands()
    statistic = cross_chain_linear_mmd(
        _four_chain_paths(),
        _perturbed_four_chain_paths(0.2),
        bandwidths=bands,
        mixture_weights=weights,
        chain_pair_schedule=tf.constant([[0, 1], [2, 3]], tf.int32),
        independent_arm_banks_verified=True,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=False,
    )
    interval = cross_chain_mmd_upper_interval(
        statistic, mmd_alpha=0.02, block_length=2
    )
    assert interval.inference_admissible
    assert _status(interval.status) == "VALID"
    assert int(interval.block_count) == 8
    assert float(interval.upper) > float(interval.estimate)
    normal_critical = tf.sqrt(tf.constant(2.0, F64)) * tf.math.erfinv(
        tf.constant(2.0 * (1.0 - 0.02 / 2.0) - 1.0, F64)
    )
    assert float(interval.critical_value) > float(normal_critical)

    with pytest.raises(PredictiveContractError, match="authenticated"):
        cross_chain_mmd_upper_interval(
            replace(statistic), mmd_alpha=0.02, block_length=2
        )

    constant = cross_chain_linear_mmd(
        _four_chain_paths(),
        _four_chain_paths(),
        bandwidths=bands,
        mixture_weights=weights,
        chain_pair_schedule=tf.constant([[0, 1], [2, 3]], tf.int32),
        independent_arm_banks_verified=True,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=False,
    )
    invalid = cross_chain_mmd_upper_interval(
        constant, mmd_alpha=0.02, block_length=2
    )
    assert not invalid.inference_admissible
    assert _status(invalid.status) == "INVALID_HARD_VETO"


def test_decision_pass_requires_both_strict_equivalence_branches() -> None:
    decision = classify_predictive_evidence(
        _valid_feature_interval(),
        _valid_mmd_interval(),
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.05,
        feature_alpha=0.03,
        mmd_alpha=0.02,
    )
    assert decision.status == "PASS"
    assert decision.primary_interval_status == "PASS"
    assert decision.mmd_upper_bound_status == "PASS"


def test_decision_material_difference_has_precedence() -> None:
    decision = classify_predictive_evidence(
        _valid_feature_interval(lower=(0.3, -0.1), upper=(0.4, 0.1)),
        _valid_mmd_interval(),
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.05,
        feature_alpha=0.03,
        mmd_alpha=0.02,
    )
    assert decision.status == "MATERIAL_DIFFERENCE"
    assert decision.primary_interval_status == "MATERIAL_DIFFERENCE"


def test_decision_wide_zero_interval_is_inconclusive_not_pass() -> None:
    decision = classify_predictive_evidence(
        _valid_feature_interval(lower=(-0.3, -0.1), upper=(0.3, 0.1)),
        _valid_mmd_interval(),
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.05,
        feature_alpha=0.03,
        mmd_alpha=0.02,
    )
    assert decision.status == "INCONCLUSIVE_UNDERPOWERED"
    assert decision.primary_interval_status == "INCONCLUSIVE_UNDERPOWERED"


@pytest.mark.parametrize(
    "feature,mmd,mechanics,code",
    [
        (_valid_feature_interval(), _valid_mmd_interval(), True, "MECHANICS_ONLY_CANNOT_PASS"),
        (
            _valid_feature_interval(),
            replace(_valid_mmd_interval(), inference_admissible=False),
            False,
            "MMD_INTERVAL_NOT_ADMISSIBLE",
        ),
    ],
)
def test_decision_hard_vetoes_nonadmissible_evidence(
    feature: SimultaneousIntervals,
    mmd: MMDInterval,
    mechanics: bool,
    code: str,
) -> None:
    decision = classify_predictive_evidence(
        feature,
        mmd,
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.05,
        feature_alpha=0.03,
        mmd_alpha=0.02,
        mechanics_only=mechanics,
    )
    assert decision.status == "INVALID_HARD_VETO"
    assert code in decision.hard_veto_codes


def test_decision_rejects_copied_and_reconstructed_interval_objects() -> None:
    feature = _valid_feature_interval()
    mmd = _valid_mmd_interval()
    copied = classify_predictive_evidence(
        copy.copy(feature),
        copy.copy(mmd),
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.05,
        feature_alpha=0.03,
        mmd_alpha=0.02,
    )
    assert copied.status == "INVALID_HARD_VETO"
    assert "FEATURE_INTERVAL_UNAUTHENTICATED" in copied.hard_veto_codes
    assert "MMD_INTERVAL_UNAUTHENTICATED" in copied.hard_veto_codes

    reconstructed_feature = SimultaneousIntervals(**vars(feature))
    reconstructed_mmd = MMDInterval(**vars(mmd))
    reconstructed = classify_predictive_evidence(
        reconstructed_feature,
        reconstructed_mmd,
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.05,
        feature_alpha=0.03,
        mmd_alpha=0.02,
    )
    assert reconstructed.status == "INVALID_HARD_VETO"
    assert "FEATURE_INTERVAL_UNAUTHENTICATED" in reconstructed.hard_veto_codes
    assert "MMD_INTERVAL_UNAUTHENTICATED" in reconstructed.hard_veto_codes


def test_decision_rejects_invalid_joint_alpha_and_binding_drift() -> None:
    decision = classify_predictive_evidence(
        _valid_feature_interval(),
        _valid_mmd_interval(),
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.04,
        feature_alpha=0.03,
        mmd_alpha=0.02,
    )
    assert decision.status == "INVALID_HARD_VETO"
    assert "INVALID_JOINT_ALPHA_ALLOCATION" in decision.hard_veto_codes

    drifted = classify_predictive_evidence(
        _valid_feature_interval(alpha=0.025),
        _valid_mmd_interval(),
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.05,
        feature_alpha=0.03,
        mmd_alpha=0.02,
    )
    assert "FEATURE_ALPHA_BINDING_MISMATCH" in drifted.hard_veto_codes


@pytest.mark.parametrize(
    "feature,mmd,expected_code",
    [
        (
            replace(
                _valid_feature_interval(),
                lower=tf.constant([0.2, -0.1], F64),
                upper=tf.constant([0.1, 0.1], F64),
            ),
            _valid_mmd_interval(),
            "FEATURE_INTERVAL_MALFORMED",
        ),
        (
            replace(
                _valid_feature_interval(),
                lower=tf.constant([float("nan"), -0.1], F64),
            ),
            _valid_mmd_interval(),
            "FEATURE_INTERVAL_MALFORMED",
        ),
        (
            _valid_feature_interval(),
            replace(
                _valid_mmd_interval(),
                lower=tf.constant(0.2, F64),
                upper=tf.constant(0.1, F64),
            ),
            "MMD_INTERVAL_MALFORMED",
        ),
    ],
)
def test_decision_hard_vetoes_malformed_interval_objects(
    feature: SimultaneousIntervals,
    mmd: MMDInterval,
    expected_code: str,
) -> None:
    decision = classify_predictive_evidence(
        feature,
        mmd,
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.05,
        feature_alpha=0.03,
        mmd_alpha=0.02,
    )
    assert decision.status == "INVALID_HARD_VETO"
    assert expected_code in decision.hard_veto_codes


def test_decision_rejects_truthy_mechanics_flag() -> None:
    with pytest.raises(PredictiveContractError, match="Python bool"):
        classify_predictive_evidence(
            _valid_feature_interval(),
            _valid_mmd_interval(),
            margins=tf.constant([0.2, 0.2], F64),
            mmd_tolerance=tf.constant(0.05, F64),
            total_alpha=0.05,
            feature_alpha=0.03,
            mmd_alpha=0.02,
            mechanics_only="false",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "feature,mmd,expected_code",
    [
        (
            replace(_valid_feature_interval(), method="pointwise"),  # type: ignore[arg-type]
            _valid_mmd_interval(),
            "FEATURE_INTERVAL_MALFORMED",
        ),
        (
            replace(
                _valid_feature_interval(),
                status=tf.constant(["VALID"]),
            ),
            _valid_mmd_interval(),
            "FEATURE_INTERVAL_MALFORMED",
        ),
        (
            replace(
                _valid_feature_interval(),
                lower=tf.constant([-0.01, -0.01], F64),
                upper=tf.constant([0.01, 0.01], F64),
            ),
            _valid_mmd_interval(),
            "FEATURE_INTERVAL_MALFORMED",
        ),
        (
            _valid_feature_interval(),
            replace(_valid_mmd_interval(), status=tf.constant(1, tf.int32)),
            "MMD_INTERVAL_MALFORMED",
        ),
        (
            _valid_feature_interval(),
            replace(
                _valid_mmd_interval(),
                lower=tf.constant(0.0, F64),
                upper=tf.constant(0.02, F64),
            ),
            "MMD_INTERVAL_MALFORMED",
        ),
    ],
)
def test_decision_rejects_forged_status_method_and_interval_algebra(
    feature: SimultaneousIntervals,
    mmd: MMDInterval,
    expected_code: str,
) -> None:
    decision = classify_predictive_evidence(
        feature,
        mmd,
        margins=tf.constant([0.2, 0.2], F64),
        mmd_tolerance=tf.constant(0.05, F64),
        total_alpha=0.05,
        feature_alpha=0.03,
        mmd_alpha=0.02,
    )
    assert decision.status == "INVALID_HARD_VETO"
    assert expected_code in decision.hard_veto_codes


def test_mmd_interval_rejects_forged_linear_statistic() -> None:
    bands, weights = _bands()
    statistic = cross_chain_linear_mmd(
        _four_chain_paths(),
        _four_chain_paths(0.2),
        bandwidths=bands,
        mixture_weights=weights,
        chain_pair_schedule=tf.constant([[0, 1], [2, 3]], tf.int32),
        independent_arm_banks_verified=True,
        stationarity_verified=True,
        mixing_verified=True,
        jit_compile=False,
    )
    with pytest.raises(PredictiveContractError, match="authenticated"):
        cross_chain_mmd_upper_interval(
            replace(statistic, squared_mmd_linear=tf.constant(123.0, F64)),
            mmd_alpha=0.02,
            block_length=2,
        )
    with pytest.raises(PredictiveContractError, match="authenticated"):
        cross_chain_mmd_upper_interval(
            replace(statistic, independent_arm_banks_verified=False),
            mmd_alpha=0.02,
            block_length=2,
        )
    with pytest.raises(PredictiveContractError, match="authenticated"):
        cross_chain_mmd_upper_interval(
            replace(statistic, status=tf.constant(["VALID"])),
            mmd_alpha=0.02,
            block_length=2,
        )
