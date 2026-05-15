import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_filter
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_filter
from bayesfilter.testing import (
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_b_observations_tf,
    model_c_observations_tf,
)


_BACKENDS = (
    ("tf_svd_cubature", "cubature"),
    ("tf_svd_ukf", "unscented"),
    ("tf_svd_cut4", None),
)


def _value_result(backend: str, observations: tf.Tensor, model, return_filtered: bool):
    kwargs = {
        "innovation_floor": tf.constant(1e-12, dtype=tf.float64),
        "return_filtered": return_filtered,
    }
    if backend == "tf_svd_cut4":
        return tf_svd_cut4_filter(observations, model, **kwargs)
    return tf_svd_sigma_point_filter(observations, model, backend=backend, **kwargs)


def _compiled_value_fn(backend: str, model, return_filtered: bool):
    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(obs: tf.Tensor):
        result = _value_result(backend, obs, model, return_filtered)
        if return_filtered:
            return (
                result.log_likelihood,
                result.filtered_means,
                result.filtered_covariances,
            )
        return result.log_likelihood

    return compiled


def _assert_same_static_shape_single_concrete(compiled, observations: tf.Tensor) -> None:
    compiled(observations)
    compiled(tf.identity(observations))
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def _assert_value_parity(backend: str, observations: tf.Tensor, model, return_filtered: bool) -> None:
    eager = _value_result(backend, observations, model, return_filtered)
    compiled = _compiled_value_fn(backend, model, return_filtered)
    graph = compiled(observations)

    if return_filtered:
        graph_value, graph_means, graph_covariances = graph
        np.testing.assert_allclose(
            graph_value.numpy(),
            eager.log_likelihood.numpy(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            graph_means.numpy(),
            eager.filtered_means.numpy(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            graph_covariances.numpy(),
            eager.filtered_covariances.numpy(),
            atol=1e-12,
        )
    else:
        np.testing.assert_allclose(
            graph.numpy(),
            eager.log_likelihood.numpy(),
            atol=1e-12,
        )
        assert eager.filtered_means is None
        assert eager.filtered_covariances is None

    _assert_same_static_shape_single_concrete(compiled, observations)


@tf.autograph.experimental.do_not_convert
def _model_b_fixture():
    return model_b_observations_tf(), make_nonlinear_accumulation_model_tf()


@tf.autograph.experimental.do_not_convert
def _model_c_fixture():
    return model_c_observations_tf(), make_univariate_nonlinear_growth_model_tf()


def test_cpu_xla_device_hidden_before_tensorflow_runtime_probe() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert tf.config.list_physical_devices("GPU") == []


@pytest.mark.parametrize("backend,_rule", _BACKENDS)
def test_model_b_value_cpu_xla_parity_return_filtered_false(backend: str, _rule: str | None) -> None:
    observations, model = _model_b_fixture()
    _assert_value_parity(backend, observations, model, return_filtered=False)


@pytest.mark.parametrize("backend,_rule", _BACKENDS)
def test_model_b_value_cpu_xla_parity_return_filtered_true(backend: str, _rule: str | None) -> None:
    observations, model = _model_b_fixture()
    _assert_value_parity(backend, observations, model, return_filtered=True)


@pytest.mark.parametrize("backend,_rule", _BACKENDS)
def test_model_c_value_cpu_xla_parity_return_filtered_false(backend: str, _rule: str | None) -> None:
    observations, model = _model_c_fixture()
    _assert_value_parity(backend, observations, model, return_filtered=False)


@pytest.mark.parametrize("backend,_rule", _BACKENDS)
def test_model_c_value_cpu_xla_parity_return_filtered_true(backend: str, _rule: str | None) -> None:
    observations, model = _model_c_fixture()
    _assert_value_parity(backend, observations, model, return_filtered=True)
